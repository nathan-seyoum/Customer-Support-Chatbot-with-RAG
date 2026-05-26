"""NLI-based hallucination / groundedness detection.

Approach (industry standard, used by RAGAS' faithfulness, TruLens'
Groundedness, Ragas + DeepEval evaluators):

  1. Split the model's answer into atomic sentences.
  2. For each sentence, ask a Natural Language Inference (NLI) model whether
     it is ENTAILED by any of the retrieved context chunks.
  3. The sentence's groundedness score = max P(entailment) across chunks.
  4. The answer's overall groundedness = mean over sentences.
  5. If overall groundedness < threshold, flag the answer as a likely
     hallucination.

We use a cross-encoder NLI model (default: `cross-encoder/nli-deberta-v3-base`)
because cross-encoders score (premise, hypothesis) jointly and are far more
accurate than dual-encoder similarity for entailment.

References for the approach:
  - RAGAS (faithfulness):       https://docs.ragas.io/en/stable/concepts/metrics/faithfulness.html
  - TruLens (Groundedness):     https://www.trulens.org/trulens_eval/getting_started/core_concepts/feedback_functions/
  - Honovich et al., 2022 (TRUE):  https://arxiv.org/abs/2204.04991  (NLI for factual consistency)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.stores.base import RetrievedChunk

# Cross-encoder NLI heads from HF typically have labels in this order:
# [contradiction, entailment, neutral]  (e.g. cross-encoder/nli-deberta-v3-base)
# We resolve the mapping at runtime via model.config.id2label to stay robust.
ENTAILMENT_LABEL = "entailment"


@dataclass
class SentenceJudgment:
    sentence: str
    entailment_score: float           # max P(entailment) over chunks, in [0, 1]
    best_chunk_id: str | None = None  # which chunk supported it (if any)


@dataclass
class HallucinationResult:
    flagged: bool
    score: float                       # mean entailment, in [0, 1]
    threshold: float
    per_sentence: list[SentenceJudgment] = field(default_factory=list)

    def to_payload(self) -> dict:
        return {
            "flagged": self.flagged,
            "score": round(self.score, 4),
            "threshold": self.threshold,
            "per_sentence": [
                {
                    "sentence": s.sentence,
                    "entailment_score": round(s.entailment_score, 4),
                    "best_chunk_id": s.best_chunk_id,
                }
                for s in self.per_sentence
            ],
        }


# ---------------------------------------------------------------------------
# Sentence segmentation
# ---------------------------------------------------------------------------
# Intentionally light-weight (no nltk/spacy dependency). Splits on sentence
# terminators while keeping abbreviations like "e.g." mostly intact.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")
_CITATION_RE = re.compile(r"\[#\d+\]")


def split_into_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    out: list[str] = []
    for p in parts:
        # Strip inline citations like "[#1]" before NLI — they confuse the model.
        cleaned = _CITATION_RE.sub("", p).strip()
        # Skip the "I don't have enough information..." canned refusal; it's
        # not a hallucination by definition.
        if not cleaned or len(cleaned) < 3:
            continue
        out.append(cleaned)
    return out


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
class HallucinationDetector:
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base", threshold: float = 0.5):
        self.model_name = model_name
        self.threshold = threshold
        self._model = None  # lazy
        self._entailment_idx: int | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # sentence-transformers ships a thin wrapper around HF cross-encoders.
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(self.model_name)
        # Resolve label index dynamically so swapping NLI models still works.
        id2label = self._model.model.config.id2label
        self._entailment_idx = next(
            i for i, name in id2label.items() if str(name).lower().startswith("entail")
        )

    @staticmethod
    def _softmax(row: list[float]) -> list[float]:
        m = max(row)
        exps = [pow(2.71828182845904, x - m) for x in row]
        s = sum(exps)
        return [e / s for e in exps]

    def detect(self, answer: str, chunks: list[RetrievedChunk]) -> HallucinationResult:
        sentences = split_into_sentences(answer)

        # Defensive: an answer with no factual content can't hallucinate.
        if not sentences:
            return HallucinationResult(
                flagged=False, score=1.0, threshold=self.threshold, per_sentence=[]
            )

        # Refusal sentinel — treat as fully grounded.
        if answer.strip().startswith("I don't have enough information"):
            return HallucinationResult(
                flagged=False, score=1.0, threshold=self.threshold, per_sentence=[]
            )

        if not chunks:
            # Answer has claims but we retrieved no context — definitionally ungrounded.
            return HallucinationResult(
                flagged=True,
                score=0.0,
                threshold=self.threshold,
                per_sentence=[SentenceJudgment(sentence=s, entailment_score=0.0) for s in sentences],
            )

        self._ensure_loaded()

        # Build (premise, hypothesis) pairs: every sentence vs every chunk.
        pairs: list[tuple[str, str]] = []
        for chunk in chunks:
            for sent in sentences:
                pairs.append((chunk.chunk.text, sent))

        # CrossEncoder.predict returns raw logits over [contradiction, entailment, neutral].
        logits = self._model.predict(pairs, show_progress_bar=False)
        # Shape: [n_chunks * n_sentences, 3]
        n_sent = len(sentences)

        judgments: list[SentenceJudgment] = []
        for s_idx, sentence in enumerate(sentences):
            best_score = 0.0
            best_chunk_id: str | None = None
            for c_idx, chunk in enumerate(chunks):
                row = logits[c_idx * n_sent + s_idx]
                probs = self._softmax(list(row))
                p_entail = probs[self._entailment_idx]
                if p_entail > best_score:
                    best_score = float(p_entail)
                    best_chunk_id = chunk.chunk.id
            judgments.append(
                SentenceJudgment(
                    sentence=sentence,
                    entailment_score=best_score,
                    best_chunk_id=best_chunk_id,
                )
            )

        mean_score = sum(j.entailment_score for j in judgments) / len(judgments)
        flagged = mean_score < self.threshold
        return HallucinationResult(
            flagged=flagged,
            score=mean_score,
            threshold=self.threshold,
            per_sentence=judgments,
        )
