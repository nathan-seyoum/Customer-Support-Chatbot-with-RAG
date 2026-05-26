from app.rag.hallucination import (
    HallucinationDetector,
    HallucinationResult,
    split_into_sentences,
)
from app.stores.base import Chunk, RetrievedChunk


def test_sentence_splitter_basic():
    out = split_into_sentences("Hello world. This is a test! Right?")
    assert out == ["Hello world.", "This is a test!", "Right?"]


def test_sentence_splitter_strips_citations():
    out = split_into_sentences("Refunds take 5 to 7 business days [#1]. Free returns are included [#2].")
    assert out == ["Refunds take 5 to 7 business days .", "Free returns are included ."]


def test_refusal_is_not_flagged():
    det = HallucinationDetector.__new__(HallucinationDetector)
    det.model_name = "fake"
    det.threshold = 0.5
    det._model = None  # NLI model isn't loaded — the refusal path short-circuits
    det._entailment_idx = None

    result = det.detect("I don't have enough information in the knowledge base to answer that.", chunks=[])
    assert isinstance(result, HallucinationResult)
    assert result.flagged is False
    assert result.score == 1.0


def test_empty_chunks_flags_factual_answer():
    det = HallucinationDetector.__new__(HallucinationDetector)
    det.model_name = "fake"
    det.threshold = 0.5
    det._model = None
    det._entailment_idx = None

    result = det.detect("Our refunds take exactly 3 days.", chunks=[])
    assert result.flagged is True
    assert result.score == 0.0


def _fake_logits_factory(entailment_scores_per_sentence: list[float]):
    """Returns a callable that mimics CrossEncoder.predict for one chunk × N sentences."""

    class FakeEncoder:
        def predict(self, pairs, show_progress_bar=False):
            # One chunk in our tests → pairs == sentences in order.
            n_sent = len(pairs)
            logits = []
            for i in range(n_sent):
                e = entailment_scores_per_sentence[i % len(entailment_scores_per_sentence)]
                # Order: [contradiction, entailment, neutral]
                logits.append([1 - e, e * 5, 0.0])
            return logits

    class FakeModel:
        class config:
            id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}

    fake = FakeEncoder()
    fake.model = FakeModel()
    return fake


def test_high_entailment_is_not_flagged():
    det = HallucinationDetector.__new__(HallucinationDetector)
    det.model_name = "fake"
    det.threshold = 0.5
    det._model = _fake_logits_factory([0.95])
    det._entailment_idx = 1

    chunks = [RetrievedChunk(chunk=Chunk(id="c1", text="Refunds take 5-7 business days.", source="x"), score=0.9)]
    result = det.detect("Refunds take 5 to 7 business days.", chunks=chunks)
    assert result.flagged is False
    assert result.score > 0.5


def test_low_entailment_is_flagged():
    det = HallucinationDetector.__new__(HallucinationDetector)
    det.model_name = "fake"
    det.threshold = 0.5
    det._model = _fake_logits_factory([0.05])
    det._entailment_idx = 1

    chunks = [RetrievedChunk(chunk=Chunk(id="c1", text="Refunds take 5-7 business days.", source="x"), score=0.9)]
    result = det.detect("Our products are made on Mars.", chunks=chunks)
    assert result.flagged is True
    assert result.score < 0.5
