"""RAG orchestrator.

Wires the components together: retrieve → generate → detect hallucination.
The pipeline is the single object the API holds onto.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass

from app.config import Settings, get_settings
from app.providers import build_embedder, build_llm
from app.rag.generator import Generator
from app.rag.hallucination import HallucinationDetector, HallucinationResult
from app.rag.retriever import RetrievalConfig, Retriever
from app.stores import build_vector_store
from app.stores.base import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class PipelineAnswer:
    answer: str
    chunks: list[RetrievedChunk]
    hallucination: HallucinationResult
    latency_ms: dict[str, float]


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        detector: HallucinationDetector,
    ):
        self.retriever = retriever
        self.generator = generator
        self.detector = detector

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "RAGPipeline":
        settings = settings or get_settings()
        embedder = build_embedder(settings)
        store = build_vector_store(settings)
        llm = build_llm(settings)

        retriever = Retriever(
            embedder=embedder,
            store=store,
            config=RetrievalConfig(top_k=settings.top_k, mmr_lambda=settings.mmr_lambda),
        )
        generator = Generator(llm=llm)
        detector = HallucinationDetector(
            model_name=settings.nli_model,
            threshold=settings.hallucination_threshold,
            device=settings.device,
        )
        return cls(retriever=retriever, generator=generator, detector=detector)

    def warmup(self) -> None:
        """Force-load lazy components so the first /query doesn't pay the cost.

        The embedder is eagerly built in `from_settings`, but the NLI cross-
        encoder used by the hallucination detector is lazy-loaded on first
        `detect()` call. That download is ~738MB and routinely exceeds the
        frontend's request timeout, so we trigger it at startup instead.
        """
        self.detector._ensure_loaded()

    def ask(self, question: str, *, top_k: int | None = None) -> PipelineAnswer:
        # Drain the streaming variant so the two paths can't diverge.
        for event_type, data in self.ask_stream(question, top_k=top_k):
            if event_type == "result":
                return data  # type: ignore[return-value]
        raise RuntimeError("ask_stream completed without yielding a result event")

    def ask_stream(
        self, question: str, *, top_k: int | None = None
    ) -> Iterator[tuple[str, object]]:
        """Yield ('stage', {...}) events marking the start of each stage, then
        a single ('result', PipelineAnswer) at the end.

        Each 'stage' event fires *before* the work begins, so consumers reflect
        the actual current step rather than a timer-driven guess.
        """
        timings: dict[str, float] = {}

        yield ("stage", {"stage": "retrieve"})
        t0 = time.perf_counter()
        chunks = self.retriever.retrieve(question, top_k=top_k)
        timings["retrieve_ms"] = (time.perf_counter() - t0) * 1000

        yield ("stage", {"stage": "generate", "n_chunks": len(chunks)})
        t0 = time.perf_counter()
        gen = self.generator.generate(question, chunks)
        timings["generate_ms"] = (time.perf_counter() - t0) * 1000

        yield ("stage", {"stage": "detect"})
        t0 = time.perf_counter()
        hallucination = self.detector.detect(gen.answer, chunks)
        timings["detect_ms"] = (time.perf_counter() - t0) * 1000

        timings["total_ms"] = sum(timings.values())

        logger.info(
            "rag_query",
            extra={
                "question_len": len(question),
                "n_chunks": len(chunks),
                "hallucination_flagged": hallucination.flagged,
                "hallucination_score": hallucination.score,
                **{k: round(v, 1) for k, v in timings.items()},
            },
        )

        yield (
            "result",
            PipelineAnswer(
                answer=gen.answer,
                chunks=chunks,
                hallucination=hallucination,
                latency_ms=timings,
            ),
        )
