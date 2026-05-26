"""RAG orchestrator.

Wires the components together: retrieve → generate → detect hallucination.
The pipeline is the single object the API holds onto.
"""

from __future__ import annotations

import logging
import time
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
        )
        return cls(retriever=retriever, generator=generator, detector=detector)

    def ask(self, question: str, *, top_k: int | None = None) -> PipelineAnswer:
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        chunks = self.retriever.retrieve(question, top_k=top_k)
        timings["retrieve_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        gen = self.generator.generate(question, chunks)
        timings["generate_ms"] = (time.perf_counter() - t0) * 1000

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

        return PipelineAnswer(
            answer=gen.answer,
            chunks=chunks,
            hallucination=hallucination,
            latency_ms=timings,
        )
