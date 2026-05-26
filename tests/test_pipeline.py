"""End-to-end pipeline test with fakes — no model downloads, no Ollama."""

from app.rag.generator import Generator
from app.rag.hallucination import HallucinationDetector, HallucinationResult
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import RetrievalConfig, Retriever
from app.stores.base import Chunk

from tests.fakes import FakeEmbedder, FakeLLM, InMemoryVectorStore


class StubDetector(HallucinationDetector):
    """Bypass the NLI model; just return a high groundedness score."""

    def __init__(self):
        self.model_name = "stub"
        self.threshold = 0.5
        self._model = None
        self._entailment_idx = None

    def detect(self, answer, chunks):
        return HallucinationResult(flagged=False, score=0.9, threshold=0.5, per_sentence=[])


def _seeded_pipeline() -> RAGPipeline:
    embedder = FakeEmbedder()
    store = InMemoryVectorStore()
    chunks = [
        Chunk(id="ret-1", text="Refunds are issued within 5 to 7 business days.", source="returns.md"),
        Chunk(id="ship-1", text="Standard ground shipping takes 3 to 5 business days.", source="shipping.md"),
    ]
    store.upsert(chunks, embedder.embed_documents([c.text for c in chunks]))

    return RAGPipeline(
        retriever=Retriever(embedder, store, RetrievalConfig(top_k=2, mmr_lambda=1.0)),
        generator=Generator(llm=FakeLLM()),
        detector=StubDetector(),
    )


def test_pipeline_returns_grounded_answer_and_timings():
    pipeline = _seeded_pipeline()
    result = pipeline.ask("How long do refunds take?")

    assert "[#1]" in result.answer
    assert len(result.chunks) == 2
    assert result.hallucination.flagged is False
    assert "retrieve_ms" in result.latency_ms
    assert "generate_ms" in result.latency_ms
    assert "detect_ms" in result.latency_ms
    assert "total_ms" in result.latency_ms
