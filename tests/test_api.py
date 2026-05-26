"""API smoke test. Stubs the pipeline so the route layer can be exercised
without booting heavy models."""

from contextlib import asynccontextmanager

from fastapi.testclient import TestClient

from app.main import app
from app.rag.generator import Generator
from app.rag.hallucination import HallucinationDetector, HallucinationResult
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import RetrievalConfig, Retriever
from app.stores.base import Chunk
from tests.fakes import FakeEmbedder, FakeLLM, InMemoryVectorStore


class StubDetector(HallucinationDetector):
    def __init__(self):
        self.model_name = "stub"
        self.threshold = 0.5
        self._model = None
        self._entailment_idx = None

    def detect(self, answer, chunks):
        return HallucinationResult(flagged=False, score=0.85, threshold=0.5, per_sentence=[])


def _build_pipeline() -> RAGPipeline:
    embedder = FakeEmbedder()
    store = InMemoryVectorStore()
    chunks = [
        Chunk(id="r1", text="Refunds take 5 to 7 business days.", source="returns.md"),
        Chunk(id="s1", text="Ground shipping is 3 to 5 business days.", source="shipping.md"),
    ]
    store.upsert(chunks, embedder.embed_documents([c.text for c in chunks]))
    return RAGPipeline(
        retriever=Retriever(embedder, store, RetrievalConfig(top_k=2, mmr_lambda=1.0)),
        generator=Generator(llm=FakeLLM()),
        detector=StubDetector(),
    )


@asynccontextmanager
async def _stub_lifespan(_app):
    _app.state.pipeline = _build_pipeline()
    yield


def test_health_and_query_round_trip():
    # Swap in a no-op lifespan that injects our stubbed pipeline, so the real
    # lifespan (which would load multi-hundred-MB models) never runs.
    app.router.lifespan_context = _stub_lifespan

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        resp = client.post("/query", json={"question": "How long do refunds take?"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["citations"], "expected at least one citation"
        assert body["hallucination"]["flagged"] is False
        assert "total_ms" in body["latency_ms"]
