from app.rag.retriever import RetrievalConfig, Retriever, _mmr
from app.stores.base import Chunk
from tests.fakes import FakeEmbedder, InMemoryVectorStore


def test_retriever_returns_top_k():
    embedder = FakeEmbedder()
    store = InMemoryVectorStore()

    chunks = [
        Chunk(id=f"c{i}", text=f"document number {i} about topic {i}", source="t.md")
        for i in range(10)
    ]
    store.upsert(chunks, embedder.embed_documents([c.text for c in chunks]))

    r = Retriever(embedder, store, RetrievalConfig(top_k=3, fetch_k_multiplier=2, mmr_lambda=1.0))
    out = r.retrieve("document number 5")
    assert len(out) == 3
    # With pure-relevance MMR (lambda=1) the most similar should be first.
    assert any("number 5" in rc.chunk.text for rc in out[:1])


def test_mmr_diversifies_results():
    # Build a query and four candidates: two near-duplicates and two distinct.
    query = [1.0, 0.0, 0.0, 0.0]
    docs = [
        [0.9, 0.1, 0.0, 0.0],   # very close to query
        [0.89, 0.11, 0.0, 0.0], # near-duplicate of #0
        [0.5, 0.5, 0.0, 0.0],   # related but different direction
        [0.0, 0.0, 1.0, 0.0],   # unrelated
    ]
    # Relevance-only: would pick {0, 1} (the duplicates).
    pure_rel = _mmr(query_vec=query, doc_vecs=docs, k=2, lambda_=1.0)
    assert pure_rel == [0, 1]

    # Diversity-aware: should drop the duplicate.
    diverse = _mmr(query_vec=query, doc_vecs=docs, k=2, lambda_=0.5)
    assert 0 in diverse and 1 not in diverse
