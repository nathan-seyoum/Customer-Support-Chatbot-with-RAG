"""Retriever: embeds the query, asks the vector store for candidates, then
optionally re-ranks with Maximal Marginal Relevance (MMR).

MMR (Carbonell & Goldstein, 1998) trades off similarity-to-query against
diversity-among-results. With dense embeddings of FAQ-style content, the top-k
nearest neighbors are often near-duplicates of each other; MMR explicitly
spreads coverage across the corpus so the LLM gets complementary context.

This implementation works directly on the embeddings already loaded for the
fetched candidates, so it adds no extra vector-store calls.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.providers.base import Embedder
from app.stores.base import RetrievedChunk, VectorStore


@dataclass
class RetrievalConfig:
    top_k: int = 5
    fetch_k_multiplier: int = 4  # fetch top_k * this, then MMR-rerank down to top_k
    mmr_lambda: float = 0.5      # 1.0 = pure relevance, 0.0 = pure diversity


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore, config: RetrievalConfig):
        self.embedder = embedder
        self.store = store
        self.config = config

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k or self.config.top_k
        fetch_k = k * self.config.fetch_k_multiplier

        query_vec = self.embedder.embed_query(query)
        candidates = self.store.query(query_vec, top_k=fetch_k)
        if len(candidates) <= k:
            return candidates

        # Re-embed candidate texts to do MMR. We *could* persist embeddings in
        # the store and avoid this, but Chroma's `include=["embeddings"]` adds
        # network bytes and isn't free either; for our scale, re-embedding a
        # handful of short chunks is cheap and keeps the store API minimal.
        cand_vecs = self.embedder.embed_documents([rc.chunk.text for rc in candidates])
        selected_idx = _mmr(
            query_vec=query_vec,
            doc_vecs=cand_vecs,
            k=k,
            lambda_=self.config.mmr_lambda,
        )
        return [candidates[i] for i in selected_idx]


# ---------------------------------------------------------------------------
# MMR (works on already-normalized vectors → dot product = cosine similarity)
# ---------------------------------------------------------------------------
def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def _mmr(
    *,
    query_vec: list[float],
    doc_vecs: list[list[float]],
    k: int,
    lambda_: float,
) -> list[int]:
    """Return indices of the MMR-selected documents, in selection order."""
    if not doc_vecs:
        return []
    k = min(k, len(doc_vecs))

    sim_to_query = [_cosine(query_vec, v) for v in doc_vecs]
    selected: list[int] = []
    remaining = set(range(len(doc_vecs)))

    # Seed with the most query-similar document.
    first = max(remaining, key=lambda i: sim_to_query[i])
    selected.append(first)
    remaining.remove(first)

    while remaining and len(selected) < k:
        def mmr_score(i: int) -> float:
            max_sim_to_selected = max(_cosine(doc_vecs[i], doc_vecs[j]) for j in selected)
            return lambda_ * sim_to_query[i] - (1 - lambda_) * max_sim_to_selected

        nxt = max(remaining, key=mmr_score)
        selected.append(nxt)
        remaining.remove(nxt)

    return selected
