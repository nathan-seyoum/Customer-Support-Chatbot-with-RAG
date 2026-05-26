"""Lightweight fakes for the provider/store protocols.

Real components depend on multi-hundred-megabyte HuggingFace downloads and an
Ollama server. Fakes let us exercise the orchestration logic deterministically
in milliseconds.
"""

from __future__ import annotations

import hashlib
import math
from typing import Sequence

from app.stores.base import Chunk, RetrievedChunk


def _hashed_vec(text: str, dim: int = 16) -> list[float]:
    """Deterministic pseudo-embedding: hash text into a fixed-length vector."""
    h = hashlib.sha256(text.lower().encode("utf-8")).digest()
    raw = [(h[i % len(h)] - 128) / 128.0 for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / norm for x in raw]


class FakeEmbedder:
    dimension = 16

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [_hashed_vec(t, self.dimension) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return _hashed_vec(text, self.dimension)


class FakeLLM:
    """LLM that echoes the first retrieved chunk so tests can assert grounding."""

    def __init__(self, canned: str | None = None):
        self.canned = canned

    def generate(self, *, system: str, user: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
        if self.canned is not None:
            return self.canned
        # Extract first chunk from the prompt for a "grounded" fake answer.
        marker = "[#1] (source:"
        if marker in user:
            after = user.split(marker, 1)[1]
            # Strip the trailing question + Answer: scaffolding.
            chunk_text = after.split("\n", 1)[1] if "\n" in after else after
            chunk_text = chunk_text.split("\n\n[#", 1)[0]
            return chunk_text.strip()[:200] + " [#1]"
        return "I don't have enough information in the knowledge base to answer that."


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._vecs: dict[str, list[float]] = {}

    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[list[float]]) -> None:
        for c, e in zip(chunks, embeddings):
            self._chunks[c.id] = c
            self._vecs[c.id] = e

    def query(self, embedding: list[float], top_k: int, *, where: dict | None = None) -> list[RetrievedChunk]:
        scored: list[tuple[float, str]] = []
        for cid, v in self._vecs.items():
            dot = sum(a * b for a, b in zip(embedding, v))
            scored.append((dot, cid))
        scored.sort(reverse=True)
        out: list[RetrievedChunk] = []
        for score, cid in scored[:top_k]:
            out.append(RetrievedChunk(chunk=self._chunks[cid], score=max(0.0, score)))
        return out

    def count(self) -> int:
        return len(self._chunks)

    def reset(self) -> None:
        self._chunks.clear()
        self._vecs.clear()
