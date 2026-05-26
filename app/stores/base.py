"""Vector-store protocol and shared data classes.

Same idea as `providers/base.py`: narrow protocol so the store is swappable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable


@dataclass
class Chunk:
    """A single retrievable unit (a chunk of a document)."""

    id: str
    text: str
    source: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float  # similarity in [0, 1] (we store with cosine-normalized vectors)


@runtime_checkable
class VectorStore(Protocol):
    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[list[float]]) -> None: ...

    def query(
        self,
        embedding: list[float],
        top_k: int,
        *,
        where: dict | None = None,
    ) -> list[RetrievedChunk]: ...

    def count(self) -> int: ...

    def reset(self) -> None: ...
