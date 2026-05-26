"""ChromaDB-backed vector store.

Why Chroma:
  - Embedded (no server) — keeps the demo single-process and reproducible.
  - First-class metadata filters (we'll need this for per-source filtering and
    later for things like access control or freshness).
  - Persistent on disk via a single `PersistentClient(path=...)` call.
  - Apache 2.0, broad adoption in LangChain/LlamaIndex ecosystems.

We bring our own embeddings (no embedding function registered with Chroma) so
that the embedder remains a swappable provider — Chroma is only a storage +
ANN engine here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from app.stores.base import Chunk, RetrievedChunk


class ChromaVectorStore:
    def __init__(self, persist_dir: Path | str, collection_name: str):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        # Cosine distance — pairs cleanly with our L2-normalized embeddings.
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------ upsert
    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")

        ids = [c.id for c in chunks]
        documents = [c.text for c in chunks]
        # Chroma requires scalar metadata values — flatten anything richer.
        metadatas = [
            {"source": c.source, **{k: v for k, v in c.metadata.items() if isinstance(v, (str, int, float, bool))}}
            for c in chunks
        ]
        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=list(embeddings),
            metadatas=metadatas,
        )

    # ------------------------------------------------------------------- query
    def query(
        self,
        embedding: list[float],
        top_k: int,
        *,
        where: dict | None = None,
    ) -> list[RetrievedChunk]:
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        out: list[RetrievedChunk] = []
        for i, _id in enumerate(ids):
            meta = metas[i] or {}
            source = meta.pop("source", "unknown")
            # Chroma returns cosine *distance* in [0, 2]; convert to similarity.
            similarity = max(0.0, 1.0 - float(dists[i]))
            out.append(
                RetrievedChunk(
                    chunk=Chunk(id=_id, text=docs[i], source=source, metadata=dict(meta)),
                    score=similarity,
                )
            )
        return out

    # ----------------------------------------------------------------- utility
    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
