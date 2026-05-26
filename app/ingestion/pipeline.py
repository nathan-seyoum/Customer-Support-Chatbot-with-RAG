"""Ingestion pipeline: load → chunk → embed → upsert.

Kept separate from the query-time `RAGPipeline` so it can run as a one-shot
CLI, a background worker, or a future scheduled job — without booting the LLM.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings, get_settings
from app.ingestion.loaders import LoadedDocument, load_path
from app.providers import build_embedder
from app.rag.chunker import RecursiveCharacterTextSplitter
from app.stores import build_vector_store
from app.stores.base import Chunk

logger = logging.getLogger(__name__)


@dataclass
class IngestionReport:
    documents: int
    chunks: int


def _chunk_id(doc_id: str, idx: int, text: str) -> str:
    """Deterministic chunk id so re-ingesting the same content upserts rather
    than duplicates. Hash includes text so edits invalidate cleanly."""
    h = hashlib.sha1(f"{doc_id}::{idx}::{text}".encode("utf-8")).hexdigest()[:16]
    return f"{Path(doc_id).stem}-{idx:04d}-{h}"


class IngestionPipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.embedder = build_embedder(self.settings)
        self.store = build_vector_store(self.settings)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

    # ---------------------------------------------------------------- helpers
    def _documents_to_chunks(self, docs: list[LoadedDocument]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in docs:
            text_chunks = self.splitter.split(doc.text)
            for i, tc in enumerate(text_chunks):
                chunks.append(
                    Chunk(
                        id=_chunk_id(doc.id, i, tc.text),
                        text=tc.text,
                        source=doc.source,
                        metadata={"doc_id": doc.id, "chunk_index": i, **doc.metadata},
                    )
                )
        return chunks

    # ------------------------------------------------------------------- main
    def ingest_documents(self, docs: list[LoadedDocument]) -> IngestionReport:
        chunks = self._documents_to_chunks(docs)
        if not chunks:
            return IngestionReport(documents=len(docs), chunks=0)

        # Batch embedding for efficiency.
        embeddings = self.embedder.embed_documents([c.text for c in chunks])
        self.store.upsert(chunks, embeddings)

        logger.info(
            "ingestion_complete",
            extra={"documents": len(docs), "chunks": len(chunks)},
        )
        return IngestionReport(documents=len(docs), chunks=len(chunks))

    def ingest_path(self, path: Path) -> IngestionReport:
        return self.ingest_documents(load_path(path))
