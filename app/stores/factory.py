"""Vector-store factory."""

from __future__ import annotations

from app.config import Settings, get_settings
from app.stores.base import VectorStore


def build_vector_store(settings: Settings | None = None) -> VectorStore:
    settings = settings or get_settings()

    if settings.vector_store == "chroma":
        from app.stores.chroma_store import ChromaVectorStore

        return ChromaVectorStore(
            persist_dir=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection,
        )

    raise ValueError(f"Unknown vector store: {settings.vector_store}")
