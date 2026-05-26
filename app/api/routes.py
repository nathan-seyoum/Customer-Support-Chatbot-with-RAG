"""HTTP routes.

The pipeline is held in `app.state` and constructed once at startup (see
`main.py` lifespan) so heavy models (embedder + NLI) don't load per-request.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from app.ingestion.loaders import LoadedDocument
from app.ingestion.pipeline import IngestionPipeline
from app.rag.pipeline import RAGPipeline
from app.schemas import (
    Citation,
    HallucinationReport,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)

router = APIRouter()
logger = logging.getLogger(__name__)


def _get_pipeline(request: Request) -> RAGPipeline:
    pipeline: RAGPipeline | None = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    return pipeline


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    components: dict[str, str] = {}
    status = "ok"

    try:
        pipeline = _get_pipeline(request)
        components["retriever"] = "ok"
        components["vector_store_count"] = str(pipeline.retriever.store.count())
    except Exception as e:  # pragma: no cover
        components["retriever"] = f"error: {e}"
        status = "degraded"

    return HealthResponse(status=status, components=components)  # type: ignore[arg-type]


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, request: Request) -> QueryResponse:
    pipeline = _get_pipeline(request)
    try:
        result = pipeline.ask(req.question, top_k=req.top_k)
    except Exception as e:
        logger.exception("query_failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

    citations = [
        Citation(
            chunk_id=rc.chunk.id,
            source=rc.chunk.source,
            score=round(rc.score, 4),
            text=rc.chunk.text,
        )
        for rc in result.chunks
    ]
    return QueryResponse(
        answer=result.answer,
        citations=citations,
        hallucination=HallucinationReport(**result.hallucination.to_payload()),
        latency_ms={k: round(v, 2) for k, v in result.latency_ms.items()},
    )


@router.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    docs = [
        LoadedDocument(
            id=d.get("id") or d["source"],
            source=d["source"],
            text=d["text"],
            metadata=d.get("metadata", {}),
        )
        for d in req.documents
    ]
    pipeline = IngestionPipeline()
    report = pipeline.ingest_documents(docs)
    return IngestResponse(ingested_documents=report.documents, ingested_chunks=report.chunks)


@router.post("/ingest/path", response_model=IngestResponse)
def ingest_path(path: str) -> IngestResponse:
    """Ingest from a server-side path. Intended for local/dev use."""
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=400, detail=f"path not found: {path}")
    pipeline = IngestionPipeline()
    report = pipeline.ingest_path(p)
    return IngestResponse(ingested_documents=report.documents, ingested_chunks=report.chunks)
