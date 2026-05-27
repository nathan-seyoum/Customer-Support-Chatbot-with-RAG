"""HTTP routes.

The pipeline is held in `app.state` and constructed once at startup (see
`main.py` lifespan) so heavy models (embedder + NLI) don't load per-request.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.ingestion.loaders import LoadedDocument
from app.ingestion.pipeline import IngestionPipeline
from app.rag.pipeline import PipelineAnswer, RAGPipeline
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


def _format_result_payload(result: PipelineAnswer) -> dict:
    return QueryResponse(
        answer=result.answer,
        citations=[
            Citation(
                chunk_id=rc.chunk.id,
                source=rc.chunk.source,
                score=round(rc.score, 4),
                text=rc.chunk.text,
            )
            for rc in result.chunks
        ],
        hallucination=HallucinationReport(**result.hallucination.to_payload()),
        latency_ms={k: round(v, 2) for k, v in result.latency_ms.items()},
    ).model_dump()


@router.post("/query/stream")
async def query_stream(req: QueryRequest, request: Request) -> StreamingResponse:
    """SSE variant of /query. Emits a `stage` event at the start of each
    pipeline step (retrieve / generate / detect) and a final `result` event
    carrying the same payload as POST /query."""
    pipeline = _get_pipeline(request)

    async def event_source():
        # The pipeline is sync and CPU/IO-bound, so we drive its iterator from
        # a thread to avoid blocking the event loop between stages.
        it = pipeline.ask_stream(req.question, top_k=req.top_k)
        sentinel = object()

        def _next():
            return next(it, sentinel)

        try:
            while True:
                event = await asyncio.to_thread(_next)
                if event is sentinel:
                    return
                event_type, data = event  # type: ignore[misc]
                if event_type == "result":
                    payload = _format_result_payload(data)  # type: ignore[arg-type]
                else:
                    payload = data  # type: ignore[assignment]
                yield f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"
        except Exception as e:
            logger.exception("query_stream_failed")
            yield f"event: error\ndata: {json.dumps({'detail': str(e)})}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")


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
