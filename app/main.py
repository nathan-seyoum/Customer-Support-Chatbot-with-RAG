"""FastAPI application entry point.

Run locally:
    uvicorn app.main:app --reload --port 8000

The lifespan handler builds the RAG pipeline once at startup so the embedder
and NLI model are loaded a single time per worker process. If the vector store
is empty and INGEST_ON_STARTUP_PATH points at a directory, we ingest it once
so the demo is fully self-contained (`docker compose up` and you can ask
questions immediately).
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.config import get_settings, resolve_device
from app.ingestion.pipeline import IngestionPipeline
from app.rag.pipeline import RAGPipeline

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )


def _maybe_seed_knowledge_base(pipeline: RAGPipeline, logger: logging.Logger) -> None:
    seed_path = os.environ.get("INGEST_ON_STARTUP_PATH")
    if not seed_path:
        return
    if pipeline.retriever.store.count() > 0:
        logger.info("knowledge base already populated; skipping startup ingest")
        return
    p = Path(seed_path)
    if not p.exists():
        logger.warning("INGEST_ON_STARTUP_PATH=%s does not exist; skipping", seed_path)
        return
    logger.info("seeding knowledge base from %s", p)
    report = IngestionPipeline().ingest_path(p)
    logger.info("seeded %d documents / %d chunks", report.documents, report.chunks)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    _configure_logging(settings.log_level)
    logger = logging.getLogger("app.main")

    # Push HF_TOKEN into the process environment so huggingface_hub picks it up
    # for model downloads. pydantic-settings reads .env into our Settings but
    # doesn't propagate to os.environ on its own.
    if settings.hf_token and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = settings.hf_token
        logger.info("HF_TOKEN loaded from settings; authenticated downloads enabled")

    resolved_device = resolve_device(settings.device)
    logger.info(
        "starting pipeline: embedder=%s/%s llm=%s/%s store=%s device=%s (configured=%s)",
        settings.embedder_provider,
        settings.embedder_model,
        settings.llm_provider,
        settings.llm_model,
        settings.vector_store,
        resolved_device,
        settings.device,
    )
    pipeline = RAGPipeline.from_settings(settings)
    _maybe_seed_knowledge_base(pipeline, logger)

    # Pre-warm the NLI cross-encoder (~738MB) so the first /query doesn't hang
    # on a model download that routinely exceeds the frontend request timeout.
    logger.info("warming up hallucination detector (downloads NLI model on first run)…")
    try:
        pipeline.warmup()
    except Exception:
        logger.exception("warmup failed; first query will retry the load")

    app.state.pipeline = pipeline
    logger.info("pipeline ready; %d chunks in store", pipeline.retriever.store.count())

    yield

    logger.info("shutting down")


app = FastAPI(
    title="Customer-Support RAG",
    description="Retrieval-augmented chatbot with NLI-based hallucination detection.",
    version="0.1.0",
    lifespan=lifespan,
)

# ----- API routes are registered first so they win over the static catch-all.
app.include_router(api_router)


# ----- Static frontend: serve assets under /assets and index.html at /.
if WEB_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(WEB_DIR / "assets")), name="assets")

    @app.get("/", include_in_schema=False)
    def root() -> FileResponse:
        return FileResponse(str(WEB_DIR / "index.html"))
