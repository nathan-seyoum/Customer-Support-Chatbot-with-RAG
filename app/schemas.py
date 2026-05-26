"""Pydantic models for the public API surface."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class Citation(BaseModel):
    chunk_id: str
    source: str
    score: float
    text: str


class HallucinationReport(BaseModel):
    flagged: bool
    score: float = Field(..., description="Aggregate groundedness in [0, 1] — higher is more grounded.")
    per_sentence: list[dict]
    threshold: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    hallucination: HallucinationReport
    latency_ms: dict


class IngestRequest(BaseModel):
    documents: list[dict] = Field(
        ...,
        description="List of {id, source, text, metadata?} entries.",
    )


class IngestResponse(BaseModel):
    ingested_documents: int
    ingested_chunks: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    components: dict[str, str]
