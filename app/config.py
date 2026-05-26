"""Centralized configuration.

All tunables live here behind pydantic-settings so the app is fully driven by
environment variables / .env. Swapping providers, models, or stores is a config
change — never a code change.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    app_env: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"

    # Embedder
    embedder_provider: Literal["local", "openai", "voyage"] = "local"
    embedder_model: str = "BAAI/bge-small-en-v1.5"

    # LLM
    llm_provider: Literal["ollama", "openai", "anthropic"] = "ollama"
    llm_model: str = "gemma3:1b"
    llm_base_url: str = "http://localhost:11434"

    # NLI (hallucination detection)
    nli_model: str = "cross-encoder/nli-deberta-v3-base"

    # Vector store
    vector_store: Literal["chroma"] = "chroma"
    chroma_persist_dir: Path = Path("./data/chroma")
    chroma_collection: str = "support_kb"

    # Retrieval / generation
    chunk_size: int = Field(default=600, ge=128, le=4096)
    chunk_overlap: int = Field(default=100, ge=0, le=1024)
    top_k: int = Field(default=5, ge=1, le=50)
    mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0)
    hallucination_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    # Optional keys
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    voyage_api_key: str | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
