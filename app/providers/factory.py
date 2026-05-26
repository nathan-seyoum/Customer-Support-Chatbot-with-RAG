"""Provider factory.

The rest of the app calls `build_embedder()` / `build_llm()` and never touches
concrete classes. To add a new provider: implement the protocols in `base.py`
and add a branch here.
"""

from __future__ import annotations

from app.config import Settings, get_settings
from app.providers.base import Embedder, LLMClient


def build_embedder(settings: Settings | None = None) -> Embedder:
    settings = settings or get_settings()
    provider = settings.embedder_provider

    if provider == "local":
        from app.providers.local import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(model_name=settings.embedder_model)

    if provider == "openai":  # pragma: no cover - optional path
        try:
            from app.providers.cloud import OpenAIEmbedder  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "OpenAI embedder requested but `openai` extra is not installed. "
                "Install with `pip install support-rag[openai]`."
            ) from e
        return OpenAIEmbedder(model=settings.embedder_model, api_key=settings.openai_api_key)

    if provider == "voyage":  # pragma: no cover - optional path
        try:
            from app.providers.cloud import VoyageEmbedder  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "Voyage embedder requested but `voyageai` is not installed."
            ) from e
        return VoyageEmbedder(model=settings.embedder_model, api_key=settings.voyage_api_key)

    raise ValueError(f"Unknown embedder provider: {provider}")


def build_llm(settings: Settings | None = None) -> LLMClient:
    settings = settings or get_settings()
    provider = settings.llm_provider

    if provider == "ollama":
        from app.providers.local import OllamaLLM

        return OllamaLLM(model=settings.llm_model, base_url=settings.llm_base_url)

    if provider == "openai":  # pragma: no cover - optional path
        try:
            from app.providers.cloud import OpenAILLM  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "OpenAI LLM requested but `openai` extra is not installed."
            ) from e
        return OpenAILLM(model=settings.llm_model, api_key=settings.openai_api_key)

    if provider == "anthropic":  # pragma: no cover - optional path
        try:
            from app.providers.cloud import AnthropicLLM  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "Anthropic LLM requested but `anthropic` extra is not installed."
            ) from e
        return AnthropicLLM(model=settings.llm_model, api_key=settings.anthropic_api_key)

    raise ValueError(f"Unknown LLM provider: {provider}")
