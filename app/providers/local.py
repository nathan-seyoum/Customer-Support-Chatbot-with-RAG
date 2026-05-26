"""Local provider implementations.

- Embeddings: sentence-transformers (BGE family by default).
- LLM: Ollama via its official Python client.

Both classes are lazy: heavy models are loaded on first use so that importing
the module is cheap (matters for tests, CLIs, and FastAPI worker boot).
"""

from __future__ import annotations

from typing import Sequence


class SentenceTransformerEmbedder:
    """Industry-standard local embedder backed by sentence-transformers.

    The BGE family (BAAI/bge-*) consistently ranks at or near the top of the
    MTEB benchmark for its parameter count and is the de-facto local default.
    BGE expects a short instruction prefix for *query* embeddings, which we
    apply transparently.
    """

    # Models that benefit from BGE's query instruction.
    _BGE_QUERY_INSTRUCTION = (
        "Represent this sentence for searching relevant passages: "
    )

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model = None  # lazy
        self._dimension: int | None = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            # Imported lazily so importing this module stays cheap.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self._dimension = int(self._model.get_sentence_embedding_dimension())

    @property
    def dimension(self) -> int:
        self._ensure_loaded()
        assert self._dimension is not None
        return self._dimension

    def _is_bge(self) -> bool:
        return "bge" in self.model_name.lower()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self._ensure_loaded()
        # normalize_embeddings=True so cosine similarity == dot product later.
        vecs = self._model.encode(
            list(texts),
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        self._ensure_loaded()
        prompt = (self._BGE_QUERY_INSTRUCTION + text) if self._is_bge() else text
        vec = self._model.encode(
            prompt,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vec.tolist()


class OllamaLLM:
    """Local LLM client backed by Ollama (https://ollama.com).

    Ollama exposes a stable HTTP API and a thin Python wrapper. We use the
    `/api/chat` endpoint via the official client so we get the model's chat
    template applied server-side.
    """

    def __init__(self, model: str = "gemma3:1b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self._client = None  # lazy

    def _ensure_client(self):
        if self._client is None:
            from ollama import Client

            self._client = Client(host=self.base_url)
        return self._client

    def generate(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        client = self._ensure_client()
        response = client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )
        # Ollama returns {"message": {"role": "assistant", "content": "..."}, ...}
        return response["message"]["content"].strip()
