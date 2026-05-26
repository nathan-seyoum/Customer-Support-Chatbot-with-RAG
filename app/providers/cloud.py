"""Optional cloud provider implementations.

Imported lazily by `factory.py` only when a cloud provider is selected.
Each class fails fast with a clear error if its SDK isn't installed.
"""

from __future__ import annotations

from typing import Sequence


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self.model = model
        # Dimensions per https://platform.openai.com/docs/guides/embeddings/embedding-models
        self._known_dims = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        self.dimension = self._known_dims.get(model, 1536)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self.model, input=list(texts))
        return [d.embedding for d in resp.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class VoyageEmbedder:
    def __init__(self, model: str = "voyage-3", api_key: str | None = None):
        import voyageai

        self._client = voyageai.Client(api_key=api_key)
        self.model = model
        # voyage-3 is 1024-dim per https://docs.voyageai.com/docs/embeddings
        self.dimension = 1024

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._client.embed(list(texts), model=self.model, input_type="document").embeddings

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed([text], model=self.model, input_type="query").embeddings[0]


class OpenAILLM:
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, *, system: str, user: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()


class AnthropicLLM:
    def __init__(self, model: str = "claude-haiku-4-5-20251001", api_key: str | None = None):
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, *, system: str, user: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
        resp = self._client.messages.create(
            model=self.model,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": user}],
        )
        # Concatenate any text blocks in the response.
        return "".join(block.text for block in resp.content if block.type == "text").strip()
