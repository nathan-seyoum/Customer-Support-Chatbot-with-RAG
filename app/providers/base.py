"""Provider protocols.

Defines the narrow interface every embedder and every LLM client must satisfy.
Anything that follows these protocols is a drop-in replacement — no other code
in the application is allowed to depend on a concrete provider.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Maps strings to dense vectors."""

    dimension: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed many documents. Should be batched internally for efficiency."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query. Some models prepend a query instruction prefix."""
        ...


@runtime_checkable
class LLMClient(Protocol):
    """Generates text from a system + user prompt."""

    def generate(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        ...
