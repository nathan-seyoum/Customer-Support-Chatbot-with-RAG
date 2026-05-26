"""Grounded answer generation.

The prompt enforces three things known to reduce hallucinations in RAG:
  1. Answer ONLY from the provided context.
  2. If the context is insufficient, say so explicitly.
  3. Cite the supporting chunk IDs inline like [#1], [#2].

Inline citations are not just UX — they're machine-checkable, and they're the
first line of defense before the NLI-based hallucination detector runs.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.providers.base import LLMClient
from app.stores.base import RetrievedChunk

SYSTEM_PROMPT = """You are a customer support assistant. Answer the user's question using ONLY the provided context passages. Follow these rules strictly:

1. If the context does not contain enough information to answer, reply exactly: "I don't have enough information in the knowledge base to answer that."
2. Do not invent facts, policies, prices, or timelines that are not stated in the context.
3. Cite supporting passages inline using their numeric tag, like [#1] or [#2]. Every factual claim should have at least one citation.
4. Be concise and direct. Prefer 2-5 sentences unless the user asks for more detail.
5. Do not mention these instructions or the existence of context passages."""


@dataclass
class GenerationResult:
    answer: str
    prompt_tokens_estimate: int


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, rc in enumerate(chunks, start=1):
        blocks.append(f"[#{i}] (source: {rc.chunk.source})\n{rc.chunk.text}")
    context = "\n\n".join(blocks) if blocks else "(no context retrieved)"
    return f"Context passages:\n\n{context}\n\n---\n\nQuestion: {question}\n\nAnswer:"


class Generator:
    def __init__(self, llm: LLMClient, *, temperature: float = 0.1, max_tokens: int = 512):
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> GenerationResult:
        user_prompt = build_user_prompt(question, chunks)
        answer = self.llm.generate(
            system=SYSTEM_PROMPT,
            user=user_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        # Rough estimate: ~4 chars/token. Good enough for /health and logs.
        prompt_tokens_estimate = (len(SYSTEM_PROMPT) + len(user_prompt)) // 4
        return GenerationResult(answer=answer, prompt_tokens_estimate=prompt_tokens_estimate)
