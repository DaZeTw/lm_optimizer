"""
Corpus and LLM abstractions.

Variants depend only on these two protocols — never on specific
SDKs or vector stores directly. This keeps variants testable
with simple in-memory doubles.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field
from typing import Protocol

from ir.evidence import Chunk


# ── Protocols ──────────────────────────────────────────────────────


class Corpus(Protocol):
    def bm25_search(self, query: str, top_k: int = 10) -> list[Chunk]: ...
    def dense_search(self, query: str, top_k: int = 10) -> list[Chunk]: ...
    def hybrid_search(
        self, query: str, top_k: int = 10, alpha: float = 0.5
    ) -> list[Chunk]: ...
    def embed(self, text: str) -> list[float]: ...


class LLM(Protocol):
    async def complete(self, system: str, user: str, max_tokens: int = 1024) -> str: ...


# ── In-memory corpus (unit tests) ─────────────────────────────────


@dataclass
class InMemoryCorpus:
    chunks: list[Chunk] = field(default_factory=list)

    def bm25_search(self, query: str, top_k: int = 10) -> list[Chunk]:
        terms = set(query.lower().split())
        scored = sorted(
            [
                (len(terms & set(c.text.lower().split())), c)
                for c in self.chunks
                if terms & set(c.text.lower().split())
            ],
            key=lambda x: -x[0],
        )
        return [c for _, c in scored[:top_k]]

    def dense_search(self, query: str, top_k: int = 10) -> list[Chunk]:
        return self.bm25_search(query, top_k)

    def hybrid_search(
        self, query: str, top_k: int = 10, alpha: float = 0.5
    ) -> list[Chunk]:
        return self.bm25_search(query, top_k)

    def embed(self, text: str) -> list[float]:
        h = int(hashlib.md5(text.encode()).hexdigest(), 16)
        vec = [(((h >> i) & 0xFF) / 255.0) - 0.5 for i in range(64)]
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


# ── Mock LLM (unit tests) ──────────────────────────────────────────


class MockLLM:
    def __init__(self, response: str = "Mock answer based on evidence."):
        self._response = response

    async def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        return self._response


# ── OpenAI LLM (production) ────────────────────────────────────────


class OpenAILLM:
    """
    Async OpenAI wrapper.

    Usage:
        llm = OpenAILLM()          # reads OPENAI_API_KEY from env
        answer = await llm.complete(system="...", user="...")
    """

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o"):
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self.model = model

    async def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""
