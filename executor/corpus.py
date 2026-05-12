"""
Corpus and LLM abstractions.

Variants depend only on these two protocols — never on specific
SDKs or vector stores directly.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

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


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


# ── Tokenization ───────────────────────────────────────────────────


def _tokenize(text: str) -> list[str]:
    """
    Normalize to lowercase alphanumeric tokens, preserving hyphenated
    compounds (e.g. 'state-of-the-art').  Strips all punctuation so
    'accuracy,' and 'accuracy' hash to the same token.

    Used at both index time and query time so the vocabulary is consistent.
    """
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", text.lower())


# ── Chunking ───────────────────────────────────────────────────────


def chunk_text(
    text: str,
    chunk_size: int = 150,
    overlap: int = 30,
    source: str = "",
) -> list[Chunk]:
    sentences: list[str] = re.split(r"(?<=[.!?])\s+", text.strip())

    chunks: list[Chunk] = []
    current_words: list[str] = []
    chunk_idx = 0

    for sentence in sentences:
        words = sentence.split()
        if current_words and len(current_words) + len(words) > chunk_size:
            chunks.append(
                Chunk(
                    doc_id=source,
                    text=" ".join(current_words),
                    metadata={"source": source, "chunk_index": chunk_idx},
                )
            )
            chunk_idx += 1
            current_words = current_words[-overlap:] if overlap else []
        current_words.extend(words)

    if current_words:
        chunks.append(
            Chunk(
                doc_id=source,
                text=" ".join(current_words),
                metadata={"source": source, "chunk_index": chunk_idx},
            )
        )

    return chunks


# ── Vector math ────────────────────────────────────────────────────


def _cosine(a: list[float], b: list[float]) -> float:
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)

    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0

    return float(np.dot(a_arr, b_arr) / denom)


# ── In-memory corpus ───────────────────────────────────────────────


@dataclass
class InMemoryCorpus:
    """
    Production-grade in-memory corpus. Suitable up to ~50K chunks.

    embedder is required — dense_search and hybrid_search raise
    RuntimeError immediately if the corpus is empty rather than
    silently falling back to BM25.

    Requires:
        pip install rank-bm25

    Usage:
        corpus = InMemoryCorpus(embedder=OpenAIEmbedder())
        corpus.add_documents(raw_texts, source="my_docs")
    """

    embedder: Embedder
    chunks: list[Chunk] = field(default_factory=list)
    _vectors: dict[int, list[float]] = field(default_factory=dict, repr=False)
    # BM25Okapi instance; rebuilt after each add_documents call.
    # Typed as object to avoid a hard import-time dependency in the annotation.
    _bm25: object = field(default=None, repr=False)

    def add_documents(
        self,
        texts: list[str],
        chunk_size: int = 150,
        overlap: int = 30,
        source: str = "",
    ) -> None:
        """Chunk, embed, and index all texts."""
        for text in texts:
            for chunk in chunk_text(
                text, chunk_size=chunk_size, overlap=overlap, source=source
            ):
                idx = len(self.chunks)
                self.chunks.append(chunk)
                self._vectors[idx] = self.embedder.embed(chunk.text)

        # Rebuild BM25 over the full corpus so far.
        # BM25Okapi takes a list of pre-tokenised documents.
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in self.chunks])

    def bm25_search(self, query: str, top_k: int = 10) -> list[Chunk]:
        """
        BM25Okapi retrieval via rank-bm25.

        _tokenize() is applied to the query for consistency with the
        index, so punctuation differences never cause missed matches.
        Only chunks with a positive score are returned.
        """
        if not self.chunks or self._bm25 is None:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)  # numpy array, one score per chunk
        ranked_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [self.chunks[i] for i in ranked_indices if scores[i] > 0]

    def dense_search(self, query: str, top_k: int = 10) -> list[Chunk]:
        """Cosine-similarity search over pre-computed embeddings."""
        if not self._vectors:
            raise RuntimeError(
                "dense_search called on an empty corpus — call add_documents first."
            )
        q_vec = self.embedder.embed(query)
        scored = [
            (_cosine(q_vec, self._vectors[i]), self.chunks[i]) for i in self._vectors
        ]
        scored.sort(key=lambda x: -x[0])
        return [c for _, c in scored[:top_k]]

    def hybrid_search(
        self, query: str, top_k: int = 10, alpha: float = 0.5
    ) -> list[Chunk]:
        """
        Reciprocal Rank Fusion of BM25 + dense.
        alpha=0.0 → pure BM25, alpha=1.0 → pure dense.
        """
        if not self._vectors:
            raise RuntimeError(
                "hybrid_search called on an empty corpus — call add_documents first."
            )
        RRF_K = 60
        bm25_hits = self.bm25_search(query, top_k=top_k * 2)
        dense_hits = self.dense_search(query, top_k=top_k * 2)

        scores: dict[int, float] = {}

        def _rrf(hits: list[Chunk], weight: float) -> None:
            for rank, chunk in enumerate(hits):
                cid = id(chunk)
                scores[cid] = scores.get(cid, 0.0) + weight / (RRF_K + rank + 1)

        _rrf(bm25_hits, 1.0 - alpha)
        _rrf(dense_hits, alpha)

        id_to_chunk = {id(c): c for c in bm25_hits + dense_hits}
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [id_to_chunk[cid] for cid, _ in ranked[:top_k]]

    def embed(self, text: str) -> list[float]:
        return self.embedder.embed(text)


# ── OpenAI Embedder ────────────────────────────────────────────────


class OpenAIEmbedder:
    """
    text-embedding-3-small — cheapest OpenAI embedder, beats ada-002 on MTEB.

    Pass dimensions=512 to halve vector storage cost with minimal
    quality loss (Matryoshka training makes this lossless truncation).

    Usage:
        embedder = OpenAIEmbedder()            # reads OPENAI_API_KEY
        embedder = OpenAIEmbedder(dimensions=512)  # smaller vectors
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
    ):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self.model = model
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        kwargs: dict = {"model": self.model, "input": text}
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        return self._client.embeddings.create(**kwargs).data[0].embedding


# ── OpenAI LLM ────────────────────────────────────────────────────


class OpenAILLM:
    """
    Async OpenAI chat-completion wrapper.

    Usage:
        llm = OpenAILLM()                        # gpt-4o, reads OPENAI_API_KEY
        llm = OpenAILLM(model="gpt-4o-mini")     # cheaper for generation
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
