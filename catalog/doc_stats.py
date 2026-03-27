"""Document-level corpus statistics used by planning and costing."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DocStats:
    doc_id: str
    total_chunks: int
    avg_chunk_tokens: float
    sections: list[str]
    total_tokens: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DocStats":
        return cls(
            doc_id=str(data.get("doc_id", "")),
            total_chunks=int(data.get("total_chunks", 0)),
            avg_chunk_tokens=float(data.get("avg_chunk_tokens", 0.0)),
            sections=list(data.get("sections", [])),
            total_tokens=int(data.get("total_tokens", 0)),
        )


def build_doc_stats(corpus) -> dict[str, DocStats]:
    """
    Scan corpus chunks and build per-document summary stats.

    The function relies only on a `chunks` iterable with `doc_id`, `section`, and
    `token_estimate()` support (as provided by `InMemoryCorpus`).
    """
    docs: dict[str, list] = {}
    for chunk in getattr(corpus, "chunks", []):
        docs.setdefault(chunk.doc_id, []).append(chunk)

    output: dict[str, DocStats] = {}
    for doc_id, chunks in docs.items():
        token_counts = [max(1, c.token_estimate()) for c in chunks]
        sections = sorted({c.section for c in chunks if c.section})
        total_chunks = len(chunks)
        total_tokens = sum(token_counts)
        avg_chunk_tokens = (total_tokens / total_chunks) if total_chunks else 0.0
        output[doc_id] = DocStats(
            doc_id=doc_id,
            total_chunks=total_chunks,
            avg_chunk_tokens=avg_chunk_tokens,
            sections=sections,
            total_tokens=total_tokens,
        )

    return output


def average_chunk_tokens(
    doc_stats: dict[str, DocStats], default: float = 180.0
) -> float:
    """Return global average chunk size across all documents."""
    if not doc_stats:
        return default

    total_tokens = sum(ds.total_tokens for ds in doc_stats.values())
    total_chunks = sum(ds.total_chunks for ds in doc_stats.values())
    if total_chunks == 0:
        return default
    return total_tokens / total_chunks
