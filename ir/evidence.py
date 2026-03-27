"""Runtime evidence types that flow between physical operators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    text: str
    doc_id: str
    section: str = ""
    span: tuple[int, int] = (0, 0)  # character offsets in source
    score: float = 0.0  # retrieval / reranking score
    metadata: dict[str, Any] = field(default_factory=dict)

    def token_estimate(self, chars_per_token: int = 4) -> int:
        return max(1, len(self.text) // chars_per_token)


@dataclass
class EvidenceSet:
    chunks: list[Chunk]
    query_ref: str = ""
    op_trace: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.chunks)

    def token_estimate(self) -> int:
        return sum(c.token_estimate() for c in self.chunks)

    def as_text(self, separator: str = "\n\n") -> str:
        return separator.join(c.text for c in self.chunks)

    def append_trace(self, op_name: str) -> EvidenceSet:
        """Return a new EvidenceSet with op_name added to the trace."""
        return EvidenceSet(
            chunks=list(self.chunks),
            query_ref=self.query_ref,
            op_trace=self.op_trace + [op_name],
        )
