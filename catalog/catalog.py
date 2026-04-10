"""Unified system catalog: model stats + corpus doc stats."""

from __future__ import annotations

from dataclasses import dataclass, field

from catalog.doc_stats import DocStats, average_chunk_tokens
from catalog.model_stats import (
    BUILTIN_MODEL_REGISTRY,
    DEFAULT_MODEL_ID,
    ModelStats,
    get_model_stats,
)


@dataclass(frozen=True)
class SystemCatalog:
    doc_stats: dict[str, DocStats] = field(default_factory=dict)
    model_stats: dict[str, ModelStats] = field(
        default_factory=lambda: dict(BUILTIN_MODEL_REGISTRY)
    )
    default_model_id: str = DEFAULT_MODEL_ID

    def get_doc(self, doc_id: str) -> DocStats | None:
        return self.doc_stats.get(doc_id)

    def avg_chunk_tokens(self, default: float = 180.0) -> float:
        """Average chunk token size across the testing corpus."""
        return average_chunk_tokens(self.doc_stats, default=default)

    def get_model(self, model_id: str | None = None) -> ModelStats:
        mid = model_id or self.default_model_id
        return get_model_stats(
            mid, registry=self.model_stats, default_model_id=self.default_model_id
        )

    def context_window(
        self, model_id: str | None = None, default: int = 128_000
    ) -> int:
        model = self.get_model(model_id)
        return model.context_window or default


def empty_catalog() -> SystemCatalog:
    return SystemCatalog()
