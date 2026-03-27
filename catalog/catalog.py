"""Unified system catalog combining doc/model/semantic stats."""

from __future__ import annotations

from dataclasses import dataclass, field

from catalog.doc_stats import DocStats, average_chunk_tokens
from catalog.model_stats import (
    BUILTIN_MODEL_REGISTRY,
    DEFAULT_MODEL_ID,
    ModelStats,
    get_model_stats,
)
from catalog.semantic_stats import SemanticStats, query_hash


@dataclass(frozen=True)
class SystemCatalog:
    doc_stats: dict[str, DocStats] = field(default_factory=dict)
    model_stats: dict[str, ModelStats] = field(
        default_factory=lambda: dict(BUILTIN_MODEL_REGISTRY)
    )
    semantic_stats: SemanticStats = field(default_factory=SemanticStats)
    default_model_id: str = DEFAULT_MODEL_ID

    def get_doc(self, doc_id: str) -> DocStats | None:
        return self.doc_stats.get(doc_id)

    def avg_chunk_tokens(self, default: float = 180.0) -> float:
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

    def density_for_query(self, query: str, default: float = 0.2) -> float:
        qh = query_hash(query)
        if qh in self.semantic_stats.density_map:
            return self.semantic_stats.density_map[qh]
        return default

    def cached_summary(self, doc_id: str, section: str) -> str | None:
        key = f"{doc_id}/{section}" if section else doc_id
        return self.semantic_stats.summarization_map.get(key)

    def cached_summary_by_key(self, key: str) -> str | None:
        return self.semantic_stats.summarization_map.get(key)

    def estimated_overlap_ratio(self, default: float = 0.2) -> float:
        graph = self.semantic_stats.overlap_graph
        if not graph:
            return default

        total_nodes = len(graph)
        if total_nodes == 0:
            return default

        # Average normalized neighborhood size as overlap proxy.
        total_neighbors = sum(len(v) for v in graph.values())
        # Scale by node count to keep ratio in [0, 1].
        ratio = min(1.0, total_neighbors / max(1, total_nodes * 4))
        return ratio


def empty_catalog() -> SystemCatalog:
    return SystemCatalog()
