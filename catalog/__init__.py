"""System catalog package."""

from catalog.catalog import SystemCatalog, empty_catalog
from catalog.doc_stats import DocStats, average_chunk_tokens, build_doc_stats
from catalog.indexer import build_system_catalog, load_catalog, save_catalog
from catalog.model_stats import (
    BUILTIN_MODEL_REGISTRY,
    DEFAULT_MODEL_ID,
    ModelStats,
    get_model_stats,
)
from catalog.semantic_stats import (
    SemanticStats,
    build_density_map,
    build_overlap_graph,
    build_semantic_stats,
    build_summarization_map,
    build_theme_clusters,
    query_hash,
)

__all__ = [
    "BUILTIN_MODEL_REGISTRY",
    "DEFAULT_MODEL_ID",
    "DocStats",
    "ModelStats",
    "SemanticStats",
    "SystemCatalog",
    "average_chunk_tokens",
    "build_density_map",
    "build_doc_stats",
    "build_overlap_graph",
    "build_semantic_stats",
    "build_summarization_map",
    "build_system_catalog",
    "build_theme_clusters",
    "empty_catalog",
    "get_model_stats",
    "load_catalog",
    "query_hash",
    "save_catalog",
]
