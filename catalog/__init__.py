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

__all__ = [
    "BUILTIN_MODEL_REGISTRY",
    "DEFAULT_MODEL_ID",
    "DocStats",
    "ModelStats",
    "SystemCatalog",
    "average_chunk_tokens",
    "build_doc_stats",
    "build_system_catalog",
    "empty_catalog",
    "get_model_stats",
    "load_catalog",
    "save_catalog",
]
