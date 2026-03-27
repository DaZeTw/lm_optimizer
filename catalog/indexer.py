"""Offline indexer: corpus -> full system catalog JSON."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from catalog.catalog import SystemCatalog
from catalog.doc_stats import DocStats, build_doc_stats
from catalog.model_stats import BUILTIN_MODEL_REGISTRY, ModelStats
from catalog.semantic_stats import SemanticStats, build_semantic_stats


async def build_system_catalog(
    corpus,
    llm=None,
    model_registry: dict[str, ModelStats] | None = None,
    default_model_id: str = "gpt-4o",
) -> SystemCatalog:
    """Build all catalog components in one offline step."""
    docs = build_doc_stats(corpus)
    semantic = await build_semantic_stats(corpus, llm=llm)
    registry = dict(model_registry or BUILTIN_MODEL_REGISTRY)
    return SystemCatalog(
        doc_stats=docs,
        model_stats=registry,
        semantic_stats=semantic,
        default_model_id=default_model_id,
    )


def save_catalog(catalog: SystemCatalog, path: str | Path) -> None:
    """Persist catalog to JSON for read-only runtime loading."""
    payload = {
        "doc_stats": {k: v.to_dict() for k, v in catalog.doc_stats.items()},
        "model_stats": {k: v.to_dict() for k, v in catalog.model_stats.items()},
        "semantic_stats": catalog.semantic_stats.to_dict(),
        "default_model_id": catalog.default_model_id,
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_catalog(path: str | Path) -> SystemCatalog:
    """Load a persisted catalog JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    doc_stats = {
        k: DocStats.from_dict(v) for k, v in dict(data.get("doc_stats", {})).items()
    }
    model_stats = {
        k: ModelStats.from_dict(v) for k, v in dict(data.get("model_stats", {})).items()
    }
    semantic = SemanticStats.from_dict(dict(data.get("semantic_stats", {})))

    return SystemCatalog(
        doc_stats=doc_stats,
        model_stats=model_stats,
        semantic_stats=semantic,
        default_model_id=str(data.get("default_model_id", "gpt-4o")),
    )
