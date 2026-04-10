"""Offline indexer: corpus -> system catalog JSON."""

from __future__ import annotations

import json
from pathlib import Path

from catalog.catalog import SystemCatalog
from catalog.doc_stats import DocStats, build_doc_stats
from catalog.model_stats import BUILTIN_MODEL_REGISTRY, ModelStats


def build_system_catalog(
    corpus,
    model_registry: dict[str, ModelStats] | None = None,
    default_model_id: str = "gpt-4o",
) -> SystemCatalog:
    """Build catalog from corpus in one offline step."""
    docs = build_doc_stats(corpus)
    registry = dict(model_registry or BUILTIN_MODEL_REGISTRY)
    return SystemCatalog(
        doc_stats=docs,
        model_stats=registry,
        default_model_id=default_model_id,
    )


def save_catalog(catalog: SystemCatalog, path: str | Path) -> None:
    """Persist catalog to JSON for read-only runtime loading."""
    payload = {
        "doc_stats": {k: v.to_dict() for k, v in catalog.doc_stats.items()},
        "model_stats": {k: v.to_dict() for k, v in catalog.model_stats.items()},
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

    return SystemCatalog(
        doc_stats=doc_stats,
        model_stats=model_stats,
        default_model_id=str(data.get("default_model_id", "gpt-4o")),
    )
