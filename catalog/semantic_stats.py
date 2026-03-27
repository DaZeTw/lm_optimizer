"""Semantic corpus statistics and offline builders."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ir.evidence import Chunk


@dataclass(frozen=True)
class SemanticStats:
    theme_clusters: dict[str, list[str]] = field(default_factory=dict)
    summarization_map: dict[str, str] = field(default_factory=dict)
    density_map: dict[str, float] = field(default_factory=dict)
    overlap_graph: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SemanticStats":
        return cls(
            theme_clusters=dict(data.get("theme_clusters", {})),
            summarization_map=dict(data.get("summarization_map", {})),
            density_map={
                k: float(v) for k, v in dict(data.get("density_map", {})).items()
            },
            overlap_graph={
                k: list(v) for k, v in dict(data.get("overlap_graph", {})).items()
            },
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> "SemanticStats":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def to_json_file(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def query_hash(query: str) -> str:
    return hashlib.md5(query.strip().lower().encode()).hexdigest()[:12]


def chunk_key(chunk: Chunk) -> str:
    return f"{chunk.doc_id}:{chunk.section}:{chunk.span[0]}:{chunk.span[1]}"


def summary_key(doc_id: str, section: str) -> str:
    return f"{doc_id}/{section}" if section else doc_id


async def build_summarization_map(
    corpus, llm=None, max_tokens: int = 256
) -> dict[str, str]:
    """
    Build section-level summaries: one summary per `doc_id/section`.

    When `llm` is not provided, this returns a deterministic extractive fallback
    so offline indexing can still run in test/local environments.
    """
    grouped: dict[str, list[str]] = {}
    for c in getattr(corpus, "chunks", []):
        key = summary_key(c.doc_id, c.section)
        grouped.setdefault(key, []).append(c.text)

    out: dict[str, str] = {}
    for key, texts in grouped.items():
        source = "\n\n".join(texts)
        if llm is None:
            out[key] = source[:700].strip()
            continue

        summary = await llm.complete(
            system="Summarize this document section accurately and concisely.",
            user=f"Section key: {key}\n\nSource text:\n{source}",
            max_tokens=max_tokens,
        )
        out[key] = summary.strip()

    return out


def build_theme_clusters(corpus, min_token_len: int = 4) -> dict[str, list[str]]:
    """
    Build a lightweight theme map from section labels.

    This is intentionally simple and dependency-free. A richer embedding-based
    clustering pipeline can replace this function in the offline indexer.
    """
    clusters: dict[str, list[str]] = {}
    for c in getattr(corpus, "chunks", []):
        label = (c.section or "misc").strip().lower()
        if len(label) < min_token_len:
            label = "misc"
        clusters.setdefault(label, []).append(chunk_key(c))
    return clusters


def build_density_map(corpus) -> dict[str, float]:
    """Estimate retrieval density by section share as a proxy for selectivity."""
    counts: dict[str, int] = {}
    chunks = list(getattr(corpus, "chunks", []))
    total = max(1, len(chunks))

    for c in chunks:
        section = (c.section or "misc").strip().lower()
        counts[section] = counts.get(section, 0) + 1

    density: dict[str, float] = {}
    for section, count in counts.items():
        density[query_hash(section)] = count / total
    return density


def build_overlap_graph(corpus) -> dict[str, list[str]]:
    """
    Build a simple overlap graph keyed by chunk id.

    Chunks sharing the same doc_id and section are considered candidates for
    semantic overlap in this lightweight implementation.
    """
    by_bucket: dict[tuple[str, str], list[str]] = {}
    for c in getattr(corpus, "chunks", []):
        key = chunk_key(c)
        bucket = (c.doc_id, c.section)
        by_bucket.setdefault(bucket, []).append(key)

    graph: dict[str, list[str]] = {}
    for ids in by_bucket.values():
        for cid in ids:
            graph[cid] = [other for other in ids if other != cid]
    return graph


async def build_semantic_stats(corpus, llm=None) -> SemanticStats:
    """Build the full semantic stats bundle for offline indexing."""
    return SemanticStats(
        theme_clusters=build_theme_clusters(corpus),
        summarization_map=await build_summarization_map(corpus, llm=llm),
        density_map=build_density_map(corpus),
        overlap_graph=build_overlap_graph(corpus),
    )
