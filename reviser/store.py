"""Step 2: normalize and accumulate SampleFeedback dicts.

FeedbackStore is a lightweight in-memory list of SampleFeedback dicts,
one per sample per iteration.  It also attaches the TST version that was
active when the sample ran, so the aggregator can compute deltas.

No database, no files — just a list that the pipeline appends to and the
aggregator reads from.  The pipeline can persist the store by serialising
store.all() to JSON if needed.
"""

from __future__ import annotations


class FeedbackStore:
    """Accumulate and retrieve SampleFeedback dicts.

    Each entry is a normalized record::

        {
            "iteration":   int,
            "tst_version": int,       # which revision of the TST was used
            "sample":      dict,      # SampleFeedback from SampleAnalyzer
        }

    Usage::

        store = FeedbackStore()
        store.add(iteration=0, tst_version=0, sample=sample_feedback)
        current_batch = store.current_iteration()
        all_records   = store.all()
    """

    def __init__(self) -> None:
        self._records: list[dict] = []

    # ── write ─────────────────────────────────────────────────────

    def add(self, iteration: int, tst_version: int, sample: dict) -> None:
        """Append one normalized SampleFeedback record.

        Args:
            iteration:   Pipeline iteration index (0-based).
            tst_version: Which revision of the TST was active (0 = initial).
            sample:      SampleFeedback dict from SampleAnalyzer.analyze().
        """
        self._records.append(
            {
                "iteration": iteration,
                "tst_version": tst_version,
                "sample": _normalize(sample),
            }
        )

    # ── read ──────────────────────────────────────────────────────

    def all(self) -> list[dict]:
        """Return all stored records."""
        return list(self._records)

    def by_iteration(self, iteration: int) -> list[dict]:
        """Return all records for a specific iteration."""
        return [r for r in self._records if r["iteration"] == iteration]

    def current_iteration(self) -> list[dict]:
        """Return records from the most recent iteration."""
        if not self._records:
            return []
        latest = max(r["iteration"] for r in self._records)
        return self.by_iteration(latest)

    def by_tst_version(self, tst_version: int) -> list[dict]:
        """Return all records that ran under a specific TST version."""
        return [r for r in self._records if r["tst_version"] == tst_version]

    def samples(self, iteration: int | None = None) -> list[dict]:
        """Return raw SampleFeedback dicts, optionally filtered by iteration."""
        records = (
            self.by_iteration(iteration) if iteration is not None else self._records
        )
        return [r["sample"] for r in records]

    def __len__(self) -> int:
        return len(self._records)


# ── Normalizer ────────────────────────────────────────────────────


def _normalize(sample: dict) -> dict:
    """Ensure all expected keys are present with sensible defaults.

    This makes downstream aggregation code safe to write without
    defensive get() calls on every field.
    """
    return {
        "query": sample.get("query", ""),
        "accuracy": float(sample.get("accuracy", 0.0)),
        "total_tokens": int(sample.get("total_tokens", 0)),
        "total_latency_ms": float(sample.get("total_latency_ms", 0.0)),
        "query_features": dict(sample.get("query_features", {})),
        "failure_points": list(sample.get("failure_points", [])),
        "successful_adaptations": list(sample.get("successful_adaptations", [])),
        "suggested_fixes": list(sample.get("suggested_fixes", [])),
    }
