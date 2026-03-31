"""Observed runtime telemetry store.

This module intentionally stores only metrics observed from real sample runs.
No heuristic fallback estimation is applied.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TelemetryEstimate:
    token_cost: float
    call_cost: float
    latency_cost: float
    quality_risk: float
    sample_count: int
    accuracy_score: float | None = None
    key_metric: str = ""
    saturation: float = 0.0


class TelemetryStore:
    """JSON-backed store for observed `(op, variant)` runtime metrics."""

    def __init__(self, path: Path | str | None = None):
        self._path = Path(path) if path else Path(__file__).parent / "telemetry.json"
        self._data: dict[str, Any] | None = None

    def record(
        self,
        *,
        op: str,
        variant: str,
        token_cost: float,
        call_cost: float,
        latency_cost: float,
        quality_risk: float,
        accuracy_score: float | None = None,
    ) -> None:
        data = self._load()
        op_node = data.setdefault(op, {})
        entry = op_node.setdefault(
            variant,
            {
                "samples": 0,
                "token_cost_sum": 0.0,
                "call_cost_sum": 0.0,
                "latency_cost_sum": 0.0,
                "quality_risk_sum": 0.0,
                "accuracy_score_sum": 0.0,
                "accuracy_score_count": 0,
            },
        )

        entry["samples"] += 1
        entry["token_cost_sum"] += float(token_cost)
        entry["call_cost_sum"] += float(call_cost)
        entry["latency_cost_sum"] += float(latency_cost)
        entry["quality_risk_sum"] += float(quality_risk)
        if accuracy_score is not None:
            entry["accuracy_score_sum"] += float(accuracy_score)
            entry["accuracy_score_count"] += 1

        self._save(data)

    def estimate(self, *, op: str, variant: str) -> TelemetryEstimate | None:
        sample = self._load().get(op, {}).get(variant)
        if not isinstance(sample, dict):
            return None

        samples = int(sample.get("samples", 0))
        if samples <= 0:
            return None

        accuracy_count = int(sample.get("accuracy_score_count", 0))
        accuracy_score: float | None = None
        if accuracy_count > 0:
            accuracy_score = float(sample.get("accuracy_score_sum", 0.0)) / float(
                accuracy_count
            )

        return TelemetryEstimate(
            token_cost=float(sample.get("token_cost_sum", 0.0)) / float(samples),
            call_cost=float(sample.get("call_cost_sum", 0.0)) / float(samples),
            latency_cost=float(sample.get("latency_cost_sum", 0.0)) / float(samples),
            quality_risk=float(sample.get("quality_risk_sum", 0.0)) / float(samples),
            sample_count=samples,
            accuracy_score=accuracy_score,
            key_metric=f"observed samples={samples}",
        )

    def has_data(self, *, op: str, variant: str) -> bool:
        return self.estimate(op=op, variant=variant) is not None

    def _load(self) -> dict[str, Any]:
        if self._data is not None:
            return self._data
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(self._data, dict):
                    return self._data
            except (OSError, json.JSONDecodeError):
                pass
        self._data = {}
        return self._data

    def _save(self, data: dict[str, Any]) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass


default_telemetry = TelemetryStore()
