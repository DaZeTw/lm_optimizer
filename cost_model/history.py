"""
Execution history log — JSON-backed store of past operator runs.

The profiler reads from this to get real compression ratios,
failure rates, and average chunk sizes.
The executor writes to it after each run via record().

Default file location: lm_optimizer/cost_model/history.json
Override with COST_HISTORY_PATH environment variable.

Schema:
{
  "TRANSFORM": {
    "default":  {"compression_ratios": [0.28, 0.31], "avg": 0.295},
    "extract":  {"compression_ratios": [0.15, 0.12], "avg": 0.135}
  },
  "VERIFY": {
    "multi_hop":  {"failure_rates": [0.30, 0.25], "avg": 0.275},
    "structured": {"failure_rates": [0.20],        "avg": 0.200},
    "simple_qa":  {"failure_rates": [0.08, 0.12],  "avg": 0.100}
  },
  "I": {
    "avg_chunk_tokens": 180,
    "default_top_k":    10
  }
}
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# ── Defaults used when no history file exists yet ──────────────────
_DEFAULTS: dict[str, Any] = {
    "TRANSFORM": {
        "default": {"avg": 0.30},
        "extract": {"avg": 0.15},
        "summarize": {"avg": 0.25},
    },
    "VERIFY": {
        "multi_hop": {"avg": 0.30},
        "structured": {"avg": 0.20},
        "simple_qa": {"avg": 0.10},
    },
    "I": {
        "avg_chunk_tokens": 180,
        "default_top_k": 10,
    },
    "AGGREGATE": {
        "avg_output_tokens": 256,
    },
}

_MAX_SAMPLES = 100  # keep rolling window to avoid unbounded growth


def _default_path() -> Path:
    env = os.environ.get("COST_HISTORY_PATH")
    if env:
        return Path(env)
    return Path(__file__).parent / "history.json"


class HistoryLog:
    """
    Thin wrapper around a JSON file that stores execution statistics.

    All reads go through get(); all writes go through record().
    The file is only written when record() is called — reads never
    create or modify the file.
    """

    def __init__(self, path: Path | str | None = None):
        self._path = Path(path) if path else _default_path()
        self._data: dict[str, Any] | None = None  # lazy load

    # ── Public API ─────────────────────────────────────────────────

    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Retrieve a nested value by dotted key path.

        Examples:
            log.get("TRANSFORM", "default", "avg")   → 0.30
            log.get("I", "avg_chunk_tokens")          → 180
            log.get("missing", default=0.5)           → 0.5
        """
        data = self._load()
        node = data
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    def record(self, op: str, subkey: str, metric: str, value: float) -> None:
        """
        Append a new observed value to the rolling history and
        recompute the running average.

        Example:
            log.record("TRANSFORM", "summarize", "compression_ratios", 0.27)
        """
        data = self._load()
        data.setdefault(op, {}).setdefault(subkey, {})
        entry = data[op][subkey]

        samples_key = metric  # e.g. "compression_ratios"
        samples = entry.get(samples_key, [])
        samples.append(round(value, 4))

        # Rolling window
        if len(samples) > _MAX_SAMPLES:
            samples = samples[-_MAX_SAMPLES:]

        entry[samples_key] = samples
        entry["avg"] = round(sum(samples) / len(samples), 4)

        self._save(data)

    def reset(self) -> None:
        """Clear all recorded history (useful for testing)."""
        import copy

        self._data = copy.deepcopy(
            _DEFAULTS
        )  # deep copy so nested dicts are fully independent
        self._save(self._data)

    # ── Internal ───────────────────────────────────────────────────

    def _load(self) -> dict[str, Any]:
        import copy

        if self._data is not None:
            return self._data
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text())
                # Merge in any missing default keys — deepcopy so _DEFAULTS is never mutated
                for k, v in _DEFAULTS.items():
                    if k not in self._data:
                        self._data[k] = copy.deepcopy(v)
                return self._data
            except (json.JSONDecodeError, OSError):
                pass
        # No file or corrupt — start from defaults (deepcopy so _DEFAULTS is never mutated)
        self._data = copy.deepcopy(_DEFAULTS)
        return self._data

    def _save(self, data: dict) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(data, indent=2))
        except OSError:
            pass  # read-only filesystem — silently continue


# ── Module-level singleton ─────────────────────────────────────────
# Import and use this everywhere rather than creating new instances.

default_log = HistoryLog()
