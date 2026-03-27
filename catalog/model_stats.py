"""Model capability and pricing metadata used by planner and cost model."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelStats:
    model_id: str
    context_window: int
    input_cost_per_1k: float
    output_cost_per_1k: float
    avg_latency_ms: int
    supports_tools: bool

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelStats":
        return cls(
            model_id=str(data.get("model_id", "unknown")),
            context_window=int(data.get("context_window", 128_000)),
            input_cost_per_1k=float(data.get("input_cost_per_1k", 0.0)),
            output_cost_per_1k=float(data.get("output_cost_per_1k", 0.0)),
            avg_latency_ms=int(data.get("avg_latency_ms", 0)),
            supports_tools=bool(data.get("supports_tools", False)),
        )


DEFAULT_MODEL_ID = "gpt-4o"

# Lightweight built-ins. Keep this hand-maintained and conservative.
BUILTIN_MODEL_REGISTRY: dict[str, ModelStats] = {
    "gpt-4o": ModelStats(
        model_id="gpt-4o",
        context_window=128_000,
        input_cost_per_1k=0.005,
        output_cost_per_1k=0.015,
        avg_latency_ms=1200,
        supports_tools=True,
    ),
    "gpt-4o-mini": ModelStats(
        model_id="gpt-4o-mini",
        context_window=128_000,
        input_cost_per_1k=0.00015,
        output_cost_per_1k=0.0006,
        avg_latency_ms=700,
        supports_tools=True,
    ),
    "claude-sonnet-4-6": ModelStats(
        model_id="claude-sonnet-4-6",
        context_window=200_000,
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
        avg_latency_ms=1300,
        supports_tools=True,
    ),
}


def get_model_stats(
    model_id: str,
    registry: dict[str, ModelStats] | None = None,
    default_model_id: str = DEFAULT_MODEL_ID,
) -> ModelStats:
    """Fetch model stats; fall back to default model entry when unknown."""
    reg = registry or BUILTIN_MODEL_REGISTRY
    if model_id in reg:
        return reg[model_id]
    if default_model_id in reg:
        return reg[default_model_id]

    # Last-resort fallback for robustness in custom registries.
    return ModelStats(
        model_id=model_id,
        context_window=128_000,
        input_cost_per_1k=0.0,
        output_cost_per_1k=0.0,
        avg_latency_ms=0,
        supports_tools=False,
    )
