"""
Weight configuration for the cost scalar.

    Cost(P) = α·token_cost + β·call_cost + γ·latency_cost + δ·quality_risk

Four named presets cover the most common trade-off scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WeightConfig:
    alpha: float = 1.0  # token cost weight
    beta: float = 1.0  # call cost weight
    gamma: float = 1.0  # latency cost weight
    delta: float = 1.0  # quality risk weight

    def scalar(
        self,
        token_cost: float,
        call_cost: float,
        latency_cost: float,
        quality_risk: float,
    ) -> float:
        return (
            self.alpha * token_cost
            + self.beta * call_cost
            + self.gamma * latency_cost
            + self.delta * quality_risk
        )


# ── Named presets ──────────────────────────────────────────────────

PRESETS: dict[str, WeightConfig] = {
    # Minimise wall-clock time — latency is the dominant concern
    "speed": WeightConfig(alpha=0.5, beta=0.5, gamma=2.0, delta=0.5),
    # Minimise hallucination / grounding risk — quality first
    "quality": WeightConfig(alpha=0.5, beta=0.5, gamma=0.5, delta=2.0),
    # Minimise API spend — fewest tokens and calls
    "economy": WeightConfig(alpha=2.0, beta=2.0, gamma=0.5, delta=0.5),
    # Equal weight on all four dimensions (default)
    "balanced": WeightConfig(alpha=1.0, beta=1.0, gamma=1.0, delta=1.0),
}

DEFAULT_PRESET = "balanced"


def get_weights(preset: str = DEFAULT_PRESET) -> WeightConfig:
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}. Choose from: {sorted(PRESETS)}")
    return PRESETS[preset]
