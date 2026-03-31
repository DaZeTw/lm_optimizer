"""
Cost vector types produced by the scorer.

OperatorCostVector  — cost estimate for one node in the plan
PlanCostReport      — aggregated report for the full plan
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OperatorCostVector:
    """Cost estimate for a single physical operator node."""

    op_id: str  # e.g. "AGGREGATE_0", "TRANSFORM_1"
    variant: str  # chosen physical variant
    token_cost: float = 0.0
    call_cost: float = 0.0
    latency_cost: float = 0.0
    quality_risk: float = 0.0
    key_metric: str = ""  # human-readable explanation of dominant cost
    saturation: float = 0.0  # AGGREGATE only: input_tokens / context_window
    sample_count: int = 0  # telemetry samples backing this estimate
    accuracy_score: float | None = None  # optional observed accuracy proxy
    variant_costs: dict[str, dict[str, float]] = field(default_factory=dict)
    # variant_costs maps variant -> {token_cost, call_cost, latency_cost, quality_risk}

    def scalar(self, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0) -> float:
        return (
            alpha * self.token_cost
            + beta * self.call_cost
            + gamma * self.latency_cost
            + delta * self.quality_risk
        )


@dataclass
class PlanCostReport:
    """Full cost report for an annotated physical plan."""

    total_token_cost: float
    total_call_cost: float
    total_latency_cost: float
    total_quality_risk: float
    scalar: float  # weighted total
    per_node: dict[str, OperatorCostVector]  # op_id → estimate
    bottleneck: str  # op_id with highest scalar
    saturation: float  # max saturation across AGGREGATEs
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Plan cost scalar : {self.scalar:.1f}",
            f"Token cost       : {self.total_token_cost:.0f} tokens",
            f"Call cost        : {self.total_call_cost:.1f} calls",
            f"Latency steps    : {self.total_latency_cost:.1f} sequential steps",
            f"Quality risk     : {self.total_quality_risk:.3f}",
            f"Bottleneck       : {self.bottleneck}",
            f"Max saturation   : {self.saturation:.2%}",
        ]
        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  ⚠  {w}")
        return "\n".join(lines)
