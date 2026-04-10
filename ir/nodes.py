"""IR node dataclasses: LogicalNode and PhysicalNode."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .ops import Op


@dataclass(frozen=True)
class LogicalNode:
    op: Op
    inputs: tuple[LogicalNode, ...]  # children in the DAG
    params: dict[str, Any]  # op-specific config

    # ── Convenience constructors ───────────────────────────────────

    @classmethod
    def isolate(cls, query: str) -> LogicalNode:
        return cls(op=Op.I, inputs=(), params={"query": query})

    @classmethod
    def transform(cls, child: LogicalNode, schema: str = "") -> LogicalNode:
        return cls(op=Op.TRANSFORM, inputs=(child,), params={"schema": schema})

    @classmethod
    def compose(
        cls, left: LogicalNode, right: LogicalNode, condition: str = ""
    ) -> LogicalNode:
        return cls(op=Op.COMPOSE, inputs=(left, right), params={"condition": condition})

    @classmethod
    def union(cls, *children: LogicalNode) -> LogicalNode:
        return cls(op=Op.UNION, inputs=tuple(children), params={})

    @classmethod
    def diff(cls, base: LogicalNode, subtract: LogicalNode) -> LogicalNode:
        return cls(op=Op.DIFF, inputs=(base, subtract), params={})

    @classmethod
    def rank(cls, child: LogicalNode, criterion: str = "") -> LogicalNode:
        return cls(op=Op.RANK, inputs=(child,), params={"criterion": criterion})

    @classmethod
    def aggregate(cls, child: LogicalNode, goal: str = "") -> LogicalNode:
        return cls(op=Op.AGGREGATE, inputs=(child,), params={"goal": goal})

    @classmethod
    def verify(cls, child: LogicalNode, constraints: str = "") -> LogicalNode:
        return cls(op=Op.VERIFY, inputs=(child,), params={"constraints": constraints})

    @classmethod
    def decompose(cls, query: str) -> LogicalNode:
        return cls(op=Op.DECOMPOSE, inputs=(), params={"query": query})

    # ── Serialization ──────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "op": self.op.value,
            "params": dict(self.params),
            "inputs": [i.to_dict() for i in self.inputs],
        }

    @classmethod
    def from_dict(cls, d: dict) -> LogicalNode:
        return cls(
            op=Op(d["op"]),
            params=d.get("params", {}),
            inputs=tuple(cls.from_dict(i) for i in d.get("inputs", [])),
        )

    # ── Pretty print ───────────────────────────────────────────────

    def pretty(self, indent: int = 0) -> str:
        pad = "  " * indent
        param_str = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        head = f"{pad}{self.op.value}({param_str})"
        if not self.inputs:
            return head
        children = "\n".join(i.pretty(indent + 1) for i in self.inputs)
        return f"{head}\n{children}"


# ── Physical layer ─────────────────────────────────────────────────


@dataclass(frozen=True)
class CostVector:
    token_cost: float = 0.0
    latency_cost: float = 0.0

    def scalar(self, alpha: float = 1.0, beta: float = 1.0) -> float:
        return alpha * self.token_cost + beta * self.latency_cost


@dataclass(frozen=True)
class PhysicalNode:
    variant: str  # e.g. "DenseRetrieve", "LLMSummarize"
    logical_ref: LogicalNode
    inputs: tuple[PhysicalNode, ...]
    params: dict[str, Any]
    cost: CostVector = field(default_factory=CostVector)
