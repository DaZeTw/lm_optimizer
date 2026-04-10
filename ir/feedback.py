"""Feedback dataclasses produced after each execution iteration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NodeFeedback:
    """Execution record for one physical node in the DAG."""

    op_id: str          # e.g. "AGGREGATE_0"
    variant: str        # chosen variant name, e.g. "DirectGenerate"
    token_cost: int     # tokens consumed by this node
    latency_ms: float   # wall-clock execution time in milliseconds
    output_summary: str # first ~200 chars of evidence output (planner context)


@dataclass
class Feedback:
    """Full feedback bundle produced after one execution + judge pass."""

    items: list[NodeFeedback] = field(default_factory=list)
    accuracy: float = 0.0   # 0.0–1.0 from LLM-as-judge
    result: str = ""        # predicted answer (execution.answer)
    gold_ans: str = ""      # expected answer
