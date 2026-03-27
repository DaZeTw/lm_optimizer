"""
Optimizer engine: applies rewrite rules to a LogicalNode DAG
until no rule fires (fixpoint).

Usage:
    from lm_optimizer.optimizer.engine import OptimizerEngine

    engine = OptimizerEngine()
    optimized, log = engine.run(plan)
    print(optimized.pretty())
    for entry in log:
        print(entry)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ir.nodes import LogicalNode
from optimizer.rules import RULES


# ── Log entry ─────────────────────────────────────────────────────


@dataclass
class RewriteEntry:
    rule: str  # e.g. "R1_transform_pushdown"
    before: str  # node.pretty() snapshot before rewrite
    after: str  # node.pretty() snapshot after rewrite
    pass_no: int  # which fixpoint pass this fired in

    def __str__(self) -> str:
        return (
            f"[pass {self.pass_no}] {self.rule}\n"
            f"  before: {self.before.splitlines()[0]}\n"
            f"  after:  {self.after.splitlines()[0]}"
        )


# ── Engine ─────────────────────────────────────────────────────────


class OptimizerEngine:
    """
    Applies the 8 rewrite rules to a LogicalNode DAG using a
    post-order tree walk, repeating until fixpoint.

    Args:
        rules:     List of (name, rule_fn) pairs. Defaults to the
                   standard RULES registry from rules.py.
        max_passes: Safety limit on fixpoint iterations.
    """

    def __init__(
        self,
        rules: list[tuple[str, Callable]] | None = None,
        max_passes: int = 20,
    ):
        self.rules = rules if rules is not None else RULES
        self.max_passes = max_passes

    # ── Public API ─────────────────────────────────────────────────

    def run(self, root: LogicalNode) -> tuple[LogicalNode, list[RewriteEntry]]:
        """
        Optimize a logical plan to fixpoint.

        Returns:
            (optimized_root, rewrite_log)
        """
        log: list[RewriteEntry] = []
        current = root

        for pass_no in range(1, self.max_passes + 1):
            rewritten, pass_log = self._single_pass(current, pass_no)
            log.extend(pass_log)

            if not pass_log:  # nothing fired → fixpoint reached
                break
            current = rewritten
        else:
            # Exceeded max_passes — return best effort
            pass

        return current, log

    # ── Internal ───────────────────────────────────────────────────

    def _single_pass(
        self, node: LogicalNode, pass_no: int
    ) -> tuple[LogicalNode, list[RewriteEntry]]:
        """
        Post-order walk: rewrite children first, then the current node.
        Returns the (possibly rewritten) node and all log entries from
        this pass.
        """
        log: list[RewriteEntry] = []

        # ── 1. Recurse into children first (post-order) ────────────
        new_inputs = []
        for child in node.inputs:
            rewritten_child, child_log = self._single_pass(child, pass_no)
            new_inputs.append(rewritten_child)
            log.extend(child_log)

        # Rebuild node with (potentially rewritten) children
        if new_inputs != list(node.inputs):
            node = LogicalNode(
                op=node.op,
                inputs=tuple(new_inputs),
                params=node.params,
            )

        # ── 2. Try each rule on the current node ───────────────────
        # Apply at most one rule per node per pass.
        # Multiple passes handle cascading rewrites.
        for rule_name, rule_fn in self.rules:
            result, fired = rule_fn(node)
            if fired:
                log.append(
                    RewriteEntry(
                        rule=rule_name,
                        before=node.pretty(),
                        after=result.pretty(),
                        pass_no=pass_no,
                    )
                )
                node = result
                break  # one rule per node per pass — prevents infinite loops

        return node, log
