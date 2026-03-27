"""
Async DAG executor.

Walks a PhysicalNode tree bottom-up. Independent branches
(e.g. children of UNION) run in parallel via asyncio.gather.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from catalog.catalog import SystemCatalog
from executor.registry import REGISTRY
from ir.evidence import EvidenceSet
from ir.nodes import PhysicalNode


@dataclass
class ExecutionResult:
    output: EvidenceSet
    trace: list[str]  # op_trace from the final EvidenceSet
    token_counts: dict[str, int]  # variant → total tokens produced
    errors: list[str] = field(default_factory=list)

    @property
    def answer(self) -> str:
        """Text of the first output chunk, or empty string."""
        return self.output.chunks[0].text if self.output.chunks else ""


class PlanRunner:
    """
    Execute a PhysicalNode DAG asynchronously.

    Usage:
        runner = PlanRunner(corpus=corpus, llm=llm)
        result = await runner.run(physical_root)
        print(result.answer)
    """

    def __init__(self, corpus, llm, catalog: SystemCatalog | None = None):
        self.corpus = corpus
        self.llm = llm
        self.catalog = catalog

    async def run(self, root: PhysicalNode) -> ExecutionResult:
        errors: list[str] = []
        token_counts: dict[str, int] = {}
        output = await self._execute(root, errors, token_counts)
        return ExecutionResult(
            output=output,
            trace=output.op_trace,
            token_counts=token_counts,
            errors=errors,
        )

    async def _execute(
        self,
        node: PhysicalNode,
        errors: list[str],
        token_counts: dict[str, int],
    ) -> EvidenceSet:
        # ── 1. Resolve children in parallel ───────────────────────
        input_results: list[EvidenceSet] = (
            list(
                await asyncio.gather(
                    *[self._execute(c, errors, token_counts) for c in node.inputs]
                )
            )
            if node.inputs
            else []
        )

        # ── 2. Look up variant and execute ─────────────────────────
        fn = REGISTRY.get(node.variant)
        if fn is None:
            errors.append(f"Unknown variant: {node.variant!r}")
            return EvidenceSet(chunks=[]).append_trace(f"MISSING:{node.variant}")

        try:
            params = dict(node.params)
            if self.catalog is not None:
                params.setdefault("catalog", self.catalog)
            result = await fn(input_results, params, self.corpus, self.llm)
        except Exception as exc:
            errors.append(f"{node.variant} failed: {exc}")
            result = EvidenceSet(chunks=[]).append_trace(f"ERROR:{node.variant}")

        # ── 3. Record token usage ──────────────────────────────────
        tokens = result.token_estimate()
        token_counts[node.variant] = token_counts.get(node.variant, 0) + tokens

        return result
