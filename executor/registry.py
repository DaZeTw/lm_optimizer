"""
Variant registry — maps physical variant names to async callables.

Every variant function registers itself with @register("VariantName").
The runner looks up variants by name at execution time.

Usage:
    from executor.registry import register, REGISTRY

    @register("MyVariant")
    async def my_variant(inputs, params, corpus, llm):
        ...
"""

from __future__ import annotations

from typing import Awaitable, Callable

from ir.evidence import EvidenceSet

# inputs: resolved child EvidenceSets
# params: node.params dict from the PhysicalNode
# corpus: Corpus instance (injected by runner)
# llm:    LLM instance (injected by runner)
VariantFn = Callable[
    [list[EvidenceSet], dict, object, object],
    Awaitable[EvidenceSet],
]

REGISTRY: dict[str, VariantFn] = {}


def register(variant_name: str) -> Callable[[VariantFn], VariantFn]:
    """Decorator: register an async variant function by name."""

    def decorator(fn: VariantFn) -> VariantFn:
        REGISTRY[variant_name] = fn
        return fn

    return decorator
