"""Physical variant parameter schemas and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ir.nodes import PhysicalNode


RUNTIME_PARAM_NAMES = {
    "catalog",
    "client",
    "corpus",
    "embedder",
    "index",
    "llm",
    "model_client",
}


@dataclass(frozen=True)
class VariantParamSchema:
    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    descriptions: dict[str, str] | None = None

    @property
    def allowed(self) -> set[str]:
        return set(self.required) | set(self.optional)


VARIANT_PARAM_SCHEMAS: dict[str, VariantParamSchema] = {
    "BM25Retrieve": VariantParamSchema(
        required=("query",),
        optional=("top_k",),
        descriptions={
            "query": "Precise evidence retrieval query.",
            "top_k": "Number of chunks to retrieve.",
        },
    ),
    "DenseRetrieve": VariantParamSchema(
        required=("query",),
        optional=("top_k",),
        descriptions={
            "query": "Semantic evidence retrieval query.",
            "top_k": "Number of chunks to retrieve.",
        },
    ),
    "HybridRetrieve": VariantParamSchema(
        required=("query",),
        optional=("top_k", "alpha"),
        descriptions={
            "query": "Evidence retrieval query.",
            "top_k": "Number of chunks to retrieve.",
            "alpha": "Dense retrieval weight from 0 to 1.",
        },
    ),
    "SimilarityRank": VariantParamSchema(
        required=("query", "criterion"),
        optional=("top_k",),
        descriptions={
            "query": "Ranking query.",
            "criterion": "Ranking criterion for traceability.",
            "top_k": "Number of chunks to keep.",
        },
    ),
    "CrossEncoderRank": VariantParamSchema(
        required=("query",),
        optional=("top_k", "model"),
        descriptions={
            "query": "Ranking query.",
            "top_k": "Number of chunks to keep.",
            "model": "Cross-encoder model name.",
        },
    ),
    "LLMSummarize": VariantParamSchema(
        required=("schema",),
        optional=("query", "summary_key"),
        descriptions={
            "schema": "Extraction or summarization target.",
            "query": "Optional context query.",
            "summary_key": "Optional cache key; runtime catalog is not a param.",
        },
    ),
    "DirectGenerate": VariantParamSchema(
        required=("goal",),
        optional=("max_tokens", "context_window"),
        descriptions={
            "goal": "Generation goal.",
            "max_tokens": "Maximum output tokens.",
            "context_window": "Model context window size.",
        },
    ),
    "HierarchicalGenerate": VariantParamSchema(
        required=("goal",),
        optional=("batch_tokens", "max_tokens"),
        descriptions={
            "goal": "Generation goal.",
            "batch_tokens": "Tokens per evidence batch.",
            "max_tokens": "Maximum output tokens per LLM call.",
        },
    ),
}


def render_variant_param_schemas(active_variants: dict[Any, list[str]]) -> str:
    lines: list[str] = []
    for op, variants in active_variants.items():
        op_name = getattr(op, "value", str(op))
        lines.append(f"{op_name}:")
        for variant in variants:
            schema = VARIANT_PARAM_SCHEMAS.get(variant, VariantParamSchema())
            required = ", ".join(schema.required) or "none"
            optional = ", ".join(schema.optional) or "none"
            lines.append(f"  - {variant}")
            lines.append(f"    required params: {required}")
            lines.append(f"    optional params: {optional}")
            if schema.descriptions:
                for name in (*schema.required, *schema.optional):
                    description = schema.descriptions.get(name)
                    if description:
                        lines.append(f"    {name}: {description}")
    return "\n".join(lines)


def validate_physical_plan_params(root: PhysicalNode) -> None:
    errors: list[str] = []
    _validate_node(root, errors)
    if errors:
        raise ValueError("; ".join(errors))


def _validate_node(node: PhysicalNode, errors: list[str]) -> None:
    schema = VARIANT_PARAM_SCHEMAS.get(node.variant)
    params = dict(node.params)

    runtime_keys = sorted(set(params) & RUNTIME_PARAM_NAMES)
    if runtime_keys:
        errors.append(
            f"{node.variant} params include runtime-only keys: {', '.join(runtime_keys)}"
        )

    if schema is not None:
        missing = [name for name in schema.required if name not in params]
        if missing:
            errors.append(
                f"{node.variant} params missing required keys: {', '.join(missing)}"
            )
        unsupported = sorted(set(params) - schema.allowed)
        if unsupported:
            errors.append(
                f"{node.variant} params include unsupported keys: {', '.join(unsupported)}"
            )

    for child in node.inputs:
        _validate_node(child, errors)
