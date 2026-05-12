"""LLM prompts for the two reviser calls.

Step 1 — Sample analysis
    Input : execution record for one sample (query, logical/physical plan,
            node-level trace, metrics)
    Output: structured SampleFeedback text (parsed by sample_analyzer.py)

Steps 4-6 — TST revision
    Input : previous TST text + PatternSummary (aggregated across samples)
    Output: revised TST in the same three-section format used by task_prompts.py
            (parsed by expr_parser.parse_task_strategy)
"""

from __future__ import annotations

from planner.variant_candidates import CANDIDATE_VARIANTS, VARIANT_DESCRIPTIONS
from parser.operator_candidates import LOGICAL_OPERATOR_DESCRIPTIONS

def _build_variant_catalog() -> str:
    """Build variant catalog text for LLM prompts.

    Includes:
    - valid variants for each logical operator
    - default variant
    - description
    - use case
    - cost level
    """

    lines: list[str] = []

    for op, variants in CANDIDATE_VARIANTS.items():
        lines.append(f"{op.value}:")

        for idx, variant in enumerate(variants):
            meta = VARIANT_DESCRIPTIONS.get(variant, {})

            default_tag = " [default]" if idx == 0 else ""
            description = meta.get("description", "No description provided.")
            use_when = meta.get("use_when", "No usage guidance provided.")
            cost = meta.get("cost", "unknown")

            lines.append(
                f"  - {variant}{default_tag}\n"
                f"    description: {description}\n"
                f"    use_when: {use_when}\n"
                f"    cost: {cost}"
            )

        lines.append("")

    return "\n".join(lines).strip()


_VARIANT_CATALOG = _build_variant_catalog()

def _build_logical_catalog() -> str:
    lines: list[str] = []

    for op, meta in LOGICAL_OPERATOR_DESCRIPTIONS.items():
        lines.append(f"\n{op.value}")
        lines.append(f"  description: {meta['description']}")
        lines.append(f"  inputs: {meta['inputs']}")
        lines.append(f"  outputs: {meta['outputs']}")
        lines.append(f"  use_when: {meta['use_when']}")

    return "\n".join(lines)


_LOGICAL_CATALOG = _build_logical_catalog()

# ── Step 1: sample analysis ───────────────────────────────────────

SAMPLE_ANALYSIS_SYSTEM_PROMPT = """\
You are a query-plan auditor for an evidence retrieval system.

Given one execution record, analyze whether the plan retrieved the correct
source evidence and identify the exact operator responsible for each issue.

Focus on:
- evidence match against gold evidence
- retrieval query quality
- physical variant / parameter choices
- whether later operators add noise or paraphrase evidence

## Analysis order
1. Check I nodes first: did retrieval find the right evidence?
2. Check RANK nodes: did ranking keep the most relevant evidence?
3. Check TRANSFORM nodes: did extraction preserve exact evidence?
4. Check AGGREGATE nodes: did it return exact evidence or paraphrase/explain?
5. Check VERIFY nodes: did it correctly filter unsupported evidence?

## Failure attribution rule
- bad_retrieval_query belongs to I nodes
- missing_evidence usually belongs to I nodes
- noisy_evidence belongs to I nodes if retrieval is noisy, or RANK nodes if ranking kept bad evidence
- wrong_variant / bad_params belong to the node using the wrong setting
- missing_transform belongs to the parent node that needed evidence extraction before generation
- over_generation belongs to AGGREGATE nodes
- hallucination belongs to AGGREGATE or VERIFY nodes
- slow belongs to the node with high latency in the trace

## Output format
Output exactly four sections with the headers shown. No markdown. No explanation.

QUERY FEATURES
<one label per line: key: value>
<labels: query_type, complexity, evidence_scope, requires_exact_match>

FAILURE POINTS
<one line per failure: op_id | issue_type | description>
<issue_type:
  bad_retrieval_query | missing_evidence | noisy_evidence |
  wrong_variant | bad_params | missing_transform |
  over_generation | hallucination | slow>
<focus on root cause, not symptoms>
<use "(none)" if no failures>

SUCCESSFUL ADAPTATIONS
<one line per success: op_id | what_worked>
<focus on decisions that improved evidence quality>
<use "(none)" if none>

SUGGESTED FIXES
<one line per fix: op_id | fix_type | detail>
<fix_type:
  rewrite_query | tune_param | change_variant |
  add_operator | remove_operator | change_schema>
<fixes must target the same operator responsible for the failure>
"""

_SAMPLE_ANALYSIS_USER_TEMPLATE = """\
## Query
{query}

## Logical plan
{logical_plan_text}

## Physical plan
{physical_plan_text}

## Execution trace
{node_trace}

## Metrics
- Accuracy : {accuracy:.2f}
- Total tokens : {total_tokens}
- Total latency (ms) : {total_latency_ms:.1f}
- Errors : {errors}

Analyse this execution focusing on evidence quality and plan effectiveness.
"""


def build_sample_analysis_user_message(
    query: str,
    logical_plan_text: str,
    physical_plan_text: str,
    node_trace: str,
    accuracy: float,
    total_tokens: int,
    total_latency_ms: float,
    errors: list[str],
    tst_text: str,
) -> str:
    return _SAMPLE_ANALYSIS_USER_TEMPLATE.format(
        query=query,
        logical_plan_text=logical_plan_text,
        physical_plan_text=physical_plan_text,
        node_trace=node_trace,
        accuracy=accuracy,
        total_tokens=total_tokens,
        total_latency_ms=total_latency_ms,
        errors="; ".join(errors) if errors else "(none)",
        tst_text=tst_text,
    )


# ── Steps 4-6: TST revision ───────────────────────────────────────

TST_REVISION_SYSTEM_PROMPT = f"""\
You are a task-strategy optimizer for a long-context reasoning system.

Given the current Task Strategy Template (TST), available logical operators,
available physical variants, task context, and aggregated feedback from execution samples,
revise the TST based on repeated evidence from feedback.

## Logical operators available
{_LOGICAL_CATALOG}

## Physical variants available
Use ONLY these variants when revising the physical policy.

{_VARIANT_CATALOG}

## Revision priorities
1. Fix repeated structural failures in the logical skeleton.
2. Fix repeated physical failures by changing variants or parameters.
3. Fix adaptation failures by making the policy more flexible or more restrictive.

## Structural revision rules
- If the same operator repeatedly causes failure, change the skeleton instead of only changing variants.
- If AGGREGATE repeatedly causes over_generation, replace it with terminal TRANSFORM when the task requires exact extraction.
- If RANK repeatedly keeps noisy evidence, reduce top_k, change ranking criterion, or make RANK optional/removable.
- If I repeatedly misses evidence, improve query slots/adaptation rules before changing downstream operators.
- If feedback recommends the same structural fix across samples, apply it even if it changes the original skeleton.
- Prefer the smallest structural change that directly addresses repeated failure.

## Conservatism rules
- Do NOT change parts of the TST that have no corresponding feedback pattern.
- Do NOT invent operators or variants outside the lists above.
- Do NOT add VERIFY unless feedback shows grounding, schema, or consistency failures.
- Preserve working adaptations listed in the summary.
- If a pattern appears in fewer than half the samples, treat it as weak evidence and prefer adaptation-policy changes over skeleton changes.

## Output format
Output exactly three sections — the same format as the original TST.
No markdown. No explanation. Nothing else.

LOGICAL SKELETON
<algebraic expression with {{SLOT}} placeholders>

PHYSICAL POLICY
<one line per node: op_id | op_name | variant | params>

ADAPTATION POLICY
mutable_slots: <comma-separated>
immutable_slots: <comma-separated>
mutable_ops: <comma-separated>
immutable_ops: <comma-separated>
allowed_rewrites: <one per line, repeat key>
forbidden_rewrites: <one per line, repeat key>
"""

_TST_REVISION_USER_TEMPLATE = """\
## Task Description
{task_description}

## Evaluation Metrics
{evaluation_metrics}

## Current Task Strategy Template
{tst_text}

## Aggregated pattern summary across {num_samples} sample(s)

### Repeated failure patterns
{failure_patterns}

### Repeated successful adaptations
{success_patterns}

### Cost / latency patterns
{cost_patterns}

Revise the TST following the system prompt priorities and conservatism rules.
Output only the revised TST.
"""


def build_tst_revision_user_message(
    task_description: str,
    evaluation_metrics: str,
    tst_text: str,
    num_samples: int,
    failure_patterns: str,
    success_patterns: str,
    cost_patterns: str,
    recommendation: str,
) -> str:
    return _TST_REVISION_USER_TEMPLATE.format(
        task_description=task_description,
        evaluation_metrics=evaluation_metrics,
        tst_text=tst_text,
        num_samples=num_samples,
        failure_patterns=failure_patterns or "(none)",
        success_patterns=success_patterns or "(none)",
        cost_patterns=cost_patterns or "(none)",
        recommendation=recommendation or "(none)",
    )