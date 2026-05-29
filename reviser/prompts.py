"""LLM prompts for the two reviser calls.

Step 1 — Sample analysis
    Input : execution record for one sample (query, logical/physical plan,
            node-level trace, metrics)
    Output: structured SampleFeedback JSON (parsed by sample_analyzer.py)

Steps 4-6 — TST revision
    Input : previous TST text + PatternSummary (aggregated across samples)
    Output: revised TST in the same two-section format used by task_prompts.py
            (parsed by expr_parser.parse_task_strategy)
"""

from __future__ import annotations

from parser.operator_candidates import LOGICAL_OPERATOR_DESCRIPTIONS

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
source evidence and whether the logical strategy supports the task.

Focus on:
- evidence match against gold evidence
- whether the logical structure can verify exact evidence support
- whether the plan is missing extraction, verification, composition, or ranking
- whether later operators add noise or paraphrase evidence
- physical operator issues only as debugging details

## Analysis order
1. Decide whether the logical plan supports the task.
2. Identify the main structural gap, if any.
3. Explain the plan-level reason in one concise sentence.
4. Optionally list physical operator problems for debugging only.

## Output format
Output ONLY valid JSON. No markdown. No explanation.

{
  "sample_analysis": {
    "query": "<query>",
    "accuracy": <number>,
    "total_tokens": <integer>,
    "total_latency_ms": <number>,
    "plan_feedback": {
      "supports_task": <true|false>,
      "main_structural_gap": "<short phrase, or 'none'>",
      "reason": "<one concise sentence>"
    },
    "physical_feedback": [
      {
        "op_id": "<op id>",
        "variant": "<physical variant>",
        "issue_type": "<short snake_case issue>",
        "description": "<what went wrong>",
        "suggested_change": "<debugging suggestion>"
      }
    ],
    "successful_adaptations": []
  }
}

Use an empty list for physical_feedback when there are no operator-level issues.
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

Analyse this execution focusing on plan-level task support. Include physical
feedback only for debugging; TST revision will ignore it.
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
task context, and aggregated plan-level feedback from execution samples,
revise the TST based on repeated structural evidence from feedback.

Treat revision as knowledge after experience: look back at what happened during
execution, then encode the lesson into the adaptation policy so future
query-level planning knows when to expand, specialize, simplify, or preserve
the skeleton.

## Logical operators available
{_LOGICAL_CATALOG}

## Revision priorities
1. Update adaptation rules from execution experience when the skeleton is broadly right.
2. Fix repeated structural failures in the logical skeleton only when the reusable structure itself is wrong.
3. Preserve the smallest structure that directly supports the task.
4. Preserve useful successful adaptations as explicit allowed_rewrites or slot guidance.

## Learned adaptation policy rules
- If results show a query needed an extra operator, add an allowed_rewrites rule that names when to add it.
- If results show a query should avoid an operator, add a forbidden_rewrites or allowed simplification rule.
- If repeated failures come from vague query/ranking/extraction goals, update mutable_slots and allowed_rewrites so future queries specialize those goals.
- If successful adaptations recur, preserve them as learned adaptation knowledge.
- If the skeleton is broadly correct but too rigid, revise ADAPTATION POLICY before changing LOGICAL SKELETON.

## Structural revision rules
- If the same operator repeatedly causes failure, change the skeleton instead of treating it as an operator-level symptom.
- If AGGREGATE repeatedly causes over_generation, replace it with terminal TRANSFORM when the task requires exact extraction.
- If RANK repeatedly keeps noisy evidence, change the ranking criterion or make RANK optional/removable.
- If I repeatedly misses evidence, improve query slots/adaptation rules before changing downstream operators.
- If feedback recommends the same structural fix across samples, apply it even if it changes the original skeleton.
- Prefer the smallest structural change that directly addresses repeated failure.
- Prefer adaptation-policy changes when they can teach the query-level planner how to handle the next similar query.

## Conservatism rules
- Do NOT change parts of the TST that have no corresponding feedback pattern.
- Do NOT invent operators outside the list above.
- Do NOT add VERIFY unless feedback shows grounding, schema, or consistency failures.
- Preserve working adaptations listed in the summary.
- If a pattern appears in fewer than half the samples, treat it as weak evidence and prefer adaptation-policy changes over skeleton changes.
- Ignore physical feedback and physical plan variants when revising the TST.

## Output format
Output exactly two sections — the same format as the original TST.
No markdown. No explanation. Nothing else.

LOGICAL SKELETON
<algebraic expression with {{SLOT}} placeholders>

ADAPTATION POLICY
mutable_slots: <comma-separated>
immutable_slots: <comma-separated>
mutable_ops: <comma-separated>
immutable_ops: <comma-separated>
allowed_rewrites: <one learned adaptation or expansion rule per line, repeat key>
forbidden_rewrites: <one learned safety boundary per line, repeat key>
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

### Cost / latency summary
{cost_patterns}

### Recommendation
{recommendation}

Revise the TST following the system prompt priorities, conservatism rules, and
knowledge-after-experience guidance. Update adaptation rules when experience
shows how future query-level plans should expand, specialize, simplify, or
preserve the skeleton.
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
