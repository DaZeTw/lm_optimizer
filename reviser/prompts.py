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

# ── Step 1: sample analysis ───────────────────────────────────────

SAMPLE_ANALYSIS_SYSTEM_PROMPT = """\
You are a query-plan auditor for a long-context reasoning system.

Given one execution record — the query, the logical and physical plans used,
node-level traces, and quality/cost metrics — identify exactly what went wrong
or right and suggest targeted local fixes.

## Output format
Output exactly four sections with the headers shown. No markdown. No explanation.

QUERY FEATURES
<one label per line: key: value>
<labels to include: query_type, complexity, evidence_scope, comparison_required>

FAILURE POINTS
<one line per failure: op_id | issue_type | description>
<issue_type: wrong_variant | bad_params | missing_evidence | over_retrieval | hallucination | slow>
<use "(none)" if no failures>

SUCCESSFUL ADAPTATIONS
<one line per success: op_id | what_worked>
<use "(none)" if none>

SUGGESTED FIXES
<one line per fix: op_id | fix_type | detail>
<fix_type: change_variant | tune_param | add_operator | remove_operator | change_slot>
<only suggest fixes grounded in the failure points above>
"""

_SAMPLE_ANALYSIS_USER_TEMPLATE = """\
## Query
{query}

## Logical plan (algebraic)
{logical_plan_text}

## Physical plan (node summary)
{physical_plan_text}

## Node-level execution trace
{node_trace}

## Metrics
- Accuracy : {accuracy:.2f}
- Total tokens : {total_tokens}
- Total latency (ms) : {total_latency_ms:.1f}
- Errors : {errors}

## Task Strategy Template in use
{tst_text}

Analyse this execution record and output the four-section report.
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

TST_REVISION_SYSTEM_PROMPT = """\
You are a task-strategy optimizer for a long-context reasoning system.

Given the current Task Strategy Template (TST) and a pattern summary aggregated
from multiple execution samples, revise the TST conservatively.

## Revision priorities (apply in order)
1. Adaptation policy  — update mutable/immutable ops and rewrite rules first.
2. Physical policy    — change variants or param ranges only for nodes with
                        consistent failures across samples.
3. Logical skeleton   — change the operator DAG shape only if a structural
                        failure repeats across the majority of samples.

## Conservatism rules
- Only change what the pattern summary explicitly flags as a repeated issue.
- Do NOT change parts of the TST that have no corresponding pattern.
- If a pattern appears in fewer than half the samples, ignore it.
- Preserve all working adaptations listed in the summary.

## Output format
Output exactly three sections — the same format as the original TST.
No markdown. No explanation. Nothing else.

LOGICAL SKELETON
<algebraic expression with {SLOT} placeholders — unchanged if no structural fix needed>

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
## Current Task Strategy Template
{tst_text}

## Aggregated pattern summary across {num_samples} sample(s)

### Repeated failure patterns
{failure_patterns}

### Repeated successful adaptations
{success_patterns}

### Cost / latency patterns
{cost_patterns}

### Revision recommendation
{recommendation}

Revise the TST following the system prompt priorities and conservatism rules.
Output only the revised TST.
"""


def build_tst_revision_user_message(
    tst_text: str,
    num_samples: int,
    failure_patterns: str,
    success_patterns: str,
    cost_patterns: str,
    recommendation: str,
) -> str:
    return _TST_REVISION_USER_TEMPLATE.format(
        tst_text=tst_text,
        num_samples=num_samples,
        failure_patterns=failure_patterns or "(none)",
        success_patterns=success_patterns or "(none)",
        cost_patterns=cost_patterns or "(none)",
        recommendation=recommendation or "(none)",
    )
