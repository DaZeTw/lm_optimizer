"""Batch evaluation harness for iterative physical-plan feedback.

This script runs a subset of dataset samples through `LmOptimizerPipeline` in
`llm_feedback` mode and exports per-sample and aggregate metrics that can be
used to revise physical planning behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from parser.semantic_parser import LLMClient, SemanticParser
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from executor.corpus import InMemoryCorpus, OpenAILLM
from ir.evidence import Chunk
from ir.nodes import PhysicalNode
from pipeline import LmOptimizerPipeline
from planner.variant_candidates import CANDIDATE_VARIANTS


class CountingPlanningClient:
    """Thin wrapper to track planning LLM call counts."""

    def __init__(self, base: LLMClient):
        self._base = base
        self.calls = 0

    def complete(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
    ) -> str:
        self.calls += 1
        return self._base.complete(
            messages=messages, model=model, temperature=temperature
        )


class FixedPlanningClient:
    """Planning client that always returns a fixed override map."""

    def __init__(self, overrides: dict[str, str]):
        self.overrides = dict(overrides)
        self.calls = 0

    def complete(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
    ) -> str:
        del messages
        del model
        del temperature
        self.calls += 1
        return json.dumps({"variant_overrides": self.overrides})


def _strip_code_fence(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_variant_overrides(response: Any) -> dict[str, str]:
    if isinstance(response, dict):
        body = response
    else:
        text = str(response or "").strip()
        if text.startswith("```"):
            text = _strip_code_fence(text)
        try:
            body = json.loads(text)
        except json.JSONDecodeError:
            return {}

    overrides = body.get("variant_overrides") if isinstance(body, dict) else None
    if not isinstance(overrides, dict):
        return {}

    out: dict[str, str] = {}
    for k, v in overrides.items():
        if isinstance(k, str) and isinstance(v, str):
            out[k.strip()] = v.strip()
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run iterative physical-feedback evaluation on JSON dataset samples.",
    )
    parser.add_argument("--input", required=True, help="Path to dataset JSON file.")
    parser.add_argument(
        "--output",
        default="outputs/batch_feedback_eval.json",
        help="Path to output JSON report.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Maximum number of samples to run.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split filter value. Use 'all' to disable split filtering.",
    )
    parser.add_argument(
        "--planning-rounds",
        type=int,
        default=1,
        help="Internal LLM planner rounds per build_physical call.",
    )
    parser.add_argument(
        "--feedback-iterations",
        type=int,
        default=2,
        help="Execution-feedback iterations via run_sync_with_physical_feedback.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model used for parser, planning, and generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for parser/planning calls.",
    )
    parser.add_argument(
        "--protocol",
        choices=["single_pass", "test_aggregate_refine"],
        default="single_pass",
        help="Execution protocol mode.",
    )
    parser.add_argument(
        "--token-increase-cap",
        type=float,
        default=0.15,
        help="Maximum allowed increase ratio for avg token cost in revised plan.",
    )
    return parser.parse_args()


def load_samples(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        return [payload]
    raise ValueError("Input JSON must be an object or list of objects")


def select_samples(
    samples: list[dict[str, Any]],
    *,
    split: str,
    max_samples: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sample in samples:
        if bool(sample.get("is_unanswerable", False)):
            continue
        if split != "all" and str(sample.get("split", "")) != split:
            continue
        out.append(sample)
        if len(out) >= max_samples:
            break
    return out


def build_corpus(sample: dict[str, Any]) -> InMemoryCorpus:
    chunks: list[Chunk] = []
    paper_id = str(sample.get("paper_id", "unknown_paper"))

    title = str(sample.get("title", "")).strip()
    if title:
        chunks.append(Chunk(text=title, doc_id=paper_id, section="title", score=1.0))

    abstract = str(sample.get("abstract", "")).strip()
    if abstract:
        chunks.append(
            Chunk(text=abstract, doc_id=paper_id, section="abstract", score=1.0)
        )

    full_text = sample.get("full_text", [])
    if isinstance(full_text, list):
        for idx, section in enumerate(full_text):
            if not isinstance(section, dict):
                continue
            section_title = str(section.get("section_title", f"section_{idx}"))
            section_text = str(section.get("full_section_text", "")).strip()
            if not section_text:
                continue
            chunks.append(
                Chunk(
                    text=section_text,
                    doc_id=paper_id,
                    section=section_title,
                    score=0.8,
                    metadata={"section_index": idx},
                )
            )

    return InMemoryCorpus(chunks=chunks)


def resolve_sample_query(sample: dict[str, Any]) -> str:
    """Resolve a per-sample query from supported dataset fields."""
    raw = sample.get("question")
    if raw is None:
        raw = sample.get("query")
    return str(raw or "").strip()


def first_non_empty_sample_query(samples: list[dict[str, Any]]) -> str:
    for sample in samples:
        q = resolve_sample_query(sample)
        if q:
            return q
    return ""


def collect_variants(root: PhysicalNode) -> list[str]:
    variants = [root.variant]
    for child in root.inputs:
        variants.extend(collect_variants(child))
    return variants


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def token_set(text: str) -> set[str]:
    norm = normalize_text(text)
    if not norm:
        return set()
    return {tok for tok in norm.split(" ") if tok}


def score_answer(predicted: str, gold: str) -> dict[str, Any]:
    pred_norm = normalize_text(predicted)
    gold_norm = normalize_text(gold)
    exact = pred_norm == gold_norm and bool(gold_norm)
    contains = bool(gold_norm) and (gold_norm in pred_norm or pred_norm in gold_norm)

    pred_tokens = token_set(predicted)
    gold_tokens = token_set(gold)
    overlap = 0.0
    if pred_tokens and gold_tokens:
        overlap = len(pred_tokens & gold_tokens) / float(len(gold_tokens))

    return {
        "exact_match": exact,
        "contains_match": contains,
        "gold_token_recall": round(overlap, 4),
    }


def score_evidence_overlap(predicted: str, grounding_evidence: Any) -> float:
    if not isinstance(grounding_evidence, list) or not grounding_evidence:
        return 0.0

    pred_tokens = token_set(predicted)
    if not pred_tokens:
        return 0.0

    scores: list[float] = []
    for ev in grounding_evidence:
        ev_text = str(ev)
        ev_tokens = token_set(ev_text)
        if not ev_tokens:
            continue
        scores.append(len(pred_tokens & ev_tokens) / float(len(ev_tokens)))

    if not scores:
        return 0.0
    return round(sum(scores) / float(len(scores)), 4)


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {
            "sample_count": 0,
            "exact_match_rate": 0.0,
            "contains_match_rate": 0.0,
            "avg_gold_token_recall": 0.0,
            "avg_evidence_overlap": 0.0,
            "avg_total_tokens": 0.0,
            "avg_error_count": 0.0,
            "root_variant_counts": {},
            "top_bottlenecks": [],
            "revision_hints": ["No results produced."],
        }

    n = len(results)
    exact_rate = sum(1 for r in results if r["metrics"]["exact_match"]) / float(n)
    contains_rate = sum(1 for r in results if r["metrics"]["contains_match"]) / float(n)
    avg_gold_recall = sum(r["metrics"]["gold_token_recall"] for r in results) / float(n)
    avg_ev_overlap = sum(r["metrics"]["evidence_overlap"] for r in results) / float(n)
    avg_total_tokens = sum(r["execution"]["total_tokens"] for r in results) / float(n)
    avg_error_count = sum(r["execution"]["error_count"] for r in results) / float(n)

    root_variant_counts: dict[str, int] = {}
    bottleneck_counts: dict[str, int] = {}
    for r in results:
        root_variant = str(r["plan"]["root_variant"])
        root_variant_counts[root_variant] = root_variant_counts.get(root_variant, 0) + 1
        bottleneck = str(r["feedback"]["overall"].get("bottleneck", ""))
        if bottleneck:
            bottleneck_counts[bottleneck] = bottleneck_counts.get(bottleneck, 0) + 1

    top_bottlenecks = sorted(
        bottleneck_counts.items(),
        key=lambda x: (-x[1], x[0]),
    )[:5]

    revision_hints: list[str] = []
    if contains_rate < 0.4:
        revision_hints.append(
            "Low answer quality: increase `feedback_iterations` and inspect planner overrides for AGGREGATE/COMPOSE operators."
        )
    if avg_error_count > 0.0:
        revision_hints.append(
            "Execution errors detected: inspect operator errors and prefer safer variants where applicable."
        )
    if avg_total_tokens > 4500:
        revision_hints.append(
            "High token usage: bias planner toward cheaper retrieval/compose variants when quality stays stable."
        )
    if avg_ev_overlap < 0.2:
        revision_hints.append(
            "Low evidence overlap: strengthen retrieval coverage and consider adding VERIFY-oriented plans."
        )
    if not revision_hints:
        revision_hints.append(
            "Quality/cost profile is stable on this batch; validate on a larger sample before revising planner behavior."
        )

    return {
        "sample_count": n,
        "exact_match_rate": round(exact_rate, 4),
        "contains_match_rate": round(contains_rate, 4),
        "avg_gold_token_recall": round(avg_gold_recall, 4),
        "avg_evidence_overlap": round(avg_ev_overlap, 4),
        "avg_total_tokens": round(avg_total_tokens, 2),
        "avg_error_count": round(avg_error_count, 4),
        "root_variant_counts": root_variant_counts,
        "top_bottlenecks": top_bottlenecks,
        "revision_hints": revision_hints,
    }


def build_op_id_variant_map(root: PhysicalNode) -> dict[str, str]:
    counters: dict[str, int] = {}
    out: dict[str, str] = {}

    def walk(node: PhysicalNode) -> None:
        op_name = node.logical_ref.op.value
        op_index = counters.get(op_name, 0)
        counters[op_name] = op_index + 1
        op_id = f"{op_name}_{op_index}"
        out[op_id] = node.variant
        for child in node.inputs:
            walk(child)

    walk(root)
    return out


def aggregate_batch_feedback(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {
            "sample_count": 0,
            "total_error_count": 0,
            "avg_token_cost_per_operator": {},
            "most_frequent_bottleneck": None,
            "warning_frequencies": {},
            "avg_quality_risk": 0.0,
            "avg_token_cost": 0.0,
        }

    op_token_totals: dict[str, float] = {}
    op_token_counts: dict[str, int] = {}
    bottleneck_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    total_errors = 0
    quality_risk_total = 0.0
    token_cost_total = 0.0

    for item in results:
        feedback = item.get("feedback", {})
        overall = feedback.get("overall", {})
        per_operator = feedback.get("per_operator", {})

        total_errors += int(item.get("execution", {}).get("error_count", 0))
        quality_risk_total += float(overall.get("quality_risk", 0.0))
        token_cost_total += float(overall.get("token_cost", 0.0))

        bottleneck = str(overall.get("bottleneck", "")).strip()
        if bottleneck:
            bottleneck_counts[bottleneck] += 1

        warnings = overall.get("warnings", [])
        if isinstance(warnings, list):
            for warning in warnings:
                warning_counts[str(warning)] += 1

        if isinstance(per_operator, dict):
            for op_id, metrics in per_operator.items():
                op_name = str(op_id).split("_", 1)[0]
                token_cost = float(metrics.get("token_cost", 0.0))
                op_token_totals[op_name] = (
                    op_token_totals.get(op_name, 0.0) + token_cost
                )
                op_token_counts[op_name] = op_token_counts.get(op_name, 0) + 1

    avg_token_cost_per_operator = {
        op: round(op_token_totals[op] / max(1, op_token_counts[op]), 4)
        for op in sorted(op_token_totals)
    }
    most_frequent_bottleneck = None
    if bottleneck_counts:
        most_frequent_bottleneck = bottleneck_counts.most_common(1)[0][0]

    n = len(results)
    return {
        "sample_count": n,
        "total_error_count": total_errors,
        "avg_token_cost_per_operator": avg_token_cost_per_operator,
        "most_frequent_bottleneck": most_frequent_bottleneck,
        "warning_frequencies": dict(warning_counts),
        "avg_quality_risk": round(quality_risk_total / float(n), 6),
        "avg_token_cost": round(token_cost_total / float(n), 6),
    }


def validate_single_change(
    key: str,
    variant: str,
) -> bool:
    op_name = key.split("_", 1)[0]
    allowed: list[str] | None = None
    for op, variants in CANDIDATE_VARIANTS.items():
        if op.value == op_name:
            allowed = list(variants)
            break
    if allowed is None:
        return False
    return variant in allowed


def request_single_global_change(
    *,
    planning_client: CountingPlanningClient,
    model: str,
    temperature: float,
    baseline_overrides: dict[str, str],
    batch_report: dict[str, Any],
) -> dict[str, Any]:
    allowed = {op.value: list(variants) for op, variants in CANDIDATE_VARIANTS.items()}
    payload = {
        "task": "Recommend exactly one global physical variant change for this batch.",
        "objective": (
            "Minimize average quality risk while keeping average token cost increase "
            "within 15% compared to baseline."
        ),
        "baseline_plan_overrides": baseline_overrides,
        "batch_report": batch_report,
        "allowed_variants": allowed,
        "output_schema": {
            "variant_overrides": {"<OP or OP_index>": "<variant>"},
            "rationale": "<short explanation>",
        },
        "rules": [
            "Return only JSON.",
            "Return exactly one override entry.",
            "Choose a variant from allowed_variants for that operator.",
        ],
    }

    response = planning_client.complete(
        messages=[
            {
                "role": "system",
                "content": "You are a physical-plan optimizer that must return strict JSON.",
            },
            {"role": "user", "content": json.dumps(payload, indent=2)},
        ],
        model=model,
        temperature=temperature,
    )

    parsed = parse_variant_overrides(response)
    rationale = ""
    try:
        body = json.loads(_strip_code_fence(str(response)))
        rationale = str(body.get("rationale", "")) if isinstance(body, dict) else ""
    except json.JSONDecodeError:
        pass

    if len(parsed) != 1:
        return {
            "accepted": False,
            "reason": "Planner did not return exactly one override.",
            "overrides": {},
            "rationale": rationale,
        }

    key, variant = next(iter(parsed.items()))
    if not validate_single_change(key, variant):
        return {
            "accepted": False,
            "reason": f"Invalid single change: {key} -> {variant}",
            "overrides": {},
            "rationale": rationale,
        }

    return {
        "accepted": True,
        "reason": "ok",
        "overrides": {key: variant},
        "rationale": rationale,
    }


def run_batch_pass(
    *,
    selected: list[dict[str, Any]],
    query_provider: Callable[[dict[str, Any]], str],
    parser: SemanticParser,
    llm: OpenAILLM,
    planning_client: Any,
    model: str,
    planning_rounds: int,
    feedback_iterations: int,
    report_progress: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for idx, sample in enumerate(selected):
        paper_id = str(sample.get("paper_id", f"sample_{idx}"))
        question = query_provider(sample).strip()
        gold_answer = str(sample.get("answer", "")).strip()
        grounding_evidence = sample.get("grounding_evidence", [])

        if not question:
            continue

        corpus = build_corpus(sample)
        pipeline = LmOptimizerPipeline(
            corpus=corpus,
            llm=llm,
            parser=parser,
            use_cost_aware_planner=True,
            planner_preset="balanced",
            model_id=model,
            physical_planner_mode="llm_feedback",
            planning_client=planning_client,
            planning_rounds=max(1, planning_rounds),
        )

        calls_before = getattr(planning_client, "calls", 0)
        run_result = pipeline.run_sync_with_physical_feedback(
            question,
            corpus,
            iterations=max(1, feedback_iterations),
        )
        calls_after = getattr(planning_client, "calls", 0)

        predicted = run_result.execution.answer
        metrics = score_answer(predicted=predicted, gold=gold_answer)
        metrics["evidence_overlap"] = score_evidence_overlap(
            predicted=predicted,
            grounding_evidence=grounding_evidence,
        )
        feedback = pipeline._build_planning_feedback(  # noqa: SLF001
            run_result.physical_plan,
            run_result.execution,
        )

        result_item = {
            "sample_index": idx,
            "paper_id": paper_id,
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": predicted,
            "metrics": metrics,
            "plan": {
                "root_variant": run_result.physical_plan.variant,
                "variants": collect_variants(run_result.physical_plan),
                "op_id_variants": build_op_id_variant_map(run_result.physical_plan),
                "rewrite_count": len(run_result.rewrite_log),
            },
            "execution": {
                "trace": list(run_result.execution.trace),
                "token_counts": dict(run_result.execution.token_counts),
                "total_tokens": int(sum(run_result.execution.token_counts.values())),
                "errors": list(run_result.execution.errors),
                "error_count": len(run_result.execution.errors),
            },
            "feedback": feedback,
            "planning": {
                "planning_rounds": int(planning_rounds),
                "feedback_iterations": int(feedback_iterations),
                "planning_calls_for_sample": calls_after - calls_before,
            },
        }
        results.append(result_item)

        if report_progress:
            print(
                f"[{len(results)}/{len(selected)}] {paper_id} "
                f"root={result_item['plan']['root_variant']} "
                f"errors={result_item['execution']['error_count']} "
                f"tokens={result_item['execution']['total_tokens']}"
            )
    return results


def compare_batch_summaries(
    baseline: dict[str, Any],
    revised: dict[str, Any],
    *,
    token_increase_cap: float,
) -> dict[str, Any]:
    base_risk = float(baseline.get("avg_quality_risk", 0.0))
    rev_risk = float(revised.get("avg_quality_risk", 0.0))
    base_token = float(baseline.get("avg_token_cost", 0.0))
    rev_token = float(revised.get("avg_token_cost", 0.0))

    risk_improved = rev_risk < base_risk
    token_increase_ratio = 0.0
    if base_token > 0.0:
        token_increase_ratio = (rev_token - base_token) / base_token
    token_within_cap = token_increase_ratio <= token_increase_cap

    accepted = risk_improved and token_within_cap
    reason = "accepted"
    if not accepted:
        if not risk_improved and not token_within_cap:
            reason = (
                "rejected: quality risk did not improve and token increase exceeded cap"
            )
        elif not risk_improved:
            reason = "rejected: quality risk did not improve"
        else:
            reason = "rejected: token increase exceeded cap"

    return {
        "accepted": accepted,
        "reason": reason,
        "risk_improved": risk_improved,
        "token_within_cap": token_within_cap,
        "token_increase_ratio": round(token_increase_ratio, 6),
        "quality_risk_delta": round(rev_risk - base_risk, 6),
        "avg_token_cost_delta": round(rev_token - base_token, 6),
    }


def main() -> None:
    # Load environment variables from .env if present.
    load_dotenv()

    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    all_samples = load_samples(input_path)
    selected = select_samples(
        all_samples,
        split=args.split,
        max_samples=args.max_samples,
    )
    if not selected:
        raise RuntimeError("No samples selected. Check --split or dataset contents.")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your shell environment or .env file, "
            "then rerun the command."
        )

    parser_client = LLMClient()
    planning_client = CountingPlanningClient(parser_client)
    parser = SemanticParser(
        client=parser_client,
        model=args.model,
        temperature=args.temperature,
    )
    llm = OpenAILLM(model=args.model)

    report: dict[str, Any]
    if args.protocol == "single_pass":
        results = run_batch_pass(
            selected=selected,
            query_provider=resolve_sample_query,
            parser=parser,
            llm=llm,
            planning_client=planning_client,
            model=args.model,
            planning_rounds=max(1, int(args.planning_rounds)),
            feedback_iterations=max(1, int(args.feedback_iterations)),
            report_progress=True,
        )
        aggregate = summarize(results)
        report = {
            "config": {
                "protocol": args.protocol,
                "input": str(input_path),
                "output": str(output_path),
                "max_samples": int(args.max_samples),
                "split": args.split,
                "planning_rounds": int(args.planning_rounds),
                "feedback_iterations": int(args.feedback_iterations),
                "model": args.model,
                "temperature": float(args.temperature),
                "selected_samples": len(selected),
                "completed_samples": len(results),
                "planning_calls_total": planning_client.calls,
            },
            "aggregate": aggregate,
            "batch_feedback": aggregate_batch_feedback(results),
            "results": results,
        }
    else:
        seed_query = first_non_empty_sample_query(selected)
        if not seed_query:
            raise RuntimeError(
                "No non-empty sample question/query found in selected samples."
            )

        # Build one baseline global plan signature from an actual sample query.
        seed_corpus = build_corpus(selected[0])
        seed_pipeline = LmOptimizerPipeline(
            corpus=seed_corpus,
            llm=llm,
            parser=parser,
            use_cost_aware_planner=True,
            planner_preset="balanced",
            model_id=args.model,
            physical_planner_mode="cost",
            planning_rounds=1,
        )
        baseline_logical = seed_pipeline.parse(seed_query)
        baseline_optimized, _ = seed_pipeline.optimize(baseline_logical)
        baseline_plan = seed_pipeline.build_physical(baseline_optimized)
        baseline_overrides = build_op_id_variant_map(baseline_plan)

        # Baseline phase: fixed plan, no per-sample internal learning.
        baseline_planning_client = FixedPlanningClient(baseline_overrides)
        baseline_results = run_batch_pass(
            selected=selected,
            query_provider=resolve_sample_query,
            parser=parser,
            llm=llm,
            planning_client=baseline_planning_client,
            model=args.model,
            planning_rounds=1,
            feedback_iterations=1,
            report_progress=True,
        )
        baseline_summary = aggregate_batch_feedback(baseline_results)

        # Revision phase: request one global variant change from aggregated report.
        change_request = request_single_global_change(
            planning_client=planning_client,
            model=args.model,
            temperature=args.temperature,
            baseline_overrides=baseline_overrides,
            batch_report=baseline_summary,
        )

        revised_overrides = dict(baseline_overrides)
        if change_request["accepted"]:
            revised_overrides.update(change_request["overrides"])

        revised_planning_client = FixedPlanningClient(revised_overrides)
        revised_results = run_batch_pass(
            selected=selected,
            query_provider=resolve_sample_query,
            parser=parser,
            llm=llm,
            planning_client=revised_planning_client,
            model=args.model,
            planning_rounds=1,
            feedback_iterations=1,
            report_progress=True,
        )
        revised_summary = aggregate_batch_feedback(revised_results)

        decision = compare_batch_summaries(
            baseline_summary,
            revised_summary,
            token_increase_cap=float(args.token_increase_cap),
        )

        winner = "revised" if decision["accepted"] else "baseline"
        report = {
            "config": {
                "protocol": args.protocol,
                "input": str(input_path),
                "output": str(output_path),
                "max_samples": int(args.max_samples),
                "split": args.split,
                "model": args.model,
                "temperature": float(args.temperature),
                "token_increase_cap": float(args.token_increase_cap),
                "selected_samples": len(selected),
            },
            "baseline": {
                "global_plan": {
                    "root_variant": baseline_plan.variant,
                    "op_id_variants": baseline_overrides,
                },
                "aggregate_summary": baseline_summary,
                "results": baseline_results,
            },
            "revision": {
                "single_change_request": change_request,
                "revised_overrides": revised_overrides,
            },
            "revised": {
                "aggregate_summary": revised_summary,
                "results": revised_results,
            },
            "delta": decision,
            "winner": winner,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    final_summary = (
        report.get("aggregate")
        if args.protocol == "single_pass"
        else report.get("revised", {}).get("aggregate_summary")
    )
    print(f"Saved report: {output_path}")
    print("Aggregate summary:")
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
