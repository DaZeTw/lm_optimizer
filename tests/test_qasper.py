"""
End-to-end tests against the QASPER top-100 dataset using the real OpenAI API.

Requires OPENAI_API_KEY to be set (load from .env).

Run tests:
    pytest tests/test_qasper.py -s

Direct script:
    python tests/test_qasper.py [--n 5]
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
import pytest

# ── resolve project root ──────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from executor.corpus import InMemoryCorpus, OpenAILLM
from ir.evidence import Chunk
from pipeline import LmOptimizerPipeline

DATA_PATH = ROOT / "qasper_top_100_context.json"

# ── data helpers ──────────────────────────────────────────────────────────────


@dataclass
class QasperSample:
    paper_id: str
    question: str
    answer: str
    grounding_evidence: list[str]
    is_unanswerable: bool
    corpus: InMemoryCorpus


def load_samples(path: Path = DATA_PATH, n: int | None = None) -> list[QasperSample]:
    """Load QASPER records and build one InMemoryCorpus per paper."""
    records = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(records, dict):
        records = [records]

    samples = []
    for rec in records[:n] if n else records:
        if rec.get("is_unanswerable"):
            continue
        corpus = _build_corpus(rec)
        samples.append(
            QasperSample(
                paper_id=rec["paper_id"],
                question=rec["question"],
                answer=rec["answer"],
                grounding_evidence=rec.get("grounding_evidence", []),
                is_unanswerable=False,
                corpus=corpus,
            )
        )
    return samples


def _build_corpus(rec: dict) -> InMemoryCorpus:
    chunks: list[Chunk] = []

    abstract = (rec.get("abstract") or "").strip()
    if abstract:
        chunks.append(Chunk(text=abstract, doc_id=rec["paper_id"], section="abstract"))

    for sec in rec.get("full_text", []):
        text = (sec.get("full_section_text") or "").strip()
        if text:
            chunks.append(
                Chunk(
                    text=text,
                    doc_id=rec["paper_id"],
                    section=sec.get("section_title") or "body",
                )
            )
    return InMemoryCorpus(chunks=chunks)


# ── scoring helpers ───────────────────────────────────────────────────────────


def token_recall(predicted: str, gold: str) -> float:
    """Fraction of gold tokens that appear in predicted."""
    gold_toks = set(gold.lower().split())
    pred_toks = set(predicted.lower().split())
    if not gold_toks:
        return 0.0
    return len(gold_toks & pred_toks) / len(gold_toks)


def exact_match(predicted: str, gold: str) -> bool:
    return predicted.strip().lower() == gold.strip().lower()


# ── pipeline factory ──────────────────────────────────────────────────────────


def make_pipeline(corpus: InMemoryCorpus) -> LmOptimizerPipeline:
    from parser.semantic_parser import LLMClient, SemanticParser

    llm = OpenAILLM(model="gpt-4o-mini")
    planning_client = LLMClient()
    parser = SemanticParser(client=planning_client)

    return LmOptimizerPipeline(
        corpus=corpus,
        llm=llm,
        planning_client=planning_client,
        parser=parser,
    )


# ── tests ─────────────────────────────────────────────────────────────────────


class TestQasperIntegration:
    """Real OpenAI calls — skipped unless OPENAI_API_KEY is set."""

    @pytest.fixture(autouse=True)
    def require_api_key(self):
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

    def test_corpus_builds_from_first_sample(self):
        samples = load_samples(n=1)
        assert samples, "No answerable samples found in dataset"
        s = samples[0]
        assert len(s.corpus.chunks) > 0
        assert all(isinstance(c, Chunk) for c in s.corpus.chunks)

    def test_bm25_search_returns_chunks(self):
        samples = load_samples(n=1)
        s = samples[0]
        results = s.corpus.bm25_search(s.question, top_k=3)
        assert isinstance(results, list)

    def test_single_sample_runs_end_to_end(self):
        samples = load_samples(n=1)
        s = samples[0]
        pipeline = make_pipeline(s.corpus)
        result = pipeline.run_sync_with_samples(
            "QA over scientific papers", [(s.question, "")], iterations=1
        )

        assert result.logical_plan is not None
        assert result.optimized_plan is not None
        assert result.physical_plan is not None
        assert result.execution is not None
        assert isinstance(result.execution.answer, str)

    def test_single_sample_with_feedback(self):
        samples = load_samples(n=1)
        s = samples[0]
        pipeline = make_pipeline(s.corpus)
        result = pipeline.run_sync_with_samples(
            "QA over scientific papers",
            [(s.question, s.answer)],
            iterations=2,
        )

        assert result.feedback is not None
        assert 0.0 <= result.feedback.accuracy <= 1.0
        assert result.feedback.gold_ans == s.answer
        assert len(result.feedback.items) > 0

        for item in result.feedback.items:
            assert item.op_id
            assert item.variant
            assert item.latency_ms >= 0.0

    def test_rewrite_log_populated(self):
        samples = load_samples(n=1)
        s = samples[0]
        pipeline = make_pipeline(s.corpus)
        result = pipeline.run_sync_with_samples(
            "QA over scientific papers", [(s.question, "")], iterations=1
        )
        assert isinstance(result.rewrite_log, list)

    def test_node_feedbacks_cover_all_variants(self):
        samples = load_samples(n=1)
        s = samples[0]
        pipeline = make_pipeline(s.corpus)
        result = pipeline.run_sync_with_samples(
            "QA over scientific papers", [(s.question, s.answer)], iterations=1
        )
        fb = result.feedback
        assert fb is not None
        variants_in_feedback = {item.variant for item in fb.items}
        assert len(variants_in_feedback) >= 1

    def test_single_sample_real(self):
        samples = load_samples(n=1)
        s = samples[0]
        pipeline = make_pipeline(s.corpus)
        result = pipeline.run_sync_with_samples(
            "QA over scientific papers",
            [(s.question, s.answer)],
            iterations=1,
        )
        recall = token_recall(result.execution.answer, s.answer)
        print(f"\n[{s.paper_id}] Q: {s.question}")
        print(f"  Gold      : {s.answer}")
        print(f"  Predicted : {result.execution.answer}")
        print(f"  Token recall  : {recall:.2f}")
        print(f"  Judge accuracy: {result.feedback.accuracy:.2f}")

        assert result.feedback is not None


# ── CLI runner ────────────────────────────────────────────────────────────────


@dataclass
class RunResult:
    paper_id: str
    question: str
    gold_ans: str
    predicted: str
    token_recall: float
    judge_accuracy: float
    errors: list[str] = field(default_factory=list)


def run_batch(
    samples: list[QasperSample], iterations: int = 1, log_dir: Path | None = None
) -> list[RunResult]:
    """Run each sample through the pipeline and return results."""
    run_results: list[RunResult] = []

    for i, s in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] {s.paper_id} — {s.question[:60]}...")
        pipeline = make_pipeline(s.corpus)
        log_path = (
            log_dir / f"round{iterations}_{s.paper_id}_{i}.json"
            if log_dir is not None
            else None
        )
        try:
            result = pipeline.run_sync_with_samples(
                "QA over scientific papers",
                [(s.question, s.answer)],
                iterations=iterations,
                log_path=log_path,
            )
            predicted = result.execution.answer
            recall = token_recall(predicted, s.answer)
            accuracy = result.feedback.accuracy if result.feedback else 0.0
            errors = result.execution.errors
        except Exception as exc:
            predicted = ""
            recall = 0.0
            accuracy = 0.0
            errors = [str(exc)]
            print(f"    ERROR: {exc}")

        run_results.append(
            RunResult(
                paper_id=s.paper_id,
                question=s.question,
                gold_ans=s.answer,
                predicted=predicted,
                token_recall=recall,
                judge_accuracy=accuracy,
                errors=errors,
            )
        )
        print(f"    Gold      : {s.answer}")
        print(f"    Predicted : {predicted[:100]}")
        print(f"    Recall={recall:.2f}  Judge={accuracy:.2f}  Errors={len(errors)}")

    return run_results


def _print_summary(label: str, results: list[RunResult]) -> None:
    avg_recall = sum(r.token_recall for r in results) / max(1, len(results))
    avg_accuracy = sum(r.judge_accuracy for r in results) / max(1, len(results))
    print(f"\n── {label} ({len(results)} samples) ──")
    print(f"  Avg token recall : {avg_recall:.3f}")
    print(f"  Avg judge score  : {avg_accuracy:.3f}")


def _print_comparison(round1: list[RunResult], round2: list[RunResult]) -> None:
    print("\n── Round 1 → Round 2 improvement ──")
    print(f"  {'Paper':>12}  {'Recall R1':>10}  {'Recall R2':>10}  {'Judge R1':>9}  {'Judge R2':>9}")
    for r1, r2 in zip(round1, round2):
        recall_arrow = "▲" if r2.token_recall > r1.token_recall else ("▼" if r2.token_recall < r1.token_recall else "=")
        print(
            f"  {r1.paper_id:>12}  {r1.token_recall:>10.2f}  {r2.token_recall:>10.2f} {recall_arrow}"
            f"  {r1.judge_accuracy:>9.2f}  {r2.judge_accuracy:>9.2f}"
        )
    avg_r1_recall = sum(r.token_recall for r in round1) / max(1, len(round1))
    avg_r2_recall = sum(r.token_recall for r in round2) / max(1, len(round2))
    avg_r1_judge = sum(r.judge_accuracy for r in round1) / max(1, len(round1))
    avg_r2_judge = sum(r.judge_accuracy for r in round2) / max(1, len(round2))
    print(f"\n  Avg recall : {avg_r1_recall:.3f} → {avg_r2_recall:.3f}  (Δ {avg_r2_recall - avg_r1_recall:+.3f})")
    print(f"  Avg judge  : {avg_r1_judge:.3f} → {avg_r2_judge:.3f}  (Δ {avg_r2_judge - avg_r1_judge:+.3f})")


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(description="Run QASPER pipeline evaluation")
    arg_parser.add_argument("--n", type=int, default=5, help="Number of samples")
    arg_parser.add_argument("--log-dir", type=str, default=None, help="Directory to save per-sample JSON logs")
    args = arg_parser.parse_args()

    log_dir = Path(args.log_dir) if args.log_dir else None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"Logging to: {log_dir}/")

    samples = load_samples(n=args.n)

    print(f"\n{'='*60}")
    print("Round 1 — Initial plan (no feedback)")
    print('='*60)
    round1 = run_batch(samples, iterations=1, log_dir=log_dir)
    _print_summary("Round 1", round1)

    print(f"\n{'='*60}")
    print("Round 2 — Revised plan (init → execute → judge → revise → execute)")
    print('='*60)
    round2 = run_batch(samples, iterations=2, log_dir=log_dir)
    _print_summary("Round 2", round2)

    _print_comparison(round1, round2)
