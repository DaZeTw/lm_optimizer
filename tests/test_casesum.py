"""
Run the LmOptimizer pipeline on QASPER samples.

Usage:
    python tests/test_casesum.py [--n 5] [--iterations 2] [--log-dir logs/]
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from executor.corpus import InMemoryCorpus, OpenAIEmbedder, OpenAILLM
from ir.evidence import Chunk
from pipeline import LmOptimizerPipeline

DATA_PATH = ROOT / "qasper_top_100_context.json"

_TASK_DESCRIPTION = (
    "Given a scientific paper, a question, and a candidate answer, "
    "retrieve the exact evidence sentences from the paper that support the answer."
)

_EVALUATION_CRITERIA = (
    "Return only exact text spans from the source document as evidence. "
    "Do not modify, paraphrase, or explain. "
    "Each output must be directly grounded in the corpus. "
    "Evaluation is based on overlap with gold evidence passages."
)


# ── Data ──────────────────────────────────────────────────────────


@dataclass
class QasperSample:
    paper_id: str
    question: str
    answer: str
    grounding_evidence: list[str]
    corpus: InMemoryCorpus


def load_samples(path: Path = DATA_PATH, n: int | None = None) -> list[QasperSample]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(records, dict):
        records = [records]

    embedder = OpenAIEmbedder()
    samples: list[QasperSample] = []

    for rec in records[:n] if n else records:
        if rec.get("is_unanswerable"):
            continue
        samples.append(
            QasperSample(
                paper_id=rec["paper_id"],
                question=rec["question"],
                answer=rec["answer"],
                grounding_evidence=rec.get("grounding_evidence", []),
                corpus=_build_corpus(rec, embedder),
            )
        )
    return samples


def _build_corpus(rec: dict, embedder: OpenAIEmbedder) -> InMemoryCorpus:
    texts: list[str] = []
    abstract = (rec.get("abstract") or "").strip()
    if abstract:
        texts.append(abstract)
    for sec in rec.get("full_text", []):
        text = (sec.get("full_section_text") or "").strip()
        if text:
            texts.append(text)
    corpus = InMemoryCorpus(embedder=embedder)
    corpus.add_documents(texts, source=rec["paper_id"])
    return corpus


# ── Pipeline ──────────────────────────────────────────────────────


def make_pipeline(corpus: InMemoryCorpus) -> LmOptimizerPipeline:
    from parser.semantic_parser import LLMClient

    return LmOptimizerPipeline(
        corpus=corpus,
        llm=OpenAILLM(model="gpt-4o-mini"),
        planning_client=LLMClient(),
    )


def sample_tuples(samples: list[QasperSample]) -> list[tuple[str, str, list[str]]]:
    return [(s.question, s.answer, s.grounding_evidence) for s in samples]


# ── Scoring ───────────────────────────────────────────────────────


def evidence_recall(predicted: str, gold_evidence: list[str]) -> float:
    """Fraction of gold evidence passages with any token overlap in predicted."""
    if not gold_evidence:
        return 0.0
    pred_toks = set(predicted.lower().split())
    hits = sum(bool(set(e.lower().split()) & pred_toks) for e in gold_evidence)
    return hits / len(gold_evidence)


# ── Batch runner ──────────────────────────────────────────────────


@dataclass
class RunResult:
    paper_id: str
    question: str
    gold_evidence: list[str]
    predicted: str
    evidence_recall: float
    judge_accuracy: float
    tst_versions: int
    errors: list[str] = field(default_factory=list)


def run_batch(
    samples: list[QasperSample],
    iterations: int = 1,
    log_dir: Path | None = None,
) -> list[RunResult]:
    if not samples:
        return []

    pipeline = make_pipeline(samples[0].corpus)
    log_path = (log_dir / "round_shared_plan.json") if log_dir else None
    node_feedbacks_path = (log_dir / "round_node_feedbacks.json") if log_dir else None

    result = pipeline.run_sync_with_samples(
        task_description=_TASK_DESCRIPTION,
        evaluation_criteria=_EVALUATION_CRITERIA,
        samples=sample_tuples(samples),
        iterations=iterations,
        log_path=log_path,
        node_feedbacks_path=node_feedbacks_path,
    )

    run_results: list[RunResult] = []
    for i, (s, execution, feedback) in enumerate(
        zip(samples, result.executions, result.feedbacks)
    ):
        print(f"  [{i+1}/{len(samples)}] {s.paper_id} — {s.question[:60]}...")
        predicted = execution.answer
        recall = evidence_recall(predicted, s.grounding_evidence)
        accuracy = feedback.accuracy if feedback else 0.0

        run_results.append(
            RunResult(
                paper_id=s.paper_id,
                question=s.question,
                gold_evidence=s.grounding_evidence,
                predicted=predicted,
                evidence_recall=recall,
                judge_accuracy=accuracy,
                tst_versions=len(result.tst_versions),
                errors=execution.errors,
            )
        )

        print(
            f"    Gold      : {s.grounding_evidence[0][:80] if s.grounding_evidence else '(none)'}..."
        )
        print(f"    Predicted : {predicted[:100]}")
        print(
            f"    Recall={recall:.2f}  Judge={accuracy:.2f}  TST versions={len(result.tst_versions)}  Errors={len(execution.errors)}"
        )

    return run_results


# ── Reporting ─────────────────────────────────────────────────────


def print_summary(label: str, results: list[RunResult]) -> None:
    n = max(1, len(results))
    print(f"\n── {label} ({len(results)} samples) ──")
    print(f"  Avg evidence recall : {sum(r.evidence_recall for r in results) / n:.3f}")
    print(f"  Avg judge score     : {sum(r.judge_accuracy for r in results) / n:.3f}")
    print(f"  Avg TST versions    : {sum(r.tst_versions for r in results) / n:.1f}")


# ── Entrypoint ────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run QASPER pipeline evaluation")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument(
        "--iterations",
        type=int,
        default=2,
        help="Number of rounds to run: round 1 is initial, later rounds use TST revision.",
    )
    parser.add_argument("--log-dir", type=str, default=None)
    args = parser.parse_args()

    log_dir = Path(args.log_dir) if args.log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"Logging to: {log_dir}/")

    samples = load_samples(n=args.n)
    print(f"Loaded {len(samples)} answerable samples.")

    print(
        f"\n{'='*60}\n"
        f"{args.iterations} round(s): round 1 initial, later rounds revised\n"
        f"{'='*60}"
    )
    results = run_batch(samples, iterations=args.iterations, log_dir=log_dir)
    print_summary(f"Final round ({args.iterations})", results)
