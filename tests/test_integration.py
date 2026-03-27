"""
Integration test for the full Phase 1 pipeline.

Calls OpenAI, parses the algebraic expression into a LogicalNode DAG,
and saves the result as JSON to tests/fixtures/.

Run with:
    uv run pytest tests/test_integration.py -v -s

Requires:
    OPENAI_API_KEY set in your .env file
"""

import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

from parser.semantic_parser import LLMClient, SemanticParser

from ir.nodes import LogicalNode

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURES_DIR.mkdir(exist_ok=True)

# ── Queries to test ────────────────────────────────────────────────
# Each entry: (test_id, query, expected_root_op)

TEST_CASES = [
    (
        "simple_qa",
        "What is the main contribution of this paper?",
        "AGGREGATE",
    ),
    (
        "multi_doc_compare",
        "Compare the evaluation metrics used in paper A and paper B.",
        "VERIFY",
    ),
    (
        "holistic_qa",
        (
            "Find the methodology sections in these 10 PDFs, "
            "extract the evaluation metrics, and compare how they handle formula accuracy."
        ),
        "AGGREGATE",
    ),
    (
        "multi_hop",
        (
            "What datasets were used across all papers, "
            "and which paper achieved the best results on each dataset?"
        ),
        "AGGREGATE",
    ),
]


# ── Fixture: one shared parser for the whole session ──────────────


@pytest.fixture(scope="session")
def parser():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set — skipping integration tests")
    client = LLMClient(api_key=api_key)
    return SemanticParser(client=client, model="gpt-4o", temperature=0.0)


# ── Helpers ────────────────────────────────────────────────────────


def _save(test_id: str, query: str, node: LogicalNode, raw_expr: str) -> Path:
    """Save the plan to tests/fixtures/<test_id>.json"""
    payload = {
        "test_id": test_id,
        "query": query,
        "raw_expression": raw_expr,
        "plan": node.to_dict(),
    }
    path = FIXTURES_DIR / f"{test_id}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def _print_result(test_id: str, query: str, node: LogicalNode, path: Path) -> None:
    print(f"\n{'─' * 60}")
    print(f"Test    : {test_id}")
    print(f"Query   : {query}")
    print(f"Plan    :\n{node.pretty()}")
    print(f"Saved   : {path}")
    print(f"{'─' * 60}")


# ── Parametrized integration tests ────────────────────────────────


@pytest.mark.parametrize("test_id,query,expected_root_op", TEST_CASES)
def test_parse_and_save(parser, test_id, query, expected_root_op, monkeypatch):
    """
    For each query:
      1. Call the LLM to produce an algebraic plan expression.
      2. Parse the expression into a LogicalNode DAG.
      3. Assert the root operator matches expectation.
      4. Assert the plan roundtrips through to_dict / from_dict.
      5. Save the result to tests/fixtures/<test_id>.json.
    """
    # Capture the raw LLM expression before parsing
    raw_expressions = []
    original_complete = parser.client.complete

    def capturing_complete(messages, model, temperature):
        result = original_complete(messages, model, temperature)
        raw_expressions.append(result)
        return result

    monkeypatch.setattr(parser.client, "complete", capturing_complete)

    # Run the parser
    node = parser.parse(query)

    # Structural assertions
    assert node is not None
    assert node.op.value == expected_root_op, (
        f"Expected root op {expected_root_op!r}, got {node.op.value!r}\n"
        f"Plan:\n{node.pretty()}"
    )

    # Every leaf must be an I() node
    def check_leaves(n: LogicalNode):
        if not n.inputs:
            assert n.op.value == "I", f"Leaf node must be I(), got {n.op.value!r}"
        for child in n.inputs:
            check_leaves(child)

    check_leaves(node)

    # Roundtrip
    restored = LogicalNode.from_dict(node.to_dict())
    assert restored == node, "Roundtrip to_dict/from_dict failed"

    # Save + print
    raw_expr = raw_expressions[0] if raw_expressions else ""
    path = _save(test_id, query, node, raw_expr)
    _print_result(test_id, query, node, path)


# ── Summary test: load all saved fixtures and validate ─────────────


def test_fixtures_are_valid_json():
    """
    After the parametrized tests run, confirm every fixture file
    can be loaded back into a LogicalNode without error.
    """
    fixture_files = list(FIXTURES_DIR.glob("*.json"))
    if not fixture_files:
        pytest.skip("No fixture files found — run test_parse_and_save first")

    for path in fixture_files:
        data = json.loads(path.read_text())
        plan_dict = (
            data.get("plan") or data.get("optimized_plan") or data.get("naive_plan")
        )
        if plan_dict is None:
            pytest.skip(f"No plan payload in fixture: {path.name}")
        node = LogicalNode.from_dict(plan_dict)
        assert node.op is not None, f"Failed to restore plan from {path.name}"
        print(f"  loaded {path.name}  root={node.op.value}")
        print(f"  loaded {path.name}  root={node.op.value}")
