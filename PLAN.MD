# LM Optimizer Framework Redesign

## Context
The codebase has a working parse → optimize → plan → execute pipeline, but the physical planner is heuristic/cost-table based and feedback is coarse (variant-level averages). The user wants a cleaner architecture with:
- An **LLM-driven physical planner** with a distinct init phase and revise phase
- **Per-node execution wrappers** that produce structured feedback
- **LLM-as-judge** accuracy scoring
- A **typed Feedback dataclass** that ties together per-node cost, result, and gold answer

---

## Architecture Overview

```
Query
  │
  ▼
SemanticParser          → LogicalNode DAG          [EXISTING: parser/semantic_parser.py]
  │
  ▼
OptimizerEngine         → optimized LogicalNode    [EXISTING: optimizer/engine.py]
  │
  ▼
LLMPhysicalPlanner ─────────────────────────────── [NEW: planner/llm_planner.py]
  ├─ init(query, catalog, logical_plan)
  │    LLM output (JSON) → PhysicalPlanParser → PhysicalNode DAG
  └─ revise(query, catalog, prev_physical, feedback)
       LLM output (JSON) → PhysicalPlanParser → PhysicalNode DAG
  │
  ▼
PlanRunner (with per-node wrappers)  ─────────────  [MODIFY: executor/runner.py]
  │   records latency_ms, token_cost, output per node
  │
  ▼
ExecutionResult + list[NodeFeedback]
  │
  ▼
AccuracyJudge           → accuracy: float          [NEW: cost_model/judge.py]
  │
  ▼
Feedback(items, accuracy, result, gold_ans)  ──────  [NEW: ir/feedback.py]
  │
  └─ fed back to LLMPhysicalPlanner.revise()
```

---

## Files to Create / Modify

### 1. `ir/feedback.py` — NEW
Typed feedback dataclasses.

```python
@dataclass
class NodeFeedback:
    op_id: str          # e.g. "AGGREGATE_0"
    variant: str        # chosen variant name
    token_cost: int     # tokens consumed
    latency_ms: float   # wall-clock time for this node
    output_summary: str # first ~200 chars of evidence output (for planner context)

@dataclass
class Feedback:
    items: list[NodeFeedback]
    accuracy: float     # 0.0–1.0 from LLM-as-judge
    result: str         # predicted answer (execution.answer)
    gold_ans: str       # expected answer
```

### 2. `planner/variant_candidates.py` — RESTORE (was deleted in refactor)
Restore the `CANDIDATE_VARIANTS: dict[Op, list[str]]` mapping that the new planner prompt uses to enumerate valid variants for each operator.

### 3. `planner/plan_parser.py` — NEW
Parses LLM-produced JSON into a `PhysicalNode` DAG.

Expected JSON format:
```json
{
  "op": "AGGREGATE", "variant": "DirectGenerate",
  "params": {"goal": "..."},
  "inputs": [{ "op": "RANK", "variant": "SimilarityRank", "params": {...}, "inputs": [...] }]
}
```

```python
def parse_physical_plan(data: dict) -> PhysicalNode
```
- Validates `variant` is in `CANDIDATE_VARIANTS[op]`; falls back to first candidate if invalid
- Recursively builds `PhysicalNode` tree with `cost=CostVector()` (populated later by scorer)

### 4. `planner/llm_planner.py` — NEW
LLM-driven physical planner with two distinct phases.

```python
class LLMPhysicalPlanner:
    def __init__(self, client: LLMClient, model: str, catalog: SystemCatalog | None)

    def init(self, query: str, logical: LogicalNode) -> PhysicalNode
        # Sends: query, catalog stats, logical plan (to_dict()), CANDIDATE_VARIANTS
        # Returns: PhysicalNode via parse_physical_plan()

    def revise(self, query: str, logical: LogicalNode,
               prev_physical: PhysicalNode, feedback: Feedback) -> PhysicalNode
        # Sends: same as init + serialized prev_physical + feedback summary
        # Returns: PhysicalNode via parse_physical_plan()
```

Prompt context for both phases:
- Task description (query)
- Catalog: `context_window`, `avg_chunk_tokens`
- Logical plan JSON
- Available variants per operator (from `CANDIDATE_VARIANTS`)

Additional context for revise phase:
- Previous physical plan (variant choices + params)
- Feedback: per-node (op_id, variant, token_cost, latency_ms, output_summary), accuracy, result vs gold_ans

### 5. `cost_model/judge.py` — NEW
LLM-as-judge accuracy scorer.

```python
class AccuracyJudge:
    def __init__(self, client: LLMClient, model: str = "gpt-4o-mini")
    async def score(self, result: str, gold_ans: str) -> float
        # Calls LLM: "Does this answer correctly address the question given gold answer?"
        # Returns float 0.0–1.0
```

### 6. `executor/runner.py` — MODIFY
Wrap each variant call with per-node timing and token recording. `_execute()` returns both `EvidenceSet` and a `NodeFeedback` item.

Changes:
- Add `import time` 
- In `_execute()`: record `start = time.monotonic()` before variant call, `latency_ms` after
- Track `token_cost` per node (from `result.token_estimate()`)
- Collect `NodeFeedback` items into a list returned alongside `ExecutionResult`
- `run()` returns `tuple[ExecutionResult, list[NodeFeedback]]`

### 7. `pipeline.py` — MODIFY
Simplify to use the new planner and feedback structures.

Changes:
- Import `LLMPhysicalPlanner`, `Feedback`, `AccuracyJudge`
- `build_physical()` → delegates to `LLMPhysicalPlanner.init()`
- Add `run_with_feedback(query, gold_ans, iterations)` loop:
  ```
  physical = planner.init(query, logical)
  for i in range(iterations):
      execution, node_feedbacks = await runner.run(physical)
      accuracy = await judge.score(execution.answer, gold_ans)
      feedback = Feedback(items=node_feedbacks, accuracy=accuracy,
                          result=execution.answer, gold_ans=gold_ans)
      if i < iterations - 1:
          physical = planner.revise(query, logical, physical, feedback)
  ```
- Remove `LlmFeedbackPlanner`, `CostAwarePlanner`, `build_physical_plan` imports (they're replaced)
- Keep `_record_execution_feedback` for telemetry (optional, can remain)
- Update `PipelineResult` to include `feedback: Feedback | None`

---

## Reuse / Keep As-Is
- `parser/` — fully retained
- `optimizer/` — fully retained  
- `ir/nodes.py`, `ir/evidence.py`, `ir/ops.py` — fully retained
- `catalog/` — fully retained
- `executor/registry.py`, `executor/ops/` — fully retained (18 variant implementations)
- `cost_model/telemetry.py`, `cost_model/scorer.py` — retained, scorer still useful for annotating costs post-execution
- `executor/corpus.py` — retained (`LLMClient` in `parser/semantic_parser.py` reused for planner and judge)

---

## Verification
1. Unit test `planner/plan_parser.py`: feed well-formed and malformed JSON, assert correct `PhysicalNode` or graceful fallback
2. Unit test `LLMPhysicalPlanner` init/revise with a mock `LLMClient` returning a fixed JSON plan
3. Unit test `AccuracyJudge` with mock LLM
4. Integration test: `run_with_feedback(query, gold_ans, iterations=2)` on `InMemoryCorpus` + `MockLLM` — verify `Feedback` fields populated and second iteration uses revised plan
5. Existing executor tests (`test_physical.py`) — confirm `runner.run()` still works with updated return type
