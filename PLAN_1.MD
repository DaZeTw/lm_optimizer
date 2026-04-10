# Plan: Multi-sample task-description refactor

## Context

The pipeline currently operates on a single query at a time. The refactor shifts to a
task-description + sample-queries model where:
- The **semantic parser** understands what the overall task is before producing a plan
- The **LLM planner** can revise based on feedback from *multiple* sample runs
- The **pipeline** accepts a set of samples and a task description instead of one query

## 1. `parser/prompts.py`

**Change**: `build_user_message()` signature and body.

```python
def build_user_message(
    task_description: str,
    sample_queries: list[str],
    context_window: int | None = None,
    avg_chunk_tokens: float | None = None,
) -> str:
```

**New user message structure**:
```
## Task context
{task_description}

## Representative sample queries
These are example questions this plan must handle:
1. {sample_queries[0]}
2. {sample_queries[1]}
...

## Examples of query → plan mappings
{FEW_SHOT_EXAMPLES block — unchanged}

{optional corpus context block — unchanged}

## Your job
Produce a single general-purpose logical plan that can handle the sample queries above.
The plan should be a reusable operator DAG covering the general reasoning pattern.

Plan:
```

If `sample_queries` is empty, replace the numbered list with `(none provided)`.

The final line changes from `Query: {query}\nPlan:` → `Plan:` (task framing replaces per-query prompt).

## 2. `parser/semantic_parser.py`

**Change**: `parse()` signature.

```python
def parse(self, task_description: str, sample_queries: list[str]) -> LogicalNode:
```

Only the call to `build_user_message()` changes. All retry/self-correction logic is
preserved verbatim — it operates on the `messages` list generically.

## 3. `planner/llm_planner.py`

**Change**: `revise()` signature and `_REVISE_USER_TEMPLATE`.

```python
def revise(
    self,
    query: str,
    logical: LogicalNode,
    prev_physical: PhysicalNode,
    feedbacks: list[Feedback],          # was: feedback: Feedback
) -> PhysicalNode:
```

**New `_REVISE_USER_TEMPLATE`**:
```
## Task
{query}

## Corpus / model context
...

## Logical plan (JSON)
{logical_json}

## Previous physical plan
{prev_physical_json}

## Execution feedback across {num_samples} sample(s)

{feedback_blocks}

Revise the physical plan to improve accuracy and/or reduce cost across all samples.
Output only the revised physical plan JSON.
```

**Private helper** formats each `Feedback` as a labeled block:
```python
def _format_feedback_block(idx: int, fb: Feedback) -> str:
    node_lines = "\n".join(
        f"    {item.op_id} ({item.variant}): "
        f"tokens={item.token_cost}, latency={item.latency_ms:.1f}ms, "
        f"output={item.output_summary!r}"
        for item in fb.items
    )
    return (
        f"### Sample {idx + 1}\n"
        f"Accuracy: {fb.accuracy:.2f}\n"
        f"Result  : {fb.result}\n"
        f"Gold ans: {fb.gold_ans}\n"
        f"Per-node:\n"
        f"{node_lines or '    (none)'}"
    )
```

`feedback_blocks = "\n\n".join(_format_feedback_block(i, fb) for i, fb in enumerate(feedbacks))`

A single-element `list[Feedback]` renders as `### Sample 1` — equivalent to old behavior.

## 4. `pipeline.py`

### `PipelineResult` changes

```python
@dataclass(frozen=True)
class PipelineResult:
    task_description: str
    sample_queries: list[str]
    logical_plan: LogicalNode
    optimized_plan: LogicalNode
    physical_plan: PhysicalNode
    rewrite_log: list[RewriteEntry]
    executions: list[ExecutionResult]    # one per sample (last iteration)
    feedbacks: list[Feedback]            # one per sample (last iteration)

    # Backward-compat convenience properties
    @property
    def execution(self) -> ExecutionResult | None:
        return self.executions[-1] if self.executions else None

    @property
    def feedback(self) -> Feedback | None:
        return self.feedbacks[-1] if self.feedbacks else None
```

`query: str` is removed. `feedback: Feedback | None` is removed (replaced by properties).

### `parse()` method

```python
def parse(self, task_description: str, sample_queries: list[str]) -> LogicalNode:
    return self.parser.parse(task_description, sample_queries)
```

### New `run_with_samples()` async method

```python
async def run_with_samples(
    self,
    task_description: str,
    samples: list[tuple[str, str]],   # (query, gold_ans), shared corpus
    iterations: int = 2,
    log_path: str | Path | None = None,
) -> PipelineResult:
```

Flow:
1. Extract `sample_queries` and `gold_answers` from `samples`
2. `logical = self.parse(task_description, sample_queries)`
3. `optimized, rewrites = self.optimize(logical)`
4. `physical = self._init_physical(task_description, optimized)`
5. For each iteration:
   - `await asyncio.gather(*[self.runner.run(physical) for _ in samples])`
   - Judge each result → build one `Feedback` per sample
   - If not last iteration and planner exists: `physical = self.planner.revise(task_description, optimized, physical, feedbacks)`
6. Return `PipelineResult(task_description=..., sample_queries=..., executions=..., feedbacks=..., ...)`

### Sync wrapper

```python
def run_sync_with_samples(
    self,
    task_description: str,
    samples: list[tuple[str, str]],
    iterations: int = 2,
) -> PipelineResult:
    return self._run_async(self.run_with_samples(task_description, samples, iterations))
```

### Remove old methods

Remove: `run()`, `run_with_feedback()`, `run_sync()`, `run_sync_with_feedback()`

### `_save_logs()` update

Update to iterate over `result.executions` and `result.feedbacks` with `result.sample_queries`.

## 5. `tests/test_qasper.py`

Update all callsites from old API → new API:

| Old | New |
|-----|-----|
| `pipeline.run_sync(s.question)` | `pipeline.run_sync_with_samples("QA over scientific papers", [(s.question, "")], iterations=1)` |
| `pipeline.run_sync_with_feedback(query=s.question, gold_ans=s.answer, iterations=N)` | `pipeline.run_sync_with_samples("QA over scientific papers", [(s.question, s.answer)], iterations=N)` |
| `result.execution` | `result.execution` (via property — no change) |
| `result.feedback` | `result.feedback` (via property — no change) |

`run_batch()`: update inner call from `run_sync_with_feedback` → `run_sync_with_samples`.

`MockPlanningClient.complete()`: The SemanticParser branch returns a fixed expression and
doesn't inspect the query content — no change needed. The `"physical query planner"` check
continues to work.

> **Note on per-sample corpora (QASPER)**: Each QASPER sample has its own `InMemoryCorpus`.
> `run_batch` continues to instantiate one pipeline per sample (one corpus per pipeline).
> The new multi-sample API works within a single pipeline's shared corpus. A per-sample
> corpus extension (`list[tuple[str, str, corpus]]`) is a follow-on, not part of this refactor.

## Critical files

- [parser/prompts.py](parser/prompts.py)
- [parser/semantic_parser.py](parser/semantic_parser.py)
- [planner/llm_planner.py](planner/llm_planner.py)
- [pipeline.py](pipeline.py)
- [tests/test_qasper.py](tests/test_qasper.py)

## Verification

1. `pytest tests/test_qasper.py -x` — all mock tests pass (no API key needed)
2. Manually trace: `pipeline.run_sync_with_samples("QA over scientific papers", [(q, gold)], iterations=2)`
   - Check `result.task_description`, `result.sample_queries`, `result.feedbacks`
3. Verify `result.execution` and `result.feedback` properties return last-element values
4. Confirm `planner.revise()` receives `list[Feedback]` with correct per-sample blocks in prompt
