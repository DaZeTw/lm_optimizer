# Fix Plan: Make Iterations Equal Rounds

## Summary

Unify terminology so `--iterations 10` means exactly 10 rounds total. Round 1 uses the initial TST, and rounds 2-10 use revised TSTs. Remove the extra baseline-vs-revision wrapper in `tests/test_casesum.py`.

## Key Changes

- Keep `pipeline.py` execution behavior unchanged:
  - `for iteration in range(iterations)` remains the core round loop.
  - `physical = self._build_physical(grounding_query, logical, tst)` remains unchanged.
- Change `tests/test_casesum.py` so it calls `run_batch(samples, iterations=args.iterations, ...)` only once.
- Remove the separate baseline call:
  - no extra `run_batch(..., iterations=1)` before the main run.
  - no “Round 2 — With TST revision” wrapper.
- Rename output language:
  - use “round” in user-facing text.
  - keep the CLI argument `--iterations` for compatibility, but describe it as number of rounds.
- Save logs like the current `round1_*` pattern, but per actual round:
  - `round1_shared_plan.json`
  - `round1_node_feedbacks.json`
  - `round2_shared_plan.json`
  - `round2_node_feedbacks.json`
  - ...
  - `round10_shared_plan.json`
  - `round10_node_feedbacks.json`
- Each round log contains all samples for that round only.

## Implementation Details

- In `pipeline.py`, split combined `log_entries` by `iteration`.
- Write one shared-plan file per iteration using one-based round numbering:
  - iteration `0` becomes `round1_shared_plan.json`
  - iteration `9` becomes `round10_shared_plan.json`
- Split `node_feedback_entries` the same way:
  - iteration `0` becomes `round1_node_feedbacks.json`
  - iteration `9` becomes `round10_node_feedbacks.json`
- Keep the final `PipelineResult` shape unchanged for now; it can still return final-round executions and feedbacks.
- In `tests/test_casesum.py`, print one final summary from the final round result. Do not print baseline/revision comparison sections.

## Test Plan

- Run:
  `python tests/test_casesum.py --n 10 --iterations 10 --log-dir logs_scifact/`
- Confirm the console no longer shows separate “Round 1” and “Round 2” experiment sections.
- Confirm the run produces exactly 10 round log pairs:
  - `round1_shared_plan.json` through `round10_shared_plan.json`
  - `round1_node_feedbacks.json` through `round10_node_feedbacks.json`
- Confirm each round log contains 10 sample records when `--n 10`.
- Confirm `round1_shared_plan.json` has `iteration: 0`.
- Confirm `round10_shared_plan.json` has `iteration: 9`.
- Confirm `pipeline.py` still passes `logical` into `_build_physical`.

## Assumptions

- “Round” and “iteration” should mean the same thing.
- Round 1 is the initial plan.
- Rounds 2 through N are revision rounds.
- The old baseline-vs-revision comparison run is no longer wanted.
