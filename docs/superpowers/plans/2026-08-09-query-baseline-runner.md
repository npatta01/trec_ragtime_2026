# Query Baseline Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a resumable Baseline B/C retrieval runner using the existing query plans and shared cache.

**Architecture:** A focused module validates plans, checkpoints exact organizer requests, derives topic candidates, hydrates final IDs locally, and atomically publishes manifests. The CLI selects B or C and resumes an immutable shared run directory.

**Tech Stack:** Python 3.13, `trec_ragtime.RagtimeRetriever`, JSONL, `unittest`.

## Global Constraints

- Use only existing `single_query` and `facet_queries` values.
- Never add `title` or `collection_id` to retrieval queries.
- Use `pipeline="ragtime2"` and `top_k=100`; never use `ragtime2%100`.
- Use RRF `k=60` for Baseline C.
- Checkpoint every query and preserve every failure explicitly.
- Store artifacts under the shared git-common-dir cache.
- Never print or persist credentials.
- Use at most four independent retriever instances and one artifact-writer
  thread; fall back to two workers if the live load test or run shows timeouts.

---

### Task 1: Resumable baseline runner

**Files:**
- Create: `src/trec_ragtime/query_baselines.py`
- Create: `tests/test_query_baselines.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: `RagtimeRetriever.retrieve(query, pipeline="ragtime2", top_k=100, max_content_fallbacks=0, fallback_on_hf_error=False)`.
- Produces: `run_query_baseline(...) -> Path`, `build_parser()`, and `main()`.

- [x] **Step 1: Write failing tests**

Cover literal plan loading, B organizer ordering, C `sum(1/(60+rank))`
fusion, success/partial/error states, resume of successful query checkpoints,
retry of failed checkpoints, atomic latest-record JSONL replacement, and CLI
defaults (`ragtime2`, `top_k=100`, shared output root).

- [x] **Step 2: Verify the tests fail for the missing module**

Run: `uv run python -m unittest tests.test_query_baselines -v`

Expected: import failure for `trec_ragtime.query_baselines`.

- [x] **Step 3: Implement the minimal runner**

Create typed plan/checkpoint helpers, deterministic B/C candidate builders,
atomic JSON/JSONL writers, resume loading keyed by topic/query index, local
hydration mapping, manifest counts, and the `b|c` CLI. Keep the module focused;
reuse existing client, retriever, path, and model classes.

- [x] **Step 4: Verify targeted and full tests**

Run:

```bash
uv run python -m unittest tests.test_query_baselines -v
uv run python -m unittest discover -s tests -v
git diff --check
```

Expected: all tests pass and `git diff --check` exits zero.

- [x] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-08-09-query-baseline-runner-design.md \
  docs/superpowers/plans/2026-08-09-query-baseline-runner.md \
  src/trec_ragtime/query_baselines.py tests/test_query_baselines.py pyproject.toml
git commit -m "feat: add resumable query baseline runner"
```

### Task 1B: Bounded concurrent retrieval

**Files:**
- Modify: `src/trec_ragtime/query_baselines.py`
- Modify: `tests/test_query_baselines.py`
- Modify: `docs/superpowers/specs/2026-08-09-query-baseline-runner-design.md`

**Interfaces:**
- Consumes: `run_query_baseline(..., retrievers: Sequence[RetrieverLike], concurrency: int)`.
- Produces: immediate deterministic checkpoints from at most four in-flight
  organizer requests; CLI option `--concurrency {1,2,4}` with default `1`.

- [x] **Step 1: Write failing scheduler tests**

Cover a measured maximum of four overlapping retrieval calls, one in-flight
call per retriever instance, deterministic plan-ordered JSONL after
out-of-order completion, immediate checkpoint writes before topic hydration,
rebuilding stale topic records, stopping after the first failed concurrent
wave, exact resume of the existing sequential B artifact, and parser rejection
above four.

- [x] **Step 2: Verify the focused tests fail**

Run: `uv run python -m unittest tests.test_query_baselines -v`

Expected: failures because `run_query_baseline` and the CLI do not expose
bounded concurrency.

- [x] **Step 3: Implement the bounded wave scheduler**

Use `ThreadPoolExecutor` with one independent retriever per slot. Submit one
bounded wave at a time, process futures in the main thread, and atomically write
each completed query checkpoint before topic hydration. Stop before the next
wave if any query fails. Only the main thread may mutate checkpoint/topic
mappings or artifact files.

- [x] **Step 4: Verify focused and complete offline suites**

Run:

```bash
uv run python -m unittest tests.test_query_baselines -v
uv run python -m unittest discover -s tests -v
git diff --check
```

Expected: all tests pass and `git diff --check` exits zero.

- [x] **Step 5: Commit**

```bash
git add src/trec_ragtime/query_baselines.py tests/test_query_baselines.py \
  docs/superpowers/specs/2026-08-09-query-baseline-runner-design.md \
  docs/superpowers/plans/2026-08-09-query-baseline-runner.md
git commit -m "feat: add bounded baseline retrieval concurrency"
```

---

### Task 2: Execute and validate B/C

**Files:**
- No source changes.
- Create runtime artifacts only under `<git-common-dir>/ragtime-cache/experiments/query-baselines/`.

**Interfaces:**
- Consumes: Task 1B CLI.
- Produces: complete B and C run directories.

- [x] **Step 1: Decrypt and verify the live endpoint**

Run one short `ragtime2` query with `retries=0`; require 100 hits before the
batch.

- [x] **Step 2: Run Baseline B**

Run the CLI with baseline `b`, the existing query-plan path, and a unique run
ID. Resume the same run ID after interruption.

- [x] **Step 3: Validate Baseline B**

Require 103 topic records; reconcile manifest success/error and candidate
counts against JSONL records.

- [x] **Step 4: Run Baseline C**

Run the CLI with baseline `c`, the same plan path, and a unique run ID. Resume
the same run ID after interruption.

- [x] **Step 5: Validate Baseline C and compare A/B/C**

Require 103 topic records; recompute RRF for sampled topics; reconcile all
manifest counts; report every failed query/topic and artifact path.

**Completed 2026-08-09:**

- Baseline B: 103/103 successful queries and topics, 10,300 candidates, no
  unresolved documents or missing translations.
- Baseline C: 752/752 successful facet queries and 103/103 successful topics,
  10,300 candidates, no unresolved documents or missing translations.
- Recomputed RRF over all 103 Baseline C topics and verified exact top-100
  ordering, scores, ranks, source-query provenance, and manifest counts.
- Baseline A remains an explicit partial baseline: 90/103 successful topics,
  13 recorded endpoint failures, and 9,000 candidates.
