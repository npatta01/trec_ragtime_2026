# Control-A Retrieval Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested batch runner that retrieves every checked-in RAGTIME topic from the organizer at a user-facing `top_k=100` depth and writes a durable, self-contained Control-A artifact.

**Architecture:** Add one focused `topic_candidates` module containing topic parsing, deterministic query construction, artifact serialization, and the batch runner. The module will call the existing `RagtimeRetriever` without modifying its API; the current organizer pipeline name is fixed as `ragtime2`, while the public `top_k` is passed to the retriever for client-side truncation of its 100-result response. The CLI will default to the shared project cache and require a unique run ID, while unit tests inject a fake retriever and use temporary directories.

**Tech Stack:** Python 3.11+, standard-library `dataclasses`, `argparse`, `hashlib`, `json`, `pathlib`, `tempfile`, `subprocess`, `datetime`; existing `RagtimeRetriever`, `RetrievalResult`, and `default_project_cache_dir()`; `unittest`.

## Global Constraints

- Decrypt credentials before running tests or application code: `sops decrypt --output .env config/secrets.sops.env`.
- Use `RagtimeRetriever.from_env()` and the complete shared local DuckDB corpus for live runs.
- Use the verified organizer pipeline name `ragtime2`; do not add a `%100` suffix.
- Pass the public `top_k` into the retriever call; Control-A defaults to `top_k=100`.
- Set `max_content_fallbacks=0`; unresolved IDs must remain explicit in the artifact.
- Preserve canonical `document.text`; use `document.report_text` only as the English report representation.
- Do not send or save bearer tokens; do not create a second corpus or vector index.
- Write artifacts below the shared project cache, not in the tracked repository.
- Do not run the full live 103-topic batch as part of unit-test verification.

---

### Task 1: Define and test topic parsing and query construction

**Files:**
- Create: `tests/test_topic_candidates.py`
- Create: `src/trec_ragtime/topic_candidates.py`

**Interfaces:**
- Produces `Topic`, `load_topics(path)`, `build_topic_query(topic)`, `pipeline_for_top_k(top_k)`, and `query_hash(query)` for the later runner tasks.

- [x] **Step 1: Write the failing tests**

Add `unittest` cases covering:

```python
def test_load_topics_preserves_source_order_and_validates_fields(self):
    topics = load_topics(topic_file)
    self.assertEqual([topic.topic_id for topic in topics], ["2000", "2001"])
    self.assertEqual(topics[0].limit, 5000)

def test_build_topic_query_uses_problem_statement_only(self):
    topic = Topic("1", "ragtime/2/all", "problem", "background", 5000)
    self.assertEqual(build_topic_query(topic), "problem")

def test_pipeline_for_top_k_translates_depth_to_organizer_protocol(self):
    self.assertEqual(pipeline_for_top_k(100), "ragtime2")
    with self.assertRaises(ValueError):
        pipeline_for_top_k(0)
```

Also test that malformed JSON, missing required fields, duplicate topic IDs, non-positive limits, and blank text fail with `ValueError` naming the source line or field.

- [x] **Step 2: Run the focused tests and verify the expected failure**

Run:

```bash
uv run python -m unittest tests.test_topic_candidates -v
```

Expected: import failure because `trec_ragtime.topic_candidates` does not yet exist.

- [x] **Step 3: Implement the minimal pure helpers**

Define:

```python
@dataclass(frozen=True, slots=True)
class Topic:
    topic_id: str
    collection_id: str
    problem_statement: str
    background: str
    limit: int

def load_topics(path: Path) -> tuple[Topic, ...]: ...
def build_topic_query(topic: Topic) -> str: ...
def pipeline_for_top_k(top_k: int) -> str: ...
def query_hash(query: str) -> str: ...
```

Read UTF-8 JSONL line-by-line, reject blank lines and unexpected/missing fields, preserve file order, and hash the exact UTF-8 query with SHA-256. Keep `limit` as metadata and exclude it from the query.

- [x] **Step 4: Run the focused tests and verify they pass**

Run the same focused command. Expected: all parsing and query-construction tests pass.

- [x] **Step 5: Commit the pure topic layer**

```bash
git add src/trec_ragtime/topic_candidates.py tests/test_topic_candidates.py
git commit -m "feat: add Control-A topic query helpers"
```

### Task 2: Add deterministic artifact serialization

**Files:**
- Modify: `tests/test_topic_candidates.py`
- Modify: `src/trec_ragtime/topic_candidates.py`

**Interfaces:**
- Produces `write_control_a_artifact(run_dir, manifest, records)` and a JSON-safe candidate-record helper used by the batch runner.

- [x] **Step 1: Write the failing serialization tests**

Test that the writer creates `manifest.json` and `candidates.jsonl`, emits UTF-8 JSON with stable key ordering, preserves canonical text and `report_text`, records unresolved/translation status, and refuses to write into an existing run directory. Test that a temporary file is not left as the completed artifact after a successful write.

- [x] **Step 2: Run the serialization tests and verify they fail**

Run:

```bash
uv run python -m unittest tests.test_topic_candidates.ControlAArtifactTests -v
```

Expected: failure because the writer helpers do not exist.

- [x] **Step 3: Implement atomic, no-overwrite serialization**

Implement:

```python
def write_control_a_artifact(
    run_dir: Path,
    *,
    manifest: Mapping[str, object],
    records: Iterable[Mapping[str, object]],
) -> None: ...
```

Create the run directory exactly once, write each file through a same-directory temporary file, flush and `os.fsync()` it, then publish with `os.replace()`. Reject an existing `run_dir` before writing. Serialize each candidate on one JSONL line with `ensure_ascii=False`, sorted keys, and newline termination.

- [x] **Step 4: Run the serialization tests and verify they pass**

Run the focused serialization command, then rerun `uv run python -m unittest tests.test_topic_candidates -v`.

- [x] **Step 5: Commit the artifact layer**

```bash
git add src/trec_ragtime/topic_candidates.py tests/test_topic_candidates.py
git commit -m "feat: write durable Control-A artifacts"
```

### Task 3: Implement the fake-retriever-tested batch runner

**Files:**
- Modify: `tests/test_topic_candidates.py`
- Modify: `src/trec_ragtime/topic_candidates.py`

**Interfaces:**
- Consumes `Topic`, `build_topic_query`, `pipeline_for_top_k`, and `write_control_a_artifact`.
- Produces:

```python
def run_control_a(
    topics_path: Path,
    *,
    retriever: RetrieverLike,
    output_root: Path,
    run_id: str,
    top_k: int = 100,
    git_revision: str | None = None,
    created_at: str | None = None,
) -> Path: ...
```

Define the internal `RetrieverLike` protocol with `cache_path`, `corpus_path`,
`corpus`, and a `retrieve(query, *, pipeline, top_k, max_content_fallbacks,
fallback_on_hf_error)` method returning `RetrievalResult`, so the fake test
retriever and the production `RagtimeRetriever` share the same checked
boundary.

- [x] **Step 1: Write the failing runner tests**

Create a fake retriever that records calls and returns real `RetrievalResult` objects assembled from `PipelineResult` and `RetrievedDocument`. Assert that two topics produce two calls with:

```python
{
    "pipeline": "ragtime2",
    "top_k": 100,
    "max_content_fallbacks": 0,
}
```

Assert that the resulting JSONL is ordered by topic source order and organizer rank, includes query hashes and full text, and explicitly records one unresolved document and one missing translation. Assert that the manifest records `requested_top_k=100`, `pipeline="ragtime2"`, topic count, candidate count, source hash, supplied Git revision, and corpus metadata without any token-like field.

- [x] **Step 2: Run the runner tests and verify they fail**

Run:

```bash
uv run python -m unittest tests.test_topic_candidates.ControlARunnerTests -v
```

Expected: failure because `run_control_a` is not implemented.

- [x] **Step 3: Implement the minimal batch runner**

For every loaded topic, call `retriever.retrieve()` once with the problem-only
query, `pipeline="ragtime2"`, the requested `top_k`, `max_content_fallbacks=0`,
and no fallback-on-Hugging-Face errors. Convert each `RetrievedDocument` with
`as_record(include_text=True)`, add topic ID, collection ID, query, query hash,
topic limit, `hydrated`, `report_ready`, and `missing_translation` fields, and
preserve the result order. Build a manifest with schema version, run ID, UTC
creation time, topic source path/hash, Git revision, query-construction
version, requested depth, pipeline, retrieval options, API cache path, corpus
path/revision, counts, and per-topic unresolved/translation IDs. The topic
background remains available to the later `P+F(B)` decomposition arm and is not
included verbatim in Control-A.

Use the supplied `created_at` and `git_revision` in tests; when omitted, use UTC now and `git rev-parse HEAD` from the topic repository. Do not include exception strings that could contain credentials.

- [x] **Step 4: Run the runner tests and verify they pass**

Run the focused runner tests, then all module tests:

```bash
uv run python -m unittest tests.test_topic_candidates -v
```

- [x] **Step 5: Commit the batch runner**

```bash
git add src/trec_ragtime/topic_candidates.py tests/test_topic_candidates.py
git commit -m "feat: add Control-A batch runner"
```

### Task 4: Add the explicit CLI entry point and repository verification

**Files:**
- Modify: `src/trec_ragtime/topic_candidates.py`
- Modify: `tests/test_topic_candidates.py`

**Interfaces:**
- Adds `main(argv: Sequence[str] | None = None) -> int`, runnable with `uv run python -m trec_ragtime.topic_candidates`.

- [x] **Step 1: Write failing CLI tests**

Test `--help`, required/validated `--run-id`, default `--top-k 100`, custom `--topics`, and custom `--output-root` argument parsing without constructing a live retriever. Test that `--top-k 0` fails through argparse.

- [x] **Step 2: Run the CLI tests and verify they fail**

Run the focused CLI test. Expected: failure because `main` and module execution are not implemented.

- [x] **Step 3: Implement the CLI**

Add arguments for `--topics` (default `topics/topics.all.2026.v0625.jsonl`), `--top-k` (default `100`), `--run-id` (required), and `--output-root` (default `default_project_cache_dir() / "experiments" / "control-a"`). Require the operator to run the repository's decrypt-first command before invoking the CLI, then construct `RagtimeRetriever.from_env()`. Run the batch in a context manager, print only the artifact path and non-secret counts, and return zero on success.

- [x] **Step 4: Run focused, full, and smoke verification**

First run:

```bash
sops decrypt --output .env config/secrets.sops.env
uv run python -m unittest tests.test_topic_candidates -v
uv run python -m unittest discover -s tests -v
```

Then run a no-network smoke test through the fake retriever already covered by the suite. Do not execute the full live batch unless separately requested; the CLI itself is the explicit live operation.

- [x] **Step 5: Commit the CLI and verification changes**

```bash
git add src/trec_ragtime/topic_candidates.py tests/test_topic_candidates.py
git commit -m "feat: expose Control-A runner CLI"
```

## Self-review checklist

- The design's `top_k=100` public contract is covered by Tasks 1, 3, and 4; only the organizer transport string is internal.
- The existing retriever is reused without changes, with local hydration and zero private-content fallback covered by Task 3.
- Canonical text, optional English report text, unresolved IDs, and missing translations are covered by Task 2 and Task 3.
- Shared-cache placement, no-overwrite behavior, atomic writes, provenance, and secret exclusion are covered by Task 2 and Task 3.
- Query decomposition, reranking, and out-of-pool retrieval are explicitly out of scope and receive only a stable artifact handoff.
- No placeholders or unspecified implementation decisions remain in the task steps.
