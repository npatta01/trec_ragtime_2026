# Topic-Union Maximum-Need Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce immutable, resumable topic-level candidate rankings by unioning all successful richer retrievals, deduplicating canonical document IDs, and assigning each document its maximum passage score across every extracted information need, without RRF or facet round-robin.

**Architecture:** A pure scoring module owns validated topic inputs, document union/deduplication, conservative one-time chunking, max-over-needs scoring, and canonical artifact encoding. A thin CLI reconstructs exact frozen-plan retrieval identities from immutable snapshots, opens the existing artifact store and passage scorer, and writes one owner-only checkpoint per topic plus a sanitized manifest. The first run is a bounded one-topic smoke followed by a 25-topic wave chosen by most successful retrieval queries while the independent two-worker retriever continues.

**Tech Stack:** Python 3.12, frozen dataclasses, `msgspec`, existing `RetrievalArtifactCache`, `PassageChunker`, `PassageScoreCache`, local/Flash passage scorer boundaries, `unittest`, SOPS-decrypted `.env`.

## Global Constraints

- Baseline C remains unchanged as the historical facet-RRF comparison arm.
- The richer arm performs no RRF, rank averaging, per-query quota, per-need quota, or facet round-robin selection.
- Deduplicate strictly by canonical `document_id`; conflicting canonical or English content fails the topic closed.
- Score every unique passage against every sealed information need and rank documents by the maximum pair score.
- Original retrieval rank and duplicate frequency are provenance only and cannot affect scoring or tie-breaking.
- Exact score ties resolve by canonical document ID.
- Every scoring wave consumes an immutable list of validated retrieval object identities.
- Credentials and plaintext environment values never enter cache identities, logs, manifests, or committed artifacts.
- The active two-worker retrieval owner process must not be stopped, duplicated, or modified.

---

### Task 1: Pure topic-union scoring contract

**Files:**
- Create: `src/trec_ragtime/topic_union_scoring.py`
- Create: `tests/test_topic_union_scoring.py`

**Interfaces:**
- Consumes: `InformationNeed`, `TopicRetrievalInput`, existing `RetrievalResult`, `LocalCrossEncoderConfig`, `PassageScorer`, `ArtifactStoreLike`, and representation name.
- Produces: `TopicScoreResult`, `deduplicate_topic_documents(inputs)`, `select_conservative_pair_length(config, needs, loader=None)`, and `score_topic_union(topic_id, needs, retrievals, *, scorer_config, scorer, store, representation)`.

- [ ] **Step 1: Write failing union and conflict tests**

Create fixtures with two retrievals whose document IDs overlap. Assert first-seen source order is preserved only as provenance, the unique pool contains one row per ID, and differing `text` or `english_text` for the same ID raises `TopicUnionScoringError`.

```python
def test_union_deduplicates_document_ids_and_keeps_all_sources():
    result = deduplicate_topic_documents((input_a, input_b))
    assert [row.document.document_id for row in result] == ["doc-a", "doc-shared", "doc-b"]
    assert [source.task_id for source in result[1].sources] == ["T1", "T2"]

def test_union_rejects_conflicting_duplicate_content():
    with self.assertRaisesRegex(TopicUnionScoringError, "conflicting duplicate"):
        deduplicate_topic_documents((input_a, conflicting_input))
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `uv run python -m unittest tests.test_topic_union_scoring -v`

Expected: import failure because `trec_ragtime.topic_union_scoring` does not exist.

- [ ] **Step 3: Implement strict input and union types**

Implement frozen validated dataclasses:

```python
@dataclass(frozen=True, slots=True)
class InformationNeed:
    need_id: str
    text: str

@dataclass(frozen=True, slots=True)
class RetrievalSource:
    task_id: str
    query: str
    object_key: str
    original_rank: int
    original_score: float

@dataclass(frozen=True, slots=True)
class TopicRetrievalInput:
    topic_id: str
    task_id: str
    query: str
    object_key: str
    result: RetrievalResult

@dataclass(frozen=True, slots=True)
class UnionDocument:
    document: RetrievedDocument
    sources: tuple[RetrievalSource, ...]
```

`deduplicate_topic_documents` must validate complete hydration, matching topic-independent retrieval query/object provenance, finite scores, and byte-identical canonical/English content for duplicate IDs. Sort output by document ID so retrieval order cannot leak into ties.

- [ ] **Step 4: Write failing max-over-needs and rank-independence tests**

Use a fake chunker and recording scorer. Assert every need receives every unique passage exactly once, the winning need/passage is retained, duplicate occurrence and organizer ranks do not change the result, and document-ID tie-breaking is deterministic.

```python
scores = {("N1", b"a"): 0.3, ("N2", b"a"): 0.9,
          ("N1", b"b"): 0.8, ("N2", b"b"): 0.1}
result = score_topic_union(
    "2000", needs, retrievals,
    scorer_config=fake_config,
    scorer=scorer_for(scores),
    store=fake_store,
    representation="report_text",
)
assert [(row.document_id, row.score, row.winning_need_id) for row in result.documents] == [
    ("doc-a", 0.9, "N2"), ("doc-b", 0.8, "N1")
]
```

- [ ] **Step 5: Implement conservative one-time chunking and scoring**

Build one `CrossEncoderPairLengthFunction` per need, select the one with the smallest positive `max_passage_length`, and create one `PassageChunker` from it. Chunk each unique document once. For every need, score the complete unique passage tuple through a query-specific `PassageScoreCache` that shares the selected chunker fingerprint. Validate complete finite coverage, reduce by maximum `(score, need_id, document_id, start, end, content_hash)`, and sort documents by `(-score, document_id)`.

The public result types must include:

```python
@dataclass(frozen=True, slots=True)
class ScoredTopicDocument:
    document_id: str
    score: float
    winning_need_id: str
    passage_start: int
    passage_end: int
    passage_content_hash: bytes
    sources: tuple[RetrievalSource, ...]

@dataclass(frozen=True, slots=True)
class TopicScoreResult:
    topic_id: str
    representation: str
    needs: tuple[InformationNeed, ...]
    input_object_keys: tuple[str, ...]
    unique_document_count: int
    passage_count: int
    documents: tuple[ScoredTopicDocument, ...]
    scorer_fingerprint: str
    chunker_fingerprint: str
```

- [ ] **Step 6: Run focused tests and verify GREEN**

Run: `uv run python -m unittest tests.test_topic_union_scoring -v`

Expected: all topic-union scoring tests pass.

- [ ] **Step 7: Commit the pure scoring contract**

```bash
git add src/trec_ragtime/topic_union_scoring.py tests/test_topic_union_scoring.py
git commit -m "feat: score deduplicated topic document unions"
```

---

### Task 2: Immutable snapshot loader and topic artifact cache

**Files:**
- Create: `src/trec_ragtime/topic_union_cache.py`
- Create: `tests/test_topic_union_cache.py`
- Create: `src/trec_ragtime/topic_union_snapshot.py`
- Create: `tests/test_topic_union_snapshot.py`

**Interfaces:**
- Consumes: frozen static plan JSONL, coverage index JSONL, full-run checkpoint directory, `RagtimeRetriever.get_cached_retrieval`, and Task 1 result types.
- Produces: `TopicScoreFingerprint`, `TopicScoreRequest`, `TopicScoreArtifactCache`, `TopicSnapshot`, `build_snapshot(...)`, and `load_snapshot_retrievals(...)`.

- [ ] **Step 1: Write failing cache isolation and canonical round-trip tests**

Assert fingerprint changes for scorer/chunker/representation/policy changes; request digest changes for any need text, plan hash, or ordered input object key; malformed or non-canonical MessagePack fails; and a stored result round-trips exactly.

- [ ] **Step 2: Implement the typed cache**

Use object keys:

```text
topic-scores/v1/<fingerprint>/<request-prefix>/<request-digest>.mpk
```

The fingerprint pins schema, scorer, chunker, representation, `max-need-passage-v1`, and `document-id-tie-v1`. The request pins topic ID, plan SHA-256, exact ordered needs, and sorted unique retrieval object keys. `put` validates result/request consistency, writes only complete results, evicts L1, and independently rereads through the typed decoder.

- [ ] **Step 3: Write failing snapshot tests**

Create temporary coverage pools and full-run checkpoints. Assert only validated successes enter the snapshot, breadth and current results are deduplicated by task ID/object key, error checkpoints are excluded, unknown tasks fail, and the snapshot digest is stable regardless of filesystem traversal order.

- [ ] **Step 4: Implement immutable snapshot construction**

Parse frozen plans with `static_plan_from_record`, map each task through `allocate_tasks(plan, len(plan.tasks))`, and reconstruct the exact `planner_static_v2` query context:

```python
{
    "kind": "planner_static_v2",
    "plan_sha256": plan.plan_sha256,
    "information_need_id": allocated.need_id,
    "task_id": allocated.task.task_id,
    "allocation_index": allocated.allocation_index,
}
```

Read breadth pool and full-run checkpoint metadata, validate plan/task/query/object-key lineage, and write a canonical owner-only snapshot JSON containing no document text or credentials. `load_snapshot_retrievals` calls `get_cached_retrieval` with the exact identity and fails closed on a miss or incomplete hydration; it never calls hosted retrieval.

- [ ] **Step 5: Run cache and snapshot tests**

Run: `uv run python -m unittest tests.test_topic_union_cache tests.test_topic_union_snapshot -v`

Expected: all tests pass.

- [ ] **Step 6: Commit cache and snapshot support**

```bash
git add src/trec_ragtime/topic_union_cache.py src/trec_ragtime/topic_union_snapshot.py tests/test_topic_union_cache.py tests/test_topic_union_snapshot.py
git commit -m "feat: snapshot richer topic retrieval inputs"
```

---

### Task 3: Resumable scoring CLI and sanitized manifest

**Files:**
- Create: `src/trec_ragtime/topic_union_cli.py`
- Create: `tests/test_topic_union_cli.py`
- Modify: `pyproject.toml`
- Modify: `src/trec_ragtime/__init__.py`

**Interfaces:**
- Consumes: Task 1 scoring API, Task 2 snapshot/cache API, `ArtifactStore.from_env`, and local/Flash scorer configuration patterns from `nuggets_cli._open_scorer`.
- Produces: console command `ragtime-topic-score` and owner-only `snapshot.json`, `progress.json`, and `manifest.json`.

- [ ] **Step 1: Write failing CLI selection and resume tests**

Assert `--topic-limit 25` selects topics by descending successful-query count then topic ID; explicit `--topic-id` is exact; `--dry-run` opens no scorer; existing validated topic artifacts become cache hits; failures are sanitized and do not stop other topics; and manifests contain counts/digests/object keys but no query text, document text, environment values, or credentials.

- [ ] **Step 2: Implement CLI parsing and dry-run snapshot**

Add:

```toml
ragtime-topic-score = "trec_ragtime.topic_union_cli:main"
```

Support `--plans`, `--coverage-index`, `--checkpoint-dir`, `--output`, repeated `--topic-id`, `--topic-limit`, `--representation report_text`, `--scorer local|flash`, `--device`, `--flash-endpoint-id`, `--flash-workers`, `--concurrency`, and `--dry-run`. Decrypt-first remains an operator precondition; CLI loads `.env` but never prints it.

- [ ] **Step 3: Implement resumable topic execution**

For each selected topic, load snapshot retrievals cache-only, return a typed topic-score cache hit when present, otherwise score and put. Write progress atomically after every topic. One worker owns one topic; concurrent workers may not score the same need query in separate processes.

- [ ] **Step 4: Run focused and regression tests**

Run:

```bash
uv run python -m unittest tests.test_topic_union_cli -v
uv run python -m unittest tests.test_topic_union_scoring tests.test_topic_union_cache tests.test_topic_union_snapshot tests.test_topic_union_cli tests.test_passage_scoring tests.test_passage_reranking_integration -v
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit CLI integration**

```bash
git add pyproject.toml src/trec_ragtime/__init__.py src/trec_ragtime/topic_union_cli.py tests/test_topic_union_cli.py
git commit -m "feat: add resumable topic-union scoring command"
```

---

### Task 4: Verify and launch the first parallel scoring wave

**Files:**
- Modify: `docs/implementation/2026-08-12-planner-static-v1-evidence.md`
- Modify: `docs/implementation/2026-08-12-planner-static-v1-plan.md`
- Create outside worktree: `<git-common-dir>/ragtime-cache/experiments/planner-static-final-v1/topic-scoring/<wave-id>/`

**Interfaces:**
- Consumes: `ragtime-topic-score`, active retrieval progress, local ROCm or configured Flash scorer.
- Produces: validated one-topic smoke, then a resumable 25-topic scoring wave and evidence-backed throughput/cost status.

- [ ] **Step 1: Verify prerequisites without exposing secrets**

Run:

```bash
sops decrypt --output .env config/secrets.sops.env
git check-ignore .env
uv run ragtime-corpus status
python -c 'import os; from dotenv import load_dotenv; load_dotenv(); print({k: bool(os.getenv(k)) for k in ("RUNPOD_API_KEY",)})'
```

Load `.env` before the final environment-presence check and print booleans only. On the Framework host, inspect ROCm/PyTorch compatibility and available memory before selecting local scoring. Do not pull a model or create infrastructure unless the selected existing scorer requires it and the user has already authorized that in scope.

- [ ] **Step 2: Create and inspect a dry-run snapshot**

Run `ragtime-topic-score --dry-run --topic-limit 25` into a new owner-only wave directory. Verify the manifest selects the 25 topics with the most validated successful queries, includes all breadth inputs, excludes failed checkpoints, and contains no sensitive payloads.

- [ ] **Step 3: Run a one-topic end-to-end smoke**

Choose the selected topic with the fewest unique documents. Run scoring for that topic using the available scorer, then independently reread its topic-score artifact. Record unique documents, passages, needs, pair count, cache hits/misses, elapsed time, and estimated 25-topic duration.

- [ ] **Step 4: Launch or stop based on the smoke gate**

If the smoke produces complete finite scores, a validated durable artifact, and a practical deadline estimate, launch the remaining selected topics in a separate named tmux session with bounded concurrency. If hardware/model/provider prerequisites are absent or the estimate misses the deadline, leave the retriever running, record the verified blocker, and do not create external infrastructure or silently substitute another scoring policy.

- [ ] **Step 5: Verify independent processes and update evidence**

Confirm the two-worker retriever remains alive and advancing, the scoring process owns a different lock/output directory, and both progress files are readable. Record exact commands, counts, timing, cache behavior, and any failures in the canonical plan/evidence log.

- [ ] **Step 6: Run final verification**

Run:

```bash
uv run python -m unittest tests.test_topic_union_scoring tests.test_topic_union_cache tests.test_topic_union_snapshot tests.test_topic_union_cli -v
git diff --check
```

Expected: all new tests pass and no whitespace errors are reported.

- [ ] **Step 7: Commit evidence updates**

```bash
git add docs/implementation/2026-08-12-planner-static-v1-plan.md docs/implementation/2026-08-12-planner-static-v1-evidence.md
git commit -m "docs: record first topic-union scoring wave"
```
