# Retrieval Artifact Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cache one complete hydrated `RetrievalResult` per exact semantic query through the issue #12 `ArtifactStore`, integrate deterministic references into Control-A/B/C, and import existing successful run artifacts without participant API calls.

**Architecture:** `RetrievalArtifactCache` defines strict identities and wire models. `RagtimeRetriever.retrieve()` performs cache-aside lookup/compute/write. Runners retain one-query ownership and main-thread checkpoint writes, then assemble topics from immutable hydrated query objects rather than rehydrating through DuckDB.

**Tech Stack:** Python 3.11+, issue #12 `ArtifactStore`, `msgspec`, SHA-256, read-only DuckDB `LocalCorpusIndex`, `unittest`, `uv`.

## Global Constraints

- Begin only after issue #12 passes its full offline and bounded storage verification gates.
- Implement GitHub issue #13 exactly using the issue #12 `ArtifactStore` interface and failure semantics.
- One immutable `.mpk` object represents one exact complete hydrated retrieval request.
- Cache only results with deterministic complete ranking, no operational hydration failures, no unresolved base document IDs, and validated canonical source text.
- Missing English translations remain explicit valid status only when canonical source hydration succeeded.
- Any result-affecting request, fallback, client/retriever behavior, query-context, or corpus change must change the identity; machine path, credentials, run ID, worker ID, timestamp, concurrency, and refresh must not.
- L1/L2 hits perform no organizer request and no corpus hydration lookup.
- Keep Control-A/B/C ordering, rankings, RRF `k=60`, bounded concurrency, failure stopping, deterministic JSONL, and main-thread-only writes unchanged.
- Checkpoints contain lightweight immutable references, not hydrated document text or whole mutable run uploads.
- Existing-run importers never call organizer, `/content`, or Dataset Viewer and never modify the source runs.
- Follow red-green-refactor and record RED/GREEN evidence in task reports.

---

### Task 1: Strict retrieval identities, wire models, and cache adapter

**Files:**
- Create: `src/trec_ragtime/retrieval_cache.py`
- Create: `tests/test_retrieval_cache.py`
- Modify: `src/trec_ragtime/models.py`
- Modify: `src/trec_ragtime/__init__.py`

**Interfaces:**
- Produces: `RetrievalRequestIdentity`, `RetrievalArtifactReference`, `RetrievalArtifactCache`, strict wire structs, behavior-version constants, `retriever_fingerprint(...)`, and `retrieval_object_key(...)`.
- `RetrievalArtifactCache.get(request: RetrievalRequestIdentity) -> RetrievalResult | None`.
- `RetrievalArtifactCache.put(request: RetrievalRequestIdentity, result: RetrievalResult) -> RetrievalResult`.
- Produces domain provenance fields sufficient for runners to record object key, binary request digest/fingerprint as hex in JSON, stored timestamp, and cache-hit status.

- [ ] **Step 1: Write failing identity and strict round-trip tests**

Use hand-derived fixtures and cover exact query UTF-8 bytes, normalized base URL, pipeline, `top_k`/`None`, endpoint-depth version, both fallback settings, content collection, backend policy, query context, all corpus identity fields except path, and client/retriever behavior versions. Assert path/concurrency/refresh/token changes do not change keys and each result-affecting mutation does.

Add strict round-trip tests for every nested field and rejection tests for wrong schema/request/fingerprint, truncated bytes, unknown fields, non-finite scores/times, duplicate/noncontiguous ranks, duplicate IDs, hit/document misalignment, malformed translation status, unresolved IDs, hydration failures, and hidden unhydrated documents.

- [ ] **Step 2: Verify RED**

Run `uv run python -m unittest tests.test_retrieval_cache -v`; require missing-module/interface failures.

- [ ] **Step 3: Implement canonical identity and wire conversion**

Strip only the machine-local `path` from `RagtimeRetriever.corpus_identity`; retain all immutable dataset/revision/schema/split/count provenance. Hash strict canonical MessagePack with SHA-256 and use layout `retrieval/v1/<fingerprint>/<prefix>/<request-hash>.mpk`. Encode source and English text exactly. `get` uses `ArtifactStore.read_validated`; `put` validates and writes once without rereading.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run python -m unittest tests.test_retrieval_cache -v
uv run python -m unittest tests.test_artifact_store tests.test_response_cache -v
git diff --check
```

- [ ] **Step 5: Commit**

```bash
git add src/trec_ragtime/retrieval_cache.py src/trec_ragtime/models.py src/trec_ragtime/__init__.py tests/test_retrieval_cache.py
git commit -m "feat: add typed retrieval artifact cache"
```

### Task 2: Cache-aside RagtimeRetriever integration

**Files:**
- Modify: `src/trec_ragtime/retriever.py`
- Modify: `tests/test_huggingface.py`
- Modify: `tests/test_retrieval_cache.py`

**Interfaces:**
- `RagtimeRetriever(..., retrieval_cache: RetrievalArtifactCache | None = None)`.
- `RagtimeRetriever.from_env()` shares the client's configured `ArtifactStore`.
- `RagtimeRetriever.retrieve(..., refresh: bool = False, query_context: Mapping[str, object] | None = None) -> RetrievalResult`.
- Produces a pure identity-builder/cache-only lookup seam used by runners and importers.

- [ ] **Step 1: Write failing cache-aside behavior tests**

```python
def test_l1_hit_does_not_call_search_or_corpus(): ...
def test_l2_hit_does_not_call_search_or_corpus_and_populates_l1(): ...
def test_miss_computes_once_validates_writes_and_returns_without_reread(): ...
def test_unresolved_or_hydration_failure_result_is_never_written(): ...
def test_missing_translation_with_source_text_is_cacheable(): ...
def test_refresh_bypasses_read_and_overwrites_only_after_complete_success(): ...
def test_failed_refresh_preserves_existing_retrieval_object(): ...
def test_query_context_changes_identity_without_changing_query_bytes(): ...
```

Use fakes that raise if organizer or corpus access occurs on a hit; do not assert only on mock call existence.

- [ ] **Step 2: Verify RED**

Run `uv run python -m unittest tests.test_retrieval_cache tests.test_huggingface -v` and require failures on absent retriever integration.

- [ ] **Step 3: Implement cache-aside retrieval**

Validate arguments and build identity before work. Read unless any refresh control is true. On miss run the existing body unchanged, validate completeness, then `put`. Add `refresh=True` as the combined explicit refresh while preserving current `refresh_search` and `refresh_documents`. Do not catch artifact-store failures. Provide cache-only reconstruction by exact identity so runner topic assembly cannot fall through to organizer retrieval.

- [ ] **Step 4: Verify GREEN and the full retriever suite**

Run:

```bash
uv run python -m unittest tests.test_retrieval_cache tests.test_huggingface tests.test_local_corpus -v
git diff --check
```

- [ ] **Step 5: Commit**

```bash
git add src/trec_ragtime/retriever.py tests/test_huggingface.py tests/test_retrieval_cache.py
git commit -m "feat: cache complete hydrated retrievals"
```

### Task 3: Control-A/B/C references and cache-only topic reconstruction

**Files:**
- Modify: `src/trec_ragtime/topic_candidates.py`
- Modify: `src/trec_ragtime/query_baselines.py`
- Modify: `tests/test_topic_candidates.py`
- Modify: `tests/test_query_baselines.py`

**Interfaces:**
- Control-A passes query context `control_a/problem_only_v1` and records one `RetrievalArtifactReference` per topic status.
- B/C query tasks pass baseline kind, construction version, query-plan SHA-256, and query index as context.
- Query checkpoints add a lightweight `retrieval_artifact` object; topic publication reconstructs exact identities and performs cache-only loads.

- [ ] **Step 1: Write failing runner integration tests**

Cover unchanged Control-A candidate order and manifest counts, unchanged Baseline B top-100 order/score, unchanged Baseline C RRF/ties/provenance, deterministic checkpoint order after out-of-order worker completion, reference persistence, resume, and stopped-wave behavior. Add a warm B/C topic test whose retriever raises on `retrieve`, `enrich_documents`, organizer, or DuckDB but serves cache-only artifacts. Verify same-ID text disagreement across C facets is fatal/explicit rather than silently selected.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run python -m unittest tests.test_topic_candidates tests.test_query_baselines -v
```

Expected: missing query-context/reference/cache-only reconstruction behavior.

- [ ] **Step 3: Integrate without changing scheduler or ranking**

Pass deterministic query contexts into each worker retrieval. Record object references only after successful retrieval artifact write. Replace `_hydrate_candidates` in topic publication with cache-only exact retrieval loads and a document map validated across facet results. Keep `ThreadPoolExecutor` bounds `{1,2,4}`, one distinct retriever per worker, main-thread-only checkpoint writes, wave stop policy, plan-order JSONL, B ranking, and C RRF `60` exactly.

- [ ] **Step 4: Verify focused and full offline suites**

Run:

```bash
uv run python -m unittest tests.test_topic_candidates tests.test_query_baselines tests.test_retrieval_cache -v
uv run python -m unittest discover -s tests -v
git diff --check
```

- [ ] **Step 5: Commit**

```bash
git add src/trec_ragtime/topic_candidates.py src/trec_ragtime/query_baselines.py tests/test_topic_candidates.py tests/test_query_baselines.py
git commit -m "feat: integrate retrieval artifacts with control runners"
```

### Task 4: Existing Control-A/B/C importer and documentation

**Files:**
- Create: `src/trec_ragtime/retrieval_import.py`
- Create: `tests/test_retrieval_import.py`
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: relevant notebooks only where retrieval-cache behavior is described

**Interfaces:**
- Adds CLI: `ragtime-retrieval-import = "trec_ragtime.retrieval_import:main"`.
- Imports `control_a_v1` directories and `query_baseline_v1` B/C directories through a pinned `LocalCorpusIndex` plus `RetrievalArtifactCache`.
- Writes an atomic owner-only sanitized migration manifest to an explicit local path.

- [ ] **Step 1: Write failing importer tests with real local corpus fixtures**

Cover source file hashes, supported schema/config checks, exact query-context reconstruction, rank/document validation, Control-A grouping, B/C query checkpoint hydration through local index only, C facet hits absent from fused topic output, skip of failed/partial records, no participant/HF Dataset Viewer calls, idempotency, independent artifact reread, byte preservation, and manifest credential/body exclusion.

- [ ] **Step 2: Verify RED**

Run `uv run python -m unittest tests.test_retrieval_import -v`; require missing importer behavior.

- [ ] **Step 3: Implement importer and user documentation**

Read manifests/JSONL incrementally, validate immutable input hashes computed during import, and hydrate only exact hit IDs through `LocalCorpusIndex.enrich_documents`. Construct conservative `PipelineResult` metadata where legacy files lack optional organizer metadata; never invent success for failed checkpoints. Write every complete query through the production cache and independently validate it. Update docs with object layout, runner reuse, importer commands, and the rule that HF query objects—not mutable JSONL—are the durable shared cache.

- [ ] **Step 4: Verify complete offline and source-policy checks**

Run:

```bash
sops decrypt --output .env config/secrets.sops.env
uv run python -m unittest tests.test_retrieval_import -v
uv run python -m unittest discover -s tests -v
rg -n "enrich_documents|retrieve\(" src/trec_ragtime/query_baselines.py src/trec_ragtime/topic_candidates.py
git diff --check
```

Inspect the runner call sites to confirm topic reconstruction uses cache-only exact artifacts and importer call sites cannot reach participant transports.

- [ ] **Step 5: Commit**

```bash
git add src/trec_ragtime/retrieval_import.py tests/test_retrieval_import.py pyproject.toml README.md AGENTS.md notebooks
git commit -m "feat: import and reuse control retrieval artifacts"
```

---

## Issue #13 completion gate

Run a whole-branch independent review against the foundation merge base. Fix
all load-bearing findings, then freshly decrypt and run the entire offline
suite, `git diff --check`, credential scans, and CLI help smoke tests. For
bounded integration, use one disposable exact query to prove cold write, L2
read with organizer/DuckDB disabled, and L1 read with all network disabled.
Import the known completed Control-A retry, Baseline B, and Baseline C artifacts
only after the canary passes; reconcile imported/skipped/verified counts and
sample exact rank/text bytes. Do not push, open a PR, delete SQLite/DuckDB, or
make the HF Bucket public.
