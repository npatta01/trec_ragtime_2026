# HF Artifact Response Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace every runtime SQLite response-cache path with typed MessagePack artifacts in a private HF Bucket behind an owner-only `fsspec.simplecache`, and provide a verified offline importer for the canonical SQLite rows.

**Architecture:** A deep `ArtifactStore` owns private-bucket and L1/L2 mechanics. A typed `ResponseCache` owns canonical response identities and validators. Organizer and Dataset Viewer clients retain domain validation and write only after successful validation; SQLite exists only inside the explicit importer.

**Tech Stack:** Python 3.11+, `huggingface-hub>=1.5,<2`, `fsspec`, `msgspec`, `requests`, `unittest`, `uv`.

## Global Constraints

- Implement GitHub issue #12 exactly; its locked behavioral decisions override convenience fallbacks.
- L1 is disposable and HF is durable; a valid L1 hit must perform zero HF calls.
- Only an empty `HfApi.get_bucket_paths_info(bucket_id, [object_key])` result is a remote miss. Every other HF failure is fatal.
- Do not rely on `HfFileSystem.exists()` to classify a miss.
- Cache only fully decoded, endpoint/domain-validated successful values.
- Validate envelope identity and domain payload before applying TTL.
- HF and local write failures raise and remove the exact failed L1 candidate.
- Use strict typed `msgspec` MessagePack `.mpk` artifacts without compression.
- Never include credentials in identities, paths, artifacts, exceptions, logs, or manifests.
- Reject non-empty legacy `RAGTIME_CACHE_PATH` and `RAGTIME_CACHE_DIR`; do not reinterpret them.
- Keep `<git-common-dir>/ragtime-cache/ragtime2-corpus-v2.duckdb` revision-pinned and read-only.
- Import only schema `1`, namespace `ragtime2-2026-v1`; do not import `codex-*` rows and do not delete the source SQLite file.
- Follow red-green-refactor for each behavior and record the failing-test command/output in the task report.

---

### Task 1: Deep ArtifactStore with exact L1/L2 semantics

**Files:**
- Create: `src/trec_ragtime/artifact_store.py`
- Create: `tests/test_artifact_store.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Produces: `ArtifactStore`, `ArtifactStoreLike`, `ArtifactStoreError`, `ArtifactStoreConfigurationError`, `ArtifactStoreRemoteError`, `ArtifactValidationError`, and `ArtifactStore.from_env()`.
- Produces: `ArtifactStore.read(object_key: str) -> bytes | None`, `ArtifactStore.read_validated(object_key: str, decoder: Callable[[bytes], T]) -> T | None`, `ArtifactStore.write(object_key: str, payload: bytes) -> None`, `ArtifactStore.local_candidate(object_key: str) -> Path`, and sanitized `configuration()` metadata.
- Consumes: `default_project_cache_dir()` and `ensure_private_cache_dir()`.

- [ ] **Step 1: Add direct dependencies and write failing storage tests**

Add `fsspec`, `msgspec`, and `huggingface-hub>=1.5,<2`, then regenerate the lock with `uv lock`. Write tests with a fake `HfApi` and an injectable filesystem factory. At minimum name and cover these breaks:

```python
def test_valid_l1_hit_never_constructs_or_calls_hf(): ...
def test_l2_hit_uses_exact_paths_info_and_populates_l1_once(): ...
def test_empty_paths_info_is_the_only_remote_miss(): ...
def test_public_bucket_and_remote_failures_are_fatal(): ...
def test_corrupt_l1_is_evicted_and_repaired_once(): ...
def test_corrupt_l2_is_fatal_without_overwrite(): ...
def test_upload_failure_removes_exact_candidate_and_raises(): ...
def test_successful_write_uploads_and_retains_mode_0600_candidate(): ...
def test_bucket_directory_is_mode_0700_and_symlinks_are_rejected(): ...
def test_different_keys_can_overlap_and_duplicate_writes_are_last_writer_wins(): ...
def test_configuration_and_errors_never_expose_tokens(): ...
```

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run python -m unittest tests.test_artifact_store -v
```

Expected: import failure for `trec_ragtime.artifact_store` or specific missing-interface failures—not fixture or syntax errors.

- [ ] **Step 3: Implement the minimal deep store**

Validate `owner/bucket`, prefix, absolute local directory, and relative object keys. Construct `simplecache` with target protocol `hf` and the same token passed to `HfApi`. Lazily verify `bucket_info(...).private is True` only when L2 is needed. Resolve the candidate with the configured simplecache mapper. `read_validated` catches only `ArtifactValidationError` from the decoder for repair; permission and storage errors remain fatal. After a repaired download, a second validation failure is fatal. Wrap external errors without including URLs carrying credentials or payload bytes.

- [ ] **Step 4: Verify GREEN and mutation cases**

Run:

```bash
uv run python -m unittest tests.test_artifact_store -v
uv run python -m unittest tests.test_paths -v
git diff --check
```

Expected: all targeted tests pass. Mentally mutate the exact-path existence call, candidate deletion, privacy condition, and token omission; a named test must fail for each.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock src/trec_ragtime/artifact_store.py tests/test_artifact_store.py
git commit -m "feat: add private HF artifact store"
```

### Task 2: Typed response identity, envelope, validation, and TTL

**Files:**
- Replace: `src/trec_ragtime/cache.py`
- Create: `tests/test_response_cache.py`
- Modify: `src/trec_ragtime/__init__.py`

**Interfaces:**
- Produces: `CACHE_SCHEMA_VERSION = 1`, `DEFAULT_CACHE_NAMESPACE = "ragtime2-2026-v1"`, `CacheEntry`, `ResponseCache`, `ResponseEnvelope`, and canonical request/object-key helpers.
- Consumes: `ArtifactStoreLike.read_validated` and `ArtifactStoreLike.write`.
- `ResponseCache.get(*, base_url, operation, request, ttl_seconds, validator) -> CacheEntry | None`.
- `ResponseCache.put(*, base_url, operation, request, response, stored_at: float | None = None) -> float`.

- [ ] **Step 1: Write failing canonicalization and envelope tests**

Use literal expected object keys and MessagePack byte fixtures derived independently of production helpers. Cover recursively reordered mappings, list-order preservation, credential exclusion, wrong schema/namespace/operation/base URL/request digest, finite `stored_at`, malformed/non-object payloads, domain-validator rejection, validation-before-TTL, expiry-as-miss, stable bytes with injected clock, and last-writer-wins.

```python
def test_request_key_is_stable_for_recursively_reordered_mappings(): ...
def test_credentials_do_not_affect_or_appear_in_key_or_bytes(): ...
def test_local_domain_corruption_requests_one_store_repair(): ...
def test_remote_domain_corruption_is_fatal(): ...
def test_valid_expired_envelope_is_a_miss_after_validation(): ...
def test_put_encodes_once_and_returns_stored_at_without_read(): ...
```

- [ ] **Step 2: Verify RED**

Run `uv run python -m unittest tests.test_response_cache -v` and require failures caused by the absent `ResponseCache` behavior.

- [ ] **Step 3: Implement strict typed response caching**

Use frozen `msgspec.Struct` types with forbidden unknown fields. Canonicalize JSON-like mappings recursively and reject unsupported/non-finite values. Hash canonical MessagePack identity bytes with SHA-256, use binary digests inside the envelope, and use the exact layout `api-responses/v1/<operation>/<prefix>/<hash>.mpk`. The validated decoder checks envelope identity and invokes the supplied endpoint validator before returning `CacheEntry`; TTL is checked afterward.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run python -m unittest tests.test_response_cache -v
uv run python -m unittest tests.test_artifact_store -v
git diff --check
```

- [ ] **Step 5: Commit**

```bash
git add src/trec_ragtime/cache.py src/trec_ragtime/__init__.py tests/test_response_cache.py
git commit -m "feat: add typed response artifact cache"
```

### Task 3: Remove SQLite from runtime clients and public surfaces

**Files:**
- Modify: `src/trec_ragtime/client.py`
- Modify: `src/trec_ragtime/huggingface.py`
- Modify: `src/trec_ragtime/retriever.py`
- Modify: `src/trec_ragtime/topic_candidates.py`
- Modify: `src/trec_ragtime/__init__.py`
- Modify: `tests/test_client.py`
- Modify: `tests/test_huggingface.py`
- Modify: `tests/test_corpus_cli.py`
- Modify: `tests/test_topic_candidates.py`

**Interfaces:**
- `RagtimeClient(..., cache: ResponseCache | None, artifact_store: ArtifactStoreLike | None, use_cache: bool = True, ...)` replaces `cache_path`.
- `RagtimeClient.from_env()` constructs `ArtifactStore.from_env()` only when caching is enabled.
- `RagtimeClient.artifact_store` and sanitized cache configuration replace `cache_path`.
- `HuggingFaceCorpus` consumes `ResponseCache`; no SQLite exception or warning path remains.

- [ ] **Step 1: Rewrite tests first for successful-only fatal-cache semantics**

Replace SQLite fixtures with real `ResponseCache` over an in-memory fake store. Preserve endpoint tests and add explicit assertions that transport/status/JSON/protocol failures never call `write`, write/read failures propagate, failed refresh leaves old bytes unchanged, malformed L1 repairs once, corrupt L2 never reaches participant requests, and non-empty legacy env settings are rejected. Update fakes/manifests to use sanitized artifact configuration rather than `.sqlite3` paths.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run python -m unittest tests.test_client tests.test_huggingface tests.test_corpus_cli tests.test_topic_candidates -v
```

Expected: failures on old constructor/env/public behavior.

- [ ] **Step 3: Implement runtime migration**

Remove `sqlite3` imports, `SQLiteResponseCache`, `default_cache_path`, `cache_path` parameters/properties, warning-and-bypass behavior, and runtime delete/clear calls. Pass the existing endpoint validator into `ResponseCache.get`; only call `put` after the live validator succeeds. Preserve organizer sorting, rate limiting, retries, Dataset Viewer anonymous-auth protections, first-seen order, and all DuckDB behavior.

- [ ] **Step 4: Verify focused and full offline tests**

Decrypt first, then run:

```bash
sops decrypt --output .env config/secrets.sops.env
uv run python -m unittest tests.test_client tests.test_huggingface tests.test_corpus_cli tests.test_topic_candidates -v
uv run python -m unittest discover -s tests -v
git diff --check
```

- [ ] **Step 5: Commit**

```bash
git add src/trec_ragtime/client.py src/trec_ragtime/huggingface.py src/trec_ragtime/retriever.py src/trec_ragtime/topic_candidates.py src/trec_ragtime/__init__.py tests/test_client.py tests/test_huggingface.py tests/test_corpus_cli.py tests/test_topic_candidates.py
git commit -m "refactor: remove SQLite response cache runtime"
```

### Task 4: Canonical SQLite importer, documentation, and bounded verification hooks

**Files:**
- Create: `src/trec_ragtime/cache_import.py`
- Create: `tests/test_cache_import.py`
- Modify: `pyproject.toml`
- Modify: `.env.example`
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `notebooks/ragtime_single_query.ipynb`
- Modify: `notebooks/ragtime2_mini_viewer.ipynb`

**Interfaces:**
- Adds CLI: `ragtime-cache-import = "trec_ragtime.cache_import:main"`.
- Produces: `import_sqlite_cache(source: Path, cache: ResponseCache, manifest_path: Path, *, canonical_namespace: str = "ragtime2-2026-v1") -> ImportSummary`.
- Produces independent destination verification using production validators and preserved `stored_at`.

- [ ] **Step 1: Write failing importer tests**

Create temporary SQLite fixtures matching the exact legacy table. Cover read-only source access, only schema/namespace/operation acceptance, endpoint echo/score/document validation, finite timestamps, 30-day Dataset Viewer TTL, idempotent overwrite, preserved `stored_at`, independent reread, sanitized counts/manifest, source hash, no body/query/token leakage, and unchanged source bytes.

- [ ] **Step 2: Verify RED**

Run `uv run python -m unittest tests.test_cache_import -v`; require import or missing-behavior failures.

- [ ] **Step 3: Implement importer and update user-facing configuration**

Open SQLite only via `file:<absolute-path>?mode=ro`; keep `sqlite3` imports confined to this module. Reuse the production organizer/content/Dataset Viewer validators without constructing live transports. Write manifest atomically with owner-only permissions. Update all docs/notebooks to describe private HF L2, disposable L1, fatal failures, environment names, and the explicit importer. Do not retain SQLite runtime advice or a `cache_path` display.

- [ ] **Step 4: Verify no SQLite runtime and the complete offline suite**

Run:

```bash
sops decrypt --output .env config/secrets.sops.env
uv run python -m unittest tests.test_cache_import -v
uv run python -m unittest discover -s tests -v
rg -n "SQLiteResponseCache|RAGTIME_CACHE_PATH|RAGTIME_CACHE_DIR|cache_path" src tests README.md .env.example AGENTS.md notebooks
git diff --check
```

Expected: tests pass; any `sqlite3`/SQLite hits are confined to `cache_import.py`, importer tests, and explicit historical migration documentation. DuckDB references remain unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/trec_ragtime/cache_import.py tests/test_cache_import.py pyproject.toml .env.example README.md AGENTS.md notebooks/ragtime_single_query.ipynb notebooks/ragtime2_mini_viewer.ipynb
git commit -m "feat: import canonical SQLite response artifacts"
```

---

## Issue #12 completion gate

Before issue #13 begins, decrypt `.env`, run the entire offline suite and
`git diff --check`, inspect the branch diff for SQLite runtime remnants and
credential exposure, then run only bounded private-bucket canaries. If an
explicit bucket does not yet exist, create/select one private bucket as an
in-scope external prerequisite, verify privacy through `bucket_info`, and never
make it public. Import the canonical SQLite file only after canaries pass;
reconcile expected source counts `pipeline=1202`, `content=10`, and
`huggingface_document=20`, plus independent destination verification. Never
delete the source database.
