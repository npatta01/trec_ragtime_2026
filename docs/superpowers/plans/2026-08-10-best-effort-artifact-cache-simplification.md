# Best-Effort Artifact Cache Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to execute this plan task by task.

**Goal:** Make the private HF Bucket/simplecache cache a strict-validation performance aid instead of a runtime dependency, while preserving issues #12/#13 artifact identities, typed MessagePack, complete-only retrieval caching, importer integrity, and DuckDB behavior.

**Architecture:** Retain three deep modules: `ArtifactStore` owns private HF L2 plus `fsspec.simplecache` L1; `ResponseCache` owns response identity, TTL, and typed envelopes; `RetrievalArtifactCache` owns exact retrieval identity, typed envelopes, and complete-result validation. Cache infrastructure failures degrade to a sanitized warning and miss for runtime callers. Valid in-memory work remains usable when a write fails. Importers remain strict because their purpose is durable migration, not runtime acceleration.

**Tech Stack:** Python 3.11+, `fsspec`, `huggingface_hub.HfFileSystem`, `msgspec`, `unittest`, `uv`, SOPS, DuckDB (unchanged).

## Global Constraints

- Work only on `codex/hf-artifact-retrieval-cache`; preserve unrelated changes and never rewrite existing user work.
- Decrypt the ignored root `.env` before every test phase with `sops decrypt --output .env config/secrets.sops.env`; confirm `git check-ignore .env` when relevant. Never print or commit secrets.
- Run focused and full offline tests before bounded integration checks. Do not call the participant API in integration checks for this simplification.
- Retain the exact response and retrieval object-key layouts, schema versions, canonical request bytes, and existing valid MessagePack wire compatibility.
- Cache only fully validated successful response values. Cache retrievals only when the result is complete and has no operational hydration failures. Missing optional English translations remain explicit and cacheable when canonical source text is present.
- Treat cache storage as best effort for runtime paths: missing, authentication, network, server, bucket-policy, and filesystem failures become misses or unsuccessful writes after one sanitized `RuntimeWarning`; disable further remote operations on that store instance. Invalid caller configuration and invalid object keys remain programming errors.
- A valid L1 candidate must be readable without constructing or contacting Hugging Face. Corrupt L1 is evicted and may repair from L2 once. Corrupt L2 is evicted and treated as a miss. The next fully validated live result may overwrite it.
- Keep the HF Bucket private. A public bucket must disable remote access for the store instance and must never receive a write.
- `ArtifactStore.write` and `ArtifactStoreLike.write` return `bool`: `True` only after the write completes and a complete L1 candidate is retained; `False` on storage failure. Clean any partial exact candidate.
- Preserve owner-only L1 directory protection and symlink-directory rejection. Do not add filesystem hardening beyond what the enclosing `0700` cache directory materially requires for this competition repository.
- Preserve participant request rate limiting and live-response validation. Cache failures must not consume or bypass the participant request limiter.
- Preserve the canonical read-only DuckDB corpus, publication-token checks, source/translation semantics, and no-DuckDB/no-network warm retrieval behavior.
- Keep all SQLite imports isolated to `cache_import.py`; SQLite must not return to runtime code.
- Control A/B/C must carry newly computed complete retrieval results in memory through topic assembly. A failed cache write yields no artifact reference, still permits that run to finish, and makes later resume recompute the retrieval.
- Migration importers remain strict: an unsuccessful write or a failed/mismatched verification is an import failure and is never counted as imported.
- Prefer deletion to replacement abstractions. Remove shallow aliases, pass-through helpers, duplicate validation, and unused public cache exports without introducing a universal repository/codec framework.
- Every task follows red-green-refactor: add or amend behavior tests first, run them and observe a relevant failure, implement the minimum change, rerun the focused tests, then commit. Update test doubles to the new boolean write contract.

## Task 1: Make `ArtifactStore` Best Effort

**Files:**

- Modify: `src/trec_ragtime/artifact_store.py`
- Modify: `tests/test_artifact_store.py`
- Modify: `pyproject.toml`
- Modify mechanically if dependency resolution changes: `uv.lock`

**Step 1: Add failing storage-policy tests**

Add focused tests demonstrating:

- a valid L1 candidate returns without constructing the HF filesystem;
- a normal remote absence returns `None` without warning or disabling the store;
- authentication/network/server/private-policy/filesystem read failures return `None`, emit one sanitized `RuntimeWarning`, and prevent later remote calls while preserving valid L1 reads;
- a public bucket causes a miss, warns once, disables remote access, and never writes;
- corrupt L1 is evicted and retried from L2 once;
- corrupt L2 is evicted and becomes a miss rather than an exception;
- failed writes return `False`, remove only the exact partial candidate, warn once, and disable later remote operations;
- successful writes return `True` and retain a complete L1 candidate;
- invalid configuration and invalid keys still raise their validation errors.

Keep tests at public behavior boundaries. Remove tests whose only purpose is `get_bucket_paths_info`, directory-list retry, target-cache invalidation, `_get_cached_file_before_open`, per-candidate mode enforcement, or fsspec-private-method pinning.

**Step 2: Run the tests to verify RED**

```bash
sops decrypt --output .env config/secrets.sops.env
uv run python -m unittest tests.test_artifact_store -v
```

Expected: new best-effort expectations fail against fatal exceptions and the old `None` write contract.

**Step 3: Implement the minimal store policy**

Keep this public operational surface:

```python
class ArtifactStoreLike(Protocol):
    def read(self, object_key: str) -> bytes | None: ...
    def read_validated(self, object_key: str, decoder: Callable[[bytes], T]) -> T | None: ...
    def write(self, object_key: str, payload: bytes) -> bool: ...
    def evict_local(self, object_key: str) -> None: ...
```

Implement one per-instance remote circuit breaker and one sanitized warn-once helper. The first remote read/write verifies privacy. `FileNotFoundError` on the ordinary simplecache read path is a normal miss. Other operational failures warn, disable, and return `None`/`False`. Reads always inspect the deterministic exact L1 candidate before checking the breaker or constructing HF. `read_validated` repairs only `ArtifactValidationError`: invalid L1 is evicted and L2 attempted once; invalid L2 is evicted and returned as a miss. Do not swallow invalid keys/configuration.

Use the public `fsspec.simplecache` open/close path. Delete exact-path listing probes, list-visibility retry machinery, target listing-cache invalidation, private hydration-method access, and the dependency upper bound that existed only for that private method. Retain private-bucket verification before a remote operation.

**Step 4: Run focused tests and inspect the diff**

```bash
uv run python -m unittest tests.test_artifact_store -v
git diff --check
```

**Step 5: Commit**

```bash
git add src/trec_ragtime/artifact_store.py tests/test_artifact_store.py pyproject.toml uv.lock
git commit -m "refactor: make artifact storage best effort"
```

## Task 2: Deepen `ResponseCache` and Remove Client Wrappers

**Files:**

- Modify: `src/trec_ragtime/cache.py`
- Modify: `src/trec_ragtime/client.py`
- Modify: `src/trec_ragtime/huggingface.py`
- Modify: `tests/test_response_cache.py`
- Modify: `tests/test_client.py`
- Modify: `tests/test_huggingface.py`

**Step 1: Add failing typed cache-aside tests**

Add tests for one canonical `ResponseCache.fetch`-style operation proving:

- a valid hit invokes the supplied validator exactly once total and returns the validator's typed value;
- TTL is evaluated only after strict envelope/domain validation;
- refresh bypasses reads;
- a live response is validated before any write;
- a validation failure is never written;
- a store write returning `False` still returns the fully validated live value and no false durable-success signal;
- client/Dataset Viewer runtime requests continue on cache read/write failure while preserving participant request limiting and error handling.

Delete tests for removed client/HF pass-through cache wrappers and the unused `pipeline` alias.

**Step 2: Run the tests to verify RED**

```bash
uv run python -m unittest tests.test_response_cache tests.test_client tests.test_huggingface -v
```

Expected: the typed cache-aside operation and boolean-write behavior do not exist yet, and cache hits are validated twice.

**Step 3: Implement the minimal response seam**

Keep `CacheEntry[T]` as the result of the supplied validator plus `stored_at`; do not return raw decoded JSON for a second caller validation. Put canonical identity, strict MessagePack envelope decoding, domain validation, TTL, fetch, and best-effort write-after-validation behind one response-cache operation used by both `RagtimeClient` and `HuggingFaceCorpus`.

Delete `RagtimeClient._cache_get`, `RagtimeClient._cache_put`, the second validation on a hit, and `RagtimeClient.pipeline`. Remove equivalent shallow Hugging Face wrappers where the new operation makes them redundant. `put` returns the stored timestamp on success and `None` on unsuccessful storage.

Do not change endpoint payload validation, token handling, cache keys, TTL/namespace/refresh semantics, or the rule that the RAGTIME bearer token is never sent to Hugging Face.

**Step 4: Run focused tests and inspect the diff**

```bash
uv run python -m unittest tests.test_response_cache tests.test_client tests.test_huggingface -v
git diff --check
```

**Step 5: Commit**

```bash
git add src/trec_ragtime/cache.py src/trec_ragtime/client.py src/trec_ragtime/huggingface.py tests/test_response_cache.py tests/test_client.py tests/test_huggingface.py
git commit -m "refactor: deepen validated response caching"
```

## Task 3: Prune Retrieval Cache Surface and Handle Failed Writes

**Files:**

- Modify: `src/trec_ragtime/retrieval_cache.py`
- Modify: `src/trec_ragtime/retriever.py`
- Modify: `src/trec_ragtime/__init__.py`
- Modify: `tests/test_retrieval_cache.py`
- Modify: `tests/test_retriever.py`
- Search/update repository callers of removed names under `src/` and `tests/`

**Step 1: Add failing retrieval tests**

Add tests proving:

- a complete validated retrieval whose store write returns `False` is returned unchanged and has no artifact reference;
- partial results and results with operational hydration failures are never offered to the store;
- a warm valid retrieval still performs zero participant search, content, Dataset Viewer, and DuckDB work;
- publication/corpus identity mismatch still fails closed;
- existing retrieval object keys and canonical MessagePack fixtures remain byte-compatible.

Replace tests of removed aliases with tests of the canonical names.

**Step 2: Run the tests to verify RED**

```bash
uv run python -m unittest tests.test_retrieval_cache tests.test_retriever -v
```

Expected: `RetrievalArtifactCache.put` assumes writes always succeed and currently produces a reference.

**Step 3: Implement and prune**

Change `RetrievalArtifactCache.put` so it returns the original complete result without a reference when `store.write(...)` is `False`; attach a reference only after `True`. Keep complete-result validation inside the adapter and preserve all identity/wire/provenance rules.

Remove shallow or unused surface after updating all repository callers: `RetrievalRequestWire`, redundant identity/reference compatibility aliases, retrieval behavior aliases, package-root cache implementation exports, and retriever pass-through aliases `retrieval_identity` and `retrieve_cached`. Retain exactly one public identity builder (`build_retrieval_identity`) for runners/importers and one cache-only retrieval method (`get_cached_retrieval`) for resume paths. Do not expose storage internals through a new abstraction.

**Step 4: Run focused tests and repository checks**

```bash
uv run python -m unittest tests.test_retrieval_cache tests.test_retriever -v
rg "retrieval_identity|retrieve_cached|RetrievalRequestWire|request_hash|fingerprint_hex|stored_timestamp|cache_stored_at|to_record" src tests
git diff --check
```

Classify any remaining search hits: retain only names proven canonical/required by persisted manifests; otherwise remove them.

**Step 5: Commit**

```bash
git add src/trec_ragtime/retrieval_cache.py src/trec_ragtime/retriever.py src/trec_ragtime/__init__.py tests
git commit -m "refactor: prune retrieval cache wrappers"
```

## Task 4: Carry Fresh Retrievals Through Control A/B/C In Memory

**Files:**

- Modify: `src/trec_ragtime/query_baselines.py`
- Modify: `src/trec_ragtime/topic_candidates.py`
- Modify: `tests/test_query_baselines.py`
- Modify: `tests/test_topic_candidates.py`

**Step 1: Add failing runner tests**

Add representative tests proving:

- Control A completes when a fully hydrated retrieval has no artifact reference after an unsuccessful cache write; the optional reference is omitted from its checkpoint/output;
- Control B and C carry a newly computed complete `RetrievalResult` in memory into topic assembly when no reference was persisted;
- a resumed checkpoint with no reusable reference recomputes the retrieval instead of pretending it is durable;
- a prior-process reference is still loaded through the cache-only seam and is checked against the expected identity/reference;
- fatal participant/API/domain failures retain existing run-finalization behavior, while cache-only `ArtifactStoreError` systemic-failure branches are gone.

**Step 2: Run the tests to verify RED**

```bash
uv run python -m unittest tests.test_query_baselines tests.test_topic_candidates -v
```

Expected: current Control A reference validation rejects no-ref success, and B/C topic assembly reloads fresh results from the cache.

**Step 3: Implement the in-memory happy path**

Make artifact references optional for complete successful runtime results. Change B/C query-task results to carry the in-memory retrieval alongside checkpoint metadata until topic assembly. Prefer that in-memory object; use `get_cached_retrieval` only for a reference created by an earlier process. A successful no-reference checkpoint is not reusable across process restart, so resume schedules recomputation.

Delete duplicated recomputation/identity checks already guaranteed by typed retrieval-cache validation. Retain exact checkpoint-to-artifact identity comparison for prior references and preserve checkpoint wire shapes except that an optional reference may be absent. Remove runner systemic/fatal handling that exists only for cache-storage failures; do not weaken participant, corpus, checkpoint, or domain validation.

**Step 4: Run focused runner tests and inspect the diff**

```bash
uv run python -m unittest tests.test_query_baselines tests.test_topic_candidates -v
git diff --check
```

**Step 5: Commit**

```bash
git add src/trec_ragtime/query_baselines.py src/trec_ragtime/topic_candidates.py tests/test_query_baselines.py tests/test_topic_candidates.py
git commit -m "refactor: keep uncached retrievals in runner memory"
```

## Task 5: Keep Importers Strict and Align Documentation

**Files:**

- Modify: `src/trec_ragtime/cache_import.py`
- Modify: `src/trec_ragtime/retrieval_import.py`
- Modify: `tests/test_cache_import.py`
- Modify: `tests/test_retrieval_import.py`
- Modify: `README.md`
- Do not edit generated `AGENTS.md`; the separately maintained instruction manifest is not present in this worktree
- Preserve the earlier issue implementation specs as historical records; the approved 2026-08-10 simplification spec is their explicit policy revision

**Step 1: Add failing importer tests**

Add tests proving both importers:

- treat `store.write(...) is False` as an import failure;
- do not count an unsuccessful write as imported;
- independently verify the written artifact and reject a miss/mismatch;
- preserve source immutability, idempotency, sanitized manifests, exact identities, and SQLite isolation.

**Step 2: Run the tests to verify RED**

```bash
uv run python -m unittest tests.test_cache_import tests.test_retrieval_import -v
```

Expected: importers currently assume the old `None` success contract.

**Step 3: Adapt importers and docs**

Require `True` from the store before verification/counting. Keep all SQLite code in `cache_import.py` and ensure no runtime module imports `sqlite3`.

Update documentation to describe the competition-oriented best-effort runtime policy, the strict importer distinction, complete-only writes, optional runtime artifact references, private-bucket requirement, and preservation of DuckDB. Remove obsolete claims about fatal authentication/network/server/cache-corruption behavior. Do not weaken statements about payload validation or private storage.

**Step 4: Run focused and static checks**

```bash
uv run python -m unittest tests.test_cache_import tests.test_retrieval_import -v
rg "^import sqlite3|^from sqlite3" src/trec_ragtime
git diff --check
```

Expected SQLite result: `src/trec_ragtime/cache_import.py` only.

**Step 5: Commit**

```bash
git add src/trec_ragtime/cache_import.py src/trec_ragtime/retrieval_import.py tests/test_cache_import.py tests/test_retrieval_import.py README.md docs instructions
git commit -m "docs: align artifact cache migration policy"
```

## Task 6: Verify Offline, Run Bounded Cache Canaries, and Review

**Files:**

- Modify only if verification reveals a defect; fixes must receive their own focused regression tests and review.
- Record evidence in the SDD task report and final handoff; do not commit plaintext logs or credentials.

**Step 1: Run the complete offline suite**

```bash
sops decrypt --output .env config/secrets.sops.env
git check-ignore .env
uv run python -m unittest discover -s tests -v
```

Expected: all tests pass with no participant endpoint access.

**Step 2: Run static and dependency checks**

```bash
uv lock --check
uv run python -m compileall -q src tests
rg "^import sqlite3|^from sqlite3" src/trec_ragtime
git diff --check
git status --short
```

Expected: lock is current, compilation succeeds, SQLite remains importer-only, no whitespace errors, and no `.env` is tracked.

**Step 3: Run bounded private-cache integration checks**

Use existing repository canary commands/tests if present. Bound the run to one disposable uniquely named cache object and do not call the participant API. Verify:

1. The configured HF Bucket is private.
2. A validated artifact can be written to L2 and then read through L1.
3. A second read succeeds with remote access disabled/unavailable, demonstrating the L1-first path.
4. A deliberately unavailable remote operation produces one sanitized warning/miss and does not crash the caller.
5. Clean up only the exact disposable object if an existing safe cleanup command exists; otherwise leave it identified for manual cleanup rather than performing a broad deletion.

**Step 4: Run independent whole-branch review**

Review the full branch against:

- GitHub issues #12 and #13 except for the explicitly revised competition tradeoff that cache operational failures are best-effort;
- `docs/superpowers/specs/2026-08-10-best-effort-artifact-cache-simplification-design.md`;
- this plan's Global Constraints;
- all unresolved PR #23 review feedback.

Any valid finding receives one regression test and a focused fix before the final re-review.

**Step 5: Final branch/PR preparation**

Confirm commits contain only intended files, no secret material, and no unrelated changes. Push the existing branch only after all checks and reviews pass, then update PR #23's body to explain the simplified three-module architecture and revised best-effort cache policy. Do not merge the PR.
