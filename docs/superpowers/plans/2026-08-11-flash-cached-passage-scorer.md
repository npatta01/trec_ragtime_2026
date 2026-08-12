# Flash-Cached Passage Scorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `PassageScorer.score(query, passages)` perform one logical Flash request whose worker owns exact-request caching, GPU micro-batching, validation, and horizontally scalable execution.

**Architecture:** Replace client-side endpoint jobs per model batch with one strict request containing every unique passage below Flash's 10 MB limit. Each self-contained Flash worker derives a sealed exact-request cache identity, reads a private Hugging Face Bucket through a worker-local L1, lazily loads the pinned cross-encoder only on a miss, scores internal micro-batches, publishes a complete validated artifact, and rereads the durable winner. Multiple workers may duplicate a simultaneous cold computation, but no mutable score union or distributed lock is used.

**Tech Stack:** Python 3.11+, `msgspec`, `fsspec.simplecache`, `huggingface_hub`, sentence-transformers, PyTorch, Runpod Flash, `unittest`.

## Global Constraints

- The public caller remains `PassageScorer.score(query, passages) -> Mapping[bytes, float]` and exposes no cache controls.
- One scorer call creates one logical Flash job; model batch size is worker-only.
- Identity includes schema/behavior, scorer fingerprint, exact UTF-8 query hash, and sorted unique passage hashes.
- Artifacts contain no query text, passage text, credentials, headers, or raw exception strings.
- Valid hits do not load or invoke the model.
- Cache operational, permission, and malformed durable-artifact failures are fatal, not GPU misses.
- Partial, foreign, malformed, or non-finite responses are never cached or returned.
- Duplicate cold computation is allowed; distributed leases and locks are out of scope.
- Workers in one scored run use the same pinned GPU type and model fingerprint.
- No stable deployment is authorized. Only Task 6's bounded ephemeral `flash dev` smoke is authorized.
- Decrypt `.env` before tests and never print its values.

## File Structure

- Modify `src/trec_ragtime/flash_passage_scorer.py` for one-call transport and strict telemetry.
- Modify `src/trec_ragtime/flash_worker.py` for cache-first execution, durable L1/L2, and internal batches.
- Modify `tests/test_flash_passage_scorer.py` and `tests/test_passage_reranking_integration.py`.
- Modify `README.md`.
- Create `scripts/flash_cached_passage_smoke.py` and `docs/experiments/2026-08-11-flash-cached-passage-smoke.md`.

---

### Task 1: Worker-Owned Exact-Request Score Artifact

**Files:**
- Modify: `src/trec_ragtime/flash_worker.py`
- Modify: `tests/test_flash_passage_scorer.py`

**Interfaces:**
- Consumes: validated Flash request fields and finite `bytes -> float` maps.
- Produces: private `FlashPassageWorker` methods `_cache_identity()`, `_cache_object_key()`, `_encode_cache_artifact()`, and `_decode_cache_artifact()`.

- [ ] **Step 1: Write failing worker identity tests**

~~~python
class WorkerCacheIdentityTests(unittest.TestCase):
    def test_hash_order_does_not_change_identity(self) -> None:
        left = FlashPassageWorker._cache_identity(
            "scorer-v1", "query", [b"a" * 32, b"b" * 32]
        )
        right = FlashPassageWorker._cache_identity(
            "scorer-v1", "query", [b"b" * 32, b"a" * 32]
        )
        self.assertEqual(left, right)
        self.assertEqual(
            FlashPassageWorker._cache_object_key(left),
            FlashPassageWorker._cache_object_key(right),
        )
~~~

Also prove scorer, exact query bytes, passage set, schema, and behavior each rotate identity.

- [ ] **Step 2: Run and verify the missing-method failure**

Run: `sops decrypt --output .env config/secrets.sops.env && git check-ignore .env && uv run python -m unittest tests.test_flash_passage_scorer.WorkerCacheIdentityTests -v`

Expected: FAIL because the worker has no cache identity methods.

- [ ] **Step 3: Implement private worker identity methods**

~~~python
identity = (
    CACHE_SCHEMA_VERSION,
    CACHE_BEHAVIOR_VERSION,
    scorer_fingerprint,
    hashlib.sha256(query.encode("utf-8")).digest(),
    tuple(sorted(set(passage_hashes))),
)
~~~

Validate nonblank/newline-free strings, exact 32-byte hashes, and at least one passage. Hash canonical MessagePack over the identity. The object path contains only fingerprint and request digests.

- [ ] **Step 4: Write failing artifact tests**

Test stable canonical bytes, sorted complete coverage, non-finite values, wrong identity, truncation, extra fields, noncanonical bytes, and absence of source text.

~~~python
payload = FlashPassageWorker._encode_cache_artifact(identity, scores)
self.assertEqual(
    FlashPassageWorker._decode_cache_artifact(payload, identity), scores
)
self.assertNotIn(b"query", payload)
~~~

- [ ] **Step 5: Implement private canonical artifact codec**

Encode schema, behavior, scorer fingerprint, query hash, ordered passage hashes, and ordered hash/score pairs. Decode generic MessagePack, validate exact shape/types/coverage/finite values, re-encode, and require byte equality. Use fixed credential-safe worker errors.

- [ ] **Step 6: Prove class-only ownership**

Extract and execute `FlashPassageWorker` with project imports blocked, then assert the isolated class derives identical keys and artifact bytes. Do not create a local cache module, cache adapter, or public cache export.

- [ ] **Step 7: Verify and commit**

Run: `uv run python -m unittest tests.test_flash_passage_scorer.WorkerCacheIdentityTests tests.test_flash_passage_scorer.StaticFlashWorkerTests -v`

Expected: PASS.

~~~bash
git add src/trec_ragtime/flash_worker.py tests/test_flash_passage_scorer.py
git commit -m "feat: add worker-owned passage score artifacts"
~~~

### Task 2: One-Call Flash Transport

**Files:**
- Modify: `src/trec_ragtime/flash_passage_scorer.py`
- Modify: `tests/test_flash_passage_scorer.py`

**Interfaces:**
- Consumes: existing strict wire request and the worker-owned cache identity from Task 1.
- Produces: one-request `RunpodFlashPassageScorer.score()` and strict private cache telemetry.

- [ ] **Step 1: Replace wave tests with failing one-call tests**

Use a recording async runner and assert 65 passages with model batch 32 produce exactly one runner call and complete scores. Assert a fully wrapped request at or above 10 MB fails rather than splitting.

- [ ] **Step 2: Run and verify failure against `_build_batches()`**

Run: `uv run python -m unittest tests.test_flash_passage_scorer.RunpodFlashPassageScorerTests -v`

Expected: FAIL because the current scorer creates one endpoint job per model batch.

- [ ] **Step 3: Add strict response telemetry**

Require exactly:

~~~python
{
    "schema_version": 1,
    "scorer_fingerprint": fingerprint,
    "scores": [{"hash": hash_hex, "score": score}],
    "cache": {
        "status": "hit" | "miss_written" | "miss_raced",
        "object_key": object_key,
        "row_count": positive_int,
        "batch_count": nonnegative_int,
    },
}
~~~

Reject extra fields, unsafe keys, count mismatches, unknown status, `batch_count != 0` on hits, and non-finite values. `score()` returns only scores.

- [ ] **Step 4: Replace `_build_batches()` with `_build_request()`**

Deduplicate passages, encode all rows once, measure the actual Runpod wrapper, and require it below `min(max_payload_bytes, 10_000_000)`. Preserve timeout and cancellation; remove client-side batch waves.

- [ ] **Step 5: Verify transport behavior**

Run: `uv run python -m unittest tests.test_flash_passage_scorer.RunpodFlashPassageScorerTests tests.test_flash_passage_scorer.FlashEndpointBatchRunnerTests -v`

Expected: PASS with one runner call and bounded cancellation.

- [ ] **Step 6: Commit**

~~~bash
git add src/trec_ragtime/flash_passage_scorer.py tests/test_flash_passage_scorer.py
git commit -m "refactor: send one logical Flash scoring request"
~~~

### Task 3: Cache-First Self-Contained Worker

**Files:**
- Modify: `src/trec_ragtime/flash_worker.py`
- Modify: `tests/test_flash_passage_scorer.py`
- Modify: `tests/test_passage_reranking_integration.py`

**Interfaces:**
- Consumes: Task 1 worker-owned artifact format and Task 2 response envelope.
- Produces: `FlashPassageWorker(loader=None, cache_store=None)` with cache-before-model behavior and internal batches.

- [ ] **Step 1: Write failing hit/miss and micro-batch tests**

~~~python
cold = worker.score(request)
self.assertEqual(cold["cache"]["status"], "miss_written")
self.assertEqual(loader.calls, 1)

warm_worker = FlashPassageWorker(loader=failing_loader, cache_store=store)
warm = warm_worker.score(request)
self.assertEqual(warm["cache"]["status"], "hit")
self.assertEqual(warm["scores"], cold["scores"])
~~~

For 65 rows at batch size 32, assert one worker request calls `model.predict` three times and reports `batch_count == 3`.

- [ ] **Step 2: Run and verify cache behavior is absent**

Run: `uv run python -m unittest tests.test_flash_passage_scorer.StaticFlashWorkerTests -v`

Expected: FAIL because the worker always loads/scores.

- [ ] **Step 3: Connect Task 1's codec to cache I/O**

Use only the existing private worker methods from Task 1. Preserve extracted-class execution with project imports blocked; do not introduce a caller-side or public cache implementation.

- [ ] **Step 4: Add injected store and cache-first flow**

Validate request, derive identity, check cache, and return hit before `_ensure_loaded`. On a miss, load once, score internal batches, validate, write, evict, reread, and return the durable winner. Report `miss_raced` when reread bytes differ from this worker's complete candidate.

- [ ] **Step 5: Add duplicate-cold and failure tests**

Run two worker instances against a barrier-backed store. Permit both models to execute; require complete equal coverage and one readable winner. Test model failure, incomplete result, refused write, malformed hit, and reread failure; no partial object or raw exception text may escape.

- [ ] **Step 6: Verify isolated class and public integration**

Run: `uv run python -m unittest tests.test_flash_passage_scorer.StaticFlashWorkerTests tests.test_flash_passage_scorer.FlashDecorationTests tests.test_passage_reranking_integration -v`

Expected: PASS.

- [ ] **Step 7: Commit**

~~~bash
git add src/trec_ragtime/flash_worker.py tests/test_flash_passage_scorer.py tests/test_passage_reranking_integration.py
git commit -m "feat: cache passage scores inside Flash workers"
~~~

### Task 4: Durable Cache and Scalable Definition

**Files:**
- Modify: `src/trec_ragtime/flash_worker.py`
- Modify: `src/trec_ragtime/flash_passage_scorer.py`
- Modify: `tests/test_flash_passage_scorer.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: Task 3 store seam.
- Produces: self-contained L1/L2 store, tuple worker ranges, `max_concurrency`, NetworkVolume/env passthrough, pinned dependencies.

- [ ] **Step 1: Write failing self-contained store tests**

Inject fake `HfApi` and `fsspec` factories into the isolated worker. Prove private bucket enforcement, private local directory, L1 hit without remote I/O, one repair from malformed L1, clean L2 miss, fatal transport/permission/malformed-L2/write/reread failures, and no token in key/error/payload/repr.

- [ ] **Step 2: Implement worker-local durable store**

When no store is injected, lazily validate `RAGTIME_HF_BUCKET_ID`, `RAGTIME_ARTIFACT_PREFIX`, `RAGTIME_ARTIFACT_CACHE_DIR`, optional `HF_TOKEN`, and optional `HF_ENDPOINT`. Build `fsspec.filesystem("simplecache", target_protocol="hf", ...)`, verify private bucket once, and default L1 to mode-0700 `/tmp/trec-ragtime-artifacts`.

- [ ] **Step 3: Write failing definition tests**

Assert the endpoint factory receives:

~~~python
{
    "workers": (1, 1),
    "max_concurrency": 1,
    "idle_timeout": 300,
    "volume": volume,
    "env": env,
}
~~~

Require pinned sentence-transformers, transformers, tokenizers, torch, msgspec, packaging, fsspec, and huggingface-hub dependencies.

- [ ] **Step 4: Extend Flash configuration**

Support `workers: int | tuple[int, int]` with Flash's integer shorthand `(0, N)` and tuple validation `0 <= minimum <= maximum`, `maximum >= 1`. Add `max_concurrency=1` and explicit `volume`/`env` passthrough to `build_flash_worker`. The smoke definition passes a concrete `GpuType`, not a heterogeneous GPU group, and sets `HF_HUB_CACHE=/runpod-volume/models` on the attached NetworkVolume. Transport settings and secrets never affect scorer identity.

- [ ] **Step 5: Verify without external effects**

Run: `uv run python -m unittest tests.test_flash_passage_scorer.FlashDecorationTests tests.test_flash_passage_scorer.StaticFlashWorkerTests -v`

Expected: PASS without importing/provisioning Runpod, touching Hugging Face, or downloading a model.

- [ ] **Step 6: Commit**

~~~bash
git add src/trec_ragtime/flash_worker.py src/trec_ragtime/flash_passage_scorer.py tests/test_flash_passage_scorer.py pyproject.toml uv.lock
git commit -m "feat: configure durable scalable Flash score workers"
~~~

### Task 5: Integration, Documentation, and Offline Verification

**Files:**
- Modify: `tests/test_passage_reranking_integration.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces: documented dumb-agent usage and full offline evidence.

- [ ] **Step 1: Add public fake end-to-end test**

Run `rerank_retrieval()` twice through `RunpodFlashPassageScorer` and an in-process worker with memory store. Require byte-identical reranking, one model inference sequence total, cold write, warm hit, and no cache metadata in public score output.

- [ ] **Step 2: Update README**

Keep caller usage as ordinary `rerank_retrieval(..., scorer=flash_scorer)`. Explain endpoint-owned exact-request caching/internal batches, `workers=(1, 1)` for one warm pod, and increasing max workers without caller changes. Remove client-wave claims.

- [ ] **Step 3: Run focused offline verification**

~~~bash
sops decrypt --output .env config/secrets.sops.env
git check-ignore .env
uv run python -m unittest \
  tests.test_passage_scoring \
  tests.test_passage_score_cache \
  tests.test_flash_passage_scorer \
  tests.test_passage_reranking_integration -v
~~~

Expected: PASS with no organizer, Hugging Face, Runpod, or model download.

- [ ] **Step 4: Run the full repository suite**

~~~bash
uv lock --check
uv run python -m unittest discover -s tests -v
git diff --check
git status --short
~~~

Expected: PASS; only intentional files and pre-existing untracked `.flash/` appear.

- [ ] **Step 5: Commit**

~~~bash
git add README.md tests/test_passage_reranking_integration.py
git commit -m "docs: document endpoint-owned passage score caching"
~~~

### Task 6: Ephemeral Flash Smoke and Batch Measurement

**Files:**
- Create: `scripts/flash_cached_passage_smoke.py`
- Create: `docs/experiments/2026-08-11-flash-cached-passage-smoke.md`

**Interfaces:**
- Consumes: completed worker/scorer, authorized Runpod/Hugging Face credentials.
- Produces: sanitized cold/warm/concurrency/batch measurements and teardown evidence; no stable endpoint.

- [ ] **Step 1: Write smoke driver and offline fixture test**

Use fixed synthetic passages only. Accept `--base-url`, `--batch-size`, `--mode {cold-warm,concurrent}`, and `--dry-run`. For `flash dev`, post the queue envelope to the namespaced local `/runsync` proxy URL discovered from the log; that proxy dispatches to the remote worker. Output only cache status, row count, batch count, elapsed milliseconds, GPU type, and peak allocated memory. Dry-run validates hashes and prints no environment values or source text.

- [ ] **Step 2: Verify dry-run**

Run: `uv run python scripts/flash_cached_passage_smoke.py --dry-run --batch-size 32`

Expected: exit 0 with sanitized counts.

- [ ] **Step 3: Start bounded `flash dev` in background**

~~~bash
flash dev > /tmp/trec-ragtime-flash-cached.log 2>&1 &
~~~

Poll at most two minutes for the actual route. On provisioning 401, report the credential blocker without printing the key. Never run `flash deploy`.

- [ ] **Step 4: Run cold, warm, and internal-batching checks**

For batch sizes 32, 64, and 128, submit the same synthetic request twice. Require cold `miss_written`, warm `hit`, warm `batch_count == 0`, complete equal coverage, and no warm inference log. Stop after the first OOM and retain the fastest successful batch.

- [ ] **Step 5: Run bounded two-worker duplicate check**

Use an isolated prefix and temporary `workers=(2, 2)`. Submit one uncached request twice concurrently. Require complete equal coverage, statuses from `{miss_written, miss_raced, hit}`, then a third `hit`. Cap at two workers and one request pair.

- [ ] **Step 6: Prove restart persistence**

Restart the ephemeral worker and require the same request to hit the durable score artifact. Use one miss-only control request to verify the NetworkVolume model cache is reused.

- [ ] **Step 7: Tear down exact dev resources**

Stop `flash dev`. Inspect the exact dev app/endpoint; if it persists, delete only that app with `flash app delete <exact-dev-app-name>`. Verify unrelated endpoints are untouched.

- [ ] **Step 8: Write sanitized report**

Record commit, Flash version, scorer fingerprint prefix, exact GPU, successful batch sizes, timings, peak memory, duplicate outcome, restart result, and teardown. Exclude endpoint bearer URLs, tokens, passages, raw logs, and bucket identifiers.

- [ ] **Step 9: Final verification and commit**

~~~bash
uv run python -m unittest discover -s tests -v
git diff --check
rg -n "(sk-|hf_[A-Za-z0-9]|Bearer |RUNPOD_API_KEY=|HF_TOKEN=)" \
  scripts/flash_cached_passage_smoke.py \
  docs/experiments/2026-08-11-flash-cached-passage-smoke.md
~~~

Expected: tests PASS, diff clean, secret scan has no matches.

~~~bash
git add scripts/flash_cached_passage_smoke.py docs/experiments/2026-08-11-flash-cached-passage-smoke.md
git commit -m "test: verify Flash-cached passage scoring"
~~~
