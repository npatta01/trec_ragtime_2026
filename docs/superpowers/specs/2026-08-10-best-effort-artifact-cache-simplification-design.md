# Best-Effort Artifact Cache Simplification Design

## Goal

Simplify the issue #12/#13 cache implementation for competition use. Cache
availability must never decide whether a validated participant request or a
Control-A/B/C run can finish. Preserve exact request identities, strict typed
MessagePack validation, private HF storage, complete-only retrieval writes,
and the existing DuckDB corpus.

This design intentionally revises the original issue #12/#13 rule that every
non-miss HF failure is fatal. Runtime cache reads and writes become best
effort. The one-time importers still detect and report unsuccessful writes and
verification misses rather than claiming migration success.

## Module Shape

Keep three modules with distinct responsibilities:

1. `ArtifactStore` owns private HF Bucket plus `fsspec.simplecache` byte I/O.
2. `ResponseCache` owns response request identity, TTL, and response envelopes.
3. `RetrievalArtifactCache` owns exact retrieval identity, retrieval envelopes,
   and complete-hydration validation.

Do not introduce a universal cache repository, codec/spec framework, or a
single combined response/retrieval envelope. Those would relocate rather than
remove complexity.

The public package interface remains the high-level `RagtimeClient` and
`RagtimeRetriever` interfaces plus domain models. Storage, wire, digest,
version, and compatibility names are internal implementation details unless a
repository importer directly requires them.

## ArtifactStore Runtime Policy

`ArtifactStore` retains this compact interface:

```python
read(object_key: str) -> bytes | None
read_validated(object_key: str, decoder: Callable[[bytes], T]) -> T | None
write(object_key: str, payload: bytes) -> bool
evict_local(object_key: str) -> None
```

The implementation follows this flow:

- Inspect the deterministic simplecache L1 candidate first. A valid L1 hit
  performs no HF network call.
- On an L1 miss, use normal `simplecache` reading. `FileNotFoundError` is a
  normal miss without a warning or circuit-breaker change. HF may mask an
  availability failure as that miss; the subsequent best-effort write will
  expose the failure and disable remote caching for the process.
- Other HF authentication/authorization/network/server failures, bucket
  inspection failures, and filesystem failures become a cache miss, warn
  once, and disable remote caching for that store instance.
- Emit one sanitized `RuntimeWarning` per store instance on the first remote
  availability failure. Never include a token, bucket path, query, response,
  or exception body.
- After a remote availability failure, disable remote reads and writes for
  that store instance. Existing valid L1 files remain readable. A new process
  may try HF again after the operator fixes the problem.
- Verify the bucket is private before the first remote read or write. A public
  bucket disables the cache and never receives a write.
- If a decoder rejects an L1 candidate with `ArtifactValidationError`, evict
  that exact candidate and try L2 once.
- If the downloaded L2 candidate also fails validation, evict it and return a
  miss. A later fully validated live result may replace the bad remote object.
- `write` returns `True` only after simplecache closes successfully and retains
  the local candidate. On failure it removes the exact partial candidate,
  warns/disables remote caching, and returns `False`.
- Keep the cache directory mode `0700`. Per-file `0600` enforcement and
  repeated symlink/status checks are unnecessary inside that owner-only
  directory; reject a symlinked cache directory at construction.

Remove exact `get_bucket_paths_info` miss classification, direct use of
`_get_cached_file_before_open`, listing-visibility retries, target-cache
invalidation, the fsspec upper pin tied to that private method, and fatal
runtime transport propagation.

Invalid caller configuration and invalid object keys remain programming
errors. Runtime construction through `from_env` catches cache-only
configuration failures at the high-level client/retriever factory, warns once,
and continues with caching disabled.

## Response Cache Flow

`ResponseCache.get` returns the validator's typed result with `stored_at`, not
the raw mapping that callers must validate a second time:

```python
CacheEntry[T](value: T, stored_at: float)
```

On a hit, decoding, canonical re-encoding, identity checks, domain validation,
and TTL evaluation still occur in that order. An invalid artifact becomes a
miss through `ArtifactStore.read_validated`.

On a miss, the caller performs the participant or Dataset Viewer request,
fully validates it, then invokes `ResponseCache.put`. `put` returns the stored
timestamp on success and `None` when the best-effort write did not persist.
Failed live requests and malformed responses are never passed to `put`.

Remove the client's pass-through `_cache_get`/`_cache_put` methods and the
second validator invocation. Use one canonical `search` method; remove the
unused `pipeline` alias.

## Retrieval Cache Flow

Keep the current object keys and MessagePack bytes compatible. Identity,
fingerprint, corpus revision, query context, ranking, hydration, source text,
translation, and provenance checks remain unchanged.

`RetrievalArtifactCache.put` returns a result with a
`RetrievalArtifactReference` only when `ArtifactStore.write` returns `True`.
When a best-effort write fails, it returns the original complete validated
result without a durable reference. Partial or operationally failed retrievals
remain uncacheable.

Remove unused version aliases, wire aliases, request-wire exports, identity
aliases, reference aliases, package-root cache exports, and the retriever's
pass-through `retrieval_identity`/`retrieve_cached` methods. Keep one identity
builder for importers and one cache-only retrieval method for resume.

## Control-A/B/C Continuity

Control-A already consumes retrieval results immediately. A complete result
without a durable artifact reference remains a successful candidate result;
its manifest omits the optional reference.

Baseline B/C must retain newly computed `RetrievalResult` objects in memory
until their topic is assembled. Topic assembly uses the in-memory result first
and uses cache-only lookup only for successful checkpoints from an earlier
process. A successful checkpoint without a durable reference is not reusable
and is recomputed on resume.

Remove runner-side reconstruction and comparison of an identity that the
retrieval cache already validated. Keep checkpoint-to-artifact comparison when
loading prior durable references, because that protects resume correctness.

Artifact-store failures no longer require systemic runner exception handling
or failed-manifest finalization. Participant, hydration, and corpus failures
retain the existing runner behavior.

## Importers

Keep `cache_import.py` and `retrieval_import.py` isolated from runtime and keep
SQLite confined to `cache_import.py`.

Importers treat `write == False`, a verification miss, or a mismatched decoded
artifact as an import failure. They never increment imported/verified counts
for a best-effort failure. Existing object layouts, source read-only checks,
source hashes, idempotency, and sanitized manifests remain unchanged.

## Tests

Use TDD and preserve wire-compatibility fixtures. Replace fatal-runtime tests
with these observable contracts:

- valid L1 hit performs no HF call;
- missing or unavailable L2 returns a miss and warns once;
- first remote failure disables later remote reads and writes but preserves L1;
- public bucket never receives a write;
- corrupt L1 retries L2 once; corrupt L2 becomes a miss;
- write failure returns `False` and removes its partial candidate;
- only validated successful responses/results invoke `write`;
- response hits are domain-validated exactly once;
- retrieval write failure returns the complete result without a reference;
- Control-A/B/C finish from in-memory validated results while cache is down;
- a resumed no-reference query is recomputed;
- importers refuse to count a failed write or verification miss;
- existing request keys, fingerprints, MessagePack fixtures, ordering, and
  DuckDB publication checks remain unchanged.

Run the offline suite after decrypting `.env`, then run one bounded private
bucket L2-to-L1 read and one best-effort failure canary. Do not call the
participant endpoint during cache-failure testing.

## Success Criteria

- Cache failure cannot abort an otherwise valid client request or Control run.
- Bad or partial data is never cached as success.
- Public storage never receives artifacts.
- Existing private artifacts remain readable without migration.
- Runtime contains no SQLite response-cache path and DuckDB remains unchanged.
- Public cache concepts and duplicate runner logic are materially reduced.
- All offline and bounded integration checks pass.
