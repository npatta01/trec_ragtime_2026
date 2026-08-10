# HF Artifact Response Cache Design

## Goal

Replace the SQLite runtime response cache with immutable typed MessagePack
objects stored in a private Hugging Face Bucket, read through an owner-only
machine-local `fsspec.simplecache`, while leaving the canonical DuckDB corpus
unchanged and read-only.

## Locked boundaries

- `ArtifactStore` owns HF Bucket authentication and privacy checks, exact
  remote existence checks, `simplecache` construction, candidate resolution,
  L1 repair, uploads, local permissions, and sanitized storage errors.
- `ResponseCache` owns request identities, response object keys, typed
  MessagePack envelopes, recursive canonicalization, `stored_at`, TTL, and
  response identity/schema validation.
- `RagtimeClient` and `HuggingFaceCorpus` own endpoint/domain validation. They
  pass validators into `ResponseCache` so a semantically invalid L1 object is
  repaired from L2 once and a semantically invalid L2 object is fatal.
- The one-time SQLite importer is an offline command. No runtime path imports,
  reads, opens, or depends on SQLite.
- `LocalCorpusIndex` and its DuckDB files are out of scope.

## Considered approaches

1. **One deep store plus typed domain adapters (selected).** Preserve the
   requested `read(object_key) -> bytes | None` and `write(...)` API, and add a
   validated-read operation that accepts a pure decoder. This keeps HF and
   `simplecache` mechanics out of response and retrieval clients while giving
   the store enough information to repair one exact corrupt L1 candidate.
2. Separate L1 and L2 services coordinated by each domain cache. This makes
   domain caches understand bucket failure classification and duplicates the
   repair algorithm, so it is rejected.
3. Put MessagePack and endpoint validation directly in `ArtifactStore`. This
   makes the storage module shallow and couples it to organizer and Dataset
   Viewer schemas, so it is rejected.

## Configuration

`ArtifactStore.from_env()` reads:

- `RAGTIME_HF_BUCKET_ID`: required `owner/bucket` identity for cached runtime
  use;
- `RAGTIME_ARTIFACT_PREFIX`: optional normalized relative object prefix;
- `RAGTIME_ARTIFACT_CACHE_DIR`: optional absolute L1 directory, defaulting to
  `<git-common-dir>/ragtime-cache/artifacts-v1`;
- `HF_TOKEN`: optional explicit token passed to `HfApi` and `HfFileSystem`;
  when absent, the normal authenticated Hugging Face token lookup is used;
- `RAGTIME_CACHE_NAMESPACE`: response identity namespace, default
  `ragtime2-2026-v1`.

Non-empty `RAGTIME_CACHE_PATH` or `RAGTIME_CACHE_DIR` is rejected with a clear
configuration error. Neither is reinterpreted. `RAGTIME_PROJECT_CACHE_DIR`
continues to locate the project cache and DuckDB corpus. Local directories are
absolute, non-symlinked, and mode `0700`; retained artifact files are mode
`0600`.

`huggingface-hub>=1.5,<2` is the verified dependency floor: tag `v1.4.0` lacks
`HfApi.get_bucket_paths_info`, while `v1.5.0` contains both required Bucket
APIs. `fsspec` and `msgspec` become direct dependencies.

## ArtifactStore behavior

The remote path is
`hf://buckets/<owner>/<bucket>/<optional-prefix>/<object-key>`. Object keys are
strict relative POSIX paths without credentials, traversal, empty segments, or
query/fragment data.

Read behavior:

1. Resolve the exact `simplecache` candidate using the configured filesystem's
   mapper.
2. If the candidate exists, read and decode it locally. A valid hit performs
   no HF API or filesystem call.
3. If the decoder rejects the local bytes, remove only that candidate and
   attempt one authenticated L2 read.
4. Before any L2 operation, lazily call `bucket_info` once and require
   `private is True`.
5. Call `get_bucket_paths_info(bucket_id, [object_key])`. Only an empty result
   is a miss. Authentication, authorization, bucket, network, timeout, server,
   and protocol errors are fatal.
6. For an existing object, open it through `simplecache` once. A valid download
   remains in L1. Invalid downloaded bytes are fatal and remain distinguishable
   from a miss; they do not trigger upstream refresh or overwrite.

Write behavior:

1. The caller supplies bytes produced from a fully validated domain value.
2. Open the `simplecache` URL for binary write. Closing uploads the complete
   file and retains the local candidate.
3. On close/upload failure, remove the exact candidate and propagate a
   sanitized fatal exception.
4. Sequential and cross-machine duplicate valid writes are last-writer-wins.
   Different object keys may be accessed concurrently; no global query lock is
   added.

## Response identity and envelope

The object layout is:

```text
api-responses/v1/<operation>/<first-two-hash-chars>/<request-hash>.mpk
```

The SHA-256 request identity is computed from canonical MessagePack containing
schema version `1`, cache namespace, normalized base URL, operation, and the
exact recursively key-sorted request mapping. Authorization values and tokens
are never inputs.

`ResponseEnvelope` is a frozen, strict `msgspec.Struct` containing response
schema version, namespace, operation, SHA-256 base-URL digest bytes, SHA-256
request-identity digest bytes, finite `stored_at`, and the canonical validated
response mapping. Reads validate envelope identity and the endpoint/domain
payload before TTL evaluation. Expiry is a miss but does not delete the valid
old artifact. A later fully validated refresh may replace it.

`ResponseCache.get(...)` accepts the endpoint validator used for the live
response. That same validator is applied inside the validated artifact read.
`put(...)` encodes exactly once and returns `stored_at` without rereading.
Cache read and write errors are fatal; clients no longer warn and bypass the
shared cache.

## Runtime integration

`RagtimeClient._validated_request` retains this order: transport, successful
HTTP status, JSON object decode, endpoint validation, then cache write. Cache
hits are validated by the same endpoint validator before use. A failed live
refresh never calls `write` and therefore cannot replace an existing artifact.

`HuggingFaceCorpus` uses operation `huggingface_document`. It writes only rows
fully parsed from a complete Dataset Viewer response. The participant token is
never attached to Hugging Face requests or cache identities.

SQLite cache exports, `cache_path` forwarding, `clear/delete` runtime APIs, and
SQLite-specific settings are removed. Runtime metadata reports the artifact
bucket identity/prefix and local cache directory without recording tokens.

## Offline SQLite import

`ragtime-cache-import` opens the source with SQLite URI `mode=ro`, accepts only
schema version `1` and namespace `ragtime2-2026-v1`, and recognizes exactly
`pipeline`, `content`, and `huggingface_document`. For every row it decodes the
canonical request/response JSON, re-runs the production operation validator,
checks operation TTL, preserves finite `stored_at`, writes through
`ResponseCache`, then independently reads and validates the destination.

Invalid, corrupt, unsupported, noncanonical-namespace, and expired rows are
skipped and counted. The source database is never modified or deleted. The
local JSON migration manifest contains the source SHA-256, source size,
destination bucket/prefix/namespace, per-operation imported/skipped/verified
counts, and sanitized errors only—never request text, response bodies, URLs
with credentials, or tokens. The importer is idempotent and last-writer-wins.

## Verification

Offline tests use fake `HfApi` and fsspec-compatible filesystems to cover L1,
L2, miss, public/missing bucket, auth/network/server failure, upload cleanup,
one-shot repair, corrupt L2, deterministic bytes/keys, permissions,
concurrency, validation ordering, credential absence, and importer filtering.
The full decrypted suite must pass before bounded live checks.

Bounded live verification creates or selects one explicit private bucket,
writes and rereads disposable canary objects, verifies an L1 hit with HF access
disabled, imports the canonical 1,202 pipeline, 10 content, and 20 Dataset
Viewer rows, and independently verifies the migration manifest. It never
deletes the SQLite backup or DuckDB corpus.
