# Flash-Cached Passage Scorer Design

## Status

Approved direction from the user on 2026-08-11. The user also authorized an
ephemeral `flash dev` smoke, including a bounded two-worker concurrency check.
This does not authorize a stable deployment.

## Objective

Make the caller deliberately unaware of passage-score caching and batching.
The caller uses the existing synchronous `PassageScorer` contract:

```python
scores = scorer.score(query, passages)
```

For the remote path, that method makes one logical HTTP request. The Runpod
Flash endpoint owns request validation, cache lookup, GPU micro-batching,
complete response validation, durable cache publication, and cache telemetry.
The same endpoint must scale from one warm worker to several interchangeable
workers without requiring topic affinity or a coordinator.

Occasional duplicate GPU computation for simultaneous identical cold requests
is acceptable. Incorrect, partial, or cross-request cache reuse is not.

## External contract

`RunpodFlashPassageScorer` remains a normal `PassageScorer`. It receives a
query and passages, deduplicates exact passage content, encodes one request,
invokes the configured Flash endpoint, validates complete score coverage, and
returns `Mapping[bytes, float]`.

The public method exposes no cache flags, cache keys, batching controls, or
hit/miss branches. Cache behavior is an endpoint implementation detail. The
endpoint may return cache telemetry in its transport envelope for diagnostics;
the scorer consumes that metadata and returns only the score map.

The request contains:

- the exact scorer configuration and fingerprint;
- the exact query;
- unique passage rows containing UTF-8 text and its SHA-256 content hash;
- a wire schema version.

The worker rejects malformed hashes, duplicate hashes, unknown fields,
fingerprint drift, tokenizer overflow, and unsupported runtime versions before
using a cached value or invoking the model.

## Cache identity and artifact

Use one sealed artifact for the exact scoring request. Its identity is derived
from canonical MessagePack over:

1. a cache schema and behavior version;
2. the scorer fingerprint;
3. SHA-256 of the exact UTF-8 query;
4. the sorted unique passage content hashes.

The key is content-addressed and sharded by request digest:

```text
passage-score-requests/v1/<scorer-fingerprint>/<prefix>/<request-digest>.mpk
```

The artifact contains only the schema/behavior versions, scorer fingerprint,
query hash, ordered passage hashes, and finite scores. It stores neither query
text nor passage text. Encoding is canonical and deterministic.

Every cache read validates the envelope, exact request identity, complete hash
coverage, finite numeric values, and canonical bytes. A malformed artifact is
never a hit. Cache transport, permission, private-bucket, or malformed durable
artifact failures remain fatal rather than silently becoming expensive GPU
misses.

This exact-request layout replaces the current mutable per-query union for the
remote endpoint. It avoids the shared read/merge/write race across Flash
workers. Requests with partially overlapping passage sets do not share cache
entries; simplicity and safe horizontal scaling take precedence over partial
reuse at this boundary.

## Worker data flow

The self-contained class-style Flash worker performs:

1. validate and canonicalize the complete request;
2. derive the exact-request cache key;
3. read and validate the local L1 and shared private Hugging Face Bucket L2;
4. on a hit, return the sealed score map without loading or invoking the model;
5. on a miss, load the pinned cross-encoder once per worker if it is not warm;
6. score all rows in internal GPU micro-batches;
7. validate exact complete coverage and finite scores;
8. encode and publish the complete artifact;
9. independently reread and validate the durable artifact;
10. return scores plus non-sensitive cache telemetry.

The worker never calls the organizer retrieval endpoint. It receives only the
query and already-selected passage text needed for scoring.

## Batching and single-pod efficiency

Transport request size and GPU inference batch size are separate controls. One
HTTP job may carry many passages below Flash's strict 10 MB request limit. The
worker splits that request into model batches sized for VRAM, rather than the
client creating one endpoint job per model batch.

The initial production shape is one class worker with:

- one model/tokenizer loaded and retained per worker;
- `max_concurrency=1` for GPU inference;
- a zero minimum worker count so idle endpoints do not bill;
- a persistent NetworkVolume-backed `HF_HUB_CACHE` for cold-worker model reuse;
- a private worker-local artifact L1 and shared Hugging Face Bucket L2;
- an empirically selected model batch size, starting with 32 and probing 64 and
  128 under `flash dev` without changing any other result-affecting setting.

Scaling changes only the Flash worker range, for example `(0, 4)`. Each worker
is interchangeable and uses the same scorer fingerprint and shared durable
cache. No query or topic is pinned to a worker.

## Concurrent cold requests

Two workers may miss the same exact request and both run inference. Each must
produce a complete independently valid artifact for the same semantic key.
The shared object store may select the last complete writer. There is no
mutable union to lose, and no partial artifact is accepted.

After publishing, a worker rereads the durable object and returns the validated
winner. If supported cheaply by the durable backend, create-if-absent may be
used as an optimization, but a distributed lease or lock is explicitly out of
scope.

All workers use a pinned GPU type for a scored run rather than a heterogeneous
GPU group. This reduces hardware-dependent numerical drift between duplicate
writers. Precision and every result-affecting runtime setting remain part of
the scorer fingerprint.

## Failure handling and safety

- A valid hit never invokes or loads the model.
- A model failure, timeout, partial response, or non-finite score writes
  nothing.
- A cache operational failure fails the request with an allowlisted error code;
  it does not leak credentials, query text, passage text, provider bodies, or
  raw exception strings.
- Cancellation must stop outstanding endpoint work where Flash exposes job
  cancellation.
- Cache telemetry is limited to `hit`, `miss_written`, `miss_raced`, object key,
  row count, batch count, and timing fields. It contains no source text.
- Tokens enter the worker only through endpoint environment configuration and
  are excluded from request identity and artifacts.

## Components

### Local remote scorer

`RunpodFlashPassageScorer` becomes a one-request transport adapter. It retains
strict request/response validation and the synchronous-to-asynchronous bridge,
but removes client-side model-batch job waves.

### Worker-owned exact-request cache codec

The self-contained Flash worker is the sole owner of request-digest
construction, object-key construction, canonical artifact encoding, and typed
validation. The local caller neither imports nor duplicates this cache code.
Tests exercise the worker directly and execute its extracted class body with
project imports blocked, matching `flash dev` transport.

### Flash worker

`FlashPassageWorker` owns durable cache access and warm model inference. Its
constructor initializes only lightweight cache configuration; model loading
remains lazy so a durable cache hit does not allocate GPU model memory.

### Flash definition

`build_flash_worker` exposes explicit worker ranges, maximum per-worker
concurrency, NetworkVolume, cache environment, pinned GPU type, idle timeout,
and pinned dependencies. Constructing the definition remains side-effect free.

## Verification

Offline tests must prove:

- exact identity changes for query, passage set, scorer, schema, or behavior;
- canonical encoding and rejection of corrupt, partial, foreign, or non-finite
  artifacts;
- a warm exact request performs zero model calls;
- one cold request scores in internal micro-batches and writes once;
- simultaneous identical cold requests may both score but leave one complete
  readable artifact;
- distinct requests cannot cross-hit;
- transport size and model batch size are independent;
- public scorer output is unchanged and contains no cache metadata;
- worker codec behavior survives class-only transport with identical bytes;
- cache and worker errors contain no queries, passages, credentials, or raw
  exception payloads;
- focused passage-scoring tests and the complete offline suite pass.

The authorized ephemeral `flash dev` smoke then proves:

1. a cold request reports `miss_written` and returns complete scores;
2. an identical request reports `hit` with no model inference log;
3. a larger request uses multiple internal GPU batches in one endpoint job;
4. two simultaneous identical cold requests across a temporary two-worker
   range both return complete equal coverage and leave a readable artifact;
5. a worker restart reuses both the durable score artifact and persisted model
   files;
6. measured batch sizes 32, 64, and 128 identify the fastest non-OOM setting.

The smoke uses only synthetic/non-sensitive passages, records sanitized timing,
cache status, GPU type, and peak memory, and tears down the ephemeral Flash app
after verification. No stable deployment is created.

## Rejected alternatives

- **Caller-side cache checks:** rejected because they make the agent understand
  execution policy and duplicate endpoint behavior.
- **One immutable object per query/passage pair:** rejected because object-store
  round trips and object count dominate the modest partial-reuse benefit.
- **The existing mutable per-query score union inside every worker:** rejected
  because distributed read/merge/write races can discard entries.
- **A distributed lock or lease:** rejected because occasional duplicate cold
  computation is acceptable and exact-request artifacts make it unnecessary.
- **Client-side endpoint jobs per GPU batch:** rejected because it adds queue and
  HTTP overhead and prevents the worker from owning efficient micro-batching.
