# Retrieval Artifact Cache Design

## Goal

Store every exact, complete hydrated retrieval as one strict MessagePack object
through the issue #12 `ArtifactStore`, then make Control-A and Baseline B/C
reuse those objects across processes, worktrees, machines, and remote workers
without organizer or DuckDB work on a warm hit.

## Selected architecture

Three approaches were considered:

1. **Automatic cache-aside inside `RagtimeRetriever.retrieve()` (selected).**
   This centralizes the complete semantic identity and successful-only rule,
   covers every runner, and gives callers the same `RetrievalResult` API.
2. Runner-specific cache wrappers. This duplicates identity and completeness
   rules across Control-A and B/C and leaves non-runner callers uncached.
3. Cache the existing mutable JSONL run files. This prevents natural
   cross-machine merging, retains partial/error records, and violates the
   required one-query object layout.

`RetrievalArtifactCache` owns typed wire conversion and validation.
`RagtimeRetriever` owns request construction and cache-aside compute.
Control-A/B/C own lightweight deterministic references and topic-level
assembly. The canonical local corpus remains the only hydration source on a
miss and the only hydration source used by importers.

## Semantic identity

`RetrievalRequestIdentity` is a frozen strict type built before organizer or
corpus work. The retriever fingerprint covers:

- retrieval schema version `1`;
- normalized RAGTIME base-URL digest;
- explicit client/retriever ranking, hydration, fallback, and translation
  behavior version constants;
- canonical local corpus dataset, config, pinned revision, schema, source
  splits, translation config/splits, shard/document/translation counts.

The machine-local corpus path is excluded.

The request hash covers:

- exact UTF-8 query bytes;
- exact pipeline/RoutIR string;
- requested `top_k`, including `None`, and the endpoint-depth/truncation
  behavior version;
- `max_content_fallbacks`, `fallback_on_hf_error`, and content collection;
- Dataset Viewer fallback/backend policy;
- a canonical optional `query_context` for derived requests.

Control-A passes `{"kind":"control_a","query_construction":"problem_only_v1"}`.
Baseline B/C pass their query construction version, baseline kind, query-plan
SHA-256, and one-based query index. RRF `k` is topic-fusion provenance and is
not part of an individual raw facet retrieval identity. Credentials, cache
paths, worker IDs, timestamps, concurrency, run IDs, and refresh flags are
excluded.

The layout is:

```text
retrieval/v1/<retriever-fingerprint>/<first-two-request-hash-chars>/<request-hash>.mpk
```

## Typed artifact and validation

The envelope and all nested wire records are frozen strict `msgspec.Struct`
types. Binary identity and content hashes stay bytes. The payload preserves
ordered `PipelineResult` fields and hits, ordered `RetrievedDocument` fields,
canonical source text byte-for-byte, optional English text, hydration failures,
unresolved IDs, missing translation IDs, and source/cache provenance needed to
reconstruct domain dataclasses.

Validation rejects wrong schema/fingerprint/request, non-finite times/scores,
noncontiguous or duplicate ranks, duplicate hit/document IDs, hit/document
count or order mismatch, malformed optional metadata, source text loss,
unresolved documents, operational hydration failures, hidden partial/error
records, and malformed MessagePack. Missing English translation is valid only
when canonical source hydration succeeded and the ID is listed explicitly.

`get(identity)` reads through `ArtifactStore.read_validated`, strictly decodes,
validates identity and result, and reconstructs the domain result without
organizer or corpus access. `put(identity, result)` validates, encodes once,
writes, and returns the original logical result without rereading. Missing is a
cache miss; every other store failure is fatal.

## Retriever integration

`RagtimeRetriever` receives an optional `RetrievalArtifactCache`. `from_env()`
shares the client's configured `ArtifactStore`. Before normal search/hydration,
`retrieve()` validates arguments, builds the complete identity, and reads the
retrieval artifact unless refresh is requested. A hit returns the reconstructed
result. A miss executes the current search → deterministic truncation → local
hydration → bounded content fallback pipeline, then caches only if all returned
IDs are hydrated, no operational hydration failure exists, and the final result
passes artifact validation.

`refresh=True` bypasses retrieval-artifact reads and refreshes the underlying
search/documents. Existing `refresh_search` or `refresh_documents` also bypass
the retrieval-artifact read. Refresh controls do not affect identity, and a
failed refresh cannot overwrite a prior complete artifact.

Returned results expose lightweight retrieval-artifact provenance (object key,
request digest, retriever fingerprint, and whether the result came from the
retrieval cache). The miss path constructs this provenance from the validated
write without rereading the object.

## Control-A/B/C integration

Control-A supplies the stable query context, uses the normal retriever path,
and records one artifact reference per topic status. Candidate ordering and
serialization remain unchanged.

Baseline workers keep one query per retriever and the existing bounded waves.
Each successful query checkpoint records only the exact query, hits, and an
immutable retrieval-object reference with lightweight provenance. The main
thread remains the sole checkpoint/manifest writer, and output ordering remains
query-plan deterministic.

Topic reconstruction no longer calls `enrich_documents`. For every successful
query checkpoint it performs a cache-only retrieval-artifact load using the
reconstructed identity, verifies the checkpoint reference, and uses the cached
documents to materialize B ordering or C fused candidates. C verifies that any
same-ID document appearing across facets has identical canonical metadata and
text. Thus an L1/L2 warm run does no organizer request and no DuckDB hydration.

Legacy successful checkpoints without references are not silently trusted.
The explicit importer populates deterministic objects; new runs discover them
by identity. Existing mutable runs remain unchanged.

## Existing-run importer

`ragtime-retrieval-import` accepts the known `control_a_v1` and
`query_baseline_v1` run schemas. It hashes and validates manifests and JSONL
inputs, reconstructs exact query contexts and identities, validates rank and ID
alignment, and fills missing passage fields only with the pinned read-only
`LocalCorpusIndex`. It never calls the organizer, `/content`, or Dataset Viewer.

Control-A imports complete topic query groups. Baseline B/C import complete
successful query checkpoints; their hit IDs are hydrated directly from the
local corpus, including facet hits not present in fused topic output. Failed or
partial query checkpoints are skipped and counted. The importer writes one
query object per complete request, independently rereads it, and emits a
sanitized local manifest containing source file hashes, corpus identity,
imported/verified/skipped counts, and object identities without query text,
document text, credentials, or raw payloads.

## Verification

Offline tests cover stable cross-path identities, every identity-changing
input, strict nested round trips, corruption and partial rejection, L1/L2/miss,
refresh, no write on organizer/hydration failure, explicit missing
translations, different-query concurrency, same-key last-writer-wins,
Control-A/B/C ordering/checkpoint semantics, cache-only topic reconstruction,
and importer behavior without participant requests.

After the #12 canonical response import, bounded live checks issue one cold
query, verify its retrieval object in L2, remove only its disposable L1
candidate, verify an L2 hit without organizer/DuckDB work, then verify an L1
hit with all network access disabled. Existing A/B/C import is bounded by the
known completed artifacts and is reconciled by sanitized counts and sampled
byte/rank checks.
