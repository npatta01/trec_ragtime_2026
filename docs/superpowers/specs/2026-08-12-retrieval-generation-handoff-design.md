# Retrieval-to-Generation Handoff Design

## Goal

Produce one immutable, self-contained retrieval handoff bundle per topic that
report generation can consume by opening exactly one artifact. The bundle is
the final output of retrieval, reranking, and internal nugget compression. It
contains only evidence selected by retained nuggets, including the exact
English passage text needed for synthesis and citation validation.

This is not another generation-input transformation stage. It is the stable
consumer-facing output contract of the retrieval pipeline.

## Scope

The handoff builder:

- consumes one validated sealed nugget bundle from issue #16;
- resolves that bundle's exact retrieval artifacts without issuing new
  organizer requests or hydrating from another corpus source;
- copies only evidence referenced by retained nuggets;
- validates every alias, offset, content hash, representation, and upstream
  identity;
- writes one complete typed MessagePack artifact per topic; and
- exposes a cache-only reader for report generation.

It does not call a report model, write report prose, perform new retrieval,
change nugget clustering, or produce an Autonuggetization submission.

## Selected architecture

Three approaches were considered:

1. **A derived self-contained retrieval handoff artifact (selected).** Resolve
   the sealed nugget bundle against its immutable retrieval inputs once, copy
   the selected evidence text, and validate the complete result before writing.
   Generation receives one small artifact with no join logic.
2. Add passage text to the nugget artifact. This would make nugget caches
   larger, change their compact lineage contract, and couple an internal
   compression artifact to one downstream consumer.
3. Make generation join nugget and retrieval artifacts at runtime. This avoids
   a new artifact but duplicates identity, text-resolution, and validation
   logic in every generator.

The derived artifact preserves the compact nugget cache while making the
retrieval boundary simple and difficult to misuse.

## Pipeline boundary

```text
exact retrieval artifacts
          +
cached passage reranking
          +
sealed deduplicated nugget bundle
          |
          v
self-contained retrieval handoff bundle
          |
          v
report generation
```

Report generation reads only the topic record, selected evidence, nuggets, and
lineage contained in the handoff. It must not reopen raw retrieval artifacts,
query the organizer, hydrate the corpus, or inspect provider/package events.

## Bundle schema

Use frozen strict `msgspec.Struct` records as both domain and wire types. The
top-level `RetrievalHandoffBundle` contains:

- schema version and handoff behavior version;
- the complete official topic record: `topic_id`, `collection_id`, `title`,
  `problem_statement`, `background`, and the maximum normalized report
  character `limit`;
- the source representation, fixed to English `report_text` in version 1;
- ordered selected evidence records;
- ordered deduplicated nugget records;
- facet/query coverage and provenance;
- exact upstream retrieval and nugget artifact references;
- sanitized selection and rejection counts; and
- no mutable status or timestamp fields: presence of a valid bundle means the
  topic handoff is complete, while execution status belongs in the manifest.

### Selected evidence record

Each `HandoffEvidence` record contains:

- the stable evidence alias from the nugget bundle;
- original document ID;
- exact selected English passage text;
- representation and language;
- exact character offsets into the selected source representation;
- SHA-256 content hash of the passage text;
- passage score and source document rank;
- originating facet index and facet query; and
- the immutable retrieval artifact reference that owns the source document.

Aliases are not renumbered during projection. Preserving them makes nugget
support links directly resolvable and prevents the exporter from silently
changing citation identity.

### Nugget record

Each `HandoffNugget` contains:

- stable nugget and cluster IDs;
- the canonical atomic English claim;
- normalized merged variants;
- `vital` or `okay` importance;
- ordered evidence links; and
- the facet coverage inherited from the sealed nugget bundle.

Each evidence link preserves:

- evidence alias;
- `support` or `partial_support` assignment;
- the nugget variant index against which that passage was actually judged; and
- `citation_ready`, which is true only for full `support` by default.

`partial_support` evidence remains available to generation as labeled context,
but is not citation-ready under the default policy. A later generator may use
it to understand the topic, but it must not silently promote it to a fully
supporting citation.

## Selection rule

The output evidence table is exactly the ordered union of aliases referenced
by retained nuggets. Evidence offered to nugget extraction but not linked to a
retained nugget is omitted. This implements "selected evidence only" and keeps
the generation context bounded.

An alias referenced by both full and partial links appears once. Its links keep
their individual assignments. The builder does not perform another semantic
deduplication pass: nugget clustering and exact passage-location deduplication
remain owned by the upstream stages.

## Source resolution and validation

The builder begins with a validated nugget artifact and its recorded exact
inputs. It loads those retrieval objects cache-only and builds a document table.
For every selected evidence row it must prove:

1. the alias exists exactly once in the nugget evidence table;
2. every retained support link names a known selected alias;
3. the owning retrieval artifact is one of the nugget request's recorded
   inputs;
4. the document ID resolves exactly once, or resolves identically wherever it
   appears across facets;
5. English `report_text` exists and matches the bundle representation;
6. offsets are ordered and remain within the source text;
7. slicing the source text at those offsets equals the copied passage text;
8. SHA-256 of the copied UTF-8 text equals the stored content hash;
9. scores are finite and ranks and facet indices are valid; and
10. all topic, query, facet, retrieval, nugget, and corpus identities agree.

Any disagreement fails the topic without writing an artifact. There is no
fallback to the local corpus, Dataset Viewer, participant API, or unrestricted
document context.

The final bundle validator repeats all checks possible from the self-contained
artifact: strict decoding, canonical encoding, unique/canonical aliases,
support-link closure, offsets, copied-text hashes, finite values, stable IDs,
ordering, topic identity, and citation-readiness policy. Checks requiring the
upstream source are performed before construction and bound into the request
identity through the upstream artifact references and evidence digest.

## Identity, cache, and storage

Store bundles under:

```text
retrieval-handoff/v1/<handoff-fingerprint>/<request-prefix>/<request-digest>.mpk
```

The handoff fingerprint covers every non-topic-specific behavior:

- schema and behavior versions;
- projection and evidence-selection policy versions;
- support-to-citation-readiness policy;
- source representation policy; and
- validation rules that can change accepted output.

The request identity covers:

- exact official topic record and topic-catalog hash;
- exact nugget artifact object key, fingerprint, request digest, and content
  digest;
- all exact retrieval artifact references used by that nugget request;
- ordered selected alias/evidence digest; and
- the topic's report character limit.

Credentials, local paths, worker IDs, timestamps, concurrency, and cache-hit
state are excluded.

A cache hit returns the complete validated bundle without opening upstream
artifacts. On a miss, the builder constructs and validates the bundle, writes
it, evicts its disposable L1 candidate, then independently rereads and validates
the durable L2 object. Only a confirmed absence is a miss; operational storage
failures are topic failures. Cache hits are the resume mechanism, so no custom
completed/missing prescan is required.

## Execution and manifest

Parallelize by topic. One worker owns exactly one handoff bundle. The builder
returns sanitized per-topic outcomes: `cached`, `written`, or `failed`, plus
object key and compact counts. It never writes raw passage text, topic text, or
credentials to logs or the manifest.

After all topics finish, write one atomic owner-only manifest in official topic
order. The manifest includes:

- handoff schema and fingerprint;
- topic catalog hash;
- one outcome per required topic;
- counts for topics, nuggets, selected evidence rows, full-support links,
  partial-support links, and citation-ready links; and
- sanitized failures.

The manifest is successful only when every required topic has a validated
bundle. Aggregate report generation must refuse a missing or failed topic under
strict mode.

## Consumer API

Expose a small cache-only interface rather than storage internals:

```python
bundle = handoff_cache.get(fingerprint, request)
```

Generation receives `RetrievalHandoffBundle` and needs no knowledge of nugget
cache envelopes, retrieval cache wire records, reranking objects, or corpus
hydration. A convenience iterator may yield citation-ready nugget/evidence
pairs, but it must preserve access to labeled partial-support context.

## Error handling

- Unknown or duplicate aliases fail closed.
- Missing retrieval inputs, unresolved documents, representation drift,
  offset/hash mismatch, and upstream identity disagreement fail closed.
- A nugget without any evidence link is invalid.
- A bundle containing no retained nuggets or no selected evidence is invalid.
- Partial support is retained and labeled, never promoted implicitly.
- Corrupt local artifacts may be repaired once from valid L2; malformed L2 is
  fatal.
- Provider, organizer, or corpus calls are forbidden in this stage.
- Error records are sanitized and bounded; passage and topic contents never
  appear in them.

## Testing

Unit and integration tests cover:

- projection of one sealed nugget fixture into one self-contained handoff;
- exact selected-alias union and omission of unused evidence;
- preservation of evidence alias, document ID, passage text, offsets, hash,
  rank, score, facet, query, and retrieval reference;
- full versus partial support and default `citation_ready` behavior;
- preservation of clustered variants and variant-specific judgment lineage;
- duplicate/unknown aliases and support-link closure;
- wrong document, representation, facet, topic, or retrieval identity;
- out-of-bounds offsets, altered source text, and hash mismatch;
- same-document disagreement across facet retrievals;
- strict/canonical MessagePack round trips and stable bytes;
- isolation for every fingerprint and request field;
- cold write, L1 eviction, independent L2 reread, warm L1/L2 hits, confirmed
  miss, and operational storage failure;
- no organizer, Dataset Viewer, corpus hydration, or provider calls;
- parallel different-topic ownership and cache-first resume;
- atomic owner-only manifest and deterministic official topic order;
- missing/failed topic rejection in strict aggregate mode; and
- secret and raw-payload exclusion from artifacts, manifests, and errors.

Run a final live validation against the completed issue #16 corpus: require one
handoff bundle for every official topic, independently verify every copied
passage hash and upstream reference, evict and reread sampled objects from L2,
and prove a warm full pass makes no retrieval, hydration, or provider calls.

## Dependency and rollout

Implementation begins after the issue #16 schema-v2 nugget bundle is merged.
Version 1 of this handoff requires its variant-specific support lineage and
carries it directly. The initial implementation targets the fixed
retrieval/reranking producer only. A future agentic producer may emit the same
handoff contract, but agentic traces and producer-specific state stay outside
this artifact.

Issue #17 report generation should consume this bundle directly. Its official
submission JSONL remains a separate downstream artifact.

## Acceptance criteria

- One self-contained retrieval handoff bundle is produced per official topic.
- Generation can use it without reading any other retrieval or nugget artifact.
- It contains only evidence referenced by retained deduplicated nuggets.
- Exact passage text, document identity, offsets, hashes, scores, ranks, facets,
  queries, and upstream references are preserved and validated.
- Full and partial support remain distinct; only full support is citation-ready
  by default.
- Invalid or incomplete output is never cached as successful.
- Exact input/configuration changes isolate cache identities.
- Warm reuse performs no organizer, corpus, reranking, nuggetizer, or provider
  work.
- The aggregate manifest proves complete topic coverage and remains sanitized.
- Report generation needs no upstream-format-specific join logic.
