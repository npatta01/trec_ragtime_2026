# Topic-Union Maximum-Need Scoring Design

## Objective

Build the richer retrieval arm's downstream candidate pool without reciprocal-rank fusion. For each topic, union all successful query results available in an immutable retrieval snapshot, deduplicate documents by canonical document ID, and score every unique document against every extracted information need. The resulting topic-level ranking feeds nugget extraction.

This does not alter Baseline C, which remains the historical facet-RRF comparison arm.

## Inputs and snapshot boundary

The producer consumes:

- the sealed 103-topic static plans and their ordered information-need ledger;
- the validated breadth-pass retrieval pools;
- validated successful checkpoints from the active full richer retrieval run; and
- the canonical local corpus used to hydrate those retrieval artifacts.

Each scoring wave records the exact set and hashes of its input retrieval objects. A running retriever may add later checkpoints, but an in-progress wave never observes them. A later wave creates a new request identity and reuses cached scores for unchanged need-passage pairs.

## Topic-level candidate construction

For each topic:

1. Read every validated successful query result included in the snapshot.
2. Union all returned candidates.
3. Deduplicate strictly by canonical `document_id`.
4. Require repeated IDs to resolve to identical canonical source content and compatible English report representations. A mismatch fails the topic rather than selecting one copy.
5. Preserve query, task, requirement, retrieval-object, and original-rank provenance as metadata only. None of these ranks contributes to the final score.

There is no RRF, rank averaging, per-query quota, per-need quota, or facet round-robin candidate selection.

## Scoring

Each unique document is chunked once with the repository's pinned passage chunker. Each unique passage is scored against every extracted information need for the topic using the existing passage-scoring boundary and cache.

For passage `p`, information need `n`, and document `d`:

```text
pair_score(p, n) = scorer(n.text, p)
document_score(d) = max pair_score(p, n) for every passage p in d and need n
```

The producer retains the winning information-need ID, passage offsets, content hash, and pair score. Exact score ties are resolved by canonical document ID; hosted-retrieval rank is not a tie-breaker. All finite scored documents form one descending topic-level ranking.

The score-cache identity includes the scorer fingerprint, exact need text, representation, and passage content hash. Later waves therefore score only previously unseen need-passage pairs.

## Downstream boundary

The output is one immutable topic candidate artifact containing:

- topic and plan identity;
- snapshot/input identities;
- unique-document and source-query counts;
- deterministic ranked documents;
- each document's maximum score and winning need/passage provenance; and
- sanitized failure and exclusion counts.

Nugget extraction consumes the topic-level ranking directly. It must not reconstruct query facets or apply RRF or round-robin evidence allocation. Passage and document limits, if needed for provider context, are applied only after global topic scoring and are recorded in the output fingerprint.

## Parallel execution

Retrieval continues with its existing two-worker owner process. Scoring runs in a separate owner-only output directory against a fixed snapshot. Topics are independent and may be scored concurrently, subject to the scorer's configured resource limits.

All 103 topics are eligible because the validated breadth pass includes at least one successful richer query per topic. The active full run contributes additional successful queries to 99 topics at the time of design. Later waves incorporate newly completed queries without mutating earlier artifacts.

## Failure handling

A topic fails closed when an input checkpoint is malformed, a retrieval object cannot be validated, duplicate document content disagrees, hydration is incomplete, a required representation is missing under the selected policy, or the scorer returns missing or non-finite values. One failed topic does not stop other topics.

Provider, retrieval, and scorer credentials never enter logs, payload identities, manifests, or committed artifacts. Failed or partial results are not upgraded to successful cache entries.

## Verification

Tests must prove:

- union and document-ID deduplication across multiple queries;
- duplicate-content disagreement fails closed;
- all needs are scored for every unique passage;
- document score is the maximum need-passage score;
- original query ranks and duplicate frequency cannot change the final ranking;
- deterministic document-ID tie-breaking;
- winning need and passage provenance is exact;
- warm waves make no scorer call for cached need-passage pairs;
- later snapshots score only newly introduced pairs; and
- nugget input contains one topic-level ranking and performs no RRF or facet round-robin selection.

Before a second submission, the resulting reports must pass the repository validator and the organizers' AutoJudge wrapper in strict mode.
