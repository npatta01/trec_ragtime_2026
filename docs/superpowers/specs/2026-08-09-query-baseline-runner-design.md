# Query Baseline Runner Design

## Goal

Produce auditable Baseline B and Baseline C retrieval artifacts for all 103
corrected 2026 topics without changing the agreed query definitions.

## Inputs and invariants

- Read the existing `query_plans.jsonl`; do not generate new queries.
- Baseline B uses exactly one `single_query` per topic.
- Baseline C uses exactly the ordered `facet_queries` per topic.
- Do not use `title` or `collection_id` in query construction.
- Call `pipeline="ragtime2"` and apply `top_k=100` through
  `RagtimeRetriever.retrieve`. Do not use a RoutIR depth suffix.
- Use the shared git-common-dir cache and write artifacts beneath it.

## Architecture

Add a focused `query_baselines` module with three boundaries:

1. Load and validate immutable query plans.
2. Run and immediately checkpoint each exact query result.
3. Build one topic artifact from checkpointed queries, hydrate final candidates
   through the local corpus, and update an auditable manifest.

The runner accepts `b` or `c`. Baselines run sequentially so they do not
compete for organizer capacity. Existing successful cache entries for the exact
`ragtime2` request are reused by `RagtimeClient`; historical `ragtime2%100`
artifact records are not imported into the new baseline.

## Ranking

Baseline B preserves organizer rank and score for the first 100 documents.
Baseline C applies reciprocal-rank fusion over successful facet lists:

```text
fused_score(document) = sum(1 / (60 + rank))
```

Deduplicate by document ID, order by descending fused score and ascending
document ID for ties, then keep 100. Record the contributing one-based query
indices and best source rank.

## Checkpoint and artifact contract

Each run directory contains:

- `manifest.json`: configuration, source hashes, lifecycle status, and counts.
- `queries.jsonl`: one latest record per `(topic_id, query_index)` with exact
  query, pipeline, status, timing, hits or sanitized error, and cache metadata.
- `topics.jsonl`: one latest record per topic with status, query results,
  candidates, and hydration diagnostics.

Writes are atomic. On resume, successful query checkpoints are reused and
failed checkpoints are attempted again. Topic records are rebuilt from current
query checkpoints. A failed B query yields an `error` topic. C is `success`
when every facet succeeds, `partial` when at least one but not all succeeds,
and `error` when none succeeds. Failures are never silently replaced.

## Error handling and observability

- Use zero transport retries per run; explicit reruns provide retry policy.
- Catch retrieval/hydration errors per query and preserve only sanitized error
  type/detail.
- Print one concise progress line per topic.
- Never print document text, bearer tokens, or raw response bodies.

## Verification

Unit tests cover plan validation, resume semantics, B ranking, C RRF, explicit
partial/error states, atomic artifact replacement, and CLI defaults. Offline
tests use fakes; one live sanity request precedes expensive runs.
