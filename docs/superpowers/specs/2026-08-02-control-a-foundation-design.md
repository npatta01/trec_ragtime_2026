# Control-A Retrieval Foundation

## Objective

Create a small, reproducible organizer-retrieval control for all RAGTIME 2026 topics. The control must be a stable candidate/document artifact that later query-decomposition, reranking, and report-generation experiments can consume without re-running or changing the baseline acquisition step.

## Verified existing seam

`trec_ragtime.RagtimeRetriever.retrieve()` already provides the required end-to-end primitive:

- `pipeline="ragtime2%100"` requests the organizer's 100-result RoutIR variant.
- `top_k=None` preserves every result returned by that endpoint.
- `RagtimeRetriever.from_env()` requires the canonical shared local DuckDB corpus.
- `max_content_fallbacks=0` prevents private `/content` fallback from masking local corpus problems.
- `RetrievalResult` preserves organizer rank/score, canonical `text`, optional `report_text`, language, URLs, dates, hydration failures, unresolved IDs, and missing translation IDs.

No change to `client.py`, `retriever.py`, `local_corpus.py`, or the corpus schema is needed for Control-A.

## Scope

Add a topic-batch runner and pure serialization helpers, with tests:

- `src/trec_ragtime/topic_candidates.py`
- `tests/test_topic_candidates.py`
- a small command or experiment entry point for running the batch against the checked-in topic JSONL

The runner will read `topics/topics.all.2026.v0625.jsonl`, preserving source order and validating each record's `topic_id`, `collection_id`, `problem_statement`, `background`, and `limit`. The baseline query is deterministic and labeled:

```text
Problem: {problem_statement}

Background: {background}
```

The `limit` field remains topic metadata and is not inserted into the retrieval query.

The Control-A runner exposes the requested candidate depth as `top_k=100`.
The organizer protocol encodes that server-side depth in the pipeline string,
so the runner translates it internally to `ragtime2%100` and calls:

```python
retriever.retrieve(
    query,
    pipeline="ragtime2%100",
    top_k=None,
    max_content_fallbacks=0,
)
```

Here `pipeline="ragtime2%100"` is a transport detail: it asks the endpoint to
return up to 100 candidates. The retriever's `top_k` argument is a separate
client-side truncation applied after the endpoint response; setting it to
`100` cannot increase the default endpoint response from 20. Control-A uses
`top_k=None` internally so it retains the full server response, while its
manifest records the user-facing requested depth as `top_k=100`.

The runner will not decompose, rerank, fuse, or add a local retrieval index.

## Artifact contract

Write to the shared common-git cache, never to a tracked directory or a worktree-relative cache:

```text
<git-common-dir>/ragtime-cache/experiments/control-a/<run-id>/
  manifest.json
  candidates.jsonl
```

`manifest.json` records the run ID, topic source path and hash, Git revision, pipeline, query-construction version, corpus path and revision, topic count, candidate count, unresolved IDs, missing translations, and creation metadata. It must never contain credentials.

Each JSONL candidate record contains the topic ID, query hash, organizer rank and score, document ID, canonical source text, `report_text` when available, language, URL, date, content source, and hydration/cache status. Unresolved IDs are represented explicitly rather than silently discarded. Writes use a temporary file followed by an atomic replace so an interrupted run cannot look complete.

The artifact is deliberately self-contained for downstream offline work. Later stages may omit text from derived files, but the Control-A artifact retains it as the canonical snapshot of what was acquired.

## Reuse from `/home/npatta01/data/competitions/trec_rag_2026`

Reuse the external work's validated concepts, not its transport implementation:

- strict bounded facet-plan validation;
- stable IDs derived from source order;
- original-query-first semantics and fail-closed fallback;
- query hashes, provenance, and durable manifests;
- later round-robin/union selection for decomposition experiments.

Do not copy its OpenRouter planner, Pyserini retrieval, TREC 2025 identifiers, or cache/ledger contracts into this repository.

## Parallel handoff

Once this artifact contract is stable, independent work can own separate modules and directories:

1. Query decomposition consumes Control-A and adds frozen facet plans, per-facet organizer retrieval, and RRF/round-robin fusion.
2. Reranking consumes a frozen candidate pool and compares organizer order, BM25, multilingual cross-encoder, and English `report_text` reranking.
3. Out-of-pool retrieval remains a separate pilot and does not modify the canonical DuckDB index.
4. Report generation and submission validation consume derived artifacts only.

All live organizer acquisition should remain owned by one runner/cache namespace to avoid accidental duplicate calls. The parallel stages themselves should be deterministic and offline once their input artifact exists.

## Testing strategy

Use fake retrievers and temporary directories for unit tests. Cover:

- topic parsing and source-order preservation;
- deterministic query construction and query hashes;
- exact retrieval arguments, including `%100`, `top_k=None`, and zero fallback;
- rank/document alignment and explicit unresolved/translation status;
- manifest contents without secret values;
- atomic artifact output and refusal to overwrite an existing run;
- malformed topic and malformed retrieval input failures.

After implementation, decrypt `.env` before running the repository test suite and run the offline tests normally. A live batch run is a separate explicitly named operation and must use the existing shared cache.
