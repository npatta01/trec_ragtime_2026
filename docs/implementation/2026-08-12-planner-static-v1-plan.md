# planner-static-v1 implementation plan

## Objective

Build and evaluate a sealed, topic-specific static research planner between
the existing LLM-generated flat facet baseline and the later adaptive agentic
researcher. The first comparison holds the existing Baseline C query budget
constant per topic and changes only how the queries are allocated across
hierarchical information needs.

## Scope

- Add typed, schema-validated static plans and validated planner artifacts.
- Add deterministic breadth-first/round-robin task admission for matched
  Baseline C budgets and larger finite frontier caps.
- Add a retrieval executor seam that uses `RagtimeRetriever`, its existing
  retrieval artifacts, local hydration, source/English representation fields,
  and bounded passage lineage.
- Keep planning sealed before retrieval: no result-aware replanning,
  coordinator loop, follow-up search, or subresearcher execution.
- Reuse `provider_settings()` and the repository `ResponseCache`/artifact-store
  interfaces; do not add a second provider configuration namespace.
- Preserve existing RAG/MLIR export code and formats; this experiment produces
  private retrieval-stage artifacts only.

## Decisions

- Plan schema: `planner_static_plan_v2`; execution artifact schema:
  `planner_static_run_v2`.
- Stable IDs are derived locally from topic/need/task content and ordinal
  tie-breakers; model-supplied IDs are not trusted.
- A plan has variable requirements, needs, and tasks. Ordering round-robins
  needs in stable priority/order order. The richer arm retains every audited
  task; the matched arm takes the exact ordered `B_i` prefix.
- The default planner-call cap is 103, matching the current complete topic set;
  every runner rejects a larger invocation before making provider calls.
- Matched-budget execution requires exactly `B_i`, where `B_i` is the number
  of successful Baseline C query records for topic `i`; an insufficient sealed
  plan is an explicit operational failure, never an empty success.
- Passage output is grouped by information-need/task and preserves canonical
  document ID, source language/text, optional verified English text, document
  rank/score, representation kind, and exact character lineage. Nuggetization
  remains downstream.
- Topic aggregation is serial and plan-order deterministic even when query
  retrieval is later run concurrently.

### Superseding two-pass decision

- The richer arm is no longer truncated to `max_total_tasks=24`. A `None` cap
  retains every distinct task in the audited sealed plan; the provider call
  remains bounded by finite output tokens and timeout.
- The matched-control arm alone retains exact per-topic Baseline C budget
  `B_i`.
- Planning now has two pre-retrieval stages. The draft extracts atomic topic
  requirements and mapped query variants; the independent auditor rereads the
  original topic and emits a complete repaired replacement plan.
- Plan schema `planner_static_plan_v2` records requirement coverage, query
  roles, planning stage, and draft fingerprint. Execution artifacts use
  `planner_static_run_v2`, and retrieval rejects any draft-stage plan.
- Draft and audited artifacts are cached independently. Only validated audited
  plans can cross the retrieval boundary.

## Work sequence

- [x] Inspect current `origin/main`, instructions, Baseline C artifacts, and
  shared cache location.
- [x] Run the read-only Baseline C marginal-yield/overlap analysis before
  selecting a task cap.
- [x] Implement typed plan validation, cache identity, and deterministic task
  admission with fake-model tests.
- [x] Run the fixture eight-topic planner-only dry run and one bounded
  provider-backed diagnostic; record model, usage, latency, cache hits, and
  failures without retrying the incomplete eight-topic invocation.
- [x] Implement the smallest executor slice and fake end-to-end tests for
  matched-budget allocation, retrieval cache reuse, hydration, passage
  lineage, explicit partial/failure states, and deterministic merging.
- [x] Run one bounded live/static retrieval smoke for topic 2000 using exactly
  its Baseline C budget; do not start a 103-topic live run without a separate
  decision record.
- [x] Run a cache-only matched-vs-flat proxy comparison for topic 2000 using
  the same seven-query budget and top-k; label it as single-topic and
  non-judged.
- [x] Run focused tests, then the complete offline suite and `git diff --check`.
- [x] Update this plan and the evidence log with final status and next action.
- [x] Add a two-pass requirement-aware draft and independent coverage audit.
- [x] Remove the richer arm's artificial 24-task cap while retaining finite
  provider token and timeout bounds.
- [x] Run one topic-2000 two-pass planner-only trial and manually review it
  before retrieval.
- [x] Run a 12-topic representative planner-only gate across varied domains.
- [x] Audit every sealed sample plan for requirement coverage, query necessity,
  semantic duplication, unsupported restrictions, and role accuracy.
- [x] Stop broad execution after the gate failed; obtain read-only Fable
  failure advice and verify its decisive claims against repository evidence.
- [x] After explicit user reauthorization, resume richer breadth retrieval from
  existing checkpoints without repeating successful queries. Preserve the new
  live success for `2007`, promote the exact cached query for `2031` with
  organizer networking disabled, and stop after repeated bounded organizer
  failures.
- [x] After the user explicitly requested another attempt, resume all 96 topics
  still lacking a richer supplement with one serial zero-retry query each.
  Complete 96/96 successfully, using the next sealed variant for the nine
  historically failed topics, then rebuild and verify the all-topic coverage
  index.
- [ ] Complete the remaining 3,165 sealed richer-plan queries. A private,
  resumable, two-worker, zero-retry run is active in tmux as
  `20260813T073518Z`. Its `topic-major-v1` queue finishes every remaining
  query for one topic before advancing. Only two requests are in flight.
  Three- and four-worker gates were slower than two; four also timed out.

## Verification gates

- No planner artifact is cached unless its plan schema and cross-record IDs
  validate completely.
- No retrieval-stage success is published when a required query, hydration,
  or passage lineage check fails.
- No call has an unbounded retry or a five-minute individual timeout.
- No prompt, trace, provider event, credential, or plaintext `.env` is written
  to committed or submission artifacts.
- Judged retrieval quality is reported only if authoritative judgments are
  found; otherwise results are labeled as proxy metrics.

## Current next action

Do not repeat successful retrievals or regenerate plans. Baseline C and richer
breadth supplements now both cover all 103 topics. This is one richer query per
topic for most topics, not execution of all 3,292 sealed queries: the retrieved
tasks initially covered 118/882 planned requirements, and only three multi-query
topics covered every planned requirement. At the user's explicit request, the
remaining full-plan run is now active in topic-major order. Monitor its
owner-only `progress.json`; do not launch a duplicate runner. Downstream
topic-union scoring is gated to topics whose successful task count exactly
equals the sealed plan task count. Passage scoring is held: no scorer process
or new infrastructure is active. When scoring resumes, consume complete topics
only, deduplicate canonical document IDs across all query results, score every
passage against every information need, and retain the maximum pair score with
no RRF. When the first retrieval pass finishes, rerun the same checkpoint-aware
command only if its final status records errors.
