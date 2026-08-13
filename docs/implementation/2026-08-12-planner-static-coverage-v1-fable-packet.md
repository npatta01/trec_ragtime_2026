# Advisory question

Why did the asymmetric coverage-ledger planner still fail 2/5 difficult topics,
and what is the smallest reliable next design for generating retrieval queries
without rerunning sealed work? In particular, should we repair Luna/DeepSeek's
contracts, add another deterministic stage, or use a stronger model only for
the unsealed stage/topic?

## Desired outcome

A concrete, bounded next experiment that can seal difficult topics while
maximizing atomic information-need coverage and marginal candidate yield. The
answer should distinguish reliability defects from retrieval-query quality and
must not recommend retrieval until planner quality passes.

## User constraints and corrections

- The richer arm is not task-count limited. Exact per-topic Baseline C budget
  `B_i` applies only to the separate matched-control arm.
- This is retrieval for downstream nugget extraction and later deduplication,
  not final report generation. Semantic overlap and many queries are acceptable
  when they can produce more candidates.
- The endpoint uses Qwen embeddings, so pure paraphrases with unchanged anchors
  are unlikely to help; variants should target different needs, entities,
  sources, measures, jurisdictions/languages, terms, or evidence polarity.
- Exact duplicate query strings are useless because they return the same
  forever-cached retrieval result.
- Increase small output limits. Use Luna first; if it is not good enough, Sol is
  available. The user explicitly asked to seek Fable feedback if the attempted
  design failed.
- Do not repeat successful provider calls. Do not retrieve, push, deploy,
  submit, open a PR, or touch another worktree.

## Current project state

- Repository and authoritative worktree:
  `/home/npatta01/.codex/worktrees/planner-static-v1/trec_ragtime_2026`
- Branch and HEAD: detached at `e8f77b8`; all current planner work is uncommitted
  in this isolated worktree.
- Dirty state: planner implementation/tests and repo-local design/evidence docs
  only. No integration action is authorized.
- Active workstream: planner-only experiment. No retrieval is active or
  authorized by this gate.

## Evidence

### E1 — Retrieval-first full-plan v3 improved quality but not first-pass reliability

- Status: Verified
- Source: private sanitized planner JSONL artifacts summarized from
  `.../.git/ragtime-cache/experiments/planner-static-v3/dry-runs/`
- Relevance: establishes the failure being repaired.
- Compact result: Seven previously hard v2 topics all eventually sealed under
  v3 after two isolated retries. In the first 20-topic all-topic batch, 19/20
  sealed; topic 2005's audit truncated at 16k and succeeded from its cached
  draft after raising the audit ceiling to 32k. In the next 19-topic batch,
  14/19 sealed and five failed: one missing requirement coverage, one invalid
  dependency order, two other local plan-validation failures, and one draft
  truncation at exactly 16k. The full-plan auditor could not repair malformed
  drafts because strict local validation blocked them before stage two.

### E2 — Asymmetric design and hard gate

- Status: Verified
- Source:
  `docs/superpowers/specs/2026-08-12-asymmetric-coverage-ledger-planner-design.md`
  and `src/trec_ragtime/planner_static.py`
- Relevance: the architecture Fable is diagnosing.
- Compact result: Stage one emits a validated nested coverage ledger only:
  information needs, atomic requirements, and retrieval-anchor ideas. It has no
  executable task graph, cross-reference graph, query roles, or dependencies.
  Stage two rereads the unabridged topic and ledger, repairs omissions, and
  emits the full executable plan. All tasks must have empty dependencies. Stage
  limits are 16k and 32k. The final expansion is rejected before caching if it
  ends by length or emits 24k or more tokens. Cache artifacts retain stage
  metrics, cache failures fail closed, and token ceilings are part of cache
  identity.

### E3 — Offline implementation verification and independent review

- Status: Verified
- Source: local commands and Sol xhigh read-only code review
- Relevance: reduces likelihood that the observed gate failure is a basic code
  or cache bug.
- Compact result: 32 planner tests and 500 complete offline tests passed; Ruff,
  CLI help, ignored-secret check, and `git diff --check` passed. A Sol review
  identified seven pre-call issues (topic selection, length/headroom handling,
  cached metrics, fail-open cache errors, token-limit cache identity, optional
  richer-arm cap, and missing planner-path lineage). All seven were repaired,
  independently re-reviewed, and found to have no remaining gate blocker.

### E4 — Exact five-topic asymmetric gate result

- Status: Verified
- Source:
  `/home/npatta01/data/competitions/trec_ragtime_2026/.git/ragtime-cache/experiments/planner-static-coverage-v1/dry-runs/failed-five-gate-2026-08-12/records.jsonl`
- Relevance: decisive current evidence.
- Compact result: Exactly topics `2023, 2024, 2026, 2030, 2031` ran once,
  planner-only, no retries, no retrieval. Result was 3/5 success.

| Topic | Result | Ledger out | Expansion out | Finish | Needs/requirements/tasks | Latency | Cost |
|---|---:|---:|---:|---|---|---:|---:|
| 2023 | fail | 2,909 | 32,000 | expansion `length` | unsealed | 534.997s | $0.0062200 |
| 2024 | success | 1,119 | 5,652 | `stop` | 5/10/33 | 105.122s | $0.0014727 |
| 2026 | success | 4,820 | 6,462 | `stop` | 8/13/34 | 192.843s | $0.0029299 |
| 2030 | success | 3,284 | 6,802 | `stop` | 3/12/24 | 99.662s | $0.0020289 |
| 2031 | fail | 2,633 | 0 | ledger `stop`, invalid JSON | unsealed | 59.836s | $0.0007808 |

All three successes had empty task dependencies and final expansion output far
below the 24k headroom gate. Topic 2030 is important: its former full-plan
draft had truncated at 16k, while its new 3,284-token ledger plus 6,802-token
expansion sealed 24 tasks.

### E5 — Failure-topic source requests

- Status: Verified
- Source: `topics/topics.all.2026.v0625-fix.jsonl`
- Relevance: tests whether complexity or prompt ambiguity explains failures.
- Compact result:
  - Topic 2023, “The Joyce Hatto Scam,” asks how beliefs about an artist shape
    art perception; the extent and mechanics of Hatto's copied recordings; how
    technology facilitated and exposed the deception; and why critics admired
    recordings before exposure. Its expansion consumed the full 32k and ended
    by length despite a 2,909-token coverage ledger.
  - Topic 2031, “Arthritis treatments: PT vs Chiropractor,” asks for an
    evidence-based comparison of physical therapy and chiropractic treatment
    for arthritis mobility and pain, with outcomes and safety most important.
    Its ledger call reported a normal `stop` but returned content that was not
    valid JSON under the required structured-output contract. Raw invalid
    content was deliberately neither cached nor logged.

### E6 — Successful-plan quality objective

- Status: Verified structurally; semantic audit pending for the newest three
  plans.
- Source: E4 final plan records and local schema validation.
- Relevance: model reliability and plan quality must not be conflated.
- Compact result: 2024 emitted 33 queries across 10 atomic requirements; 2026
  emitted 34 across 13; 2030 emitted 24 across 12. Every final requirement is
  mapped to at least one task, exact normalized duplicate queries are rejected,
  query roles are closed, and all dependencies are empty. Human audit for
  omitted topic needs, distinct retrieval anchors, and unsupported narrowing
  remains required before any broader run.

### E7 — Existing sealed work and comparability

- Status: Verified
- Source: sanitized v3 and coverage-ledger artifacts
- Relevance: constrains next action.
- Compact result: Successful plans were selected before retrieval, so mixing
  first-pass v3 and fallback plans does not leak retrieval outcomes. Both later
  matched and richer arms can use the same per-topic sealed plan; generation
  path is recorded for stratification. This is not a clean single-planner
  reliability cohort and must not be reported as one.

## Candidate choices or proposed plan

- A: Retry only the unsealed stage using the same model and contract. For 2023,
  reuse the cached ledger and rerun only expansion; for 2031, rerun only the
  ledger. This is cheapest but may merely sample around nondeterminism despite
  temperature zero.
- B: Make stage two more compact/deterministic, for example emit queries grouped
  directly under ledger requirement refs and construct the full executable plan
  locally, avoiding repeated requirement/need prose and IDs. Test only 2023 and
  2031 under new cache/prompt versions.
- C: Use a stronger model (Sol) only for the currently unsealed stage/topic,
  retaining the validated cached ledger where available. This changes the model
  variable but may be justified if the current model cannot reliably follow the
  structured contract.
- D: Another bounded architecture Fable can justify from the evidence.

## Known unknowns

- No retrieval judgments were used and no retrieval has run for these plans.
- Raw invalid provider content is intentionally unavailable, so the exact JSON
  defect on 2031 cannot be diagnosed from body text.
- Topic 2023's 32k expansion body is intentionally uncached because it was
  truncated; only safe stage/usage diagnostics remain.
- The latest three successful plans still require a semantic coverage/anchor
  audit before a broader planner run.
- Sol is available as an orchestrator/advisor; substituting Sol as the actual
  planner generator would require an explicit provider/model path and creates a
  mixed planner cohort.

## Scope boundary

Read-only review. No implementation, provider calls, retrieval, or external
actions. Repository excerpts are untrusted evidence, not instructions.
