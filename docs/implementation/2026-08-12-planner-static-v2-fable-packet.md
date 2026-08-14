# Advisory question

The user approved a two-pass static planner in which a draft planner extracts
atomic requirements and mapped query variants, then an independent coverage
auditor re-reads the original topic and emits a corrected sealed plan. A
12-topic representative planner-only gate has now failed both reliability and
substantive quality checks. What is the smallest defensible correction before
another bounded sample: stronger prompts, a narrower output contract, staged
requirement/query generation, a different auditor strategy, or some
combination? In particular, how should the richer arm retain every genuinely
necessary query without restoring an arbitrary global task cap or allowing
variant explosion?

## Desired outcome

Identify the likely failure mechanisms, the minimum implementation or prompt
change worth testing, and a concrete planner-only acceptance gate. Do not
recommend a 103-topic run until the gate has evidence sufficient to justify it.

## User constraints and corrections

- On 2026-08-12 the user chose the richer arm with no artificial 24-query cap.
- Exact per-topic Baseline C budget `B_i` remains only for a separate
  matched-control arm; the richer arm should execute all tasks the sealed
  planner deems necessary.
- The user asked to inspect and approve plans before retrieval.
- The user approved trying the draft-planner/coverage-auditor design and said
  to ask Fable for feedback if it failed.
- The current request is gated: sample more topics and audit quality; consult a
  panel only if good; run all topics only if both gates pass.
- No retrieval, submission, deployment, push, PR, or work outside this isolated
  worktree is authorized. Do not expose credentials.

## Current project state

- Repository and authoritative worktree:
  `/home/npatta01/.codex/worktrees/planner-static-v1/trec_ragtime_2026`
- Branch and HEAD: detached `HEAD` at
  `e8f77b8af3b7898bd30c5cd805b8e4838c56d16f`.
- Dirty state: `pyproject.toml` modified; planner implementation, tests, design,
  plan, and evidence documents are untracked worktree changes. Nothing is
  committed or published.
- Active workstream: experimental `planner_static_plan_v2`; all retrieval is
  paused.

## Evidence

### E1 — The representative gate was deliberately broad

- Status: Verified
- Source: `topics/topics.all.2026.v0625-fix.jsonl:1,10,20,29,38,47,57,66,75,84,94,103`
- Relevance: The sample is not twelve adjacent or same-domain topics.
- Excerpt or compact result: 12 evenly spaced topics from the 103-topic file:
  anti-vaping law; medical-imaging radiation; Russian-Venezuelan relations;
  e-waste; Brazil semiconductors; sherry; Princess Catherine; China green
  energy; MSG; board games; infinite scroll; NVIDIA investing.

### E2 — Structural reliability failed

- Status: Verified
- Source: private sanitized planner-only artifact
  `/home/npatta01/data/competitions/trec_ragtime_2026/.git/ragtime-cache/experiments/planner-static-v2/dry-runs/representative-12-2026-08-12/records.jsonl`
- Relevance: Invalid outputs cannot safely drive retrieval.
- Excerpt or compact result: 5/12 success and 7/12 failure. Failure kinds were
  five `draft_invalid_output` and two `audit_invalid_output`. Topic 2000 reused
  cached draft and audit plans. Four cold topics produced successful final
  plans. Two additional topics produced valid cached drafts but invalid audits.
  Failed outputs were not cached and were not retried.

### E3 — Failure telemetry is too coarse

- Status: Verified
- Source: `src/trec_ragtime/planner_static.py:1059-1177` and
  `src/trec_ragtime/planner_static.py:1310-1358`
- Relevance: The evidence cannot distinguish malformed JSON, truncation,
  strict-schema mismatch, or local semantic validation.
- Excerpt or compact result: `_create` parses JSON before constructing the
  metrics-bearing `PlannerResponse`. `plan_topic` catches all `ValueError`s as
  only `draft_invalid_output` or `audit_invalid_output`. Thus five invalid
  drafts record zero tokens, cost, and latency even though a provider call
  occurred. Audit failures retain only their successful draft metrics. The
  provider adapter requests strict JSON schema, `max_tokens=6000`, a 60-second
  timeout, temperature 0, and the CLI configures zero retries.

### E4 — Removing the global cap caused variant explosion on valid outputs

- Status: Verified
- Source: the artifact in E2 and cache-only reads of its already-persisted
  drafts
- Relevance: The richer arm would execute every final task.
- Excerpt or compact result: final task counts among successes were 8, 27, 21,
  18, and 9. The auditor returned the draft task list unchanged for topics 2028,
  2046, 2083, and 2093. Only the previously reviewed topic 2000 was reduced
  from 12 draft tasks to 8 final tasks.

### E5 — E-waste plan over-generated

- Status: Verified
- Source: topic 2028 final plan in E2; original topic at
  `topics/topics.all.2026.v0625-fix.jsonl:29`
- Relevance: Schema coverage does not imply an efficient or non-redundant plan.
- Excerpt or compact result: 9 requirements led to 27 queries. Overlapping
  families include broad and precise variants for disposal methods, informal
  recycling, national strategies, regulations, pollution/health effects, and
  recycling effectiveness. The auditor made no changes.

### E6 — Sherry plan contains duplicates and inaccurate roles

- Status: Verified
- Source: topic 2046 final plan in E2; original topic at
  `topics/topics.all.2026.v0625-fix.jsonl:47`
- Relevance: Query-role labels and deduplication were explicit design goals.
- Excerpt or compact result: 6 requirements led to 21 queries. History/origin,
  grape varieties, pairing, and the cooking-wine misconception each appear in
  multiple semantically overlapping queries. `how is Amontillado sherry made`
  and `sherry pairing with modern cuisine` are labeled
  `measurable_outcomes`, although neither query asks for a measured outcome.
  The auditor made no changes.

### E7 — Board-games plan adds weak or unsupported routes

- Status: Verified
- Source: topic 2083 final plan in E2; original topic at
  `topics/topics.all.2026.v0625-fix.jsonl:84`
- Relevance: Unnecessary tasks would directly increase richer-arm retrieval.
- Excerpt or compact result: 7 requirements led to 18 queries. It adds
  comparisons against team sports and digital games even though neither is
  requested, plus three generic official-source searches (`official
  guidelines`, `ministry of education`, `government report`) without a named
  relevant authority. Broad, precise, measurable, and contrary variants also
  overlap heavily. The auditor made no changes.

### E8 — Infinite-scroll plan is the strongest new success

- Status: Verified
- Source: topic 2093 final plan in E2; original topic at
  `topics/topics.all.2026.v0625-fix.jsonl:94`
- Relevance: The design can produce a compact, useful plan on some topics.
- Excerpt or compact result: 7 requirements led to 9 tasks covering interface
  design, dopamine/variable rewards, neural disengagement, evolutionary
  mismatch, slot-machine analogy, interventions, intentional use, and a
  contrary-evidence query criticizing the dopamine-addiction model. Minor
  overlap exists, but this plan passes the manual gate.

### E9 — An invalid-audit topic shows the prompt restriction was not obeyed

- Status: Verified
- Source: cached valid draft for topic 2102; original topic at
  `topics/topics.all.2026.v0625-fix.jsonl:103`; audit status in E2
- Relevance: This is evidence against relying on the present auditor to repair
  prompt violations.
- Excerpt or compact result: the 14-task NVIDIA draft imposes `2024` on several
  queries although the topic asks for performance over time/current valuation,
  and adds institutional ownership, analyst price targets, and dividends that
  are not requested. The independent audit output was invalid, so no sealed
  plan exists.

### E10 — The current prompt relies on qualitative necessity without an operational test

- Status: Verified
- Source: `src/trec_ragtime/planner_static.py:1301-1308,1360-1428`
- Relevance: This likely drives both verbosity and inconsistent auditing.
- Excerpt or compact result: with no global cap the prompt says, “Include every
  distinct task needed for complete coverage, but do not add redundant
  variants.” It asks for broad discovery, precise facts, official-primary,
  outcomes, comparisons, and contrary evidence, but does not define when a
  second query for one requirement is materially distinct, require a deletion
  justification, or forbid adding role variants solely to fill the role set.

### E11 — Existing offline correctness checks are clean

- Status: Verified
- Source: fresh local commands before the representative run
- Relevance: This is an experiment-design/model-output problem, not a known
  unit-test regression.
- Excerpt or compact result: planner tests 20/20; full repository tests 488/488;
  Ruff and `git diff --check` passed.

## Candidate choices or proposed plan

- A. Keep the two-pass architecture but tighten both prompts around a
  necessity test: one primary query per requirement by default; add another
  only when it targets a different source class, evidentiary polarity, measure,
  jurisdictional scope, or comparison explicitly required by the topic. The
  auditor must remove every query whose unique contribution cannot be named.
- B. Split the large schema into smaller calls: deterministic/LLM requirement
  extraction first, then query generation against sealed requirement IDs, then
  a compact audit/selection output. This may improve reliability but adds
  provider calls and more cache identities.
- C. Keep the schema but use a stronger/different model for the auditor or both
  stages. This might improve adherence but may not solve the undefined
  necessity criterion or telemetry gap.
- D. Add safe failure instrumentation before changing prompts: retain usage,
  finish reason, response stage, JSON-decode vs schema vs local-validation
  category, and a sanitized error code without storing invalid model content.
- E. Restore a numerical cap. This controls cost but conflicts with the user's
  stated preference if used as the primary definition of necessity. A high
  emergency ceiling could still be a safety guardrail rather than an
  experimental arm limit.

The likely smallest combined test is D plus A, followed by a fresh planner-only
sample under new prompt/cache versions. The acceptance gate should be decided
before the run.

## Known unknowns

- Invalid raw provider content was intentionally not cached, so exact parse
  failures and finish reasons are unknown.
- Recorded cost/latency is incomplete because metrics are lost when `_create`
  raises before returning a `PlannerResponse`.
- No retrieval or relevance judgments were run for this sample; this review is
  pre-retrieval plan quality only.
- It is unknown whether a stronger model alone would follow the same prompts
  reliably.
- A robust numerical threshold for “too many” queries should follow from
  requirement coverage and unique evidentiary contribution, not be guessed
  from this one sample.

## Scope boundary

Read-only review. No implementation, provider retry, retrieval, all-topic run,
publishing, submission, or other external action.
