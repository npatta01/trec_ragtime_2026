# Advisory question

Is the retrieval-first two-pass static planner now good enough to run
planner-only across all 103 RAGTIME topics, followed by a quality audit, without
running retrieval? If so, should the all-topic procedure run once and retry only
failed topics once, or should the runner itself gain a single bounded retry for
local-validation failures first? Identify any evidence that should still block
the all-topic planner run.

## Desired outcome

A go/no-go recommendation for a private all-topic planner-only run, a bounded
failure-handling policy, and concrete quality criteria aligned with downstream
candidate retrieval and nugget extraction rather than query compactness.

## User constraints and corrections

- The user corrected the objective on 2026-08-12: overlapping and duplicate-like
  semantic variants are acceptable because the system wants more retrieval
  candidates and will extract/deduplicate nuggets downstream.
- Because the hosted endpoint uses Qwen embeddings, pure paraphrases may add
  little. Query diversity should target distinct information needs, entities,
  source families, measures, jurisdictions, terminology, or evidence polarity.
- The user explicitly approved maximizing information-need coverage and
  marginal candidate yield.
- The user said the old 6,000-token planner output limit was too small and
  approved increasing it. They asked to use Luna first and allow Sol escalation
  if Luna was not good enough.
- The richer arm has no artificial global task cap. Exact per-topic Baseline C
  budget `B_i` applies only to a separate matched-control arm.
- Plans must be inspected before retrieval. No retrieval, all-topic retrieval,
  submission, push, deployment, or PR is authorized in this decision.

## Current project state

- Repository and authoritative worktree:
  `/home/npatta01/.codex/worktrees/planner-static-v1/trec_ragtime_2026`
- Detached HEAD:
  `e8f77b8af3b7898bd30c5cd805b8e4838c56d16f`.
- Dirty state: experimental planner code, tests, plans, specs, and evidence are
  uncommitted local changes. Nothing has been pushed or published.
- Luna xhigh performed implementation and local review. The actual static
  planner provider remains `deepseek/deepseek-v4-flash-20260423`; do not confuse
  the orchestrator with the planner model. Sol is requested here as an
  independent advisor lane.

## Evidence

### E1 — Corrected retrieval quality objective

- Status: Verified
- Source: current user corrections and
  `docs/superpowers/specs/2026-08-12-retrieval-first-planner-reliability-design.md`
- Relevance: Prior compactness-based audit criteria were wrong.
- Excerpt or compact result: A good plan extracts every information need and
  creates plausible retrieval directions. Semantic overlap and large task
  counts are allowed. Exact normalized duplicate strings remain invalid because
  they return the same immutable cached retrieval. Imperfect labels and
  non-minimal plans are not blockers.

### E2 — Minimal v3 implementation

- Status: Verified
- Source: `src/trec_ragtime/planner_static.py`
- Relevance: The new gate uses isolated prompt/cache identities and better
  failure evidence.
- Excerpt or compact result: provider output allowance increased from 6,000 to
  16,000 tokens; prompt versions advanced to `planner-static-draft-v3` and
  `planner-static-audit-v3`; cache operation advanced to v3. Draft prompts seek
  candidate recall and distinct anchors rather than role-filling. Audit prompts
  preserve overlapping candidate-expanding routes and repair omissions or
  harmful restrictions. Sanitized diagnostics now retain stage, finish reason,
  usage, cost, latency, and a closed error category without raw invalid output.
  Timeout remains 60 seconds per request, temperature zero, and SDK retries zero.

### E3 — Offline verification is clean

- Status: Verified
- Source: fresh local commands after the v3 change
- Relevance: The result is not explained by a known unit regression.
- Excerpt or compact result: planner tests 22/22, complete repository tests
  490/490, Ruff, CLI help, and `git diff --check` all passed. A synthetic
  `finish_reason=length` response is classified `output_truncated` while
  retaining usage and excluding raw content. A duplicate-query local failure is
  classified without payload text.

### E4 — Seven previously failed topics all obtained sealed v3 plans

- Status: Verified
- Source: owner-only artifacts under
  `<git-common-dir>/ragtime-cache/experiments/planner-static-v3/dry-runs/failed-sample-v3-2026-08-12/`
- Relevance: This directly retests the prior v2 failures across medicine,
  geopolitics, industry, public opinion, energy, nutrition, and finance.
- Excerpt or compact result: final topic shapes were:
  - `2009`: 5 requirements, 20 tasks;
  - `2019`: 6 requirements, 36 tasks;
  - `2037`: 7 requirements, 29 tasks;
  - `2056`: 13 requirements, 22 tasks;
  - `2065`: 15 requirements, 32 tasks;
  - `2074`: 8 requirements, 13 tasks; and
  - `2102`: 12 requirements, 36 tasks.
  No plan was truncated and no retrieval ran.

### E5 — Reliability was 5/7 first-pass, then 7/7 after one failed-topic retry

- Status: Verified
- Source: same artifacts as E4
- Relevance: This is the main unresolved operational decision.
- Excerpt or compact result: First pass sealed `2009`, `2019`, `2056`, `2065`,
  and `2102`. Topics `2037` and `2074` returned complete parsed responses with
  `finish_reason=stop`, but local plan validation rejected them. A safe
  validator subcode was added after those attempts; one retry per failed topic
  then sealed both, so the original exact validator rules remain unknown. No
  successful provider stage was repeated. Every failed output remained
  uncached.

### E6 — Larger output allowance is materially used

- Status: Verified
- Source: same artifacts as E4
- Relevance: The increase is not merely theoretical.
- Excerpt or compact result: final draft+audit output-token totals were 6,356;
  12,066; 14,014; 15,802; 15,085; 9,632; and 15,125. Because each record
  aggregates two stages, totals above 12,000 prove at least one stage exceeded
  the old 6,000 ceiling for five topics. No v3 failure was classified as
  truncation.

### E7 — Manual retrieval-first quality audit passed all seven

- Status: Verified
- Source: final plan records in E4 compared with canonical topic text
- Relevance: Structural success alone is insufficient.
- Excerpt or compact result:
  - `2009` covers bodily harm, cancer, DNA damage, IARC, NTP, and safe clinical
    dose with source-specific, outcome, comparison, and contrary routes.
  - `2019` covers military, diplomatic, economic, third-country involvement,
    regional effects, and negative effects using arms, advisors, UN voting,
    trade/debt, sanctions, intermediaries, security, and counterevidence routes.
  - `2037` covers history, current status, industry categories, competitiveness,
    challenges, investment conditions, and balanced risks, including a
    Portuguese-language route.
  - `2056` covers polls and royal comparisons, duties, parenting, cancer,
    personal/fashion style, commoner/media arc, domestic/global popularity, and
    modernization/humanization of the monarchy.
  - `2065` covers climate outcomes, policy transferability, solar/wind/battery/EV
    dominance and overcapacity, US/other-country views, and domestic/international
    challenges.
  - `2074` covers overall health, symptom complex, sodium reduction, natural
    glutamate comparison, official safety positions, adverse reactions, and
    long-term effects.
  - `2102` covers AI/gaming/chip growth, historical stock behavior, benefits,
    current valuation risks, volatility/drawdowns, retirement-income suitability,
    official filings, comparisons, and contrary investment cases.

### E8 — Quality caveats are non-blocking under the corrected objective

- Status: Verified facts with an Inference on impact
- Source: final plans in E4
- Relevance: Reviewers should not silently restore a compactness objective.
- Excerpt or compact result: Some specific routes impose older date anchors,
  such as Brazil market size `2023 2024` and China market shares in 2023/2024.
  The same plans also contain unrestricted broad/current routes, so these dated
  routes add historical candidates rather than narrowing the entire plan.
  Topic `2019` does not put the word `current` in a query, but has broad
  relationship queries likely to retrieve recent as well as historical
  documents. These are imperfect, not omitted information needs.

### E9 — Cost and latency are low-dollar but operationally slow

- Status: Verified
- Source: all nine v3 attempt records (seven final successes plus two failed
  first attempts)
- Relevance: All-topic planning may take hours serially.
- Excerpt or compact result: 44,074 input tokens, 99,517 output tokens,
  `$0.0281786253` reported cost, and 1,498.217483 provider-seconds. Successful
  per-topic aggregated latency ranged from 109.61 to 320.60 seconds. The nominal
  60-second SDK request timeout did not bound aggregate or observed stage wall
  time as tightly as expected. No automatic retries occurred.

## Candidate choices or proposed plan

- A. Run all 103 planner-only with v3, no automatic retry; inspect classified
  failures and run one separate failed-topic retry batch. This preserves exact
  first-pass evidence and never repeats successes.
- B. Add one automatic retry for `local_validation` only, then run all 103.
  This yields a more complete single artifact but obscures first-pass reliability
  unless attempts are separately recorded.
- C. Stop and redesign or switch planner models before all-topic planning. This
  is warranted only if the panel considers 5/7 first-pass reliability or the
  quality caveats disqualifying despite 7/7 eventual sealed plans.

Luna recommends A: retain first-pass observability, then retry only failures
once. Do not retrieve. Audit every sealed all-topic plan for requirement
coverage, unusable queries, and globally harmful narrowing; do not reject plans
for semantic overlap, task count, or imperfect role labels.

## Known unknowns

- The two original v3 local-validator subcodes were not captured because that
  safe field was added after the attempts; both retry outputs were valid.
- Temperature zero did not make output shape fully deterministic.
- Planner-only quality does not establish retrieval recall or nugget quality.
- The all-topic serial wall time is uncertain and could be several hours.
- No direct comparison of marginal candidate yield has yet run; retrieval is
  intentionally outside this gate.

## Scope boundary

Read-only review. No implementation, provider call, retrieval, all-topic run,
publication, submission, push, deployment, or PR by the panel.
