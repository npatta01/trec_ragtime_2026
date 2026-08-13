# planner-static-v1 evidence log

## 2026-08-12 — repository and artifact inspection

Environment: isolated worktree
`/home/npatta01/.codex/worktrees/planner-static-v1/trec_ragtime_2026/`,
detached at `e8f77b8` (`origin/main`). The worktree was clean before changes.
The shared cache is the Git common directory cache:
`/home/npatta01/data/competitions/trec_ragtime_2026/.git/ragtime-cache`.
`.env` was produced with `sops decrypt --output .env config/secrets.sops.env`
and `git check-ignore .env` confirmed it is ignored. No credential value was
printed.

### Baseline C artifacts

Primary immutable run artifact inspected:

`/home/npatta01/data/competitions/trec_ragtime_2026/.git/ragtime-cache/experiments/query-baselines/baseline-c-facet-rrf-alias-2026-08-09/manifest.json`

Associated hits and fused topic records:

- `.../queries.jsonl`
- `.../topics.jsonl`

The manifest records:

- `schema_version`: `query_baseline_v1`
- `status`: `complete`
- 103 planned/recorded/successful topics
- 752 successful query records and zero failed queries
- 10,300 fused candidates
- zero unresolved documents and zero missing translations
- pipeline `ragtime2`, `top_k=100`, RRF `k=60`
- corpus revision `c99bf3e2e8c18c5172498875d7522285f658e2c9`
- query-plan SHA recorded by the run: `83dacea6961986bcbd9aeb1423e8da321d80f0d185d39a875ef148f8f22185d7`

Independent parsing of `topics.jsonl` confirmed:

- 103/103 topic records are `success`.
- 752/752 query records are `success`.
- plan status is 102 `model_generated` and 1 `fallback` (topic `2066`).
- facet count distribution is 1 topic with 4, 9 with 6, 50 with 7, and 43
  with 8 queries: total 752, mean 7.301, median 7.
- every query result contains 100 hits; exact duplicate query text does not
  occur within a topic.

This establishes operational completion and deterministic artifact creation,
not judged recall. No authoritative judged-recall artifact was found in the
current `origin/main` tree or inspected cache paths.

### Artifact discrepancy

The mutable current file
`.../experiments/query-variants/problem-background-2026-08-03/query_plans.jsonl`
has SHA-256
`04331b016f4896f25dd6cdd404510a7d95b46bcddf1f6c2fdb36350ee9cc28ae`, not the
SHA pinned by the completed Baseline C run. It currently contains 103
`model_generated` rows and 756 facets, with 6–8 facets per topic. A separate
topic-2066 refresh artifact exists under
`.../experiments/query-variants/problem-background-2026-08-03/topic-2066-refresh-2026-08-12/`.
The completed run’s own topic records are therefore the authoritative source
for the requested historical 752-query facts; the mutable plan file must not
be substituted for them.

## 2026-08-12 — Baseline C marginal-yield analysis

Command:

```bash
uv run python - "$CACHE/experiments/query-baselines/baseline-c-facet-rrf-alias-2026-08-09/topics.jsonl" <<'PY'
# read-only JSONL analysis; see terminal command history for the full script
PY
```

Results over all 103 successful topic records and 752 successful ranked lists:

- final unique-document union per topic: mean 514.38, median 519, range
  268–743.
- duplicate mass among the 100×query hit slots: mean 29.41%, median 27.67%,
  range 5.14–60.75%.
- all pairwise list overlap: mean 16.85 shared documents and mean Jaccard
  0.1024 over 2,395 pairs; adjacent-query overlap mean 17.04 and Jaccard
  0.1051 over 649 pairs.
- mean new documents by query position: 100.0, 77.2, 74.5, 70.3, 63.6,
  57.8, 56.7, and 50.1 for positions 1–8 (positions 7/8 apply only to topics
  with that many queries).
- mean cumulative unique documents after positions 1–8: 100.0, 177.2, 251.7,
  322.0, 385.4, 443.1, 504.6, and 546.8 where applicable.

Interpretation: later queries still add substantial unique-document coverage,
so a tiny static cap would be unjustified. The selected 24-task frontier cap
allows a bounded richer-plan experiment while the matched arm uses exactly the
observed per-topic Baseline C budget.

## Blockers and unknowns

- No authoritative judged retrieval labels or recall artifact were found; any
  pre-judgment evaluation must use clearly labeled proxies.
- The current `origin/main` tree does not include the later adaptive agentic
  implementation; this task will expose a plan/evidence seam that a later
  strategy can consume without importing that worktree.

## 2026-08-12 — planner implementation and bounded dry runs

Focused fake-model and fake-retriever tests are in
`tests/test_planner_static.py`; the implementation is
`src/trec_ragtime/planner_static.py`. The first vertical slice passed 14
focused tests and Ruff/diff checks. The fixture planner-only run covered the
deterministic representative topics `2000, 2015, 2029, 2044, 2058, 2073, 2087,
2102`; all eight validated with two needs and four admitted tasks. Its sanitized
metrics artifact is:

`/home/npatta01/data/competitions/trec_ragtime_2026/.git/ragtime-cache/experiments/planner-static-v1/dry-runs/fixture-2026-08-12/records.jsonl`

Provider-backed bounded work used the shared `provider_settings()` route and a
single `OpenAI` client configured with a 60-second timeout and zero client
retries. The eight-topic invocation eventually published this sanitized
metrics artifact:

`/home/npatta01/data/competitions/trec_ragtime_2026/.git/ragtime-cache/experiments/planner-static-v1/dry-runs/provider-2026-08-12/records.jsonl`

It contains 3 successful and 5 `invalid_output` planner records; the three
successful plans were topics `2073`, `2087`, and `2102`, with shapes of 6/19,
6/13, and 4/12 needs/tasks. Their mean latency was 145.421 seconds (range
82.452–263.814), with 264/1944, 308/1874, and 320/1482 input/output tokens.
The failed calls were not retried. A separate one-topic diagnostic for topic
`2000` completed and wrote `/tmp/planner-static-diagnostic.jsonl`; it was not
committed. Its sanitized record reports:

- model `deepseek/deepseek-v4-flash-20260423`, prompt version
  `planner-static-v1`;
- status `success`, cache miss, 307 input tokens, 1,019 output tokens,
  reported cost `$0.0003283`, and 17.463261 seconds latency;
- 3 information needs, 10 raw/admitted tasks, zero truncation;
- plan SHA-256 `4c4832f403d397bcafd05164c61494a79448b10966c83a302c6b2c2817a3da4d`.

An L1 cache-only probe confirmed that this validated topic-2000 plan is
readable through `ResponseCachePlanArtifactCache`; no provider call was made
by that probe. The artifact-cache warning is an external operational issue,
not a planner-schema failure.

### Real retrieval smoke

Using the cached validated topic-2000 plan, the existing
`RagtimeRetriever.from_env()` path, exact `B_2000=7` from the Baseline C topic
artifact, `pipeline="ragtime2"`, `top_k=100`, and the shared finite timeout
policy, the static executor completed all seven precommitted tasks with 14
bounded passages (two per task). The sanitized operational record is:

`/home/npatta01/data/competitions/trec_ragtime_2026/.git/ragtime-cache/experiments/planner-static-v1/smokes/topic-2000-matched-2026-08-12.json`

This proves the planner-cache → exact-budget admission → shared retrieval →
hydration → bounded passage output path. It is not a quality or judged-recall
result; no authoritative judgments were available.

### Cache-only matched-vs-flat proxy

For topic `2000`, the seven static retrieval artifacts were reread through the
same `RagtimeRetriever.get_cached_retrieval()` contract and compared with the
first seven successful Baseline C query lists at `top_k=100`. The sanitized
comparison is:

`/home/npatta01/data/competitions/trec_ragtime_2026/.git/ragtime-cache/experiments/planner-static-v1/proxies/topic-2000-matched-vs-flat-2026-08-12.json`

- static matched plan: 7/7 cache hits, 388 unique documents, 44.5714%
  duplicate mass;
- Baseline C seven-query prefix: 403 unique documents, 42.4286% duplicate
  mass;
- static plan shape before matching: 3 information needs, 10 admitted tasks
  under the 24-task cap.

This is a one-topic operational proxy comparison only. It does not establish
relevance, recall, or a static-planning gain; no judgments or downstream
nugget evaluation were available.

## 2026-08-12 — verification

After decrypting `.env` and confirming it is ignored:

```text
uv run python -m unittest tests.test_planner_static -v     # 14 passed
uv run python -m unittest discover -s tests -v              # 482 passed
uv run ruff check src/trec_ragtime/planner_static.py tests/test_planner_static.py
git diff --check
```

The complete offline suite finished `Ran 482 tests ... OK`. No official RAG or
MLIR export was changed or produced.

## Historical next action (superseded)

The smallest representative static-planning path is implemented and verified.
Do not start a 103-topic live matched-budget or frontier run yet: the provider
sample had five invalid-output failures and there are no judged labels. The
next decision is to stabilize the planner/provider route, then run matched,
flat same-count, and richer finite-frontier controls over identical topics
before making any quality claim.

## 2026-08-12 — two-pass coverage planner trial

The user approved a draft-then-audit planning gate and removed the artificial
24-task limit from the richer arm. The implementation is specified in
`docs/superpowers/specs/2026-08-12-two-pass-static-planner-design.md` and uses:

- `planner_static_plan_v2` with locally derived atomic requirement IDs;
- per-task requirement mappings and closed query-role labels;
- separate validated draft and audited cache identities;
- an audited-plan-only retrieval boundary;
- exact `B_i` only for the matched-control arm; and
- all audited tasks for the richer arm (`max_total_tasks=None`).

Fresh offline verification after the change:

```text
uv run python -m unittest tests.test_planner_static -v   # 20 passed
uv run python -m unittest discover -s tests -v            # 488 passed
uv run ruff check src/trec_ragtime/planner_static.py tests/test_planner_static.py
uv run ragtime-static-plan --help
git diff --check
```

All commands exited successfully. `.env` was refreshed with SOPS and remained
ignored. No credential value was printed.

Exactly one provider-backed planner-only trial ran for topic `2000`. It made
one draft call and one audit call, with zero client retries and no retrieval.
The private trial artifact is:

`/home/npatta01/data/competitions/trec_ragtime_2026/.git/ragtime-cache/experiments/planner-static-v2/dry-runs/topic-2000-two-pass-2026-08-12/records.jsonl`

Sanitized combined metrics:

- status `success`; draft cache miss; audit cache miss;
- model `deepseek/deepseek-v4-flash-20260423`;
- final prompt version `planner-static-audit-v2`;
- 3,352 input tokens and 8,566 output tokens across both stages;
- reported total cost `$0.00243442458`;
- combined provider latency `133.01096` seconds;
- draft SHA-256
  `6db392f05d7fd047b66ae4ec01b1a66373c2719286e31884ef1b40794e4518b0`;
- audited SHA-256
  `1d19aaeab000d6897ddbeb58c5215d71e64c07f8a6754d2942b9807b4e15e946`;
- 6 requirements, 2 information needs, 8 audited tasks, zero truncation.

A no-provider cache-only probe then confirmed both draft and audit cache hits
and reread the audited plan successfully.

### Draft-to-audit effect

The draft had the same six requirements and 12 tasks. The auditor reduced it
to eight by removing four semantically overlapping variants: a second global
overview, a second youth-prevalence outcome query, a second legal-rationale
query, and a second parts/accessories query. It retained a contrary-evidence
query and made the legal-reasoning task explicitly include court rulings.

### Pre-retrieval quality review

The audited plan passes the topic-2000 structural coverage gate:

- sale and consumption regulation: explicit;
- parts and accessories: explicit;
- countries, passage dates, and specific provisions: explicit;
- legal reasoning: explicit;
- measurable effectiveness and youth vaping prevalence: explicit;
- teenager/young-adult health outcomes beyond prevalence: explicit; and
- contrary evidence or unintended consequences: explicit.

Compared with the seven broad Baseline C queries, the audited plan adds a
dedicated official-primary-source route, contrary-evidence route, flavor-ban
outcome route, and direct respiratory/hospital health-outcome route. It avoids
the previous unsupported `since 2015` restriction and selected-country bias.

Two caveats remain. The model used role names rather than descriptive prose in
`evidence_target`, and country-specific follow-up queries are deferred in favor
of broad discovery plus official databases. Neither blocks a bounded
topic-2000 retrieval pilot, but both should be reviewed in a varied
planner-only sample before broad execution.

Verdict: the topic-2000 audited plan is good enough for a bounded retrieval
pilot using all eight tasks. No retrieval was run in this turn. The trial did
not fail, so Fable feedback was not invoked.

## Current next action

The user approved broadening the planner-only quality check before any
retrieval. The representative gate and its result are recorded below; this
earlier topic-2000 action is superseded.

## 2026-08-12 — representative 12-topic planner gate

A planner-only sample selected 12 evenly spaced topics from the 103-topic file:
`2000`, `2009`, `2019`, `2028`, `2037`, `2046`, `2056`, `2065`, `2074`,
`2083`, `2093`, and `2102`. The domains span law, medicine, geopolitics,
environmental policy, industry, food and culture, public opinion, energy,
nutrition, education, behavioral science, and finance. No retrieval ran.

The command was:

```text
uv run ragtime-static-plan --count 12 --max-planner-calls 12 \
  --output <git-common-dir>/ragtime-cache/experiments/planner-static-v2/\
dry-runs/representative-12-2026-08-12/records.jsonl
```

The runner completed all 12 records and exited `1` because the sample contained
failures:

- 5/12 sealed successfully;
- 5/12 failed as `draft_invalid_output`;
- 2/12 failed as `audit_invalid_output`;
- topic `2000` was a complete draft+audit cache hit; and
- no failed provider output was cached or retried.

Failure telemetry is incomplete by design in the current adapter. `_create`
parses JSON before constructing the metrics-bearing response envelope, and
`plan_topic` collapses all `ValueError` cases into a stage-level invalid-output
kind. Therefore failed draft calls report zero tokens, latency, and cost, while
audit failures retain only successful draft metrics. The artifact's known
lower-bound totals are 20,964 input tokens, 45,130 output tokens,
`$0.01505736176`, and 527.892117 provider-seconds. These are not total sample
cost or latency.

### Manual pre-retrieval quality audit

Only two of the 12 sampled topics pass the strict substantive plan gate:

- `2000` passes with 8 complementary tasks and full requirement coverage;
- `2093` passes with 9 tasks covering interface design, neurobiology,
  evolutionary mismatch, interventions, intentional use, and criticism of the
  dopamine-addiction model;
- `2028` is structurally valid but fails compactness: 9 requirements expanded
  to 27 overlapping e-waste queries;
- `2046` is structurally valid but fails compactness and role accuracy: 6
  requirements expanded to 21 tasks with repeated history, grape, pairing, and
  cooking-wine-myth families, while production and pairing queries are labeled
  `measurable_outcomes`;
- `2083` is structurally valid but fails necessity: 7 requirements expanded to
  18 tasks, including unrequested comparisons against team sports and digital
  games and generic official-source searches without a named relevant
  authority; and
- the other seven topics have no sealed plan because of invalid draft or audit
  output.

Cache-only inspection of already-persisted drafts confirmed that the fresh
auditor made no query-level change on topics `2028`, `2046`, `2083`, or `2093`.
For `2028` it normalized role-like `evidence_target` strings. For `2046` it
translated Chinese requirement/evidence text to English while retaining all 21
queries. Topics `2083` and `2093` were semantically unchanged. The only observed
query deletion remains cached topic `2000` from the earlier trial. Fresh audits
for `2074` and `2102` were invalid.

The valid NVIDIA draft for `2102` also demonstrates prompt noncompliance: it
imposed `2024` on several queries and added institutional ownership, analyst
price targets, and dividends despite those not being requested. Its attempted
audit failed validation.

### Fable failure consultation

Following the user's explicit fallback instruction, a read-only Fable
consultation used the frozen sanitized packet at
`docs/implementation/2026-08-12-planner-static-v2-fable-packet.md`. Fable ran
through first-party `claude.ai` subscription authentication with model `fable`
and max effort. Codex and Gemini lanes were intentionally disabled; this was
failure diagnosis, not the approval panel requested only after a passing
sample.

Fable recommended:

1. preserve sanitized stage/finish-reason/usage/error-class telemetry before
   parsing so truncation, JSON, schema, local validation, refusal, and timeout
   cases can be distinguished;
2. encode query necessity in the schema: one primary query per requirement by
   default, with each extra query naming a topic-licensed distinctness axis and
   unique contribution; and
3. replace full-plan audit re-emission with a compact per-query
   keep/drop/rewrite/add decision list applied and revalidated deterministically
   in code.

Fable also hypothesized that the two audit failures may correlate with cached
draft input. Repository evidence contradicts that claim: both run records have
`draft_cache_hit=false`, and every persisted draft round-trips exactly through
`static_plan_from_record`. The drafts were generated fresh and passed directly
to the auditor in the same `plan_topic` call. No cache-path defect was found.

The verified central diagnosis remains: the role taxonomy acts as an implicit
checklist, while the prompt has no operational definition of when a variant
adds unique evidence. Full-plan re-emission lets the auditor copy a bloated
draft or fail while reproducing it.

## Historical gate decision (superseded)

The representative quality gate failed. Do not run the approval panel, all 103
topics, or any richer-arm retrieval. The next action requires a user-approved
design change: add safe failure telemetry, structural necessity fields, and a
compact decision-list auditor under new prompt/schema/cache versions; then run
a fresh bounded planner-only gate. Retain no artificial global task cap. A
high task-count threshold may flag a plan for human review but must not truncate
the richer arm.

## 2026-08-12 — final ledger-v4 / expansion-v3 planner cohort

The user approved a final two-stage planner contract and then authorized a
three-topic planner-only gate followed by all-topic plan generation if the
gate passed. No retrieval ran.

The final contract is recorded in:

- `docs/superpowers/specs/2026-08-12-deadline-final-planner-contract-design.md`;
- `docs/superpowers/plans/2026-08-12-deadline-final-planner-contract.md`; and
- `src/trec_ragtime/planner_static.py` under provider identities
  `planner-static-ledger-v4` and `planner-static-expand-v3`.

The problem statement is authoritative, the title disambiguates, and the
background contributes factual and decision-relevant context without turning
biography or workflow into standalone requirements. The ledger seeks the
smallest complete atomic requirement set. Expansion emits one or more queries
per requirement with no query-count cap, preserves candidate-shifting semantic
overlap, and merges exact normalized duplicates.

Output ceilings are provider-response safety controls, not query budgets:
ledger generation allows 16,000 tokens with an 8,000-token quarantine gate;
query expansion allows 32,000 tokens with a 24,000-token quarantine gate.
`max_total_tasks` remains `None` throughout the richer path.

### Final offline verification

Test-first implementation produced the expected red failures for field
authority, ledger headroom, and retrieval-anchor bounds before the fix. After
the final reliability changes:

```text
UV_OFFLINE=1 uv run python -m unittest tests.test_planner_static -v
# Ran 49 tests in 0.029s — OK

UV_OFFLINE=1 uv run python -m unittest discover -s tests -v
# Ran 517 tests in 17.649s — OK

UV_OFFLINE=1 uv run ruff check \
  src/trec_ragtime/planner_static.py tests/test_planner_static.py
# All checks passed!

UV_OFFLINE=1 uv run ragtime-static-plan --help
git diff --check
# both exited 0
```

The final 103 plans also independently round-tripped through
`static_plan_from_record`; all 103 recomputed plan hashes and topic hashes
matched.

### Three-topic gate and Fable advice

The gate topics were `2048` (propolis), `2055` (earthquake forecasting), and
`2057` (Crocs). Final selected results were:

- `2048`: 5 requirements, 15 queries, `$0.00101205`, 35.660551 seconds;
- `2055`: 4 requirements, 12 queries, `$0.00050340794`, 53.966929 seconds,
  using its cached validated ledger after an initial local-sealing failure;
- `2057`: 9 requirements, 29 queries, `$0.00204820676`, 135.591879 seconds.

All three had zero exact duplicates, uncovered requirements, query truncation,
or fallback use. Manual inspection found that the propolis ledger no longer
turned promotional workflow into a requirement; earthquake forecasting
preserved progress, verified cases, new methods, uncertainty, false alarms,
and reproducibility; and Crocs used the background's influencer/comfort context
as factual retrieval needs.

The initial `2055` expansion had returned `finish_reason=stop` but failed local
assembly. Per the user's instruction, a read-only Fable consultation used the
sanitized frozen packet at
`docs/implementation/2026-08-12-planner-final-fable-context.md`. Fable ran as
model `fable`, max effort, through first-party `claude.ai` subscription auth;
Codex and Gemini lanes were intentionally disabled. Fable recommended keeping
valid two-pass plans, adding precise sanitized local-validation codes, allowing
exact duplicate merges across needs, and using a visible deterministic ledger
fallback below a 10% systemic-fallback pause threshold.

The implementation followed that advice. Exact duplicates now retain every
requirement mapping even if a later information need has no unique task. A
schema-valid, headroom-safe v3 expansion that fails local assembly falls back
to each validated requirement description plus its one-to-five anchor seeds.
Fallback plan identities include the precise validation code and a sanitized
source-payload hash. Legacy v2 remains fail-closed.

### All-topic result

The 100 topics outside the gate ran with four bounded workers and one private
checkpoint per topic. The main queue took approximately 3,021 seconds
(50 minutes 21 seconds) wall-clock. Original failures were preserved rather
than overwritten.

The canonical final artifact is:

`/home/npatta01/data/competitions/trec_ragtime_2026/.git/ragtime-cache/experiments/planner-static-final-v1/dry-runs/all-103-2026-08-12/all-103-final-plans.jsonl`

It contains exactly 103 successful topic plans, 882 requirements, and 3,292
queries. Query count is uncapped: min 7, median 29, max 90 per topic. Requirement
count is min 3, median 7, max 25.

Ninety-six final plans sealed through the normal provider or one-retry path.
Seven (6.80%) are visibly marked deterministic ledger fallbacks:

- local validation: `2023` and `2035` duplicate requirement refs, `2024`
  malformed query sets, and `2032` incomplete ledger-ref coverage;
- output truncation: `2006` and `2088` reached the 32,000-token expansion
  ceiling; and
- wall timeout: `2022` reached the separate 600-second process limit.

The two truncated outputs and timeout were not accepted or retried. Their
validated cached ledgers produced offline plans with 15, 28, and 60 queries.
Topic `2044` had one malformed ledger (`unknown_plan_field`); its single
authorized retry produced a direct 3-requirement, 27-query plan. The fallback
rate stayed below Fable's 10% pause threshold.

### Quality audit

The machine-readable audit is:

`/home/npatta01/data/competitions/trec_ragtime_2026/.git/ragtime-cache/experiments/planner-static-final-v1/dry-runs/all-103-2026-08-12/quality-audit.json`

Across all 3,292 queries it found zero:

- uncovered requirements;
- exact normalized duplicates within a topic;
- task caps or admitted-query truncation;
- task dependencies; or
- manifest/plan hash mismatches.

Manual inspection covered every fallback plan, the three gate plans, the six
largest ledgers, dated-query outliers, all short-query patterns, and a
stratified sample of low lexical-overlap anchors. The fallbacks all retain an
exact direct requirement query plus three to five useful anchor queries. Large
ledgers were detailed but aligned with multifaceted topics rather than report
workflow. The 239 query/year occurrences absent literally from topic text were
concentrated in topics that requested recent/current data, historic trends,
projections, elections, or named dated events; they were retained as useful
candidate-shifting variants rather than treated as a cohort-wide date cutoff.

One non-blocking metadata defect remains: 41 topics label all query routes
`contrary_evidence`. The query strings and mappings remain useful and roles do
not affect retrieval execution, but downstream analysis must not interpret the
role distribution as reliable.

### Comparison with compact-v2

On the same 18 topics that compact-v2 successfully sealed, the final contract
changed:

- requirements: 218 to 172 (`-21.10%`);
- queries: 1,050 to 483 (`-54.00%`);
- selected successful-plan cost: `$0.03014556688` to `$0.02672957608`
  (`-11.33%`); and
- selected provider latency: 2,410.464022 to 2,135.288323 seconds
  (`-11.42%`).

This is a compactness and reliability comparison, not a retrieval-quality
measurement. Qualitatively, propolis fell from 11 requirements/90 queries to
5/15 without losing oral, topical, case, trial, or statistical evidence;
India/China fell from 20/93 to 8/32 while retaining demographics, economy,
innovation, labor, and projection needs; earthquake forecasting fell from
9/35 to 4/12 while preserving uncertainty and verification; and Crocs moved
from 8/26 to 9/29 because the final contract recovered the background's
influencer context.

### Cost, latency, and cache behavior

The complete planning campaign made an inferred 208 provider calls: 104 ledger
calls and 104 expansion calls. This is two calls per topic plus one extra
malformed-ledger retry for `2044` and one extra failed expansion attempt for
`2055`.

Known usage is a lower bound because `2022` was killed before its expansion
returned usage:

- 233,757 known input tokens;
- 668,787 known output tokens;
- `$0.171157404696` known cost; and
- 11,392.641117 recorded provider-seconds excluding the 600-second `2022`
  process timeout.

For the 96 directly sealed provider plans, median latency was 81.8121335
seconds, p95 was 241.3920495 seconds, and max was 352.430475 seconds. Across
the selected 103 final artifacts including recoveries, median recorded latency
was 85.255939 seconds, p95 331.2915094 seconds, and max 600 seconds.

All 103 validated ledgers are in the durable forever cache. One hundred final
expansion plans are cached; only the three offline recoveries (`2006`, `2022`,
`2088`) lack an expansion-cache object and instead live in the canonical final
manifest. The only provider-run ledger cache hit was the `2055` recovery; the
three offline recoveries also reread their ledgers from cache. There were no
expansion cache hits because this was the first final-contract cohort.

## Deadline retrieval gate

The user authorized a retrieval gate for the three audited topics. The run used
the frozen plans, `ragtime2`, `top_k=100`, the canonical local corpus, no
content fallbacks, and exact `planner_static_v2` query contexts. It did not call
the planner provider or regenerate a plan.

The organizer was unstable under deadline load. A four-worker load test
returned three complete queries and one `RagtimeApiError` after retries; the run
stopped before scheduling later work. Two workers recovered the failed query
and initially completed further waves, but later two-worker and one-worker
requests also exhausted retries. Successful retrievals were checkpointed and
published to the durable retrieval cache before later failures. Repeatedly
failing queries were replaced with unused variants for the same requirement;
those alternates failed while the organizer was in the same degraded state.
Live calls then stopped.

The canonical gate directory is:

`/home/npatta01/data/competitions/trec_ragtime_2026/.git/ragtime-cache/experiments/planner-static-final-v1/retrieval-gates/three-topic-2026-08-12/`

After the organizer recovered, a coverage-first retry used untouched variants
and completed every requirement. The runner then resumed the original 56-query
gate with two workers. Its first pair succeeded, but the next pair exhausted
retries in 244 and 362 seconds, so the remaining 28-query wave tail was not
scheduled. This second stop preserved the deadline cutoff while keeping all
new successes.

The user requested one further retry. An untouched Crocs history variant
returned 100 hydrated candidates in 110 seconds, but increased the cross-topic
unique pool by only four documents. The immediately following two untouched
earthquake variants both exhausted retries in 244 seconds, and the remaining
queue stopped again.

The final `manifest.json` validates 27 successful queries, 2,700 candidate
slots, 1,668 unique document IDs, 61.78% unique-slot yield, 27/27 completely
hydrated results, zero missing translations, and 18/18 requirements covered.
Two retrieval-artifact hits and one lower response-cache hit were observed.
Final topic status is:

- `2048`: 13/15 queries, 945 unique candidates, 5/5 requirements;
- `2055`: 4/12 queries, 241 unique candidates, 4/4 requirements; and
- `2057`: 10/29 queries, 485 unique candidates, 9/9 requirements.

Every successful query contained exactly 100 unique ranks and finite scores.
No successful query had zero marginal yield. Mean marginal unique yield was
72.69 documents for `2048`, 60.25 for `2055`, and 53.44 for `2057`. The
per-topic candidate pools use RRF with `k=60` and retain task, requirement, and
durable retrieval-object provenance.

A cache-only content spot-check found materially different precision proxies.
Earthquake/seismic anchors occurred in 96.75% of `2055` top-100 slots, and
Crocs/ugly-fashion anchors in 58.56% of `2057` slots. Only 4.62% of `2048` slots
explicitly mentioned propolis: its embedding results frequently followed
generic medical terms instead of the entity. These lexical rates are not
relevance judgments or recall measurements, but the observed propolis drift is
large enough to treat as a high-confidence precision warning. Preserve its
pool for nugget extraction rather than deleting candidates.

## Other-topic breadth retry

At the user's request, retrieval moved to untouched topics instead of spending
more deadline time on failed gate strings. A two-query health wave succeeded
for topics `2000` and `2001`; each returned 100 unique, fully hydrated
candidates in approximately 183 seconds with no missing translations. The
cache-only lexical precision proxy was strong: 88/100 candidates for the
e-cigarette-law query contained an e-cigarette/vaping/tobacco anchor, and
90/100 for the Nordic-walking query contained a walking/energy/calorie anchor.

The next breadth wave attempted the first frozen queries for `2002` and `2003`.
Both exhausted retries after approximately 244 seconds. The runner stopped, so
`2004` through `2009` were not called. The private breadth artifacts are under:

`/home/npatta01/data/competitions/trec_ragtime_2026/.git/ragtime-cache/experiments/planner-static-final-v1/retrieval-gates/deadline-breadth-2026-08-12/`

Its manifest validates two successful topics, two error checkpoints, six
unattempted topics, 200/200 unique candidates, complete hydration, and zero
missing translations.

An adaptive all-topic continuation then tried untouched topics with one fresh
retriever/session at a time. Topics `2004` and `2005` failed consecutively in
244 and 362 seconds, so the two-failure serial circuit breaker stopped the
remaining 96 queued jobs. The canonical `progress.json` in the breadth
directory records that cutoff. No successful checkpoint was repeated.

### Completed all-topic candidate fallback

The live endpoint was not required to obtain deadline candidate coverage. The
repository already contains the final completed Baseline C run at:

`/home/npatta01/data/competitions/trec_ragtime_2026/.git/ragtime-cache/experiments/query-baselines/baseline-c-facet-rrf-alias-2026-08-09/`

Fresh streaming validation confirmed 103/103 successful topics, 756/756
successful query lists, exactly 100 unique ranked candidates per topic, 10,300
topic-candidate slots, complete candidate text/hydration, finite fused scores,
zero unresolved documents, and zero missing translations. The source uses the
same pinned local-corpus revision as the richer retrievals.

The deadline coverage index is:

`/home/npatta01/data/competitions/trec_ragtime_2026/.git/ragtime-cache/experiments/planner-static-final-v1/retrieval-gates/deadline-all-topics-coverage-2026-08-12/`

Its owner-only manifest and 103-row topic index pin the Baseline C manifest,
topics, queries, and frozen richer-plan SHA-256 values. Baseline C is the
guaranteed primary pool for every topic. Successful richer retrievals supplement
topics `2000`, `2001`, `2048`, `2055`, and `2057`. The primary pools contain
8,424 globally unique documents. Primary-plus-supplement pools contain 9,853
globally unique documents; summed within topic scope, the unions contain 11,842
candidates. This is complete candidate coverage, not a claim that all 3,292
richer-plan queries ran or that judged recall was measured.

## Current next action

Do not regenerate plans or repeat successful retrievals. All 103 topics now
have validated candidate pools, and the five successful richer-topic pools are
indexed as supplements. Move immediately to nugget extraction using the
all-topic coverage index. Do not prioritize another organizer retry over
extraction before the deadline: the service fails in both concurrent and fresh
serial sessions, while the cached primary pools already provide complete topic
coverage. No submission, deployment, push, PR, or commit occurred.

## User-authorized remaining-topic continuation

After the user explicitly requested another attempt, retrieval resumed from the
existing checkpoints. No sealed plan was regenerated and no previously
successful richer query was repeated. Each live topic used a fresh retriever,
`ragtime2`, `top_k=100`, the pinned local corpus, no content fallbacks, exact
`planner_static_v2` lineage, and a finite timeout.

The bounded live outcomes were:

- `2006`: `RagtimeApiError` after 121.371 seconds with one retry;
- `2007`: success after 47.193 seconds, with 100 unique fully hydrated
  candidates, no missing translations, and a durable retrieval object;
- `2008`: `RagtimeApiError` after 121.148 seconds with one retry;
- `2009`: `RagtimeApiError` after 120.779 seconds with one retry;
- `2010` and `2011`: concurrent zero-retry probes, both `RagtimeApiError`
  after 60.750 and 60.751 seconds.

Thus one of six new live topic probes succeeded. After the serial breaker and
the final zero-retry pair both demonstrated the same organizer failure mode,
live calls stopped. The continuation did not call the planner provider.

An offline exact-query comparison found only three richer-plan strings already
present in the completed Baseline C query set: two for `2031` and one for the
already supplemented `2055`. The first matching `2031` query was promoted in
2.124 seconds through the forever response cache with organizer networking
explicitly disabled. It produced another 100 unique, fully hydrated candidates
and a durable static-lineage retrieval object.

The owner-only continuation artifacts are under:

`/home/npatta01/data/competitions/trec_ragtime_2026/.git/ragtime-cache/experiments/planner-static-final-v1/retrieval-gates/remaining-breadth-2026-08-12/`

The refreshed all-topic coverage index now records seven richer supplement
topics (`2000`, `2001`, `2007`, `2031`, `2048`, `2055`, and `2057`), 2,071
supplement candidate slots, 11,971 candidates summed within topic-local union
scope, and 9,970 globally unique documents across Baseline C plus supplements.
Baseline C remains complete for 103/103 topics. Nine topics have known live
errors, 87 have not been attempted by this richer breadth pass, and 96 remain
without a richer supplement.

### Current next action after continuation

Do not repeat the failed probes while the organizer remains degraded. Preserve
the 96-topic resumable queue, but use the already completed 103-topic nugget
artifacts on synchronized `origin/main` for the deadline path. No submission,
deployment, push, PR, or commit occurred.

## Endpoint recovery and complete richer breadth coverage

The user explicitly requested another attempt. A new serial pass removed the
circuit breaker and gave every topic still lacking a richer supplement one
bounded chance. It used zero automatic retries and no concurrency. The 87
untouched topics used their first allocated sealed query; the nine historically
failed topics (`2002`–`2006` and `2008`–`2011`) used their next allocated query
variant rather than repeating the failed string.

The endpoint had recovered. Run `20260813T052240Z` completed 96/96 topics with
zero errors in 1,662.105 seconds. Median topic latency was 17.191 seconds, p95
was 26.161 seconds, and maximum latency was 31.438 seconds. The run observed no
retrieval-artifact or lower response-cache hits, so all 96 results were new live
organizer successes. Each result passed the pre-publication gate: exactly 100
unique ranks, finite scores, full local hydration, no hydration failures, no
missing translations, exact `planner_static_v2` lineage, and a durable
retrieval-object key. No planner provider call occurred.

The refreshed coverage index now contains at least one richer supplement for
103/103 topics. Across the prior multi-query gates and breadth runs it records:

- 127 successful richer queries;
- 11,671 supplement candidate slots and 10,398 globally unique supplement
  documents;
- 18,740 candidates summed within topic-local Baseline-C-plus-supplement union
  scope;
- 15,618 globally unique documents across Baseline C plus all supplements;
- 8,440 topic-local documents new relative to Baseline C, with median 72 new
  documents per topic, minimum 29, maximum 856, and zero zero-yield topics.

A fresh direct lookup of all 10,398 unique supplement IDs through the pinned
read-only local DuckDB corpus returned 10,398 hydrated report-ready documents,
zero unresolved IDs, zero failures, and zero missing translations. Source
hashes matched and all index, summary, progress, and pool artifacts checked were
owner-only.

This is complete **topic breadth**, not complete richer-plan execution. The
retrieved tasks map to 118 of 882 planned requirements, and only the three
earlier multi-query topics cover every planned requirement. Running all 3,292
sealed queries remains a separate large retrieval job.

### Current next action after endpoint recovery

Do not repeat these 127 successful richer queries. Use the completed 103-topic
candidate and nugget artifacts for the deadline path. Only start the remaining
full-plan queries if the user explicitly prioritizes that substantially larger
run over downstream use. No submission, deployment, push, PR, or commit
occurred.

## Full richer-plan query run queued

The user explicitly authorized the larger run. Offline queue reconciliation
against the frozen 103-topic manifest and every successful richer pool proved
3,292 sealed unique task queries, 127 successful task queries, and exactly
3,165 remaining unique task queries across all 103 topics.

The owner-only resumable runner and artifacts are under:

`/home/npatta01/data/competitions/trec_ragtime_2026/.git/ragtime-cache/experiments/planner-static-final-v1/retrieval-gates/full-richer-2026-08-13/`

Run `20260813T060134Z` is active with four workers, `ragtime2`, `top_k=100`,
the pinned local corpus, no content fallbacks, exact `planner_static_v2`
lineage, and zero automatic retries. It round-robins topics so partial progress
improves cross-topic coverage, writes one atomic owner-only checkpoint per
query, and skips every validated success when resumed. Its frozen-plan hash is
`fbc32b5c3154f66d7c545beca9a8af68dd3373f4b48918219e5908986378ee10`.

The first health snapshot recorded 6/6 successful new query checkpoints and
zero errors. Every checkpoint contained exactly 100 unique candidates and a
durable retrieval-object key. Initial four-worker query durations were
46.373–61.433 seconds, implying a rough 10–14 hour completion window if early
throughput persists. This is a live estimate, not a completion claim.

The initial detached `nohup` launch was reaped by the execution environment
before issuing a checkpoint. It made no recorded successful or failed query
call. Reproducing under an attached execution session proved the runner itself
remained healthy; that attached session is the active run above.

### Current next action for the full run

Monitor `progress.json` and do not start a second runner while its lock is
held. If the first pass ends with query errors, launch the same runner again;
validated successes will be skipped and only failed or absent query checkpoints
will be retried. No submission, deployment, push, PR, or commit occurred.

### Adaptive concurrency correction and durable queue

The initial four-worker run eventually saturated the organizer. It preserved
eight valid success checkpoints but then produced repeated 60-second
`RagtimeApiError` timeouts. The exact launcher and Python child were stopped;
all atomic checkpoints remained valid. An immediate one-worker restart also
timed out on its first query, showing that the organizer needed recovery time
rather than merely lower concurrency.

After a two-minute cooldown, the runner was relaunched with one worker in the
host's persistent tmux session `ragtime-full-20260813`. The active run is
`20260813T061017Z`; it resumed eight validated full-run successes and queued
3,157 remaining queries. Its first five new serial checkpoints all succeeded,
with zero errors, median latency 14.132 seconds, and maximum latency 19.425
seconds. This early rate implies roughly 12–13 hours for the queue, but actual
completion time remains endpoint-dependent.

The live status files are `queue.json`, `runner.json`, `manifest.json`, and
`progress.json` in the full-run directory. The tmux server was independently
verified as detached and alive after the launching tool sessions ended. Do not
start another runner while that named session or the owner-only lock exists.

### Bounded two-worker acceleration gate

At the user's request, the healthy serial runner was checkpoint-stopped and
resumed with two workers as run `20260813T063801Z`. It inherited 107 validated
full-run successes and queued 3,058 remaining queries. The first 21-query gate
completed 21/21 successfully with zero errors. Median query duration rose to
26.513 seconds, p95 was 35.054 seconds, and maximum was 38.877 seconds, but
overlap raised observed throughput to about 258 queries/hour versus roughly
230/hour in the earlier serial measurement.

This is a modest ~12% throughput improvement, not a doubling: the organizer
slows individual requests under concurrency. The resulting live ETA was about
11.75 hours, roughly 1.7 hours faster than the serial estimate. Two workers are
being retained; four workers remain rejected because they previously caused a
cluster of 60-second timeouts.

### User-requested four-worker retest

The user explicitly requested another four- or five-worker attempt. Four was
selected as the smallest requested increase and tested checkpoint-safely. Run
`20260813T064901Z` completed 12 queries: nine successes and three
`RagtimeApiError` timeouts at approximately 60.15 seconds. Median duration was
52.886 seconds and effective completed throughput was about 238.5 queries/hour,
which was slower than the two-worker gate's ~258 queries/hour. Five workers was
therefore not attempted.

All nine four-worker successes were preserved. After a two-minute cooldown,
the durable tmux runner resumed with two workers as `20260813T065431Z`, with
155 validated full-run successes already checkpointed and 3,010 queries
queued. Its first five new queries succeeded with zero errors. Two workers
remain the fastest demonstrated reliable setting.

### User-requested three-worker gate

Because the user wanted to finish in time for another submission, a
checkpoint-safe three-worker gate was also tested. Run `20260813T065725Z`
completed 6/6 queries without errors, but median latency rose to 46.935 seconds
and effective throughput was approximately 230 queries/hour—slower than the
two-worker gate's ~258 queries/hour. The six successes were preserved and the
runner reverted to two workers as `20260813T070005Z`, with 178 validated
full-run successes and 2,987 queries queued at restart.

Two workers are therefore the measured throughput optimum among one through
four workers: one is reliable but slower, three adds server-side latency, and
four adds both latency and timeouts. Five or ten workers are not justified by
the measured saturation curve.

Finishing retrieval does not itself create another submission. The full richer
results still require deterministic topic fusion, downstream passage scoring
or evidence selection, nugget/report regeneration, and submission-format
validation. Existing submission artifacts were generated from earlier
candidate/nugget inputs and will not silently incorporate this run.

The immediate two-worker restart after the three-worker gate did not include a
cooldown. Its first two queries timed out, while the next five completed in
10.336–25.109 seconds. This confirms residual endpoint pressure rather than a
query-shape failure. The two-worker run remains active; those two error
checkpoints require a checkpoint-only retry pass after the first pass ends.

### Topic-major completion policy and held scoring stage

The user requested that retrieval finish every sealed query for a topic before
that topic enters later stages. The private queue changed from round-robin to
`topic-major-v1`, and the executor changed from eagerly submitting the entire
queue to keeping at most two requests in flight. An offline audit proved 2,919
pending queries formed 102 contiguous topic blocks with no repeated block; one
topic was already complete. The prior process stopped cleanly, all atomic
checkpoints remained valid, and run `20260813T073518Z` started with two workers,
246 validated full-run successes resumed, and 2,919 queries queued.

The downstream snapshot records each plan's sealed query count and has a hard
`complete-only` gate. Fourteen focused tests cover document-ID union, duplicate
content conflicts, max-over-needs scoring, rank and duplicate-frequency
independence, cache isolation, snapshot lineage, and complete-topic selection;
all 14 passed. At the gate, topic 2048 was complete with 15/15 queries, 1,008
unique candidate IDs, and one information need.

Passage scoring remains held. No `ragtime-topic-score` process, local model
run, or Runpod scoring job launched. The Framework host exposes its AMD GPU,
but the installed PyTorch build is CUDA 12.8 and reports no CUDA device; no
existing Flash endpoint is configured. Retrieval continues independently and
complete topics accumulate behind the hard gate.
