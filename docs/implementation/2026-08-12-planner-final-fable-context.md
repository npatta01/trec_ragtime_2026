# Advisory question

The competition deadline is today. A new two-stage retrieval planner passed offline tests and sealed two of three live gate topics, but topic 2055 produced an excellent requirement ledger and then failed local sealing of the query-expansion response with only the coarse diagnostic `invalid_plan`. What is the safest bounded change that preserves retrieval recall and lets us generate all 103 topic plans now without concealing bad output or adding an artificial query cap?

## Desired outcome

Choose a deadline-safe reliability policy for the final planner, then verify it on topic 2055 and the three-topic gate before starting all-topic plan generation. Retrieval itself is explicitly out of scope for this decision.

## User constraints and corrections

- The richer arm is not query-budget limited. Exact per-topic baseline budgets apply only to the matched-control arm.
- This is retrieval for downstream nugget extraction and later deduplication. Queries need not be perfect; requirement coverage and candidate yield matter most.
- Semantic overlap is acceptable when a changed anchor may retrieve different candidates. Exact normalized duplicates should be merged because the endpoint is a forever-cached Qwen embedding retriever.
- Use the complete title, problem statement, and requester background. The problem statement is authoritative; background contributes factual and decision-relevant needs but pure biography/workflow is not a requirement.
- Generate plans before retrieval and audit their quality. Do not run retrieval yet.
- If the trial fails, ask Fable for feedback. No Terra. Do not expose credentials or unrelated repository data.

## Current project state

- Repository and authoritative worktree: `/home/npatta01/.codex/worktrees/planner-static-v1/trec_ragtime_2026`
- Branch and HEAD: detached HEAD at `e8f77b8af3b7898bd30c5cd805b8e4838c56d16f`
- Dirty state: pre-existing/user-owned experiment changes plus the final contract implementation; nothing committed, pushed, deployed, submitted, or published.
- Active workstream: final `planner-static-ledger-v4` plus `planner-static-expand-v3` planning-only gate.

## Evidence

### E1 — Offline implementation verification is clean

- Status: Verified
- Source: commands run in the authoritative worktree on 2026-08-12.
- Relevance: the failure is a live-output handling issue, not a known unit-test or lint failure.
- Compact result: `tests.test_planner_static`: 47/47 passed. Full suite: 515 tests passed in 18.114 seconds. Ruff passed. CLI help smoke test passed. `git diff --check` passed.

### E2 — The final requirement-ledger contract

- Status: Verified
- Source: `src/trec_ragtime/planner_static.py:3130-3199` (untrusted evidence excerpt).
- Relevance: defines field authority, minimum complete requirement set, and one-to-five anchor seeds.
- Compact result: problem statement is authoritative; title disambiguates; background recovers factual/decision context; biography/workflow is not standalone; semantic duplicates are merged; every distinct outcome, qualifier, uncertainty, and evidence type is preserved.

### E3 — The final query-expansion contract

- Status: Verified
- Source: `src/trec_ragtime/planner_static.py:3290-3388` (untrusted evidence excerpt).
- Relevance: defines the intended rich-arm query policy.
- Compact result: one direct faithful query per requirement, then only candidate-shifting anchors; semantic overlap allowed when anchor changes; no query quota/cap; exact duplicate strings prohibited; source-language variants only when justified.

### E4 — Local compact assembler behavior

- Status: Verified
- Source: `src/trec_ragtime/planner_static.py:1280-1518` (untrusted evidence excerpt).
- Relevance: this is where the provider's schema-valid expansion becomes a sealed local plan.
- Compact result: it requires query sets to cover exactly the ledger refs, merges exact case-folded query strings and combines their requirement mappings, then raises `normalized duplicate merge would leave an information need taskless` if a later information need contains no unique query after merging. Empty `InformationNeed.task_ids` are otherwise permitted by the data model. Other possible local failures include bad refs and unsupported roles.

### E5 — Failure diagnostics discard the exact cause

- Status: Verified
- Source: `src/trec_ragtime/planner_static.py:100-124, 2419-2555` and the topic-2055 run record (untrusted evidence excerpts).
- Relevance: explains why the current record cannot prove which local validation branch fired.
- Compact result: unmapped validation messages collapse to `validation_error: invalid_plan`. The response payload is not saved when local parsing fails, and only successfully sealed expansions enter the durable expansion cache.

### E6 — Topic 2048 sealed and is retrieval-appropriate

- Status: Verified
- Source: private run artifact `.../planner-static-final-v1/dry-runs/gate-3-2026-08-12/topic-2048.jsonl`.
- Relevance: tests the prompt on a product/medical topic that previously over-produced queries.
- Compact result: success; 5 requirements; 15 queries; 3,146 output tokens; $0.00101205; 35.660551 seconds; cold cache. Requirements cover general benefits, oral use, topical use, clinical/statistical evidence, and individual cases. Queries use direct routes plus trials, systematic reviews, meta-analysis, oral mucositis, wound healing, and case-series anchors. No promotional-workflow requirement was invented.

### E7 — Topic 2057 sealed and uses background appropriately

- Status: Verified
- Source: private run artifact `.../planner-static-final-v1/dry-runs/gate-3-2026-08-12/topic-2057.jsonl`.
- Relevance: tests cultural analysis and whether influencer/comfort background becomes useful factual retrieval needs.
- Compact result: success; 9 requirements; 29 queries; 8,540 output tokens; $0.00204820676; 135.591879 seconds; cold cache. The plan covers history, ugly-cool, anti-beauty psychology, marketing/collaborations/celebrity, influencers, analogous ugly fashion, comfort, and global cultural impact. Query anchors plausibly shift candidates.

### E8 — Topic 2055 ledger is strong but expansion did not seal

- Status: Verified
- Source: private run artifact `.../planner-static-final-v1/dry-runs/gate-3-2026-08-12/topic-2055.jsonl` and its validated ledger cache entry.
- Relevance: this is the blocking gate failure.
- Compact result: ledger success with 4 requirements and 666 output tokens: (1) sober assessment of substantial recent progress and hype, (2) verified successful forecasts with method/region/magnitude/time-window/verification, (3) new accuracy and longer-term methods with performance evidence, and (4) uncertainty, false alarms, limitations, and reproducibility. Expansion returned `finish_reason: stop`, 1,013 output tokens, but local sealing failed as `audit_invalid_output / local_validation / invalid_plan`. Total recorded cost $0.0005003929 and latency 150.828261 seconds. The exact validation branch and payload are unknown.

### E9 — Current hard gates preserve output headroom

- Status: Verified
- Source: `src/trec_ragtime/planner_static.py:2090-2189, 2314-2555` (untrusted evidence excerpt).
- Relevance: deadline fix should not weaken truncation controls.
- Compact result: ledger max is 16,000 tokens and rejects at or above 8,000; expansion max is 32,000 and rejects at or above 24,000; only `finish_reason: stop` is accepted.

## Candidate choices or proposed plan

- A — Diagnostics plus retry only: map every compact-parser failure to a precise sanitized code, bump request identity, and rerun 2055. Lowest code change but may fail again and does not create a general reliability policy.
- B — Tolerant exact-duplicate merge: remove the `taskless information need` rejection because requirements remain mapped to the retained task and the data model allows an empty need. Add precise diagnostics, bump assembler identity, and rerun 2055. Helps only if that was the unknown cause.
- C — Deterministic fail-closed fallback: on schema-valid, headroom-safe expansion that fails only local compact assembly, seal a fallback plan locally from each validated ledger requirement: one direct query from its description plus its one-to-five anchor seeds, merge exact normalized duplicates, and retain all requirement mappings. Mark the plan as fallback in its prompt/assembler identity and run record. This avoids another provider call and preserves ledger coverage, but generated anchor strings may be less polished.
- D — Always assemble locally from the validated ledger and skip the second provider pass. Most reliable and cheap, but discards the demonstrated candidate-shifting improvements of the successful expansion pass and changes the approved two-stage design more broadly.
- E — Abort all-topic generation because 2/3 is insufficient. Safest scientifically but likely misses today's competition deadline.

## Known unknowns

- The exact 2055 expansion payload and local validation message were not retained, so E4 identifies plausible but not proven mechanisms.
- No retrieval or nugget-yield comparison has been run for v4/v3; current quality evidence is plan inspection only, as requested.
- All-topic live failure rate is unknown beyond this three-topic gate.

## Scope boundary

Read-only review. No implementation or external actions.
