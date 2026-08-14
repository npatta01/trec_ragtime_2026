# Two-Pass Static Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the one-pass planner with a cached draft-and-audit planning gate that exposes requirement coverage and seals only audited plans.

**Architecture:** Extend the typed static-plan schema with atomic requirements, task coverage references, query roles, stage identity, and draft anchoring. A two-pass orchestrator caches the validated draft and audited final plan separately; the existing executor accepts only the final audited plan and otherwise remains unchanged.

**Tech Stack:** Python 3.13, dataclasses, OpenAI-compatible chat completions, repository `ResponseCache`, `unittest`.

## Global Constraints

- No retrieval occurs during the planner trial.
- No provider retries are implicit or unbounded.
- No prompt, raw provider response, credential, or plaintext `.env` is persisted.
- The richer arm executes all tasks in the audited plan with no artificial 24-task cap.
- The matched arm continues to request exact per-topic Baseline C budget `B_i`.
- No result-aware replanning is introduced.

---

### Task 1: Requirement-aware plan schema

**Files:**
- Modify: `src/trec_ragtime/planner_static.py`
- Test: `tests/test_planner_static.py`

**Interfaces:**
- Produces: `PlanningRequirement`, `QueryRole`, extended `PlanTask`, and audited/draft fields on `StaticResearchPlan`.
- Produces: `parse_static_plan(..., planning_stage, draft_plan_sha256)` with deterministic coverage validation.

- [ ] **Step 1: Write a failing public-parser test**

Add a payload fixture with atomic requirements and task `requirement_refs` plus
`query_role`. Assert locally derived requirement IDs, complete coverage, and
canonical round-trip behavior.

- [ ] **Step 2: Run the focused test and confirm the expected schema failure**

Run: `uv run python -m unittest tests.test_planner_static.PlannerStaticTests.test_requirement_coverage_and_query_roles_are_sealed -v`

Expected: failure because the parser does not accept `requirements`,
`requirement_refs`, or `query_role`.

- [ ] **Step 3: Implement the minimal schema and parser behavior**

Add the requirement and role fields, derive stable IDs locally, map model refs,
reject unknown/missing refs, and require every requirement to be covered.
Version the plan schema because existing cache records are incompatible.

- [ ] **Step 4: Run focused planner schema tests**

Run: `uv run python -m unittest tests.test_planner_static.PlannerStaticTests -v`

Expected: all schema tests pass.

### Task 2: Two-pass orchestration and independent caches

**Files:**
- Modify: `src/trec_ragtime/planner_static.py`
- Test: `tests/test_planner_static.py`

**Interfaces:**
- Produces: `PlannerAuditor.audit(topic, draft, *, max_total_tasks)`.
- Produces: `StaticPlanner(..., auditor=...)` that returns only an audited final plan.

- [ ] **Step 1: Write a failing two-pass behavior test**

Use fake draft and auditor models. Assert one draft call, one audit call, the
auditor receives the original topic and draft, and the returned plan has
`planning_stage="audited"` anchored to the draft SHA-256.

- [ ] **Step 2: Run the two-pass test and confirm it fails for the missing API**

Run: `uv run python -m unittest tests.test_planner_static.PlannerStaticTests.test_planner_audits_draft_before_sealing -v`

Expected: failure because `StaticPlanner` has no auditor seam.

- [ ] **Step 3: Implement draft and audit orchestration**

Use stage-qualified cache keys. Read the audited key first, then draft. Generate
and validate only on cache misses. Aggregate stage metrics and return explicit
`draft_invalid_output`, `draft_model_failure`, `audit_invalid_output`, or
`audit_model_failure`. Cache only validated plans.

- [ ] **Step 4: Add and pass cache/failure tests**

Cover audited cache reuse, draft-only reuse after audit failure, and rejection
of unaudited plans at the executor boundary.

Run: `uv run python -m unittest tests.test_planner_static -v`

Expected: all focused tests pass.

### Task 3: Provider prompts and uncapped richer mode

**Files:**
- Modify: `src/trec_ragtime/planner_static.py`
- Test: `tests/test_planner_static.py`

**Interfaces:**
- Produces: draft and audit JSON schemas and provider methods.
- Changes: `max_total_tasks: int | None`; `None` means retain every validated task.

- [ ] **Step 1: Write failing tests for uncapped plans and prompt contracts**

Assert that a plan with more than 24 tasks is preserved when the cap is `None`,
that frontier execution requests every task, and that draft/audit provider
requests use their respective strict schemas and original topic input.

- [ ] **Step 2: Run tests and confirm expected failures**

Run: `uv run python -m unittest tests.test_planner_static -v`

Expected: failures because caps are required integers and no audit provider
method exists.

- [ ] **Step 3: Implement the provider and cap changes**

Set the CLI/default richer plan cap to `None`, retain finite provider token and
timeout limits, add explicit draft coverage instructions, and add the audit
prompt that emits a complete corrected plan.

- [ ] **Step 4: Run focused and full offline verification**

Run:

```bash
uv run python -m unittest tests.test_planner_static -v
uv run python -m unittest discover -s tests -v
uv run ruff check src/trec_ragtime/planner_static.py tests/test_planner_static.py
git diff --check
```

Expected: all tests and checks pass.

### Task 4: Bounded topic-2000 planner trial

**Files:**
- Modify: `docs/implementation/2026-08-12-planner-static-v1-evidence.md`
- Modify: `docs/implementation/2026-08-12-planner-static-v1-plan.md`
- Runtime artifact: shared private planner-static experiment cache only

**Interfaces:**
- Consumes: two-pass planner CLI with one selected topic.
- Produces: sanitized stage metrics and a validated audited plan; no retrieval.

- [ ] **Step 1: Decrypt `.env` and verify it remains ignored**

Run: `sops decrypt --output .env config/secrets.sops.env && git check-ignore .env`

Expected: `.env` is printed by `git check-ignore` and no value is printed.

- [ ] **Step 2: Run exactly one topic-2000 planner-only trial**

Use the two-pass prompt version and an explicit one-topic output path under the
shared experiment cache. Do not run retrieval.

- [ ] **Step 3: Classify failure or review the final plan**

On failure, stop and ask Fable for sanitized feedback. On success, compare the
audited requirement list and tasks against sale, consumption, parts/accessories,
jurisdictions/dates, exact law content, legal reasoning, effectiveness, and
youth/young-adult health.

- [ ] **Step 4: Record evidence and next decision**

Record stage costs, tokens, latency, cache status, plan shape, coverage verdict,
and whether retrieval remains blocked. Do not claim retrieval quality.
