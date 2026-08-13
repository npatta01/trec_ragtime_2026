# Deadline-Final Planner Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze, verify, and gate the final two-stage requirement-ledger and query-generation contract before generating all remaining RAGTIME topic plans.

**Architecture:** Keep the existing `OpenAIStaticPlannerModel` and `AsymmetricStaticPlanner` boundaries. Version and tighten the ledger and compact-expansion prompts, enforce a pre-cache ledger output quarantine and bounded anchor list, and reuse deterministic local plan assembly.

**Tech Stack:** Python 3.13, `unittest`, OpenAI-compatible chat completions, repository artifact caches, Ruff.

## Global Constraints

- Use title, problem statement, and background; do not turn pure biography or workflow into requirements.
- Preserve the richer arm with `max_total_tasks=None`; exact `B_i` belongs only to matched control.
- Preserve semantic variants with distinct anchors; merge only whitespace-normalized, case-folded exact strings.
- Ledger provider ceiling is 16,000 tokens and quarantine threshold is 8,000 output tokens.
- Expansion provider ceiling is 32,000 tokens and headroom threshold is 24,000 output tokens.
- Do not retrieve, submit, deploy, push, open a PR, commit, expose credentials, or touch another worktree.

---

### Task 1: Versioned prompt contracts

**Files:**
- Modify: `src/trec_ragtime/planner_static.py`
- Test: `tests/test_planner_static.py`

**Interfaces:**
- Consumes: `OpenAIStaticPlannerModel.generate_coverage_ledger()` and `expand_coverage_ledger()`.
- Produces: `planner-static-ledger-v4` and `planner-static-expand-v3` provider request identities.

- [ ] Add provider-request tests asserting the complete field-authority,
      deduplication, granularity, source-language, marginal-yield, and no-cap
      instructions.
- [ ] Run the new tests and confirm they fail because the v4/v3 instructions
      and identities are absent.
- [ ] Implement the minimal versioned prompts and select the existing ledger
      and compact query schemas for those versions.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Ledger size and anchor gates

**Files:**
- Modify: `src/trec_ragtime/planner_static.py`
- Test: `tests/test_planner_static.py`

**Interfaces:**
- Consumes: `PlannerStageMetrics` and `COVERAGE_LEDGER_JSON_SCHEMA`.
- Produces: `ledger_gate_max_output_tokens=8_000` in planner identity and
  pre-cache validation; one-to-five anchor schema.

- [ ] Add tests proving a ledger at 8,000 output tokens is rejected before
      caching, a smaller valid ledger is accepted, and changing the quarantine
      threshold changes the cache key.
- [ ] Run the tests and confirm the 8,000-token case is currently accepted.
- [ ] Add `maxItems: 5`, the planner threshold, cache identity, live check,
      and cached-entry validation.
- [ ] Run focused planner tests and confirm they pass.

### Task 3: Offline regression verification

**Files:**
- Verify: `src/trec_ragtime/planner_static.py`
- Verify: `tests/test_planner_static.py`

**Interfaces:**
- Consumes: the completed prompt and ledger-gate implementation.
- Produces: fresh test, lint, CLI, and diff evidence.

- [ ] Run `UV_OFFLINE=1 uv run python -m unittest tests.test_planner_static -v`.
- [ ] Run `UV_OFFLINE=1 uv run python -m unittest discover -s tests -v`.
- [ ] Run `UV_OFFLINE=1 uv run ruff check src/trec_ragtime/planner_static.py tests/test_planner_static.py`.
- [ ] Run `UV_OFFLINE=1 uv run ragtime-static-plan --help` and `git diff --check`.

### Task 4: Small provider gate and manual audit

**Files:**
- Write private artifacts under: `/home/npatta01/data/competitions/trec_ragtime_2026/.git/ragtime-cache/experiments/planner-static-final-v1/dry-runs/`
- Update: `docs/implementation/2026-08-12-planner-static-v1-evidence.md`

**Interfaces:**
- Consumes: exact topic selection, final v4/v3 planner path, shared validated caches.
- Produces: sanitized JSONL plans and a pass/fail audit decision.

- [ ] Select three varied topics that expose background handling, empirical
      uncertainty, and broad multi-facet coverage.
- [ ] Run only those planner calls with no retrieval.
- [ ] Audit every requirement against title/problem/background and inspect
      every query for a plausible candidate-shifting route.
- [ ] Stop if any topic fails or has workflow-only requirements, unsupported
      narrowing, missing coverage, degenerate output, or unusable query mass.

### Task 5: Remaining-topic planning and aggregate audit

**Files:**
- Write private artifacts under: `/home/npatta01/data/competitions/trec_ragtime_2026/.git/ragtime-cache/experiments/planner-static-final-v1/dry-runs/`
- Update: `docs/implementation/2026-08-12-planner-static-v1-evidence.md`
- Update: `docs/implementation/2026-08-12-planner-static-v1-plan.md`

**Interfaces:**
- Consumes: the passed small gate and exact list of topics lacking a final-v1 plan.
- Produces: one final sealed plan per topic plus sanitized aggregate quality,
  cost, latency, and cache evidence.

- [ ] Generate only missing final-v1 plans in checkpointed batches.
- [ ] After each batch, verify status, requirement coverage, duplicate count,
      dependency count, task count, output headroom, cache behavior, cost, and
      latency before continuing.
- [ ] Manually audit a stratified sample including the largest plans and all
      unusual failures or retries.
- [ ] Stop without retrieval if the final cohort does not meet the design’s
      success criteria.
