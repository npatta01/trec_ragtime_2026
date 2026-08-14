# Retrieval-First Planner Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve two-pass planner reliability and candidate-recall query generation, then rerun only the seven previously failed topics without retrieval.

**Architecture:** Keep the existing draft-to-audit planner and sealed plan schema. Increase the provider output budget, revise both prompts around distinct dense-retrieval anchors, and carry sanitized provider failure diagnostics through a typed exception into planner-only run records. New prompt versions isolate all new calls from historical v2 cache entries.

**Tech Stack:** Python 3.13, OpenAI-compatible chat completions, dataclasses, unittest, Ruff, repository artifact cache.

## Global Constraints

- Use Luna xhigh; escalate to Sol only if the bounded revised gate fails.
- Keep `max_total_tasks=None` for the richer arm.
- Set provider `max_tokens=16000`, timeout 60 seconds, temperature zero, and zero SDK retries.
- Keep exact normalized duplicate queries invalid; permit semantic overlap with distinct retrieval anchors.
- Never cache or store raw invalid provider content.
- Do not run retrieval, all 103 topics, push, deploy, submit, or open a PR.
- Do not commit because the owner task explicitly prohibits repository integration actions.

---

### Task 1: Retrieval-first prompts and output allowance

**Files:**
- Modify: `src/trec_ragtime/planner_static.py`
- Test: `tests/test_planner_static.py`

**Interfaces:**
- Consumes: `OpenAIStaticPlannerModel.generate()` and `.audit()`.
- Produces: v3 prompt defaults and provider requests with `max_tokens=16000`.

- [ ] Add a provider-contract test asserting the 16,000-token request, v3 prompt versions, candidate-recall objective, distinct retrieval anchors, no role-filling, and recall-preserving auditor language.
- [ ] Run the focused test and confirm it fails against the v2/6,000-token implementation.
- [ ] Update constants, default output allowance, and draft/audit prompt text with no plan-schema change.
- [ ] Run the focused test and all existing planner tests.

### Task 2: Sanitized invalid-output diagnostics

**Files:**
- Modify: `src/trec_ragtime/planner_static.py`
- Test: `tests/test_planner_static.py`

**Interfaces:**
- Produces: `PlannerCallError`, extended `PlannerResponse`/`PlannerOutcome` diagnostics, and planner-only JSONL fields `failure_stage`, `failure_detail`, and `finish_reason`.
- Guarantees: raw message content is not attached to the exception or outcome.

- [ ] Add failing tests for a `finish_reason="length"` invalid JSON response and a well-formed but locally invalid plan.
- [ ] Assert both retain available usage, cost, and latency; classify respectively as `output_truncated` and `local_validation`; and serialize no message or response body.
- [ ] Implement the typed sanitized provider error and pass its metrics into `_failed_outcome`.
- [ ] Add finish reason to successful response envelopes and map parse/envelope failures to the closed categories.
- [ ] Include sanitized diagnostics in planner-only and static-run records.
- [ ] Run focused planner tests.

### Task 3: Offline verification and failed-topic gate

**Files:**
- Modify: `docs/implementation/2026-08-12-planner-static-v1-evidence.md`
- Modify: `docs/implementation/2026-08-12-planner-static-v1-plan.md`

**Interfaces:**
- Consumes: the v3 CLI and seven-topic temporary input derived from the canonical topic file.
- Produces: an owner-only planner-only JSONL artifact and a manual need/query audit.

- [ ] Decrypt `.env`, confirm it is ignored, run focused planner tests, the complete offline suite, Ruff, CLI help, and `git diff --check`.
- [ ] Build a temporary seven-topic input containing only `2009`, `2019`, `2037`, `2056`, `2065`, `2074`, and `2102`.
- [ ] Run exactly one v3 planner-only attempt for those seven topics with no retrieval and no retries.
- [ ] Inspect status, classified failure diagnostics, tokens, cost, latency, cache behavior, requirements, and queries.
- [ ] Manually compare each successful plan to its original topic for omitted needs, harmful narrowing, and plausible distinct retrieval anchors; allow semantic overlap and large task counts.
- [ ] If all seven pass, request the approval panel. If any systemic failure remains, stop and seek Sol advice instead of running all topics.
- [ ] Update the evidence and canonical plan with the verified outcome and next action.
