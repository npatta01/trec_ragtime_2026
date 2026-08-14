# Asymmetric Coverage-Ledger Planner Implementation Plan

**Goal:** Recover unsealed topics with a compact coverage-ledger pass followed
by a strict dependency-free query-expansion pass, without repeating sealed
provider work.

## Constraints

- Keep the generator model fixed; Luna xhigh orchestrates and Sol is advisory.
- Ledger output limit: 16,000. Expansion output limit: 32,000.
- No automatic retries and no retrieval.
- Keep semantic variants with distinct retrieval anchors; reject exact query
  strings after normalization.
- Preserve all existing sealed v3 plans and record the planner path.
- Do not commit, push, deploy, submit, or open a PR.

## Tasks

- [x] Add failing tests for the nested ledger schema, local fingerprint,
  independent ledger/final cache identities, dependency-free final boundary,
  and separate stage metrics.
- [x] Implement typed ledger parsing/caching and the asymmetric planner without
  changing the legacy v3 full-plan path.
- [x] Add provider-contract tests for the short ledger prompt and 32k final
  expansion prompt.
- [x] Run planner tests, the complete offline suite, Ruff, CLI help, and
  `git diff --check`.
- [ ] Run one no-retry planner-only attempt for topics 2023, 2024, 2026, 2030,
  and 2031.
- [ ] Audit coverage, distinct anchors, unsupported narrowing, exact
  duplicates, empty dependencies, finish reasons, and expansion headroom.
- [ ] If the gate passes, run only unseen topics; otherwise stop.
- [ ] Assemble exactly one final sealed plan per topic, audit quality, and
  update the canonical evidence log.
