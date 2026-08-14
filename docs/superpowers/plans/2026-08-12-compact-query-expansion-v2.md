# Compact Query Expansion v2 Implementation Plan

**Goal:** Seal topics 2023 and 2031 without repeating successful stages by
shrinking model serialization and hardening JSON envelope handling.

## Tasks

- [ ] Add failing tests for compact requirement-to-query parsing, exact ledger
  coverage, deterministic final-plan assembly, no dependencies, and uncapped
  task retention.
- [ ] Add failing tests for fenced/prefixed JSON extraction, sanitized parse
  fingerprints, and one ledger-v2 repair call with aggregate metrics.
- [ ] Implement the compact expansion schema/parser and select it only for
  `planner-static-expand-v2`.
- [ ] Implement in-memory envelope hardening without storing raw content.
- [ ] Run planner tests, full offline tests, Ruff, CLI help, and diff checks;
  obtain independent review.
- [ ] Run topic 2023 with ledger v1/expansion v2 and topic 2031 with ledger
  v2/expansion v2. Do not retry beyond the ledger-v2 contract.
- [ ] Audit both plans, then all five asymmetric-gate plans, before any unseen
  topic call.
- [ ] Update the canonical plan and evidence log.
