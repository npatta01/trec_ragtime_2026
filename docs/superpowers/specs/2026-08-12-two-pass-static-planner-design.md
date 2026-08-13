# Two-Pass Static Planner Design

## Goal

Produce a sealed static retrieval plan only after a draft planner and an
independent coverage auditor have both compared their work with the original
topic. Retrieval must never begin from an unaudited draft.

## Planning stages

The draft stage atomizes the topic into explicit requirements, groups those
requirements into information needs, and creates complementary query variants.
Every task declares which requirements it targets and one query role:

- `broad_discovery`
- `precise_facts`
- `official_primary`
- `measurable_outcomes`
- `contrary_evidence`
- `comparison`

The audit stage receives the unabridged original topic and the complete draft.
It emits a repaired replacement plan, not comments or a patch. Its prompt
requires it to recover omissions, remove unsupported temporal or geographic
restrictions, reduce semantic duplication, and ensure requested outcomes are
operationalized rather than replaced by easier proxy questions.

## Validation and sealing

Model references are untrusted labels. Stable requirement, need, and task IDs
are derived locally from topic content, normalized record content, and ordinal
tie-breakers. The parser rejects a plan unless:

- every declared requirement is targeted by at least one task;
- every task targets at least one known requirement;
- query roles are from the closed vocabulary;
- normalized query text is unique;
- dependencies reference earlier tasks; and
- all existing structural and fingerprint checks pass.

The final plan records `planning_stage="audited"` and the exact draft plan
SHA-256. The executor rejects a draft-stage plan. Planning remains sealed before
retrieval and never inspects retrieval results.

## Caching and failure behavior

Draft and audited plans use separate cache identities. The audited identity
includes the draft plan SHA-256. A valid audited cache hit skips both provider
calls. A draft cache hit may skip draft generation while still allowing an
audit retry. Invalid outputs and failures are never cached.

Planner metrics report stage-specific cache hits, tokens, latency, cost, and
failure stage without storing prompts, provider response bodies, or secrets.

## Query budgets

The richer arm executes every task in the audited plan. It has no artificial
24-task admission limit. The provider response remains finite because each call
has a token and timeout limit. The matched-control arm continues to execute
exactly the per-topic Baseline C query count `B_i`; plan shortfall remains an
explicit failure.

## Trial and review gate

Offline tests cover schema validation, draft/audit cache behavior, audit
failure, full richer execution, and matched-prefix execution. After offline
verification, one planner-only topic-2000 trial may run. No retrieval is part
of the trial. Its final plan is manually compared with the topic requirements
before any richer-arm execution decision.

If the trial fails, provide Fable only the sanitized failure classification,
the relevant prompt/schema source, and the topic text for feedback. Do not send
credentials, raw logs, or unrelated repository context.
