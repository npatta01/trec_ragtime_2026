# Compact Query Expansion v2 Design

## Decision

Replace full-plan model re-emission with a compact expansion schema for
currently unsealed coverage-ledger topics. The model emits one query set for
each validated ledger requirement. Each query carries only its text, role, and
the retrieval-anchor dimension that makes it candidate-expanding. Local code
deterministically constructs requirement records, information needs, task IDs,
evidence targets, priorities, mappings, and empty dependencies before applying
the existing strict final-plan validator.

This is a serialization reliability change, not a smaller retrieval plan. No
global or per-requirement task-count cap is introduced. Exact normalized query
strings remain invalid; semantic overlap remains acceptable when anchor
dimensions differ.

## Envelope hardening

Continue to request strict provider JSON schema. Before declaring invalid JSON,
allow an in-memory parser to extract the first complete JSON object from a
markdown fence or prose prefix, then apply the same strict local schema. Never
store or log the raw body.

For ledger prompt v2 only, one invalid-JSON response may trigger one explicit
JSON-only repair call using the same topic and a sanitized defect class, not the
raw response. Aggregate both calls' usage, cost, and latency in the stage
metrics. A final invalid response records only a closed content-shape class and
parse-error offset; it never records response text.

## Bounded gate

- Topic 2023: reuse its validated `planner-static-ledger-v1` artifact and call
  only `planner-static-expand-v2`.
- Topic 2031: call `planner-static-ledger-v2`, with at most one JSON repair,
  followed by `planner-static-expand-v2`.
- Keep the 32k provider ceiling, hard `finish_reason=length` rejection, and
  pre-cache 24k headroom gate.
- Do not rerun any sealed stage or topic and do not retrieve.

Both topics must seal with complete ledger-requirement coverage, exact-string
deduplication, empty dependencies, plausible distinct retrieval anchors, no
harmful unsupported narrowing, and normal expansion completion in the healthy
1k–8k range. If 2023 still truncates, use Sol only for that expansion under the
same compact contract. If 2031 still has an envelope failure after hardening,
inspect only its sanitized fingerprint before deciding whether to use Sol.
