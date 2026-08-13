# Deadline-Final Planner Contract Design

## Objective

Freeze a reliable two-stage static planner for all 103 RAGTIME 2026 topics.
The planner must use the complete title, problem statement, and requester
background while distinguishing required report content from incidental user
biography or workflow. It must maximize information-need coverage and marginal
candidate yield before retrieval.

## Chosen approach

Retain the provider-tested `coverage ledger -> compact query expansion -> local
sealed plan` architecture. Do not switch to the less-tested anchorless ledger
or return to full-plan re-emission. Version both prompt identities so existing
validated cache artifacts remain immutable and distinguishable.

## Input authority

- The problem statement is authoritative for what must and must not appear.
- The title disambiguates the subject.
- The background contributes factual questions, comparisons, constraints,
  outcomes, audience needs, and decision-relevant context not already present.
- Pure biography or workflow, such as preparing a presentation, affects
  priority or framing but is not a standalone retrieval requirement.
- The report character limit is not semantic query content.

## Requirement ledger

The ledger emits the smallest complete set of atomic, independently answerable
requirements. It merges restatements across the problem statement and
background. It does not split a requirement by source type, query strategy,
terminology, or evidence polarity. It preserves distinct requested entities,
comparisons, qualifiers, populations, jurisdictions, outcomes, timeframes,
uncertainty, limitations, and evidence types.

Each requirement may include one to five short retrieval-anchor seeds. They
are non-exhaustive hints for query generation, not a query or task cap.
Requests such as “if any,” “whether,” “balanced,” “sober assessment,” or
“honest numbers” license verification and contrary-evidence routes.

The ledger retains the 16,000-token provider ceiling but is rejected and not
cached at 8,000 output tokens or more. That threshold quarantines degenerate
emission; it is not a requirement-count cap.

## Query generation

The expansion stage rereads the complete original topic and consumes the
validated ledger. It emits every requirement reference exactly once with one
or more queries. Each requirement starts with a faithful direct query.
Additional queries are included only when they introduce a plausible
candidate-shifting anchor: entity or alias, technical terminology, source
family, requested measure or case, topic-licensed jurisdiction/language/
population/timeframe, or contrary/negative/limitation evidence.

Semantic overlap remains allowed when the anchor differs. Pure paraphrases are
discouraged. Exact whitespace-normalized, case-folded duplicates are merged
locally and all requirement mappings are retained. Source-language variants
are generated only when a named jurisdiction, entity, or terminology makes
them plausibly candidate-expanding. Query-role labels are audit metadata, not
a quota. The richer arm has no query cap; matched control alone uses `B_i`.

## Reliability and execution gate

- Expansion retains its 32,000-token ceiling and 24,000-token headroom gate.
- Provider output must be one schema-valid JSON object with no prose.
- Failed or quarantined stages are not cached as successes.
- A small varied-topic planner-only gate is manually audited before broader
  planning.
- The gate checks source-clause coverage, absence of workflow-only
  requirements, query usefulness, unsupported narrowing, exact duplicates,
  requirement mappings, dependencies, output size, cost, and latency.
- Only after the gate passes may the remaining topic plans be generated.
- Retrieval, report writing, submission, and deployment are out of scope.

## Success criteria

- Every source-backed factual need is represented once in the ledger.
- No standalone requirement exists solely because of requester biography or
  workflow.
- Every ledger requirement has at least one sealed retrieval query.
- No plan has unresolved mappings, dependencies, exact normalized duplicate
  queries, or richer-arm truncation.
- The small provider gate passes manual quality review before all-topic
  planner generation begins.
