# Asymmetric Coverage-Ledger Planner Design

## Decision

Keep every already sealed `planner-static-v3` plan. For unsealed topics, use a
versioned two-stage fallback:

1. extract a compact coverage ledger containing nested information needs,
   atomic requirements, and retrieval-anchor ideas; and
2. independently reread the original topic, repair that ledger, and emit the
   complete executable query plan.

Only the final plan crosses the strict executable-plan validator. The ledger
has its own smaller schema and validation boundary, so task graph mistakes
cannot prevent the second stage from repairing coverage.

## Retrieval contract

The objective is information-need coverage and marginal candidate yield for the
hosted dense retriever. Semantic overlap is allowed when queries use different
entities, sources, measures, jurisdictions, languages, technical terms, or
evidence polarity. Pure paraphrases with unchanged anchors are discouraged.
Exact case-normalized duplicate strings remain invalid.

All executable retrieval tasks are independent and therefore have empty
`depends_on` arrays. The richer arm keeps every final task; only the separate
matched-control arm applies exact per-topic Baseline C budget `B_i`.

## Stage contracts

The coverage ledger nests requirements under information needs and therefore
does not require task IDs, dependencies, or query-to-requirement cross-reference
graphs. Each atomic requirement carries one or more short retrieval-anchor
ideas. The ledger is validated, fingerprinted, and cached independently.

The expansion stage receives the unabridged topic and validated ledger. It must
reconstruct and repair atomic coverage rather than treating the ledger as
authoritative, then emit the existing complete static-plan shape. The final
plan must cover every final requirement, contain no exact normalized duplicate
query, contain only supported query roles, impose no unsupported narrowing, and
contain no dependencies.

Use the existing generator model for both stages to isolate architecture from
model choice. The ledger allowance is 16,000 output tokens and the expansion
allowance is 32,000. Temperature remains zero, SDK retries remain zero, and the
request timeout remains finite. New operation and prompt versions isolate both
cache stages from the earlier full-plan drafts and audits.

## Gate

Run exactly one planner-only attempt on the five currently unsealed topics:
`2023`, `2024`, `2026`, `2030`, and `2031`. Do not retrieve and do not retry
inside the gate.

Continue to unseen topics only if all five seal, every original requirement is
covered by plausible candidate-expanding queries, all dependencies are empty,
there are no exact duplicates or harmful unsupported restrictions, and each
expansion finishes normally below 24,000 of its 32,000-token allowance.

The final cohort is intentionally mixed: already sealed v3 plans plus
coverage-ledger fallback plans selected before retrieval. Record generation
path per topic and stratify later retrieval results by that path. Do not report
the mixed cohort as a clean single-planner reliability estimate.
