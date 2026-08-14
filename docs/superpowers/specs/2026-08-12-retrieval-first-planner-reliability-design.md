# Retrieval-First Planner Reliability Design

## Objective

Make the two-pass static planner reliable enough to generate candidate-recall
queries for the downstream nugget extractor. Plan quality is judged by
information-need coverage and distinct retrieval directions, not by compactness
or report-ready precision.

## Retrieval-quality contract

The planner must extract every independently searchable information need in the
topic. It may emit multiple semantically related queries when they use different
retrieval anchors likely to move a dense retriever toward additional candidate
documents, including different entities, laws or programs, source families,
measurements, outcomes, jurisdictions, languages, technical terms, or evidence
polarity.

Pure paraphrases with the same anchors are discouraged because the hosted Qwen
embedding retriever is likely to rank nearly the same documents. Exact
case-normalized duplicate query strings remain invalid because they request the
same immutable cached result and cannot add candidates. Query-role labels are
metadata, not a checklist: the planner must not create a query merely to fill a
role category.

The coverage auditor is recall-oriented. It repairs omitted information needs,
harmful unsupported narrowing, unusable queries, and missing retrieval routes.
It preserves overlapping queries whenever their anchors may yield additional
candidates. It does not optimize for the smallest plan.

## Reliability and diagnostics

Increase the provider completion allowance from 6,000 to 16,000 tokens. Keep
the existing finite 60-second request timeout, temperature zero, and zero SDK
retries so one bounded gate has attributable calls and costs.

Invalid provider output must remain uncached and its raw content must never be
written. The sanitized run record must nevertheless preserve available input
tokens, output tokens, reported cost, latency, provider finish reason, failed
stage, and a closed failure category:

- `output_truncated`
- `invalid_json`
- `missing_content`
- `invalid_response_envelope`
- `local_validation`
- `model_failure`

Prompt versions advance to `planner-static-draft-v3` and
`planner-static-audit-v3`, creating independent cache identities without
changing or invalidating historical v2 artifacts.

## Bounded experiment

After offline verification, rerun only the seven topics that failed the prior
representative sample: `2009`, `2019`, `2037`, `2056`, `2065`, `2074`, and
`2102`. Run planning only. Do not retrieve documents.

The Luna gate passes when all seven produce sealed audited plans, every original
information need has at least one plausible retrieval query, and no query adds
a harmful unsupported restriction. Overlapping retrieval routes and large task
counts are allowed. Exact duplicates remain invalid.

If structural failures remain or information needs are systematically omitted,
stop and seek Sol advice before changing the architecture or running all 103
topics. If the gate passes, inspect the seven plans and obtain the previously
requested approval panel before an all-topic planner run.

## Scope and safety

- No artificial richer-arm task cap.
- No retrieval, report synthesis, nugget extraction, submission, push, deploy,
  or PR in this revision.
- No credentials, raw invalid provider output, prompts, or response bodies in
  run artifacts or logs.
- Preserve the exact per-topic Baseline C budget only for the separate matched
  control arm.
