---
name: validate-submission
description: Validate a TREC RAGTIME 2026 submission JSONL against the organizers' AutoJudge checker (autojudge-base report_tool, ragtime26 spec) before submitting. Use when asked to check, validate, or lint a submission/run file, when someone mentions autojudge, report_tool, the ragtime26 or rag26 spec, or when a run is about to be submitted to NIST.
---

# Validate a RAGTIME submission

Runs Laura Dietz's `autojudge-base` report checker over one or more submission
JSONL files and reports what would be rejected. **Exit code 255 means the
submission has at least one hard failure; 0 means it passes.**

## Run it

```bash
uv run ragtime-validate-submission path/to/ragtime26.submission.jsonl
```

That is the whole workflow for the common case. It defaults to
`--spec ragtime26` and to the official
[`topics/topics.all.2026.v0625-fix.jsonl`](../../../topics/topics.all.2026.v0625-fix.jsonl),
and it accepts several run files at once.

Useful flags:

| Flag | Effect |
| --- | --- |
| `--strict` | Treat every smell (warning) as a hard failure. Use before a final submit. |
| `--suppress-warnings` | Hide smells; show only hard problems. |
| `--no-topics` | Skip the topic-coverage and per-topic length cross-check. |
| `--spec rag26` | Check against a different track spec. |
| `--requests-out PATH` | Keep the translated topics file (see below) instead of discarding it. |

The command echoes the exact underlying `report_tool` invocation before running
it, so the raw command can be copied and rerun by hand.

## Reading the output

- **PROBLEM** — hard failure. Fix before submitting; causes exit 255.
- **SMELL** — warning only. Accepted by the organizers but usually worth fixing;
  `--strict` promotes these to failures.
- **`[MISSING]` / `[DUP]` / `[NEW]`** — topic-coverage findings, only produced
  when topics are supplied. For `ragtime26`, a missing or unknown topic is a
  hard error; a duplicate topic id is a smell.
- Findings are collated by kind: one example message, then the run files and
  topic ids where that kind occurs.

## What `ragtime26` actually enforces

Grounded in `autojudge_base/track_specs.yml` (v0.4.4). Report this rather than
guessing when someone asks why a check fired:

- Sentences live under **`responses`**; each is `{"text": ..., "citations": ...}`.
  Citations are a `{doc_id: score}` map (or a plain doc-id list).
- Required metadata: **`team_id`, `topic_id`, `run_id`**. `run_desc` is
  recommended — its absence is a smell.
- `run_id` is capped at **20 characters** (`A-Z a-z 0-9 - _ .`, not starting
  with a period). Over-length is a **hard failure**.
- Cited doc-ids must match the NeuCLIR pattern
  `[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_\d+`
  (uuid + `_` + segment). A malformed id is a **hard failure**.
- Max **3 citations per sentence** — a smell, not a failure (extra citations
  are simply not judged).
- Length is the topic's own `limit` counted in NFKC **characters**, not words.
  Over-length is a smell: organizers truncate rather than reject.
- **`references` is not checked at all** for 2026 — absent, empty, or
  mismatched all pass. Do not spend effort on it.
- Empty answers, blank sentences, and duplicate topic ids are smells.

## Why the wrapper exists

Two things the raw command cannot do here:

1. The official topics file uses `topic_id` / `collection_id`, but the checker
   parses `--topics` into a model requiring `request_id` / `collection_ids`.
   Passing the official file straight to `report_tool` fails to load, silently
   costing the coverage and length checks. The wrapper translates it into a
   throwaway requests file and leaves the pinned official topics untouched.
2. `autojudge-base` is a submission-time checker, not a pipeline dependency, so
   it is deliberately absent from `pyproject.toml`. The wrapper uses it if it is
   importable and otherwise fetches it through
   `uv run --isolated --no-project --with 'autojudge-base>=0.4.3'`, leaving the
   project environment and lock file alone.

The implementation is
[`src/trec_ragtime/submission_validation.py`](../../../src/trec_ragtime/submission_validation.py),
covered by [`tests/test_submission_validation.py`](../../../tests/test_submission_validation.py).

## Raw checker, without the wrapper

Only for a submission whose topics are already in `request_id` form, or when
checking without topics:

```bash
uv run --isolated --no-project --with 'autojudge-base>=0.4.3' python -m autojudge_base.report_tool check --spec ragtime26 path/to/submission.jsonl
```

`report_tool` also offers `verify` (fails fast at the first violation, but
ignores `--topics`, so it skips coverage and length) and `convert --to <spec>`.
