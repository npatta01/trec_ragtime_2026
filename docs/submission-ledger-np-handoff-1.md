# Submission ledger — `np-handoff-1`

TREC RAGTIME 2026, **Report generation** task.
Recorded 2026-08-13. Everything below was read off the artifacts, not from memory.

---

## 0. Values entered in EvalBase

Copy-paste exactly. Form: *Submission to RAG TREC Instrument for Multilingual
Evaluation, Report generation*.

| Field | Value entered |
| --- | --- |
| User name | `npatta01` *(prefilled)* |
| Email | `npatta01@gmail.com` *(prefilled)* |
| Organization | `(NP Labs) NP` *(prefilled)* |
| Submission file | `NP.np-handoff-1.ragtime26.jsonl` |
| **Runtag** | `np-handoff-1` |
| **Description** | the block in §4 below |
| **Priority** | `1` |

The uploaded file is committed to this repository at
[`submissions/NP.np-handoff-1.ragtime26.jsonl`](../submissions/NP.np-handoff-1.ragtime26.jsonl),
with its checksum beside it. It is byte-identical to the validated run output —
same SHA-256 in all three places (run directory, upload staging directory, and
this repository).

**Submitted:** _record the date/time here once the upload is confirmed._

---

## 1. What to put in the EvalBase form

The form is *Submission to RAG TREC Instrument for Multilingual Evaluation, Report generation*.
Three fields are prefilled from the account; three need values.

| Form field | Value | Notes |
| --- | --- | --- |
| User name | `npatta01` | prefilled |
| Email | `npatta01@gmail.com` | prefilled |
| Organization | `(NP Labs) NP` | prefilled |
| **Submission file** | `NP.np-handoff-1.ragtime26.jsonl` | see §2 for the path |
| **Runtag** | `np-handoff-1` | see §3 — must match `run_id` inside the file |
| **Description** | see §4 | free text |
| **Priority** | `1` | 1 = highest. See §5 before changing. |

---

## 2. The file to upload

In this repository (tracked, reviewable in the PR):

```
submissions/NP.np-handoff-1.ragtime26.jsonl
```

Staging copy outside the repo, identical bytes:

```
/home/npatta01/data/competitions/trec_ragtime_2026/.git/ragtime-cache/experiments/reports/submit/NP.np-handoff-1.ragtime26.jsonl
```

| | |
| --- | --- |
| Size | 765,155 bytes (747 KB) |
| SHA-256 | `e39dd2617ceeb53bd1c2c1be8bd0b78bd81edf879adffb379416f0b9f6a73e52` |
| Lines | 103 — one JSON object per topic |
| Format | plain JSONL (EvalBase also accepts gzip/bzip2/xz; no need to compress at this size) |

A `.sha256` file sits beside it. To confirm the upload is the file you expect:

```bash
cd submissions && sha256sum -c NP.np-handoff-1.ragtime26.jsonl.sha256
```

This is a byte-identical copy of the run's own `submission.jsonl`, renamed for a
recognisable upload. The original stays untouched at
`…/experiments/reports/np-handoff-1/submission.jsonl`.

---

## 3. Runtag — why `np-handoff-1`

EvalBase requires: letters, numbers, underscores, periods (not first character)
or hyphens only, **20 characters or less**.

| Check | Result |
| --- | --- |
| Length | 12 ✅ (limit 20) |
| Charset `[A-Za-z0-9._-]` | ✅ |
| Leading period | none ✅ |
| Matches `run_id` inside every record | ✅ all 103 |

**Keep the runtag identical to the `run_id` in the file.** The AutoJudge spec
independently caps `run_id` at 20 characters and treats an over-length value as
a hard failure, so a mismatch between form and file is the easiest way to get a
run rejected or mis-attributed.

---

## 4. Description of the run

Paste this into the description box:

```
Nugget-grounded report generation over a sealed retrieval handoff.

Pipeline: multilingual dense retrieval (RoutIR/Qwen3-Embedding-8B) with
facet-decomposed queries, English report_text hydration from a
revision-pinned local corpus, GPU passage scoring, nuggetization into
atomic claims with a per-passage support matrix, then a self-contained
retrieval handoff bundle carrying only evidence that a retained nugget
cites.

Generation is one call per topic against openai/gpt-5.6-sol at
temperature 0, seed 7, constrained by a strict JSON schema whose evidence
enum contains only citation-ready passage handles. Canonical document IDs
are never shown to the model. Only full-support links are citable;
partial-support evidence is carried as context and cannot be cited.
Sentences that are not exactly one sentence, or whose cited source does
not support the whole sentence, are dropped rather than repaired into
place. Reports are packed to the topic character limit by whole sentences
only.

Topic 2023 carries a single uncited abstention sentence: its evidence
contains no citation-ready passage, and no honest citation was available.
```

Facts behind that text, all read off the run manifest:

| Setting | Value |
| --- | --- |
| Model | `openai/gpt-5.6-sol` |
| Temperature / seed | `0.0` / `7` |
| Structured output | `json_schema` (strict) |
| Max citations per sentence | 3 |
| Evidence passages per topic | up to 30, untruncated text |
| Character budget ratio | 0.98 of the topic limit |
| Generator fingerprint | `783fe29684a42b0df14ba3514249be49…` |
| Handoff fingerprint | `28972dd580ee96930147157de3cc8694…` |
| Topic catalog digest | `be2f70cbaa47a06251b461fce91b3eaf…` |
| Input manifest | `complete-103.json` (`retrieval_handoff_v1`, complete) |

---

## 5. Priority

`1` is highest, `10` lowest — it sets how eagerly the run is pulled into the
manual judging pool.

Leave it at **1** if this is your only report-generation run: you want it judged.
Lower it only if you submit several runs and want a different one assessed first.

---

## 6. What is in the submission

| | |
| --- | --- |
| Topics | **103 / 103**, in the official topic-file order, no extras or duplicates |
| Sentences | 3,919 (min 1, median 39, max 54 per topic) |
| Citations | 4,196 links over 1,060 distinct documents |
| Uncited sentences | 2 — topic 2023 (abstention) and topic 2075 s#36 (a closing statement of evidentiary limits) |
| team_id | `NP` |
| run_desc in file | `openai/gpt-5.6-sol report generation` |

102 topics carry a generated report. Topic 2023 carries the abstention.

---

## 7. Validation evidence

Three independent checkers, all run against the exact file above.

| Checker | Command | Result |
| --- | --- | --- |
| **AutoJudge `ragtime26`** (organizers') | `uv run ragtime-validate-submission <file> --strict` | **103/103 valid**, all topics present, no extras or duplicates, exit 0 |
| Official `rag_run_validator` | `rag_run_validator <file> --topics topics/topics.all.2026.v0625-fix.jsonl` | exit 0, zero errors |
| Local strict | `uv run ragtime-report-validate <file> --topics … --team-id NP --run-id np-handoff-1 --strict-order` | 103 validated |

`--strict` promotes every warning to a failure, and it still passes — so there
are no outstanding smells either.

Re-run the organizers' check any time with:

```bash
uv run ragtime-validate-submission \
  /home/npatta01/data/competitions/trec_ragtime_2026/.git/ragtime-cache/experiments/reports/submit/NP.np-handoff-1.ragtime26.jsonl \
  --strict
```

---

## 8. Known limitations — disclose if asked, none block submission

- **Topic 2023** abstains. The corpus contains zero documents about its subject
  (Joyce Hatto), verified directly against 4,000,380 documents and 3,000,285
  translations. Its paired topic 2043 shares the same problem statement with a
  different requester background and did produce a report, so the *subject* is
  answerable — 2023's own evidence bundle simply had no citation-ready passage.
  Re-running that topic's handoff upstream could replace the abstention.
- **Two sentences overreach their cited passage** (topic 2040 s#30 adds two list
  items the source does not support; topic 2051 s#13 states an age the source
  only implies arithmetically). 2 of 3,917 cited sentences. Reported by an audit;
  I have not personally re-verified these two.
- **Reports use a median 72% of the character limit.** Facet coverage is
  essentially complete, so this reflects the evidence being exhausted rather than
  the budget binding.

---

## 9. Reproducing this exact file

```bash
sops decrypt --output .env config/secrets.sops.env

uv run ragtime-report \
  --handoff-manifest "$(git rev-parse --git-common-dir)/ragtime-cache/experiments/retrieval-handoff/complete-103.json" \
  --team-id NP \
  --run-id np-handoff-1 \
  --model openai/gpt-5.6-sol \
  --concurrency 16 \
  --abstain-uncitable
```

Warm, this makes **zero provider calls** and reproduces the submission byte for
byte: all 102 reports are served from the artifact cache under generator
fingerprint `783fe296…`, and the abstention is deterministic.
