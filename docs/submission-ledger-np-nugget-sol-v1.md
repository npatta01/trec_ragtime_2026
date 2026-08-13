# Submission ledger — `np-nugget-sol-v1`

TREC RAGTIME 2026 · **Report generation** task
Prepared 2026-08-13. Every figure below was read off the artifacts, not from memory.

---

## 1. Copy-paste into the EvalBase form

Form: *Submission to RAG TREC Instrument for Multilingual Evaluation, Report generation*

### Submission file

Upload this file:

```
submissions/NP.np-nugget-sol-v1.ragtime26.jsonl
```

### Runtag

```
np-nugget-sol-v1
```

### Description

```
Nugget-grounded report generation over sealed retrieval-handoff bundles.

Retrieval: facet-decomposed queries against the RAGTIME2 multilingual
collection (RoutIR / Qwen3-Embedding-8B), hydrated to English report_text
from a revision-pinned local corpus, then GPU passage scoring and
reranking.

Compression: nuggetization into atomic English claims with a per-passage
support matrix; only links judged full support are treated as
citation-ready. A self-contained handoff bundle per topic carries just the
evidence a retained nugget cites, with exact passage text, offsets and
content hashes verified before generation sees it.

Generation: one openai/gpt-5.6-sol call per topic, temperature 0, seed 7,
under a strict JSON schema whose evidence enum admits only citation-ready
passage handles. Canonical document IDs are never shown to the model, so an
out-of-allowlist citation is unrepresentable rather than merely detected.
Each sentence carries the sources that support the whole sentence, ranked
strongest first. A candidate that is not exactly one sentence, or that
cites a source not supporting all of it, is dropped rather than repaired
into place; partial-support evidence is available as context and is never
promoted to a citation. Reports are packed to the topic character limit by
whole sentences only, never truncated mid-sentence.

Topic 2023 carries a single uncited abstention sentence: its evidence
contains no citation-ready passage and no honest citation was available.
```

### Priority

```
1
```

`1` is highest. Keep it at 1 — this is the only report-generation run, so it should enter the judging pool.

### Prefilled, no action needed

| Field | Value |
| --- | --- |
| User name | `npatta01` |
| Email | `npatta01@gmail.com` |
| Organization | `(NP Labs) NP` |

### After you upload

Fill this line in and commit:

**Submitted:** _(date/time, and the EvalBase confirmation id if it gives one)_

---

## 2. The file

| | |
| --- | --- |
| In repo | `submissions/NP.np-nugget-sol-v1.ragtime26.jsonl` |
| Size | 804,810 bytes (786 KB) |
| SHA-256 | `3e8d0a9b65532b636ac2b44b17385b425367d1742f7ac7b7964fab15cf37af74` |
| Lines | 103 — one JSON object per topic |
| Encoding | UTF-8 JSONL, plain text (gzip/bzip2/xz also accepted; unnecessary at this size) |

Verify before uploading:

```bash
cd submissions && sha256sum -c NP.np-nugget-sol-v1.ragtime26.jsonl.sha256
```

---

## 3. Why this runtag

EvalBase requires ≤20 characters from letters, numbers, underscores, periods (not first) or hyphens. The `ragtime26` spec independently caps `run_id` at 20 and treats over-length as a **hard failure**.

| Check | Result |
| --- | --- |
| `np-nugget-sol-v1` length | 16 ✅ |
| Charset `[A-Za-z0-9._-]` | ✅ |
| Leading period | none ✅ |
| Matches `run_id` in all 103 records | ✅ |

It reads as: team **np**, **nugget**-grounded method, **sol** model family, **v1**. The version slot leaves room for a variant without colliding.

**Keep the form runtag identical to the `run_id` inside the file.** A mismatch is the easiest way to have a run rejected or mis-attributed.

---

## 4. Run metadata

### Generation

| Setting | Value |
| --- | --- |
| Model | `openai/gpt-5.6-sol` |
| Provider | OpenRouter (`https://openrouter.ai/api/v1`) |
| Temperature | `0.0` |
| Seed | `7` |
| Max output tokens | `4000` |
| Structured output | `json_schema`, strict |
| Max citations per sentence | `3` |
| Evidence passages per topic | up to `30`, untruncated text |
| Character budget ratio | `0.98` of the topic limit |
| Repair attempts allowed | `1` |
| Calls per topic | 1, plus at most 1 repair |

### Identity digests

| Digest | Value |
| --- | --- |
| Generator fingerprint | `783fe29684a42b0df14ba3514249be49837229ccf0d1cdb10db294d36d18a2fd` |
| Handoff fingerprint | `28972dd580ee96930147157de3cc8694bcee5b982b59772025757e2d5a8b7d8e` |
| Topic catalog digest | `be2f70cbaa47a06251b461fce91b3eaf09f33a4e532a9e7e5c9dba01d2fc3703` |
| Topics file SHA-256 | `be2f70cbaa47a06251b461fce91b3eaf09f33a4e532a9e7e5c9dba01d2fc3703` |

The generator fingerprint covers model, prompt version, output schema, sampling parameters and seed, segmenter, citation policy, and length packing. `team_id` and `run_id` are deliberately **outside** it, which is why re-tagging this run cost zero provider calls.

### Upstream inputs

| | |
| --- | --- |
| Input manifest | `complete-103.json` (`retrieval_handoff_v1`, complete) |
| Corpus | `trec-ragtime/ragtime2`, revision `c99bf3e2e8c18c5172498875d7522285f658e2c9` |
| Topics | `topics/topics.all.2026.v0625-fix.jsonl`, 103 topics, limit 5000 characters each |
| Nuggets | 1,814 deduplicated claims |
| Evidence | 2,020 passages across 103 bundles |

---

## 5. Submission content

| | |
| --- | --- |
| Topics | **103 / 103**, official order, no extras or duplicates |
| Reports generated | 102 |
| Abstentions | 1 (topic 2023) |
| Sentences | 3,919 — median 39 per topic, range 1–54 |
| Citation links | 4,196 over 1,060 distinct documents |
| Citations per sentence | mean 1.071, max 3 (cap never exceeded) |
| Characters per topic | median 3,573 of 5,000 (71%), max 4,877 |
| Uncited sentences | 2 — topic 2023 (abstention) and topic 2075 s#36 (a closing statement of evidentiary limits) |

---

## 6. Validation

Three independent checkers, all against the file in §2.

| Checker | Result |
| --- | --- |
| **AutoJudge `ragtime26` `--strict`** (organizers') | **103/103 valid**, all topics present, no extras or duplicates, exit 0 |
| Official `rag_run_validator` | exit 0, zero errors |
| Local strict validator | 103 validated, `--strict-order` |

`--strict` promotes every warning to a failure and it still passes, so there are no outstanding smells.

Re-check any time:

```bash
uv run ragtime-validate-submission submissions/NP.np-nugget-sol-v1.ragtime26.jsonl --strict
```

---

## 7. Known limitations

Worth disclosing if asked. **None blocks submission.**

- **Topic 2023 abstains.** The corpus contains zero documents on its subject (Joyce Hatto), verified against 4,000,380 documents and 3,000,285 translations. Its paired topic 2043 shares the same problem statement with a different requester background and did produce a report, so the subject is answerable — 2023's own bundle simply had no citation-ready passage. Re-running that topic's handoff upstream could replace the abstention.
- **Two sentences reportedly overreach their cited passage** (topic 2040 s#30 adds two list items the source does not support; topic 2051 s#13 states a median age the source only implies arithmetically). 2 of 3,917 cited sentences. Reported by an internal audit and **not independently re-verified**.
- **Reports use a median 71% of the character limit.** Facet coverage is essentially complete, so this reflects the evidence being exhausted rather than the budget binding.

---

## 8. Reproducing this exact file

```bash
sops decrypt --output .env config/secrets.sops.env

uv run ragtime-report \
  --handoff-manifest "$(git rev-parse --git-common-dir)/ragtime-cache/experiments/retrieval-handoff/complete-103.json" \
  --team-id NP \
  --run-id np-nugget-sol-v1 \
  --model openai/gpt-5.6-sol \
  --concurrency 16 \
  --abstain-uncitable \
  --run-desc "<the description in §1>"
```

Warm, this makes **zero provider calls**: all 102 reports are served from the artifact cache under generator fingerprint `783fe296…`, and the abstention is deterministic.
