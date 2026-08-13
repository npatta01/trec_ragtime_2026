# Submission ledger — `np-nugget-sol-v1`

TREC RAGTIME 2026 · **Report generation** task
Prepared 2026-08-13. Every figure below was read off the artifacts, not from memory.

---

## 1. Copy-paste into the EvalBase form

Form: *Submission to RAG TREC Instrument for Multilingual Evaluation, Report generation*

### Submission file

Upload this file:

```
submissions/np-nugget-sol-v1/NP.np-nugget-sol-v1.ragtime26.jsonl
```

### Runtag

```
np-nugget-sol-v1
```

### Description

Paste this into the form's description box.

The `run_desc` stored inside the JSONL is a deliberately concise 417-character
summary of the same system — that field forbids newlines and caps at 1000
characters, while this box is a free-text area. Exact version pins for every
component are in §4.

```
Faceted multilingual RAG with nugget-grounded, citation-strict generation.

RETRIEVAL. Each topic request is decomposed into roughly seven facet queries
covering the distinct aspects the requester asked about. Each is issued to the
organizers' participant endpoint via the ragtime2 pipeline alias, which performs
dense retrieval with Qwen/Qwen3-Embedding-8B over a FAISS index served by RoutIR
and fuses the Chinese, Spanish, Russian and English branches. Per-facet result
lists are combined by reciprocal rank fusion (k=60, depth 100). Documents are
hydrated to English report_text from a local copy of the collection pinned to
revision c99bf3e2e8c18c5172498875d7522285f658e2c9 (4,000,380 documents and
3,000,285 machine translations); the original-language text is preserved as the
canonical record and only the English rendition is used for synthesis.

RERANKING. Candidate passages are scored on GPU with
mixedbread-ai/mxbai-rerank-base-v2 pinned to revision
3ea9d4dffa7d12a4f366be8e275c349de9fc9865. Passages are kept at document scale
rather than chunked, because splitting them stranded a claim's subject in a
neighbouring passage and cost grounding accuracy.

COMPRESSION TO NUGGETS. nuggetizer 0.0.5 extracts atomic English claims from the
reranked evidence, then judges each claim against each passage individually as
support, partial_support or not_support. That support matrix is the lineage.
Paraphrases are merged using sentence-transformers/all-MiniLM-L6-v2 at cosine
>= 0.92, guarded by a numeric and negation signature so that "fell 30%" can
never collapse into "fell 13%". This run retained 1,814 deduplicated nuggets
over 2,020 evidence passages across 103 topics.

SEALED HANDOFF. A self-contained bundle per topic carries only the evidence a
retained nugget actually cites, with the exact passage text, character offsets
and SHA-256 content hashes verified before generation is allowed to read it.
Only links judged full support are marked citation-ready; partial_support is
retained and labelled but is never promoted to a citation.

GENERATION. One openai/gpt-5.6-sol call per topic at temperature 0 with seed 7,
constrained by a strict JSON schema whose evidence enum contains only the
citation-ready passage handles for that topic. Canonical document identifiers
are never shown to the model, so a citation outside the allowlist is
unrepresentable rather than merely detected afterwards. The topic's character
budget is planned across its nugget claims in proportion to how much citable
evidence backs each claim, computed deterministically from the sealed bundle
rather than by a separate planning call. A bounded repair pass re-asks when
candidates are rejected. A candidate that is not exactly one sentence, or whose
cited source does not support the entire sentence, is dropped rather than
repaired into place, and conflicting accounts are kept as separate attributed
sentences rather than blended. Citations are ranked so the strongest fully
supporting source is scored highest, since assessors check the first citation.
Reports are packed to the topic's character limit by whole sentences only and
are never truncated mid-sentence.

ABSTENTION. Topic 2023 carries a single uncited sentence stating that no
citation-ready evidence was available. Its evidence contains no fully supporting
passage, and the collection contains no document on the topic's subject, so no
honest citation exists. Fabricating one or promoting partial support was
rejected in favour of an explicit abstention.
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
| In repo | [`NP.np-nugget-sol-v1.ragtime26.jsonl`](NP.np-nugget-sol-v1.ragtime26.jsonl) — beside this ledger |
| Size | 804,810 bytes (786 KB) |
| SHA-256 | `ffe055b4e5f506fbf127489dfa94bac76558b4bf10f7762c18b378425fa87242` |
| Lines | 103 — one JSON object per topic |
| Encoding | UTF-8 JSONL, plain text (gzip/bzip2/xz also accepted; unnecessary at this size) |

Verify before uploading:

```bash
cd submissions/np-nugget-sol-v1 && sha256sum -c NP.np-nugget-sol-v1.ragtime26.jsonl.sha256
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

### Component stack

| Stage | Component |
| --- | --- |
| Dense retrieval | `Qwen/Qwen3-Embedding-8B` over FAISS, served by RoutIR — organizers' `ragtime2` endpoint, fusing zh/es/ru/en |
| Query construction | ~7 facet queries per topic, reciprocal-rank fusion |
| Hydration | English `report_text` from a revision-pinned local collection |
| Reranking | `mixedbread-ai/mxbai-rerank-base-v2` @ `3ea9d4df`, GPU, document-scale passages |
| Nuggetization | `nuggetizer==0.0.5` — atomic claims plus a per-passage support matrix |
| Paraphrase merging | `sentence-transformers/all-MiniLM-L6-v2`, cosine ≥ 0.92 with numeric/negation guard |
| Generation | `openai/gpt-5.6-sol` via OpenRouter, one call per topic |

### Exact versions

Everything that can change a result, pinned. Read from `pyproject.toml`,
`uv.lock`, and the module constants at the commit that produced this run.

| Component | Exact version |
| --- | --- |
| Python | `3.12.13` |
| Retrieval endpoint | organizers' `ragtime2` pipeline alias (deployed embedding dimension, query instruction and FAISS index parameters are not published) |
| Retrieval model | `Qwen/Qwen3-Embedding-8B` (endpoint-side, no revision published) |
| Corpus | `trec-ragtime/ragtime2` revision `c99bf3e2e8c18c5172498875d7522285f658e2c9` |
| Reranker | `mixedbread-ai/mxbai-rerank-base-v2` revision `3ea9d4dffa7d12a4f366be8e275c349de9fc9865` |
| Nuggetizer | `nuggetizer==0.0.5` |
| Paraphrase embedder | `sentence-transformers/all-MiniLM-L6-v2` |
| Generation model | `openai/gpt-5.6-sol` (OpenRouter) |
| `sentence-transformers` | `5.7.0` |
| `torch` | `2.9.1` |
| `transformers` | `5.14.1` |
| `tokenizers` | `0.22.2` |
| `openai` | `2.53.0` |
| `msgspec` | `0.21.1` |
| `spacy` | `3.8.15` |
| `duckdb` | `1.5.4` |

### Behavior versions

Each names a pinned algorithm and participates in an artifact fingerprint, so
changing one retires the artifacts it produced.

| Constant | Value |
| --- | --- |
| Nuggets | `nuggets-v1` |
| Paraphrase guard | `numeric-negation-guard-v1` |
| Retrieval handoff | `retrieval-handoff-v1` |
| Evidence bundle builder | `evidence-bundle-v1` |
| Prompt | `report-prompt-v1` |
| Output schema | `report-sentences-v1` |
| Sentence segmenter | `sentence-segmenter-v1` |
| Citation ranking policy | `citation-rank-v2` |
| Facet budget policy | `facet-budget-v2` |
| Length packing | `whole-sentence-pack-v1` |
| Generator behavior | `report-generator-v1` |

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
uv run ragtime-validate-submission \
  submissions/np-nugget-sol-v1/NP.np-nugget-sol-v1.ragtime26.jsonl --strict
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
