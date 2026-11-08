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

Paste this into the form's description box. Kept short and organizer-facing;
exact model revisions and dependency pins are internal and live in §4.

```
Facet-decomposed multilingual retrieval, nugget compression, and
citation-strict generation.

We split each request into roughly seven facet queries and send them to the
organizers' ragtime2 endpoint, fusing the per-facet rankings with RRF. Pooled
documents are hydrated to their English text and reranked with
mxbai-rerank-base-v2. We keep passages at document scale; chunking them cost
grounding accuracy because a claim's subject often ended up in a neighbouring
chunk.

Nuggetizer extracts atomic claims from the reranked evidence and labels each
passage as full, partial, or no support for each claim. Only full support is
treated as citable. Generation then receives a sealed bundle holding just the
passages a retained claim cites; document IDs are replaced by opaque handles,
so the model cannot emit an identifier it was not given.

Reports are written in one gpt-5.6-sol call per topic at temperature 0 under a
JSON schema restricted to that topic's citable handles. Character budget is
distributed across the claims in proportion to the evidence backing each. A
sentence whose cited source does not support the whole sentence is dropped
rather than rewritten, and conflicting accounts are kept as separate attributed
sentences. Reports are trimmed to the character limit on sentence boundaries.

Topic 2023 contains a single uncited sentence saying no citation-ready evidence
was available. Nothing in its evidence fully supports any claim, and we would
rather abstain than manufacture a citation.
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

### Submitted

| | |
| --- | --- |
| Status | **Submitted to NIST via EvalBase** |
| Date | 13 August 2026 |
| Track / task | `trec2026-ragtime-repgen` — ragtime, Report generation |
| Runtag as accepted | `np-nugget-sol-v1` |
| Submitted by | `npatta01` (NP Labs) |
| EvalBase state | *waiting for validation* |

This is the only report-generation run submitted for the track.

---

## Organizer results received

The user-supplied `ragtime2026-results.1102.NP_Labs.tgz` reports scores for this
run under `almost-human` (17 topics) and `autoargue` (34 topics). See the
[results record](../../results/2026-11-02/README.md) for the original archives,
separate metric summaries, provenance, and verification. The 34 autoargue
reports preserve this submission's response text and citations exactly.
This result record supplements the historical EvalBase state above; it does
not establish a final ranked leaderboard or scores for all 103 topics.

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
