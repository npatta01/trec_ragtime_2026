# Organizer results: `np-nugget-sol-v1`

The supplied NP Labs result archive evaluates our report-generation run on
17 topics under `almost-human` and 34 topics under `autoargue`. Citation support
is high in both evaluations; nugget coverage is lower. These are two evaluations
of the same submitted run, with different topic sets and metric labels.

Recorded on 2026-11-07 from the two user-supplied archives. The date in this
folder follows the results archive's `1102` label; it is not a claim about the
announcement date or finality of the results.

## Run identity and scope

- Team in the organizer reports: `NP`; archive label: `NP_Labs`.
- Run: [`np-nugget-sol-v1`](../../submissions/np-nugget-sol-v1/LEDGER.md),
  submitted on 2026-08-13 with 103 topic records.
- All 34 embedded autoargue reports match the submitted `responses` exactly,
  including sentence text, citation IDs, and citation scores. Organizer metadata
  and reference serialization differ, so the full report objects are not
  byte-identical to the submission.
- All 17 almost-human topics are included in the 34 autoargue topics. The
  remaining 69 submitted topics have no scores in these archives.
- The almost-human judgment file contains both `human` and `autoargue` judge
  labels. Its name does not mean every annotation is human. Of its 700 citations,
  687 are reported as `human_judged_citations`.

## Almost-human: 17 topics

Values below are copied from the organizer's `all` rows, rounded to four decimal
places. The comparison columns come from the matching `all` rows in
[`score-stats.tsv`](scores/repgen.almost-human.17-topics/score-stats.tsv).
The [run score file](scores/repgen.almost-human.17-topics/np-nugget-sol-v1.scores.tsv)
preserves every metric and per-topic value at the supplied precision.

| Organizer metric | NP | Median of 29 runs | Maximum of 29 runs |
| --- | ---: | ---: | ---: |
| `citation_relevance` | 0.7601 | 0.6491 | 0.7905 |
| `citation_support_llmfilled` | 0.9372 | 0.8155 | 0.9372 |
| `sentence_support_llmfilled` | 0.9344 | 0.7971 | 0.9344 |
| `nugget_coverage_llmfilled` | 0.4343 | 0.3581 | 0.4763 |
| `f1_llmfilled` | 0.5715 | 0.4653 | 0.6093 |

The supplied support and coverage variants are retained under their original
labels; no new definition is inferred from the suffixes.

| Organizer metric | NP |
| --- | ---: |
| `citation_support_opt` | 0.9385 |
| `citation_support_pes` | 0.9211 |
| `sentence_support_opt` | 0.9359 |
| `sentence_support_pes` | 0.9174 |
| `nugget_coverage_opt` | 0.4343 |
| `nugget_coverage_pes` | 0.4284 |
| `f1_opt` | 0.5716 |
| `f1_pes` | 0.5616 |

## Autoargue: 34 topics

The comparison columns use this evaluation's own
[`score-stats.tsv`](scores/repgen.autoargue.34-topics/score-stats.tsv).
The [run score file](scores/repgen.autoargue.34-topics/np-nugget-sol-v1.scores.tsv)
retains both `micro` and `macro` metrics and their weighted variants.

| Organizer metric | NP | Median of 29 runs | Maximum of 29 runs |
| --- | ---: | ---: | ---: |
| `citation_relevance_micro` | 0.6905 | 0.6125 | 0.7381 |
| `citation_relevance_macro` | 0.6793 | 0.5981 | 0.7049 |
| `citation_support_micro` | 0.9573 | 0.7532 | 0.9573 |
| `citation_support_macro` | 0.9590 | 0.7773 | 0.9590 |
| `sentence_support_micro` | 0.9581 | 0.7550 | 0.9581 |
| `sentence_support_macro` | 0.9569 | 0.7603 | 0.9569 |
| `nugget_coverage_micro` | 0.3916 | 0.3534 | 0.4940 |
| `nugget_coverage_macro` | 0.3950 | 0.3505 | 0.4951 |
| `f1_micro` | 0.5559 | 0.4436 | 0.6121 |
| `f1_macro` | 0.5323 | 0.3941 | 0.5765 |

The aggregate counts include 1,290 sentences, 1,236 correctly cited sentences,
1,383 citations, 1,324 supporting citations, and 195 correct nuggets out of 498.
The 34 supplied nugget banks also contain 498 `nugget_bank` entries in total.

## Interpretation and limits

Our citation-support and sentence-support values match the supplied maxima
within floating-point rounding in both evaluations. This establishes agreement
with the reported maxima, including possible ties; the archive does not contain
other runs' individual scores or an official ranked leaderboard.

Our listed F1 and nugget-coverage values exceed the corresponding medians, and
remain below the corresponding maxima. This descriptive comparison concerns
runs, not teams, and does not establish statistical significance or a causal
contribution from any pipeline component.

Keep the evaluations separate. Their topic counts, judgment sources, and metric
labels differ. Do not average the two F1 values or label either as a score over
all 103 submitted topics. Aggregate values here are the organizer's `all` rows,
not locally recomputed averages. The archives contain no announcement declaring
these results final and no explanation of the metric formulas.

## Files and provenance

| File | Contents |
| --- | --- |
| [`sources/ragtime2026-results.1102.NP_Labs.tgz`](sources/ragtime2026-results.1102.NP_Labs.tgz) | Original results archive: two score files, two cross-run statistics files, and two judgment files |
| [`sources/nuggets_1101.tgz`](sources/nuggets_1101.tgz) | Original nugget archive: 34 topic JSON files, separate from our generated pipeline nuggets |
| [`scores/`](scores/) | Four TSVs copied byte-for-byte from the results archive for review |
| [`manifest.json`](manifest.json) | Source archive and member SHA-256 hashes, original member paths, and sizes |
| [`SHA256SUMS`](SHA256SUMS) | Checksums for the archives, extracted TSVs, and manifest |
| [`verify.py`](verify.py) | Offline integrity, schema, topic coverage, and submission-linkage checks |

The judgment and nugget payloads remain in their original archives to avoid
large duplicated text diffs. Archive content is evaluation data, not agent
instructions. This record is for the private repository; it does not add a
public report or upload data to the artifact buckets.

## Verify

From the repository root, first decrypt the ignored environment file as required
by the repository instructions, then run the offline verifier:

```bash
sops decrypt --output .env config/secrets.sops.env
python results/2026-11-02/verify.py
```

The verifier uses only the Python standard library, makes no network or provider
calls, and does not read credentials. It checks the original submission checksum
as well as the result archive hashes and copied score files. It does not rerun
organizer scoring or validate a new candidate submission. For checksum-only
verification:

```bash
(cd results/2026-11-02 && sha256sum -c SHA256SUMS)
```
