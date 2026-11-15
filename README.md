# TREC RAGTIME 2026

NP Labs’ system turns research requests into short English reports with
sentence-level citations to the original news articles.

[Project site](https://npatta01.github.io/trec_ragtime_2026/) ·
[Notebook paper (PDF)](paper/ragtime2026-np-draft.pdf) ·
[Workshop slides](https://npatta01.github.io/trec_ragtime_2026/slides/ragtime2026.html) ·
[92-second video](https://npatta01.github.io/trec_ragtime_2026/#film)

## Approach

1. **Plan queries.** DeepSeek V4 Flash turns each request into a broad English
   query and focused queries for parts of the request.
2. **Retrieve and rerank.** The organizers’ index searches English, Spanish,
   Russian, and Chinese news. Mixedbread reranks the English originals and
   organizer-supplied translations for each query.
3. **Extract nuggets.** We select up to 24 distinct articles round-robin.
   Nuggetizer extracts factual claims and labels article support. We merge
   similar claims and keep up to 20 per topic.
4. **Write the report.** Fully supported claims guide GPT-5.6 Sol. Each report
   fits within 5,000 characters.

## Organizer results

We submitted one run, `np-nugget-sol-v1`, with 103 topics. In the preliminary
results, AutoARGUE scored 34 topics. The mostly-human assessment scored 17
of those topics, with AutoARGUE filling gaps in the human judgments.

| Measure | AutoARGUE (micro) | Mostly human (LLM-filled) |
| --- | ---: | ---: |
| Citation support | 95.73% | 93.72% |
| Nugget coverage | 39.16% | 43.43% |
| F1 | 55.59% | 57.15% |

[Organizer score notes](results/2026-11-02/README.md) ·
[Submission ledger](submissions/np-nugget-sol-v1/LEDGER.md)

## Start here

- Explore public articles and translations. [Mini data viewer](notebooks/ragtime2_mini_viewer.ipynb)
- Set up retrieval, generate reports, or validate a submission. [Pipeline guide](docs/pipeline-guide.md)
- Check the collection, API, and track format. [Track reference](https://npatta01.github.io/trec_ragtime_2026/reports/2026-quick-reference-and-2025-reproduction-playbook/)
