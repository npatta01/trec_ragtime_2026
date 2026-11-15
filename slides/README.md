# TREC RAGTIME 2026 talk

[Self-contained HTML deck](ragtime2026.html) · [PDF slides](ragtime2026.pdf)

The opening illustration shows a request and news in four languages becoming a cited English report. The talk follows Topic 2014, African penguin conservation, from the original request and DeepSeek queries to the organizer scores. Advance with the right arrow to reveal the supplied English translation and the proposed coverage check. Press **S** for speaker notes and **Esc** for an overview. Use the appearance button or press **D** to cycle Auto, Dark, Light, and Storybook. The PDF shows both versions of the translated excerpt.

Edit `ragtime2026.qmd` and `ragtime.scss`. Build the self-contained browser deck from the repository root.

```sh
quarto render slides/ragtime2026.qmd --to revealjs
```

Open the HTML with `?print-pdf` to print a copy.

The notes link to the notebook paper, topic file and submitted report. The translation uses the pinned corpus revision `c99bf3e2e8c18c5172498875d7522285f658e2c9`. A China Times article on the bird-flu response at Boulders Beach, South Africa, shows how the supplied English translation led to a citation to the original Chinese document. The News24 article is the main example through reranking, extraction, writing and scoring. The News24 walkthrough shows how an extracted nugget guides the writing plan and how the linked article supplies the study result. The cited-sentence slide shows separate AutoARGUE judgments for citation support and the reference answer. Slide 15 shows Topic 2014’s organizer article matches, preliminary AutoARGUE totals and three short examples of credited nuggets. Slide 16 compares the six missing reference details with the submitted wording and selected sources. The two comparison slides use the organizers’ preliminary aggregate rows and field statistics for 29 runs. The appendix includes all eight penguin queries, the 13 organizer article matches, the settings, the comparison table, the writing-cost breakdown, the abstention rule and the proposed coverage check.

The visual reference is the [final TREC RAG 2026 talk](https://npatta01.github.io/trec_rag_2026/reports/2026-trec-rag-talk.html). The two library scenes are reused from that deck. The model marks come from the [DeepSeek brand assets](https://github.com/deepseek-ai/awesome-deepseek-integration/blob/main/docs/_logo%20svg/ICON.svg) and the [Mixedbread press kit](https://www.mixedbread.com/press). At the transition from the penguin example to the aggregate scores, the illustration shows a supported citation and a partially filled coverage checklist. The other illustrations are in [art](art/).

The request slide uses a [generated illustration of an African penguin at Boulders Beach](art/african-penguin.jpg). The [Two Oceans Aquarium species profile](https://www.aquarium.co.za/foundation/animals/african-penguin) supplies the native range.

The built-in ImageGen prompts use the reference deck’s thick hand-drawn lines, warm cream, flat colors and oak tabletop. The daylight scenes show a report request beside multilingual newspapers and a cited English report, news articles becoming short claims, and a citation check beside a partially filled checklist. The task, report-generation and evaluation scenes have night variants with the same composition, slate paper, amber highlights and a charcoal background.

Built-in ImageGen prompt for the penguin illustration

> Create a square editorial illustration for a research talk. Show one adult African penguin (Spheniscus demersus), full body in three-quarter view, standing on warm rounded granite boulders at Boulders Beach, South Africa. Distinctive black-and-white plumage with a single black chest band, sparse black belly spots, a white stripe around the face, and subtle pink skin just above the eye. A sheltered pale teal bay behind it. Use clean dark outlines, warm cream paper, softly textured flat gouache colors, and a clear silhouette that reads at slide size. Keep the bird’s head and feet comfortably inside the frame. No ice, snow, crests, lettering, or borders.

The closing slides use the saved writer records at `<git-common-dir>/ragtime-cache/experiments/reports/np-nugget-sol-v1`. The local submission matches the checked-in submission by SHA-256. The 115 writer attempts sum one draft for each of 102 generated reports and the 13 recorded repairs. Token totals come from the checkpoints, and elapsed time comes from their corresponding report artifacts. The Topic 2014 trace follows the saved fishing-restrictions query to a News24 article. It verifies the source’s search and reranked positions, the retained nugget, and the final cited sentence. The two omitted details come from the same article and the organizer reference bank.

The Topic 2014 trace uses the saved plan, the eight retrieval inputs, cached Mixedbread score maps, and the sealed nugget bundle. The eight lists contain 800 query–article pairs across 601 unique IDs. The reranking replay reproduces all 24 selected articles in order. In the saved round-robin pass, rounds 1 to 9 contribute 4, 2, 4, 3, 3, 3, 2, 2 and 1 articles. The pool has 14 original English articles, two Spanish, two Russian and six Chinese. The China Times result moves from 60 to 2 for q4 and is cited for the bird-flu sentence. News24 moves from 16 to 1 for q7 and supports the cited sentence about the 2008 restricted-fishing study. All 19 retained nuggets have full support. Eleven distinct articles provided full support and seven were cited in the final report.


## Slide evidence

| Slides | Source used in the audit |
|---|---|
| Task and search | [Topic catalog](https://github.com/npatta01/trec_ragtime_2026/blob/a4561cf097dcfaf249ddfc3dcf25845a897d187a/topics/topics.all.2026.v0625-fix.jsonl), [organizer API](https://trec-ragtime.github.io/search_api.html), [dataset card](https://huggingface.co/datasets/trec-ragtime/ragtime2), saved query plan and retrieval inputs |
| Ranks and claim extraction | Topic 2014’s cached retrieval results, per-query score maps, and sealed nugget bundle |
| Writing and abstention | Saved handoff and writer manifest, [report-generation code](https://github.com/npatta01/trec_ragtime_2026/blob/95faef084ef7409eb5ac576c0689f48a476852cb/src/trec_ragtime/report_generation.py), writer report records and submitted JSONL |
| Organizer results and the penguin example | [Organizer results archive](https://github.com/npatta01/trec_ragtime_2026/blob/a4561cf097dcfaf249ddfc3dcf25845a897d187a/results/2026-11-02/README.md), Topic 2014’s original nugget bank, report judgments and exact retrieval inputs |
| Writer cost | The 115 cached OpenRouter responses for this run. Match the 102 selected completions to saved report sentences, then match the remaining attempts by each repaired report’s token remainder. The sum of usage.cost is USD 20.192178. Two attempts had been cached earlier. The saved Nuggetizer run does not record charges. |

The slide audit uses the [merged notebook paper](https://github.com/npatta01/trec_ragtime_2026/blob/a4561cf097dcfaf249ddfc3dcf25845a897d187a/paper/ragtime2026-np-draft.tex) for the method and limitations. The saved handoff records 851 query-list entries. The writer allocates 4,900 characters by the number of fully supporting articles. The fishing-restrictions nugget has two supporting articles and a 239-character allowance. The Topic 2014 report uses 3,437 budgeted characters, including the separators. The paper’s 3,573-character median and 4,877-character maximum omit separators.

## Topic 2014 coverage audit

Preliminary AutoARGUE credited 10 of 16 reference nuggets and supported 36 of 36 citations. Matching answers in the saved sentence judgments and applying the organizer bank’s AND and OR rules reproduces the aggregate. Five questions had no matched answer. The conservation-measures question matched three of four required answers.

Joining full document IDs from the organizer’s `nuggets_1101/2014.json` to the saved run gives 13 of 13 reference articles retrieved, 10 of 13 among the 24 selected and 7 of 13 cited. The results use the union of 601 unique hits across the eight search lists. The report cites seven distinct articles. The linked appendix table lists each exact match, its best original search rank and its selection and citation status. The source checks use the English evidence spans in the submitted run's sealed bundle, with their text hashes verified. Every missing detail was present in a selected organizer-listed article. The 19 saved claim variants and submitted report omit all six details. The bundle records zero rejections for atomicity, support or the 20-nugget cap. Deduplication preserves all distinct normalized variants, which puts the gaps at the output of Nuggetizer. The extraction run did not save the interim candidate lists or the raw importance-labeling responses. All 19 retained nuggets qualified for writing. The writer’s cached draft and submitted report contain the same 34 sentences, with zero repairs or drops.

| Missing detail | Selected source | Available for writing |
|---|---|---|
| 9,900 breeding pairs worldwide | Daily Maverick | Eligible and cited for other facts |
| 13,200 breeding pairs in 2019 | News24 | Eligible and cited for other facts |
| 97% decline over a century | NBC Boston | Eligible and cited for other facts |
| Bee swarm behind 63 deaths near Cape Town | Two Gazeta.ua articles and one RFI article | All six listed articles were retrieved, with three selected. The selected three were outside the eligible source list. |
| Names of the targeted colonies | News24 | Eligible and cited for other facts |
| Fishing in less sensitive areas | News24 | Eligible and cited for other facts. This was the fourth required conservation measure. |

The observations follow the organizer bank, retrieved document IDs, selected passages, retained claims and variants, writer eligibility links and exact submitted sentences. Slide 16 pairs the organizer's expected details with the submitted report's wording. The source and question details are in its speaker notes.
