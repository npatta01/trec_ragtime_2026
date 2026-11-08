# NP at TREC RAGTIME 2026

[Read the notebook paper](ragtime2026-np-draft.pdf) · [LaTeX source](ragtime2026-np-draft.tex)

**Select the Evidence, Then Write** describes the submitted `np-nugget-sol-v1` run by Nidhin Pattaniyil, Tanwir Zaman, and Semih Yagli (Independent Researchers — Team NP Labs). It follows one vaping-legislation request through query planning, retrieval, claim selection, and cited report generation.

The five-page manuscript is self-contained: figures are drawn in TikZ and references are embedded in the source. No external image files or bibliography are needed. The filename is retained for continuity with the document editor; the PDF has no working-draft label.

## Build

Open the `.tex` file in the LaTeX editor, or compile twice with a TeX installation that includes TikZ, Latin Modern, and the packages listed in the preamble:

```bash
mkdir -p /tmp/ragtime-paper-build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=/tmp/ragtime-paper-build paper/ragtime2026-np-draft.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=/tmp/ragtime-paper-build paper/ragtime2026-np-draft.tex
```

Inspect the rendered PDF after edits. The checked PDF is committed beside its source.

## Results and scope

Table 2 uses the preliminary organizer results released on November 4, 2026, recorded in [results/2026-11-02](../results/2026-11-02/README.md). The table keeps the 17-topic mostly-human and 34-topic AutoARGUE assessments separate and shows our run, the median, and the maximum across 29 runs. [organizer-results-table.json](organizer-results-table.json) records the selected metric keys, full-precision values, and source archive checksum.

The organizer snapshot verifier checks the score files and matches all 34 embedded reports to the [submitted run](../submissions/np-nugget-sol-v1/LEDGER.md). Follow the repository environment setup, then run:

```bash
python results/2026-11-02/verify.py
```

The organizers permitted these preliminary scores for the notebook and presentation, but requested that they not be used in subsequent paper submissions. This PR does not submit the manuscript. The appendix identifies the unrecovered prompt and configuration details; the paper does not claim an ablation result or a unique leaderboard rank.
