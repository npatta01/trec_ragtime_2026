# TREC RAGTIME 2026 Reference and Mini Viewer

This repository is a small working companion for TREC RAGTIME 2026. It combines
an at-a-glance track reference with a runnable notebook that previews the
RAGTIME2 data without downloading the full collection.

## Start Here

1. Read the [hosted quick reference and reproduction playbook](https://npatta01.github.io/trec_ragtime_2026/reports/2026-quick-reference-and-2025-reproduction-playbook/)
   for the dataset layout, retrieval API, submission formats, timeline, and
   useful 2025 system context.
2. Open the [mini data viewer](notebooks/ragtime2_mini_viewer.ipynb) to inspect
   live original-language and machine-translated samples from Hugging Face.
3. Use the optional notebook helper with participant credentials when you are
   ready to query the private RAGTIME retrieval service.

## Run the Notebook

Install [uv](https://docs.astral.sh/uv/) and run:

```bash
uv run jupyter lab notebooks/ragtime2_mini_viewer.ipynb
```

To execute the whole notebook without opening Jupyter:

```bash
uv run jupyter nbconvert \
  --execute \
  --to notebook \
  --inplace notebooks/ragtime2_mini_viewer.ipynb
```

The notebook uses the Hugging Face Dataset Viewer API and loads only a small,
configurable number of rows. It requires internet access, Python 3.11 or newer,
and no Hugging Face token for the public samples.

## Private Search API

Registered TREC participants receive the base URL and bearer token separately.
The notebooks load the ignored root `.env` file automatically. The mini-viewer
skips its optional private retrieval when either value is absent; the dedicated
single-query notebook reports the missing configuration. Do not add credentials
to notebooks, repository, or saved cell outputs.

The shared encrypted configuration is committed as
`config/secrets.sops.env` using SOPS and the GCP KMS key configured in
`.sops.yaml`. GCP Application Default Credentials must have decrypt access to
that key.

Decrypt the configuration before starting the notebook, then run Jupyter
normally:

```bash
sops decrypt --output .env config/secrets.sops.env
uv run jupyter lab notebooks/ragtime2_mini_viewer.ipynb
```

For a minimal one-query API check, open the single-query notebook and edit its
`QUERY` value. Build the complete local corpus once before opening it:

```bash
uv run ragtime-corpus build
uv run jupyter lab notebooks/ragtime_single_query.ipynb
```

Always decrypt first and then run code normally. Do not wrap notebook or
application commands with `sops exec-env` or another SOPS execution wrapper.

To change the encrypted values, edit them through SOPS and decrypt the updated
file again:

```bash
sops config/secrets.sops.env
sops decrypt --output .env config/secrets.sops.env
```

For an unshared local configuration, copy `.env.example` to `.env` and fill in
the token. The plaintext `.env` file is ignored by Git.

## Hosted Retrieval Model

The official participant endpoint currently uses
[`Qwen/Qwen3-Embedding-8B`](https://huggingface.co/Qwen/Qwen3-Embedding-8B)
for dense retrieval, a FAISS index for vector similarity search, and
[RoutIR](https://github.com/hltcoe/routir) for serving and pipeline composition.
The `ragtime2` alias searches Chinese, Spanish, Russian, and English branches
and fuses their scores.

The model is an 8-billion-parameter, instruction-aware text embedding model.
Its model card reports support for more than 100 languages, a context window up
to 32K tokens, and embeddings up to 4,096 dimensions. Those are model
capabilities, not confirmed endpoint settings: the organizers have not
published the deployed embedding dimension, query instruction, truncation,
FAISS quantization/index parameters, or an immutable index version.

This endpoint is retrieval-only. It returns ranked document IDs and scores,
and `/content` returns document text. The default `ragtime2` pipeline does not
write the final report, extract nuggets, apply a cross-encoder reranker, or
verify citations. A separate generation model is therefore required for
report-generation runs.

### When To Consider Another Retrieval Model

Use the hosted Qwen3 retriever as the first baseline. It is already a strong
multilingual model and avoids maintaining a four-language index. Before
replacing it, measure retrieval failures on judged development topics and try:

1. Decomposing each report request into focused queries and fusing the result
   lists.
2. Fetching candidate text and applying a multilingual reranker, such as a
   Qwen3-Reranker model, to improve ordering and citation precision.
3. Adding a lexical BM25/SPLADE signal for exact names, dates, law numbers, and
   other facts that dense retrieval may miss.

Build a separate embedding index only when evaluation shows that the hosted
candidate pool is missing relevant documents, or when you need full control
over candidate depth, offline reproducibility, metadata filters, custom query
instructions, or index snapshots. If local cost or latency matters, smaller
Qwen3 embedding variants are available, but their quality should be measured
on the RAGTIME topics rather than inferred from generic benchmarks.

## Reusable Retrieval And Hydration

Use `RagtimeRetriever` for the normal end-to-end path. It searches the private
RAGTIME endpoint, then hydrates the ranked document IDs with canonical source
text, URL, date, language, and—where needed—an English translation from the
public [Hugging Face corpus](https://huggingface.co/datasets/trec-ragtime/ragtime2):

```python
from trec_ragtime import RagtimeRetriever

with RagtimeRetriever.from_env() as retriever:
    result = retriever.retrieve(
        "health benefits of Nordic walking",
        pipeline="ragtime2%100",
        top_k=None,
    )

for document in result.documents:
    print(
        document.rank,
        document.document_id,
        document.score,
        document.language,
        document.url,
        document.report_text,
    )
```

`RagtimeRetriever` uses the cached `RagtimeClient.search()` method for retrieval
and automatically selects the local corpus index when it exists. `top_k=None`
enriches every result returned by the endpoint; it does not impose a 20-result
limit. To request more than 20 from RoutIR, change the literal pipeline suffix,
for example `ragtime2%100`. A `top_k` value can only truncate results that the
endpoint already returned.

The `default` configuration remains canonical endpoint content. All 11 cached
private `/content` samples matched it byte-for-byte, including two documents
from each of English, Spanish, Russian, and Chinese. The `mt_docs` rows reuse
the Spanish, Russian, and Chinese IDs with English machine translations, so the
local index joins them as a second text representation rather than treating
them as additional retrieval documents.

`document.text` is always canonical; `document.english_text` is optional;
`document.report_text` selects English when available and otherwise preserves
the English source document. This is the representation intended for the
track's required English report. The dataset card says HLTCOE produced these
translations with Sockeye, but it does not identify an exact model checkpoint,
so translated text remains explicitly distinguishable from source evidence.

### Build The Complete Local Corpus

The corpus command resolves the generated Parquet branch to an immutable Hub
commit and downloads all 45 shards: 25 canonical `default` shards and 20
`mt_docs` English-translation shards. It materializes 4,000,380 unique documents
plus 3,000,285 ID-joined translations into DuckDB. Before indexing, it verifies
the exact Hub manifest, every shard's byte size and LFS SHA-256 digest, unique
IDs, source-language mapping, and bidirectional translation coverage:

```bash
sops decrypt --output .env config/secrets.sops.env
uv run ragtime-corpus build
uv run ragtime-corpus status
```

`status` performs a deep indexed-translation integrity check before reporting
the corpus as canonical and English-complete.

The pinned source snapshot is 11.62 GB and the unified DuckDB index is about
24.90 GB. Project artifacts live at `<git-common-dir>/ragtime-cache/`, one
untracked location shared by every worktree. Installed packages used outside
this Git checkout fall back to
`${XDG_CACHE_HOME:-~/.cache}/trec-ragtime-2026/`. Override the shared root with
an absolute `RAGTIME_PROJECT_CACHE_DIR` or the database alone with
`RAGTIME_LOCAL_CORPUS_PATH`.

`RagtimeRetriever.from_env()` always opens the canonical local index. Its path
comes from `RAGTIME_LOCAL_CORPUS_PATH`, or defaults beneath the configured
project cache. A missing, corrupt, or incomplete index raises a clear startup
error; run `uv run ragtime-corpus build` before retrieval. There is no silent
Dataset Viewer fallback. Local lookups are read-only, require no Hugging Face
response cache, and never call Dataset Viewer. Use `HuggingFaceCorpus` directly
only when explicitly opting into Dataset Viewer hydration without the complete
local index.

For direct enrichment without running a search, pass any number of exact IDs:

```python
from trec_ragtime import LocalCorpusIndex

with LocalCorpusIndex() as corpus:
    hydration = corpus.enrich_documents(document_ids)

for document in hydration.documents:
    print(document.document_id, document.text, document.english_text)
    if document.report_text is None:
        raise RuntimeError(f"English text unavailable for {document.document_id}")
    use_for_report = document.report_text
```

The method deduplicates IDs for lookup, preserves first-seen order, chunks
arbitrarily large batches, and reports absent base IDs in
`hydration.unresolved_ids`. Any expected English-translation gap is reported
separately in `hydration.missing_translation_ids`. The local index is
revision-pinned, so `refresh=True` does not contact the network. The high-level
retriever exposes the same operation as `retriever.enrich_documents(ids)` when
a search client is already open.

True corpus misses automatically fall back to the private endpoint:

```python
result = retriever.retrieve(
    "health benefits of Nordic walking",
    max_content_fallbacks=5,
)
```

The default permits at most five rank-ordered `/content` requests for IDs absent
from an otherwise healthy corpus; those responses are cached without expiry.
`0` disables fallback and `None` explicitly permits every true miss. The loop
stops after its first endpoint error. Across all currently cached searches,
130/130 unique returned IDs resolved locally, so this path has not yet been
needed. Any fallback document has text but no URL, date, language, or verified
English variant. Inspect `result.fully_hydrated`, `result.fully_report_ready`,
`result.unresolved_ids`, and `result.missing_translation_ids` before generation.

Dataset Viewer failures that leave IDs unresolved raise
`HuggingFaceHydrationError` rather than causing a burst of private requests.
Set `fallback_on_hf_error=True` only when the same bounded private fallback is
desired after a remote hydration failure; that mode requires a finite cap.
Inspect `result.hydration_failures` for any recorded failures.

Use the lower-level `RagtimeClient` when only document IDs and scores, or an
explicit single `/content` lookup, are needed:

```python
from trec_ragtime import RagtimeClient

with RagtimeClient.from_env() as client:
    result = client.search("health benefits of Nordic walking")

    for hit in result.hits:
        print(hit.rank, hit.document_id, hit.score)

    document = client.get_content(result.hits[0].document_id)
    print(document.text)
```

Private responses and Dataset Viewer fallback documents share the
permission-restricted SQLite database at
`<git-common-dir>/ragtime-cache/api-v1.sqlite3`. The complete local corpus is a
separate revision-pinned DuckDB database and does not use
per-document response caching. Both live in Git's untracked common metadata
area and are shared across worktrees. Cache overrides must be absolute paths
inside a user-only directory.

Private `/pipeline` and `/content` entries do not expire by default because the
participant endpoint is treated as immutable. Numeric
`RAGTIME_SEARCH_CACHE_TTL_SECONDS` and `RAGTIME_CONTENT_CACHE_TTL_SECONDS`
values can still impose an expiry if needed. Remotely hydrated Dataset Viewer
documents live for 30 days by default; override that TTL with
`RAGTIME_HF_DOCUMENT_CACHE_TTL_SECONDS`, or set it to `none`, `off`, or
`disabled` for no expiry. `refresh_search=True` and `refresh_documents=True`
independently bypass and replace the corresponding entries through
`RagtimeRetriever.retrieve()`. Set `use_cache=False` only when intentionally
running without a cache.

Live Dataset Viewer fallback fetches use the dataset's current default
revision, and their sparse cache remains subject to the configured TTL. The
complete local index instead records the immutable Parquet conversion commit
and is replaced only by an explicit `ragtime-corpus build --replace`. FAISS is
unnecessary for hydration because the hosted endpoint has already done
semantic retrieval; the remaining operation is an exact lookup by document ID.

Although the private endpoint does not publish an immutable version identifier,
this project follows the participant guarantee that its results will not
change. Use `refresh=True`, rotate `RAGTIME_CACHE_NAMESPACE`, or call
`client.cache.clear()` only for explicit manual invalidation. Cache keys include
the base URL, operation, namespace, and exact JSON request, but never the bearer
token.

### RAGTIME Contract Details

| Purpose | Required value |
| --- | --- |
| Topic-file collection ID | `ragtime/2/all` |
| Multilingual pipeline alias | `ragtime2` |
| `/content` collection | `ragtime2` |

- `ragtime2%20` is literal RoutIR syntax, not an encoded space. It expands the
  `ragtime2` alias and keeps the final top 20 fused results.
- The alias currently searches the top 25 results from each of Chinese,
  Spanish, Russian, and English before score fusion. Increasing only the outer
  `%20` does not increase those per-language depths.
- The API returns `scores` as an unordered document-ID-to-score object. The
  client sorts it by descending score and then document ID for stable ties.
- The client canonicalizes authentication to `Authorization: Bearer <token>`
  and distinguishes a local SQLite hit from the service's own `cached` field.
- Cached private queries and responses must be treated as participant data. The
  cache directory and database are created with user-only permissions and must
  not be committed.

## Repository Map

- [`AGENTS.md`](AGENTS.md): contributor and agent instructions, including the
  required decrypt-first secret workflow.
- [`src/trec_ragtime`](src/trec_ragtime): reusable private API client, local
  DuckDB corpus index and builder, Dataset Viewer fallback, high-level
  retriever, typed responses, and cross-worktree caches.
- [`topics/topics.all.2026.v0625.jsonl`](topics/topics.all.2026.v0625.jsonl):
  [official June 25 test topics](https://trec.nist.gov/act_part/tracks/ragtime/topics.all.2026.v0625.jsonl),
  normalized to one JSON object per line.
- [`index.html`](index.html): compact GitHub Pages landing page for the project.
- [`reports/2026-quick-reference-and-2025-reproduction-playbook/index.html`](reports/2026-quick-reference-and-2025-reproduction-playbook/index.html):
  static 2026 quick reference and 2025 reproduction playbook.
- [`notebooks/ragtime2_mini_viewer.ipynb`](notebooks/ragtime2_mini_viewer.ipynb):
  bounded dataset preview and optional retrieval API example.
- [`notebooks/ragtime_single_query.ipynb`](notebooks/ragtime_single_query.ipynb):
  one-query cached search with complete local-corpus document enrichment.
- [`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock): reproducible Python
  environment managed by uv.
- [`tests`](tests): offline unit tests for HTTP, cache, corpus build, and local
  enrichment behavior.

The RAGTIME2 snapshot and generated DuckDB index live in the Git-common project
cache, not in any tracked worktree. Track details can change, so use the
official sources linked from the [reference report](reports/2026-quick-reference-and-2025-reproduction-playbook/index.html)
when making submission decisions.
