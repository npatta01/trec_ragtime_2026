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
`QUERY` value:

```bash
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

## Reusable API Client And Shared Cache

Use the `trec_ragtime` package from notebooks, scripts, or other Python code:

```python
from trec_ragtime import RagtimeClient

client = RagtimeClient.from_env()
result = client.search("health benefits of Nordic walking")

for hit in result.hits:
    print(hit.rank, hit.document_id, hit.score)

document = client.get_content(result.hits[0].document_id)
print(document.text)
```

Search and content responses are cached in a permission-restricted SQLite
database at `${XDG_CACHE_HOME:-~/.cache}/trec-ragtime-2026/api-v1.sqlite3`.
Because this path is outside the repository, all worktrees for the same user
share it automatically. Set `RAGTIME_CACHE_DIR` or `RAGTIME_CACHE_PATH` to
override the location; overrides must be absolute paths inside a user-only
directory. The client refuses a cache directory accessible to other users. Set
`use_cache=False` only when intentionally running without a cache. Search
entries live for 24 hours and content entries for 30 days by default; use
`refresh=True` on either client method to bypass and replace a cached response.

The endpoint does not expose an immutable index version. Use
`RAGTIME_CACHE_NAMESPACE` to separate experiment/index generations, or call
`client.cache.clear()` to clear the active namespace. Cache keys include the
base URL, operation, namespace, and exact JSON request, but never the bearer
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
- Cached queries and document text may be participant-only data. The cache
  directory and database are created with user-only permissions and must not be
  committed.

## Repository Map

- [`AGENTS.md`](AGENTS.md): contributor and agent instructions, including the
  required decrypt-first secret workflow.
- [`src/trec_ragtime`](src/trec_ragtime): reusable API client, typed responses,
  and cross-worktree SQLite cache.
- [`topics/topics.all.2026.v0625.jsonl`](topics/topics.all.2026.v0625.jsonl):
  [official June 25 test topics](https://trec.nist.gov/act_part/tracks/ragtime/topics.all.2026.v0625.jsonl),
  normalized to one JSON object per line.
- [`index.html`](index.html): compact GitHub Pages landing page for the project.
- [`reports/2026-quick-reference-and-2025-reproduction-playbook/index.html`](reports/2026-quick-reference-and-2025-reproduction-playbook/index.html):
  static 2026 quick reference and 2025 reproduction playbook.
- [`notebooks/ragtime2_mini_viewer.ipynb`](notebooks/ragtime2_mini_viewer.ipynb):
  bounded dataset preview and optional retrieval API example.
- [`notebooks/ragtime_single_query.ipynb`](notebooks/ragtime_single_query.ipynb):
  minimal one-query search against the private RAGTIME endpoint.
- [`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock): reproducible Python
  environment managed by uv.
- [`tests`](tests): offline unit tests for HTTP validation and cache behavior.

The RAGTIME2 corpus is accessed remotely and is not vendored in this repository.
Track details can change, so use the official sources linked from the
[reference report](reports/2026-quick-reference-and-2025-reproduction-playbook/index.html)
when making submission decisions.
