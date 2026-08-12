# TREC RAGTIME 2026 Reference and Mini Viewer

This repository is a small working companion for TREC RAGTIME 2026. It combines
an at-a-glance track reference with a runnable notebook that previews the
RAGTIME2 data without downloading the full collection. Passage reranking starts
with the local scorer; the optional Flash path is only an explicit transport
seam.

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

## Passage Reranking

Passage reranking is a local-first layer over an existing `RetrievalResult`.
The normal English report path selects each document's `report_text`: verified
English translation for Spanish, Russian, and Chinese source documents, or the
canonical text for an English source. `representation="canonical"` selects the
original-language source instead. Missing report translations and unresolved
documents remain in the result with explicit statuses.

The pinned local model is `mixedbread-ai/mxbai-rerank-base-v2` at revision
`3ea9d4dffa7d12a4f366be8e275c349de9fc9865`, loaded through the supported direct
`sentence_transformers.CrossEncoder` path. The model-owned tokenizer name and
revision must match the model name and revision. The tokenizer-only
`CrossEncoderPairLengthFunction` captures the retrieval query and loads no
model weights. `LocalCrossEncoderScorer` loads model weights lazily, on the
first score call, and returns raw cross-encoder logits.
The supported local flow is below. `retrieval` is an existing hydrated
`RetrievalResult`, such as the value returned by
[Reusable Retrieval And Hydration](#reusable-retrieval-and-hydration).

```python
from trec_ragtime import (
    CachedPassageScorer,
    CrossEncoderPairLengthFunction,
    LocalCrossEncoderConfig,
    LocalCrossEncoderScorer,
    PassageChunker,
    PassageScoreCache,
    rerank_retrieval,
)

policy = LocalCrossEncoderConfig(device="cpu")
pair_length = CrossEncoderPairLengthFunction(policy, retrieval.query)
chunker = PassageChunker(pair_length, overlap_tokens=64)
scorer = LocalCrossEncoderScorer(policy)

# Optional complete per-query cache. `store` may be the landed `ArtifactStore`
# or an `ArtifactStoreLike` fake supplied by the caller.
cache = PassageScoreCache(
    store,
    scorer,
    chunker_fingerprint=chunker.fingerprint,
    representation="report_text",
)
scorer = CachedPassageScorer(cache)

report_result = rerank_retrieval(
    retrieval,
    chunker=chunker,
    scorer=scorer,
    representation="report_text",
)
```

`PassageChunker` uses the pinned language-specific spaCy pipelines through
LangChain's sentence splitter. It preserves exact Python-character offsets,
exact UTF-8 SHA-256 content hashes, and complete selected-text coverage. Normal
overlap contains complete sentence units only; it never splits a sentence to
manufacture overlap. If one sentence exceeds the full pair budget, only that
sentence is passed to `RecursiveCharacterTextSplitter` (RCTS), and every
fragment is rechecked against the exact pair ceiling. Keep the pair-length
callable aligned with the scorer: a character count can admit a passage that
overflows after query and special tokens are included.

Use `device="cuda"` for CUDA PyTorch, or the user-facing `device="rocm"`
alias on ROCm builds; `rocm` is normalized to PyTorch's `cuda` device. The
scorer and chunker fingerprints include result-affecting model, tokenizer,
device, precision, batching, query format, query policy, and pair-length
settings. Changing one of these settings isolates its score cache.

`PassageScoreCache` stores one complete validated `dict[bytes, float]` per
query. It accepts the concrete `ArtifactStore` or an `ArtifactStoreLike` fake.
The small protocol includes `read`, `read_validated`, `write`, and `evict_local`.
Cache reads return the requested hashes from a complete union, and a successful
miss merges new scores into that union. Partial or failed scoring is never
written.

Reranking aggregates each document by its maximum passage score. Evidence is
deduplicated by positional `(document_id, start_char, end_char)` identity, so
identical text at distinct offsets remains distinct evidence while sharing a
safe content-hash score. Ordering is deterministic by descending score,
organizer rank, and document ID; evidence is ordered by source offsets.

Flash remains an optional transport for the same `PassageScorer` contract; the
local path above is the primary complete execution path. Install Flash
separately with `uv sync --extra runpod`. Bounded payload size, batch size,
concurrency, request timeout, worker count, and idle timeout remain explicit.
The adapter accepts the same complete score map, and the static worker accepts
mapping-compatible Transformers `BatchEncoding` values. An existing endpoint
caller and a later worker definition use the same pinned model policy:

> **Flash scaling rule:** always configure workers as `(0, N)`: `0` is the
> minimum and `N` is the desired maximum. The integer shorthand `workers=N`
> also means `(0, N)`. Never use `(1, N)`, because it keeps a worker running
> and billing while idle.

```python
from trec_ragtime import (
    CrossEncoderPairLengthFunction,
    FlashEndpointBatchRunner,
    PassageChunker,
    RunpodFlashPassageScorer,
    RunpodFlashConfig,
    build_flash_worker,
    rerank_retrieval,
)

flash_policy = RunpodFlashConfig(
    endpoint_id="existing-endpoint-id",
    model_config=policy,
    device="cuda",
    batch_size=32,
    max_payload_bytes=9_000_000,
    concurrency=4,
    request_timeout=60.0,
    workers=1,  # Flash shorthand for (0, 1): scale to zero when idle.
    idle_timeout=300,
)

remote_policy = flash_policy.effective_model_config
remote_pair_length = CrossEncoderPairLengthFunction(remote_policy, retrieval.query)
remote_chunker = PassageChunker(remote_pair_length, overlap_tokens=64)

# Caller and scorer for an endpoint that already exists.
with FlashEndpointBatchRunner(flash_policy) as endpoint_runner:
    with RunpodFlashPassageScorer(flash_policy, endpoint_runner) as flash_scorer:
        flash_result = rerank_retrieval(
            retrieval,
            chunker=remote_chunker,
            scorer=flash_scorer,
            representation="report_text",
        )

# Definition seam for a later, explicit worker-definition workflow.
from runpod_flash import Endpoint

worker_definition = build_flash_worker(
    flash_policy,
    endpoint_factory=Endpoint,
)
```

`FlashEndpointBatchRunner` constructs an `Endpoint(id=...)`, submits with
`run()`, and waits with `job.wait()`. `build_flash_worker` only returns the
static worker or applies the explicit endpoint decorator with pinned
dependencies. Building a worker definition does not deploy, provision, or
make a paid request. No paid or deployment action is authorized by this
refactor.

Offline reranking verification starts after decrypting `.env` and makes no
organizer, Hugging Face, Runpod, or model-download calls:

```bash
sops decrypt --output .env config/secrets.sops.env && git check-ignore .env && \
  uv run python -m unittest tests.test_passage_reranking_integration -v
sops decrypt --output .env config/secrets.sops.env && git check-ignore .env && \
  uv run python -m unittest \
  tests.test_passages \
  tests.test_passage_scoring \
  tests.test_passage_score_cache \
  tests.test_reranking \
  tests.test_flash_passage_scorer \
  tests.test_passage_reranking_integration -v
sops decrypt --output .env config/secrets.sops.env && git check-ignore .env && \
  uv run python scripts/benchmark_passage_chunking.py
sops decrypt --output .env config/secrets.sops.env && git check-ignore .env && \
  uv run python -m unittest discover -s tests -v
```

Offline fakes and an in-memory Flash wire runner intentionally replace all
organizer, Dataset Viewer, endpoint, deployment, and model-weight calls in
these checks. A real local model batch and live endpoint behavior remain
explicit external acceptance gates.

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
    retrieval = retriever.retrieve(
        "health benefits of Nordic walking",
        pipeline="ragtime2%100",
        top_k=None,
    )

for document in retrieval.documents:
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
response cache, and never call Dataset Viewer. `HuggingFaceCorpus`, the former
opt-in Dataset Viewer backend, is deprecated; build the local corpus instead.
Its HTTP 429 handling remains temporarily for compatibility and is not part of
normal local-corpus retrieval.

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

Private responses and Dataset Viewer fallback documents are durable typed
artifacts in a private Hugging Face Bucket (L2), read through a disposable
owner-only `fsspec.simplecache` directory (L1). Configure the bucket with
`RAGTIME_HF_BUCKET_ID`, an optional `RAGTIME_ARTIFACT_PREFIX`, and an optional
absolute `RAGTIME_ARTIFACT_CACHE_DIR` (defaulting to
`<git-common-dir>/ragtime-cache/artifacts-v1`). `HF_TOKEN` is optional when the
normal Hugging Face authentication lookup is available. Runtime artifact
caching is best effort: a private-bucket authentication, transport, permission,
server, or write failure emits one credential-safe warning, disables remote
access for that store instance, and leaves the live result usable as an
uncached response. A cache miss is therefore normal after the breaker opens.
L1 artifacts may be discarded and rebuilt from the private bucket; a malformed
L1 candidate is repaired once from L2, and malformed or mismatched typed
payloads are never accepted. The adapters use a strict typed `msgspec`
MessagePack envelope, enforce identity and domain validation, and admit writes
only for fully validated successful results. Retrieval artifacts are eligible
only for complete hydrated results without hydration failures. Runner
checkpoints may include an optional artifact reference and omit it when
best-effort storage is unavailable; a successful reference-less checkpoint is
usable only in-process and is recomputed on resume or restart.
The durable Hugging Face Bucket must remain private. The complete local corpus
remains a separate revision-pinned DuckDB database and does not use
per-document response caching.

### Cache OpenAI-Compatible Chat Completions

`CachedChatCompletions` wraps an already configured OpenAI-compatible client and
stores validated `ChatCompletion` responses in the same private artifact store.
The shared setup is:

```python
from openai import OpenAI

from trec_ragtime import CachedChatCompletions, provider_settings
from trec_ragtime.artifact_store import ArtifactStore

provider_url, provider_key = provider_settings()
client = OpenAI(api_key=provider_key, base_url=provider_url, max_retries=0)
cached = CachedChatCompletions(client, ArtifactStore.from_env())
response = cached.create(model=model, messages=messages)
```

When `namespace` is omitted, the wrapper inherits a store's validated
`namespace` metadata; stores without namespace metadata use the default
`ragtime2-2026-v1`. Pass `namespace=` explicitly to override the store.

The LLM provider credential is not a special case: `OPENROUTER_API_KEY` lives
in the committed `config/secrets.sops.env` ciphertext and reaches the process
through the same decrypt-first workflow as every other secret. Decrypt into the
ignored root `.env`, and the notebooks and entry points that load `.env` pick it
up from there. The variable is listed with an empty value in the tracked
`.env.example` for an unshared local configuration. OpenRouter's base URL is
`https://openrouter.ai/api/v1`.

```bash
sops decrypt --output .env config/secrets.sops.env
```

For a direct OpenAI account or a LiteLLM deployment, use `OPENAI_API_KEY`
instead and set `OPENAI_BASE_URL` to that deployment's OpenAI-compatible `/v1`
URL. The proxy address is deployment-specific, so nothing here assumes one;
omitting `OPENAI_BASE_URL` reaches OpenAI itself at `https://api.openai.com/v1`.

`trec_ragtime.provider_settings()` owns that resolution so callers do not
re-derive it from the environment: `OPENROUTER_API_KEY` wins and implies the
OpenRouter base URL, `OPENAI_API_KEY` (or the `OPEN_AI_API_KEY` spelling some
upstream packages read) with an optional `OPENAI_BASE_URL` is the fallback,
and an explicit `OPENAI_BASE_URL` overrides the default in either case. A
blank value is an unfilled `.env` placeholder rather than a credential, and no
credential at all raises `ProviderConfigurationError` instead of reaching a
provider anonymously. Pass an explicit mapping to resolve without reading the
process environment. The returned key is for the SDK to use: it is never
logged, stored, or part of a cache key or fingerprint.

The cache identity includes the base URL, operation, namespace, and the exact
JSON request. Mapping key order is canonicalized recursively, while list order
(including message order) remains significant. Output-affecting options such
as the model, tools, response format, temperature, and provider request body
therefore get distinct entries. Normal entries are immutable and do not expire;
pass `refresh=True` to bypass the hit and replace that exact entry with a fresh
provider response. Responses are round-tripped as fully typed
`openai.types.chat.ChatCompletion` objects, including provider extension fields.
Streaming requests (`stream=True`) always delegate directly and are not cached.
Common OpenAI SDK request values, including omitted-value sentinels, Pydantic
models, enums, dates, iterables, and tuples, are normalized to their JSON form;
unordered collections are rejected because they cannot produce stable keys.

Artifact caching is best effort: a cache read or write failure leaves the live
provider response usable as an uncached result. If a refresh write fails, the
fresh response is still returned but the previous durable entry may remain
readable. The OpenAI client's API key is never part of a cache key or stored
payload; transport-only options such as
`timeout`, `extra_headers`, and `extra_query` are also excluded from identity.
Clients sharing a provider base URL and request identity also share entries;
use a distinct `namespace=` when accounts or routing policies must be isolated.
Top-level credential-shaped request fields are rejected before a provider call
so they cannot be persisted accidentally. Nested structured request data is
canonicalized as supplied, allowing tool schemas and provider extensions to
legitimately use property names such as `password` or `auth_token`.

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

Live participant-API cache misses are proactively limited to one request per
second per host with `requests-ratelimiter`, before the existing retry/backoff
policy is applied. Override the positive rate with
`RAGTIME_REQUESTS_PER_SECOND`. Local cache hits and local DuckDB hydration make
no HTTP request and consume no rate-limit capacity. Supplying a custom
`session=` is an explicit transport override and bypasses the built-in limiter
and retry policy.

Live Dataset Viewer fallback fetches use the dataset's current default
revision, and their sparse cache remains subject to the configured TTL. The
complete local index instead records the immutable Parquet conversion commit
and is replaced only by an explicit `ragtime-corpus build --replace`. FAISS is
unnecessary for hydration because the hosted endpoint has already done
semantic retrieval; the remaining operation is an exact lookup by document ID.

Although the private endpoint does not publish an immutable version identifier,
this project follows the participant guarantee that its results will not
change. Use `refresh=True`, rotate `RAGTIME_CACHE_NAMESPACE`, or discard the
disposable L1 directory only for explicit manual invalidation. Cache keys
include the base URL, operation, namespace, and exact JSON request, but never
the bearer token.

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
  and distinguishes a local artifact hit from the service's own `cached` field.
- Cached private queries and responses must be treated as participant data. The
  L1 artifact directory is created with user-only permissions and must not be
  committed; the durable L2 bucket must remain private.

### One-Time Legacy Cache Import

The runtime has no SQLite cache path. If an existing legacy response database
must be migrated, use the explicit offline importer after configuring the
private bucket and decrypting `.env`:

```bash
sops decrypt --output .env config/secrets.sops.env
uv run ragtime-cache-import \
  /absolute/path/to/api-v1.sqlite3 \
  --manifest /absolute/path/to/cache-import-manifest.json
```

The importer opens the source with SQLite `mode=ro`, accepts only schema `1`,
namespace `ragtime2-2026-v1`, and the `pipeline`, `content`, and
`huggingface_document` operations. Every accepted row is checked with the
production endpoint validator, applies the 30-day Dataset Viewer TTL,
preserves its finite `stored_at`, and is independently reread from the
destination. Unsupported, malformed, and expired rows are counted as skipped;
`write() is False`, an independent verification miss, or an independent
verification mismatch is a strict migration failure and is never counted as an
imported or verified row. The importer does not publish a successful manifest
after any such failure.
The manifest contains only source hash/size, destination bucket metadata, and
sanitized counts; it contains no request text, response bodies, URLs with
credentials, or tokens. The source database is never deleted or modified.

### Reuse Existing Retrieval Runs

Control-A and Baseline B/C runs are mutable JSONL checkpoints for audit and
resume. Their durable cross-process cache is one complete hydrated query object
per exact semantic request in the private artifact store:

```text
retrieval/v1/<retriever-fingerprint>/<request-prefix>/<request-digest>.mpk
```

The runners first try that object. A warm B/C topic reconstruction loads the
objects by exact query context and performs no organizer request or DuckDB
hydration. Baseline C imports every successful facet checkpoint, including IDs
that were not retained in the fused topic's top-100 candidates.

To migrate an existing immutable run directory, decrypt `.env` first and give
the importer an explicit local manifest path:

```bash
sops decrypt --output .env config/secrets.sops.env
uv run ragtime-retrieval-import \
  /absolute/path/to/control-a-or-query-baseline-run \
  --corpus /absolute/path/to/ragtime2-corpus-v2.duckdb \
  --manifest /absolute/path/to/retrieval-import-manifest.json
```

The importer accepts only `control_a_v1` and `query_baseline_v1` B/C
directories. It validates source hashes and pinned corpus identity, hydrates
exact IDs only through `LocalCorpusIndex.enrich_documents`, writes complete
objects through `RetrievalArtifactCache`, then evicts its disposable L1 copy
and independently rereads the private L2 object. Malformed or partial input
records may be counted as skipped; an unsuccessful write, L2 verification miss,
or decoded-artifact mismatch is a strict migration failure, never a counted
success, and produces no successful output manifest. The source run files are
read-only and the owner-only manifest contains only file hashes, sanitized
counts, corpus provenance, and retrieval object keys—not queries, document
bodies, credentials, or local cache secrets.

### Source-Linked Nuggets

`trec_ragtime.nuggets` compresses one topic's ranked passages into atomic
English claims where every claim resolves to a document and an exact character
span. It is an internal evidence-compression stage, not a Nuggetizer-track
submission, and the package's `vital`/`okay` scores are signals rather than
human ground truth.

Upstream [`nuggetizer`](https://github.com/castorini/nuggetizer) returns claim
text with no document identity, so linkage comes from its assigner: every claim
is judged against every evidence passage, and that support matrix is the
lineage. A claim no passage supports is quarantined, and a topic where nothing
survives fails without writing anything.

```text
create(query, all passages)  ->  claims + vital/okay
assign(query, ONE passage, claims) x N  ->  support / partial_support / not_support
```

The pinned version is `nuggetizer==0.0.5`. Later upstream commits move the
prompts into `src/nuggetizer/prompts/` without adding that package to the
setuptools `include` list, so a Git-pinned build installs and then fails on
`from ..prompts import ...`. Revisit the pin when upstream publishes a fixed
release; the installed version and revision are part of every artifact
fingerprint, so a change retires the previous objects.

The provider credential is resolved by `trec_ragtime.providers.provider_settings`
from the decrypted `.env`, so `OPENROUTER_API_KEY` is picked up automatically and
implies the OpenRouter base URL. Pinned 0.0.5 reads only `OPEN_AI_API_KEY` or
`OPENAI_API_KEY` and always points at OpenAI, so the resolved base URL and key
are handed to it explicitly rather than left to its own lookup. Upstream exposes
no seed, so identical inputs may still produce different claims on a cold run:
determinism comes from the artifact cache, not from the provider. Upstream also
retries a failed provider call forever, so this stage caps each topic's provider
calls and turns exhaustion into a failed topic rather than a hung batch.

Bundles are cached per topic, keyed by a slow-moving behaviour fingerprint and
the topic's exact inputs:

```text
nuggets/v1/<nuggetizer-fingerprint>/<request-prefix>/<request-digest>.mpk
```

The request identity covers the topic, the exact question asked, the source
representation, the input retrieval artifacts, the passage-source fingerprint,
and a digest of the ordered alias table. Rewording the question, or altering an
alias, document, or offset, therefore yields a different key rather than a stale
hit.

The bundle carries the complete evidence-alias table, ordered nuggets with
their attesting passages, facet coverage, and sanitized rejection counts. It
deliberately stores no passage text: the offsets and `content_hash` prove which
text was used, and the text itself is recoverable from the retrieval artifact,
so the bundle cannot drift from the corpus. Paraphrases are merged with a
pinned `sentence-transformers/all-MiniLM-L6-v2` cosine threshold guarded by a
numeric and negation check, so `fell 30%` never collapses into `fell 13%`, and
a merge keeps every attesting source.

Run it over the planned topics after decrypting `.env`:

```bash
uv run ragtime-nuggets --model <model-id> --topic-id 2000 --evidence-budget 24
```

Retrieval is read from the artifact cache only; a topic whose facets have not
been retrieved is reported rather than turned into an organizer request. The
query plan's own facet numbering is carried through to both the projected
evidence and the recorded inputs, so a facet with no cached retrieval is skipped
without renumbering the ones around it. `trec_ragtime.nuggets_runner` remains
the supported Python API for callers that want the stage without the command.

### Retrieval bundle for report generation

After the nugget cache is complete, materialize the retrieval pipeline's final
consumer artifact with the same model and scorer identity used above:

```bash
uv run ragtime-retrieval-handoff \
  --model <model-id> \
  --manifest /absolute/path/to/retrieval-handoff-manifest.json
```

This command is cache-only. It reconstructs the exact retrieval, passage-score,
and nugget identities and fails a topic when any required artifact is absent;
it never starts retrieval, hydration, GPU scoring, or provider work. Each
successful topic produces one self-contained MessagePack bundle under
`retrieval-handoff/v1/`. The bundle includes the official topic record, retained
deduplicated nuggets, and only the exact English passage slices linked by those
nuggets. Full `support` links are citation-ready. `partial_support` is retained
as labeled context but is not citation-ready.

Report generation consumes the typed bundle directly:

```python
bundle = handoff_cache.get(fingerprint, request)
if bundle is None:
    raise RuntimeError("retrieval handoff is incomplete")
for nugget in bundle.nuggets:
    citations = tuple(link.alias for link in nugget.support if link.citation_ready)
```

Generation does not reopen raw retrieval or nugget artifacts and does not need
to deduplicate evidence. The owner-only manifest records per-topic object keys
and counts without topic or passage text.

## Repository Map

- [`AGENTS.md`](AGENTS.md): contributor and agent instructions, including the
  required decrypt-first secret workflow.
- [`src/trec_ragtime`](src/trec_ragtime): reusable private API client, local
  DuckDB corpus index and builder, Dataset Viewer fallback, high-level
  retriever, typed responses, and cross-worktree caches.
- [`topics/topics.all.2026.v0625-fix.jsonl`](topics/topics.all.2026.v0625-fix.jsonl):
  [official corrected June 25 test topics](https://trec.nist.gov/act_part/tracks/ragtime/topics.all.2026.v0625-fix.jsonl),
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
