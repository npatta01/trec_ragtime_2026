# Agent Instructions

## TREC RAGTIME 2026 Competition Rules

- Treat the [TREC 2026 RAGTIME Track Guidelines](https://docs.google.com/document/d/1XqypNTHxIdazxfnUebrp-EsUuGNhodhQGrJIDnYGaB4/edit) as the source of truth for competition rules, submission formats, scoring, deadlines, and other track requirements. Consult the document whenever the repository instructions or task context leave a competition rule unclear.

## Secrets And Environment

- Use a decrypt-first workflow. Before running notebooks, scripts, tests, or
  application code, decrypt the committed SOPS file into the ignored root
  `.env` file:

  ```bash
  sops decrypt --output .env config/secrets.sops.env
  ```

- Run code normally after decryption. Do not wrap notebook or application
  commands with `sops exec-env` or another SOPS execution wrapper.
- The notebooks load the root `.env` automatically through `python-dotenv`.
  New Python entry points that require these credentials should do the same.
- Keep `.env` and `.env.*` ignored by Git, while retaining the tracked
  `.env.example`. Confirm with `git check-ignore .env` when changing ignore
  rules.
- Never commit, print, log, or save the plaintext bearer token in notebook
  outputs. The API endpoint is public configuration; the token is secret.
- The committed ciphertext is `config/secrets.sops.env`. Its GCP KMS creation
  rule is in `.sops.yaml` and uses the `np-public-training` project key.
- To update credentials, edit the ciphertext through SOPS and then refresh the
  local `.env` before running code:

  ```bash
  sops config/secrets.sops.env
  sops decrypt --output .env config/secrets.sops.env
  ```

## Running The Notebooks

After decrypting `.env`, run notebooks without a SOPS wrapper:

```bash
uv run jupyter lab notebooks/ragtime2_mini_viewer.ipynb
uv run jupyter lab notebooks/ragtime_single_query.ipynb
```

## Reusable Client And Cache

- Use `trec_ragtime.RagtimeRetriever` for search followed by document hydration.
  Use `trec_ragtime.RagtimeClient` only when raw search results or an explicit
  single `/content` lookup is required. Do not duplicate HTTP or bearer-token
  handling in notebooks and scripts.
- Use `LocalCorpusIndex.enrich_documents(ids)` for direct local hydration, or
  `RagtimeRetriever.enrich_documents(ids)` when the search client is already
  open. Both must preserve first-seen order, deduplicate lookup work, and return
  unresolved IDs explicitly. Never impose a 20-document hydration cap.
- `RagtimeRetriever.from_env()` must require the shared canonical local DuckDB
  index and fail clearly when it is missing, corrupt, or incomplete. Configure
  its location with `RAGTIME_PROJECT_CACHE_DIR` or
  `RAGTIME_LOCAL_CORPUS_PATH`; do not silently fall back to Dataset Viewer.
- Use `HuggingFaceCorpus` directly only when an explicit Dataset Viewer client
  is required in an environment without the complete local index.
- The canonical corpus is Hugging Face `default/{eng,spa,rus,zho}`. Never mix in
  `mt_docs` as additional retrieval documents. Its rows reuse Spanish, Russian,
  and Chinese source IDs; join them as optional English representations.
- Preserve `document.text` as canonical source content. Use
  `document.report_text` at the English report-synthesis boundary and inspect
  `missing_translation_ids` separately from unknown base `unresolved_ids`.
- All 11 cached private `/content` samples matched the Hugging Face source text
  byte-for-byte, including two per original-language split. Preserve the
  original-language `default` configuration as canonical hydration evidence;
  keep `mt_docs` explicitly separate as an English synthesis representation.
- Request counts above 20 by changing the RoutIR suffix, such as
  `ragtime2%100`; `top_k=None` enriches everything the endpoint returned, while
  `top_k=N` only truncates that existing result set.
- `RagtimeRetriever.retrieve()` automatically makes at most five forever-cached
  `/content` calls for true missing base IDs, stopping after the first error.
  `max_content_fallbacks=0` disables this and `None` explicitly allows all true
  misses. A Dataset Viewer split failure that leaves IDs unresolved must raise
  unless `fallback_on_hf_error=True` is explicitly supplied; that option must
  still use a finite `max_content_fallbacks`.
- Keep `ragtime2%20` as literal RoutIR DSL. It is not a URL-encoded space.
- Use `ragtime/2/all` only for topic records. Use `ragtime2` for the pipeline
  alias and for `/content` collection requests.
- Never trust JSON object order for search ranking. The reusable client sorts
  scores by descending score and then document ID.
- The project cache is `<git-common-dir>/ragtime-cache`, so the API cache,
  source snapshot, and DuckDB index are inherently shared by every Git
  worktree and cannot be committed. Do not use a worktree-relative cache.
- The complete local corpus defaults to
  `<git-common-dir>/ragtime-cache/ragtime2-corpus-v2.duckdb`.
  It is revision-pinned, read-only at runtime, and shared across worktrees.
  Build or inspect it only after decrypting `.env`:

  ```bash
  uv run ragtime-corpus build
  uv run ragtime-corpus status
  ```

- A local corpus lookup must not call Dataset Viewer or write per-document
  response-cache entries; the materialized indexed corpus is already the
  durable source. Keep source Parquet outside every tracked worktree for
  rebuilds. The builder must verify the exact pinned Hub manifest and every
  source shard's size and LFS SHA-256 before indexing. It must also prove
  `mt_docs` unique-ID, source-split, English-language, and bidirectional
  coverage invariants. `ragtime-corpus status` must deep-check those indexed
  invariants before reporting English translations as complete.
- Treat the participant API as immutable. `/pipeline` and `/content` cache
  entries do not expire by default; retain explicit refresh and namespace
  rotation only as manual invalidation controls.
- Private pipeline/content responses and Dataset Viewer documents use the
  private Hugging Face Bucket as durable L2 storage, read through the
  disposable owner-only `fsspec.simplecache` L1. Configure it with
  `RAGTIME_HF_BUCKET_ID`, optional `RAGTIME_ARTIFACT_PREFIX`, and optional
  `RAGTIME_ARTIFACT_CACHE_DIR`; `HF_TOKEN` is the optional Hugging Face token.
  L1 cache, transport, permission, and malformed L2 failures are fatal; a
  malformed L1 object may be repaired once from L2. `HuggingFaceCorpus` (the
  deprecated Dataset Viewer backend) uses a 30-day document TTL, configurable
  with `RAGTIME_HF_DOCUMENT_CACHE_TTL_SECONDS`; `none`, `off`, or `disabled`
  means no expiry. Keep search and document refresh controls independent.
- The explicit `ragtime-cache-import` command is the only SQLite reader. It
  opens the legacy source read-only, validates canonical rows, preserves
  `stored_at`, independently verifies every destination artifact, and writes a
  sanitized owner-only manifest. It never deletes or modifies the source.
- The explicit `ragtime-retrieval-import` command is the only importer for
  existing Control-A/B/C retrieval run directories. It accepts only
  `control_a_v1` and `query_baseline_v1`, validates immutable run/source hashes
  and the pinned local corpus identity, and hydrates exact hit IDs only through
  `LocalCorpusIndex.enrich_documents`. It must not construct a participant
  client, call `/content`, or use Dataset Viewer. Complete queries are written
  through `RetrievalArtifactCache`, then their L1 candidate is evicted and the
  typed object is independently reread from L2. Failed/partial records are
  skipped rather than upgraded to success. The output manifest is explicit,
  atomic, owner-only, and sanitized: source hashes, counts, corpus provenance,
  and object keys are allowed; query/body payloads, credentials, and local
  cache secrets are not. The mutable JSONL run remains an audit/checkpoint
  input; the `retrieval/v1/<fingerprint>/<prefix>/<request>.mpk` object is the
  durable shared query cache reused by warm runners.
- The Dataset Viewer fallback cache is TTL/namespace based and does not pin a
  Hub revision. Keep it only for environments without the complete local
  index. Do not build another FAISS/vector index for hydration; document IDs
  are exact lookup keys after hosted retrieval.
- `trec_ragtime.nuggetizer_adapter` is the only module that may import
  `nuggetizer`. Keep the pin at the published `nuggetizer==0.0.5`: later
  upstream commits move the prompts into `src/nuggetizer/prompts/` without
  adding that package to the setuptools `include` list, so a Git-pinned build
  installs and then fails on `from ..prompts import ...`. Provider routing is
  environment driven through `OPENAI_API_KEY` and `OPENAI_BASE_URL`; never pass
  credentials through the adapter. Upstream retries a failed call forever, so
  every topic runs under a finite provider-call budget.
- Nugget source linkage comes from the package's assigner, never from a model
  echoing a document ID. A claim no passage supports is quarantined, and a topic
  where nothing survives fails without writing an artifact. Upstream
  `vital`/`okay` and support labels are signals, not qrels or human judgments.
- A nugget bundle never stores passage text. Offsets plus `content_hash` prove
  which text was used and the text stays recoverable from the retrieval
  artifact. The request identity pins the ordered alias table, so an altered
  alias, document, or offset cannot be read back under the same key.
- Cache keys must include the endpoint, operation, namespace, and exact request,
  but never the bearer token. Cache only validated successful responses.
- The OpenAI-compatible chat-completion cache stores the complete validated
  provider response payload, including usage and provider extension fields.
  Client API keys and transport headers/options remain outside the identity and
  stored artifact; top-level credential-shaped request fields are rejected,
  while nested tool schemas and provider bodies retain their exact properties.
- The public Hugging Face dataset does not require the private RAGTIME token.
  Never send the RAGTIME authorization header to Hugging Face or include it in
  document-cache keys, payloads, errors, or logs.
- The hosted endpoint uses `Qwen/Qwen3-Embedding-8B`, FAISS, and RoutIR. It is a
  retriever, not a report-generation model. Do not describe unreported serving
  details such as the deployed vector dimension, instruction, quantization, or
  immutable index version as known facts.
- Prefer measured query decomposition, result fusion, or multilingual
  reranking before introducing a separately maintained embedding index.
- Decrypt `.env` first, then run the offline tests normally:

  ```bash
  uv run python -m unittest discover -s tests -v
  ```
