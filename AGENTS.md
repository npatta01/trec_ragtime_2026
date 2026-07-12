# Agent Instructions

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
- `RAGTIME_DOCUMENT_BACKEND=auto` must prefer the shared local DuckDB index when
  present. An explicit local path also requires that artifact. `local` requires
  it and `viewer` explicitly selects the remote fallback. Do not silently fall
  back from a missing or corrupt configured local index.
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
- `HuggingFaceCorpus` stores validated documents individually in that same
  SQLite database. Its default TTL is 30 days and can be overridden with
  `RAGTIME_HF_DOCUMENT_CACHE_TTL_SECONDS`; `none`, `off`, or `disabled` means no
  expiry. Keep search and document refresh controls independent.
- The Dataset Viewer fallback cache is TTL/namespace based and does not pin a
  Hub revision. Keep it only for environments without the complete local
  index. Do not build another FAISS/vector index for hydration; document IDs
  are exact lookup keys after hosted retrieval.
- Cache keys must include the endpoint, operation, namespace, and exact request,
  but never the bearer token. Cache only validated successful responses.
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
