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
- Use `trec_ragtime.HuggingFaceCorpus.get_many()` for direct batch hydration.
  It reads original-language rows from the Hugging Face `default` configuration
  across `eng`, `spa`, `rus`, and `zho`; do not silently substitute `mt_docs`,
  whose rows reuse the source IDs but contain different translated text.
- Eight measured documents, two per original-language split, matched the
  private `/content` text byte-for-byte. Preserve the original-language
  `default` configuration as the canonical hydration source unless a separate,
  explicit translation feature is being implemented and tested.
- For the default top 20, a cold hydration requires at most four sequential
  Hugging Face `/filter` operations rather than up to 20 private `/content`
  calls. Retry/backoff can make more than four wire-level GET attempts.
  Preserve batching, remove IDs found in earlier splits, and never depend on
  Dataset Viewer row order for retrieval rank.
- `RagtimeRetriever.retrieve()` makes no `/content` fallback calls by default
  because `max_content_fallbacks=0`. Keep fallback explicit and bounded. A
  Hugging Face split failure that leaves IDs unresolved must raise unless
  `fallback_on_hf_error=True` is explicitly supplied; that option must still
  respect `max_content_fallbacks`.
- Keep `ragtime2%20` as literal RoutIR DSL. It is not a URL-encoded space.
- Use `ragtime/2/all` only for topic records. Use `ragtime2` for the pipeline
  alias and for `/content` collection requests.
- Never trust JSON object order for search ranking. The reusable client sorts
  scores by descending score and then document ID.
- The default SQLite cache is outside the repository at
  `${XDG_CACHE_HOME:-~/.cache}/trec-ragtime-2026/api-v1.sqlite3`, so it is
  shared across worktrees. Do not replace it with a worktree-relative cache.
- `HuggingFaceCorpus` stores validated documents individually in that same
  SQLite database. Its default TTL is 30 days and can be overridden with
  `RAGTIME_HF_DOCUMENT_CACHE_TTL_SECONDS`; `none`, `off`, or `disabled` means no
  expiry. Keep search and document refresh controls independent.
- The Hugging Face cache is TTL/namespace based and does not pin a Hub
  revision. Refresh documents or rotate `RAGTIME_CACHE_NAMESPACE` after a known
  corpus update. Prefer a downloaded snapshot plus local ID index for large
  sweeps of mostly uncached queries.
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
