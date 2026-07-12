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

- Use `trec_ragtime.RagtimeClient` instead of duplicating HTTP or bearer-token
  handling in notebooks and scripts.
- Keep `ragtime2%20` as literal RoutIR DSL. It is not a URL-encoded space.
- Use `ragtime/2/all` only for topic records. Use `ragtime2` for the pipeline
  alias and for `/content` collection requests.
- Never trust JSON object order for search ranking. The reusable client sorts
  scores by descending score and then document ID.
- The default SQLite cache is outside the repository at
  `${XDG_CACHE_HOME:-~/.cache}/trec-ragtime-2026/api-v1.sqlite3`, so it is
  shared across worktrees. Do not replace it with a worktree-relative cache.
- Cache keys must include the endpoint, operation, namespace, and exact request,
  but never the bearer token. Cache only validated successful responses.
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
