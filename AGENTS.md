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
