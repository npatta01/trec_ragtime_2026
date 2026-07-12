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
The notebook loads the ignored root `.env` file automatically and skips private
retrieval when either value is absent. Do not add credentials to the notebook,
repository, or saved cell outputs.

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

To change the encrypted values, edit them through SOPS and decrypt the updated
file again:

```bash
sops config/secrets.sops.env
sops decrypt --output .env config/secrets.sops.env
```

For an unshared local configuration, copy `.env.example` to `.env` and fill in
the token. The plaintext `.env` file is ignored by Git.

## Repository Map

- [`topics/topics.all.2026.v0625.jsonl`](topics/topics.all.2026.v0625.jsonl):
  [official June 25 test topics](https://trec.nist.gov/act_part/tracks/ragtime/topics.all.2026.v0625.jsonl),
  normalized to one JSON object per line.
- [`index.html`](index.html): compact GitHub Pages landing page for the project.
- [`reports/2026-quick-reference-and-2025-reproduction-playbook/index.html`](reports/2026-quick-reference-and-2025-reproduction-playbook/index.html):
  static 2026 quick reference and 2025 reproduction playbook.
- [`notebooks/ragtime2_mini_viewer.ipynb`](notebooks/ragtime2_mini_viewer.ipynb):
  bounded dataset preview and optional retrieval API example.
- [`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock): reproducible Python
  environment managed by uv.

The RAGTIME2 corpus is accessed remotely and is not vendored in this repository.
Track details can change, so use the official sources linked from the
[reference report](reports/2026-quick-reference-and-2025-reproduction-playbook/index.html)
when making submission decisions.
