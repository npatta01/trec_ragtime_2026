# Task 3 Report — Minimal Real Flash Worker Vertical Slice

## Status

Implemented the deadline-focused worker slice: exact-request cache-first scoring, genuine hit/miss telemetry, complete canonical artifact write/reread, fail-closed cache paths, private HF Bucket/simplecache production construction, and worker range/concurrency configuration.

## Files

- `src/trec_ragtime/flash_worker.py`
- `src/trec_ragtime/flash_passage_scorer.py`
- `tests/test_flash_passage_scorer.py`

## Commit

Pending commit at report-writing time; production and test changes will be committed together.

## RED/GREEN evidence

RED before implementation:

```text
uv run python -m unittest tests.test_flash_passage_scorer.StaticFlashWorkerTests.test_cold_miss_writes_rereads_and_returns_complete_scores tests.test_flash_passage_scorer.StaticFlashWorkerTests.test_warm_hit_on_fresh_worker_does_not_load_model tests.test_flash_passage_scorer.StaticFlashWorkerTests.test_65_rows_use_three_model_batches_and_report_three tests.test_flash_passage_scorer.RunpodFlashConfigTests.test_worker_tuple_range_and_max_concurrency_reach_endpoint_factory -v
Ran 4 tests ...
FAILED (errors=4)
TypeError: FlashPassageWorker.__init__() got an unexpected keyword argument cache_store
TypeError: RunpodFlashConfig.__init__() got an unexpected keyword argument max_concurrency
```

GREEN:

```text
sops decrypt --output .env config/secrets.sops.env
git check-ignore .env
=> .env
uv run python -m unittest tests.test_flash_passage_scorer -v
=> Ran 63 tests in 1.018s — OK
uv run python -m unittest tests.test_passage_reranking_integration -v
=> Ran 3 tests in 4.647s — OK
git diff --check
=> exit 0, no output
```

## Self-review

- Cache identity/codec remain private to `FlashPassageWorker`; no caller-side cache API was added.
- Hits return before model loading; misses count actual `predict` batches, require strict `True` writes, evict only the exact local key, and return only independently reread durable scores.
- Fixed credential-safe errors cover malformed hits, model failures, false writes, and bad/missing rereads.
- Extracted class execution with project imports blocked remains covered with an injected strict store. Production construction uses method-local imports and the minimum class-body-safe fallback.
- `.flash/` was pre-existing and untouched.

## Concerns

- Live Hugging Face Bucket permissions, simplecache transport, Runpod Flash deployment, and one-GPU smoke remain for the parent task. Offline tests cover configuration and store seams, not live remote behavior.
- Existing `ruff check` reports the pre-existing unused `PropertyMock` import in `tests/test_flash_passage_scorer.py`.
