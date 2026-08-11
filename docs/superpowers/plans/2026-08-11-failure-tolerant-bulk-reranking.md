# Failure-Tolerant Bulk Reranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Discard only document representations whose strict passage chunking fails, while scoring and persisting every valid document and query.

**Architecture:** Catch the typed `PassageChunkingError` at the per-document boundary in `rerank_retrieval` and emit an unscored `invalid_passages` document row. Keep unexpected chunker, scorer, and artifact failures fatal. Extend the disposable bulk runner's output validator to accept the new typed status, then resume the two incomplete manifests from their first known failures so durable outputs are skipped.

**Tech Stack:** Python 3.11, `unittest`, msgspec MessagePack artifacts, dstack, Runpod A40/A6000, Hugging Face Bucket artifacts.

## Global Constraints

- Passage validators remain strict and malformed text is never rewritten.
- Only `PassageChunkingError` is recoverable at the document boundary.
- `invalid_passages` rows have no score or evidence and sort with other unscored rows.
- Cache, model, decoding, GPU, credential, and upload failures remain fatal.
- Recovery keeps CUDA/bfloat16, batch size 32, `$0.60/GPU-hour`, and three-hour limits.

---

### Task 1: Preserve queries when one document cannot be chunked

**Files:**
- Modify: `tests/test_reranking.py`
- Modify: `src/trec_ragtime/reranking.py`
- Modify: `/tmp/trec-ragtime-bulk-rerank/runner/bulk_rerank.py`
- Modify: `/tmp/trec-ragtime-bulk-rerank/runner/tests/test_runner.py`

**Interfaces:**
- Consumes: `PassageChunkingError`, `rerank_retrieval(...) -> RerankingResult`
- Produces: document status literal `invalid_passages`; output-wire validation accepting that literal

- [ ] **Step 1: Add a failing core behavior test**

Add a test chunker that raises `PassageChunkingError` only for one document and assert that `rerank_retrieval` returns that document as `invalid_passages`, scores the valid document, and emits no evidence for the invalid document. Add a second assertion that a plain `RuntimeError` from a chunker is still wrapped as `RerankingError`.

- [ ] **Step 2: Verify the core test fails for the missing status behavior**

Run: `uv run python -m unittest tests.test_reranking.RerankingTests.test_invalid_passages_discard_only_the_failing_document -v`

Expected: ERROR with `RerankingError: chunker failed for a selected representation` caused by `PassageChunkingError`.

- [ ] **Step 3: Implement the narrow document-level discard**

Import `PassageChunkingError`, add `invalid_passages` to `_Status`, catch only that exception around `chunker.chunk`, append `(document, "invalid_passages", ())`, and continue. Preserve the existing catch-and-wrap behavior for every other exception.

- [ ] **Step 4: Verify core behavior and regressions**

Run: `uv run python -m unittest tests.test_reranking -v`

Expected: all reranking tests pass.

- [ ] **Step 5: Add and verify a failing runner-wire test**

Construct a minimal `RerankingResult` containing one `invalid_passages` row, encode it with `encode_output`, and assert `decode_output` round-trips it. Run the focused runner test and verify it fails with `RunnerError: reranking output failed validation` because the status is not allowlisted.

- [ ] **Step 6: Extend only the runner output status allowlist**

Add `invalid_passages` beside `scored`, `missing_representation`, `unresolved`, and `no_passages`. Do not change score/evidence validation.

- [ ] **Step 7: Run complete local verification**

Run:

```bash
uv run python -m unittest tests.test_reranking -v
uv run python -m unittest discover -s /tmp/trec-ragtime-bulk-rerank/runner/tests -v
uv run python -m unittest discover -s tests -v
```

Expected: all commands exit 0 with no failed tests.

- [ ] **Step 8: Prepare and dry-run recovery selections**

Configure worker 2 to start at zero-based manifest offset 136 and worker 3 at offset 71, retaining their original worker labels so earlier output artifacts are skipped. Dry-run totals must be 76 and 141 respectively.

- [ ] **Step 9: Preview and launch recovery tasks**

Preview both exact dstack configs, confirm available offers remain within `$0.60/GPU-hour`, then submit detached. Verify both reach running state and cross their previously failing objects without `Traceback`, `RerankingError`, or `ERROR` in normal logs.

- [ ] **Step 10: Verify final artifact completion**

Require successful runner summaries covering all selected entries. Report total completed/skipped queries, `invalid_passages` document count, final cost, and any remaining fatal failure without exposing query or document content.
