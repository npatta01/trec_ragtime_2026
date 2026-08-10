# Passage Reranking Implementation Plan

## Implementation status — 2026-08-10

- Objective and merge base remain unchanged: issue #15 on
  `codex/passage-reranking`, based on `72596ad` / `origin/codex/control-a-foundation`.
- Tasks 1–6 are implemented and task-reviewed. The main implementation commits
  cover LangChain/spaCy passage lineage, local scoring, complete ArtifactStoreLike
  score maps, deterministic reranking artifacts, optional Flash transport, and
  public integration/documentation.
- Final Sol standards/spec review requested four implementation fixes. Passage
  configuration/runtime identity is fixed in `f841547`; pair-token and scorer
  runtime identity is fixed in `945b304`; Flash startup/runtime/provisioning is
  fixed in `604f4f4`. Public integration of those settled interfaces is in
  progress.
- Latest evidence before public re-integration: 26 passage tests, 22 scorer
  tests, 64 Flash+scorer tests, 246 full repository tests, and the multilingual
  benchmark are green. A real pinned-tokenizer-only CJK smoke proved 502
  characters / 513 pair tokens become two exact passages with a 512-token
  maximum; no model weights or inference were used.
- Next action: finish exports/README/integration regression, run scoped Sol
  re-review, then repeat focused, benchmark, full, lock, diff, and secret gates.
- External gates remain deferred and must not be represented as complete: real
  model-weight inference, organizer/canonical/report ranking comparison, paid
  `flash dev`, and cross-machine reuse after issue #12 lands.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add boundary-aware passage preparation, local and optional Runpod Flash cross-encoder scoring, complete per-query ArtifactStore score reuse, and deterministic document reranking for hydrated RAGTIME retrievals.

**Architecture:** `passages.py` adapts maintained LangChain/spaCy splitters into strict contiguous lineage records. Independent scoring, passage-score cache, ranking, and Flash transport modules communicate through small protocols; reranking remains downstream of `RagtimeRetriever` and does not change Control-A/B/C or the pending #12/#13 implementations.

**Tech Stack:** Python 3.11+, `langchain-text-splitters`, spaCy, `sentence-transformers`, target-model Hugging Face tokenizer, `msgspec` MessagePack, optional `runpod-flash`, `unittest`, and `uv`.

## Global Constraints

- Base every change on `origin/codex/control-a-foundation` and work only on `codex/passage-reranking`.
- Decrypt `config/secrets.sops.env` to ignored root `.env` before running Python, tests, scripts, or notebooks.
- Preserve unrelated work; do not push, open a pull request, create a persistent Runpod deployment, run `flash deploy`, or make live HF Bucket/organizer writes. Ephemeral `flash dev` provisioning is allowed only as a separately authorized deferred gate and is not part of offline implementation.
- Because #12/#13 remain pending and this branch must stay based on Control-A, do not copy or preemptively rebase their work. Task 1 exclusively owns `pyproject.toml`/`uv.lock`, Task 6 exclusively owns `src/trec_ragtime/__init__.py`/README/environment documentation, and Sol owns final follow-up reconciliation of those shared files after #12/#13 land. No other worker may edit them.
- Normal passage preparation uses English `report_text`; canonical English, Spanish, Russian, and Chinese remain explicit supported representations.
- Use LangChain/spaCy for paragraph packing, sentence segmentation, and adjacent-sentence packing; do not implement a project-owned paragraph/sentence parser or greedy packer.
- Use `RecursiveCharacterTextSplitter` only when one individual sentence exceeds the configured model token ceiling.
- Require an explicit length callable; production uses the pinned tokenizer to
  count the complete formatted query/passage pair, while explicit `len` is
  boundary-test-only.
- Combine only adjacent source paragraphs/sentences, never reorder text, use zero overlap in v1, and require maintained `create_documents()` start indexes, cursor equality, full reconstruction, and exact source slices at every nested splitter stage.
- Preserve exact Python character offsets and exact UTF-8 SHA-256 bytes without Unicode normalization or whitespace stripping.
- `PassageScorer.score(query, passages)` returns exactly one finite non-Boolean float per unique binary passage-content hash.
- Store successful validated score unions as deterministic `dict[bytes, float]` MessagePack under `passage-scores/v1/<effective-score-fingerprint>/<query-prefix>/<query-hash>.mpk`; the effective fingerprint covers schema, scorer, chunker, and representation.
- Consume the issue #12 `ArtifactStoreLike` shape (`read`, `read_validated`,
  `write`, `evict_local`) while the cache adapter itself uses only validated
  reads and writes; do not edit or reimplement HF Bucket/simplecache mechanics.
- Failed, partial, malformed, non-finite, truncated, transport-error, or storage-error scoring never becomes a successful cache artifact.
- Aggregate documents with versioned `max_passage_v1`, preserve organizer provenance, and retain unscored documents with explicit statuses.
- Local scoring is the default and supports CPU/CUDA plus a user-facing ROCm alias normalized to PyTorch `cuda`; Runpod Flash is optional transport using the same scorer contract.
- Runpod payloads remain below 10 MB, batches/concurrency/timeouts are bounded, workers load the model once, and logs contain no queries, passage text, tokens, or credentials.

---

### Task 1: Exact LangChain/spaCy passage preparation

**Files:**
- Create: `src/trec_ragtime/passages.py`
- Create: `tests/test_passages.py`
- Create: `tests/fixtures/passage_chunking.json`
- Create: `scripts/benchmark_passage_chunking.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: a spaCy pipeline name and a Hugging Face tokenizer-compatible length function.
- Produces: `Passage`, `PassageChunker`, `PassageChunkingError`, `PassageValidationError`, `ChunkerConfig`, `LangChainPassageChunker`, `sha256_bytes`, and `validate_passages(selected_text, passages)`.

- [ ] **Step 1: Add fixture-driven failing tests for exact sentence chunks and lineage**

  Add literal fixtures for English, Spanish, Russian, Chinese, English machine translation, CRLF, headings/lists, repeated identical sentences, short whole-document text, paragraphs, and one deliberately oversized sentence. Tests must assert literal start/end offsets and text, source/representation/content SHA-256 bytes, monotonic spans, cursor equality, full reconstruction at every splitter level, adjacency, zero overlap, neighbor IDs, stable IDs across a subprocess, and exact source slices. Prove the recursive fallback runs exactly once for the individually oversized sentence and never for a packed multi-sentence chunk. Name each tested mutation: whitespace stripping, offset reconstruction, sentence reordering, non-deterministic IDs, or long-sentence overflow.

- [ ] **Step 2: Run the passage tests and capture the expected red state**

  Run:

  ```bash
  uv run python -m unittest tests.test_passages -v
  ```

  Expected: import failure for `trec_ragtime.passages`.

- [ ] **Step 3: Add maintained chunking dependencies and lock them**

  Add direct compatible ranges for `langchain-text-splitters>=1.1.2,<2`, `spacy>=3.8,<4`, `sentence-transformers>=5.6,<6`, and `msgspec>=0.19,<1`. Add pinned spaCy 3.8.0 model wheels for `en_core_web_sm`, `es_core_news_sm`, `ru_core_news_sm`, and `zh_core_web_sm`. Add optional extra `runpod = ["runpod-flash>=1.18,<2"]`. Run `uv lock` and record resolved versions in the benchmark output/fingerprint tests.

- [ ] **Step 4: Implement strict Passage and ChunkerConfig validation**

  Implement frozen slotted dataclasses. Reject blank document IDs, unknown representations, malformed/non-32-byte hashes, invalid offsets, empty text, mismatched source slices, non-monotonic or overlapping spans, inconsistent indexes/neighbors, nonzero v1 overlap, and invalid chunker fingerprints. Hash exact UTF-8 bytes and encode passage IDs from a canonical `msgspec.msgpack` lineage tuple.

- [ ] **Step 5: Implement LangChainPassageChunker with exceptional recursive fallback**

  Use the identical target-model `chunk_size` and tokenizer `length_function` at all three stages. First use `CharacterTextSplitter(separator=r"(?:\r?\n){2,}", is_separator_regex=True, keep_separator="end", strip_whitespace=False, chunk_overlap=0, add_start_index=True)` to pack complete adjacent LF/CRLF paragraphs. Descend only an individually over-budget paragraph into `SpacyTextSplitter(separator="", strip_whitespace=False, chunk_overlap=0, add_start_index=True)` using the pinned language-to-pipeline mapping. Descend only an individually over-budget sentence into `RecursiveCharacterTextSplitter(separators=[" ", ""], keep_separator="end", strip_whitespace=False, chunk_overlap=0, add_start_index=True)`. Use `create_documents()` at all levels; validate nonnegative starts, cursor equality, exact parent slices, monotonic spans, and `"".join(chunks) == parent_text`, adjust nested offsets, and repeat final document-coordinate checks. Assign stable IDs and neighbors only after all exact spans exist. Record boundary metadata as `paragraph_packed`, `sentence_packed`, or `oversized_sentence_fallback`; do not invent sentence ordinal ranges.

- [ ] **Step 6: Run focused tests to green and refactor without behavior changes**

  Run the passage test command until all tests pass, then rerun after any naming or helper extraction. Output must have zero failures/errors and no unexpected warnings.

- [ ] **Step 7: Implement and run the cheap dependency benchmark**

  The script loads the literal fixture file, requires every pinned spaCy language model, runs LangChain/spaCy twice, verifies byte-identical records, exact offsets, per-level reconstruction, paragraph/sentence boundary behavior, and configured ceilings, then prints resolved package/model versions, fixture/pass/failure counts, startup seconds, documents/second, and passages/second. It must fail clearly when any required model is missing and must not download a cross-encoder or call a remote service.

  Run:

  ```bash
  uv run python scripts/benchmark_passage_chunking.py
  ```

- [ ] **Step 8: Commit Task 1**

  ```bash
  git add pyproject.toml uv.lock src/trec_ragtime/passages.py tests/test_passages.py tests/fixtures/passage_chunking.json scripts/benchmark_passage_chunking.py
  git commit -m "feat: add exact boundary-aware passages"
  ```

### Task 2: Strict scorer protocol and local cross-encoder

**Files:**
- Create: `src/trec_ragtime/passage_scoring.py`
- Create: `tests/test_passage_scoring.py`

**Interfaces:**
- Consumes: `Passage` from Task 1 and an injected CrossEncoder/tokenizer loader.
- Produces: `PassageScorer`, `PassageScoringError`, `PassageScoreValidationError`, `LocalCrossEncoderConfig`, `LocalCrossEncoderScorer`, and `validate_passage_scores(passages, scores)`.

- [ ] **Step 1: Write failing scorer-contract tests**

  Cover exact unique hash coverage, duplicate-content deduplication, missing/extra/malformed keys, Boolean/non-numeric/NaN/infinite scores, empty query, lazy one-time model loading, deterministic batching, CPU/CUDA device forwarding, normalization of user `rocm` to PyTorch `cuda`, raw-logit preservation, pair formatting, tokenizer overflow rejection before model invocation, and fingerprint changes for every result-affecting configuration field. Use a complete fake model/tokenizer at the external inference seam; assert returned behavior rather than fake existence.

- [ ] **Step 2: Verify the tests fail because the module is absent**

  ```bash
  uv run python -m unittest tests.test_passage_scoring -v
  ```

- [ ] **Step 3: Implement the Protocol and strict validator**

  Expose the exact `score(query: str, passages: Sequence[Passage]) -> Mapping[bytes, float]` contract plus immutable fingerprint. Normalize valid integer/float outputs to float only after rejecting Boolean and non-finite values. Require equality with the requested unique 32-byte content-hash set.

- [ ] **Step 4: Implement the lazy pinned local scorer**

  Default to `BAAI/bge-reranker-v2-m3` revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`. Load `sentence_transformers.CrossEncoder` and its tokenizer once under a lock, accept explicit device/precision/batch/max-length/query-format settings, tokenize each formatted pair without truncation, reject overflow, call `predict` in bounded batches with identity activation, and return raw finite logits by content hash. Do not download a model during tests.

- [ ] **Step 5: Run scorer and passage tests to green**

  ```bash
  uv run python -m unittest tests.test_passage_scoring tests.test_passages -v
  ```

- [ ] **Step 6: Commit Task 2**

  ```bash
  git add src/trec_ragtime/passage_scoring.py tests/test_passage_scoring.py
  git commit -m "feat: add local passage scorer protocol"
  ```

### Task 3: Complete per-query score cache over ArtifactStoreLike

**Files:**
- Create: `src/trec_ragtime/passage_score_cache.py`
- Create: `tests/test_passage_score_cache.py`

**Interfaces:**
- Consumes: Task 2 `PassageScorer`/validator and the future issue #12 `ArtifactStoreLike` shape (`read`, `read_validated`, `write`).
- Produces: `PassageScoreCache`, `CachedPassageScorer`, `PassageScoreArtifactValidationError`, `score_object_key`, and strict MessagePack envelope codecs.

- [ ] **Step 1: Write failing fake-store cache tests**

  Implement a test-only fake storing `dict[str, bytes]` and invoking decoders on hits. Cover stable key/bytes across subprocesses, exact query-byte isolation, scorer/chunker/representation/fingerprint isolation, cold miss, complete warm hit, overlapping-map reuse, exact missing-only scorer calls, deterministic complete-union rewrite, no rewrite on full hit, malformed/truncated/wrong-schema/wrong-query/wrong-fingerprint/non-finite/missing coverage rejection, storage error propagation, and no write after scorer failure or partial result.

- [ ] **Step 2: Verify red**

  ```bash
  uv run python -m unittest tests.test_passage_score_cache -v
  ```

- [ ] **Step 3: Implement exact interface compatibility and MessagePack schema**

  Reference issue #12 `ArtifactStoreLike` only under `TYPE_CHECKING`; use structural duck typing at runtime and do not ship a fallback store or counterfeit #12 validation exception. Use frozen strict `msgspec.Struct` envelopes with forbidden unknown fields. `PassageScoreArtifactValidationError` owns domain failures. Accept the exact artifact-store validation exception type/factory in the cache constructor, and wrap domain decoder failures in that exact type so the real store can repair L1. Tests inject the fake store's exception; after #12 lands, callers inject its `ArtifactValidationError`. Sort binary score keys before creating the dict, encode once, and immediately decode/domain-validate before write.

- [ ] **Step 4: Implement cache-aside merge semantics**

  Hash exact query UTF-8 only for overlap-reusing query identity. Build an effective score fingerprint from schema, model scorer fingerprint, chunker fingerprint, and representation for both object path and envelope. Record sorted coverage plus its digest in the envelope. On read, validate the whole stored union; call the scorer only for requested hashes absent from it; validate the exact missing response; merge and validate the complete union; write once. Return only the current request's unique hash map. Never catch store/model failures as misses.

- [ ] **Step 5: Run cache/scorer tests to green**

  ```bash
  uv run python -m unittest tests.test_passage_score_cache tests.test_passage_scoring -v
  ```

- [ ] **Step 6: Commit Task 3**

  ```bash
  git add src/trec_ragtime/passage_score_cache.py tests/test_passage_score_cache.py
  git commit -m "feat: cache complete passage score maps"
  ```

### Task 4: Deterministic retrieval reranking and artifacts

**Files:**
- Create: `src/trec_ragtime/reranking.py`
- Create: `tests/test_reranking.py`

**Interfaces:**
- Consumes: base `RetrievalResult`, Task 1 `PassageChunker`, and Task 2/3 scorer contract.
- Produces: `RerankedDocument`, `RerankingResult`, `rerank_retrieval`, `write_reranking_artifact`, and aggregation constant `max_passage_v1`.

- [ ] **Step 1: Write failing deterministic ranking tests**

  Cover canonical and report-text selection, report-text language `eng`, organizer/query/document provenance, max passage aggregation, winning-hash ties, document-score ties, shuffled input determinism, missing representation, unresolved/no-passage documents retained after scored rows, malformed rank/document alignment rejection, stable manifest/JSONL bytes, atomic output, and refusal to overwrite an existing run.

- [ ] **Step 2: Verify red**

  ```bash
  uv run python -m unittest tests.test_reranking -v
  ```

- [ ] **Step 3: Implement versioned aggregation and result types**

  Prepare passages only for valid selected representations. Score all unique passages once. Choose each document's maximum score and lexicographically smallest winning content hash. Sort scored rows by descending score, organizer rank, document ID, and winning hash; append explicit `missing_representation`, `unresolved`, or `no_passages` rows deterministically. Preserve original organizer rank/score and all fingerprints.

- [ ] **Step 4: Implement stable per-topic artifact writing**

  Require `write_reranking_artifact(..., source_retrieval_ref: str)` as an explicit immutable provenance input; never infer it from `RetrievalResult`. Write canonical compact sorted-key JSON manifest plus deterministic JSONL records through a temporary sibling directory, fsync/rename where existing repository helpers do so, and refuse any existing destination. Include schema, topic/query, representation, chunker/scorer/cache fingerprints, aggregation version, source retrieval reference, counts/statuses, and file SHA-256.

- [ ] **Step 5: Run ranking and upstream focused tests to green**

  ```bash
  uv run python -m unittest tests.test_reranking tests.test_passages tests.test_passage_scoring tests.test_passage_score_cache -v
  ```

- [ ] **Step 6: Commit Task 4**

  ```bash
  git add src/trec_ragtime/reranking.py tests/test_reranking.py
  git commit -m "feat: add deterministic passage reranking"
  ```

### Task 5: Optional local-parity Runpod Flash adapter

**Files:**
- Create: `src/trec_ragtime/flash_passage_scorer.py`
- Create: `src/trec_ragtime/flash_worker.py`
- Create: `tests/test_flash_passage_scorer.py`

**Interfaces:**
- Consumes: Task 1 `Passage`, Task 2 scorer validator/config, and an injected async Flash batch runner.
- Produces: `RunpodFlashConfig`, `RunpodFlashPassageScorer`, `FlashEndpointBatchRunner`, `build_flash_worker`, and strict wire codecs.

- [ ] **Step 1: Write failing fake-endpoint tests**

  Cover the same exact scorer contract, count batching, actual UTF-8 JSON payload limits below 10 MB, one oversized passage rejection, bounded concurrent batches, async `run()` plus `job.wait()`, timeout/error/malformed/missing/extra/non-finite response rejection, deterministic payload rows, exact local/remote parity under the same deterministic fake model, model/result-policy fingerprint isolation, transport-only setting non-isolation, and stdout/stderr capture proving no query/text/token/credential logging.

- [ ] **Step 2: Verify red**

  ```bash
  uv run python -m unittest tests.test_flash_passage_scorer -v
  ```

- [ ] **Step 3: Implement transport-only scorer and event-loop bridge**

  Serialize compact hash/text rows, measure final encoded bytes before dispatch, partition without crossing count/byte limits, and gather no more than configured concurrent batches. Validate every batch response and exact global coverage. Bridge the sync scorer method to async jobs with one private dedicated event-loop thread; never use nested `asyncio.run()`.

- [ ] **Step 4: Implement optional class-style warm worker definition**

  Lazily import `runpod_flash`; do not require it for local use or tests. Build a class endpoint whose initializer imports/loads the pinned CrossEncoder and tokenizer once per worker and whose scoring method performs the same pair-overflow and score validation. Keep worker dependencies/config explicit and do not write artifacts or logs from the worker. Endpoint ID, worker count, idle timeout, request timeout, and concurrency must not affect result identity; precision/device/batch policy may when it affects logits.

- [ ] **Step 5: Run Flash plus local scorer tests to green**

  ```bash
  uv run python -m unittest tests.test_flash_passage_scorer tests.test_passage_scoring -v
  ```

- [ ] **Step 6: Commit Task 5**

  ```bash
  git add src/trec_ragtime/flash_passage_scorer.py src/trec_ragtime/flash_worker.py tests/test_flash_passage_scorer.py
  git commit -m "feat: add optional Flash passage scorer"
  ```

### Task 6: Public integration, documentation, and complete verification

**Files:**
- Modify: `src/trec_ragtime/__init__.py`
- Modify: `README.md`
- Modify: `.env.example`
- Create: `tests/test_passage_reranking_integration.py`

**Interfaces:**
- Consumes: all Task 1-5 public types and base retrieval models.
- Produces: stable package exports, documented local-first usage, optional Flash configuration, and one representative fake end-to-end retrieval-to-ranking test.

- [ ] **Step 1: Write a failing public integration test**

  Import the documented public API, construct a complete multilingual `RetrievalResult`, run report-text and canonical variants through real chunk/cache/ranking components with only model/store/remote external seams faked, verify second-run score reuse, compare local and remote fake scorer rankings, and assert unresolved/missing-translation status preservation. The test must not call the network or require secrets/model weights.

- [ ] **Step 2: Verify red**

  ```bash
  uv run python -m unittest tests.test_passage_reranking_integration -v
  ```

- [ ] **Step 3: Export the supported API and document local-first operation**

  Export Passage/chunker/scorer/cache/ranking types without exporting test fakes. Document dependency/model pins, `report_text` default, canonical multilingual comparison, exact-offset/hash semantics, ArtifactStore handoff, CPU/CUDA plus ROCm-alias examples, local scoring as the primary runnable path, optional Runpod extra/config, payload and timeout controls, no-deploy status, and commands for the benchmark/focused/full tests. Add only non-secret configuration names to `.env.example`.

- [ ] **Step 4: Run focused integration and complete offline verification**

  Ensure `.env` is freshly decrypted, then run:

  ```bash
  uv run python -m unittest \
    tests.test_passages \
    tests.test_passage_scoring \
    tests.test_passage_score_cache \
    tests.test_reranking \
    tests.test_flash_passage_scorer \
    tests.test_passage_reranking_integration -v
  uv run python scripts/benchmark_passage_chunking.py
  uv run python -m unittest discover -s tests -v
  ```

  Verify zero failures/errors, no secret/plaintext leakage in tracked files or diff, and no Runpod/HF/organizer external mutation.

  Separately report four acceptance gates as authorized/evidenced or deferred: a real local pinned-model batch (large download), organizer-only versus canonical versus `report_text` comparison, ephemeral `flash dev` real-worker parity smoke (external cost), and L2/cross-machine reuse after #12 lands. Offline fakes do not satisfy those gates.

- [ ] **Step 5: Commit Task 6**

  ```bash
  git add src/trec_ragtime/__init__.py README.md .env.example tests/test_passage_reranking_integration.py
  git commit -m "docs: integrate passage reranking workflow"
  ```

### Task 7: Sol architecture integration and final review

**Files:**
- Modify only files identified by review findings.

**Interfaces:**
- Consumes: the complete branch diff from `origin/codex/control-a-foundation` and issue #15/spec requirements.
- Produces: a reviewed, verified branch with no unaddressed load-bearing findings.

- [ ] **Step 1: Generate a complete branch review package**

  Include commits, diff stat, full diff, issue #15 requirements, approved design, test/benchmark evidence, and any deferred task-review findings.

- [ ] **Step 2: Dispatch Sol xhigh for independent spec and quality review**

  Require separate verdicts for boundary/offset lineage, dependency policy, scorer/cache integrity, deterministic ranking, local/Flash parity, transport safety, secret handling, regression risk, and verification sufficiency.

- [ ] **Step 3: Dispatch one Luna xhigh fix wave if findings exist**

  Give one worker the complete findings list and explicit affected-file ownership. Require failing regression tests before fixes, focused reruns, and a written fix report. Then ask Sol xhigh for one scoped re-review of that fix diff.

- [ ] **Step 4: Run fresh final verification and inspect branch state**

  Re-run the complete Task 6 command sequence plus `git diff --check`, inspect `git status --short`, compare changed files to scope, and verify no secrets, logs, model weights, generated benchmark outputs, or unrelated artifacts are tracked.

- [ ] **Step 5: Hand off without push or pull request**

  Report branch, commits, changed modules, focused/full test counts, benchmark result, review outcome, and any remaining user action. Do not push or open a PR.
