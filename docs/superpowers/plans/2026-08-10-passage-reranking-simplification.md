# Passage Reranking Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broad issue #15 passage-reranking implementation with a compact LangChain/spaCy chunker, deterministic local scorer/cache, small reranker, and static Flash adapter while preserving exact text lineage, score integrity, and deterministic evidence.

**Architecture:** Workstream A establishes the only passage data contract and owns chunking, pair-length/scoring, score-cache, fixtures, and the benchmark. Workstreams B and C consume that settled contract without editing A-owned files: B normalizes retrieval rows once and returns compact reranking data; C transports the same scorer contract through bounded Flash batches and a static class-only worker. Workstream D updates only the public exports, integration coverage, and documentation after B and C are complete; reranking does not publish JSONL/files, because persistence remains the future `ArtifactStoreLike` concern.

**Tech Stack:** Python 3.11+, `langchain-text-splitters` 1.1.x, spaCy 3.8.x with the four pinned pipelines already declared in `pyproject.toml`, `sentence-transformers` 5.6.x, the pinned Hugging Face tokenizer/model policy, `msgspec`, optional `runpod-flash`, `unittest`, `uv`, and SOPS.

## Global Constraints

- Start from commit `94bd6a4`; preserve unrelated and concurrent edits. The implementation must not change files outside the ownership map below.
- Decrypt the committed SOPS file before every Python, benchmark, compile, import, formatter, linter, or test command: `sops decrypt --output .env config/secrets.sops.env`. Never print, log, commit, or save the bearer token; `.env` must remain ignored.
- Workstream A owns only `src/trec_ragtime/passages.py`, `src/trec_ragtime/passage_scoring.py`, `src/trec_ragtime/passage_score_cache.py`, `tests/test_passages.py`, `tests/test_passage_scoring.py`, `tests/test_passage_score_cache.py`, `tests/fixtures/passage_chunking.json`, and `scripts/benchmark_passage_chunking.py`.
- Workstream B owns only `src/trec_ragtime/reranking.py` and `tests/test_reranking.py`.
- Workstream C owns only `src/trec_ragtime/flash_passage_scorer.py`, `src/trec_ragtime/flash_worker.py`, and `tests/test_flash_passage_scorer.py`.
- Workstream D owns only `src/trec_ragtime/__init__.py`, `README.md`, and `tests/test_passage_reranking_integration.py`. Later workstreams read earlier interfaces but do not modify earlier workstream files.
- Settle the compact `Passage` and `PassageChunker` API in A before B or C starts. B and C must use the exact symbols and field names produced by A; neither may add compatibility fields to A-owned structures.
- Preserve the approved invariants: exact source slices with authoritative start/exclusive-end character offsets; stable SHA-256 bytes over exact UTF-8 passage text; complete selected-text union coverage; complete adjacent sentences in normal chunks; sentence-only overlap; `RecursiveCharacterTextSplitter` only for an individually oversized sentence; and exact formatted query/passage token validation before scoring.
- Use one domain exception per module (`PassageChunkingError`, `PassageScoringError`, `PassageScoreArtifactValidationError`, `RerankingError`, and `FlashScoringError`); do not keep parallel validation-error hierarchies or compatibility aliases.
- Deduplicate scoring by content hash only. Deduplicate evidence only by `(document_id, start_char, end_char)`; identical text at different offsets remains separate evidence while sharing one score.
- Retain maximum-passage document aggregation and deterministic ordering: descending reranker score, organizer rank, then document ID. Use deterministic source-offset evidence ordering and deterministic positional tie selection.
- Keep the score-cache result as a complete validated `dict[bytes, float]` per exact query, with binary hash keys, finite float values, exact requested coverage, and an `ArtifactStoreLike` seam. Do not add an `artifact_store.py` implementation or move artifact publication into the reranker.
- Keep local scoring as the primary path. The Flash adapter, worker definition, and tests must not provision, deploy, call a paid endpoint, download model weights, or require the optional Flash package merely to import or construct local fakes.
- The static worker must load the configured model and explicit tokenizer once on its first request, accept Transformers `BatchEncoding` mappings, enforce the exact pair-token ceiling, return complete finite score mappings, and fail closed if a later request changes the model fingerprint.
- Keep worker dependencies pinned from the existing installed runtime metadata, and preserve worker GPU, worker-count, idle-timeout, and decorator seams without dynamic generated source, source digests, module caches, or class caches.
- Delete internal-only tests for removed fields, recursive result-graph forgeries, platform-specific JSONL publication, generated-class identity, and other implementation details instead of mechanically rewriting them around the new API.
- No deploy, `flash dev`, paid call, organizer/Hugging Face write, push, pull request, or public publication is part of this implementation.

## File Ownership and Dependency Map

| Workstream | Files | Owns | Depends on |
| --- | --- | --- | --- |
| A — chunking/scorer/cache | `src/trec_ragtime/passages.py`, `src/trec_ragtime/passage_scoring.py`, `src/trec_ragtime/passage_score_cache.py`, their three focused tests, `tests/fixtures/passage_chunking.json`, `scripts/benchmark_passage_chunking.py` | compact passage records, exact chunking, pair-length/tokenizer policy, local scorer, validated score cache | base models and installed dependencies |
| B — reranking | `src/trec_ragtime/reranking.py`, `tests/test_reranking.py` | compact immutable result records, selection/status handling, score/evidence aggregation | A’s settled `Passage`, `PassageChunker`, `PassageScorer`, and cache-facing scorer contract |
| C — Flash | `src/trec_ragtime/flash_passage_scorer.py`, `src/trec_ragtime/flash_worker.py`, `tests/test_flash_passage_scorer.py` | thin wire/endpoint adapter and one static class-only worker | A’s `PassageScorer`, config, exact pair validation, and hash-score contract |
| D — integration/public/docs | `src/trec_ragtime/__init__.py`, `README.md`, `tests/test_passage_reranking_integration.py` | supported exports and local-only end-to-end documentation/tests | completed A, B, and C interfaces |

There is no planned `pyproject.toml` or `uv.lock` edit: the required LangChain, spaCy, model, scorer, `msgspec`, and optional Flash dependency declarations already exist. If a worker discovers that a required dependency is absent from the lock, stop at that boundary and report it rather than editing another workstream’s files.

---

### Task 1: A — Settle the compact passage contract and exact chunker

**Files:**
- Modify: `src/trec_ragtime/passages.py`
- Modify: `tests/test_passages.py`
- Modify: `tests/fixtures/passage_chunking.json`

**Interfaces:**
- Consumes: a structural pair-length callable with `fingerprint: str`, `max_length: int`, `max_passage_length: int`, `passage_length_function: Callable[[str], int]`, and `__call__(text: str) -> int`; production passes `CrossEncoderPairLengthFunction` from Task 2, while A tests use an offline fake.
- Produces: `Representation = Literal["canonical", "report_text"]`; `Boundary = Literal["sentence", "oversized_sentence"]`; frozen slotted `Passage` with exactly `document_id`, `representation`, `language`, `text`, `start_char`, `end_char`, `content_hash`, and `boundary`; `PassageChunkingError`; `sha256_bytes(value: str | bytes) -> bytes`; `validate_passages(selected_text: str, passages: Sequence[Passage]) -> None`; and `PassageChunker(pair_length, *, overlap_tokens=64)` with `fingerprint` and this exact method:

  ```python
  def chunk(
      self,
      *,
      document_id: str,
      canonical_text: str,
      selected_text: str,
      representation: Representation,
      language: str,
  ) -> tuple[Passage, ...]:
      raise NotImplementedError
  ```

- [ ] **Step 1: Replace the old lineage tests with a red compact-API test**

  In `tests/test_passages.py`, remove tests and helpers that construct `ChunkerConfig`, `LangChainPassageChunker`, `chunk_index`, `passage_id`, source/representation hashes, neighbor IDs, caller namespaces, and v1 zero-overlap restrictions. Add this contract test before changing production code:

  ```python
  class FakePairLength:
      fingerprint = "fake-pair-v1"
      max_length = 8
      max_passage_length = 6
      passage_length_function = staticmethod(lambda text: len(text.split()))

      def __call__(self, text: str) -> int:
          return len(text.split()) + 2


  def test_passage_contract_is_compact_and_immutable(self) -> None:
      fields = tuple(Passage.__dataclass_fields__)
      self.assertEqual(
          fields,
          (
              "document_id",
              "representation",
              "language",
              "text",
              "start_char",
              "end_char",
              "content_hash",
              "boundary",
          ),
      )
      passage = Passage(
          document_id="doc-1",
          representation="canonical",
          language="eng",
          text="One.",
          start_char=0,
          end_char=4,
          content_hash=sha256_bytes("One."),
          boundary="sentence",
      )
      with self.assertRaises((AttributeError, TypeError)):
          passage.text = "changed"  # type: ignore[misc]
  ```

- [ ] **Step 2: Run the compact passage tests to verify the expected red state**

  Run:

  ```bash
  sops decrypt --output .env config/secrets.sops.env
  git check-ignore .env
  uv run python -m unittest tests.test_passages -v
  ```

  Expected: the test run fails because the old `Passage` field set and `LangChainPassageChunker` interface do not match the compact contract. The command must not print `.env` contents.

- [ ] **Step 3: Implement the compact frozen `Passage` and one domain validation seam**

  In `src/trec_ragtime/passages.py`, delete `ChunkerConfig`, `_passage_id`, all source/representation hash and neighbor fields, overlap counters, compatibility properties, caller namespace composition, and recursive object validation. Implement the exact eight-field slotted dataclass shown below; validate nonblank IDs/language, the two representations, `report_text` language `eng`, `0 <= start_char < end_char`, `end_char - start_char == len(text)`, 32-byte hash equality, and the two boundary values at construction time. Use `PassageChunkingError` for both chunker and collection validation; delete the parallel `PassageValidationError` hierarchy:

  ```python
  @dataclass(frozen=True, slots=True)
  class Passage:
      document_id: str
      representation: Representation
      language: str
      text: str
      start_char: int
      end_char: int
      content_hash: bytes
      boundary: Boundary
  ```

  Keep `sha256_bytes` as the only hash helper. Make `validate_passages` validate the load-bearing collection seam only: a nonempty selected text; nonempty `Passage` sequence; each exact source slice and content hash; strictly increasing passage positions with no duplicate positional interval; and interval-union coverage from character zero through `len(selected_text)` with no gap. Allow overlap, but never treat overlap as additional source coverage. Do not add a custom sentence parser to the validator.

- [ ] **Step 4: Add fixture-driven red tests for exact slices, union coverage, and sentence-only overlap**

  Extend `tests/fixtures/passage_chunking.json` with literal English, Spanish, Russian, Chinese, English `report_text`, CRLF, leading/trailing whitespace, repeated-identical-sentence, paragraph, normal sentence-packing, overlap, and individually oversized-sentence cases. Store expected `(start_char, end_char)` intervals and expected boundary values, not generated IDs. Add tests that compare every returned text to `selected_text[start:end]`, every hash to `sha256_bytes(text)`, and the interval union to the full selected representation:

  ```python
  def assert_exact_passages(
      self,
      selected_text: str,
      passages: tuple[Passage, ...],
      expected_spans: list[tuple[int, int]],
  ) -> None:
      self.assertEqual(
          [(passage.start_char, passage.end_char) for passage in passages],
          expected_spans,
      )
      for passage in passages:
          self.assertEqual(
              selected_text[passage.start_char : passage.end_char],
              passage.text,
          )
          self.assertEqual(passage.content_hash, sha256_bytes(passage.text))
      validate_passages(selected_text, passages)
      covered: set[int] = set()
      for passage in passages:
          covered.update(range(passage.start_char, passage.end_char))
      self.assertEqual(covered, set(range(len(selected_text))))
  ```

  Add a sentence-only-overlap assertion that computes each adjacent intersection and requires it to equal a complete sentence from the literal source; also assert that no overlap begins or ends in the middle of a sentence. Add a repeated-text case proving equal content hashes do not collapse distinct `(start_char, end_char)` intervals.

- [ ] **Step 5: Implement `PassageChunker` with maintained splitters and a single exceptional RCTS path**

  Replace the paragraph-packing/configuration framework with `PassageChunker.__init__(pair_length, *, overlap_tokens=64)`. Derive one fingerprint from the pair-length fingerprint, overlap budget, installed `langchain-text-splitters` version, installed spaCy version, and the pinned language-to-pipeline mapping; do not include a caller namespace or interchangeable configuration object. Resolve `eng -> en_core_web_sm`, `spa -> es_core_news_sm`, `rus -> ru_core_news_sm`, and `zho -> zh_core_web_sm`; report unavailable pipelines through `PassageChunkingError` without including credentials or source text.

  Use the installed `SpacyTextSplitter` for normal selected-text chunking with exactly `chunk_size=pair_length.max_passage_length`, `chunk_overlap=overlap_tokens`, `length_function=pair_length.passage_length_function`, `separator=""`, `strip_whitespace=False`, and `add_start_index=True`. Validate every returned `Document` start index and exact source slice. Normal outputs must contain only complete adjacent sentence units. If a returned unit is individually over the exact full-pair ceiling, confirm through the same pinned spaCy sentence splitter that it is one sentence, pass only that sentence to `RecursiveCharacterTextSplitter` with `separators=[" ", ""]`, `chunk_overlap=0`, `strip_whitespace=False`, and `add_start_index=True`, then map relative offsets back to selected-text coordinates. Never send a multi-sentence unit to RCTS.

  Revalidate each final passage with `pair_length(passage.text) <= pair_length.max_length`, exact source slice, and interval-union coverage. If a LangChain/spaCy update stops providing authoritative offsets, use a small spaCy span adapter only for offset extraction; do not introduce a project-owned sentence parser or greedy packing framework. Return `boundary="sentence"` for normal chunks and `boundary="oversized_sentence"` for RCTS fragments.

- [ ] **Step 6: Run the chunker tests to green and remove stale implementation-only tests**

  Run:

  ```bash
  sops decrypt --output .env config/secrets.sops.env
  git check-ignore .env
  uv run python -m unittest tests.test_passages -v
  ```

  Expected: all compact contract, multilingual exact-slice, overlap, repeated-text, whitespace, pair-ceiling, and oversized-sentence tests pass. Delete tests that assert paragraph boundary names, `passage_id` stability, neighbor links, zero overlap, `ChunkerConfig` fingerprint composition, or private splitter helper calls. The remaining tests must verify behavior at the chunker seam.

- [ ] **Step 7: Commit the settled A passage seam before any later workstream starts**

  ```bash
  git add src/trec_ragtime/passages.py tests/test_passages.py tests/fixtures/passage_chunking.json
  git commit -m "refactor: simplify exact passage chunking"
  ```

  Later workers must read this commit and use the exact eight-field `Passage` plus `PassageChunker` constructor/method; they must not edit these files.

### Task 2: A — Align pair-length validation and local scoring to the compact passage API

**Files:**
- Modify: `src/trec_ragtime/passage_scoring.py`
- Modify: `tests/test_passage_scoring.py`

**Interfaces:**
- Consumes: Task 1 `Passage` and the structural pair-length seam; injected tokenizer/model loader fakes.
- Produces: one module exception `PassageScoringError`; `PassageScorer` with `fingerprint` and `score(query: str, passages: Sequence[Passage]) -> Mapping[bytes, float]`; `validate_passage_scores(passages, scores) -> dict[bytes, float]`; immutable `LocalCrossEncoderConfig`; `CrossEncoderPairLengthFunction` with `max_length`, `max_passage_length`, and `passage_length_function`; and lazy `LocalCrossEncoderScorer`.

- [ ] **Step 1: Rewrite scorer helpers/tests around hash/text identity and verify red**

  In `tests/test_passage_scoring.py`, replace the old `_passage` helper with named construction of the eight-field `Passage`, delete tests for removed passage lineage and `PassageScoreValidationError`, and keep tests for finite exact hash coverage, duplicate-content deduplication, tokenizer-only lazy loading, exact pair formatting, special-token overhead, runtime/model/tokenizer fingerprint changes, device/precision normalization, bounded batches, raw logits, and failure-before-predict. Add this representative failing contract test:

  ```python
  def test_score_deduplicates_equal_content_and_returns_finite_float_map(self) -> None:
      first = _passage("same", start_char=0, end_char=4)
      repeated = _passage("same", start_char=5, end_char=9)
      other = _passage("other", start_char=10, end_char=15)
      model = RecordingModel({"same": 2.5, "other": -1})
      scorer = LocalCrossEncoderScorer(
          _config(batch_size=2),
          loader=lambda _: (model, RecordingTokenizer()),
      )

      scores = scorer.score("exact query", (first, repeated, other))

      self.assertEqual(scores, {first.content_hash: 2.5, other.content_hash: -1.0})
      self.assertEqual(len(model.predict_calls), 1)
      self.assertTrue(all(type(value) is float for value in scores.values()))
  ```

  Run:

  ```bash
  sops decrypt --output .env config/secrets.sops.env
  git check-ignore .env
  uv run python -m unittest tests.test_passage_scoring -v
  ```

  Expected: old imports/fields fail, giving a red state before implementation.

- [ ] **Step 2: Implement the compact validator and scorer protocol**

  Delete `_passage_lineage` and every check for removed `passage_id`, representation hash, chunk index, and neighbor fields. Keep one ordered `dict[bytes, str]` of first-seen content hashes and texts; reject a contradictory hash/text pair, malformed/extra/missing keys, Boolean/non-real/non-finite values, and any result whose key set differs from the unique requested hashes. Normalize values to ordinary `float` and return keys in binary sorted order. The protocol must expose only the immutable fingerprint and synchronous `score` method.

- [ ] **Step 3: Preserve the exact pair-length/tokenizer seam**

  Keep `CrossEncoderPairLengthFunction` lazy and tokenizer-only. It must format the exact query, count the query’s no-special-token length, call `num_special_tokens_to_add(pair=True)`, expose `max_length`, expose `max_passage_length`, and expose a passage-only `length_function` used only as a packing estimate. Its `__call__` must tokenize the complete formatted query/passage pair with `truncation=False` and reject overflow/truncation reported by the tokenizer. Include the pinned scorer model, tokenizer, runtime versions, device, precision, batch size, max length, query format, and query policy in the stable scorer fingerprint.

- [ ] **Step 4: Preserve the lazy local scorer and exact pre-predict validation**

  Keep the default model `BAAI/bge-reranker-v2-m3` at revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`; load the explicit tokenizer/model pair once under a lock; validate every unique formatted pair with the exact max-length tokenizer call before any `predict`; use identity activation and `apply_softmax=False` for raw logits; batch at `LocalCrossEncoderConfig.batch_size`; and return the validated hash map. Keep `rocm` normalized to the effective `cuda` device. All tests use injected fakes and must not download weights.

- [ ] **Step 5: Run the scorer and upstream passage tests to green**

  ```bash
  sops decrypt --output .env config/secrets.sops.env
  git check-ignore .env
  uv run python -m unittest tests.test_passage_scoring tests.test_passages -v
  ```

  Expected: both modules pass, including exact pair-token validation, contiguous-CJK budget behavior, one-time model loading, raw score preservation, and no scorer-stack eager import from `passages`.

- [ ] **Step 6: Commit the scorer seam**

  ```bash
  git add src/trec_ragtime/passage_scoring.py tests/test_passage_scoring.py
  git commit -m "refactor: simplify local passage scoring"
  ```

### Task 3: A — Replace the large score-envelope framework with a validated complete map cache

**Files:**
- Modify: `src/trec_ragtime/passage_score_cache.py`
- Modify: `tests/test_passage_score_cache.py`

**Interfaces:**
- Consumes: Task 1 `Passage` content hashes and chunker fingerprint; Task 2 `PassageScorer` and `validate_passage_scores`; a structural future `ArtifactStoreLike` with `read`, `read_validated`, `write`, and `evict_local`.
- Produces: `PassageScoreCache(store, scorer, *, chunker_fingerprint, representation, artifact_validation_error)`, `CachedPassageScorer(cache)`, `score_object_key`, `query_hash`, and strict map encode/decode helpers. `PassageScoreCache.score(query, passages) -> dict[bytes, float]` returns only requested hashes while the stored artifact remains the complete validated union for that query.

- [ ] **Step 1: Replace envelope tests with a red complete-map cache test**

  In `tests/test_passage_score_cache.py`, retain a test-local fake with the complete structural seam and no production implementation:

  ```python
  class FakeArtifactStore:
      class ArtifactValidationError(Exception):
          pass

      def __init__(self) -> None:
          self.objects: dict[str, bytes] = {}
          self.read_calls: list[str] = []
          self.read_validated_calls: list[str] = []
          self.write_calls: list[tuple[str, bytes]] = []

      def read(self, key: str) -> bytes | None:
          self.read_calls.append(key)
          return self.objects.get(key)

      def read_validated(self, key: str, decoder: Callable[[bytes], object]) -> object | None:
          self.read_validated_calls.append(key)
          payload = self.objects.get(key)
          return None if payload is None else decoder(payload)

      def write(self, key: str, payload: bytes) -> None:
          self.write_calls.append((key, payload))
          self.objects[key] = payload

      def evict_local(self, key: str) -> None:
          self.objects.pop(key, None)
  ```

  Add tests for stable exact-query keys, scorer/chunker/representation isolation, cold miss, complete warm hit, missing-only scoring, complete-union rewrite, duplicate content, malformed/truncated/non-map/non-finite/wrong-key artifacts, storage failure propagation, and no write after scorer failure or partial coverage. Delete `PassageScoreEnvelope`, coverage-digest, schema-envelope, fingerprint-only constructor, and duplicate exception-factory compatibility tests. Run:

  ```bash
  sops decrypt --output .env config/secrets.sops.env
  git check-ignore .env
  uv run python -m unittest tests.test_passage_score_cache -v
  ```

  Expected: imports for the removed envelope API and old constructor fail, establishing the deletion-first red state.

- [ ] **Step 2: Implement strict map serialization and identity**

  Keep `OBJECT_ROOT = "passage-scores/v1"` and derive the object key from the effective scorer fingerprint, the stable chunker fingerprint, the selected representation, and the exact UTF-8 query hash. Encode only a deterministic `dict[bytes, float]` map with `msgspec`; decode only bytes into a mapping whose keys are exactly 32-byte hashes, whose values are exact finite floats after normalization, and whose coverage is nonempty or explicitly empty only for the no-request path. Do not include the bearer token or query text in keys, payloads, exceptions, or logs.

- [ ] **Step 3: Implement cache-aside merge semantics over the future seam**

  Reference `ArtifactStoreLike` under `TYPE_CHECKING` only; do not add a concrete store or call `read`/`evict_local` from this adapter. On `read_validated`, decode and validate the complete map. Determine missing requested hashes by first-seen unique content hash, call the configured scorer exactly once for those hashes, validate the exact response with `validate_passage_scores`, merge into the prior map, revalidate the complete union, and write once. Return a new dictionary containing only the requested unique hashes. Any storage, decode, scorer, or partial-coverage failure propagates and never becomes a cache miss or successful write.

- [ ] **Step 4: Preserve the cache wrapper seam without duplicated identity logic**

  Make `CachedPassageScorer` delegate to one `PassageScoreCache`, expose the underlying scorer fingerprint, and satisfy `PassageScorer`. Remove the current optional scorer/fingerprint overloads, `_CacheIdentity` dataclass, repeated request validation, and compatibility aliases that only supported the old envelope publication framework. Keep the injected artifact-store validation exception seam so a future #12 store can repair local cache entries without importing an implementation from this branch.

- [ ] **Step 5: Run cache, scorer, and passage tests to green**

  ```bash
  sops decrypt --output .env config/secrets.sops.env
  git check-ignore .env
  uv run python -m unittest tests.test_passage_score_cache tests.test_passage_scoring tests.test_passages -v
  ```

  Expected: all map key/bytes stability, warm/cold/overlap reuse, finite exact coverage, error propagation, and local scorer tests pass with no envelope symbols remaining in the focused test imports.

- [ ] **Step 6: Commit the complete-map cache**

  ```bash
  git add src/trec_ragtime/passage_score_cache.py tests/test_passage_score_cache.py
  git commit -m "refactor: simplify complete passage score cache"
  ```

### Task 4: A — Update the multilingual fixture benchmark and finish A verification

**Files:**
- Modify: `scripts/benchmark_passage_chunking.py`
- Modify: `tests/fixtures/passage_chunking.json`
- Modify: `tests/test_passages.py`

**Interfaces:**
- Consumes: the compact `PassageChunker`, exact `Passage`, `validate_passages`, and the fake pair-length seam from Tasks 1–3.
- Produces: a local-only benchmark that proves exact offsets/hashes, complete selected-text coverage, sentence-only overlap, RCTS-only oversized-sentence handling, deterministic records, and a pinned-tokenizer-only contiguous-CJK smoke.

- [ ] **Step 1: Add a red benchmark assertion for all required invariants**

  Replace `record_passages` fields for removed IDs/hashes/neighbors with the compact fields. Add a benchmark assertion shaped like this to make the required output explicit:

  ```python
  def assert_benchmark_invariants(
      selected_text: str,
      passages: tuple[Passage, ...],
      pair_length: Callable[[str], int],
      max_length: int,
  ) -> None:
      validate_passages(selected_text, passages)
      covered = set()
      for passage in passages:
          assert selected_text[passage.start_char : passage.end_char] == passage.text
          assert passage.content_hash == sha256_bytes(passage.text)
          assert pair_length(passage.text) <= max_length
          covered.update(range(passage.start_char, passage.end_char))
      assert covered == set(range(len(selected_text)))
  ```

  Add a sentence-overlap fixture whose records are run twice and compared byte-for-byte. Add a contiguous `"中" * 502` case using only the pinned-tokenizer fake/length seam, assert no model loader is called, and assert every resulting pair is within the exact ceiling. The benchmark must not call the organizer, Hugging Face, Runpod, or model-weight loader.

- [ ] **Step 2: Implement the benchmark and run it to green**

  Resolve the installed package and four spaCy pipeline versions, run the fixture cases twice through `PassageChunker`, compare canonical sorted JSON bytes, and print `full_source_coverage=true`, `sentence_only_overlap=true`, `oversized_sentence_rcts_only=true`, `pair_ceiling=true`, `deterministic_records=true`, fixture/pass counts, and throughput. Run:

  ```bash
  sops decrypt --output .env config/secrets.sops.env
  git check-ignore .env
  uv run python scripts/benchmark_passage_chunking.py
  ```

  Expected: exit code 0, zero failures, two identical record runs, and a contiguous-CJK result that fits the exact full-pair budget without model inference.

- [ ] **Step 3: Run every A-focused test and commit the completed dependency seam**

  ```bash
  sops decrypt --output .env config/secrets.sops.env
  git check-ignore .env
  uv run python -m unittest tests.test_passages tests.test_passage_scoring tests.test_passage_score_cache -v
  git diff --check
  git status --short
  ```

  Expected: all A tests pass; `git diff --check` is clean; only A-owned files plus the plan are changed. Commit:

  ```bash
  git add src/trec_ragtime/passages.py src/trec_ragtime/passage_scoring.py src/trec_ragtime/passage_score_cache.py tests/test_passages.py tests/test_passage_scoring.py tests/test_passage_score_cache.py tests/fixtures/passage_chunking.json scripts/benchmark_passage_chunking.py
  git commit -m "test: verify simplified passage scoring seam"
  ```

  This is the dependency checkpoint. B and C may now proceed independently, but they must not modify any A-owned file.

### Task 5: B — Replace reranking’s recursive graph and artifact publisher with one compact function

**Files:**
- Modify: `src/trec_ragtime/reranking.py`
- Modify: `tests/test_reranking.py`

**Interfaces:**
- Consumes: `RetrievalResult`; Task 1 `PassageChunker`; Task 2/3 `PassageScorer` or `CachedPassageScorer`.
- Produces exactly:
  - `RerankingError`, the single module exception used for retrieval/selection/result normalization failures;
  - `PassageEvidence(document_id: str, start_char: int, end_char: int, score: float, content_hash: bytes)` as a frozen slotted dataclass;
  - `RerankedDocument(document: RetrievedDocument, score: float | None, evidence: tuple[PassageEvidence, ...], reranked_rank: int, status: Literal["scored", "missing_representation", "unresolved", "no_passages"])` as a frozen slotted dataclass;
  - `RerankingResult(query: str, representation: Literal["canonical", "report_text"], documents: tuple[RerankedDocument, ...])` as a frozen slotted dataclass;
  - `rerank_retrieval(retrieval: RetrievalResult, *, chunker: PassageChunker, scorer: PassageScorer, representation: Literal["canonical", "report_text"]) -> RerankingResult`; and
  - `max_passage_v1 = "max_passage_v1"` as the internal aggregation label used by the maximum-passage implementation.

- [ ] **Step 1: Replace artifact/publication tests with red interface tests**

  In `tests/test_reranking.py`, delete `write_reranking_artifact`, JSONL/manifest, fsync, platform rename, `topic_id`, source-ref, and recursive forged-result tests. Replace the passage helper so it constructs only A’s eight fields. Keep fake retrieval construction from `SearchHit`, `RetrievedDocument`, `PipelineResult`, and `RetrievalResult`. Add this direct interface test:

  ```python
  def test_rerank_returns_compact_deterministic_evidence(self) -> None:
      retrieval = _retrieval(
          (
              _document("doc-b", rank=2, score=0.4, text="same. same."),
              _document("doc-a", rank=1, score=0.5, text="winner."),
          )
      )
      chunker = RecordingChunker(
          {
              "doc-b": (("same.", 0, 5), ("same.", 6, 11), ("same.", 6, 11)),
              "doc-a": (("winner.", 0, 7),),
          }
      )
      scorer = RecordingScorer({"same.": 1.0, "winner.": 2.0})

      result = rerank_retrieval(
          retrieval,
          chunker=chunker,
          scorer=scorer,
          representation="canonical",
      )

      self.assertEqual([row.document.document_id for row in result.documents], ["doc-a", "doc-b"])
      self.assertEqual(result.documents[0].score, 2.0)
      self.assertEqual(
          [(item.start_char, item.end_char) for item in result.documents[1].evidence],
          [(0, 5), (6, 11)],
      )
      self.assertEqual(len(scorer.calls), 1)
      self.assertEqual(len(scorer.calls[0][1]), 2)
  ```

  The `RecordingChunker` must return real `Passage` instances with the supplied offsets; the tuple syntax above is the helper input, not a production interface. Run:

  ```bash
  sops decrypt --output .env config/secrets.sops.env
  git check-ignore .env
  uv run python -m unittest tests.test_reranking -v
  ```

  Expected: imports for removed artifact symbols and the old `topic_id` parameter fail.

- [ ] **Step 2: Implement construction-time normalization and compact result records**

  Delete the 1,000-line recursive `__post_init__` graph validators, convenience-field duplication, passage lineage recreation, compatibility aliases, JSON serializers, and all filesystem publication helpers. Keep frozen/slotted result records with no recursive revalidation. Normalize and validate once inside `rerank_retrieval`: check the retrieval’s hit/document alignment; select `document.text` for `canonical` or `document.report_text` with language `eng` for `report_text`; retain unresolved and missing-representation rows; call `chunker.chunk` only for valid selected text; and create source-offset evidence from the returned passages.

- [ ] **Step 3: Implement hash deduplication, positional evidence deduplication, and deterministic ranking**

  Gather one first-seen `dict[bytes, Passage]` across all scorable documents and call `scorer.score` once. Reattach each content-hash score to every positional passage. Within each document, deduplicate only exact `(document_id, start_char, end_char)` evidence identities, preserve identical content at distinct offsets, sort evidence by `(start_char, end_char, content_hash)`, and aggregate `max(score)`. Choose a winning passage by `(start_char, end_char, content_hash)` among max-score passages. Sort scored documents by `(-score, document.rank, document.document_id)` and append unscored status rows by `(document.rank, document.document_id)`; assign `reranked_rank` after sorting.

- [ ] **Step 4: Add focused tests for all retained behavior and remove shallow tests**

  Retain tests for canonical/report-text selection, unresolved/missing representation retention, exact source slices and hashes, maximum aggregation, scorer-call deduplication, repeated identical sentences at distinct offsets, exact-location overlap deduplication, warm cache delegation, stable score/document ties, frozen/slotted records, and multilingual report language. Remove tests for private helpers, recursive result-model forgeries, duplicated convenience fields, topic/source-ref metadata, and file publication internals. Add an explicit tie test:

  ```python
  def test_document_ties_use_organizer_rank_then_document_id(self) -> None:
      retrieval = _retrieval(
          (
              _document("doc-z", rank=2, score=0.5, text="same"),
              _document("doc-a", rank=1, score=0.5, text="same"),
          )
      )
      result = rerank_retrieval(
          retrieval,
          chunker=RecordingChunker(),
          scorer=RecordingScorer({"same": 3.0}),
          representation="canonical",
      )
      self.assertEqual(
          [row.document.document_id for row in result.documents],
          ["doc-a", "doc-z"],
      )
  ```

- [ ] **Step 5: Run B with the completed A seam and commit it**

  ```bash
  sops decrypt --output .env config/secrets.sops.env
  git check-ignore .env
  uv run python -m unittest tests.test_reranking tests.test_passages tests.test_passage_scoring tests.test_passage_score_cache -v
  git diff --check
  ```

  Expected: B and A focused tests pass, no writer/publication test remains, and no A file is modified by B. Commit:

  ```bash
  git add src/trec_ragtime/reranking.py tests/test_reranking.py
  git commit -m "refactor: simplify deterministic passage reranking"
  ```

### Task 6: C — Reduce Flash to a thin adapter and one static self-contained worker

**Files:**
- Modify: `src/trec_ragtime/flash_passage_scorer.py`
- Modify: `src/trec_ragtime/flash_worker.py`
- Modify: `tests/test_flash_passage_scorer.py`

**Interfaces:**
- Consumes: Task 2 `LocalCrossEncoderConfig`, `PassageScorer`, `PassageScoringError`, and compact `Passage`; an injected async runner or endpoint object.
- Produces: strict request/response codecs carrying `schema_version`, `scorer_fingerprint`, `model_config`, `query`, and hash/text rows; `RunpodFlashConfig`; `RunpodFlashPassageScorer` implementing `PassageScorer`; `FlashEndpointBatchRunner`; static `FlashPassageWorker`; and `build_flash_worker(config, *, endpoint_factory=None, gpu_resolver=None) -> type[FlashPassageWorker]`.

- [ ] **Step 1: Replace generated-worker tests with red static-worker and adapter tests**

  Delete tests for generated source digests, module/class caches, cache growth caps, normalized-policy class identity, dynamic source rendering, and duplicated worker config objects. Keep tests for wire hash/hex conversion, exact payload byte limits, bounded count/concurrency/timeouts, complete per-batch/global coverage, endpoint `run()` then `job.wait()`, cancellation, error sanitization, and local/Flash fake parity. Add this static class-only test shape:

  ```python
  def test_static_worker_source_executes_without_project_imports(self) -> None:
      source = inspect.getsource(FlashPassageWorker)
      namespace = {"__name__": "flash_worker_isolated"}
      with patch.dict(sys.modules, {"trec_ragtime": None}, clear=False):
          exec(compile(source, "<flash-worker>", "exec"), namespace)
      isolated_worker = namespace["FlashPassageWorker"]
      model = FakeModel()
      tokenizer = BatchEncodingTokenizer()
      worker = isolated_worker(loader=lambda config: (model, tokenizer))
      response = worker.score(_request_for_config(_worker_config()))
      self.assertEqual(response["scorer_fingerprint"], _worker_config().fingerprint)
      self.assertEqual(model.load_calls, 1)
  ```

  The extracted class body must not resolve a project module; the fake tokenizer must be a mapping-compatible `BatchEncoding` shape with `input_ids`, `length`, and no overflow/truncation. Run:

  ```bash
  sops decrypt --output .env config/secrets.sops.env
  git check-ignore .env
  uv run python -m unittest tests.test_flash_passage_scorer -v
  ```

  Expected: old `build_flash_worker` generated-source assumptions fail, while the retained transport tests identify the behavior to preserve.

- [ ] **Step 2: Implement the thin wire/endpoint adapter**

  In `flash_passage_scorer.py`, keep only one Flash domain exception (`FlashScoringError`), strict JSON codecs, config normalization, bounded batch partitioning, injected endpoint invocation, complete response validation, and the minimal private event-loop bridge required by synchronous `score`. Request rows must convert `bytes` hashes to lowercase hex and validate `sha256(text)`; responses must convert hex back to bytes and validate finite scores. Measure final UTF-8 encoded request bytes before dispatch, keep every payload strictly below both the configured limit and 10,000,000 bytes, bound concurrent waves, cancel/await sibling tasks after failure, and require exact global coverage before returning a validated `dict[bytes, float]`.

  Add `model_config` to the wire request as a strict primitive mapping covering every result-affecting scorer setting. The adapter must send the effective model fingerprint with every request and reject response fingerprint drift. Transport-only endpoint ID, worker name, GPU, worker count, idle timeout, request timeout, and payload/concurrency settings must not change the model/scorer fingerprint.

- [ ] **Step 3: Implement the static `FlashPassageWorker` without generated source**

  Define one ordinary top-level `FlashPassageWorker` class. Its body must contain all helpers required by the class-only transport and import only the standard library plus declared pinned third-party dependencies from inside methods. The constructor accepts an injected `loader` for local tests and does not load anything. Its `score(request: Mapping[str, object]) -> dict[str, object]` validates exact request fields and model configuration, computes the fingerprint from the received configuration, and on the first valid request loads the model and explicit tokenizer once. Later requests must use the same fingerprint or raise `FlashScoringError`/the worker’s injected error type.

  Accept a Transformers `BatchEncoding` mapping, validate every full formatted query/passage pair with `truncation=False`, `max_length=config.max_length`, and no overflow/truncation, call raw-logit `predict` in bounded batches, and return all requested finite scores keyed by lowercase hash hex. Keep model loading and prediction errors sanitized. Do not import `trec_ragtime`, read credentials, write artifacts, or log query/passage content.

- [ ] **Step 4: Keep decoration lazy and pinned without provisioning on construction**

  Make `build_flash_worker` return or decorate the static class directly. Resolve `runpod_flash`, GPU enum/type, and pinned dependency strings only when an `endpoint_factory` is explicitly supplied; preserve the exact decorator kwargs for worker name, resolved GPU, worker count, idle timeout, and pinned third-party versions. Constructing `build_flash_worker(config)` locally must not resolve GPU, create an endpoint, or provision anything. Remove `_WORKER_CLASS_CACHE`, `_WORKER_MODULE_CACHE`, `linecache`, generated source, source digest, and per-policy generated-class mutation.

- [ ] **Step 5: Run C focused tests and commit without touching A or B**

  ```bash
  sops decrypt --output .env config/secrets.sops.env
  git check-ignore .env
  uv run python -m unittest tests.test_flash_passage_scorer tests.test_passage_scoring tests.test_passages -v
  git diff --check
  ```

  Expected: static class extraction, fake local execution, `BatchEncoding` acceptance, one-time load, exact pair ceiling, wire coverage, batching, cancellation, endpoint lifecycle, and local/Flash parity all pass. Commit:

  ```bash
  git add src/trec_ragtime/flash_passage_scorer.py src/trec_ragtime/flash_worker.py tests/test_flash_passage_scorer.py
  git commit -m "refactor: simplify Flash passage scoring"
  ```

### Task 7: D — Reconcile public exports, integration coverage, and local-first documentation

**Files:**
- Modify: `src/trec_ragtime/__init__.py`
- Modify: `README.md`
- Modify: `tests/test_passage_reranking_integration.py`

**Interfaces:**
- Consumes: the final A/B/C symbols exactly as defined above and existing `RetrievalResult`/`RetrievedDocument` models.
- Produces: a unique `__all__` containing only supported compact chunking, scoring, cache, reranking, and optional Flash symbols; a public local-first example; and an offline integration test covering report text, canonical multilingual evidence, cache warm reuse, CJK pair budgeting, and Flash fake parity.

- [ ] **Step 1: Add the red public-surface test before changing exports/docs**

  Replace imports of `ChunkerConfig`, `LangChainPassageChunker`, `RerankedPassage`, `write_reranking_artifact`, and envelope-only cache symbols. Add this assertion:

  ```python
  def test_public_surface_has_no_deleted_compatibility_symbols(self) -> None:
      expected = {
          "Passage",
          "PassageChunker",
          "CrossEncoderPairLengthFunction",
          "LocalCrossEncoderConfig",
          "LocalCrossEncoderScorer",
          "PassageScoreCache",
          "CachedPassageScorer",
          "PassageEvidence",
          "RerankedDocument",
          "RerankingResult",
          "rerank_retrieval",
          "RunpodFlashPassageScorer",
          "FlashPassageWorker",
      }
      self.assertEqual(len(trec_ragtime.__all__), len(set(trec_ragtime.__all__)))
      self.assertTrue(expected <= set(trec_ragtime.__all__))
      self.assertFalse(hasattr(trec_ragtime, "write_reranking_artifact"))
      self.assertFalse(hasattr(trec_ragtime, "ChunkerConfig"))
  ```

  Add a red end-to-end call with the exact new signature:

  ```python
  result = rerank_retrieval(
      retrieval,
      chunker=PassageChunker(pair_length),
      scorer=CachedPassageScorer(cache),
      representation="report_text",
  )
  self.assertEqual(result.query, retrieval.query)
  ```

  Run:

  ```bash
  sops decrypt --output .env config/secrets.sops.env
  git check-ignore .env
  uv run python -m unittest tests.test_passage_reranking_integration -v
  ```

  Expected: old exports and the old `topic_id`/chunker construction fail.

- [ ] **Step 2: Update `__init__.py` to export only the settled interfaces**

  Import and list the compact A symbols, B result/function symbols, and C adapter/static-worker symbols once. Remove deleted `ChunkerConfig`, `LangChainPassageChunker`, `PassageValidationError`, `RerankedPassage`, all JSONL writer/schema constants, and envelope-only cache symbols. Keep `PassageChunkingError`, `sha256_bytes`, and `validate_passages` as the supported exact-lineage utilities used by the final integration test and benchmark. Confirm optional `runpod_flash` remains lazy at package import.

- [ ] **Step 3: Rewrite the integration fixture as one offline decision test**

  Keep a test-local `ArtifactStoreLike` fake implementing `read`, `read_validated`, `write`, and `evict_local`; a tokenizer fake that records exact single/pair calls; a raw-logit model fake; and an in-memory Flash wire runner. Use English, Spanish, Russian, Chinese, missing-report-translation, and unresolved rows. Assert the following behavior in one representative test:

  ```python
  first = rerank_retrieval(
      retrieval,
      chunker=chunker,
      scorer=CachedPassageScorer(cache),
      representation="report_text",
  )
  model_calls = model.predict_calls
  second = rerank_retrieval(
      retrieval,
      chunker=chunker,
      scorer=CachedPassageScorer(cache),
      representation="report_text",
  )
  self.assertEqual(first, second)
  self.assertEqual(model.predict_calls, model_calls)
  self.assertEqual(first.documents[-1].status, "unresolved")
  self.assertIn("missing-translation", {row.document.document_id for row in first.documents})
  self.assertTrue(
      all(
          row.status == "scored" or not row.evidence
          for row in first.documents
      )
  )
  ```

  Add a canonical run and Flash fake run with the same deterministic model and compare document IDs, statuses, maximum scores, winning hashes, and evidence offsets. Keep the contiguous-CJK smoke and assert all full-pair token counts stay at or below the configured model ceiling. No test may call a live organizer, HF Dataset Viewer, Runpod endpoint, or model-weight loader.

- [ ] **Step 4: Rewrite README usage and constraints**

  Replace the old `ChunkerConfig`/`LangChainPassageChunker`/`topic_id`/artifact-writer examples with the exact local-first flow: create `LocalCrossEncoderConfig`, create the tokenizer-only `CrossEncoderPairLengthFunction` for the retrieval query, construct `PassageChunker(pair_length, overlap_tokens=64)`, construct `LocalCrossEncoderScorer`, optionally wrap it with `PassageScoreCache` and `CachedPassageScorer`, and call `rerank_retrieval` with only `retrieval`, `chunker`, `scorer`, and explicit `representation`. Document `report_text` versus `canonical`, exact offsets/hashes/full coverage, sentence-only overlap, oversized-sentence RCTS, max-passage aggregation, positional evidence deduplication, complete dict cache semantics, and the future `ArtifactStoreLike` seam.

  Keep the pinned model/tokenizer revision, CPU/CUDA plus `rocm` alias, lazy local model loading, optional Flash config/runner/worker definition, bounded payload/concurrency/timeouts, pinned worker dependencies, `BatchEncoding` support, and the statement that building a worker definition does not deploy. Remove claims about deleted lineage IDs, zero overlap, JSONL persistence, dynamic class caches, or a live Flash smoke. Keep the decrypt-first focused/full-test and benchmark commands, and state that no paid/deploy action is authorized by this refactor.

- [ ] **Step 5: Run D integration and all focused tests to green**

  ```bash
  sops decrypt --output .env config/secrets.sops.env
  git check-ignore .env
  uv run python -m unittest tests.test_passage_reranking_integration tests.test_reranking tests.test_flash_passage_scorer tests.test_passage_score_cache tests.test_passage_scoring tests.test_passages -v
  ```

  Expected: the complete offline reranking workflow passes, second-run cache reuse performs no model prediction/write, missing translation and unresolved rows remain explicit, and local/Flash fake outputs match.

- [ ] **Step 6: Commit the public/documentation checkpoint**

  ```bash
  git add src/trec_ragtime/__init__.py README.md tests/test_passage_reranking_integration.py
  git commit -m "docs: integrate simplified passage reranking API"
  ```

### Task 8: Final deletion, verification, and handoff checks

**Files:**
- Read: all changed files and `94bd6a4` versions of the four target production modules
- Modify: only an already-owned file if a verification failure reveals a concrete defect; do not expand scope

**Interfaces:**
- Consumes: completed A/B/C/D checkpoints and the approved design spec.
- Produces: verified evidence of behavior preservation, material net deletion, clean ownership, and no external side effects.

- [ ] **Step 1: Run the exact production deletion gate against `94bd6a4`**

  Count nonblank source lines in exactly `passages.py`, `reranking.py`, `flash_passage_scorer.py`, and `flash_worker.py` at the baseline and working tree. At `94bd6a4`, the descriptive physical/nonblank counts are `835/753`, `1,451/1,323`, `1,027/916`, and `747/675`, respectively: 4,060 physical lines but 3,667 nonblank lines. The physical total is not the gate baseline. Preserve the script's `awk 'NF'` nonblank methodology and accept only when `current * 100 <= 3667 * 70`, equivalent to at most 2,566 current nonblank lines and therefore at least 30% net deletion. Run after decrypting `.env` because this is the final verification session:

  ```bash
  sops decrypt --output .env config/secrets.sops.env
  git check-ignore .env
  files=(
    src/trec_ragtime/passages.py
    src/trec_ragtime/reranking.py
    src/trec_ragtime/flash_passage_scorer.py
    src/trec_ragtime/flash_worker.py
  )
  baseline=0
  current=0
  for file in "${files[@]}"; do
      before=$(git show "94bd6a4:${file}" | awk 'NF { count += 1 } END { print count + 0 }')
      after=$(awk 'NF { count += 1 } END { print count + 0 }' "${file}")
      printf '%s baseline=%s current=%s\n' "${file}" "${before}" "${after}"
      baseline=$((baseline + before))
      current=$((current + after))
  done
  printf 'production_nonblank_baseline=%s production_nonblank_current=%s\n' "${baseline}" "${current}"
  test "${baseline}" -eq 3667
  test $((current * 100)) -le $((3667 * 70))
  ```

  Expected: the command exits 0, confirms the 3,667-nonblank-line baseline, and prints a current production size of no more than 2,566 nonblank lines. If it fails, delete shallow compatibility/publication/generated-worker code or remove internal-only tests; do not simply move it to another file.

- [ ] **Step 2: Verify focused behavior and the complete repository**

  ```bash
  sops decrypt --output .env config/secrets.sops.env
  git check-ignore .env
  uv run python -m unittest tests.test_passages tests.test_passage_scoring tests.test_passage_score_cache tests.test_reranking tests.test_flash_passage_scorer tests.test_passage_reranking_integration -v
  uv run python scripts/benchmark_passage_chunking.py
  uv run python -m unittest discover -s tests -v
  ```

  Expected: focused tests, the multilingual benchmark, and complete unittest discovery all exit 0. The benchmark must report exact slice/hash/coverage, sentence-only-overlap, RCTS-only oversized-sentence, exact pair ceiling, and deterministic-record success.

- [ ] **Step 3: Run compile, import, lock, static, and formatting checks**

  ```bash
  sops decrypt --output .env config/secrets.sops.env
  git check-ignore .env
  uv run python -m compileall -q src tests scripts
  uv run python -c 'import trec_ragtime; import trec_ragtime.passages; import trec_ragtime.passage_scoring; import trec_ragtime.passage_score_cache; import trec_ragtime.reranking; import trec_ragtime.flash_passage_scorer; import trec_ragtime.flash_worker; print("imports=ok")'
  uv run --with ruff ruff check src tests scripts
  uv run --with ruff ruff format --check src tests scripts
  uv lock --check
  git diff --check
  ```

  Expected: `imports=ok`, no compile errors, Ruff check/format success, lock consistency, and no whitespace errors. Importing the package must not import or provision `runpod_flash`, model weights, or a live endpoint.

- [ ] **Step 4: Check ownership, secrets, ignored environment, and net test deletion**

  ```bash
  allowed='^(docs/superpowers/plans/2026-08-10-passage-reranking-simplification\.md|src/trec_ragtime/(passages|passage_scoring|passage_score_cache|reranking|flash_passage_scorer|flash_worker|__init__)\.py|tests/(test_passages|test_passage_scoring|test_passage_score_cache|test_reranking|test_flash_passage_scorer|test_passage_reranking_integration)\.py|tests/fixtures/passage_chunking\.json|scripts/benchmark_passage_chunking\.py|README\.md)$'
  git diff --name-only 94bd6a4 -- | tee /tmp/issue-15-files.txt
  ! grep -Ev "${allowed}" /tmp/issue-15-files.txt
  git check-ignore .env
  if git ls-files --error-unmatch .env >/dev/null 2>&1; then exit 1; fi
  git grep -n -I -E 'RAGTIME_TOKEN|HF_TOKEN|Authorization:[[:space:]]*Bearer|BEGIN[[:space:]].*PRIVATE[[:space:]]KEY' -- . ':(exclude)config/secrets.sops.env'
  ```

  Expected: the allowed-file grep emits no out-of-scope path; `git check-ignore .env` prints `.env`; tracked-file lookup fails as intended; secret grep exits nonzero because no matching secret is present. Do not include `/tmp/issue-15-files.txt` in the repository.

  Compare focused test size against `94bd6a4` and record net deletion:

  ```bash
  test_before=$(git show 94bd6a4:tests/test_passages.py | wc -l)
  test_before=$((test_before + $(git show 94bd6a4:tests/test_passage_scoring.py | wc -l)))
  test_before=$((test_before + $(git show 94bd6a4:tests/test_passage_score_cache.py | wc -l)))
  test_before=$((test_before + $(git show 94bd6a4:tests/test_reranking.py | wc -l)))
  test_before=$((test_before + $(git show 94bd6a4:tests/test_flash_passage_scorer.py | wc -l)))
  test_after=$(wc -l < tests/test_passages.py)
  test_after=$((test_after + $(wc -l < tests/test_passage_scoring.py)))
  test_after=$((test_after + $(wc -l < tests/test_passage_score_cache.py)))
  test_after=$((test_after + $(wc -l < tests/test_reranking.py)))
  test_after=$((test_after + $(wc -l < tests/test_flash_passage_scorer.py)))
  printf 'focused_test_lines_before=%s focused_test_lines_after=%s\n' "${test_before}" "${test_after}"
  test "${test_after}" -lt "${test_before}"
  ```

  Expected: focused test code is also materially smaller, not just moved, while the complete suite remains green.

- [ ] **Step 5: Confirm no external side effects and hand off without commit/push/PR**

  Inspect `git status --short`, `git diff --stat 94bd6a4`, and the final diff. Confirm no model weights, plaintext `.env`, API responses, benchmark output files, logs, generated worker modules, deployment artifacts, or unrelated changes are tracked. Do not run `flash dev`, `flash deploy`, a paid endpoint call, an organizer/HF write, `git push`, or a pull-request command. The implementer reports the final production/test deletion counts, focused/full test results, benchmark result, and any concrete deferred external gates.

## Completion Checklist

- [ ] A’s compact eight-field `Passage` and `PassageChunker` contract is settled before B/C work and is not edited by later workstreams.
- [ ] Exact offsets, source slices, SHA-256 bytes, full selected-text union coverage, sentence-only overlap, and individually oversized-sentence-only RCTS are covered by focused tests and the fixture benchmark.
- [ ] Exact pair token validation remains authoritative before local, cached, and Flash scoring; the local scorer is lazy and primary.
- [ ] The cache stores complete validated `dict[bytes, float]` maps through the future `ArtifactStoreLike` seam and never writes partial/invalid results.
- [ ] Reranking returns only compact immutable result data, deduplicates scoring by hash and evidence by exact location, retains statuses, and sorts deterministically.
- [ ] Flash is a thin bounded adapter plus a static self-contained class-only worker with pinned dependencies, one-time model load, `BatchEncoding` support, and no provisioning on local construction.
- [ ] Public exports/docs/integration use only the final interfaces; filesystem JSONL publication and dynamic worker generation are gone.
- [ ] Focused tests, benchmark, full unittest discovery, compile/import/static/format/lock/diff/secret checks pass, the production LOC gate proves at least 30% deletion against `94bd6a4`, and no external paid/deploy/push/PR action occurs.
