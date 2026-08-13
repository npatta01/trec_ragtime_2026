# Retrieval-to-Generation Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one immutable, self-contained, selected-evidence retrieval bundle per official topic that report generation can consume without reopening retrieval or nugget artifacts.

## Current status (2026-08-12)

- Implementation and the typed generation boundary are complete on
  `codex/retrieval-generation-handoff-design`.
- The corrected warm path discovers durable handoff objects before opening any
  upstream stage; the official 103-topic pass completed as 103 cache hits in
  1.97 seconds.
- The complete owner-only manifest is stored under the shared project cache at
  `experiments/retrieval-handoff/complete-103.json`.
- The manifest covers 1,814 retained nuggets and 2,020 selected evidence rows;
  all 103 producer identities and object keys are present.
- The independent review findings covering exact producer identity, warm
  reuse, strict malformed-L2 handling, direct manifest-key reads, request
  lineage, runtime validation, and durable rereads have been addressed.

**Architecture:** Reconstruct the exact merged PR #26 nugget request using cache-only retrieval and reranking seams, require its sealed nugget artifact to exist, and project its retained support links into a strict MessagePack handoff bundle. Store that derived bundle in the shared artifact store under its own complete identity, validate it before and after durable storage, and expose only a typed cache reader to generation.

**Tech Stack:** Python 3.12, `msgspec` strict structs and MessagePack, existing `ArtifactStoreLike`/Hugging Face Bucket storage, `unittest`, `uv`.

## Global Constraints

- Version 1 consumes only the merged schema-v2 nugget bundle and requires every `NuggetSupport.variant_index`.
- Version 1 source representation is exactly `report_text`; no source-text or Dataset Viewer fallback is allowed.
- Output evidence is exactly the ordered union of aliases linked by retained nuggets; aliases are preserved and never renumbered.
- `support` links are citation-ready; `partial_support` links are retained but are not citation-ready.
- The handoff stage must make no organizer, `/content`, corpus hydration, passage-scoring, nuggetizer, embedding, or provider calls.
- A warm handoff hit must return after validating the handoff object, without opening upstream artifacts.
- Only confirmed artifact absence is a cache miss; transport, permission, malformed L1/L2, and validation failures are fatal.
- On a cold write, validate, write L2, evict the L1 candidate, then independently reread and validate the durable object.
- Never include credentials, local paths, raw topic text, passage text, or provider payloads in logs, failures, or manifests.
- Presence of a valid bundle means success; timestamps, mutable status, and worker identity remain outside the artifact.

---

## File map

- `src/trec_ragtime/nuggetizer_adapter.py`: expose pure nuggetizer fingerprint calculation without constructing a provider client.
- `src/trec_ragtime/nuggets_runner.py`: expose exact nugget rows/request preparation separately from execution.
- `src/trec_ragtime/passage_score_cache.py`: provide a scorer facade that can only return already-cached scores.
- `src/trec_ragtime/retrieval_handoff.py`: own strict bundle records, projection, validation, and content digest.
- `src/trec_ragtime/retrieval_handoff_cache.py`: own handoff identity, envelope, object keys, strict cache reads/writes.
- `src/trec_ragtime/retrieval_handoff_runner.py`: orchestrate one topic, parallel topics, outcomes, and manifest.
- `src/trec_ragtime/retrieval_handoff_cli.py`: reproduce exact upstream identities cache-only and build requested handoffs.
- `src/trec_ragtime/__init__.py`, `pyproject.toml`, `README.md`: public API, command registration, and operator documentation.

### Task 1: Add pure upstream identity and cache-only preparation seams

**Files:**
- Modify: `src/trec_ragtime/nuggetizer_adapter.py`
- Modify: `src/trec_ragtime/nuggets_runner.py`
- Modify: `src/trec_ragtime/passage_score_cache.py`
- Test: `tests/test_nuggetizer_adapter.py`
- Test: `tests/test_nuggets_runner.py`
- Test: `tests/test_passage_score_cache.py`

**Interfaces:**
- Consumes: `NuggetizerConfig`, `TopicWork`, `NuggetFingerprint`, `NuggetRequest`, `PassageScoreCache`.
- Produces: `backend_fingerprint(config: NuggetizerConfig) -> Mapping[str, object]`; `prepare_nugget_request(work: TopicWork, fingerprint: NuggetFingerprint) -> tuple[tuple[FacetEvidence, ...], NuggetRequest]`; `CacheOnlyPassageScorer(cache: PassageScoreCache, fingerprint: str)` with `score(query: str, passages: Sequence[Passage]) -> Sequence[float]`.

- [ ] **Step 1: Write failing pure-identity and exact-request tests**

```python
def test_backend_fingerprint_does_not_build_provider(self):
    with patch("trec_ragtime.nuggetizer_adapter.build", side_effect=AssertionError):
        identity = backend_fingerprint(self.config)
    self.assertEqual(identity["model"], self.config.model)

def test_prepare_nugget_request_matches_run_topic_identity(self):
    rows, request = prepare_nugget_request(work(), fingerprint())
    self.assertEqual(request.evidence_digest, evidence_digest(tuple(
        item for row in rows for item in row.candidates
    )))
    self.assertEqual(tuple(item.facet_index for item in request.inputs), (1, 2))
```

- [ ] **Step 2: Run the focused tests and verify missing imports/functions fail**

Run: `uv run python -m unittest tests.test_nuggetizer_adapter tests.test_nuggets_runner -v`

Expected: FAIL because `backend_fingerprint` and `prepare_nugget_request` do not exist.

- [ ] **Step 3: Extract the existing production calculations without changing their bytes**

```python
def backend_fingerprint(config: NuggetizerConfig) -> Mapping[str, object]:
    return {
        "adapter": ADAPTER_VERSION,
        "model": config.model,
        "provider": config.provider,
        "package": dict(package_identity()),
        "parameters": dict(config.parameters),
    }

def prepare_nugget_request(
    work: TopicWork,
    fingerprint: NuggetFingerprint,
) -> tuple[tuple[FacetEvidence, ...], NuggetRequest]:
    rows = evidence_from_reranking(work)
    request = NuggetRequest(
        topic_id=work.topic_id,
        query=work.query,
        representation=work.representation,
        inputs=work.inputs,
        passage_source_fingerprint=work.passage_source_fingerprint,
        evidence_digest=evidence_digest(tuple(
            evidence for row in rows for evidence in row.candidates
        )),
    )
    return rows, request
```

Make `NuggetizerBackend.fingerprint` delegate to `backend_fingerprint`, and make `run_topic` call `prepare_nugget_request` before its cache lookup. Preserve the existing request field ordering and evidence flattening order exactly.

- [ ] **Step 4: Write failing cache-only scorer tests**

```python
def test_cache_only_scorer_returns_cached_scores_without_delegate(self):
    scorer = CacheOnlyPassageScorer(self.cache, self.fingerprint)
    self.assertEqual(scorer.score("query", self.passages), [0.8, 0.2])

def test_cache_only_scorer_fails_on_confirmed_miss(self):
    scorer = CacheOnlyPassageScorer(self.cache, self.fingerprint)
    with self.assertRaisesRegex(KeyError, "cached passage scores required"):
        scorer.score("uncached", self.passages)
```

- [ ] **Step 5: Implement cache-only batch lookup**

Add `PassageScoreCache.get_many(fingerprint, query, passages) -> list[float] | None` by extracting the read/validation half of `CachedPassageScorer.score`. Implement `CacheOnlyPassageScorer.score` as one call to `get_many`; raise a bounded `KeyError` containing hashes/counts, never query or passage text, when the result is `None`. Do not accept a delegate scorer.

- [ ] **Step 6: Run all affected tests**

Run: `uv run python -m unittest tests.test_nuggetizer_adapter tests.test_nuggets_runner tests.test_passage_score_cache -v`

Expected: PASS, including existing byte-identity/cache behavior tests.

- [ ] **Step 7: Commit the independently usable seams**

```bash
git add src/trec_ragtime/nuggetizer_adapter.py src/trec_ragtime/nuggets_runner.py src/trec_ragtime/passage_score_cache.py tests/test_nuggetizer_adapter.py tests/test_nuggets_runner.py tests/test_passage_score_cache.py
git commit -m "refactor: expose cache-only nugget preparation"
```

### Task 2: Define and validate the self-contained handoff domain model

**Files:**
- Create: `src/trec_ragtime/retrieval_handoff.py`
- Create: `tests/test_retrieval_handoff.py`

**Interfaces:**
- Consumes: `Topic`, `NuggetBundle`, `NuggetRequest`, ordered `NuggetInput` and corresponding validated retrieval objects.
- Produces: frozen strict records `HandoffTopic`, `HandoffRetrievalInput`, `HandoffEvidence`, `HandoffSupport`, `HandoffNugget`, `HandoffFacet`, `HandoffNuggetSource`, `HandoffCounts`, `RetrievalHandoffBundle`; functions `project_retrieval_handoff(...) -> RetrievalHandoffBundle`, `check_retrieval_handoff(bundle) -> None`, and `handoff_bundle_digest(bundle) -> bytes`.

- [ ] **Step 1: Write the happy-path projection fixture and assertions**

```python
def test_projection_is_self_contained_and_selected_only(self):
    bundle = project_retrieval_handoff(
        topic=self.topic,
        nugget_bundle=self.nuggets,
        nugget_source=self.nugget_source,
        retrievals=self.retrievals,
    )
    self.assertEqual([row.alias for row in bundle.evidence], ["E001", "E003"])
    self.assertEqual(bundle.evidence[0].text, "exact English passage")
    self.assertEqual(bundle.nuggets[0].support[0].citation_ready, True)
    self.assertEqual(bundle.nuggets[0].support[1].citation_ready, False)
    check_retrieval_handoff(bundle)
```

- [ ] **Step 2: Run the focused test and verify the new module is absent**

Run: `uv run python -m unittest tests.test_retrieval_handoff -v`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Add strict schema-v1 records with explicit fields**

```python
class HandoffEvidence(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    alias: str
    document_id: str
    text: str
    representation: Literal["report_text"]
    language: str
    start: int
    end: int
    content_hash: bytes
    passage_score: float
    document_rank: int
    facet_index: int
    facet_query: str
    retrieval_input_index: int

class HandoffSupport(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    alias: str
    assignment: Literal["support", "partial_support"]
    variant_index: int
    citation_ready: bool

class RetrievalHandoffBundle(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    schema_version: Literal[1]
    behavior_version: str
    representation: Literal["report_text"]
    topic: HandoffTopic
    retrieval_inputs: tuple[HandoffRetrievalInput, ...]
    nugget_source: HandoffNuggetSource
    evidence: tuple[HandoffEvidence, ...]
    nuggets: tuple[HandoffNugget, ...]
    facets: tuple[HandoffFacet, ...]
    counts: HandoffCounts
```

Define every nested record in this module, including complete official topic fields, retrieval object key/fingerprint/request digest, nugget object key/fingerprint/request/content digest, nugget variants and facet coverage, and sanitized selection/rejection counts.

- [ ] **Step 4: Implement deterministic projection and validation**

Compute selected aliases by walking ordered retained nuggets and their ordered support links, then filtering the upstream evidence table in its existing alias order. Resolve each alias through its owning facet retrieval input and copy exact `report_text`; verify document identity, slice offsets, SHA-256, finite score, rank, facet/query, corpus identity, and identical duplicate documents. Set `citation_ready = (assignment == "support")`. Reject zero nuggets/evidence and every unknown, duplicate, or unclosed alias.

- [ ] **Step 5: Add explicit corruption and policy tests**

```python
def test_partial_support_is_never_citation_ready(self): ...
def test_unused_evidence_is_omitted_without_alias_renumbering(self): ...
def test_unknown_support_alias_fails_closed(self): ...
def test_offset_slice_or_hash_mismatch_fails_closed(self): ...
def test_same_document_disagreement_between_facets_fails_closed(self): ...
def test_wrong_topic_query_representation_or_retrieval_identity_fails(self): ...
def test_variant_index_must_address_preserved_variant(self): ...
def test_strict_msgpack_round_trip_has_stable_bytes(self): ...
```

Use table-driven `subTest` mutations and assert bounded exception messages contain field names/identities but no topic or passage text.

- [ ] **Step 6: Run the domain suite**

Run: `uv run python -m unittest tests.test_retrieval_handoff -v`

Expected: PASS for projection, closure, policy, corruption, stable encoding, and digest tests.

- [ ] **Step 7: Commit the domain model**

```bash
git add src/trec_ragtime/retrieval_handoff.py tests/test_retrieval_handoff.py
git commit -m "feat: define retrieval generation handoff"
```

### Task 3: Add immutable handoff identity and artifact cache

**Files:**
- Create: `src/trec_ragtime/retrieval_handoff_cache.py`
- Create: `tests/test_retrieval_handoff_cache.py`

**Interfaces:**
- Consumes: `RetrievalHandoffBundle`, `handoff_bundle_digest`, `ArtifactStoreLike`.
- Produces: `HandoffFingerprint`, `HandoffRequest`, `HandoffEnvelope`, `RetrievalHandoffArtifactCache`; `selected_evidence_digest(bundle) -> bytes`; `handoff_object_key(fingerprint: bytes, request_digest: bytes) -> str`.

- [ ] **Step 1: Write failing identity and strict round-trip tests**

```python
def test_object_key_is_namespaced_and_sharded(self):
    self.assertEqual(
        handoff_object_key(bytes.fromhex("11" * 32), bytes.fromhex("22" * 32)),
        "retrieval-handoff/v1/" + "11" * 32 + "/22/" + "22" * 32 + ".mpk",
    )

def test_put_evicts_l1_then_independently_rereads_l2(self):
    stored = self.cache.put(self.fingerprint, self.request, self.bundle)
    self.assertEqual(stored, self.bundle)
    self.assertEqual(self.store.evictions, [self.expected_key])
    self.assertGreaterEqual(self.store.reads, 1)
```

- [ ] **Step 2: Run the cache test and verify imports fail**

Run: `uv run python -m unittest tests.test_retrieval_handoff_cache -v`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement full behavior and request identity**

```python
class HandoffFingerprint(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    schema_version: Literal[1]
    behavior_version: str
    selection_policy: str
    citation_policy: str
    representation: Literal["report_text"]
    validation_version: str

class HandoffRequest(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    topic_catalog_digest: bytes
    topic: HandoffTopic
    nugget_source: HandoffNuggetSource
    retrieval_inputs: tuple[HandoffRetrievalInput, ...]
    selected_evidence_digest: bytes

class HandoffEnvelope(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    fingerprint_digest: bytes
    request_digest: bytes
    bundle_digest: bytes
    bundle: RetrievalHandoffBundle
```

Encode fingerprints/requests canonically with the project MessagePack helpers and SHA-256. Validate every fixed 32-byte digest, nonblank version, exact representation, topic/source agreement, and exact request-to-bundle field equality.

- [ ] **Step 4: Implement strict cache semantics**

```python
class RetrievalHandoffArtifactCache:
    def get(self, fingerprint: HandoffFingerprint, request: HandoffRequest) -> RetrievalHandoffBundle | None: ...
    def put(
        self,
        fingerprint: HandoffFingerprint,
        request: HandoffRequest,
        bundle: RetrievalHandoffBundle,
    ) -> RetrievalHandoffBundle: ...
```

Follow `NuggetArtifactCache`: `get` returns `None` only for confirmed absence and validates envelope identities plus `check_retrieval_handoff`; `put` validates before writing, requires a successful write, evicts local L1, rereads through `read_validated`, and returns only the independently verified durable bundle. Do not catch operational store errors as misses.

- [ ] **Step 5: Add isolation and failure tests**

```python
def test_every_fingerprint_field_isolates_object_key(self): ...
def test_every_request_field_isolates_object_key(self): ...
def test_warm_get_opens_no_upstream_object(self): ...
def test_malformed_l1_repairs_once_from_valid_l2(self): ...
def test_malformed_l2_and_operational_reads_are_fatal(self): ...
def test_put_rejects_bundle_request_disagreement_before_write(self): ...
def test_unknown_msgpack_fields_and_noncanonical_bytes_fail(self): ...
```

- [ ] **Step 6: Run cache and domain suites**

Run: `uv run python -m unittest tests.test_retrieval_handoff tests.test_retrieval_handoff_cache -v`

Expected: PASS with exact cache miss/failure boundaries.

- [ ] **Step 7: Commit the cache**

```bash
git add src/trec_ragtime/retrieval_handoff_cache.py tests/test_retrieval_handoff_cache.py
git commit -m "feat: cache retrieval handoff bundles"
```

### Task 4: Orchestrate topic builds and write the sanitized manifest

**Files:**
- Create: `src/trec_ragtime/retrieval_handoff_runner.py`
- Create: `tests/test_retrieval_handoff_runner.py`

**Interfaces:**
- Consumes: `Topic`, validated nugget/retrieval inputs, `HandoffFingerprint`, `RetrievalHandoffArtifactCache`, `project_retrieval_handoff`.
- Produces: `HandoffWork`, `HandoffOutcome`; `build_handoff_request(work) -> HandoffRequest`; `run_topic(work, fingerprint, cache) -> HandoffOutcome`; `run_topics(works, fingerprint, cache, workers) -> tuple[HandoffOutcome, ...]`; `write_handoff_manifest(path, outcomes, *, topic_catalog_digest, fingerprint) -> None`.

- [ ] **Step 1: Write failing cold-build and warm-resume tests**

```python
def test_run_topic_builds_one_bundle_on_miss(self):
    outcome = run_topic(self.work, self.fingerprint, self.cache)
    self.assertEqual(outcome.status, "written")
    self.assertEqual(outcome.topic_id, self.work.topic.topic_id)

def test_run_topic_returns_warm_hit_without_loading_upstream(self):
    self.cache.seed(self.bundle)
    self.work.upstream_loader = Mock(side_effect=AssertionError("must not open"))
    outcome = run_topic(self.work, self.fingerprint, self.cache)
    self.assertEqual(outcome.status, "cached")
```

Model `HandoffWork` so request identity and the handoff cache lookup can be built before invoking an `upstream_loader` callback. The callback returns the already-validated objects needed only for cold projection.

- [ ] **Step 2: Run the focused runner test and verify imports fail**

Run: `uv run python -m unittest tests.test_retrieval_handoff_runner -v`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement request-first one-topic orchestration**

```python
def run_topic(work, fingerprint, cache):
    request = build_handoff_request(work)
    cached = cache.get(fingerprint, request)
    if cached is not None:
        return outcome_for(cached, status="cached")
    nugget_bundle, retrievals = work.upstream_loader()
    bundle = project_retrieval_handoff(
        topic=work.topic,
        nugget_bundle=nugget_bundle,
        nugget_source=work.nugget_source,
        retrievals=retrievals,
    )
    durable = cache.put(fingerprint, request, bundle)
    return outcome_for(durable, status="written")
```

Catch topic-local exceptions only in `run_topics`; sanitize them to exception class plus bounded identities. Preserve official input ordering in returned outcomes even when workers finish out of order. One worker owns one topic.

- [ ] **Step 4: Write concurrency and manifest tests**

```python
def test_parallel_results_remain_in_official_topic_order(self): ...
def test_same_topic_is_not_written_twice(self): ...
def test_manifest_is_atomic_owner_only_and_contains_no_payload_text(self): ...
def test_manifest_fails_overall_when_any_required_topic_failed(self): ...
def test_manifest_counts_full_partial_and_citation_ready_links(self): ...
```

- [ ] **Step 5: Implement deterministic outcomes and manifest**

Write JSON to a sibling temporary file opened with mode `0o600`, `fsync`, then `os.replace`. Include schema/fingerprint/catalog hashes, official-order topic outcomes, status, object key, and aggregate counts only. Set top-level `complete` only when every outcome is `cached` or `written`; reject duplicate/missing topic IDs supplied by the CLI.

- [ ] **Step 6: Run runner, cache, and domain suites**

Run: `uv run python -m unittest tests.test_retrieval_handoff_runner tests.test_retrieval_handoff_cache tests.test_retrieval_handoff -v`

Expected: PASS for cache-first resume, ownership, ordering, sanitized failures, and atomic manifest behavior.

- [ ] **Step 7: Commit orchestration**

```bash
git add src/trec_ragtime/retrieval_handoff_runner.py tests/test_retrieval_handoff_runner.py
git commit -m "feat: orchestrate retrieval handoff builds"
```

### Task 5: Add the cache-only CLI and public generation boundary

**Files:**
- Create: `src/trec_ragtime/retrieval_handoff_cli.py`
- Modify: `src/trec_ragtime/__init__.py`
- Modify: `pyproject.toml`
- Create: `tests/test_retrieval_handoff_cli.py`

**Interfaces:**
- Consumes: merged `ragtime-nuggets` query/topic/model configuration, `prepare_topic`, `prepare_nugget_request`, `backend_fingerprint`, `CacheOnlyPassageScorer`, nugget/retrieval artifact caches.
- Produces: console command `ragtime-retrieval-handoff`; public handoff bundle/cache types for report generation.

- [ ] **Step 1: Write failing CLI cache-only tests**

```python
def test_cli_builds_handoff_from_existing_artifacts(self): ...
def test_cli_fails_when_retrieval_score_or_nugget_cache_is_missing(self): ...
def test_cli_never_constructs_provider_or_computing_scorer(self): ...
def test_warm_cli_run_opens_only_handoff_cache(self): ...
def test_cli_requires_every_selected_official_topic(self): ...
```

Patch all organizer, corpus, scorer delegate, and nuggetizer constructors to raise `AssertionError`; seed only exact retrieval, passage-score, and nugget artifacts.

- [ ] **Step 2: Run the focused test and verify command/module absence**

Run: `uv run python -m unittest tests.test_retrieval_handoff_cli -v`

Expected: FAIL because the CLI module and script do not exist.

- [ ] **Step 3: Implement the CLI by reusing nugget identity logic**

Parse the same topic catalog, query plan, retrieval namespace, reranking config, nuggetizer config, topic selection, workers, artifact store, and manifest arguments as `ragtime-nuggets`. Calculate the nugget fingerprint with `backend_fingerprint`; construct `CacheOnlyPassageScorer`; call `prepare_topic` and `prepare_nugget_request`; require `NuggetArtifactCache.get(...)` to hit; build `HandoffWork` with a cold-only retrieval loader; then call `run_topics` and `write_handoff_manifest`. Return nonzero when any topic fails.

- [ ] **Step 4: Register the command and exports**

```toml
[project.scripts]
ragtime-retrieval-handoff = "trec_ragtime.retrieval_handoff_cli:main"
```

Export `RetrievalHandoffBundle`, `RetrievalHandoffArtifactCache`, `HandoffFingerprint`, and `HandoffRequest` from `trec_ragtime.__init__` so generation never imports runner/CLI internals.

- [ ] **Step 5: Run CLI and full offline tests**

Run: `uv run python -m unittest tests.test_retrieval_handoff_cli -v`

Run: `uv run python -m unittest discover -s tests -v`

Expected: PASS; existing retrieval and nugget command behavior remains unchanged.

- [ ] **Step 6: Commit the executable path**

```bash
git add src/trec_ragtime/retrieval_handoff_cli.py src/trec_ragtime/__init__.py pyproject.toml tests/test_retrieval_handoff_cli.py
git commit -m "feat: build generation handoffs cache only"
```

### Task 6: Document and verify the complete official-topic handoff

**Files:**
- Modify: `README.md`
- Create: `tests/test_retrieval_handoff_live.py`

**Interfaces:**
- Consumes: `ragtime-retrieval-handoff`, completed merged PR #26 artifacts, official topic catalog.
- Produces: documented operator command, complete sanitized manifest, one validated handoff bundle per official topic.

- [ ] **Step 1: Add the operator workflow and consumer example**

Document prerequisite decryption, required existing retrieval/reranking/nugget caches, command invocation, cache-only failure semantics, manifest meaning, and generation usage:

```python
bundle = handoff_cache.get(fingerprint, request)
assert bundle is not None
for nugget in bundle.nuggets:
    citation_aliases = tuple(link.alias for link in nugget.support if link.citation_ready)
```

- [ ] **Step 2: Add an opt-in live invariant test**

```python
@unittest.skipUnless(os.environ.get("RAGTIME_LIVE_HANDOFF") == "1", "opt-in")
def test_all_official_topics_are_complete_and_warm(self):
    first = run_official_handoffs()
    self.assertTrue(first.complete)
    self.assertEqual(first.topic_count, official_topic_count())
    second = run_official_handoffs(forbid_upstream=True)
    self.assertTrue(all(row.status == "cached" for row in second.outcomes))
```

Also iterate every durable bundle, recompute copied passage hashes, check upstream references, and evict/reread a deterministic sample from L2.

- [ ] **Step 3: Run cheap verification before the live pass**

Run: `git diff --check`

Run: `uv run python -m unittest discover -s tests -v`

Expected: no whitespace errors and all offline tests PASS.

- [ ] **Step 4: Run the complete official-topic build when caches are present**

Run: `sops decrypt --output .env config/secrets.sops.env`

Run: `git check-ignore .env`

Run: `RAGTIME_LIVE_HANDOFF=1 uv run python -m unittest tests.test_retrieval_handoff_live -v`

Expected: one validated bundle per official topic; sampled L2 rereads pass; the second full pass is entirely `cached` and makes no upstream call. If a required upstream cache is absent, report its exact sanitized identity and do not substitute live computation.

- [ ] **Step 5: Commit documentation and live verification**

```bash
git add README.md tests/test_retrieval_handoff_live.py
git commit -m "docs: verify retrieval generation handoff"
```

## Two-day critical path

Day 1 completes Tasks 1–3: exact cache-only seams, bundle contract, and durable cache. Day 2 completes Tasks 4–5 and the offline portion of Task 6. The full live 103-topic verification runs only if all merged upstream artifacts are already present; missing caches are reported as prerequisites rather than regenerated by custom code.
