# Nuggets CLI Handoff

**Status:** The nuggetization stage for issue #16 is complete, merged with the
reranking work, and open as PR #26 (branch `claude/trec-ragtime-issue-16-9d4643`).
A three-lane advisory review plus a Codex PR review found two identity defects,
both now fixed with regression coverage; see Part 2. One task remains: a `ragtime-nuggets` terminal command layered
over the existing Python API. The API stays; the CLI is additive.

**Goal of the remaining task:** Give the nuggets stage a terminal entry point
matching the other runners in this repository, without moving any
result-affecting logic out of `nuggets_runner`.

---

## Part 1 — What already shipped

Four modules and four test files. 517 tests pass; 90 are new.

| Module | Responsibility |
| --- | --- |
| `src/trec_ragtime/nuggetizer_adapter.py` | The only module that imports `nuggetizer`. Provider calls, bounded. |
| `src/trec_ragtime/nuggets.py` | Records, evidence table, clustering, `check_bundle`, the stage. |
| `src/trec_ragtime/nugget_cache.py` | `nuggets/v1/<fingerprint>/<xx>/<request>.mpk` identity and typed cache. |
| `src/trec_ragtime/nuggets_runner.py` | Per-topic orchestration, manifest, reranking projection. |

### How linkage works

Upstream `create()` returns bare claim strings with **no document identity**.
`Document.docid` is carried in the request object but never rendered into the
prompt (`prompts/service.py` builds context as `f"[{i+1}] {doc.segment}"`), and
`ScoredNugget` has no docid field. Linkage therefore comes from the package's
own assigner:

```
create(query, all passages)              -> claims + vital/okay
assign(query, ONE passage, claims) x N   -> support / partial_support / not_support
```

That support matrix **is** the lineage. A claim no passage supports is
quarantined; a topic where nothing survives fails and writes nothing.

### Public API

```python
evidence_from_reranking(results: Sequence[Any], *, queries: Sequence[str],
                        facet_indices: Sequence[int]) -> tuple[FacetEvidence, ...]
build_fingerprint(config: NuggetizerConfig, *, provider_base_url: str,
                  evidence_budget: int = 24, max_nuggets: int = 20,
                  similarity_threshold: float = 0.92) -> NuggetFingerprint
run_topic(work: TopicWork, *, cache, fingerprint, backend_factory, clusterer) -> TopicOutcome
run_topics(work: Sequence[TopicWork], *, cache, fingerprint, backend_factory,
           clusterer, concurrency: int = 4) -> tuple[TopicOutcome, ...]
write_manifest(path: Path, outcomes: Sequence[TopicOutcome], **extra: Any) -> None
```

```python
TopicWork(topic_id, query, representation, facets, inputs, passage_source_fingerprint)
TopicOutcome(topic_id, status, object_key, nugget_count, evidence_count, reason)
NuggetizerConfig(model, max_nuggets=20, creator_window_size=10,
                 scorer_window_size=20, max_calls=64)
NuggetRequest(topic_id, query, representation, inputs, passage_source_fingerprint,
              evidence_digest)
NuggetFingerprint(..., evidence_budget, max_nuggets, output_limits)
NuggetInput(facet_index, query, object_key, request_digest)
```

`TopicOutcome.status` is one of `written`, `cached`, `failed`.

---

## Part 2 — Decisions that must not be undone

These are non-obvious. Each was reached from evidence, and reverting any of them
reintroduces a specific defect.

### The `nuggetizer` pin is `==0.0.5` on purpose

Do **not** "upgrade" it to a Git revision. Every upstream commit since roughly
October 2025 moved the prompts into `src/nuggetizer/prompts/` **without adding
that package to the setuptools `include` list** in upstream's `pyproject.toml`.
A Git-pinned build installs successfully and then fails at import:

```
ModuleNotFoundError: No module named 'nuggetizer.prompts'
```

That is almost certainly why PyPI still stops at 0.0.5, where the prompts are
inline in `models/nuggetizer.py` and the distribution is self-contained. This is
verified, not inferred: `d00dc3eb` was installed, imported, and failed. Revisit
only when upstream publishes a fixed release. The installed version is part of
every artifact fingerprint, so a change retires all previous objects.

### `BoundedHandler` is not optional

0.0.5's `LLMHandler.run` is `while True:` with **no exit condition** — it catches
every exception, rotates the API key, sleeps 0.1s, and loops. One bad key hangs
every topic in a batch silently rather than failing. `BoundedHandler` gives each
topic a finite provider-call budget (`NuggetizerConfig.max_calls`, default 64) so
exhaustion raises and fails that topic.

This is required by the issue's own criteria: a timed-out result cannot be kept
out of the cache if the code never times out. It is also why the handler is
assigned to all three of `creator_llm`, `scorer_llm`, and `assigner_llm`.

Related: upstream's `_create_window` retries a **parse** failure up to 500 times
at `temperature=0.2`. The call budget bounds that too.

### No seed exists upstream

`LLMHandler.run` exposes no `seed`, so identical inputs may produce different
claims on a cold run. Determinism comes from the artifact cache (same key, same
bytes), not from the provider. Do not add a determinism claim stronger than that.

### One `msgspec.Struct` per record, used as domain *and* wire type

`Meta` and `Literal` constraints are enforced on **decode**, so
`msgpack.decode(msgpack.encode(bundle), type=NuggetBundle)` validates alias
formats, offset ranges, enum membership, string lengths, and unknown fields in
one call — at write time and at cache-read time. `check_bundle` therefore only
does referential work.

This deliberately deviates from `retrieval_cache.py`'s dataclass/Struct split.
That split exists there because `RetrievalResult` is a pre-existing type owned by
`models.py`. Nuggets are new, so there is no second representation to bridge.
Do not add parallel dataclasses or `_to_wire`/`_from_wire` converters.

### Bundles store no passage text

Offsets plus `content_hash` prove which text was used, and the text stays
recoverable from the retrieval artifact. Adding text would bloat the artifact and
let it drift from the corpus. `tests/test_nugget_cache.py::test_stores_no_passage_text_or_provider_material`
enforces this.

### `NuggetRequest` carries the topic query and the evidence digest

Both were added after review found holes:

- **`query`.** It drives `create()` *and* `assign()`, but was absent from the
  identity. Reproduced: rewording the topic question while the evidence stayed
  fixed produced the same object key, and the second run returned `cached`
  without invoking the backend — serving claims that answered the previous
  question. `_validate` now also compares `bundle.query` to `request.query`.
- **`evidence_digest`.** An earlier draft validated documents against a set
  derived from the bundle itself, which could never fail. The request now pins
  `evidence_digest(bundle.evidence)`, so an altered alias, document, or offset
  cannot be read back under the same key.

Keep both checks non-tautological. Note that `check_bundle`'s `document_ids`
argument *is* tautological at cache-read time, where it is derived from the
bundle; it is real on the write path, where it comes from the candidate rows.
That is intentional — do not "fix" it by deleting the parameter.

Any new result-affecting input must join `NuggetFingerprint` or `NuggetRequest`
and the isolation test matrix. `NuggetFingerprint.output_limits` covers the
knobs that decide what the model may emit and what survives the atomicity
guard — `max_claim_characters`, `max_claim_words`, `max_completion_tokens` —
and `build_fingerprint` reads them from the modules that enforce them, so
editing a constant retires artifacts without anyone remembering to plumb it.
`REQUEST_TIMEOUT_SECONDS` is deliberately excluded: a timeout aborts a call, it
does not change a successful result.

### `partial_support` retains a claim, and says so

`SUPPORT_LABELS` is `("support", "partial_support")`, so a nugget may be
attested only partially. The distinction is preserved per link on
`NuggetSupport.assignment` rather than collapsed. Nothing in the bundle asserts
entailment — a consumer that needs fully attested claims filters on the label.
Do not silently drop the weaker label when merging paraphrases; `_merge_cluster`
already promotes `support` over `partial_support` for the same alias only.

### The atomicity guard's regex shape is deliberate

`_SENTENCE_BREAK = re.compile(r"[a-z0-9][.!?][\"'’”)\]]*\s+[A-Z]")`

Requiring a lowercase or digit **before** the terminator and a capital **after**
the whitespace keeps `exports fell 3.5%` and `U.S. exports fell` out of the
quarantine while still catching `Grain fell. Ports also closed`. An earlier
version without those anchors quarantined every decimal and abbreviation —
exactly the factual claims this stage exists to preserve. Regression coverage:
`tests/test_nuggets.py::test_keeps_claims_with_decimals_and_abbreviations`.

### Other invariants

- **No fallback path.** Empty or fully-unsupported output fails the topic. Do not
  reintroduce an extractive fallback nugget kind.
- **No checkpoint file.** A warm topic is a cache hit at ~0 cost, so the artifact
  cache is the resume mechanism. `query_baselines.py` needs JSONL checkpoints
  because organizer calls are rate-limited; this stage does not.
- **Clustering guard.** Paraphrases merge on pinned MiniLM cosine ≥ 0.92 **and**
  a matching numeric/negation signature, so `fell 30%` never collapses into
  `fell 13%`. A merge unions evidence so every attesting source survives.

---

## Part 3 — The remaining task: `ragtime-nuggets`

Create `src/trec_ragtime/nuggets_cli.py` and register it:

```toml
ragtime-nuggets = "trec_ragtime.nuggets_cli:main"
```

Keep it a thin shell. It resolves inputs, prepares passages, and reports; it must
not contain result-affecting logic, which belongs in `nuggets_runner`.

### Argument surface

`--model` (required), `--topic-id` (repeatable), `--query-plan`,
`--representation {canonical,report_text}` (default `report_text`), `--top-k`
(default 100), `--evidence-budget`, `--max-nuggets`, `--concurrency`,
`--manifest`. Follow the repository's `build_parser()` / `main(argv) -> int` /
`load_dotenv()` shape. Return `0` on success, `1` if any topic failed, `2` for
caller error.

### Passage preparation

Retrieval is **cache-only**: use `RagtimeRetriever.get_cached_retrieval(...)`. A
topic with no cached facet retrieval is reported as failed, never turned into an
organizer request from this command.

Per facet query:

```python
from trec_ragtime.passage_score_cache import CachedPassageScorer, PassageScoreCache
from trec_ragtime.passage_scoring import (
    CrossEncoderPairLengthFunction, LocalCrossEncoderConfig, LocalCrossEncoderScorer,
)
from trec_ragtime.passages import PassageChunker
from trec_ragtime.reranking import rerank_retrieval

scorer_config = LocalCrossEncoderConfig()          # once per run
scorer = LocalCrossEncoderScorer(scorer_config)    # once per run

# CrossEncoderPairLengthFunction takes the query, so this is PER QUERY.
pair_length = CrossEncoderPairLengthFunction(scorer_config, query)
chunker = PassageChunker(pair_length)
score_cache = PassageScoreCache(
    store, scorer,
    chunker_fingerprint=chunker.fingerprint,
    representation=representation,
)
result = rerank_retrieval(
    retrieval,
    chunker=chunker,
    scorer=CachedPassageScorer(score_cache),
    representation=representation,
)
```

Then, passing the query plan's own numbering rather than positions:

```python
facets = evidence_from_reranking(
    results, queries=kept_queries, facet_indices=kept_facet_indices
)
```

### Facet indices are explicit, not positional

`evidence_from_reranking` requires `facet_indices` alongside `queries`. Pass the
query plan's own numbering — for a topic whose facet 1 has no cached retrieval,
pass `facet_indices=[0, 2]`, not `[0, 1]`.

This used to be positional (`enumerate`), which was a live trap: a CLI that
filtered out the missing facet and built `results`, `queries`, and `inputs` from
the same surviving list produced `{0, 1}` on both sides. `_validate` compares
facet *sets*, so a consistent shift is **equal and passes** — it would have
sealed a bundle attributing plan-facet-2's evidence to `facet_index 1`, silently,
under a key a later fully-cached run would not reproduce. The validator can only
catch *inconsistent* numbering, never consistently wrong numbering, because
`facet_index` is caller-supplied data.

The parameter is now required and duplicates are rejected, so the bug is
unrepresentable. Keep it that way; do not add a positional default.

### Wiring

```python
NuggetInput(facet_index, query, object_key, request_digest)
```

`object_key` and `request_digest` come from each retrieval's
`retrieval_artifact` reference. `facet_index` must match the value passed to
`evidence_from_reranking` for the same facet. `passage_source_fingerprint` should
combine the chunker fingerprint, the score-cache fingerprint, and the
representation, so a chunker or scorer change retires old bundles.

`TopicWork.query` is the topic-level question handed to `create()` and
`assign()`. It is part of `NuggetRequest` and therefore part of the cache key —
see the note in Part 2.

### Tests

Add `tests/test_nuggets_cli.py`. Follow the existing fakes rather than inventing
new ones: `MemoryArtifactStore` from `tests/test_nugget_cache.py`,
`ScriptedBackend` / `candidate` / `fixed_embedder` from `tests/test_nuggets.py`.
Cover at minimum:

- a topic with no cached retrieval is reported, not requested from the organizer;
- facet indices survive a missing middle facet;
- `--topic-id` filtering, and an unknown topic exits 2;
- the manifest is atomic, owner-only (`mode & 0o077 == 0`), and carries no query
  or passage text;
- exit code is 1 when any topic fails.

---

## Part 4 — Working notes

### Verification

```bash
sops decrypt --output .env config/secrets.sops.env
uv run python -m unittest discover -s tests -v
```

**`tests/test_corpus_cli.py::test_download_selects_original_and_english_translation_splits`
fails on this machine and is pre-existing.** It is a umask issue —
`ensure_private_cache_dir` rejects a directory created with group/other bits.
Confirmed identical with all nugget changes stashed. Do not chase it as part of
this work.

Checks run beyond the suite, worth repeating after changes:

- byte-identical artifacts and nugget IDs across two independent runs;
- L1 evict then L2 reread round-trips the typed bundle;
- all five fingerprint fields and all four request fields isolate with no cross-hit;
- a real spaCy `PassageChunker` → `rerank_retrieval` → sealed bundle, asserting
  `english_text[start:end] == text` and `sha256(text) == content_hash` per passage
  (`tests/test_nuggets_runner.py::RealRerankingIntegrationTestCase`).

### Provider configuration

Routing is environment driven and needs no wrapper. Upstream constructs its own
`OpenAI(api_key=...)` with no explicit base URL, so the SDK resolves
`OPENAI_API_KEY` and, when set, `OPENAI_BASE_URL`. That reaches OpenRouter, a
LiteLLM proxy, or OpenAI itself. Never pass credentials through the adapter.

Upstream resolves the key eagerly in `__init__`, so tests that build a real
`Nuggetizer` must set `OPENAI_API_KEY` (see `ProviderKeyTestCase`). No network
call happens at construction.

### Repository footgun

There is a local branch literally named `origin/main` (`refs/heads/origin/main`,
stale at `e4001fd`). Every `git log origin/main` warns that the refname is
ambiguous and may resolve to that stale branch instead of the remote. Use
`refs/remotes/origin/main` explicitly, or delete the local branch.

### Cost shape

With a 24-passage evidence budget, one cold topic is roughly 28 provider calls
(~24 assign + ~3 create + ~1 score) and a warm topic is zero. Parallelize by
topic; one worker owns exactly one bundle and never patches another's.
