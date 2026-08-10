# Passage Reranking Design

## Objective

Implement GitHub issue #15 as a derived, deterministic layer downstream of
hydrated retrieval. The layer prepares exact source passages, scores them with
one local-or-remote protocol, reuses complete per-query score maps through the
issue #12 `ArtifactStoreLike` interface, and emits auditable document rankings.
It does not modify organizer retrieval, the canonical DuckDB corpus, or the
Control-A/B/C artifact formats.

## Approved decisions

- Normal report synthesis chunks English `report_text`. Canonical source text
  remains supported for the required English/Spanish/Russian/Chinese comparison
  and for documents without a verified English representation.
- Maintained LangChain and spaCy components own paragraph packing, sentence
  segmentation, and adjacent-sentence packing. The project does not implement a
  paragraph/sentence parser or a greedy packing algorithm.
- `RecursiveCharacterTextSplitter` is used only when one sentence by itself
  exceeds the configured passage token budget.
- Production chunking counts the complete formatted query/passage pair with the
  same pinned tokenizer used for scoring, including special tokens. Character
  `len` remains available only as an explicit boundary-test/benchmark policy;
  it is never an implicit production ceiling.
- Overlap is zero in schema version 1.
- Local scoring is the default path. Runpod Flash is an optional transport
  adapter implementing the same scorer protocol.
- The issue #12/#13 implementation remains owned by its other branch/session.
  This branch consumes only the published `ArtifactStoreLike` protocol and uses
  in-memory fakes in tests.
- This implementation creates no persistent Runpod deployment. An ephemeral
  `flash dev` worker smoke is a separately authorized deferred gate and is not
  part of the offline implementation run.
- Within this branch, Task 1 exclusively owns dependency/lock changes and Task 6
  exclusively owns exports/documentation. Sol owns final reconciliation of
  those shared files after #12/#13 land; this branch does not preemptively rebase
  or copy their pending implementation.

## Module boundaries

### `passages.py`

This module owns passage lineage and the LangChain adapter. It exposes:

```python
class PassageChunker(Protocol):
    @property
    def fingerprint(self) -> str: ...

    def chunk(
        self,
        *,
        document_id: str,
        canonical_text: str,
        selected_text: str,
        representation: Literal["canonical", "report_text"],
        language: str,
    ) -> tuple[Passage, ...]: ...
```

`LangChainPassageChunker` composes three maintained stages:

1. `CharacterTextSplitter` configured with regex separator
   `r"(?:\r?\n){2,}"`, `is_separator_regex=True`, `keep_separator="end"`,
   `strip_whitespace=False`, `chunk_overlap=0`, `add_start_index=True`, and the
   same target-model `chunk_size`/tokenizer `length_function` used at every
   stage. It packs adjacent complete LF or CRLF paragraphs while retaining
   exact blank-line bytes. Only a single over-budget paragraph descends.
2. `SpacyTextSplitter` configured with `separator=""`,
   `strip_whitespace=False`, `chunk_overlap=0`, `add_start_index=True`, and the
   target-model tokenizer length function. It packs adjacent complete
   `Span.text_with_ws` values without reconstructing whitespace. Only a single
   over-budget sentence descends.
3. `RecursiveCharacterTextSplitter` configured with the same tokenizer,
   zero overlap, no whitespace stripping, and separators `(" ", "")`. It is
   invoked only for that individually over-budget sentence.

Every stage uses `create_documents()` so its maintained splitter supplies
`start_index`. The adapter rejects missing or negative indexes and, within each
parent span, requires cursor equality, exact source slices, monotonic spans,
and full reconstruction (`"".join(chunks) == parent_text`). Nested offsets are
adjusted to document coordinates and all checks are repeated on the final
sequence.

`ChunkerConfig` requires an explicit length callable. The production callable
is `CrossEncoderPairLengthFunction`, which captures one exact query and local
model policy, lazy-loads tokenizer assets without model weights, and exposes
its own stable fingerprint. A custom callable must likewise expose a stable
fingerprint or be paired with an explicit one.

The production pipeline mapping is pinned to spaCy 3.8.0 model wheels:
`eng -> en_core_web_sm`, `spa -> es_core_news_sm`, `rus -> ru_core_news_sm`, and
`zho -> zh_core_web_sm`; English `report_text` also uses `en_core_web_sm`.
Callers may override a pipeline name, but the benchmark fails clearly if a
required pinned model is unavailable.

The chunker fingerprint composes configuration and any caller namespace with
the installed `langchain-text-splitters`, spaCy, and selected spaCy-model
distribution versions. A caller namespace cannot replace those identities.

The selected dependency is recorded only after a fixture benchmark checks
short text, multiple paragraphs, headings/lists, abbreviations, a very long
sentence, CJK without Western whitespace, repeated identical sentences, CRLF,
and translated report text. The benchmark measures exact-slice validity,
determinism, boundary behavior, startup time, and throughput. It is a small
script, not a reusable framework.

### Passage lineage

`Passage` is a frozen, slotted dataclass containing:

- `document_id`;
- `source_content_hash`: SHA-256 bytes of exact canonical UTF-8 text;
- `representation`: `canonical` or `report_text`;
- `representation_hash`: SHA-256 bytes of exact selected UTF-8 text;
- `language` (the selected representation language; English for report text);
- stable `chunk_index` and binary `passage_id`;
- exact `text`, `start_char`, and exclusive `end_char`;
- boundary-path metadata: `paragraph_packed`, `sentence_packed`, or
  `oversized_sentence_fallback`;
- `previous_passage_id` and `next_passage_id`;
- binary `content_hash`: SHA-256 bytes of exact passage UTF-8 text;
- chunker fingerprint and explicit overlap characters/tokens.

No Unicode normalization or whitespace stripping occurs before hashing.
`passage_id` is SHA-256 over a canonical MessagePack lineage tuple containing
document ID, representation, representation hash, offsets, chunk index, and
chunker fingerprint. Neighbor IDs are assigned after all passage IDs exist.

### `passage_scoring.py`

The common contract is:

```python
class PassageScorer(Protocol):
    @property
    def fingerprint(self) -> str: ...

    def score(
        self,
        query: str,
        passages: Sequence[Passage],
    ) -> Mapping[bytes, float]: ...
```

`validate_passage_scores` requires exactly one finite non-Boolean float for
each unique requested 32-byte content hash and rejects missing or extra keys.

`LocalCrossEncoderScorer` lazily loads a pinned
`sentence_transformers.CrossEncoder`. The initial default is
`BAAI/bge-reranker-v2-m3` at revision
`953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`, using raw logits. Device,
precision, batch size, maximum pair length, query format, library version, and
model/tokenizer identity are fingerprinted. CPU and CUDA PyTorch device strings
are accepted; a user-facing `rocm` alias normalizes to PyTorch's `cuda` device
on ROCm builds. Before inference, formatted query/passage pairs are tokenized
without truncation and any overflow is rejected; silent truncation would make a
score invalid for the recorded passage.

The immutable policy records explicit tokenizer name/revision and the installed
Sentence Transformers, Transformers, Tokenizers, and PyTorch versions. Policy
construction fails closed if declared runtime metadata differs from the local
environment. `CrossEncoderPairLengthFunction` uses the same query formatting
and tokenizer identity as inference. It exposes a passage-only no-special-token
length and a passage budget derived from the formatted query plus the
tokenizer's official pair-special-token count for LangChain's additive packing.
The complete pair length remains authoritative and is revalidated for every
emitted passage before prediction.

### `passage_score_cache.py`

This module owns only the passage-score domain adapter. It references the issue
#12 `ArtifactStoreLike` under `TYPE_CHECKING` and otherwise uses structural duck
typing; it does not ship a fallback artifact store or counterfeit the #12
exception identity. The domain decoder raises
`PassageScoreArtifactValidationError`. The cache constructor accepts the exact
artifact-store validation exception type/factory and adapts domain decode
failures to it, so tests inject the fake-store exception now and callers inject
#12 `ArtifactValidationError` after that implementation lands.

The object layout is:

```text
passage-scores/v1/
  <effective-score-fingerprint>/
    <query-hash-prefix>/
      <query-hash>.mpk
```

`query_hash` is SHA-256 of exact query UTF-8 bytes only. Passage-set identity is
stored in the strict MessagePack envelope, not the path, because including it
in the path would make overlapping score reuse impossible.

The effective score fingerprint covers the cache schema, model scorer
fingerprint, chunker fingerprint, and selected representation. The envelope
records those fields, the exact query digest, sorted coverage hashes, coverage
digest, and a deterministically ordered `dict[bytes, float]`. Reads go through
`read_validated`. A true miss begins with an empty map; other storage failures
propagate. Missing hashes alone are sent to the scorer, the returned exact map
is validated, and the union is encoded, decoded, and domain-validated before
one write. Failed or partial scoring never writes. The public return value
contains exactly the current request's unique hashes even when the stored union
is larger. One worker owning one query is an orchestration invariant;
last-writer-wins duplicates remain semantically interchangeable.

### `reranking.py`

`rerank_retrieval` consumes an existing `RetrievalResult`, chooses canonical or
English report representation explicitly, prepares passages, obtains cached
scores, and aggregates with versioned `max_passage_v1`.

Max passage score avoids rewarding documents merely for producing more
variable-length chunks. Equal maximum passages break ties by content hash.
Scored documents sort by descending document score, organizer rank, document
ID, and winning passage hash. Documents without a selected representation or
without passages remain in the output after scored documents with an explicit
status. Organizer rank/score, query, document metadata, passage IDs, chunker
fingerprint, scorer fingerprint, cache key, representation, and aggregation
version are retained.

`write_reranking_artifact(..., source_retrieval_ref: str)` requires the immutable
source retrieval reference explicitly; it never infers provenance from the
mutable in-memory result. It writes a stable manifest and per-topic JSONL into
a new directory atomically and refuses to overwrite an existing run, matching
the repository's current artifact conventions. This does not alter Control-A/B/C.

### `flash_passage_scorer.py`

`RunpodFlashPassageScorer` implements `PassageScorer` and is transport-only.
The caller still owns validation and artifact writes. It batches by both count
and actual serialized UTF-8 payload below Flash's 10 MB limit, sends query plus
hash/text rows, and accepts only hash/finite-score rows. Concurrency, endpoint
ID, timeout, workers, idle timeout, GPU selection, and batch size are
configuration values. Only result-affecting model/precision/device/batch policy
is included in the score fingerprint; endpoint ID, worker count, idle timeout,
request timeout, and transport concurrency are excluded.

An optional endpoint runner uses `Endpoint(id=...)`, `run()`, and
`job.wait()` rather than the synchronous timeout path. A dedicated private
event-loop thread bridges that async API to the synchronous scorer protocol.
Startup uses a stop-request handshake and bounded cleanup so a late or broken
event-loop factory cannot hang the host process. An Endpoint-style worker
decorator receives the configured worker name, resolved GPU group/type,
workers, idle timeout, and exact runtime dependencies; constructing the raw
class performs no provisioning. The generated worker reconstructs the full
embedded local policy and rejects runtime-version drift before loading or
reporting a scorer fingerprint.
The worker definition uses a class-style endpoint whose initializer loads the
same pinned local scorer once per worker. Worker-required imports and helpers
remain inside the shipped class scope. Neither worker nor transport logs query
or passage text, tokens, hashes tied to text, or credentials.

## Error behavior

- Empty or malformed document/query inputs raise before model or storage work.
- Unknown/reconstructed offsets, mismatched source slices, malformed hashes,
  overlapping/reordered source spans, non-finite scores, missing/extra score
  keys, tokenizer overflow, malformed remote payloads, and corrupt MessagePack
  are fatal validation errors.
- An absent score object is the only nonfatal artifact-store miss.
- A document-level missing representation is a valid explicit reranking status,
  not an exception and not a silently dropped document.
- Model, transport, timeout, and storage failures are never cached as scores.

## Verification

Focused tests cover multilingual and translated fixtures, repeated text,
paragraph/sentence behavior, exact offsets and reconstruction at every splitter
level, binary hashes, stable IDs across processes, long-sentence fallback,
whole-document short chunks, scorer schema, local fake-model batching, no
truncation, a real pinned-tokenizer CJK ceiling smoke, cache fingerprint
isolation, warm reuse, overlap merge,
complete-only writes, deterministic aggregation, exact parity under one
deterministic local/remote fake, payload/timeout/concurrency bounds, and log
redaction.

After decrypting `.env`, run every focused test and then:

```bash
uv run python -m unittest discover -s tests -v
```

Offline verification does not download model weights, call the organizer,
mutate HF Bucket, provision Runpod, run `flash dev`, deploy, push, or open a pull
request. Four acceptance gates require separate authorization or dependencies:

- one real local batch using the pinned cross-encoder (large model download);
- organizer-only versus canonical versus `report_text` ranking comparison;
- ephemeral `flash dev` real-worker parity smoke (external GPU cost);
- L2/cross-machine score reuse after the issue #12 ArtifactStore lands.

The handoff reports these as deferred unless independently authorized and
evidenced; fake-only tests are not represented as completion of those gates.
