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
- Maintained LangChain and spaCy components own sentence segmentation and
  adjacent-sentence packing. The project does not implement a sentence parser or
  a greedy packing algorithm.
- `RecursiveCharacterTextSplitter` is used only when one sentence by itself
  exceeds the configured passage token budget.
- Overlap is zero in schema version 1.
- Local scoring is the default path. Runpod Flash is an optional transport
  adapter implementing the same scorer protocol.
- The issue #12/#13 implementation remains owned by its other branch/session.
  This branch consumes only the published `ArtifactStoreLike` protocol and uses
  in-memory fakes in tests.
- No Runpod endpoint is created or deployed by this work.

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

`LangChainPassageChunker` composes:

1. `SpacyTextSplitter` configured with `separator=""`,
   `strip_whitespace=False`, `chunk_overlap=0`, `add_start_index=True`, and a
   target-model tokenizer length function. This makes LangChain join adjacent
   complete `Span.text_with_ws` values without reconstructing whitespace.
2. `RecursiveCharacterTextSplitter` configured with the same tokenizer,
   zero overlap, no whitespace stripping, and separators `(" ", "")`. It is
   invoked only for an over-budget output from the sentence splitter, which
   proves the over-budget source unit is one individual sentence.

The adapter rejects a missing or negative start index and validates every final
chunk using `selected_text[start_char:end_char] == text`. Nested fallback
offsets are adjusted to document coordinates and revalidated. The production
default uses spaCy's `en_core_web_sm` pipeline for English `report_text`; callers
may supply another installed spaCy pipeline for canonical-language experiments.

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
- sentence start/end indexes when the normal sentence path supplies them;
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
model/tokenizer identity are fingerprinted. CPU, CUDA, and ROCm-compatible
PyTorch device strings are accepted. Before inference, formatted query/passage
pairs are tokenized without truncation and any overflow is rejected; silent
truncation would make a score invalid for the recorded passage.

### `passage_score_cache.py`

This module owns only the passage-score domain adapter. It imports the issue
#12 `ArtifactStoreLike` and `ArtifactValidationError` when available and defines
an interface-compatible fallback solely while that branch has not landed.

The object layout is:

```text
passage-scores/v1/
  <scorer-fingerprint>/
    <query-hash-prefix>/
      <query-hash>.mpk
```

`query_hash` is SHA-256 of exact query UTF-8 bytes only. Passage-set identity is
stored in the strict MessagePack envelope, not the path, because including it
in the path would make overlapping score reuse impossible.

The envelope records schema version, scorer fingerprint, exact query digest,
sorted coverage hashes, coverage digest, and a deterministically ordered
`dict[bytes, float]`. Reads go through `read_validated`. A true miss begins with
an empty map; other storage failures propagate. Missing hashes alone are sent
to the scorer, the returned exact map is validated, and the union is encoded,
decoded, and domain-validated before one write. Failed or partial scoring never
writes. The public return value contains exactly the current request's unique
hashes even when the stored union is larger. One worker owning one query is an
orchestration invariant; last-writer-wins duplicates remain semantically
interchangeable.

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

`write_reranking_artifact` writes a stable manifest and per-topic JSONL into a
new directory atomically and refuses to overwrite an existing run, matching the
repository's current artifact conventions. This does not alter Control-A/B/C.

### `flash_passage_scorer.py`

`RunpodFlashPassageScorer` implements `PassageScorer` and is transport-only.
The caller still owns validation and artifact writes. It batches by both count
and actual serialized UTF-8 payload below Flash's 10 MB limit, sends query plus
hash/text rows, and accepts only hash/finite-score rows. Concurrency, endpoint
ID, timeout, workers, idle timeout, GPU selection, and batch size are
configuration values included where result-affecting.

An optional endpoint runner uses `Endpoint(id=...)`, `run()`, and
`job.wait()` rather than the synchronous timeout path. A dedicated private
event-loop thread bridges that async API to the synchronous scorer protocol.
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
paragraph/sentence behavior, exact offsets, binary hashes, stable IDs across
processes, long-sentence fallback, whole-document short chunks, scorer schema,
local fake-model batching, no truncation, cache fingerprint isolation, warm
reuse, overlap merge, complete-only writes, deterministic aggregation, fake
Flash parity, payload/timeout/concurrency bounds, and log redaction.

After decrypting `.env`, run every focused test and then:

```bash
uv run python -m unittest discover -s tests -v
```

No model-weight download, live organizer request, HF Bucket mutation, Runpod
provisioning, `flash dev`, deployment, push, or pull request is part of this
implementation verification.
