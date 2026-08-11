# Passage Reranking Simplification Design

## Status

Approved direction from the user on 2026-08-10. This design refines issue #15
with a deletion-first implementation while preserving its load-bearing behavior.

## Objective

Replace the current broad implementation with three deeper modules:

1. one small passage-chunking interface backed by LangChain and spaCy;
2. one compact deterministic reranking function and immutable result model;
3. one thin Runpod Flash adapter plus one static self-contained worker.

The refactor must materially delete production and test code. Moving the same
logic to new files does not count. Across `passages.py`, `reranking.py`,
`flash_passage_scorer.py`, and `flash_worker.py`, production lines should fall
by at least 30%, with a stretch target of 40%.

## Preserved invariants

- No selected source text is lost. The union of passage intervals covers the
  complete selected representation.
- Normal chunks contain only complete adjacent sentences. A sentence may be
  split internally only when that sentence alone exceeds the model budget.
- Overlap is allowed and contains complete sentence units wherever LangChain
  can satisfy the configured approximate token budget.
- Every passage is an exact source slice with authoritative start and exclusive
  end character offsets.
- Passage content hashes are stable SHA-256 bytes over exact UTF-8 text.
- Exact formatted query/passage token length remains the final authority before
  scoring; splitter token accounting is only a packing estimate.
- Overlapping evidence is deduplicated by
  `(document_id, start_char, end_char)`. Content hashes are not positional:
  identical text at distinct offsets remains distinct evidence while sharing a
  query score safely.
- Score cache values remain complete validated `dict[bytes, float]` mappings per
  query through the future `ArtifactStoreLike` seam.
- Document scores use the existing maximum-passage aggregation. Ordering is
  deterministic: descending reranker score, then organizer rank, then document
  ID. Evidence ordering is deterministic by source offsets.
- Local scoring remains the primary path. No Runpod deployment or paid call is
  part of this refactor.

## 1. Passage chunking module

### External interface

Keep one simple class rather than separate public configuration and protocol
types:

```python
class PassageChunker:
    def __init__(
        self,
        pair_length: CrossEncoderPairLengthFunction,
        *,
        overlap_tokens: int = 64,
    ) -> None: ...

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

`Passage` becomes a compact frozen, slotted structure containing only fields
used by scoring, caching, and reranking:

- document ID;
- selected representation and language;
- exact text, start, and end offsets;
- stable binary content hash;
- boundary kind (`sentence` or `oversized_sentence`), if retained for evidence.

Delete source/representation hashes, neighbor passage IDs, overlap counters,
chunker IDs, and related validation unless a direct caller or persisted schema
requires them. Chunk identity is the positional tuple
`(document_id, start_char, end_char)`; content identity is the SHA-256 hash.

### Normal path

Use the installed `langchain-text-splitters` `SpacyTextSplitter` with:

- the pinned language-specific spaCy pipeline;
- `chunk_size=pair_length.max_passage_length`;
- `chunk_overlap=overlap_tokens`;
- `length_function=pair_length.passage_length_function`;
- `separator=""`;
- `strip_whitespace=False`;
- `add_start_index=True`.

The installed combination was probed before this design. It retained exact
source slices and start-index metadata for representative English, Spanish,
Russian, Chinese, repeated-identical-sentence, leading/trailing-whitespace, and
complete-sentence-overlap cases. When no complete prior sentence fits the
overlap budget, it emits no overlap rather than splitting a sentence.

For every returned document, verify only the load-bearing properties:

1. `start_index` is an integer and the chunk equals the corresponding exact
   source slice;
2. the union of intervals covers the selected text;
3. exact full-pair token length is within the scorer limit.

If a pinned LangChain/spaCy update breaks authoritative offsets, replace only
offset extraction with a small spaCy `doc.sents` span adapter. Do not add a
custom sentence parser or general packing framework.

### Oversized sentence path

If an individual spaCy sentence exceeds the full-pair ceiling, pass only that
sentence to `RecursiveCharacterTextSplitter`. Its returned fragments must be
mapped back to exact parent offsets and revalidated with the exact pair length.
No multi-sentence chunk may use this fallback.

### Fingerprints

Use one stable chunker fingerprint derived from the model/tokenizer pair-length
fingerprint, overlap budget, LangChain version, spaCy version, and selected
pipeline version. Remove caller namespaces and interchangeable configuration
machinery that has no second production adapter.

## 2. Reranking module

### External interface

Retain one function:

```python
def rerank_retrieval(
    retrieval: RetrievalResult,
    *,
    chunker: PassageChunker,
    scorer: PassageScorer,
    representation: Literal["canonical", "report_text"],
) -> RerankingResult: ...
```

Replace the current large defensive models with compact frozen structures:

- `PassageEvidence(document_id, start_char, end_char, score, content_hash)`;
- `RerankedDocument(document, score, evidence, reranked_rank, status)`;
- `RerankingResult(query, representation, documents)`.

Use construction-time normalization in `rerank_retrieval`; do not make every
result object recursively revalidate the entire graph in `__post_init__`.

### Data flow

1. Select canonical or English report text for each hydrated document.
2. Chunk selected text and gather passages.
3. Deduplicate scoring work by content hash because equal query/text pairs have
   equal scores.
4. Ask the scorer once for the complete unique passage set; retain strict score
   validation and cache behavior.
5. Reattach scores to positional passages.
6. Deduplicate only overlapping duplicate evidence by positional identity.
7. Score each document by its maximum passage score and sort deterministically.

Delete the standalone filesystem JSONL publication framework. Persistence is a
separate concern owned by `ArtifactStoreLike` after issues #12/#13 land. The
reranker returns data; it does not implement cross-platform atomic file writes.

### Tests

Keep interface-level tests for aggregation, ties, unresolved/missing
representations, repeated identical sentences at distinct offsets, overlap
deduplication, warm score-cache reuse, and multilingual integration. Delete
tests for private helpers, recursive result-model forgeries, and platform file
publication internals removed by this design.

## 3. Runpod Flash modules

### Thin endpoint adapter

`RunpodFlashPassageScorer` remains a `PassageScorer` adapter. It owns only:

- bounded request batching;
- bytes/hex wire conversion;
- invocation of an injected endpoint runner;
- complete response validation;
- minimal sync/async bridging needed by the synchronous scorer interface.

Delete dynamic generated-class source, source digests, class/module caches, and
duplicated endpoint orchestration.

### Static self-contained worker

Define one ordinary static `FlashPassageWorker` class whose body contains every
helper required by Runpod Flash class-only transport. It imports only the
standard library and declared pinned third-party dependencies.

The request includes the scorer model configuration and fingerprint. On the
first request, the worker validates the configuration, loads the model and
explicit tokenizer once, and retains that model. Later requests must use the
same fingerprint or fail closed. This removes the need to generate a new class
whose source embeds each configuration.

The static worker must:

- run directly with injected fake model/tokenizer loaders;
- accept Transformers `BatchEncoding` mappings;
- enforce exact full-pair token limits;
- return complete finite score mappings;
- remain decoratable by Runpod Flash with pinned dependencies, GPU, worker
  count, and idle timeout;
- perform no provisioning merely by importing or constructing the local fake.

Tests extract and execute only the static class body with project imports
blocked, matching Runpod's class-only transport, and also exercise local fake
execution. No deployment or paid endpoint call is authorized.

## Error handling

Use one domain exception per module. Validate data where it crosses an actual
seam: splitter output, scorer output, cache artifact decode, and Flash wire
input/output. Delete repeated validations of immutable objects created by our
own code.

All external failures propagate with query text, credentials, and bearer tokens
excluded from messages and logs.

## Verification and acceptance

- Decrypt root `.env` before every test/application run.
- Focused passage, scorer, cache, reranking, Flash, and integration tests pass.
- Complete repository tests pass.
- Multilingual fixture benchmark proves exact slices, full source coverage,
  sentence-only overlap, deterministic offsets/hashes, and oversized-sentence
  fallback.
- A pinned-tokenizer-only contiguous-CJK smoke remains within the exact pair
  ceiling without model-weight inference.
- Lock, Ruff, formatting, diff, secret, ignored-environment, and tracked-artifact
  checks pass.
- Production and test code show material net deletion; internal-only tests are
  removed rather than mechanically rewritten.
- Sol reviews the combined diff for requirement preservation and accidental
  reintroduction of shallow interfaces.
- Nothing is pushed, no pull request is opened, and nothing is deployed.

## Rejected alternatives

- **Overlap then discard partial sentences:** rejected because it can lose
  source text and complicates offset lineage.
- **Keep the current generic configuration and validation framework:** rejected
  because it is the main source of shallow interface and test growth.
- **Move filesystem artifact publication to another reranking file:** rejected
  because that moves code rather than deleting it; ArtifactStore owns
  persistence.
- **Ship the local package to Flash workers:** rejected for this branch because
  the package/commit is not an independently installable pinned worker
  dependency. The static worker remains self-contained.
