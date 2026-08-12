# Failure-Tolerant Bulk Reranking

## Objective

Finish passage scoring for the 849 unique retrieval artifacts without allowing
an invalid document representation to terminate its query or an entire GPU
shard. Preserve all valid results in the existing private Hugging Face artifact
bucket and identify discarded document representations in the typed result.

## Scope

This change applies at the core reranking document boundary and to the
disposable bulk runner's output-wire validation. It does not weaken the passage
validators or silently repair malformed source text.

## Behavior

For each retrieval artifact, reranking attempts to chunk every selected document
representation independently. A `PassageChunkingError` discards only that
document's invalid passage set and assigns the document the explicit
`invalid_passages` status. Other documents in the query continue through cached
passage scoring, ranking, validation, and upload.

Only `PassageChunkingError` from the document chunking boundary is recoverable.
Unexpected chunker exceptions, invalid returned passage objects, artifact-store,
decoding, model, GPU, credential, and upload failures remain fatal so software,
infrastructure, or data-integrity problems cannot be mistaken for discarded
content.

The existing output artifact records `invalid_passages` beside the affected
document without storing raw exception details. The query remains completed and
auditable, while no invalid passage or fabricated score is emitted.

## Recovery Run

Resume worker 2 at manifest line 137 and worker 3 at line 72. Existing output
artifacts are skipped through the durable output cache; the four previously
failing retrieval objects are retried under document-level discard behavior,
and the remaining 164 unattempted objects are processed. The prior 681
successful outputs remain authoritative and are not recomputed.

The recovery uses the existing CUDA/bfloat16 scorer identity, private Hugging
Face bucket, four-worker cache namespace, A40/A6000 resource constraint,
`$0.60/GPU-hour` maximum price, and three-hour task limit.

## Verification

Tests must first demonstrate that a `PassageChunkingError` from one document
terminates the current query. After implementation they must prove that the
document receives `invalid_passages`, other documents are still scored, and an
unexpected chunker exception remains fatal. Output wire validation must accept
the new status. The complete reranking and runner test suites and dry-run
manifest selections must pass before dstack preview and submission.

Completion requires successful dstack summaries for both recovery manifests and
an explicit count of `invalid_passages` documents. No query is considered scored
merely because its retrieval artifact was read.
