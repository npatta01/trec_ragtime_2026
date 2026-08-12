"""Self-contained selected retrieval evidence for report generation."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Literal

import msgspec

from .models import RetrievalResult
from .nugget_cache import NuggetRequest
from .nuggets import NuggetBundle, check_bundle
from .passages import sha256_bytes
from .topic_candidates import Topic


SCHEMA_VERSION = 1
BEHAVIOR_VERSION = "retrieval-handoff-v1"
REPRESENTATION = "report_text"


class HandoffError(ValueError):
    """Raised when upstream artifacts cannot form a valid handoff."""


class HandoffTopic(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    topic_id: str
    collection_id: str
    title: str | None
    problem_statement: str
    background: str
    limit: int


class HandoffRetrievalInput(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    facet_index: int
    query: str
    object_key: str
    request_digest: bytes


class HandoffNuggetSource(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    object_key: str
    fingerprint_digest: bytes
    request_digest: bytes
    bundle_digest: bytes


class HandoffEvidence(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    alias: str
    document_id: str
    text: str
    representation: Literal["report_text"]
    language: str
    start_char: int
    end_char: int
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


class HandoffNugget(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    nugget_id: str
    cluster_id: str
    claim: str
    variants: tuple[str, ...]
    importance: Literal["vital", "okay"]
    support: tuple[HandoffSupport, ...]


class HandoffFacet(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    facet_index: int
    query: str
    evidence_count: int
    nugget_count: int


class HandoffCounts(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    offered_evidence: int
    selected_evidence: int
    omitted_evidence: int
    nuggets: int
    full_support: int
    partial_support: int
    citation_ready: int
    rejected: int


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


def topic_record(topic: Topic) -> HandoffTopic:
    if not isinstance(topic, Topic):
        raise TypeError("topic must be a Topic")
    return HandoffTopic(
        topic_id=topic.topic_id,
        collection_id=topic.collection_id,
        title=topic.title,
        problem_statement=topic.problem_statement,
        background=topic.background,
        limit=topic.limit,
    )


def retrieval_input_records(request: NuggetRequest) -> tuple[HandoffRetrievalInput, ...]:
    return tuple(
        HandoffRetrievalInput(
            facet_index=row.facet_index,
            query=row.query,
            object_key=row.object_key,
            request_digest=row.request_digest,
        )
        for row in request.inputs
    )


def project_retrieval_handoff(
    *,
    topic: Topic,
    nugget_bundle: NuggetBundle,
    nugget_request: NuggetRequest,
    nugget_source: HandoffNuggetSource,
    retrievals: Mapping[int, RetrievalResult],
) -> RetrievalHandoffBundle:
    """Copy exactly the evidence linked by retained nuggets into one bundle."""

    if not isinstance(nugget_bundle, NuggetBundle):
        raise TypeError("nugget_bundle must be a NuggetBundle")
    check_bundle(nugget_bundle)
    if topic.topic_id != nugget_bundle.topic_id or topic.topic_id != nugget_request.topic_id:
        raise HandoffError("topic identity disagrees with nugget artifacts")
    if nugget_bundle.query != nugget_request.query:
        raise HandoffError("nugget query disagrees with its request")
    if nugget_bundle.representation != REPRESENTATION or nugget_request.representation != REPRESENTATION:
        raise HandoffError("handoff requires report_text evidence")
    inputs = retrieval_input_records(nugget_request)
    input_by_facet = {row.facet_index: (index, row) for index, row in enumerate(inputs)}
    if len(input_by_facet) != len(inputs):
        raise HandoffError("retrieval facet inputs are duplicated")

    selected_aliases = {
        link.alias for nugget in nugget_bundle.nuggets for link in nugget.support
    }
    if not nugget_bundle.nuggets or not selected_aliases:
        raise HandoffError("handoff requires retained nuggets and selected evidence")

    documents_seen: dict[str, str] = {}
    evidence: list[HandoffEvidence] = []
    for row in nugget_bundle.evidence:
        if row.alias not in selected_aliases:
            continue
        owner = input_by_facet.get(row.facet_index)
        retrieval = retrievals.get(row.facet_index)
        if owner is None or retrieval is None:
            raise HandoffError("selected evidence has no retrieval input")
        input_index, source = owner
        if retrieval.query != source.query:
            raise HandoffError("retrieval query disagrees with its input")
        reference = retrieval.retrieval_artifact
        if (
            reference is None
            or getattr(reference, "object_key", None) != source.object_key
            or getattr(reference, "request_digest", None) != source.request_digest
        ):
            raise HandoffError("retrieval artifact identity disagrees with nugget input")
        matches = tuple(doc for doc in retrieval.documents if doc.document_id == row.document_id)
        if len(matches) != 1:
            raise HandoffError("selected document does not resolve exactly once")
        text = matches[0].report_text
        if not isinstance(text, str) or not text:
            raise HandoffError("selected document has no report_text")
        previous = documents_seen.setdefault(row.document_id, text)
        if previous != text:
            raise HandoffError("same document disagrees across retrieval inputs")
        if row.start_char < 0 or row.end_char > len(text) or row.start_char >= row.end_char:
            raise HandoffError("selected evidence offsets are invalid")
        passage = text[row.start_char : row.end_char]
        if sha256_bytes(passage) != row.content_hash:
            raise HandoffError("selected evidence content hash is invalid")
        evidence.append(
            HandoffEvidence(
                alias=row.alias,
                document_id=row.document_id,
                text=passage,
                representation=REPRESENTATION,
                language="eng",
                start_char=row.start_char,
                end_char=row.end_char,
                content_hash=row.content_hash,
                passage_score=float(row.passage_score),
                document_rank=row.document_rank,
                facet_index=row.facet_index,
                facet_query=source.query,
                retrieval_input_index=input_index,
            )
        )
    if {row.alias for row in evidence} != selected_aliases:
        raise HandoffError("support links do not close over selected evidence")

    nuggets = tuple(
        HandoffNugget(
            nugget_id=row.nugget_id,
            cluster_id=row.cluster_id,
            claim=row.claim,
            variants=row.variants,
            importance=row.importance,
            support=tuple(
                HandoffSupport(
                    alias=link.alias,
                    assignment=link.assignment,
                    variant_index=link.variant_index,
                    citation_ready=link.assignment == "support",
                )
                for link in row.support
            ),
        )
        for row in nugget_bundle.nuggets
    )
    facets = tuple(
        HandoffFacet(row.facet_index, row.query, row.evidence_count, row.nugget_count)
        for row in nugget_bundle.facets
    )
    full = sum(link.assignment == "support" for row in nuggets for link in row.support)
    partial = sum(link.assignment == "partial_support" for row in nuggets for link in row.support)
    bundle = RetrievalHandoffBundle(
        schema_version=SCHEMA_VERSION,
        behavior_version=BEHAVIOR_VERSION,
        representation=REPRESENTATION,
        topic=topic_record(topic),
        retrieval_inputs=inputs,
        nugget_source=nugget_source,
        evidence=tuple(evidence),
        nuggets=nuggets,
        facets=facets,
        counts=HandoffCounts(
            offered_evidence=len(nugget_bundle.evidence),
            selected_evidence=len(evidence),
            omitted_evidence=len(nugget_bundle.evidence) - len(evidence),
            nuggets=len(nuggets),
            full_support=full,
            partial_support=partial,
            citation_ready=full,
            rejected=sum(row.count for row in nugget_bundle.rejected),
        ),
    )
    check_retrieval_handoff(bundle)
    return bundle


def check_retrieval_handoff(bundle: RetrievalHandoffBundle) -> None:
    """Validate every invariant available from the self-contained object."""

    if not isinstance(bundle, RetrievalHandoffBundle):
        raise TypeError("bundle must be a RetrievalHandoffBundle")
    if bundle.schema_version != SCHEMA_VERSION or bundle.behavior_version != BEHAVIOR_VERSION:
        raise HandoffError("unsupported handoff version")
    if bundle.representation != REPRESENTATION or bundle.topic.limit < 1:
        raise HandoffError("invalid topic or representation")
    if not bundle.nuggets or not bundle.evidence:
        raise HandoffError("handoff must contain nuggets and evidence")
    aliases = tuple(row.alias for row in bundle.evidence)
    if len(set(aliases)) != len(aliases):
        raise HandoffError("evidence aliases must be unique")
    input_facets = {row.facet_index: (index, row) for index, row in enumerate(bundle.retrieval_inputs)}
    if len(input_facets) != len(bundle.retrieval_inputs):
        raise HandoffError("retrieval facets must be unique")
    for row in bundle.evidence:
        owner = input_facets.get(row.facet_index)
        if (
            owner is None
            or owner[0] != row.retrieval_input_index
            or owner[1].query != row.facet_query
            or row.representation != REPRESENTATION
            or row.language != "eng"
            or row.start_char < 0
            or row.end_char <= row.start_char
            or sha256_bytes(row.text) != row.content_hash
            or not math.isfinite(row.passage_score)
            or row.document_rank < 1
        ):
            raise HandoffError("invalid selected evidence record")
    linked: set[str] = set()
    full = partial = 0
    for nugget in bundle.nuggets:
        if not nugget.support or not nugget.variants:
            raise HandoffError("nugget must retain variants and support")
        for link in nugget.support:
            if link.alias not in aliases or link.variant_index >= len(nugget.variants):
                raise HandoffError("nugget support does not resolve")
            if link.citation_ready != (link.assignment == "support"):
                raise HandoffError("citation readiness contradicts support policy")
            linked.add(link.alias)
            full += link.assignment == "support"
            partial += link.assignment == "partial_support"
    if linked != set(aliases):
        raise HandoffError("evidence table is not the selected alias union")
    expected = HandoffCounts(
        offered_evidence=bundle.counts.offered_evidence,
        selected_evidence=len(bundle.evidence),
        omitted_evidence=bundle.counts.offered_evidence - len(bundle.evidence),
        nuggets=len(bundle.nuggets),
        full_support=full,
        partial_support=partial,
        citation_ready=full,
        rejected=bundle.counts.rejected,
    )
    if bundle.counts != expected or min(bundle.counts.offered_evidence, bundle.counts.rejected) < 0:
        raise HandoffError("handoff counts are inconsistent")
    for value, name in (
        (bundle.nugget_source.fingerprint_digest, "nugget fingerprint"),
        (bundle.nugget_source.request_digest, "nugget request"),
        (bundle.nugget_source.bundle_digest, "nugget bundle"),
        *((row.request_digest, "retrieval request") for row in bundle.retrieval_inputs),
    ):
        if not isinstance(value, bytes) or len(value) != 32:
            raise HandoffError(f"{name} digest must be 32 bytes")


def handoff_bundle_digest(bundle: RetrievalHandoffBundle) -> bytes:
    check_retrieval_handoff(bundle)
    return hashlib.sha256(msgspec.msgpack.encode(bundle)).digest()


__all__ = [
    "BEHAVIOR_VERSION",
    "HandoffCounts",
    "HandoffError",
    "HandoffEvidence",
    "HandoffFacet",
    "HandoffNugget",
    "HandoffNuggetSource",
    "HandoffRetrievalInput",
    "HandoffSupport",
    "HandoffTopic",
    "RetrievalHandoffBundle",
    "check_retrieval_handoff",
    "handoff_bundle_digest",
    "project_retrieval_handoff",
    "retrieval_input_records",
    "topic_record",
]
