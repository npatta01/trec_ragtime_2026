"""Self-contained selected retrieval evidence for report generation."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Literal

import msgspec

from .models import RetrievalResult, RetrievedDocument
from .nugget_cache import NuggetRequest, evidence_digest, nugget_object_key
from .nuggets import (
    MAX_CLAIM_CHARACTERS,
    MAX_CLAIM_WORDS,
    SCHEMA_VERSION as NUGGET_SCHEMA_VERSION,
    NuggetBundle,
    check_bundle,
    normalize_claim,
)
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

    if not isinstance(topic, Topic):
        raise TypeError("topic must be a Topic")
    if not isinstance(nugget_bundle, NuggetBundle):
        raise TypeError("nugget_bundle must be a NuggetBundle")
    if not isinstance(nugget_request, NuggetRequest):
        raise TypeError("nugget_request must be a NuggetRequest")
    if not isinstance(nugget_source, HandoffNuggetSource):
        raise TypeError("nugget_source must be a HandoffNuggetSource")
    if not isinstance(retrievals, Mapping):
        raise TypeError("retrievals must be a mapping")
    _validate_topic_input(topic)
    _validate_nugget_request(nugget_request)
    _validate_nugget_source(nugget_source)
    check_bundle(nugget_bundle)
    if topic.topic_id != nugget_bundle.topic_id or topic.topic_id != nugget_request.topic_id:
        raise HandoffError("topic identity disagrees with nugget artifacts")
    if nugget_bundle.query != nugget_request.query:
        raise HandoffError("nugget query disagrees with its request")
    if evidence_digest(nugget_bundle.evidence) != nugget_request.evidence_digest:
        raise HandoffError("nugget evidence disagrees with its request")
    if nugget_bundle.representation != REPRESENTATION or nugget_request.representation != REPRESENTATION:
        raise HandoffError("handoff requires report_text evidence")
    if (
        nugget_source.request_digest != nugget_request.digest
        or nugget_source.bundle_digest
        != hashlib.sha256(msgspec.msgpack.encode(nugget_bundle)).digest()
        or nugget_source.object_key
        != nugget_object_key(
            nugget_source.fingerprint_digest, nugget_source.request_digest
        )
    ):
        raise HandoffError("nugget source identity disagrees with its bundle")
    inputs = retrieval_input_records(nugget_request)
    if tuple((facet.facet_index, facet.query) for facet in nugget_bundle.facets) != tuple(
        (row.facet_index, row.query) for row in inputs
    ):
        raise HandoffError("nugget facets disagree with retrieval inputs")
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
        if not isinstance(retrieval, RetrievalResult):
            raise HandoffError("selected retrieval result has an invalid type")
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
        if len(matches) != 1 or not isinstance(matches[0], RetrievedDocument):
            raise HandoffError("selected document does not resolve exactly once")
        if row.representation != REPRESENTATION or row.language != "eng":
            raise HandoffError("selected evidence language or representation is invalid")
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
    _validate_handoff_schema(bundle)
    if bundle.schema_version != SCHEMA_VERSION or bundle.behavior_version != BEHAVIOR_VERSION:
        raise HandoffError("unsupported handoff version")
    if bundle.representation != REPRESENTATION:
        raise HandoffError("invalid topic or representation")
    _validate_topic_record(bundle.topic)
    if not bundle.nuggets or not bundle.evidence:
        raise HandoffError("handoff must contain nuggets and evidence")
    if not bundle.retrieval_inputs or not bundle.facets:
        raise HandoffError("handoff must retain retrieval facets")
    aliases = tuple(row.alias for row in bundle.evidence)
    if (
        len(set(aliases)) != len(aliases)
        or any(not _valid_alias(alias) for alias in aliases)
        or tuple(sorted(aliases, key=_alias_number)) != aliases
    ):
        raise HandoffError("evidence aliases must be unique and canonical")
    input_facets: dict[int, tuple[int, HandoffRetrievalInput]] = {}
    for index, row in enumerate(bundle.retrieval_inputs):
        if (
            isinstance(row.facet_index, bool)
            or row.facet_index < 0
            or not _nonblank(row.query)
            or not row.object_key
            or not isinstance(row.object_key, str)
            or not _is_digest(row.request_digest)
        ):
            raise HandoffError("invalid retrieval input")
        input_facets[row.facet_index] = (index, row)
    if len(input_facets) != len(bundle.retrieval_inputs):
        raise HandoffError("retrieval facets must be unique")
    input_order = tuple(row.facet_index for row in bundle.retrieval_inputs)
    if tuple(sorted(input_order)) != input_order:
        raise HandoffError("retrieval facets are not in deterministic order")
    facet_order = tuple(row.facet_index for row in bundle.facets)
    if facet_order != input_order:
        raise HandoffError("handoff facets do not match retrieval inputs")
    if len(set(facet_order)) != len(facet_order):
        raise HandoffError("handoff facets must be unique")
    evidence_by_facet: dict[int, set[str]] = {index: set() for index in input_order}
    for row in bundle.evidence:
        owner = input_facets.get(row.facet_index)
        if (
            owner is None
            or owner[0] != row.retrieval_input_index
            or owner[1].query != row.facet_query
            or row.representation != REPRESENTATION
            or row.language != "eng"
            or not _nonblank(row.document_id)
            or not row.text
            or not isinstance(row.start_char, int)
            or isinstance(row.start_char, bool)
            or row.start_char < 0
            or not isinstance(row.end_char, int)
            or isinstance(row.end_char, bool)
            or row.end_char <= row.start_char
            or row.end_char - row.start_char != len(row.text)
            or sha256_bytes(row.text) != row.content_hash
            or not math.isfinite(row.passage_score)
            or isinstance(row.document_rank, bool)
            or row.document_rank < 1
        ):
            raise HandoffError("invalid selected evidence record")
        evidence_by_facet[row.facet_index].add(row.alias)
    linked: set[str] = set()
    full = partial = 0
    nugget_ids: set[str] = set()
    cluster_ids: set[str] = set()
    scores = {row.alias: row.passage_score for row in bundle.evidence}
    for nugget in bundle.nuggets:
        nugget_aliases = tuple(link.alias for link in nugget.support)
        if (
            not nugget.support
            or not nugget.variants
            or not nugget.claim.strip()
            or len(nugget.claim) > MAX_CLAIM_CHARACTERS
            or len(nugget.claim.split()) > MAX_CLAIM_WORDS
            or tuple(sorted(set(nugget.variants))) != nugget.variants
            or any(
                not variant.strip()
                or variant != normalize_claim(variant)
                or len(variant) > MAX_CLAIM_CHARACTERS
                or len(variant.split()) > MAX_CLAIM_WORDS
                for variant in nugget.variants
            )
            or normalize_claim(nugget.claim) not in nugget.variants
            or nugget.nugget_id
            != _expected_nugget_id(bundle.topic.topic_id, nugget.claim, nugget_aliases)
            or nugget.cluster_id
            != _expected_cluster_id(bundle.topic.topic_id, nugget.variants)
            or nugget.nugget_id in nugget_ids
            or nugget.cluster_id in cluster_ids
            or len(set(nugget_aliases)) != len(nugget_aliases)
            or tuple(sorted(nugget_aliases)) != nugget_aliases
        ):
            raise HandoffError("nugget must retain variants and support")
        nugget_ids.add(nugget.nugget_id)
        cluster_ids.add(nugget.cluster_id)
        for link in nugget.support:
            if (
                link.alias not in aliases
                or link.assignment not in ("support", "partial_support")
                or link.variant_index < 0
                or isinstance(link.variant_index, bool)
                or link.variant_index >= len(nugget.variants)
            ):
                raise HandoffError("nugget support does not resolve")
            if link.citation_ready != (link.assignment == "support"):
                raise HandoffError("citation readiness contradicts support policy")
            linked.add(link.alias)
            full += link.assignment == "support"
            partial += link.assignment == "partial_support"
    ordered = sorted(
        bundle.nuggets,
        key=lambda nugget: (
            0 if nugget.importance == "vital" else 1,
            -max(scores[link.alias] for link in nugget.support),
            nugget.nugget_id,
        ),
    )
    if list(bundle.nuggets) != ordered:
        raise HandoffError("nuggets are not in deterministic order")
    if linked != set(aliases):
        raise HandoffError("evidence table is not the selected alias union")
    for facet in bundle.facets:
        owner = input_facets[facet.facet_index]
        selected_for_facet = evidence_by_facet[facet.facet_index]
        if (
            owner[1].query != facet.query
            or facet.evidence_count < 0
            or facet.nugget_count < 0
            or len(selected_for_facet) > facet.evidence_count
            or facet.nugget_count
            != sum(
                1
                for nugget in bundle.nuggets
                if any(link.alias in selected_for_facet for link in nugget.support)
            )
        ):
            raise HandoffError("facet coverage is inconsistent")
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
    if (
        any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in (
                bundle.counts.offered_evidence,
                bundle.counts.selected_evidence,
                bundle.counts.omitted_evidence,
                bundle.counts.nuggets,
                bundle.counts.full_support,
                bundle.counts.partial_support,
                bundle.counts.citation_ready,
                bundle.counts.rejected,
            )
        )
        or bundle.counts.offered_evidence < len(bundle.evidence)
        or sum(facet.evidence_count for facet in bundle.facets)
        != bundle.counts.offered_evidence
        or bundle.counts != expected
    ):
        raise HandoffError("handoff counts are inconsistent")
    for value, name in (
        (bundle.nugget_source.fingerprint_digest, "nugget fingerprint"),
        (bundle.nugget_source.request_digest, "nugget request"),
        (bundle.nugget_source.bundle_digest, "nugget bundle"),
        *((row.request_digest, "retrieval request") for row in bundle.retrieval_inputs),
    ):
        if not _is_digest(value):
            raise HandoffError(f"{name} digest must be 32 bytes")
    if not _nugget_source_key_matches(bundle.nugget_source):
        raise HandoffError("nugget source object key is invalid")


def _valid_alias(alias: object) -> bool:
    return (
        isinstance(alias, str)
        and len(alias) >= 4
        and alias.startswith("e")
        and alias[1:].isdigit()
        and len(alias[1:]) >= 3
        and int(alias[1:]) >= 1
        and alias == f"e{int(alias[1:]):03d}"
    )


def _alias_number(alias: str) -> int:
    return int(alias[1:])


def _nonblank(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_digest(value: object) -> bool:
    return isinstance(value, bytes) and len(value) == hashlib.sha256().digest_size


def _validate_handoff_schema(bundle: RetrievalHandoffBundle) -> None:
    try:
        payload = msgspec.msgpack.encode(bundle)
        decoded = msgspec.msgpack.decode(payload, type=RetrievalHandoffBundle)
        if decoded != bundle or msgspec.msgpack.encode(decoded) != payload:
            raise ValueError("non-canonical handoff encoding")
    except (msgspec.DecodeError, msgspec.EncodeError, msgspec.ValidationError, TypeError, ValueError):
        raise HandoffError("handoff schema validation failed") from None


def _validate_topic_input(topic: Topic) -> None:
    if (
        not _nonblank(topic.topic_id)
        or not _nonblank(topic.collection_id)
        or not _nonblank(topic.problem_statement)
        or not _nonblank(topic.background)
        or (not isinstance(topic.limit, int) or isinstance(topic.limit, bool))
        or topic.limit < 1
        or (topic.title is not None and not _nonblank(topic.title))
    ):
        raise HandoffError("invalid topic record")


def _validate_topic_record(topic: HandoffTopic) -> None:
    if (
        not _nonblank(topic.topic_id)
        or not _nonblank(topic.collection_id)
        or (topic.title is not None and not _nonblank(topic.title))
        or not _nonblank(topic.problem_statement)
        or not _nonblank(topic.background)
        or not isinstance(topic.limit, int)
        or isinstance(topic.limit, bool)
        or topic.limit < 1
    ):
        raise HandoffError("invalid topic record")


def _validate_nugget_request(request: NuggetRequest) -> None:
    try:
        payload = msgspec.msgpack.encode(request)
        decoded = msgspec.msgpack.decode(payload, type=NuggetRequest)
        if decoded != request or msgspec.msgpack.encode(decoded) != payload:
            raise ValueError("non-canonical nugget request encoding")
    except (msgspec.DecodeError, msgspec.EncodeError, msgspec.ValidationError, TypeError, ValueError):
        raise HandoffError("nugget request schema validation failed") from None
    if (
        not _nonblank(request.topic_id)
        or not _nonblank(request.query)
        or request.representation != REPRESENTATION
        or not _nonblank(request.passage_source_fingerprint)
        or not request.inputs
        or not _is_digest(request.evidence_digest)
    ):
        raise HandoffError("invalid nugget request")
    seen: set[int] = set()
    previous = -1
    for row in request.inputs:
        if (
            isinstance(row.facet_index, bool)
            or not isinstance(row.facet_index, int)
            or row.facet_index < 0
            or row.facet_index in seen
            or row.facet_index < previous
            or not _nonblank(row.query)
            or not isinstance(row.object_key, str)
            or not row.object_key
            or not _is_digest(row.request_digest)
        ):
            raise HandoffError("invalid nugget retrieval input")
        seen.add(row.facet_index)
        previous = row.facet_index


def _validate_nugget_source(source: HandoffNuggetSource) -> None:
    if (
        not isinstance(source.object_key, str)
        or not source.object_key
        or not _is_digest(source.fingerprint_digest)
        or not _is_digest(source.request_digest)
        or not _is_digest(source.bundle_digest)
        or not _nugget_source_key_matches(source)
    ):
        raise HandoffError("invalid nugget source reference")


def _nugget_source_key_matches(source: HandoffNuggetSource) -> bool:
    try:
        return source.object_key == nugget_object_key(
            source.fingerprint_digest, source.request_digest
        )
    except (TypeError, ValueError):
        return False


def _expected_nugget_id(topic_id: str, claim: str, aliases: tuple[str, ...]) -> str:
    payload = msgspec.msgpack.encode(
        {
            "schema_version": NUGGET_SCHEMA_VERSION,
            "topic_id": topic_id,
            "claim": normalize_claim(claim),
            "aliases": sorted(aliases),
        }
    )
    return "nug1_" + hashlib.sha256(payload).hexdigest()[:24]


def _expected_cluster_id(topic_id: str, variants: tuple[str, ...]) -> str:
    payload = msgspec.msgpack.encode(
        {
            "schema_version": NUGGET_SCHEMA_VERSION,
            "topic_id": topic_id,
            "variants": sorted(variants),
        }
    )
    return "nugc1_" + hashlib.sha256(payload).hexdigest()[:24]


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
