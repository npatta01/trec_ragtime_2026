"""Durable per-topic report artifacts keyed by exact generation identity.

This is ``retrieval_cache`` re-instantiated for the generation domain: the
adapter owns semantic identity, a pinned MessagePack wire form, a canonical
round-trip check, and one ``validate_report`` gate applied on both write and
decode. ``ArtifactStoreLike`` owns lookup and transport.

Identity splits in two, exactly as the retrieval cache splits it. The
*generator fingerprint* covers everything about how a report is produced --
schema and official-format version, provider and model, prompt and output
schema, sampling parameters and seed, segmentation, citation policy, and length
packing. The *request digest* covers what is being produced: the exact topic
request, its character limit, and the sealed evidence bundle. ``team_id`` and
``run_id`` are deliberately outside both, because re-tagging a run must not
invalidate reports that are already generated and validated.

Only a complete report that passes every check is cacheable. Refusals, truncated
completions, malformed output, unknown handles, uncited substantive sentences,
and over-limit reports all raise before a write is attempted. Dropped-sentence
diagnostics keep a reason code and a digest, never the text, so an artifact can
never become an evidence dump.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import msgspec

from .artifact_store import (
    ArtifactStoreLike,
    ArtifactStoreRemoteError,
    ArtifactValidationError,
)
from .evidence_bundle import EvidenceBundle, UnknownEvidenceHandleError, topic_digest
from .report_sentences import (
    LENGTH_PACKING_VERSION,
    SENTENCE_SEGMENTER_VERSION,
    SentenceSegmenter,
    budgeted_character_count,
    default_segmenter,
    normalize_report_text,
    report_character_count,
)
from .report_submission import SUBMISSION_FORMAT_VERSION
from .topic_candidates import Topic

REPORT_SCHEMA_VERSION = 1
REPORT_OBJECT_KEY_PREFIX = "reports/v1"
OFFICIAL_FORMAT_VERSION = SUBMISSION_FORMAT_VERSION

PROMPT_VERSION = "report-prompt-v1"
OUTPUT_SCHEMA_VERSION = "report-sentences-v1"
CITATION_POLICY_VERSION = "citation-rank-v1"
GENERATOR_BEHAVIOR_VERSION = "report-generator-v1"

RESPONSE_FORMAT_MODES = frozenset({"json_schema", "json_object"})

# Closed set of reasons a candidate sentence may be dropped or repaired.
DROP_REASONS = frozenset(
    {
        "unknown_handle",
        "not_single_sentence",
        "missing_citation",
        "empty_text",
        "duplicate_text",
        "not_nfkc",
        "over_limit_pack",
        "citation_cap",
        "unsupported_claim",
        "handle_mismatch",
    }
)


class ReportValidationError(ValueError):
    """A report violates the domain contract and must not be cached."""


def _digest(value: Any, *, name: str) -> bytes:
    if isinstance(value, str):
        try:
            value = bytes.fromhex(value)
        except ValueError:
            raise ValueError(f"{name} must be a 32-byte digest") from None
    if not isinstance(value, (bytes, bytearray)) or len(value) != 32:
        raise ValueError(f"{name} must be a 32-byte digest")
    return bytes(value)


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _normalize_base_url(base_url: str) -> str:
    return _text(base_url, name="base_url").rstrip("/")


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    """Everything about *how* a report is produced, in fingerprint form."""

    base_url: str
    model: str
    temperature: float
    top_p: float | None
    max_output_tokens: int
    seed: int | None
    max_repairs: int
    max_citations: int
    max_passages: int
    evidence_chars: int
    budget_ratio: float
    response_format_mode: str = "json_schema"
    prompt_version: str = PROMPT_VERSION
    output_schema_version: str = OUTPUT_SCHEMA_VERSION
    segmenter_version: str = SENTENCE_SEGMENTER_VERSION
    citation_policy_version: str = CITATION_POLICY_VERSION
    length_packing_version: str = LENGTH_PACKING_VERSION
    generator_behavior_version: str = GENERATOR_BEHAVIOR_VERSION

    def __post_init__(self) -> None:
        _normalize_base_url(self.base_url)
        _text(self.model, name="model")
        _finite(self.temperature, name="temperature")
        if self.top_p is not None:
            _finite(self.top_p, name="top_p")
        for value, name in (
            (self.max_output_tokens, "max_output_tokens"),
            (self.max_citations, "max_citations"),
            (self.max_passages, "max_passages"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for value, name in (
            (self.max_repairs, "max_repairs"),
            (self.evidence_chars, "evidence_chars"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise ValueError("seed must be an integer or None")
        ratio = _finite(self.budget_ratio, name="budget_ratio")
        if not 0.0 < ratio <= 1.0:
            raise ValueError("budget_ratio must be in (0, 1]")
        if self.response_format_mode not in RESPONSE_FORMAT_MODES:
            raise ValueError(
                f"response_format_mode must be one of {sorted(RESPONSE_FORMAT_MODES)}"
            )
        for value, name in (
            (self.prompt_version, "prompt_version"),
            (self.output_schema_version, "output_schema_version"),
            (self.segmenter_version, "segmenter_version"),
            (self.citation_policy_version, "citation_policy_version"),
            (self.length_packing_version, "length_packing_version"),
            (self.generator_behavior_version, "generator_behavior_version"),
        ):
            _text(value, name=name)

    @property
    def fingerprint(self) -> bytes:
        return generator_fingerprint(self)


def generator_fingerprint(settings: GenerationSettings) -> bytes:
    """Return the binary fingerprint of the generation semantics."""

    if not isinstance(settings, GenerationSettings):
        raise TypeError("settings must be a GenerationSettings")
    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "official_format_version": OFFICIAL_FORMAT_VERSION,
        "base_url_digest": hashlib.sha256(
            _normalize_base_url(settings.base_url).encode("utf-8")
        ).digest(),
        "model": settings.model,
        "prompt_version": settings.prompt_version,
        "output_schema_version": settings.output_schema_version,
        "response_format_mode": settings.response_format_mode,
        "sampling": {
            "temperature": float(settings.temperature),
            "top_p": None if settings.top_p is None else float(settings.top_p),
            "max_output_tokens": settings.max_output_tokens,
            "seed": settings.seed,
        },
        "max_repairs": settings.max_repairs,
        "max_citations": settings.max_citations,
        "max_evidence_passages": settings.max_passages,
        "evidence_chars": settings.evidence_chars,
        "budget_ratio": float(settings.budget_ratio),
        "segmenter_version": settings.segmenter_version,
        "citation_policy_version": settings.citation_policy_version,
        "length_packing_version": settings.length_packing_version,
        "generator_behavior_version": settings.generator_behavior_version,
    }
    return hashlib.sha256(msgspec.msgpack.encode(payload)).digest()


def report_object_key(
    fingerprint: "ReportRequestIdentity | bytes | bytearray | str",
    request_digest: bytes | bytearray | str | None = None,
) -> str:
    """Build ``reports/v1/<fingerprint>/<prefix>/<request>.mpk``."""

    if isinstance(fingerprint, ReportRequestIdentity):
        if request_digest is not None:
            raise TypeError("request_digest is not accepted with a request identity")
        fingerprint_bytes = fingerprint.generator_fingerprint
        request_bytes = fingerprint.request_digest
    else:
        fingerprint_bytes = _digest(fingerprint, name="generator fingerprint")
        request_bytes = _digest(request_digest, name="request digest")
    request_hex = request_bytes.hex()
    return (
        f"{REPORT_OBJECT_KEY_PREFIX}/{fingerprint_bytes.hex()}/"
        f"{request_hex[:2]}/{request_hex}.mpk"
    )


@dataclass(frozen=True, slots=True)
class ReportRequestIdentity:
    """The exact semantic inputs that produce one validated report."""

    topic_id: str
    topic_digest: bytes
    character_limit: int
    bundle_seal: bytes
    settings: GenerationSettings

    def __post_init__(self) -> None:
        _text(self.topic_id, name="topic_id")
        _digest(self.topic_digest, name="topic digest")
        _digest(self.bundle_seal, name="bundle seal")
        if (
            isinstance(self.character_limit, bool)
            or not isinstance(self.character_limit, int)
            or self.character_limit < 1
        ):
            raise ValueError("character_limit must be a positive integer")
        if not isinstance(self.settings, GenerationSettings):
            raise TypeError("settings must be a GenerationSettings")

    @classmethod
    def for_bundle(
        cls,
        bundle: EvidenceBundle,
        *,
        topic: Topic,
        settings: GenerationSettings,
    ) -> "ReportRequestIdentity":
        """Build the identity for one sealed bundle and its topic request."""

        if not isinstance(bundle, EvidenceBundle):
            raise TypeError("bundle must be an EvidenceBundle")
        if bundle.topic_id != topic.topic_id:
            raise ValueError("bundle and topic identify different topics")
        if bundle.character_limit != topic.limit:
            raise ValueError("bundle and topic disagree about the character limit")
        return cls(
            topic_id=bundle.topic_id,
            topic_digest=topic_digest(topic),
            character_limit=bundle.character_limit,
            bundle_seal=bundle.seal_bytes,
            settings=settings,
        )

    @property
    def generator_fingerprint(self) -> bytes:
        return generator_fingerprint(self.settings)

    @property
    def canonical_request_bytes(self) -> bytes:
        return msgspec.msgpack.encode(
            {
                "schema_version": REPORT_SCHEMA_VERSION,
                "topic_id_utf8": self.topic_id.encode("utf-8"),
                "topic_digest": self.topic_digest,
                "character_limit": self.character_limit,
                "bundle_seal": self.bundle_seal,
            }
        )

    @property
    def request_digest(self) -> bytes:
        return hashlib.sha256(self.canonical_request_bytes).digest()

    @property
    def object_key(self) -> str:
        return report_object_key(self.generator_fingerprint, self.request_digest)


@dataclass(frozen=True, slots=True)
class ReportArtifactReference:
    """Small immutable provenance record safe for runner checkpoints."""

    object_key: str
    request_digest: bytes
    generator_fingerprint: bytes
    stored_at: float
    cache_hit: bool

    def __post_init__(self) -> None:
        request_digest = _digest(self.request_digest, name="request digest")
        fingerprint = _digest(self.generator_fingerprint, name="generator fingerprint")
        _finite(self.stored_at, name="stored_at")
        if not isinstance(self.cache_hit, bool):
            raise TypeError("cache_hit must be a boolean")
        if not isinstance(self.object_key, str) or not self.object_key:
            raise ValueError("object_key must be a non-empty string")
        if self.object_key != report_object_key(fingerprint, request_digest):
            raise ValueError("object_key does not match report identity")

    def as_record(self) -> dict[str, object]:
        return {
            "object_key": self.object_key,
            "request_digest": self.request_digest.hex(),
            "generator_fingerprint": self.generator_fingerprint.hex(),
            "stored_at": float(self.stored_at),
            "cache_hit": self.cache_hit,
        }


@dataclass(frozen=True, slots=True)
class ReportCitation:
    """One document supporting an entire sentence, with its priority score."""

    document_id: str
    handle: str
    score: float


@dataclass(frozen=True, slots=True)
class ReportSentence:
    """One report sentence in report order."""

    index: int
    text: str
    substantive: bool
    citations: tuple[ReportCitation, ...]
    facet_ids: tuple[str, ...]

    def citation_map(self) -> dict[str, float]:
        return {citation.document_id: citation.score for citation in self.citations}


@dataclass(frozen=True, slots=True)
class DroppedSentence:
    """A removed candidate, recorded as a digest so no text is retained."""

    index: int
    reason: str
    text_sha256: str
    character_count: int

    def __post_init__(self) -> None:
        if self.reason not in DROP_REASONS:
            raise ValueError(f"drop reason must be one of {sorted(DROP_REASONS)}")

    @classmethod
    def create(cls, *, index: int, reason: str, text: str) -> "DroppedSentence":
        return cls(
            index=index,
            reason=reason,
            text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            character_count=len(text),
        )


@dataclass(frozen=True, slots=True)
class ReportDiagnostics:
    """Sanitized generation telemetry; never prompts or evidence."""

    model_id: str
    repair_attempts: int
    latency_seconds: float
    usage: Mapping[str, int] | None
    dropped: tuple[DroppedSentence, ...]
    llm_cache_hit: bool


@dataclass(frozen=True, slots=True)
class TopicReport:
    """One validated report for one topic."""

    schema_version: int
    topic_id: str
    character_limit: int
    normalized_character_count: int
    budgeted_character_count: int
    sentences: tuple[ReportSentence, ...]
    references: tuple[str, ...]
    bundle_seal: bytes
    topic_digest: bytes
    generator_fingerprint: bytes
    created_at: str
    diagnostics: ReportDiagnostics
    report_artifact: Any | None = field(default=None, repr=False, compare=False)

    @classmethod
    def create(
        cls,
        *,
        request: ReportRequestIdentity,
        sentences: Sequence[ReportSentence],
        created_at: str,
        diagnostics: ReportDiagnostics,
    ) -> "TopicReport":
        """Derive references and character counts so they cannot disagree."""

        sentences = tuple(sentences)
        texts = [item.text for item in sentences]
        references = sorted(
            {citation.document_id for item in sentences for citation in item.citations}
        )
        return cls(
            schema_version=REPORT_SCHEMA_VERSION,
            topic_id=request.topic_id,
            character_limit=request.character_limit,
            normalized_character_count=report_character_count(texts),
            budgeted_character_count=budgeted_character_count(texts),
            sentences=sentences,
            references=tuple(references),
            bundle_seal=request.bundle_seal,
            topic_digest=request.topic_digest,
            generator_fingerprint=request.generator_fingerprint,
            created_at=created_at,
            diagnostics=diagnostics,
        )

    def sentence_payloads(self) -> list[dict[str, Any]]:
        """Return sentence dictionaries for ``report_submission``."""

        return [
            {"text": item.text, "citations": item.citation_map()} for item in self.sentences
        ]


class ReportCitationWire(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    document_id: str
    handle: str
    score: float


class ReportSentenceWire(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    index: int
    text: str
    substantive: bool
    citations: tuple[ReportCitationWire, ...]
    facet_ids: tuple[str, ...]


class DroppedSentenceWire(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    index: int
    reason: str
    text_sha256: str
    character_count: int


class ReportDiagnosticsWire(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    model_id: str
    repair_attempts: int
    latency_seconds: float
    usage: dict[str, int] | None
    dropped: tuple[DroppedSentenceWire, ...]
    llm_cache_hit: bool


class TopicReportWire(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    schema_version: int
    topic_id: str
    character_limit: int
    normalized_character_count: int
    budgeted_character_count: int
    sentences: tuple[ReportSentenceWire, ...]
    references: tuple[str, ...]
    bundle_seal: bytes
    topic_digest: bytes
    generator_fingerprint: bytes
    created_at: str
    diagnostics: ReportDiagnosticsWire


class ReportEnvelope(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    schema_version: int
    generator_fingerprint: bytes
    request_digest: bytes
    stored_at: float
    report: TopicReportWire


def _to_wire(report: TopicReport) -> TopicReportWire:
    return TopicReportWire(
        schema_version=report.schema_version,
        topic_id=report.topic_id,
        character_limit=report.character_limit,
        normalized_character_count=report.normalized_character_count,
        budgeted_character_count=report.budgeted_character_count,
        sentences=tuple(
            ReportSentenceWire(
                index=item.index,
                text=item.text,
                substantive=item.substantive,
                citations=tuple(
                    ReportCitationWire(
                        document_id=citation.document_id,
                        handle=citation.handle,
                        score=float(citation.score),
                    )
                    for citation in item.citations
                ),
                facet_ids=tuple(item.facet_ids),
            )
            for item in report.sentences
        ),
        references=tuple(report.references),
        bundle_seal=report.bundle_seal,
        topic_digest=report.topic_digest,
        generator_fingerprint=report.generator_fingerprint,
        created_at=report.created_at,
        diagnostics=ReportDiagnosticsWire(
            model_id=report.diagnostics.model_id,
            repair_attempts=report.diagnostics.repair_attempts,
            latency_seconds=float(report.diagnostics.latency_seconds),
            usage=None if report.diagnostics.usage is None else dict(report.diagnostics.usage),
            dropped=tuple(
                DroppedSentenceWire(
                    index=item.index,
                    reason=item.reason,
                    text_sha256=item.text_sha256,
                    character_count=item.character_count,
                )
                for item in report.diagnostics.dropped
            ),
            llm_cache_hit=report.diagnostics.llm_cache_hit,
        ),
    )


def _from_wire(wire: TopicReportWire) -> TopicReport:
    return TopicReport(
        schema_version=wire.schema_version,
        topic_id=wire.topic_id,
        character_limit=wire.character_limit,
        normalized_character_count=wire.normalized_character_count,
        budgeted_character_count=wire.budgeted_character_count,
        sentences=tuple(
            ReportSentence(
                index=item.index,
                text=item.text,
                substantive=item.substantive,
                citations=tuple(
                    ReportCitation(
                        document_id=citation.document_id,
                        handle=citation.handle,
                        score=citation.score,
                    )
                    for citation in item.citations
                ),
                facet_ids=tuple(item.facet_ids),
            )
            for item in wire.sentences
        ),
        references=tuple(wire.references),
        bundle_seal=wire.bundle_seal,
        topic_digest=wire.topic_digest,
        generator_fingerprint=wire.generator_fingerprint,
        created_at=wire.created_at,
        diagnostics=ReportDiagnostics(
            model_id=wire.diagnostics.model_id,
            repair_attempts=wire.diagnostics.repair_attempts,
            latency_seconds=wire.diagnostics.latency_seconds,
            usage=None if wire.diagnostics.usage is None else dict(wire.diagnostics.usage),
            dropped=tuple(
                DroppedSentence(
                    index=item.index,
                    reason=item.reason,
                    text_sha256=item.text_sha256,
                    character_count=item.character_count,
                )
                for item in wire.diagnostics.dropped
            ),
            llm_cache_hit=wire.diagnostics.llm_cache_hit,
        ),
    )


def _canonical_envelope_bytes(envelope: ReportEnvelope) -> bytes:
    return msgspec.msgpack.encode(
        ReportEnvelope(
            schema_version=envelope.schema_version,
            generator_fingerprint=envelope.generator_fingerprint,
            request_digest=envelope.request_digest,
            stored_at=envelope.stored_at,
            report=_to_wire(_from_wire(envelope.report)),
        )
    )


def _finite_score(value: Any, *, position: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportValidationError(f"sentence {position} citation score must be a number")
    score = float(value)
    if not math.isfinite(score):
        raise ReportValidationError(f"sentence {position} citation score must be finite")
    return score


def _report_text(value: Any, *, name: str) -> str:
    """Like ``_text`` but raising the domain error callers actually catch."""

    try:
        return _text(value, name=name)
    except ValueError as error:
        raise ReportValidationError(str(error)) from None


def _report_finite(value: Any, *, name: str) -> float:
    try:
        return _finite(value, name=name)
    except ValueError as error:
        raise ReportValidationError(str(error)) from None


def validate_report(
    request: ReportRequestIdentity,
    report: TopicReport,
    *,
    bundle: EvidenceBundle,
    segmenter: SentenceSegmenter | None = None,
) -> None:
    """Raise unless ``report`` is a complete, cacheable success.

    Applied identically on write and on decode, so a stored artifact can never
    be weaker than a freshly generated one.
    """

    segmenter = segmenter or default_segmenter()
    if not isinstance(report, TopicReport):
        raise ReportValidationError("report must be a TopicReport")
    if bundle.seal_bytes != request.bundle_seal:
        raise ReportValidationError("bundle does not match the request seal")

    if report.schema_version != REPORT_SCHEMA_VERSION:
        raise ReportValidationError("unsupported report schema version")
    if report.topic_id != request.topic_id:
        raise ReportValidationError("report topic does not match the request")
    if report.character_limit != request.character_limit:
        raise ReportValidationError("report character limit does not match the request")
    if report.bundle_seal != request.bundle_seal:
        raise ReportValidationError("report bundle seal does not match the request")
    if report.topic_digest != request.topic_digest:
        raise ReportValidationError("report topic digest does not match the request")
    if report.generator_fingerprint != request.generator_fingerprint:
        raise ReportValidationError("report fingerprint does not match the request")
    _report_text(report.created_at, name="created_at")

    if not report.sentences:
        raise ReportValidationError("a cacheable report needs at least one sentence")

    max_citations = request.settings.max_citations
    seen_text: set[str] = set()
    for position, item in enumerate(report.sentences, start=1):
        if item.index != position:
            raise ReportValidationError("sentence indices must be contiguous from one")
        if not isinstance(item.text, str) or not item.text.strip():
            raise ReportValidationError(f"sentence {position} must have text")
        if normalize_report_text(item.text) != item.text:
            raise ReportValidationError(f"sentence {position} is not NFKC-normalized")
        if not segmenter.is_single_sentence(item.text):
            raise ReportValidationError(f"sentence {position} is not exactly one sentence")
        if item.text in seen_text:
            raise ReportValidationError(f"sentence {position} duplicates earlier text")
        seen_text.add(item.text)

        if not isinstance(item.substantive, bool):
            raise ReportValidationError(f"sentence {position} substantive must be boolean")
        if item.substantive and not item.citations:
            raise ReportValidationError(
                f"sentence {position} is substantive and needs at least one citation"
            )
        if not item.substantive and item.citations:
            raise ReportValidationError(
                f"sentence {position} is non-substantive and must not cite"
            )
        if len(item.citations) > max_citations:
            raise ReportValidationError(
                f"sentence {position} exceeds the {max_citations}-citation cap"
            )

        documents: list[str] = []
        scores: list[float] = []
        for citation in item.citations:
            try:
                resolved = bundle.resolve_handle(citation.handle)
            except UnknownEvidenceHandleError as error:
                raise ReportValidationError(
                    f"sentence {position} cites an unallowlisted handle: {error}"
                ) from None
            if resolved != citation.document_id:
                raise ReportValidationError(
                    f"sentence {position} handle {citation.handle} does not resolve to"
                    f" {citation.document_id}"
                )
            scores.append(_finite_score(citation.score, position=position))
            documents.append(citation.document_id)
        if len(set(documents)) != len(documents):
            raise ReportValidationError(f"sentence {position} cites a document twice")
        if any(later >= earlier for earlier, later in zip(scores, scores[1:])):
            raise ReportValidationError(
                f"sentence {position} citation scores must strictly descend"
            )

        for facet_id in item.facet_ids:
            if facet_id not in bundle.facet_ids:
                raise ReportValidationError(
                    f"sentence {position} references unknown facet {facet_id}"
                )

    expected_references = tuple(
        sorted({citation.document_id for item in report.sentences for citation in item.citations})
    )
    if report.references != expected_references:
        raise ReportValidationError("references must equal the union of citation keys")
    if not report.references:
        raise ReportValidationError("a cacheable report must cite at least one document")

    texts = [item.text for item in report.sentences]
    normalized = report_character_count(texts)
    budgeted = budgeted_character_count(texts)
    if report.normalized_character_count != normalized:
        raise ReportValidationError("recorded normalized character count is wrong")
    if report.budgeted_character_count != budgeted:
        raise ReportValidationError("recorded budgeted character count is wrong")
    if budgeted > report.character_limit:
        raise ReportValidationError(
            f"report is {budgeted} budgeted characters, limit is {report.character_limit}"
        )

    if not isinstance(report.diagnostics, ReportDiagnostics):
        raise ReportValidationError("diagnostics must be a ReportDiagnostics")
    _report_text(report.diagnostics.model_id, name="diagnostics.model_id")
    if (
        isinstance(report.diagnostics.repair_attempts, bool)
        or not isinstance(report.diagnostics.repair_attempts, int)
        or report.diagnostics.repair_attempts < 0
    ):
        raise ReportValidationError("repair_attempts must be a non-negative integer")
    _report_finite(report.diagnostics.latency_seconds, name="latency_seconds")
    for dropped in report.diagnostics.dropped:
        if dropped.reason not in DROP_REASONS:
            raise ReportValidationError(f"unknown drop reason {dropped.reason!r}")


@dataclass(frozen=True, slots=True)
class _DecodedArtifact:
    report: TopicReport
    stored_at: float


def _with_reference(
    report: TopicReport, reference: ReportArtifactReference | None
) -> TopicReport:
    return TopicReport(
        schema_version=report.schema_version,
        topic_id=report.topic_id,
        character_limit=report.character_limit,
        normalized_character_count=report.normalized_character_count,
        budgeted_character_count=report.budgeted_character_count,
        sentences=report.sentences,
        references=report.references,
        bundle_seal=report.bundle_seal,
        topic_digest=report.topic_digest,
        generator_fingerprint=report.generator_fingerprint,
        created_at=report.created_at,
        diagnostics=report.diagnostics,
        report_artifact=reference,
    )


class ReportArtifactCache:
    """Typed cache adapter for one validated report per generation identity."""

    def __init__(
        self,
        store: ArtifactStoreLike,
        *,
        clock: Callable[[], float] = time.time,
        segmenter: SentenceSegmenter | None = None,
    ) -> None:
        if not callable(getattr(store, "read_validated", None)) or not callable(
            getattr(store, "write", None)
        ):
            raise TypeError("store must implement read_validated and write")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.store = store
        self._clock = clock
        self._segmenter = segmenter or default_segmenter()

    def _decoder(
        self, request: ReportRequestIdentity, bundle: EvidenceBundle
    ) -> Callable[[bytes], _DecodedArtifact]:
        def decode(payload: bytes) -> _DecodedArtifact:
            try:
                envelope = msgspec.msgpack.decode(payload, type=ReportEnvelope)
            except (msgspec.DecodeError, msgspec.ValidationError, TypeError, ValueError):
                raise ArtifactValidationError("cached report envelope is invalid") from None
            try:
                if envelope.schema_version != REPORT_SCHEMA_VERSION:
                    raise ValueError("unsupported report schema version")
                if (
                    envelope.generator_fingerprint != request.generator_fingerprint
                    or envelope.request_digest != request.request_digest
                ):
                    raise ValueError("cached report identity is invalid")
                if _canonical_envelope_bytes(envelope) != payload:
                    raise ValueError("cached report wire encoding is not canonical")
                _finite(envelope.stored_at, name="stored_at")
                report = _from_wire(envelope.report)
                validate_report(request, report, bundle=bundle, segmenter=self._segmenter)
            except Exception:
                raise ArtifactValidationError("cached report payload is invalid") from None
            return _DecodedArtifact(report=report, stored_at=float(envelope.stored_at))

        return decode

    def get(
        self, request: ReportRequestIdentity, *, bundle: EvidenceBundle
    ) -> TopicReport | None:
        """Return a durable validated report, or ``None`` on any miss."""

        if not isinstance(request, ReportRequestIdentity):
            raise TypeError("request must be a ReportRequestIdentity")
        if bundle.seal_bytes != request.bundle_seal:
            raise ReportValidationError("bundle does not match the request seal")
        try:
            decoded = self.store.read_validated(
                request.object_key, self._decoder(request, bundle)
            )
        except (OSError, ArtifactStoreRemoteError):
            return None
        if decoded is None:
            return None
        if not isinstance(decoded, _DecodedArtifact):
            raise ArtifactValidationError("cached report decoder returned an invalid value")
        reference = ReportArtifactReference(
            object_key=request.object_key,
            request_digest=request.request_digest,
            generator_fingerprint=request.generator_fingerprint,
            stored_at=decoded.stored_at,
            cache_hit=True,
        )
        return _with_reference(decoded.report, reference)

    def put(
        self,
        request: ReportRequestIdentity,
        report: TopicReport,
        *,
        bundle: EvidenceBundle,
    ) -> TopicReport:
        """Validate then durably store one report; failures never write."""

        if not isinstance(request, ReportRequestIdentity):
            raise TypeError("request must be a ReportRequestIdentity")
        validate_report(request, report, bundle=bundle, segmenter=self._segmenter)
        stored_at = _finite(self._clock(), name="cache clock")
        envelope = ReportEnvelope(
            schema_version=REPORT_SCHEMA_VERSION,
            generator_fingerprint=request.generator_fingerprint,
            request_digest=request.request_digest,
            stored_at=stored_at,
            report=_to_wire(report),
        )
        try:
            stored = self.store.write(request.object_key, msgspec.msgpack.encode(envelope))
        except (OSError, ArtifactStoreRemoteError):
            return _with_reference(report, None)
        if stored is not True:
            return _with_reference(report, None)
        reference = ReportArtifactReference(
            object_key=request.object_key,
            request_digest=request.request_digest,
            generator_fingerprint=request.generator_fingerprint,
            stored_at=stored_at,
            cache_hit=False,
        )
        return _with_reference(report, reference)


__all__ = [
    "CITATION_POLICY_VERSION",
    "DROP_REASONS",
    "GENERATOR_BEHAVIOR_VERSION",
    "OFFICIAL_FORMAT_VERSION",
    "OUTPUT_SCHEMA_VERSION",
    "PROMPT_VERSION",
    "REPORT_OBJECT_KEY_PREFIX",
    "REPORT_SCHEMA_VERSION",
    "RESPONSE_FORMAT_MODES",
    "DroppedSentence",
    "GenerationSettings",
    "ReportArtifactCache",
    "ReportArtifactReference",
    "ReportCitation",
    "ReportDiagnostics",
    "ReportRequestIdentity",
    "ReportSentence",
    "ReportValidationError",
    "TopicReport",
    "generator_fingerprint",
    "report_object_key",
    "validate_report",
]
