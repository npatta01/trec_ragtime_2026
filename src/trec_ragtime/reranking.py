"""Deterministic passage reranking and immutable per-topic artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import ctypes
import errno
import msgspec
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Literal

from .models import RetrievalResult, RetrievedDocument, SearchHit
from .passage_scoring import PassageScorer, validate_passage_scores
from .passages import (
    Passage,
    PassageChunker,
    PassageValidationError,
    Representation,
    sha256_bytes,
    validate_passages,
)


max_passage_v1 = "max_passage_v1"
RERANKING_SCHEMA_VERSION = "reranking_v1"
RERANKING_SCHEMA = RERANKING_SCHEMA_VERSION
RERANKED_RECORD_FILENAME = "reranked.jsonl"
_STATUSES = ("scored", "missing_representation", "unresolved", "no_passages")
_Status = Literal["scored", "missing_representation", "unresolved", "no_passages"]
_BOUNDARIES = frozenset(
    ("paragraph_packed", "sentence_packed", "oversized_sentence_fallback")
)
_HASH_LENGTH = 32
_RENAME_NOREPLACE = 1
_RENAME_EXCL = 4
_WINDOWS_ERROR_FILE_EXISTS = 80
_WINDOWS_ERROR_ALREADY_EXISTS = 183
_WINDOWS_UNSUPPORTED_ERRORS = frozenset((1, 50, 120))
_ENOTSUP = getattr(errno, "ENOTSUP", getattr(errno, "EOPNOTSUPP", 95))
_POSIX_UNSUPPORTED_ERRORS = frozenset(
    (errno.ENOSYS, _ENOTSUP, getattr(errno, "EOPNOTSUPP", _ENOTSUP))
)


def _non_blank(value: object, field_name: str, *, allow_newlines: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")
    if not allow_newlines and any(character in value for character in "\r\n"):
        raise ValueError(f"{field_name} must not contain newlines")
    return value


def _finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _optional_identity(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _non_blank(value, field_name)


def _validate_hash(value: object, field_name: str) -> bytes:
    if not isinstance(value, bytes) or len(value) != _HASH_LENGTH:
        raise ValueError(f"{field_name} must be exactly {_HASH_LENGTH} bytes")
    return value


def _optional_hash(value: object, field_name: str) -> bytes | None:
    if value is None:
        return None
    return _validate_hash(value, field_name)


def _nonnegative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer")
    return value


def _optional_text(value: object, field_name: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string or None")
    return value


def _passage_lineage_id(
    *,
    document_id: str,
    representation: str,
    representation_hash: bytes,
    start_char: int,
    end_char: int,
    chunk_index: int,
    chunker_fingerprint: str,
) -> bytes:
    return sha256_bytes(
        msgspec.msgpack.encode(
            (
                document_id,
                representation,
                representation_hash,
                start_char,
                end_char,
                chunk_index,
                chunker_fingerprint,
            )
        )
    )


@dataclass(frozen=True, slots=True)
class PassageEvidence:
    """Immutable, self-contained audit evidence for one scored passage."""

    document_id: str
    passage_id: bytes
    content_hash: bytes
    source_content_hash: bytes
    representation_hash: bytes
    representation: Representation
    language: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int
    boundary: str
    previous_passage_id: bytes | None
    next_passage_id: bytes | None
    overlap_chars: int
    overlap_tokens: int
    chunker_fingerprint: str
    score: float

    def __post_init__(self) -> None:
        _non_blank(self.document_id, "passage document_id")
        passage_id = _validate_hash(self.passage_id, "passage_id")
        content_hash = _validate_hash(self.content_hash, "content_hash")
        _validate_hash(self.source_content_hash, "source_content_hash")
        representation_hash = _validate_hash(
            self.representation_hash, "representation_hash"
        )
        if self.representation not in {"canonical", "report_text"}:
            raise ValueError("passage representation is invalid")
        language = _non_blank(self.language, "passage language")
        if self.representation == "report_text" and language != "eng":
            raise ValueError("report_text passage evidence requires language='eng'")
        chunk_index = _nonnegative_int(self.chunk_index, "chunk_index")
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("passage evidence text must not be empty")
        if (
            isinstance(self.start_char, bool)
            or not isinstance(self.start_char, int)
            or isinstance(self.end_char, bool)
            or not isinstance(self.end_char, int)
            or self.start_char < 0
            or self.end_char <= self.start_char
            or self.end_char - self.start_char != len(self.text)
        ):
            raise ValueError("passage evidence offsets must exactly cover text")
        if self.boundary not in _BOUNDARIES:
            raise ValueError("passage evidence boundary is invalid")
        _optional_hash(self.previous_passage_id, "previous_passage_id")
        _optional_hash(self.next_passage_id, "next_passage_id")
        _validate_hash(self.content_hash, "content_hash")
        if content_hash != sha256_bytes(self.text):
            raise ValueError("content_hash does not match passage evidence text")
        chunker_fingerprint = _non_blank(
            self.chunker_fingerprint, "passage chunker_fingerprint"
        )
        score = _finite_float(self.score, "passage score")
        if type(self.score) is not float:
            object.__setattr__(self, "score", score)
        overlap_chars = _nonnegative_int(self.overlap_chars, "overlap_chars")
        overlap_tokens = _nonnegative_int(self.overlap_tokens, "overlap_tokens")
        if overlap_chars != 0 or overlap_tokens != 0:
            raise ValueError("max_passage_v1 requires zero passage overlap")
        expected_id = _passage_lineage_id(
            document_id=self.document_id,
            representation=self.representation,
            representation_hash=representation_hash,
            start_char=self.start_char,
            end_char=self.end_char,
            chunk_index=chunk_index,
            chunker_fingerprint=chunker_fingerprint,
        )
        if passage_id != expected_id:
            raise ValueError("passage_id does not match passage evidence lineage")


RerankedPassage = PassageEvidence


@dataclass(frozen=True, slots=True)
class RerankedDocument:
    """One organizer document with its deterministic reranking evidence."""

    topic_id: str
    document_id: str
    status: _Status
    reranked_rank: int
    organizer_rank: int
    organizer_score: float
    score: float | None
    winning_passage_id: bytes | None
    winning_content_hash: bytes | None
    passage_ids: tuple[bytes, ...]
    passage_hashes: tuple[bytes, ...]
    passage_scores: tuple[float, ...]
    passage_evidence: tuple[PassageEvidence, ...]
    representation: Representation
    language: str | None
    source_language: str | None
    canonical_text: str | None
    selected_text: str | None
    report_text: str | None
    url: str | None
    date: str | None
    content_source: str | None
    content_local_cache_hit: bool | None
    english_text: str | None
    chunker_fingerprint: str
    scorer_fingerprint: str
    cache_fingerprint: str | None
    cache_key: str | None
    aggregation: str

    def __post_init__(self) -> None:
        _non_blank(self.topic_id, "document topic_id")
        _non_blank(self.document_id, "document document_id")
        if self.status not in _STATUSES:
            raise ValueError("document status is invalid")
        _positive_int(self.reranked_rank, "reranked_rank")
        _positive_int(self.organizer_rank, "organizer_rank")
        _finite_float(self.organizer_score, "organizer_score")
        if self.representation not in {"canonical", "report_text"}:
            raise ValueError("document representation is invalid")
        _optional_text(self.language, "document language")
        _optional_text(self.source_language, "document source_language")
        _optional_text(self.canonical_text, "canonical_text")
        _optional_text(self.selected_text, "selected_text")
        _optional_text(self.report_text, "report_text")
        _optional_text(self.url, "url")
        _optional_text(self.date, "date")
        _optional_text(self.content_source, "content_source")
        _optional_text(self.english_text, "english_text")
        if self.content_local_cache_hit is not None and not isinstance(
            self.content_local_cache_hit, bool
        ):
            raise ValueError("content_local_cache_hit must be a boolean or None")
        _non_blank(self.chunker_fingerprint, "chunker_fingerprint")
        _non_blank(self.scorer_fingerprint, "scorer_fingerprint")
        cache_fingerprint = _optional_identity(
            self.cache_fingerprint, "cache_fingerprint"
        )
        cache_key = _optional_identity(self.cache_key, "cache_key")
        if (cache_fingerprint is None) != (cache_key is None):
            raise ValueError(
                "cache_fingerprint and cache_key must be provided together"
            )
        if self.aggregation != max_passage_v1:
            raise ValueError("unsupported passage aggregation")

        if self.english_text is not None and self.report_text != self.english_text:
            raise ValueError("report_text must match english_text when present")
        if (
            self.english_text is None
            and self.source_language == "eng"
            and self.canonical_text is not None
            and self.report_text != self.canonical_text
        ):
            raise ValueError("English source report_text must match canonical_text")

        if not isinstance(self.passage_ids, tuple):
            raise ValueError("passage_ids must be a tuple")
        if not isinstance(self.passage_hashes, tuple):
            raise ValueError("passage_hashes must be a tuple")
        if not isinstance(self.passage_scores, tuple):
            raise ValueError("passage_scores must be a tuple")
        if not isinstance(self.passage_evidence, tuple):
            raise ValueError("passage_evidence must be a tuple")
        if any(
            _validate_hash(value, "passage_id") != value for value in self.passage_ids
        ):
            raise ValueError("passage_ids are malformed")
        if any(
            _validate_hash(value, "passage hash") != value
            for value in self.passage_hashes
        ):
            raise ValueError("passage_hashes are malformed")
        if (
            len(self.passage_ids) != len(self.passage_hashes)
            or len(self.passage_ids) != len(self.passage_scores)
            or len(self.passage_ids) != len(self.passage_evidence)
        ):
            raise ValueError("passage convenience fields and evidence disagree")
        if any(
            not isinstance(value, PassageEvidence) for value in self.passage_evidence
        ):
            raise ValueError("passage_evidence must contain PassageEvidence values")
        for evidence, passage_id, content_hash, passage_score in zip(
            self.passage_evidence,
            self.passage_ids,
            self.passage_hashes,
            self.passage_scores,
            strict=True,
        ):
            try:
                PassageEvidence(
                    **{
                        field: getattr(evidence, field)
                        for field in PassageEvidence.__dataclass_fields__
                    }
                )
            except (AttributeError, TypeError, ValueError) as error:
                raise ValueError(
                    "passage_evidence contains invalid evidence"
                ) from error
            if evidence.document_id != self.document_id:
                raise ValueError("passage evidence document_id disagrees")
            if evidence.representation != self.representation:
                raise ValueError("passage evidence representation disagrees")
            if evidence.passage_id != passage_id:
                raise ValueError("passage_ids disagree with passage evidence")
            if evidence.content_hash != content_hash:
                raise ValueError("passage_hashes disagree with passage evidence")
            if _finite_float(passage_score, "passage score") != evidence.score:
                raise ValueError("passage_scores disagree with passage evidence")

        if self.status == "scored":
            if self.canonical_text is None or self.selected_text is None:
                raise ValueError("scored documents require both text representations")
            if self.score is None:
                raise ValueError("scored documents require a score")
            scored_value = _finite_float(self.score, "document score")
            if self.winning_passage_id is None or self.winning_content_hash is None:
                raise ValueError("scored documents require a winning passage")
            _validate_hash(self.winning_passage_id, "winning_passage_id")
            _validate_hash(self.winning_content_hash, "winning_content_hash")
            if not self.passage_evidence:
                raise ValueError("scored documents require passage evidence")
            if self.representation == "report_text":
                if self.language != "eng" or self.report_text != self.selected_text:
                    raise ValueError(
                        "report_text documents require exact English report text"
                    )
            else:
                if self.selected_text != self.canonical_text:
                    raise ValueError(
                        "canonical documents must select exact source text"
                    )
                if self.language is None or self.language != self.source_language:
                    raise ValueError(
                        "canonical documents require their source language"
                    )
            expected_source_hash = sha256_bytes(self.canonical_text)
            expected_representation_hash = sha256_bytes(self.selected_text)
            cursor = 0
            for index, evidence in enumerate(self.passage_evidence):
                if evidence.chunk_index != index:
                    raise ValueError("passage evidence indexes must be contiguous")
                if evidence.language != self.language:
                    raise ValueError("passage evidence language disagrees")
                if evidence.chunker_fingerprint != self.chunker_fingerprint:
                    raise ValueError("passage evidence chunker fingerprint disagrees")
                if evidence.source_content_hash != expected_source_hash:
                    raise ValueError("passage evidence source hash disagrees")
                if evidence.representation_hash != expected_representation_hash:
                    raise ValueError("passage evidence representation hash disagrees")
                expected_previous = (
                    None if index == 0 else self.passage_evidence[index - 1].passage_id
                )
                expected_next = (
                    None
                    if index == len(self.passage_evidence) - 1
                    else self.passage_evidence[index + 1].passage_id
                )
                if evidence.previous_passage_id != expected_previous:
                    raise ValueError("previous passage linkage disagrees")
                if evidence.next_passage_id != expected_next:
                    raise ValueError("next passage linkage disagrees")
                if evidence.start_char != cursor:
                    raise ValueError("passage evidence must reconstruct selected_text")
                if (
                    self.selected_text[evidence.start_char : evidence.end_char]
                    != evidence.text
                ):
                    raise ValueError(
                        "passage evidence text does not match selected_text"
                    )
                cursor = evidence.end_char
            if cursor != len(self.selected_text):
                raise ValueError("passage evidence does not cover selected_text")
            winning_score = max(evidence.score for evidence in self.passage_evidence)
            if scored_value != winning_score:
                raise ValueError("document score must equal max passage score")
            winning = min(
                (
                    evidence
                    for evidence in self.passage_evidence
                    if evidence.score == winning_score
                ),
                key=lambda evidence: evidence.content_hash,
            )
            if self.winning_passage_id != winning.passage_id:
                raise ValueError("winning_passage_id disagrees with winning evidence")
            if self.winning_content_hash != winning.content_hash:
                raise ValueError("winning_content_hash disagrees with winning evidence")
        else:
            if self.score is not None:
                raise ValueError("unscored documents must not have a score")
            if (
                self.winning_passage_id is not None
                or self.winning_content_hash is not None
            ):
                raise ValueError("unscored documents must not have a winning passage")
            if (
                self.passage_evidence
                or self.passage_ids
                or self.passage_hashes
                or self.passage_scores
            ):
                raise ValueError("unscored documents must not have passage evidence")
            if self.status == "unresolved":
                if (
                    self.canonical_text is not None
                    or self.selected_text is not None
                    or self.language is not None
                ):
                    raise ValueError("unresolved documents must not have selected text")
            elif self.status == "missing_representation":
                if self.canonical_text is None:
                    raise ValueError(
                        "missing representations require hydrated source text"
                    )
                if self.selected_text is not None:
                    raise ValueError(
                        "missing representations must not have selected text"
                    )
                if self.representation == "canonical":
                    if self.language is not None:
                        raise ValueError(
                            "missing canonical representations must not have a selected language"
                        )
                else:
                    if self.language != "eng":
                        raise ValueError(
                            "missing report representations require language='eng'"
                        )
                    if self.report_text is not None:
                        raise ValueError(
                            "missing report representations must not have report_text"
                        )
            elif self.status == "no_passages":
                if self.canonical_text is None or self.selected_text is None:
                    raise ValueError("no_passages documents require selected text")
                if self.representation == "report_text":
                    if self.language != "eng" or self.report_text != self.selected_text:
                        raise ValueError(
                            "no_passages report text must be English and exact"
                        )
                else:
                    if self.selected_text != self.canonical_text:
                        raise ValueError(
                            "no_passages canonical text must be exact source text"
                        )
                    if self.language is None or self.language != self.source_language:
                        raise ValueError(
                            "no_passages canonical text requires source language"
                        )

    @property
    def winning_passage(self) -> PassageEvidence | None:
        if self.winning_passage_id is None:
            return None
        return next(
            evidence
            for evidence in self.passage_evidence
            if evidence.passage_id == self.winning_passage_id
        )

    @property
    def reranked_score(self) -> float | None:
        """Compatibility name for the aggregated document score."""

        return self.score

    @property
    def rank(self) -> int:
        """Compatibility name for the organizer rank."""

        return self.organizer_rank

    @property
    def retrieval_score(self) -> float:
        """Compatibility name for the organizer score."""

        return self.organizer_score

    @property
    def winning_hash(self) -> bytes | None:
        """Compatibility name for the winning passage content hash."""

        return self.winning_content_hash

    @property
    def selected_language(self) -> str | None:
        """Language used to prepare the selected representation."""

        return self.language

    @property
    def selected_representation(self) -> Representation:
        """Compatibility name for the selected representation."""

        return self.representation

    @property
    def winning_passage_score(self) -> float | None:
        """Score of the selected winning passage."""

        return self.score

    @property
    def passage_content_hashes(self) -> tuple[bytes, ...]:
        """Compatibility name for all passage content hashes."""

        return self.passage_hashes

    @property
    def text(self) -> str | None:
        """Return the canonical source text retained for this document."""

        return self.canonical_text


@dataclass(frozen=True, slots=True)
class RerankingResult:
    """One deterministic, auditable reranking result for an explicit topic."""

    topic_id: str
    query: str
    pipeline: str
    representation: Representation
    chunker_fingerprint: str
    scorer_fingerprint: str
    cache_fingerprint: str | None
    cache_key: str | None
    aggregation: str
    documents: tuple[RerankedDocument, ...]

    def __post_init__(self) -> None:
        _validate_reranking_result(self)

    @property
    def topic(self) -> str:
        """Compatibility name for the explicit topic identity."""

        return self.topic_id

    @property
    def status_counts(self) -> dict[str, int]:
        counts = {status: 0 for status in _STATUSES}
        for document in self.documents:
            counts[document.status] += 1
        return counts

    @property
    def counts(self) -> dict[str, int]:
        """Compatibility alias for status counts."""

        return self.status_counts


@dataclass(frozen=True, slots=True)
class _PreparedDocument:
    document: RetrievedDocument
    organizer_rank: int
    organizer_score: float
    status: _Status | None
    canonical_text: str | None
    selected_text: str | None
    selected_language: str | None
    passages: tuple[Passage, ...]


def _validate_reranking_result(result: RerankingResult) -> None:
    _non_blank(result.topic_id, "topic_id")
    _non_blank(result.query, "query", allow_newlines=True)
    _non_blank(result.pipeline, "pipeline")
    if result.representation not in {"canonical", "report_text"}:
        raise ValueError("result representation is invalid")
    _non_blank(result.chunker_fingerprint, "chunker_fingerprint")
    _non_blank(result.scorer_fingerprint, "scorer_fingerprint")
    cache_fingerprint = _optional_identity(
        result.cache_fingerprint, "cache_fingerprint"
    )
    cache_key = _optional_identity(result.cache_key, "cache_key")
    if (cache_fingerprint is None) != (cache_key is None):
        raise ValueError("result cache_fingerprint and cache_key must agree")
    if result.aggregation != max_passage_v1:
        raise ValueError("result aggregation is invalid")
    if not isinstance(result.documents, tuple):
        raise ValueError("result documents must be a tuple")
    document_ids: set[str] = set()
    organizer_ranks: set[int] = set()
    for index, document in enumerate(result.documents, start=1):
        if not isinstance(document, RerankedDocument):
            raise ValueError("result documents must contain RerankedDocument values")
        try:
            RerankedDocument(
                **{
                    field: getattr(document, field)
                    for field in document.__dataclass_fields__
                }
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("result contains an invalid RerankedDocument") from error
        if document.document_id in document_ids:
            raise ValueError("result contains duplicate document IDs")
        document_ids.add(document.document_id)
        if document.organizer_rank in organizer_ranks:
            raise ValueError("result contains duplicate organizer ranks")
        organizer_ranks.add(document.organizer_rank)
        if document.reranked_rank != index:
            raise ValueError("reranked_rank values must exactly match tuple order")
        if document.topic_id != result.topic_id:
            raise ValueError("document topic_id disagrees with result")
        if document.representation != result.representation:
            raise ValueError("document representation disagrees with result")
        if document.chunker_fingerprint != result.chunker_fingerprint:
            raise ValueError("document chunker_fingerprint disagrees with result")
        if document.scorer_fingerprint != result.scorer_fingerprint:
            raise ValueError("document scorer_fingerprint disagrees with result")
        if document.cache_fingerprint != result.cache_fingerprint:
            raise ValueError("document cache_fingerprint disagrees with result")
        if document.cache_key != result.cache_key:
            raise ValueError("document cache_key disagrees with result")
        if document.aggregation != result.aggregation:
            raise ValueError("document aggregation disagrees with result")
    expected_order = tuple(sorted(result.documents, key=_reranked_document_sort_key))
    if expected_order != result.documents:
        raise ValueError("result documents are not in deterministic reranked order")


def _reranked_document_sort_key(
    document: RerankedDocument,
) -> tuple[int, float, int, str, bytes | None]:
    if document.status == "scored":
        return (
            0,
            -float(document.score),  # type: ignore[arg-type]
            document.organizer_rank,
            document.document_id,
            document.winning_content_hash,
        )
    return (1, 0.0, document.organizer_rank, document.document_id, None)


def _validate_search_rows(
    retrieval: RetrievalResult,
) -> tuple[tuple[SearchHit, RetrievedDocument], ...]:
    if not isinstance(retrieval, RetrievalResult):
        raise TypeError("retrieval must be a RetrievalResult")
    query = _non_blank(retrieval.query, "retrieval query", allow_newlines=True)
    _non_blank(retrieval.pipeline, "retrieval pipeline")
    del query

    hits = retrieval.search_result.hits
    documents = retrieval.documents
    if isinstance(hits, (str, bytes, bytearray)) or not isinstance(hits, Sequence):
        raise ValueError("retrieval search hits must be a sequence")
    if isinstance(documents, (str, bytes, bytearray)) or not isinstance(
        documents, Sequence
    ):
        raise ValueError("retrieval documents must be a sequence")

    hits_by_id: dict[str, SearchHit] = {}
    hit_ranks: set[int] = set()
    for hit in hits:
        if not isinstance(hit, SearchHit):
            raise ValueError("retrieval search hits must contain SearchHit values")
        document_id = _non_blank(hit.document_id, "search hit document_id")
        rank = _positive_int(hit.rank, "search hit rank")
        score = _finite_float(hit.score, "search hit score")
        if document_id in hits_by_id:
            raise ValueError(f"duplicate search hit document_id: {document_id}")
        if rank in hit_ranks:
            raise ValueError(f"duplicate search hit rank: {rank}")
        hit_ranks.add(rank)
        hits_by_id[document_id] = SearchHit(
            rank=rank, document_id=document_id, score=score
        )

    documents_by_id: dict[str, RetrievedDocument] = {}
    document_ranks: set[int] = set()
    for document in documents:
        if not isinstance(document, RetrievedDocument):
            raise ValueError(
                "retrieval documents must contain RetrievedDocument values"
            )
        document_id = _non_blank(document.document_id, "retrieved document_id")
        rank = _positive_int(document.rank, "retrieved document rank")
        score = _finite_float(document.score, "retrieved document score")
        if document_id in documents_by_id:
            raise ValueError(f"duplicate retrieved document_id: {document_id}")
        if rank in document_ranks:
            raise ValueError(f"duplicate retrieved document rank: {rank}")
        document_ranks.add(rank)
        if document.text is not None and not isinstance(document.text, str):
            raise ValueError(f"document text must be a string or None: {document_id}")
        if document.english_text is not None and not isinstance(
            document.english_text, str
        ):
            raise ValueError(
                f"document english_text must be a string or None: {document_id}"
            )
        documents_by_id[document_id] = document
        del score

    if set(hits_by_id) != set(documents_by_id):
        missing = sorted(set(hits_by_id).difference(documents_by_id))
        extra = sorted(set(documents_by_id).difference(hits_by_id))
        raise ValueError(
            f"search/document IDs are misaligned; missing={missing}, extra={extra}"
        )

    rows: list[tuple[SearchHit, RetrievedDocument]] = []
    for document_id, hit in hits_by_id.items():
        document = documents_by_id[document_id]
        if document.rank != hit.rank or document.score != hit.score:
            raise ValueError(
                f"search/document rank or score is misaligned: {document_id}"
            )
        rows.append((hit, document))
    rows.sort(key=lambda row: (row[0].rank, row[0].document_id))
    return tuple(rows)


def _scorer_identity(
    scorer: PassageScorer,
) -> tuple[str, str | None, Callable[[str], object] | str | None]:
    fingerprint = _non_blank(getattr(scorer, "fingerprint", None), "scorer fingerprint")
    score_method = getattr(scorer, "score", None)
    if not callable(score_method):
        raise TypeError("scorer must expose a callable score method")

    cache = getattr(scorer, "cache", None)
    if cache is None:
        cache_fingerprint = _optional_identity(
            getattr(scorer, "cache_fingerprint", None), "cache fingerprint"
        )
        key_factory = getattr(scorer, "cache_key", None)
        if (
            key_factory is not None
            and not callable(key_factory)
            and not isinstance(key_factory, str)
        ):
            raise TypeError("scorer cache_key must be a string or callable")
        return fingerprint, cache_fingerprint, key_factory

    cache_fingerprint = _non_blank(
        getattr(cache, "effective_fingerprint", getattr(cache, "fingerprint", None)),
        "cache fingerprint",
    )
    key_factory = getattr(cache, "object_key", None)
    if key_factory is None:
        key_factory = getattr(cache, "cache_key", None)
    if key_factory is None or not callable(key_factory):
        raise TypeError("cached scorer cache must expose a callable object_key")
    return fingerprint, cache_fingerprint, key_factory


def _resolve_cache_key(
    query: str,
    cache_fingerprint: str | None,
    cache_identity: Callable[[str], object] | str | None,
) -> str | None:
    if cache_fingerprint is None:
        return None
    if callable(cache_identity):
        return _non_blank(cache_identity(query), "cache key")
    return _non_blank(cache_identity, "cache key")


def _passage_validation(
    passages: tuple[Passage, ...],
    *,
    document_id: str,
    canonical_text: str,
    selected_text: str,
    representation: Representation,
    language: str,
    chunker_fingerprint: str,
) -> None:
    validate_passages(selected_text, passages)
    expected_source_hash = sha256_bytes(canonical_text)
    expected_representation_hash = sha256_bytes(selected_text)
    for passage in passages:
        if passage.document_id != document_id:
            raise PassageValidationError(
                "chunker returned a passage for another document"
            )
        if passage.representation != representation:
            raise PassageValidationError("chunker returned another representation")
        if passage.language != language:
            raise PassageValidationError("chunker returned another language")
        if passage.chunker_fingerprint != chunker_fingerprint:
            raise PassageValidationError("chunker returned another fingerprint")
        if passage.source_content_hash != expected_source_hash:
            raise PassageValidationError(
                "passage source hash does not match canonical text"
            )
        if passage.representation_hash != expected_representation_hash:
            raise PassageValidationError(
                "passage representation hash does not match selected text"
            )


def _make_result_document(
    prepared: _PreparedDocument,
    *,
    topic_id: str,
    representation: Representation,
    chunker_fingerprint: str,
    scorer_fingerprint: str,
    cache_fingerprint: str | None,
    cache_key: str | None,
    scores: Mapping[bytes, float],
) -> RerankedDocument:
    document = prepared.document
    passage_ids = tuple(passage.passage_id for passage in prepared.passages)
    passage_hashes = tuple(passage.content_hash for passage in prepared.passages)
    passage_scores = tuple(
        float(scores[passage.content_hash]) for passage in prepared.passages
    )
    passage_evidence = tuple(
        PassageEvidence(
            document_id=passage.document_id,
            passage_id=passage.passage_id,
            content_hash=passage.content_hash,
            source_content_hash=passage.source_content_hash,
            representation_hash=passage.representation_hash,
            representation=passage.representation,
            language=passage.language,
            chunk_index=passage.chunk_index,
            text=passage.text,
            start_char=passage.start_char,
            end_char=passage.end_char,
            boundary=passage.boundary,
            previous_passage_id=passage.previous_passage_id,
            next_passage_id=passage.next_passage_id,
            overlap_chars=passage.overlap_chars,
            overlap_tokens=passage.overlap_tokens,
            chunker_fingerprint=passage.chunker_fingerprint,
            score=passage_scores[index],
        )
        for index, passage in enumerate(prepared.passages)
    )
    canonical_text = prepared.canonical_text
    selected_text = prepared.selected_text
    source_language = document.language
    if prepared.status is not None:
        return RerankedDocument(
            topic_id=topic_id,
            document_id=document.document_id,
            status=prepared.status,
            reranked_rank=1,
            organizer_rank=prepared.organizer_rank,
            organizer_score=prepared.organizer_score,
            score=None,
            winning_passage_id=None,
            winning_content_hash=None,
            passage_ids=passage_ids,
            passage_hashes=passage_hashes,
            passage_scores=passage_scores,
            passage_evidence=passage_evidence,
            representation=representation,
            language=prepared.selected_language,
            source_language=source_language,
            canonical_text=canonical_text,
            selected_text=selected_text,
            report_text=document.report_text,
            url=document.url,
            date=document.date,
            content_source=document.content_source,
            content_local_cache_hit=document.content_local_cache_hit,
            english_text=document.english_text,
            chunker_fingerprint=chunker_fingerprint,
            scorer_fingerprint=scorer_fingerprint,
            cache_fingerprint=cache_fingerprint,
            cache_key=cache_key,
            aggregation=max_passage_v1,
        )

    candidates = tuple(
        (float(scores[passage.content_hash]), passage) for passage in prepared.passages
    )
    winning_score = max(score for score, _ in candidates)
    winning_passage = min(
        (passage for score, passage in candidates if score == winning_score),
        key=lambda passage: passage.content_hash,
    )
    return RerankedDocument(
        topic_id=topic_id,
        document_id=document.document_id,
        status="scored",
        reranked_rank=1,
        organizer_rank=prepared.organizer_rank,
        organizer_score=prepared.organizer_score,
        score=winning_score,
        winning_passage_id=winning_passage.passage_id,
        winning_content_hash=winning_passage.content_hash,
        passage_ids=passage_ids,
        passage_hashes=passage_hashes,
        passage_scores=passage_scores,
        passage_evidence=passage_evidence,
        representation=representation,
        language=prepared.selected_language,
        source_language=source_language,
        canonical_text=canonical_text,
        selected_text=selected_text,
        report_text=document.report_text,
        url=document.url,
        date=document.date,
        content_source=document.content_source,
        content_local_cache_hit=document.content_local_cache_hit,
        english_text=document.english_text,
        chunker_fingerprint=chunker_fingerprint,
        scorer_fingerprint=scorer_fingerprint,
        cache_fingerprint=cache_fingerprint,
        cache_key=cache_key,
        aggregation=max_passage_v1,
    )


def rerank_retrieval(
    retrieval: RetrievalResult,
    *,
    topic_id: str,
    chunker: PassageChunker,
    scorer: PassageScorer,
    representation: Representation,
) -> RerankingResult:
    """Rerank one retrieved topic with exact, deterministic passage evidence."""

    topic = _non_blank(topic_id, "topic_id")
    if representation not in {"canonical", "report_text"}:
        raise ValueError("representation must be canonical or report_text")
    if not callable(getattr(chunker, "chunk", None)):
        raise TypeError("chunker must expose a callable chunk method")
    chunker_fingerprint = _non_blank(
        getattr(chunker, "fingerprint", None), "chunker fingerprint"
    )
    scorer_fingerprint, cache_fingerprint, cache_identity = _scorer_identity(scorer)
    rows = _validate_search_rows(retrieval)
    query = _non_blank(retrieval.query, "retrieval query", allow_newlines=True)
    pipeline = _non_blank(retrieval.pipeline, "retrieval pipeline")
    cache_key = _resolve_cache_key(query, cache_fingerprint, cache_identity)

    prepared_rows: list[_PreparedDocument] = []
    unique_passages: dict[bytes, Passage] = {}
    for hit, document in rows:
        canonical_text = document.text
        if canonical_text is None:
            prepared = _PreparedDocument(
                document=document,
                organizer_rank=hit.rank,
                organizer_score=hit.score,
                status="unresolved",
                canonical_text=None,
                selected_text=None,
                selected_language=None,
                passages=(),
            )
            prepared_rows.append(prepared)
            continue
        if representation == "canonical":
            selected_text = canonical_text
            selected_language = document.language
            if (
                selected_language is None
                or not isinstance(selected_language, str)
                or not selected_language.strip()
            ):
                prepared = _PreparedDocument(
                    document=document,
                    organizer_rank=hit.rank,
                    organizer_score=hit.score,
                    status="missing_representation",
                    canonical_text=canonical_text,
                    selected_text=None,
                    selected_language=None,
                    passages=(),
                )
                prepared_rows.append(prepared)
                continue
        else:
            selected_text = document.report_text
            selected_language = "eng"
            if selected_text is None:
                prepared = _PreparedDocument(
                    document=document,
                    organizer_rank=hit.rank,
                    organizer_score=hit.score,
                    status="missing_representation",
                    canonical_text=canonical_text,
                    selected_text=None,
                    selected_language=selected_language,
                    passages=(),
                )
                prepared_rows.append(prepared)
                continue
        if not isinstance(selected_text, str):
            raise ValueError("selected representation must be a string")

        passages = chunker.chunk(
            document_id=document.document_id,
            canonical_text=canonical_text,
            selected_text=selected_text,
            representation=representation,
            language=selected_language,
        )
        if (
            passages is None
            or isinstance(passages, (str, bytes, bytearray))
            or not isinstance(passages, Sequence)
        ):
            raise ValueError("chunker must return a sequence of passages")
        passage_tuple = tuple(passages)
        if not passage_tuple:
            prepared = _PreparedDocument(
                document=document,
                organizer_rank=hit.rank,
                organizer_score=hit.score,
                status="no_passages",
                canonical_text=canonical_text,
                selected_text=selected_text,
                selected_language=selected_language,
                passages=(),
            )
            prepared_rows.append(prepared)
            continue
        _passage_validation(
            passage_tuple,
            document_id=document.document_id,
            canonical_text=canonical_text,
            selected_text=selected_text,
            representation=representation,
            language=selected_language,
            chunker_fingerprint=chunker_fingerprint,
        )
        prepared = _PreparedDocument(
            document=document,
            organizer_rank=hit.rank,
            organizer_score=hit.score,
            status=None,
            canonical_text=canonical_text,
            selected_text=selected_text,
            selected_language=selected_language,
            passages=passage_tuple,
        )
        prepared_rows.append(prepared)
        for passage in passage_tuple:
            unique_passages.setdefault(passage.content_hash, passage)

    scores: Mapping[bytes, float]
    if unique_passages:
        passages_to_score = tuple(
            unique_passages[key] for key in sorted(unique_passages)
        )
        raw_scores = scorer.score(query, passages_to_score)
        scores = validate_passage_scores(passages_to_score, raw_scores)
    else:
        scores = {}

    reranked_documents = tuple(
        _make_result_document(
            prepared,
            topic_id=topic,
            representation=representation,
            chunker_fingerprint=chunker_fingerprint,
            scorer_fingerprint=scorer_fingerprint,
            cache_fingerprint=cache_fingerprint,
            cache_key=cache_key,
            scores=scores,
        )
        for prepared in prepared_rows
    )
    ordered_documents = tuple(
        sorted(reranked_documents, key=_reranked_document_sort_key)
    )
    ranked_documents = tuple(
        replace(document, reranked_rank=index)
        for index, document in enumerate(ordered_documents, start=1)
    )
    return RerankingResult(
        topic_id=topic,
        query=query,
        pipeline=pipeline,
        representation=representation,
        chunker_fingerprint=chunker_fingerprint,
        scorer_fingerprint=scorer_fingerprint,
        cache_fingerprint=cache_fingerprint,
        cache_key=cache_key,
        aggregation=max_passage_v1,
        documents=ranked_documents,
    )


def _json_evidence(evidence: PassageEvidence) -> dict[str, object]:
    return {
        "boundary": evidence.boundary,
        "chunk_index": evidence.chunk_index,
        "content_hash": evidence.content_hash.hex(),
        "document_id": evidence.document_id,
        "end_char": evidence.end_char,
        "language": evidence.language,
        "next_passage_id": (
            None if evidence.next_passage_id is None else evidence.next_passage_id.hex()
        ),
        "overlap_chars": evidence.overlap_chars,
        "overlap_tokens": evidence.overlap_tokens,
        "passage_id": evidence.passage_id.hex(),
        "previous_passage_id": (
            None
            if evidence.previous_passage_id is None
            else evidence.previous_passage_id.hex()
        ),
        "representation": evidence.representation,
        "representation_hash": evidence.representation_hash.hex(),
        "score": evidence.score,
        "source_content_hash": evidence.source_content_hash.hex(),
        "start_char": evidence.start_char,
        "text": evidence.text,
        "chunker_fingerprint": evidence.chunker_fingerprint,
    }


def _json_record(
    result: RerankingResult, document: RerankedDocument
) -> dict[str, object]:
    winning_passage = document.winning_passage
    return {
        "aggregation": document.aggregation,
        "cache_fingerprint": document.cache_fingerprint,
        "cache_key": document.cache_key,
        "canonical_text": document.canonical_text,
        "chunker_fingerprint": document.chunker_fingerprint,
        "content_local_cache_hit": document.content_local_cache_hit,
        "content_source": document.content_source,
        "date": document.date,
        "document_id": document.document_id,
        "english_text": document.english_text,
        "language": document.language,
        "organizer_rank": document.organizer_rank,
        "organizer_score": document.organizer_score,
        "pipeline": result.pipeline,
        "passage_evidence": [
            _json_evidence(evidence) for evidence in document.passage_evidence
        ],
        "passage_hashes": [value.hex() for value in document.passage_hashes],
        "passage_ids": [value.hex() for value in document.passage_ids],
        "passage_scores": list(document.passage_scores),
        "report_text": document.report_text,
        "representation": document.representation,
        "reranked_rank": document.reranked_rank,
        "score": document.score,
        "selected_text": document.selected_text,
        "schema_version": RERANKING_SCHEMA_VERSION,
        "scorer_fingerprint": document.scorer_fingerprint,
        "source_language": document.source_language,
        "status": document.status,
        "topic_id": document.topic_id,
        "url": document.url,
        "winning_content_hash": (
            None
            if document.winning_content_hash is None
            else document.winning_content_hash.hex()
        ),
        "winning_passage_id": (
            None
            if document.winning_passage_id is None
            else document.winning_passage_id.hex()
        ),
        "winning_passage": (
            None if winning_passage is None else _json_evidence(winning_passage)
        ),
        "query": result.query,
    }


def _sha256_file_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_fsync(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    if sys.platform.startswith("win"):
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unsupported_noreplace(message: str) -> None:
    raise OSError(_ENOTSUP, message)


def _raise_posix_publish_error(error_number: int, destination: Path) -> None:
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), str(destination))
    if error_number in _POSIX_UNSUPPORTED_ERRORS:
        _unsupported_noreplace("atomic no-replace publication is unavailable")
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _publish_noreplace_linux(source: Path, destination: Path) -> None:
    """Publish with Linux renameat2(RENAME_NOREPLACE)."""

    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
    except (AttributeError, OSError) as error:
        raise OSError(
            _ENOTSUP,
            "atomic no-replace publication is unavailable on Linux",
        ) from error
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,  # AT_FDCWD
        os.fsencode(source),
        -100,  # AT_FDCWD
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    _raise_posix_publish_error(ctypes.get_errno(), destination)


def _publish_noreplace_darwin(source: Path, destination: Path) -> None:
    """Publish with Darwin renamex_np(RENAME_EXCL)."""

    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renamex_np = libc.renamex_np
    except (AttributeError, OSError) as error:
        raise OSError(
            _ENOTSUP,
            "atomic no-replace publication is unavailable on Darwin",
        ) from error
    renamex_np.argtypes = (
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renamex_np.restype = ctypes.c_int
    result = renamex_np(
        os.fsencode(source),
        os.fsencode(destination),
        _RENAME_EXCL,
    )
    if result == 0:
        return
    _raise_posix_publish_error(ctypes.get_errno(), destination)


def _publish_noreplace_windows(source: Path, destination: Path) -> None:
    """Publish with same-volume MoveFileExW without replacement flags."""

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        move_file_ex = kernel32.MoveFileExW
    except (AttributeError, OSError) as error:
        raise OSError(
            _ENOTSUP,
            "MoveFileExW is unavailable on Windows",
        ) from error
    move_file_ex.argtypes = (
        ctypes.c_wchar_p,
        ctypes.c_wchar_p,
        ctypes.c_uint,
    )
    move_file_ex.restype = ctypes.c_int
    if move_file_ex(str(source), str(destination), 0):
        return
    get_last_error = getattr(ctypes, "get_last_error", None)
    if not callable(get_last_error):
        _unsupported_noreplace("Windows error reporting is unavailable")
    error_number = get_last_error()
    if error_number in {
        _WINDOWS_ERROR_FILE_EXISTS,
        _WINDOWS_ERROR_ALREADY_EXISTS,
    }:
        raise FileExistsError(
            errno.EEXIST,
            "destination already exists",
            str(destination),
        )
    if error_number in _WINDOWS_UNSUPPORTED_ERRORS:
        _unsupported_noreplace("MoveFileExW no-replace publication is unavailable")
    raise OSError(
        error_number,
        f"MoveFileExW failed with error {error_number}",
        str(destination),
    )


def _publish_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without ever replacing a destination."""

    if sys.platform.startswith("linux"):
        _publish_noreplace_linux(source, destination)
        return
    if sys.platform == "darwin":
        _publish_noreplace_darwin(source, destination)
        return
    if sys.platform.startswith("win"):
        _publish_noreplace_windows(source, destination)
        return
    _unsupported_noreplace(
        "atomic no-replace publication is unavailable on this platform"
    )


def write_reranking_artifact(
    destination: str | os.PathLike[str],
    result: RerankingResult,
    *,
    source_retrieval_ref: str,
) -> None:
    """Write one immutable reranking run using a sibling-directory rename."""

    if not isinstance(result, RerankingResult):
        raise TypeError("result must be a RerankingResult")
    _validate_reranking_result(result)
    source_ref = _non_blank(source_retrieval_ref, "source_retrieval_ref")
    destination_path = Path(destination)
    if os.path.lexists(destination_path):
        raise FileExistsError(f"reranking artifact already exists: {destination_path}")
    parent = destination_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination_path):
        raise FileExistsError(f"reranking artifact already exists: {destination_path}")

    records_payload = b"".join(
        (
            json.dumps(
                _json_record(result, document),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        for document in result.documents
    )
    manifest = {
        "aggregation": result.aggregation,
        "cache_fingerprint": result.cache_fingerprint,
        "cache_key": result.cache_key,
        "chunker_fingerprint": result.chunker_fingerprint,
        "counts": result.status_counts,
        "pipeline": result.pipeline,
        "query": result.query,
        "records": {
            "filename": RERANKED_RECORD_FILENAME,
            "sha256": _sha256_file_bytes(records_payload),
        },
        "representation": result.representation,
        "schema_version": RERANKING_SCHEMA_VERSION,
        "scorer_fingerprint": result.scorer_fingerprint,
        "source_retrieval_ref": source_ref,
        "topic_id": result.topic_id,
    }
    manifest_payload = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )

    temporary_path: Path | None = None
    try:
        temporary_path = Path(
            tempfile.mkdtemp(prefix=f".{destination_path.name}.", dir=parent)
        )
        _write_fsync(temporary_path / RERANKED_RECORD_FILENAME, records_payload)
        _write_fsync(temporary_path / "manifest.json", manifest_payload)
        _fsync_directory(temporary_path)
        _publish_noreplace(temporary_path, destination_path)
        temporary_path = None
        _fsync_directory(parent)
    finally:
        if temporary_path is not None:
            shutil.rmtree(temporary_path, ignore_errors=True)


__all__ = [
    "RERANKED_RECORD_FILENAME",
    "RERANKING_SCHEMA",
    "RERANKING_SCHEMA_VERSION",
    "PassageEvidence",
    "RerankedDocument",
    "RerankedPassage",
    "RerankingResult",
    "max_passage_v1",
    "rerank_retrieval",
    "write_reranking_artifact",
]
