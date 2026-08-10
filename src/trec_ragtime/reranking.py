"""Deterministic passage reranking and immutable per-topic artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class RerankedDocument:
    """One organizer document with its deterministic reranking evidence."""

    topic_id: str
    document_id: str
    status: _Status
    organizer_rank: int
    organizer_score: float
    score: float | None
    winning_passage_id: bytes | None
    winning_content_hash: bytes | None
    passage_ids: tuple[bytes, ...]
    passage_hashes: tuple[bytes, ...]
    passage_scores: tuple[float, ...]
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


def _validate_search_rows(retrieval: RetrievalResult) -> tuple[tuple[SearchHit, RetrievedDocument], ...]:
    if not isinstance(retrieval, RetrievalResult):
        raise TypeError("retrieval must be a RetrievalResult")
    query = _non_blank(retrieval.query, "retrieval query", allow_newlines=True)
    _non_blank(retrieval.pipeline, "retrieval pipeline")
    del query

    hits = retrieval.search_result.hits
    documents = retrieval.documents
    if isinstance(hits, (str, bytes, bytearray)) or not isinstance(hits, Sequence):
        raise ValueError("retrieval search hits must be a sequence")
    if isinstance(documents, (str, bytes, bytearray)) or not isinstance(documents, Sequence):
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
        hits_by_id[document_id] = SearchHit(rank=rank, document_id=document_id, score=score)

    documents_by_id: dict[str, RetrievedDocument] = {}
    document_ranks: set[int] = set()
    for document in documents:
        if not isinstance(document, RetrievedDocument):
            raise ValueError("retrieval documents must contain RetrievedDocument values")
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
        if document.english_text is not None and not isinstance(document.english_text, str):
            raise ValueError(f"document english_text must be a string or None: {document_id}")
        documents_by_id[document_id] = document
        del score

    if set(hits_by_id) != set(documents_by_id):
        missing = sorted(set(hits_by_id).difference(documents_by_id))
        extra = sorted(set(documents_by_id).difference(hits_by_id))
        raise ValueError(f"search/document IDs are misaligned; missing={missing}, extra={extra}")

    rows: list[tuple[SearchHit, RetrievedDocument]] = []
    for document_id, hit in hits_by_id.items():
        document = documents_by_id[document_id]
        if document.rank != hit.rank or document.score != hit.score:
            raise ValueError(f"search/document rank or score is misaligned: {document_id}")
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
        if key_factory is not None and not callable(key_factory) and not isinstance(key_factory, str):
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
            raise PassageValidationError("chunker returned a passage for another document")
        if passage.representation != representation:
            raise PassageValidationError("chunker returned another representation")
        if passage.language != language:
            raise PassageValidationError("chunker returned another language")
        if passage.chunker_fingerprint != chunker_fingerprint:
            raise PassageValidationError("chunker returned another fingerprint")
        if passage.source_content_hash != expected_source_hash:
            raise PassageValidationError("passage source hash does not match canonical text")
        if passage.representation_hash != expected_representation_hash:
            raise PassageValidationError("passage representation hash does not match selected text")


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
    passage_scores = tuple(float(scores[passage.content_hash]) for passage in prepared.passages)
    canonical_text = prepared.canonical_text
    selected_text = prepared.selected_text
    source_language = document.language
    if prepared.status is not None:
        return RerankedDocument(
            topic_id=topic_id,
            document_id=document.document_id,
            status=prepared.status,
            organizer_rank=prepared.organizer_rank,
            organizer_score=prepared.organizer_score,
            score=None,
            winning_passage_id=None,
            winning_content_hash=None,
            passage_ids=passage_ids,
            passage_hashes=passage_hashes,
            passage_scores=passage_scores,
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
        (float(scores[passage.content_hash]), passage)
        for passage in prepared.passages
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
        organizer_rank=prepared.organizer_rank,
        organizer_score=prepared.organizer_score,
        score=winning_score,
        winning_passage_id=winning_passage.passage_id,
        winning_content_hash=winning_passage.content_hash,
        passage_ids=passage_ids,
        passage_hashes=passage_hashes,
        passage_scores=passage_scores,
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
            if selected_language is None or not isinstance(selected_language, str) or not selected_language.strip():
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
        if passages is None or isinstance(passages, (str, bytes, bytearray)) or not isinstance(passages, Sequence):
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
        passages_to_score = tuple(unique_passages[key] for key in sorted(unique_passages))
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
    scored = sorted(
        (document for document in reranked_documents if document.status == "scored"),
        key=lambda document: (
            -document.score,  # type: ignore[operator]
            document.organizer_rank,
            document.document_id,
            document.winning_content_hash,  # type: ignore[arg-type]
        ),
    )
    unscored = sorted(
        (document for document in reranked_documents if document.status != "scored"),
        key=lambda document: (document.organizer_rank, document.document_id),
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
        documents=tuple(scored + unscored),
    )


def _json_record(document: RerankedDocument) -> dict[str, object]:
    return {
        "aggregation": document.aggregation,
        "cache_fingerprint": document.cache_fingerprint,
        "cache_key": document.cache_key,
        "canonical_text": document.canonical_text,
        "content_local_cache_hit": document.content_local_cache_hit,
        "content_source": document.content_source,
        "date": document.date,
        "document_id": document.document_id,
        "english_text": document.english_text,
        "language": document.language,
        "organizer_rank": document.organizer_rank,
        "organizer_score": document.organizer_score,
        "passage_hashes": [value.hex() for value in document.passage_hashes],
        "passage_ids": [value.hex() for value in document.passage_ids],
        "passage_scores": list(document.passage_scores),
        "report_text": document.report_text,
        "score": document.score,
        "selected_text": document.selected_text,
        "source_language": document.source_language,
        "status": document.status,
        "topic_id": document.topic_id,
        "url": document.url,
        "winning_content_hash": (
            None if document.winning_content_hash is None else document.winning_content_hash.hex()
        ),
        "winning_passage_id": (
            None if document.winning_passage_id is None else document.winning_passage_id.hex()
        ),
    }


def _sha256_file_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_fsync(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_reranking_artifact(
    destination: str | os.PathLike[str],
    result: RerankingResult,
    *,
    source_retrieval_ref: str,
) -> None:
    """Write one immutable reranking run using a sibling-directory rename."""

    if not isinstance(result, RerankingResult):
        raise TypeError("result must be a RerankingResult")
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
                _json_record(document),
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
        if os.path.lexists(destination_path):
            raise FileExistsError(f"reranking artifact already exists: {destination_path}")
        os.replace(temporary_path, destination_path)
        temporary_path = None
        _fsync_directory(parent)
    finally:
        if temporary_path is not None:
            shutil.rmtree(temporary_path, ignore_errors=True)


__all__ = [
    "RERANKED_RECORD_FILENAME",
    "RERANKING_SCHEMA",
    "RERANKING_SCHEMA_VERSION",
    "RerankedDocument",
    "RerankingResult",
    "max_passage_v1",
    "rerank_retrieval",
    "write_reranking_artifact",
]
