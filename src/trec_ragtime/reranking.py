"""Compact deterministic passage reranking."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from .models import RetrievalResult, RetrievedDocument, SearchHit
from .passage_scoring import PassageScorer, validate_passage_scores
from .passages import Passage, PassageChunker, sha256_bytes


max_passage_v1 = "max_passage_v1"
_Status = Literal["scored", "missing_representation", "unresolved", "no_passages"]


class RerankingError(ValueError):
    """Raised when retrieval rows, selected text, or reranking results are invalid."""


@dataclass(frozen=True, slots=True)
class PassageEvidence:
    document_id: str
    start_char: int
    end_char: int
    score: float
    content_hash: bytes


@dataclass(frozen=True, slots=True)
class RerankedDocument:
    document: RetrievedDocument
    score: float | None
    evidence: tuple[PassageEvidence, ...]
    reranked_rank: int
    status: _Status


@dataclass(frozen=True, slots=True)
class RerankingResult:
    query: str
    representation: Literal["canonical", "report_text"]
    documents: tuple[RerankedDocument, ...]


def rerank_retrieval(
    retrieval: RetrievalResult,
    *,
    chunker: PassageChunker,
    scorer: PassageScorer,
    representation: Literal["canonical", "report_text"],
) -> RerankingResult:
    """Select, score, and deterministically rank passages from one retrieval."""

    if not isinstance(retrieval, RetrievalResult):
        raise RerankingError("retrieval must be a RetrievalResult")
    if representation not in {"canonical", "report_text"}:
        raise RerankingError("representation must be canonical or report_text")
    if not callable(getattr(chunker, "chunk", None)):
        raise RerankingError("chunker must expose a callable chunk method")
    if not callable(getattr(scorer, "score", None)):
        raise RerankingError("scorer must expose a callable score method")

    query = retrieval.query
    if not isinstance(query, str) or not query.strip():
        raise RerankingError("retrieval query must be a non-blank string")
    hits = retrieval.search_result.hits
    documents = retrieval.documents
    if (
        isinstance(hits, (str, bytes, bytearray))
        or not isinstance(hits, Sequence)
        or isinstance(documents, (str, bytes, bytearray))
        or not isinstance(documents, Sequence)
    ):
        raise RerankingError("retrieval hits and documents must be sequences")

    hits_by_id: dict[str, SearchHit] = {}
    hit_ranks: set[int] = set()
    for hit in hits:
        if not isinstance(hit, SearchHit):
            raise RerankingError("retrieval hits must contain SearchHit values")
        if (
            not isinstance(hit.document_id, str)
            or not hit.document_id.strip()
            or isinstance(hit.rank, bool)
            or not isinstance(hit.rank, int)
            or hit.rank <= 0
            or isinstance(hit.score, bool)
            or not isinstance(hit.score, (int, float))
            or not math.isfinite(float(hit.score))
            or hit.document_id in hits_by_id
            or hit.rank in hit_ranks
        ):
            raise RerankingError("retrieval hits are invalid or duplicated")
        hits_by_id[hit.document_id] = hit
        hit_ranks.add(hit.rank)

    documents_by_id: dict[str, RetrievedDocument] = {}
    document_ranks: set[int] = set()
    for document in documents:
        if not isinstance(document, RetrievedDocument):
            raise RerankingError(
                "retrieval documents must contain RetrievedDocument values"
            )
        if (
            not isinstance(document.document_id, str)
            or not document.document_id.strip()
            or isinstance(document.rank, bool)
            or not isinstance(document.rank, int)
            or document.rank <= 0
            or isinstance(document.score, bool)
            or not isinstance(document.score, (int, float))
            or not math.isfinite(float(document.score))
            or document.text is not None
            and not isinstance(document.text, str)
            or document.english_text is not None
            and not isinstance(document.english_text, str)
            or document.document_id in documents_by_id
            or document.rank in document_ranks
        ):
            raise RerankingError("retrieval documents are invalid or duplicated")
        documents_by_id[document.document_id] = document
        document_ranks.add(document.rank)

    if set(hits_by_id) != set(documents_by_id):
        raise RerankingError("search hits and documents are misaligned")
    rows: list[tuple[SearchHit, RetrievedDocument]] = []
    for document_id, hit in hits_by_id.items():
        document = documents_by_id[document_id]
        if document.rank != hit.rank or document.score != hit.score:
            raise RerankingError("search hit and document rank or score is misaligned")
        rows.append((hit, document))
    rows.sort(key=lambda row: (row[0].rank, row[0].document_id))

    prepared: list[tuple[RetrievedDocument, _Status | None, tuple[Passage, ...]]] = []
    unique_passages: dict[bytes, Passage] = {}
    for _, document in rows:
        canonical_text = document.text
        if canonical_text is None:
            prepared.append((document, "unresolved", ()))
            continue
        if not canonical_text:
            raise RerankingError("document text must not be empty")

        if representation == "canonical":
            selected_text = canonical_text
            language = document.language
            if not isinstance(language, str) or not language.strip():
                prepared.append((document, "missing_representation", ()))
                continue
        else:
            selected_text = document.report_text
            language = "eng"
            if selected_text is None:
                prepared.append((document, "missing_representation", ()))
                continue
            if not isinstance(selected_text, str) or not selected_text:
                raise RerankingError("selected report_text must be a non-empty string")

        try:
            passages = chunker.chunk(
                document_id=document.document_id,
                canonical_text=canonical_text,
                selected_text=selected_text,
                representation=representation,
                language=language,
            )
        except Exception as error:
            raise RerankingError(
                "chunker failed for a selected representation"
            ) from error
        if (
            passages is None
            or isinstance(passages, (str, bytes, bytearray))
            or not isinstance(passages, Sequence)
        ):
            raise RerankingError("chunker must return a sequence of passages")

        passage_tuple = tuple(passages)
        if not passage_tuple:
            prepared.append((document, "no_passages", ()))
            continue
        for passage in passage_tuple:
            if not isinstance(passage, Passage):
                raise RerankingError("chunker returned a non-Passage value")
            if (
                passage.document_id != document.document_id
                or passage.representation != representation
                or passage.language != language
                or passage.start_char < 0
                or passage.end_char > len(selected_text)
                or selected_text[passage.start_char : passage.end_char] != passage.text
                or passage.content_hash != sha256_bytes(passage.text)
            ):
                raise RerankingError("chunker returned invalid source-offset evidence")
            unique_passages.setdefault(passage.content_hash, passage)
        prepared.append((document, None, passage_tuple))

    scores: Mapping[bytes, float] = {}
    if unique_passages:
        passages_to_score = tuple(unique_passages.values())
        try:
            scores = validate_passage_scores(
                passages_to_score,
                scorer.score(query, passages_to_score),
            )
        except Exception as error:
            raise RerankingError("scorer returned invalid passage scores") from error

    scored_documents: list[RerankedDocument] = []
    status_documents: list[RerankedDocument] = []
    for document, status, passages in prepared:
        if status is not None:
            status_documents.append(RerankedDocument(document, None, (), 0, status))
            continue
        evidence_by_location: dict[tuple[str, int, int], PassageEvidence] = {}
        for passage in passages:
            evidence_by_location.setdefault(
                (passage.document_id, passage.start_char, passage.end_char),
                PassageEvidence(
                    document_id=passage.document_id,
                    start_char=passage.start_char,
                    end_char=passage.end_char,
                    score=float(scores[passage.content_hash]),
                    content_hash=passage.content_hash,
                ),
            )
        evidence = tuple(
            sorted(
                evidence_by_location.values(),
                key=lambda item: (item.start_char, item.end_char, item.content_hash),
            )
        )
        maximum_score = max(item.score for item in evidence)
        winning_evidence = min(
            (item for item in evidence if item.score == maximum_score),
            key=lambda item: (item.start_char, item.end_char, item.content_hash),
        )
        scored_documents.append(
            RerankedDocument(document, winning_evidence.score, evidence, 0, "scored")
        )

    scored_documents.sort(
        key=lambda row: (-float(row.score), row.document.rank, row.document.document_id)
    )
    status_documents.sort(key=lambda row: (row.document.rank, row.document.document_id))
    ordered_documents = scored_documents + status_documents
    ranked_documents = tuple(
        RerankedDocument(
            row.document,
            row.score,
            row.evidence,
            index,
            row.status,
        )
        for index, row in enumerate(ordered_documents, start=1)
    )
    return RerankingResult(query, representation, ranked_documents)


__all__ = [
    "PassageEvidence",
    "RerankedDocument",
    "RerankingError",
    "RerankingResult",
    "max_passage_v1",
    "rerank_retrieval",
]
