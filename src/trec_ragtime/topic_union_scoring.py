"""Score one deduplicated topic-level document union without rank fusion."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Callable

from .models import RetrievalResult, RetrievedDocument
from .passage_scoring import validate_passage_scores
from .passages import Passage, sha256_bytes


class TopicUnionScoringError(ValueError):
    """Raised when topic inputs or scorer output cannot be sealed safely."""


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TopicUnionScoringError(f"{name} must be a non-blank string")
    return value


@dataclass(frozen=True, slots=True)
class InformationNeed:
    need_id: str
    text: str

    def __post_init__(self) -> None:
        _text(self.need_id, "need_id")
        _text(self.text, "need text")


@dataclass(frozen=True, slots=True)
class RetrievalSource:
    task_id: str
    query: str
    object_key: str
    original_rank: int
    original_score: float


@dataclass(frozen=True, slots=True)
class TopicRetrievalInput:
    topic_id: str
    task_id: str
    query: str
    object_key: str
    result: RetrievalResult

    def __post_init__(self) -> None:
        _text(self.topic_id, "topic_id")
        _text(self.task_id, "task_id")
        _text(self.query, "query")
        _text(self.object_key, "object_key")
        if not isinstance(self.result, RetrievalResult):
            raise TopicUnionScoringError("result must be a RetrievalResult")


@dataclass(frozen=True, slots=True)
class UnionDocument:
    document: RetrievedDocument
    sources: tuple[RetrievalSource, ...]


@dataclass(frozen=True, slots=True)
class ScoredTopicDocument:
    document_id: str
    score: float
    winning_need_id: str
    passage_start: int
    passage_end: int
    passage_content_hash: bytes
    sources: tuple[RetrievalSource, ...]


@dataclass(frozen=True, slots=True)
class TopicScoreResult:
    topic_id: str
    representation: str
    needs: tuple[InformationNeed, ...]
    input_object_keys: tuple[str, ...]
    unique_document_count: int
    passage_count: int
    documents: tuple[ScoredTopicDocument, ...]
    scorer_fingerprint: str
    chunker_fingerprint: str


def select_conservative_pair_length(
    config: object,
    needs: Sequence[InformationNeed],
    *,
    factory: Callable[[object, str], object] | None = None,
) -> object:
    """Choose the need leaving the smallest safe passage-token budget."""

    rows = tuple(needs)
    if not rows:
        raise TopicUnionScoringError("at least one information need is required")
    if factory is None:
        from .passage_scoring import CrossEncoderPairLengthFunction

        factory = CrossEncoderPairLengthFunction  # type: ignore[assignment]
    pair_lengths = tuple(factory(config, row.text) for row in rows)
    for pair_length in pair_lengths:
        budget = getattr(pair_length, "max_passage_length", None)
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
            raise TopicUnionScoringError("pair length has no positive passage budget")
    return min(
        pair_lengths,
        key=lambda row: (
            getattr(row, "max_passage_length"),
            getattr(row, "fingerprint", ""),
        ),
    )


def _same_content(left: RetrievedDocument, right: RetrievedDocument) -> bool:
    return (
        left.document_id == right.document_id
        and left.text == right.text
        and left.english_text == right.english_text
        and left.language == right.language
    )


def deduplicate_topic_documents(
    inputs: Sequence[TopicRetrievalInput],
) -> tuple[UnionDocument, ...]:
    """Union retrievals by canonical document ID, retaining rank as provenance only."""

    if not inputs:
        raise TopicUnionScoringError("at least one retrieval input is required")
    topic_id = inputs[0].topic_id
    documents: dict[str, RetrievedDocument] = {}
    sources: dict[str, list[RetrievalSource]] = {}
    seen_inputs: set[tuple[str, str]] = set()
    for item in inputs:
        if item.topic_id != topic_id:
            raise TopicUnionScoringError("retrieval inputs must belong to one topic")
        if item.result.query != item.query:
            raise TopicUnionScoringError("retrieval query does not match input lineage")
        if item.result.hydration_failures or item.result.missing_translation_ids:
            raise TopicUnionScoringError("retrieval hydration is incomplete")
        identity = (item.task_id, item.object_key)
        if identity in seen_inputs:
            raise TopicUnionScoringError("duplicate retrieval input identity")
        seen_inputs.add(identity)
        for document in item.result.documents:
            if document.text is None:
                raise TopicUnionScoringError("retrieval hydration is incomplete")
            if not math.isfinite(float(document.score)):
                raise TopicUnionScoringError("retrieval score must be finite")
            prior = documents.get(document.document_id)
            if prior is not None and not _same_content(prior, document):
                raise TopicUnionScoringError(
                    f"conflicting duplicate content for {document.document_id}"
                )
            documents.setdefault(document.document_id, document)
            sources.setdefault(document.document_id, []).append(
                RetrievalSource(
                    task_id=item.task_id,
                    query=item.query,
                    object_key=item.object_key,
                    original_rank=document.rank,
                    original_score=float(document.score),
                )
            )
    return tuple(
        UnionDocument(documents[document_id], tuple(sources[document_id]))
        for document_id in sorted(documents)
    )


def _chunk_documents(
    rows: Sequence[UnionDocument], *, chunker: object, representation: str
) -> tuple[tuple[Passage, ...], dict[str, tuple[Passage, ...]]]:
    if representation not in {"canonical", "report_text"}:
        raise TopicUnionScoringError("representation must be canonical or report_text")
    chunk = getattr(chunker, "chunk", None)
    if not callable(chunk):
        raise TopicUnionScoringError("chunker must expose chunk")
    unique: dict[bytes, Passage] = {}
    by_document: dict[str, tuple[Passage, ...]] = {}
    for row in rows:
        document = row.document
        selected = document.text if representation == "canonical" else document.report_text
        if not isinstance(selected, str) or not selected:
            raise TopicUnionScoringError(
                f"document {document.document_id} lacks {representation} content"
            )
        language = document.language if representation == "canonical" else "eng"
        if not isinstance(language, str) or not language:
            raise TopicUnionScoringError("selected representation lacks a language")
        passages = tuple(
            chunk(
                document_id=document.document_id,
                canonical_text=document.text,
                selected_text=selected,
                representation=representation,
                language=language,
            )
        )
        if not passages:
            raise TopicUnionScoringError(
                f"document {document.document_id} produced no passages"
            )
        for passage in passages:
            if not isinstance(passage, Passage):
                raise TopicUnionScoringError("chunker returned a non-Passage value")
            if passage.document_id != document.document_id:
                raise TopicUnionScoringError("passage document lineage is invalid")
            if selected[passage.start_char : passage.end_char] != passage.text:
                raise TopicUnionScoringError("passage offsets do not match source text")
            if sha256_bytes(passage.text) != passage.content_hash:
                raise TopicUnionScoringError("passage content hash is invalid")
            prior = unique.get(passage.content_hash)
            if prior is not None and prior.text != passage.text:
                raise TopicUnionScoringError("passage hash collision")
            unique.setdefault(passage.content_hash, passage)
        by_document[document.document_id] = passages
    return tuple(unique.values()), by_document


def score_topic_union(
    topic_id: str,
    needs: Sequence[InformationNeed],
    inputs: Sequence[TopicRetrievalInput],
    *,
    chunker: object,
    scorer: object,
    representation: str,
) -> TopicScoreResult:
    """Score every unique passage against every need and keep each document's maximum."""

    _text(topic_id, "topic_id")
    need_rows = tuple(needs)
    if not need_rows or any(not isinstance(need, InformationNeed) for need in need_rows):
        raise TopicUnionScoringError("at least one InformationNeed is required")
    if len({need.need_id for need in need_rows}) != len(need_rows):
        raise TopicUnionScoringError("need IDs must be unique")
    if any(item.topic_id != topic_id for item in inputs):
        raise TopicUnionScoringError("topic ID does not match retrieval inputs")
    scorer_method = getattr(scorer, "score", None)
    scorer_fingerprint = getattr(scorer, "fingerprint", None)
    chunker_fingerprint = getattr(chunker, "fingerprint", None)
    if not callable(scorer_method) or not isinstance(scorer_fingerprint, str):
        raise TopicUnionScoringError("scorer must expose fingerprint and score")
    if not isinstance(chunker_fingerprint, str) or not chunker_fingerprint:
        raise TopicUnionScoringError("chunker fingerprint is required")

    union = deduplicate_topic_documents(inputs)
    unique_passages, by_document = _chunk_documents(
        union, chunker=chunker, representation=representation
    )
    scores_by_need: dict[str, Mapping[bytes, float]] = {}
    for need in need_rows:
        try:
            scores_by_need[need.need_id] = validate_passage_scores(
                unique_passages, scorer_method(need.text, unique_passages)
            )
        except Exception as error:
            raise TopicUnionScoringError(
                f"scorer returned invalid scores for need {need.need_id}"
            ) from error

    scored: list[ScoredTopicDocument] = []
    union_by_id = {row.document.document_id: row for row in union}
    for document_id, passages in by_document.items():
        candidates = [
            (float(scores_by_need[need.need_id][passage.content_hash]), need, passage)
            for need in need_rows
            for passage in passages
        ]
        score, winning_need, winning_passage = min(
            candidates,
            key=lambda row: (
                -row[0], row[1].need_id, row[2].start_char,
                row[2].end_char, row[2].content_hash,
            ),
        )
        scored.append(
            ScoredTopicDocument(
                document_id=document_id,
                score=score,
                winning_need_id=winning_need.need_id,
                passage_start=winning_passage.start_char,
                passage_end=winning_passage.end_char,
                passage_content_hash=winning_passage.content_hash,
                sources=union_by_id[document_id].sources,
            )
        )
    scored.sort(key=lambda row: (-row.score, row.document_id))
    return TopicScoreResult(
        topic_id=topic_id,
        representation=representation,
        needs=need_rows,
        input_object_keys=tuple(sorted({item.object_key for item in inputs})),
        unique_document_count=len(union),
        passage_count=sum(len(passages) for passages in by_document.values()),
        documents=tuple(scored),
        scorer_fingerprint=scorer_fingerprint,
        chunker_fingerprint=chunker_fingerprint,
    )


__all__ = [
    "InformationNeed",
    "RetrievalSource",
    "ScoredTopicDocument",
    "TopicRetrievalInput",
    "TopicScoreResult",
    "TopicUnionScoringError",
    "UnionDocument",
    "deduplicate_topic_documents",
    "select_conservative_pair_length",
    "score_topic_union",
]
