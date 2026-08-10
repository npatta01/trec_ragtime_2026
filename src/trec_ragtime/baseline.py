"""Citation-safe extractive report baseline for the multilingual corpus."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from ._text_boundaries import continues_after_period, has_paragraph_break
from .multiretriever import FusedDocument, MultiQueryRetrievalResult
from .submission import (
    ReportMetadata,
    ReportSentence,
    ReportSubmission,
)
from .topics import TopicRequest


_SENTENCE_BREAK = re.compile(
    r"(?P<terminal>[.!?][\"'”’\)\]]*)"
    r"(?P<separator>[ \t\f\v\r\n]+)|"
    r"(?P<paragraph>(?:\r\n?|\n)[ \t]*(?:\r\n?|\n)+)"
)
_TOKEN = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)
_ENGLISH_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "after",
        "all",
        "also",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "been",
        "before",
        "but",
        "by",
        "can",
        "could",
        "do",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "include",
        "information",
        "into",
        "is",
        "it",
        "its",
        "me",
        "need",
        "of",
        "on",
        "or",
        "other",
        "report",
        "should",
        "that",
        "the",
        "their",
        "them",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
    }
)


@dataclass(frozen=True, slots=True)
class EvidenceSentence:
    """One English sentence copied from a hydrated collection document."""

    document_id: str
    language: str
    text: str
    document_rank: int
    document_score: float
    sentence_index: int
    selection_score: float
    matched_terms: tuple[str, ...]
    query_ids: tuple[str, ...]
    translation_used: bool
    url: str | None = None
    date: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str) or not self.document_id:
            raise ValueError("document_id must be a non-empty string")
        if not isinstance(self.language, str) or not self.language:
            raise ValueError("language must be a non-empty string")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must be a non-empty string")
        if (
            isinstance(self.document_rank, bool)
            or not isinstance(self.document_rank, int)
            or self.document_rank < 1
        ):
            raise ValueError("document_rank must be a positive integer")
        if (
            isinstance(self.sentence_index, bool)
            or not isinstance(self.sentence_index, int)
            or self.sentence_index < 0
        ):
            raise ValueError("sentence_index must be a non-negative integer")
        for name, value in (
            ("document_score", self.document_score),
            ("selection_score", self.selection_score),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be a finite number")
        if not isinstance(self.translation_used, bool):
            raise ValueError("translation_used must be a boolean")
        if isinstance(self.matched_terms, (str, bytes)) or isinstance(
            self.query_ids, (str, bytes)
        ):
            raise ValueError("matched_terms and query_ids must be iterables")
        try:
            matched_terms = tuple(self.matched_terms)
            query_ids = tuple(self.query_ids)
        except TypeError as error:
            raise ValueError(
                "matched_terms and query_ids must be iterables"
            ) from error
        for name, values in (
            ("matched_terms", matched_terms),
            ("query_ids", query_ids),
        ):
            if any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"{name} must contain non-empty strings")
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicates")
        object.__setattr__(self, "matched_terms", matched_terms)
        object.__setattr__(self, "query_ids", query_ids)

    def as_record(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "language": self.language,
            "text": self.text,
            "document_rank": self.document_rank,
            "document_score": self.document_score,
            "sentence_index": self.sentence_index,
            "selection_score": self.selection_score,
            "matched_terms": list(self.matched_terms),
            "query_ids": list(self.query_ids),
            "translation_used": self.translation_used,
            "url": self.url,
            "date": self.date,
        }


@dataclass(frozen=True, slots=True)
class ExtractiveGenerationResult:
    report: ReportSubmission
    evidence: tuple[EvidenceSentence, ...]


def _tokens(text: str) -> frozenset[str]:
    return frozenset(
        token
        for token in _TOKEN.findall(unicodedata.normalize("NFKC", text).casefold())
        if len(token) > 1 and token not in _ENGLISH_STOPWORDS
    )


def _sentences(text: str) -> tuple[str, ...]:
    raw_parts: list[tuple[str, bool]] = []
    start = 0
    hard_boundary_before = False
    for match in _SENTENCE_BREAK.finditer(text):
        terminal = match.group("terminal")
        end = match.start() if terminal is None else match.start() + len(terminal)
        raw_parts.append((text[start:end], hard_boundary_before))
        separator = match.group("separator") or match.group("paragraph") or ""
        hard_boundary_before = has_paragraph_break(separator)
        start = match.end()
    raw_parts.append((text[start:], hard_boundary_before))

    repaired: list[str] = []
    for raw_part, hard_boundary in raw_parts:
        part = " ".join(raw_part.split()).strip()
        if not part:
            continue
        previous = repaired[-1] if repaired else ""
        if (
            repaired
            and not hard_boundary
            and continues_after_period(previous, part)
        ):
            repaired[-1] = f"{repaired[-1]} {part}"
        else:
            repaired.append(part)
    return tuple(repaired)


@dataclass(frozen=True, slots=True)
class ExtractiveReportGenerator:
    """Select collection sentences verbatim and attach their source IDs.

    This is a deliberately conservative control run: it favors citation
    support and format correctness over abstractive fluency. It is useful both
    as a valid emergency baseline and as a regression oracle for later LLM
    generation.
    """

    max_documents: int = 30
    max_sentences: int = 24
    max_sentences_per_document: int = 2
    min_sentence_characters: int = 40
    max_sentence_characters: int = 600
    min_term_overlap: int = 1

    def __post_init__(self) -> None:
        for name in (
            "max_documents",
            "max_sentences",
            "max_sentences_per_document",
            "min_sentence_characters",
            "max_sentence_characters",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_sentence_characters < self.min_sentence_characters:
            raise ValueError(
                "max_sentence_characters must not be smaller than the minimum"
            )
        if (
            isinstance(self.min_term_overlap, bool)
            or not isinstance(self.min_term_overlap, int)
            or self.min_term_overlap < 0
        ):
            raise ValueError("min_term_overlap must be a non-negative integer")

    def _candidates(
        self,
        topic: TopicRequest,
        retrieval: MultiQueryRetrievalResult,
    ) -> tuple[EvidenceSentence, ...]:
        topic_terms = _tokens(
            " ".join((topic.title, topic.background, topic.problem_statement))
        )
        query_terms = {
            query.query_id: _tokens(query.text)
            for query in retrieval.fused_ranking.queries
        }
        candidates: list[EvidenceSentence] = []
        seen_text: set[str] = set()

        for fused_document in retrieval.documents[: self.max_documents]:
            report_text = fused_document.report_text
            if report_text is None:
                continue
            for sentence_index, sentence in enumerate(_sentences(report_text)):
                character_count = len(unicodedata.normalize("NFKC", sentence))
                if not (
                    self.min_sentence_characters
                    <= character_count
                    <= self.max_sentence_characters
                ):
                    continue
                canonical_sentence = sentence.casefold()
                if canonical_sentence in seen_text:
                    continue
                sentence_terms = _tokens(sentence)
                matched_terms = tuple(sorted(sentence_terms.intersection(topic_terms)))
                if len(matched_terms) < self.min_term_overlap:
                    continue
                matched_query_ids = tuple(
                    query.query_id
                    for query in retrieval.fused_ranking.queries
                    if sentence_terms.intersection(query_terms[query.query_id])
                )
                score = (
                    float(len(matched_terms))
                    + 0.25 * len(matched_query_ids)
                    + 1.0 / (5.0 + fused_document.rank)
                )
                candidates.append(
                    self._evidence(
                        fused_document,
                        sentence=sentence,
                        sentence_index=sentence_index,
                        score=score,
                        matched_terms=matched_terms,
                        query_ids=matched_query_ids,
                    )
                )
                seen_text.add(canonical_sentence)

        candidates.sort(
            key=lambda item: (
                -item.selection_score,
                item.document_rank,
                item.document_id,
                item.sentence_index,
            )
        )
        return tuple(candidates)

    @staticmethod
    def _evidence(
        fused_document: FusedDocument,
        *,
        sentence: str,
        sentence_index: int,
        score: float,
        matched_terms: tuple[str, ...],
        query_ids: tuple[str, ...],
    ) -> EvidenceSentence:
        document = fused_document.document
        return EvidenceSentence(
            document_id=fused_document.document_id,
            language=fused_document.language,
            text=sentence,
            document_rank=fused_document.rank,
            document_score=fused_document.score,
            sentence_index=sentence_index,
            selection_score=score,
            matched_terms=matched_terms,
            query_ids=query_ids,
            translation_used=document.english_text is not None,
            url=document.url,
            date=document.date,
        )

    def _select(
        self,
        candidates: tuple[EvidenceSentence, ...],
        *,
        limit: int,
    ) -> tuple[EvidenceSentence, ...]:
        remaining = list(candidates)
        selected: list[EvidenceSentence] = []
        document_counts: Counter[str] = Counter()
        covered_terms: set[str] = set()

        while remaining and len(selected) < self.max_sentences:
            eligible = [
                candidate
                for candidate in remaining
                if document_counts[candidate.document_id]
                < self.max_sentences_per_document
            ]
            if not eligible:
                break
            eligible.sort(
                key=lambda item: (
                    -(
                        item.selection_score
                        + 0.35 * len(set(item.matched_terms).difference(covered_terms))
                    ),
                    item.document_rank,
                    item.document_id,
                    item.sentence_index,
                )
            )
            candidate = eligible[0]
            remaining.remove(candidate)
            proposed_text = " ".join(
                [*(item.text for item in selected), candidate.text]
            )
            if len(unicodedata.normalize("NFKC", proposed_text)) > limit:
                continue
            selected.append(candidate)
            document_counts[candidate.document_id] += 1
            covered_terms.update(candidate.matched_terms)
        return tuple(selected)

    def generate(
        self,
        topic: TopicRequest,
        retrieval: MultiQueryRetrievalResult,
        *,
        team_id: str,
        run_id: str,
        run_desc: str,
    ) -> ExtractiveGenerationResult:
        """Build one validator-ready, citation-bearing extractive report."""

        retrieval.require_report_ready()
        candidates = self._candidates(topic, retrieval)
        if not candidates and self.min_term_overlap:
            fallback = ExtractiveReportGenerator(
                max_documents=self.max_documents,
                max_sentences=self.max_sentences,
                max_sentences_per_document=self.max_sentences_per_document,
                min_sentence_characters=self.min_sentence_characters,
                max_sentence_characters=self.max_sentence_characters,
                min_term_overlap=0,
            )
            candidates = fallback._candidates(topic, retrieval)
        selected = self._select(candidates, limit=topic.limit)
        if not selected:
            raise ValueError(
                f"no extractive evidence fits topic {topic.topic_id!r} and its limit"
            )

        responses = tuple(
            ReportSentence(
                text=evidence.text,
                citations={
                    evidence.document_id: max(
                        evidence.selection_score,
                        1.0 / (1.0 + evidence.document_rank),
                    )
                },
            )
            for evidence in selected
        )
        report = ReportSubmission.build(
            metadata=ReportMetadata(
                team_id=team_id,
                topic_id=topic.topic_id,
                run_id=run_id,
                run_desc=run_desc,
            ),
            responses=responses,
        )
        report.validate_limit(topic.limit)
        return ExtractiveGenerationResult(report=report, evidence=selected)


__all__ = [
    "EvidenceSentence",
    "ExtractiveGenerationResult",
    "ExtractiveReportGenerator",
]
