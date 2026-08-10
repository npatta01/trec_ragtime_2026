"""Validated, deterministic output models for the RAGTIME 2026 tasks.

This module deliberately uses only the Python standard library.  It turns
report-generation output into the official JSONL shape and multilingual
retrieval output into the standard six-column TREC run format.
"""

from __future__ import annotations

import json
import math
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from os import PathLike, fsync
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import MappingProxyType
from typing import Self

MAX_RUN_ID_LENGTH = 25
MAX_MLIR_HITS = 1_000


class SubmissionValidationError(ValueError):
    """Raised when output would violate a RAGTIME submission contract."""


def _atomic_write_bytes(path: str | PathLike[str], payload: bytes) -> None:
    """Write bytes through an adjacent file, preserving an existing destination."""

    destination = Path(path).expanduser()
    stage: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            stage = Path(handle.name)
            handle.write(payload)
            handle.flush()
            fsync(handle.fileno())
        stage.replace(destination)
    finally:
        if stage is not None:
            try:
                stage.unlink(missing_ok=True)
            except OSError:
                # A complete adjacent staging file is safer to leave behind
                # than allowing cleanup to hide the write/publication error.
                pass


def _require_token(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise SubmissionValidationError(f"{field_name} must be a string")
    if not value:
        raise SubmissionValidationError(f"{field_name} must not be empty")
    if any(character.isspace() for character in value):
        raise SubmissionValidationError(
            f"{field_name} must not contain whitespace: {value!r}"
        )
    return value


def _require_description(value: object) -> str:
    if not isinstance(value, str):
        raise SubmissionValidationError("run_desc must be a string")
    if not value.strip():
        raise SubmissionValidationError("run_desc must not be empty")
    if value != value.strip():
        raise SubmissionValidationError(
            "run_desc must not have leading or trailing whitespace"
        )
    return value


def _require_run_id(value: object) -> str:
    run_id = _require_token(value, "run_id")
    if len(run_id) > MAX_RUN_ID_LENGTH:
        raise SubmissionValidationError(
            f"run_id must contain at most {MAX_RUN_ID_LENGTH} characters"
        )
    return run_id


def _require_score(value: object, field_name: str = "score") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SubmissionValidationError(
            f"{field_name} must be an int or float, not {type(value).__name__}"
        )
    try:
        score = float(value)
    except (OverflowError, ValueError) as error:
        raise SubmissionValidationError(f"{field_name} must be finite") from error
    if not math.isfinite(score):
        raise SubmissionValidationError(f"{field_name} must be finite")
    # Avoid two textual representations for a tied zero score.
    return 0.0 if score == 0.0 else score


def _require_nonnegative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SubmissionValidationError(f"{field_name} must be an integer")
    if value < 0:
        raise SubmissionValidationError(f"{field_name} must not be negative")
    return value


@dataclass(frozen=True, slots=True)
class ReportMetadata:
    """Official metadata included with every report topic."""

    team_id: str
    topic_id: str
    run_id: str
    run_desc: str

    def __post_init__(self) -> None:
        _require_token(self.team_id, "team_id")
        _require_token(self.topic_id, "topic_id")
        _require_run_id(self.run_id)
        _require_description(self.run_desc)

    def as_dict(self) -> dict[str, str]:
        """Return metadata in official field order."""

        return {
            "team_id": self.team_id,
            "topic_id": self.topic_id,
            "run_id": self.run_id,
            "run_desc": self.run_desc,
        }


@dataclass(frozen=True, slots=True)
class ReportSentence:
    """One report sentence and its scored supporting document IDs."""

    text: str
    citations: Mapping[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise SubmissionValidationError("sentence text must be a string")
        if not self.text.strip():
            raise SubmissionValidationError("sentence text must not be empty")
        if self.text != self.text.strip():
            raise SubmissionValidationError(
                "sentence text must not have leading or trailing whitespace"
            )
        if not isinstance(self.citations, Mapping):
            raise SubmissionValidationError("citations must be a mapping")

        validated: list[tuple[str, float]] = []
        for document_id, raw_score in self.citations.items():
            document_id = _require_token(document_id, "citation document ID")
            score = _require_score(
                raw_score, f"citation score for document {document_id!r}"
            )
            validated.append((document_id, score))
        validated.sort(key=lambda item: (-item[1], item[0]))
        object.__setattr__(
            self,
            "citations",
            MappingProxyType(dict(validated)),
        )

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible sentence with deterministic citations."""

        return {"text": self.text, "citations": dict(self.citations)}


@dataclass(frozen=True, slots=True)
class ReportSubmission:
    """One validated report topic in the official JSONL schema."""

    metadata: ReportMetadata
    responses: tuple[ReportSentence, ...]
    references: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, ReportMetadata):
            raise SubmissionValidationError("metadata must be ReportMetadata")
        try:
            responses = tuple(self.responses)
        except TypeError as error:
            raise SubmissionValidationError(
                "responses must be an iterable of ReportSentence values"
            ) from error
        if any(not isinstance(response, ReportSentence) for response in responses):
            raise SubmissionValidationError(
                "responses must contain only ReportSentence values"
            )
        if isinstance(self.references, (str, bytes)):
            raise SubmissionValidationError(
                "references must be an iterable of document IDs"
            )
        try:
            references = tuple(self.references)
        except TypeError as error:
            raise SubmissionValidationError(
                "references must be an iterable of document IDs"
            ) from error
        validated_references = tuple(
            _require_token(reference, "reference document ID")
            for reference in references
        )
        if len(set(validated_references)) != len(validated_references):
            raise SubmissionValidationError("references must not contain duplicates")

        citation_union = {
            document_id
            for response in responses
            for document_id in response.citations
        }
        reference_set = set(validated_references)
        if reference_set != citation_union:
            missing = sorted(citation_union - reference_set)
            extra = sorted(reference_set - citation_union)
            details: list[str] = []
            if missing:
                details.append(f"missing {missing!r}")
            if extra:
                details.append(f"unexpected {extra!r}")
            raise SubmissionValidationError(
                "references must exactly equal the citation union ("
                + "; ".join(details)
                + ")"
            )

        object.__setattr__(self, "responses", responses)
        object.__setattr__(self, "references", tuple(sorted(reference_set)))

    @classmethod
    def build(
        cls,
        *,
        metadata: ReportMetadata,
        responses: Iterable[ReportSentence],
    ) -> Self:
        """Build a report while deriving the exact reference union."""

        try:
            frozen_responses = tuple(responses)
        except TypeError as error:
            raise SubmissionValidationError(
                "responses must be an iterable of ReportSentence values"
            ) from error
        if any(
            not isinstance(response, ReportSentence)
            for response in frozen_responses
        ):
            raise SubmissionValidationError(
                "responses must contain only ReportSentence values"
            )
        references = tuple(
            sorted(
                {
                    document_id
                    for response in frozen_responses
                    for document_id in response.citations
                }
            )
        )
        return cls(
            metadata=metadata,
            responses=frozen_responses,
            references=references,
        )

    @property
    def report_text(self) -> str:
        """Join sentence texts exactly as the official validator does."""

        return " ".join(response.text for response in self.responses)

    @property
    def nfkc_character_count(self) -> int:
        """Count NFKC-normalized characters in the complete report text."""

        return len(unicodedata.normalize("NFKC", self.report_text))

    def validate_limit(self, limit: int) -> None:
        """Raise when this report exceeds its topic-specific character limit."""

        validated_limit = _require_nonnegative_int(limit, "character limit")
        actual = self.nfkc_character_count
        if actual > validated_limit:
            raise SubmissionValidationError(
                f"report for topic {self.metadata.topic_id!r} exceeds its NFKC "
                f"character limit: {actual} > {validated_limit}"
            )

    def as_dict(self) -> dict[str, object]:
        """Return this report in the official JSON-compatible shape."""

        return {
            "metadata": self.metadata.as_dict(),
            "responses": [response.as_dict() for response in self.responses],
            "references": list(self.references),
        }


def _validated_topic_limits(topic_limits: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(topic_limits, Mapping):
        raise SubmissionValidationError("topic_limits must be a mapping")
    validated: dict[str, int] = {}
    for topic_id, limit in topic_limits.items():
        topic_id = _require_token(topic_id, "topic limit ID")
        validated[topic_id] = _require_nonnegative_int(
            limit, f"character limit for topic {topic_id!r}"
        )
    if not validated:
        raise SubmissionValidationError("topic_limits must not be empty")
    return validated


def validate_report_run(
    submissions: Iterable[ReportSubmission],
    *,
    topic_limits: Mapping[str, int],
) -> tuple[ReportSubmission, ...]:
    """Validate and canonically order one complete report run."""

    try:
        reports = tuple(submissions)
    except TypeError as error:
        raise SubmissionValidationError(
            "submissions must be an iterable of ReportSubmission values"
        ) from error
    if any(not isinstance(report, ReportSubmission) for report in reports):
        raise SubmissionValidationError(
            "submissions must contain only ReportSubmission values"
        )

    limits = _validated_topic_limits(topic_limits)
    topic_ids = [report.metadata.topic_id for report in reports]
    duplicate_ids = sorted(
        topic_id for topic_id in set(topic_ids) if topic_ids.count(topic_id) > 1
    )
    if duplicate_ids:
        raise SubmissionValidationError(
            f"report run contains duplicate topics: {duplicate_ids!r}"
        )

    actual_ids = set(topic_ids)
    expected_ids = set(limits)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_ids)
        raise SubmissionValidationError(
            "report topic set is incomplete or unexpected: "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )

    identities = {
        (
            report.metadata.team_id,
            report.metadata.run_id,
            report.metadata.run_desc,
        )
        for report in reports
    }
    if len(identities) > 1:
        raise SubmissionValidationError(
            "team_id, run_id, and run_desc must be consistent across a report run"
        )

    for report in reports:
        report.validate_limit(limits[report.metadata.topic_id])
    return tuple(sorted(reports, key=lambda report: report.metadata.topic_id))


def write_report_jsonl(
    path: str | PathLike[str],
    submissions: Iterable[ReportSubmission],
    *,
    topic_limits: Mapping[str, int],
) -> None:
    """Write one compact JSON object per line after full-run validation."""

    reports = validate_report_run(submissions, topic_limits=topic_limits)
    lines = [
        json.dumps(
            report.as_dict(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        for report in reports
    ]
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    _atomic_write_bytes(path, payload)


@dataclass(frozen=True, slots=True)
class MLIRRankedHit:
    """One ranked document in a TREC multilingual retrieval run."""

    rank: int
    document_id: str
    score: float

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int):
            raise SubmissionValidationError("MLIR rank must be an integer")
        if self.rank < 1:
            raise SubmissionValidationError("MLIR rank must be positive")
        _require_token(self.document_id, "MLIR document ID")
        object.__setattr__(self, "score", _require_score(self.score, "MLIR score"))


@dataclass(frozen=True, slots=True)
class MLIRRun:
    """A complete, deterministically ordered MLIR ranking for one topic."""

    topic_id: str
    run_id: str
    hits: tuple[MLIRRankedHit, ...]

    def __post_init__(self) -> None:
        _require_token(self.topic_id, "MLIR topic ID")
        _require_run_id(self.run_id)
        try:
            hits = tuple(self.hits)
        except TypeError as error:
            raise SubmissionValidationError(
                "MLIR hits must be an iterable of MLIRRankedHit values"
            ) from error
        if any(not isinstance(hit, MLIRRankedHit) for hit in hits):
            raise SubmissionValidationError(
                "MLIR hits must contain only MLIRRankedHit values"
            )
        if not hits:
            raise SubmissionValidationError(
                "an MLIR topic must contain at least one hit"
            )
        if len(hits) > MAX_MLIR_HITS:
            raise SubmissionValidationError(
                f"an MLIR topic may contain at most {MAX_MLIR_HITS} hits"
            )

        document_ids = [hit.document_id for hit in hits]
        if len(set(document_ids)) != len(document_ids):
            raise SubmissionValidationError(
                f"MLIR topic {self.topic_id!r} contains duplicate document IDs"
            )
        expected_ranks = tuple(range(1, len(hits) + 1))
        actual_ranks = tuple(hit.rank for hit in hits)
        if actual_ranks != expected_ranks:
            raise SubmissionValidationError(
                "MLIR ranks must be consecutive and start at 1"
            )
        deterministic = tuple(
            sorted(hits, key=lambda hit: (-hit.score, hit.document_id))
        )
        if hits != deterministic:
            raise SubmissionValidationError(
                "MLIR hits must be ordered by descending score and then document ID"
            )
        object.__setattr__(self, "hits", hits)

    @classmethod
    def from_scores(
        cls,
        *,
        topic_id: str,
        run_id: str,
        scores: Mapping[str, int | float]
        | Iterable[tuple[str, int | float]],
        max_hits: int = MAX_MLIR_HITS,
    ) -> Self:
        """Validate fused scores, rank them deterministically, and cap depth."""

        _require_token(topic_id, "MLIR topic ID")
        _require_run_id(run_id)
        if (
            isinstance(max_hits, bool)
            or not isinstance(max_hits, int)
            or max_hits < 1
            or max_hits > MAX_MLIR_HITS
        ):
            raise SubmissionValidationError(
                f"max_hits must be an integer from 1 through {MAX_MLIR_HITS}"
            )

        raw_items: Iterable[tuple[str, int | float]]
        raw_items = scores.items() if isinstance(scores, Mapping) else scores
        try:
            items = tuple(raw_items)
        except TypeError as error:
            raise SubmissionValidationError(
                "scores must be a mapping or iterable of (document_id, score) pairs"
            ) from error

        validated: list[tuple[str, float]] = []
        seen: set[str] = set()
        for index, item in enumerate(items):
            try:
                document_id, raw_score = item
            except (TypeError, ValueError) as error:
                raise SubmissionValidationError(
                    f"score entry {index} must contain document ID and score"
                ) from error
            document_id = _require_token(document_id, "MLIR document ID")
            if document_id in seen:
                raise SubmissionValidationError(
                    f"duplicate MLIR document ID: {document_id!r}"
                )
            seen.add(document_id)
            validated.append((document_id, _require_score(raw_score, "MLIR score")))
        if not validated:
            raise SubmissionValidationError(
                "an MLIR topic must contain at least one hit"
            )

        validated.sort(key=lambda item: (-item[1], item[0]))
        hits = tuple(
            MLIRRankedHit(rank=rank, document_id=document_id, score=score)
            for rank, (document_id, score) in enumerate(
                validated[:max_hits], start=1
            )
        )
        return cls(topic_id=topic_id, run_id=run_id, hits=hits)

    def as_lines(self) -> tuple[str, ...]:
        """Return standard six-field TREC run rows for this topic."""

        return tuple(
            f"{self.topic_id} Q0 {hit.document_id} {hit.rank} "
            f"{hit.score!r} {self.run_id}"
            for hit in self.hits
        )


def _validated_expected_topics(
    expected_topic_ids: Iterable[str] | None,
) -> set[str] | None:
    if expected_topic_ids is None:
        return None
    if isinstance(expected_topic_ids, (str, bytes)):
        raise SubmissionValidationError(
            "expected_topic_ids must be an iterable of topic IDs"
        )
    try:
        topic_ids = tuple(expected_topic_ids)
    except TypeError as error:
        raise SubmissionValidationError(
            "expected_topic_ids must be an iterable of topic IDs"
        ) from error
    validated = tuple(
        _require_token(topic_id, "expected MLIR topic ID") for topic_id in topic_ids
    )
    if len(set(validated)) != len(validated):
        raise SubmissionValidationError(
            "expected_topic_ids must not contain duplicates"
        )
    if not validated:
        raise SubmissionValidationError("expected_topic_ids must not be empty")
    return set(validated)


def validate_mlir_runs(
    runs: Iterable[MLIRRun],
    *,
    expected_topic_ids: Iterable[str] | None = None,
) -> tuple[MLIRRun, ...]:
    """Validate a run collection and order topics for contiguous output."""

    try:
        rankings = tuple(runs)
    except TypeError as error:
        raise SubmissionValidationError(
            "runs must be an iterable of MLIRRun values"
        ) from error
    if any(not isinstance(run, MLIRRun) for run in rankings):
        raise SubmissionValidationError("runs must contain only MLIRRun values")
    if not rankings:
        raise SubmissionValidationError("an MLIR run file must not be empty")

    topic_ids = [run.topic_id for run in rankings]
    duplicate_ids = sorted(
        topic_id for topic_id in set(topic_ids) if topic_ids.count(topic_id) > 1
    )
    if duplicate_ids:
        raise SubmissionValidationError(
            f"MLIR run contains duplicate topics: {duplicate_ids!r}"
        )
    run_ids = {run.run_id for run in rankings}
    if len(run_ids) != 1:
        raise SubmissionValidationError(
            "run_id must be consistent across an MLIR run file"
        )

    expected = _validated_expected_topics(expected_topic_ids)
    actual = set(topic_ids)
    if expected is not None and actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise SubmissionValidationError(
            "MLIR topic set is incomplete or unexpected: "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
    return tuple(sorted(rankings, key=lambda run: run.topic_id))


def write_mlir_run(
    path: str | PathLike[str],
    runs: Iterable[MLIRRun],
    *,
    expected_topic_ids: Iterable[str],
) -> None:
    """Write a complete deterministic, topic-contiguous six-column run."""

    if expected_topic_ids is None:
        raise SubmissionValidationError(
            "expected_topic_ids is required when writing an MLIR run"
        )
    rankings = validate_mlir_runs(runs, expected_topic_ids=expected_topic_ids)
    lines = [line for run in rankings for line in run.as_lines()]
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    _atomic_write_bytes(path, payload)


__all__ = [
    "MAX_MLIR_HITS",
    "MAX_RUN_ID_LENGTH",
    "MLIRRankedHit",
    "MLIRRun",
    "ReportMetadata",
    "ReportSentence",
    "ReportSubmission",
    "SubmissionValidationError",
    "validate_mlir_runs",
    "validate_report_run",
    "write_mlir_run",
    "write_report_jsonl",
]
