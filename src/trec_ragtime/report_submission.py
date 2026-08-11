"""The official TREC RAGTIME 2026 report submission contract.

This module owns the wire format and nothing about how a report is produced, so
it can check any JSONL file — one this repository generated, one edited by hand,
or one from another tool. It is the local stand-in for the official validator,
which the track publishes in the TREC Active Participants area; re-check the
live guidelines before treating a run as submission-ready.

The character limit is enforced exactly as the guidelines state it: the count of
NFKC-normalized characters across the concatenated sentence text. The generator
deliberately packs against a slightly stricter budget (see
``report_sentences.budgeted_character_count``); that margin is a generation-side
safety choice and is not a validation rule.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .report_sentences import (
    SentenceSegmenter,
    default_segmenter,
    normalize_report_text,
    report_character_count,
)
from .topic_candidates import load_topics

SUBMISSION_FORMAT_VERSION = "ragtime-report-2026-v1"
MAX_RUN_ID_LENGTH = 25
MAX_TEAM_ID_LENGTH = 64
MAX_TOPIC_ID_LENGTH = 64
MAX_RUN_DESC_LENGTH = 1000

TOP_LEVEL_KEYS = frozenset({"metadata", "responses", "references"})
METADATA_KEYS = frozenset({"team_id", "topic_id", "run_id", "run_desc"})
RESPONSE_KEYS = frozenset({"text", "citations"})


class SubmissionValidationError(ValueError):
    """A submission field violates the official contract."""


@dataclass(frozen=True, slots=True)
class SubmissionViolation:
    """One contract violation, addressed to a line and topic when known."""

    code: str
    message: str
    line: int = 0
    topic_id: str | None = None

    def render(self) -> str:
        location = f"line {self.line}" if self.line else "file"
        topic = f" [{self.topic_id}]" if self.topic_id else ""
        return f"{location}{topic} {self.code}: {self.message}"


@dataclass(frozen=True, slots=True)
class SubmissionValidation:
    """The outcome of validating one submission file."""

    path_name: str
    record_count: int
    topic_ids: tuple[str, ...]
    violations: tuple[SubmissionViolation, ...]

    @property
    def ok(self) -> bool:
        return not self.violations


def _validate_identifier(
    value: Any,
    *,
    name: str,
    max_length: int,
    allow_spaces: bool = False,
) -> str:
    if not isinstance(value, str):
        raise SubmissionValidationError(f"{name} must be a string")
    if not value.strip():
        raise SubmissionValidationError(f"{name} must not be blank")
    if value != value.strip():
        raise SubmissionValidationError(f"{name} must not have leading or trailing space")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise SubmissionValidationError(f"{name} must not contain control characters")
    if not allow_spaces and any(character.isspace() for character in value):
        raise SubmissionValidationError(f"{name} must not contain whitespace")
    if len(unicodedata.normalize("NFKC", value)) > max_length:
        raise SubmissionValidationError(
            f"{name} must be at most {max_length} characters after NFKC normalization"
        )
    return value


def validate_team_id(team_id: Any) -> str:
    """Return a valid team identifier or raise."""

    return _validate_identifier(team_id, name="team_id", max_length=MAX_TEAM_ID_LENGTH)


def validate_run_id(run_id: Any) -> str:
    """Return a run identifier of at most 25 NFKC characters or raise."""

    return _validate_identifier(run_id, name="run_id", max_length=MAX_RUN_ID_LENGTH)


def validate_topic_id(topic_id: Any) -> str:
    """Return a valid topic identifier or raise."""

    return _validate_identifier(topic_id, name="topic_id", max_length=MAX_TOPIC_ID_LENGTH)


def validate_run_desc(run_desc: Any) -> str:
    """Return a valid run description or raise."""

    return _validate_identifier(
        run_desc,
        name="run_desc",
        max_length=MAX_RUN_DESC_LENGTH,
        allow_spaces=True,
    )


def _is_valid_score(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _ordered_citations(citations: Mapping[str, Any]) -> dict[str, float]:
    """Order citations strongest first so serialized output is deterministic."""

    items = sorted(citations.items(), key=lambda item: (-float(item[1]), item[0]))
    return {document_id: float(score) for document_id, score in items}


def submission_record(
    *,
    team_id: str,
    topic_id: str,
    run_id: str,
    run_desc: str,
    sentences: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one official report object from ordered sentence dictionaries.

    ``sentences`` carry ``text`` and an optional ``citations`` mapping of
    document ID to score. Structural validation happens here; semantic checks
    such as single-sentence segmentation and the character limit belong to
    :func:`validate_submission_record`.
    """

    metadata = {
        "team_id": validate_team_id(team_id),
        "topic_id": validate_topic_id(topic_id),
        "run_id": validate_run_id(run_id),
        "run_desc": validate_run_desc(run_desc),
    }
    items = list(sentences)
    if not items:
        raise SubmissionValidationError("a report must contain at least one sentence")

    responses: list[dict[str, Any]] = []
    references: set[str] = set()
    for position, item in enumerate(items, start=1):
        if not isinstance(item, Mapping):
            raise SubmissionValidationError(f"sentence {position} must be a mapping")
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            raise SubmissionValidationError(f"sentence {position} must have text")
        citations = item.get("citations") or {}
        if not isinstance(citations, Mapping):
            raise SubmissionValidationError(
                f"sentence {position} citations must be a mapping"
            )
        for document_id, score in citations.items():
            if not isinstance(document_id, str) or not document_id.strip():
                raise SubmissionValidationError(
                    f"sentence {position} has a blank citation key"
                )
            if not _is_valid_score(score):
                raise SubmissionValidationError(
                    f"sentence {position} citation {document_id} needs a finite score"
                )
        responses.append({"text": text, "citations": _ordered_citations(citations)})
        references.update(citations)

    return {
        "metadata": metadata,
        "responses": responses,
        "references": sorted(references),
    }


def _metadata_violations(metadata: Any) -> list[SubmissionViolation]:
    violations: list[SubmissionViolation] = []
    if not isinstance(metadata, Mapping):
        return [SubmissionViolation("metadata_type", "metadata must be an object")]
    if set(metadata) != METADATA_KEYS:
        violations.append(
            SubmissionViolation(
                "metadata_keys",
                f"metadata keys must be exactly {sorted(METADATA_KEYS)}",
            )
        )
    checks = (
        ("team_id", validate_team_id),
        ("topic_id", validate_topic_id),
        ("run_id", validate_run_id),
        ("run_desc", validate_run_desc),
    )
    for field, validator in checks:
        if field not in metadata:
            continue
        try:
            validator(metadata[field])
        except SubmissionValidationError as error:
            violations.append(SubmissionViolation(field, str(error)))
    return violations


def _response_violations(
    responses: Any, segmenter: SentenceSegmenter
) -> tuple[list[SubmissionViolation], list[str], set[str]]:
    violations: list[SubmissionViolation] = []
    texts: list[str] = []
    cited: set[str] = set()

    if not isinstance(responses, list):
        return (
            [SubmissionViolation("responses_type", "responses must be an array")],
            texts,
            cited,
        )
    if not responses:
        violations.append(
            SubmissionViolation("responses_empty", "responses must not be empty")
        )

    for position, item in enumerate(responses, start=1):
        if not isinstance(item, Mapping):
            violations.append(
                SubmissionViolation(
                    "response_type", f"response {position} must be an object"
                )
            )
            continue
        if "text" not in item or not set(item) <= RESPONSE_KEYS:
            violations.append(
                SubmissionViolation(
                    "response_keys",
                    f"response {position} keys must be text and optional citations",
                )
            )
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            violations.append(
                SubmissionViolation(
                    "text_type", f"response {position} text must be a non-empty string"
                )
            )
        else:
            texts.append(text)
            if normalize_report_text(text) != text:
                violations.append(
                    SubmissionViolation(
                        "text_not_normalized",
                        f"response {position} text is not NFKC-normalized and trimmed",
                    )
                )
            elif not segmenter.is_single_sentence(text):
                violations.append(
                    SubmissionViolation(
                        "not_single_sentence",
                        f"response {position} must be exactly one sentence",
                    )
                )

        citations = item.get("citations", {})
        if not isinstance(citations, Mapping):
            violations.append(
                SubmissionViolation(
                    "citations_type", f"response {position} citations must be an object"
                )
            )
            continue
        scores: list[float] = []
        for document_id, score in citations.items():
            if not isinstance(document_id, str) or not document_id.strip():
                violations.append(
                    SubmissionViolation(
                        "citation_key",
                        f"response {position} has a blank or non-string citation key",
                    )
                )
                continue
            if not _is_valid_score(score):
                violations.append(
                    SubmissionViolation(
                        "citation_score",
                        f"response {position} citation {document_id} needs a finite score",
                    )
                )
                continue
            scores.append(float(score))
            cited.add(document_id)
        if len(scores) != len(set(scores)):
            violations.append(
                SubmissionViolation(
                    "citation_score_tie",
                    f"response {position} citation scores must be distinct so"
                    " assessment priority is unambiguous",
                )
            )
    return violations, texts, cited


def _reference_violations(references: Any, cited: set[str]) -> list[SubmissionViolation]:
    if not isinstance(references, list) or not all(
        isinstance(item, str) and item.strip() for item in references
    ):
        return [
            SubmissionViolation(
                "references_type", "references must be an array of non-empty strings"
            )
        ]
    violations: list[SubmissionViolation] = []
    if len(references) != len(set(references)):
        violations.append(
            SubmissionViolation("references_duplicate", "references must be unique")
        )
    if set(references) != cited:
        violations.append(
            SubmissionViolation(
                "references_union",
                "references must equal the union of all citation keys",
            )
        )
    return violations


def validate_submission_record(
    record: Any,
    *,
    character_limit: int | None,
    segmenter: SentenceSegmenter | None = None,
    line: int = 0,
    topic_id: str | None = None,
) -> tuple[SubmissionViolation, ...]:
    """Return every contract violation in one parsed report object."""

    segmenter = segmenter or default_segmenter()
    if not isinstance(record, Mapping):
        found = type(record).__name__
        return (
            SubmissionViolation(
                "record_type", f"report must be a JSON object, found {found}", line, topic_id
            ),
        )

    violations: list[SubmissionViolation] = []
    if set(record) != TOP_LEVEL_KEYS:
        violations.append(
            SubmissionViolation(
                "top_level_keys",
                f"report keys must be exactly {sorted(TOP_LEVEL_KEYS)}",
            )
        )

    violations.extend(_metadata_violations(record.get("metadata")))
    response_violations, texts, cited = _response_violations(
        record.get("responses"), segmenter
    )
    violations.extend(response_violations)
    violations.extend(_reference_violations(record.get("references"), cited))

    if character_limit is not None and texts:
        count = report_character_count(texts)
        if count > character_limit:
            violations.append(
                SubmissionViolation(
                    "character_limit",
                    f"report is {count} NFKC characters, limit is {character_limit}",
                )
            )

    return tuple(
        replace(violation, line=line, topic_id=topic_id) for violation in violations
    )


def _record_topic_id(record: Any) -> str | None:
    if not isinstance(record, Mapping):
        return None
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    topic_id = metadata.get("topic_id")
    return topic_id if isinstance(topic_id, str) and topic_id.strip() else None


def _metadata_value(record: Any, field: str) -> str | None:
    if not isinstance(record, Mapping):
        return None
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get(field)
    return value if isinstance(value, str) else None


def validate_submission_file(
    path: Path | str,
    *,
    topics: Mapping[str, int],
    team_id: str | None = None,
    run_id: str | None = None,
    segmenter: SentenceSegmenter | None = None,
    strict_order: bool = False,
) -> SubmissionValidation:
    """Validate a submission JSONL file against the expected topic set."""

    path = Path(path)
    segmenter = segmenter or default_segmenter()
    violations: list[SubmissionViolation] = []
    seen: list[str] = []
    record_count = 0
    observed_team: str | None = None
    observed_run: str | None = None

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped:
                violations.append(
                    SubmissionViolation("blank_line", "blank lines are not allowed", line_number)
                )
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as error:
                violations.append(
                    SubmissionViolation("json_decode", str(error), line_number)
                )
                continue

            record_count += 1
            topic = _record_topic_id(record)
            limit = topics.get(topic) if topic is not None else None
            violations.extend(
                validate_submission_record(
                    record,
                    character_limit=limit,
                    segmenter=segmenter,
                    line=line_number,
                    topic_id=topic,
                )
            )

            if topic is None:
                continue
            if topic not in topics:
                violations.append(
                    SubmissionViolation(
                        "topic_unknown",
                        f"topic {topic} is not in the expected topic set",
                        line_number,
                        topic,
                    )
                )
            elif topic in seen:
                violations.append(
                    SubmissionViolation(
                        "topic_duplicate",
                        f"topic {topic} appears more than once",
                        line_number,
                        topic,
                    )
                )
            seen.append(topic)

            for field, expected, observed, code in (
                ("team_id", team_id, observed_team, "team_id_mismatch"),
                ("run_id", run_id, observed_run, "run_id_mismatch"),
            ):
                value = _metadata_value(record, field)
                if value is None:
                    continue
                if expected is not None and value != expected:
                    violations.append(
                        SubmissionViolation(
                            code,
                            f"{field} is {value!r}, expected {expected!r}",
                            line_number,
                            topic,
                        )
                    )
                if observed is not None and value != observed:
                    violations.append(
                        SubmissionViolation(
                            "metadata_inconsistent",
                            f"{field} {value!r} differs from {observed!r} on an earlier line",
                            line_number,
                            topic,
                        )
                    )
            observed_team = observed_team or _metadata_value(record, "team_id")
            observed_run = observed_run or _metadata_value(record, "run_id")

    if record_count == 0:
        violations.append(
            SubmissionViolation("file_empty", "the submission contains no reports")
        )

    for topic in topics:
        if topic not in seen:
            violations.append(
                SubmissionViolation("topic_missing", f"topic {topic} is missing")
            )

    if strict_order:
        expected_order = [topic for topic in topics if topic in set(seen)]
        observed_order = [topic for topic in seen if topic in topics]
        if observed_order != expected_order:
            violations.append(
                SubmissionViolation(
                    "topic_order",
                    "reports must appear in the topic file's order",
                )
            )

    return SubmissionValidation(
        path_name=path.name,
        record_count=record_count,
        topic_ids=tuple(seen),
        violations=tuple(violations),
    )


def topic_character_limits(path: Path | str) -> dict[str, int]:
    """Return ``{topic_id: character limit}`` from an official topic file."""

    return {topic.topic_id: topic.limit for topic in load_topics(Path(path))}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragtime-report-validate",
        description="Validate a RAGTIME report submission JSONL file.",
    )
    parser.add_argument("submission", help="Path to the submission JSONL file")
    parser.add_argument(
        "--topics", required=True, help="Path to the official topics JSONL file"
    )
    parser.add_argument("--team-id", default=None, help="Require this exact team_id")
    parser.add_argument("--run-id", default=None, help="Require this exact run_id")
    parser.add_argument(
        "--strict-order",
        action="store_true",
        help="Require reports in the topic file's order",
    )
    parser.add_argument(
        "--max-violations",
        type=int,
        default=50,
        help="Maximum violations to print (default: 50)",
    )
    return parser


def _print_violations(
    violations: Iterable[SubmissionViolation], *, limit: int
) -> None:
    items = list(violations)
    for violation in items[:limit]:
        print(violation.render())
    remaining = len(items) - limit
    if remaining > 0:
        print(f"... and {remaining} more violation(s)")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    submission = Path(args.submission)
    if not submission.is_file():
        print(f"submission file not found: {submission}", file=sys.stderr)
        return 2
    try:
        topics = topic_character_limits(args.topics)
    except (OSError, ValueError) as error:
        print(f"could not read topics: {error}", file=sys.stderr)
        return 2

    try:
        report = validate_submission_file(
            submission,
            topics=topics,
            team_id=args.team_id,
            run_id=args.run_id,
            strict_order=args.strict_order,
        )
    except OSError as error:
        print(f"could not read submission: {error}", file=sys.stderr)
        return 2

    if report.ok:
        print(f"validated {report.record_count} report(s) in {report.path_name}")
        return 0

    print(
        f"found {len(report.violations)} violation(s) in {report.path_name}"
        f" across {report.record_count} report(s)"
    )
    _print_violations(report.violations, limit=max(0, args.max_violations))
    return 1


__all__ = [
    "MAX_RUN_ID_LENGTH",
    "MAX_TEAM_ID_LENGTH",
    "METADATA_KEYS",
    "RESPONSE_KEYS",
    "SUBMISSION_FORMAT_VERSION",
    "TOP_LEVEL_KEYS",
    "SubmissionValidation",
    "SubmissionValidationError",
    "SubmissionViolation",
    "build_parser",
    "main",
    "submission_record",
    "topic_character_limits",
    "validate_run_desc",
    "validate_run_id",
    "validate_submission_file",
    "validate_submission_record",
    "validate_team_id",
    "validate_topic_id",
]


if __name__ == "__main__":
    raise SystemExit(main())
