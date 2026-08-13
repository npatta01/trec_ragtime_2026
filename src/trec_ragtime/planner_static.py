"""Sealed, deterministic static research planning for RAGTIME experiments.

This module deliberately stops at the static-plan seam.  A planner makes one
validated plan for a topic; task admission and retrieval execution never call
the planner again and never inspect search results when choosing later tasks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

from dotenv import load_dotenv
from openai import OpenAI

from .artifact_store import ArtifactStoreLike
from .cache import ResponseCache, canonicalize_json
from .models import RetrievalResult
from .passages import Passage
from .paths import default_project_cache_dir
from .providers import provider_settings
from .retrieval_cache import RetrievalArtifactReference


PLAN_SCHEMA_VERSION = "planner_static_plan_v2"
RUN_SCHEMA_VERSION = "planner_static_run_v2"
PLANNER_OPERATION = "planner-static-plan-v3"
COVERAGE_LEDGER_SCHEMA_VERSION = "planner_coverage_ledger_v1"
COVERAGE_LEDGER_OPERATION = "planner-static-coverage-ledger-v1"
COVERAGE_EXPANSION_OPERATION = "planner-static-coverage-expansion-v1"
COMPACT_EXPANSION_ASSEMBLER_VERSION = "compact-query-assembler-v2"
LEDGER_FALLBACK_VERSION = "ledger-fallback-v1"
COMPACT_EXPANSION_PROMPT_VERSIONS = frozenset(
    {"planner-static-expand-v2", "planner-static-expand-v3"}
)
DEFAULT_MAX_TOTAL_TASKS: int | None = None
DEFAULT_MAX_PLANNER_CALLS = 103
DEFAULT_PLANNER_MODEL = "deepseek/deepseek-v4-flash-20260423"
DEFAULT_PROMPT_VERSION = "planner-static-draft-v3"
DEFAULT_AUDIT_PROMPT_VERSION = "planner-static-audit-v4"
DEFAULT_LEDGER_PROMPT_VERSION = "planner-static-ledger-v1"
DEFAULT_EXPANSION_PROMPT_VERSION = "planner-static-expand-v1"
DEFAULT_PLANNER_MAX_TOKENS = 16_000
DEFAULT_AUDIT_MAX_TOKENS = 32_000
DEFAULT_TOPIC_PATH = Path("topics/topics.all.2026.v0625-fix.jsonl")
DEFAULT_TOP_K = 100
QUERY_ROLES = frozenset(
    {
        "broad_discovery",
        "precise_facts",
        "official_primary",
        "measurable_outcomes",
        "contrary_evidence",
        "comparison",
    }
)
_HEX64 = set("0123456789abcdef")
_WS = re.compile(r"\s+")
_SENTENCE_END = re.compile(r".*?(?:[.!?。！？](?:\s+|$)|$)", re.DOTALL)


def _text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-blank text")
    return _WS.sub(" ", value.strip())


def _positive_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        canonicalize_json(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_id(prefix: str, value: Any) -> str:
    return f"{prefix}-{_sha256_json(value)[:24]}"


def _validation_error_code(error: ValueError) -> str:
    """Map local validation messages to non-payload diagnostic codes."""

    message = str(error)
    checks = (
        ("duplicate normalized query", "duplicate_query"),
        ("duplicate requirement_ref", "duplicate_requirement_ref"),
        ("unique requirement_ref", "duplicate_requirement_ref"),
        ("duplicate task_ref", "duplicate_task_ref"),
        ("unknown requirement_ref", "unknown_requirement_ref"),
        ("unknown task_ref", "unknown_task_ref"),
        ("dependencies must refer to earlier", "invalid_dependency_order"),
        ("coverage-ledger plan tasks must be dependency-free", "nonempty_dependencies"),
        ("every planning requirement must be covered", "missing_requirement_coverage"),
        ("query_role is unsupported", "unsupported_query_role"),
        ("unknown plan field", "unknown_plan_field"),
        (
            "exact ledger requirement coverage",
            "ledger_requirement_coverage",
        ),
        ("compact query_sets must be a non-empty list", "malformed_query_sets"),
        ("compact query_sets entries must be objects", "malformed_query_set"),
        ("compact queries entries must be objects", "malformed_query"),
        ("compact query set must contain queries", "empty_query_set"),
        ("coverage ledger does not match topic", "topic_mismatch"),
        ("information need taskless", "duplicate_merge_taskless"),
    )
    for pattern, code in checks:
        if pattern in message:
            return code
    return "invalid_plan"


def _expect_keys(value: Mapping[str, Any], allowed: set[str], *, label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown plan field(s) in {label}: {', '.join(unknown)}")


@dataclass(frozen=True, slots=True)
class PlanningTopic:
    """The topic fields exposed to a planner."""

    topic_id: str
    title: str
    problem_statement: str
    background: str

    def __post_init__(self) -> None:
        for name in ("topic_id", "title", "problem_statement", "background"):
            object.__setattr__(self, name, _text(getattr(self, name), field_name=name))

    def as_record(self) -> dict[str, str]:
        return {
            "topic_id": self.topic_id,
            "title": self.title,
            "problem_statement": self.problem_statement,
            "background": self.background,
        }

    @property
    def input_sha256(self) -> str:
        return _sha256_json(self.as_record())


@dataclass(frozen=True, slots=True)
class CoverageLedgerRequirement:
    requirement_ref: str
    description: str
    priority: int
    retrieval_anchors: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "requirement_ref",
            _text(self.requirement_ref, field_name="coverage requirement_ref"),
        )
        object.__setattr__(
            self,
            "description",
            _text(self.description, field_name="coverage requirement description"),
        )
        _positive_int(self.priority, field_name="coverage requirement priority")
        anchors = tuple(
            _text(value, field_name="retrieval anchor")
            for value in self.retrieval_anchors
        )
        if not anchors:
            raise ValueError("coverage requirement must have retrieval anchors")
        if len(anchors) > 5:
            raise ValueError(
                "coverage requirement must have at most five retrieval anchors"
            )
        if len({value.casefold() for value in anchors}) != len(anchors):
            raise ValueError("coverage requirement retrieval anchors must be unique")
        object.__setattr__(self, "retrieval_anchors", anchors)


@dataclass(frozen=True, slots=True)
class CoverageLedgerNeed:
    need_ref: str
    objective: str
    why_it_matters: str
    priority: int
    requirements: tuple[CoverageLedgerRequirement, ...]

    def __post_init__(self) -> None:
        for name in ("need_ref", "objective", "why_it_matters"):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), field_name=f"coverage {name}"),
            )
        _positive_int(self.priority, field_name="coverage need priority")
        if not self.requirements:
            raise ValueError("coverage need must contain requirements")


@dataclass(frozen=True, slots=True)
class CoverageLedger:
    topic_id: str
    topic_sha256: str
    model: str
    prompt_version: str
    information_needs: tuple[CoverageLedgerNeed, ...]
    schema_version: str = COVERAGE_LEDGER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != COVERAGE_LEDGER_SCHEMA_VERSION:
            raise ValueError("unsupported coverage ledger schema version")
        for name in ("topic_id", "model", "prompt_version"):
            _text(getattr(self, name), field_name=name)
        if len(self.topic_sha256) != 64 or set(self.topic_sha256) - _HEX64:
            raise ValueError("coverage topic_sha256 must be a lowercase SHA-256 digest")
        if not self.information_needs:
            raise ValueError("coverage ledger must contain information needs")
        need_refs = [need.need_ref for need in self.information_needs]
        requirement_refs = [
            requirement.requirement_ref
            for need in self.information_needs
            for requirement in need.requirements
        ]
        if len(set(need_refs)) != len(need_refs):
            raise ValueError("coverage ledger need_ref values must be unique")
        if len(set(requirement_refs)) != len(requirement_refs):
            raise ValueError("coverage ledger requirement_ref values must be unique")

    @property
    def requirements(self) -> tuple[CoverageLedgerRequirement, ...]:
        return tuple(
            requirement
            for need in self.information_needs
            for requirement in need.requirements
        )

    @property
    def ledger_sha256(self) -> str:
        return _sha256_json(self.as_record(include_fingerprint=False))

    def as_record(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        record: dict[str, Any] = {
            "schema_version": self.schema_version,
            "topic_id": self.topic_id,
            "topic_sha256": self.topic_sha256,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "information_needs": [
                {
                    "need_ref": need.need_ref,
                    "objective": need.objective,
                    "why_it_matters": need.why_it_matters,
                    "priority": need.priority,
                    "requirements": [
                        {
                            "requirement_ref": requirement.requirement_ref,
                            "description": requirement.description,
                            "priority": requirement.priority,
                            "retrieval_anchors": list(
                                requirement.retrieval_anchors
                            ),
                        }
                        for requirement in need.requirements
                    ],
                }
                for need in self.information_needs
            ],
        }
        if include_fingerprint:
            record["ledger_sha256"] = self.ledger_sha256
        return record


def parse_coverage_ledger(
    topic: PlanningTopic,
    payload: Mapping[str, Any],
    *,
    model: str,
    prompt_version: str,
) -> CoverageLedger:
    """Validate and fingerprint a compact pre-query coverage ledger."""

    if not isinstance(payload, Mapping):
        raise ValueError("coverage ledger output must be an object")
    _expect_keys(payload, {"information_needs"}, label="coverage ledger")
    raw_needs = payload.get("information_needs")
    if (
        not isinstance(raw_needs, Sequence)
        or isinstance(raw_needs, (str, bytes))
        or not raw_needs
    ):
        raise ValueError("coverage information_needs must be a non-empty list")
    needs: list[CoverageLedgerNeed] = []
    for need_order, raw_need in enumerate(raw_needs, start=1):
        if not isinstance(raw_need, Mapping):
            raise ValueError("coverage information_needs entries must be objects")
        _expect_keys(
            raw_need,
            {"need_ref", "objective", "why_it_matters", "priority", "requirements"},
            label=f"coverage information_needs[{need_order}]",
        )
        raw_requirements = raw_need.get("requirements")
        if (
            not isinstance(raw_requirements, Sequence)
            or isinstance(raw_requirements, (str, bytes))
            or not raw_requirements
        ):
            raise ValueError("coverage need requirements must be a non-empty list")
        requirements: list[CoverageLedgerRequirement] = []
        for requirement_order, raw_requirement in enumerate(
            raw_requirements, start=1
        ):
            if not isinstance(raw_requirement, Mapping):
                raise ValueError("coverage requirements entries must be objects")
            _expect_keys(
                raw_requirement,
                {
                    "requirement_ref",
                    "description",
                    "priority",
                    "retrieval_anchors",
                },
                label=(
                    f"coverage requirements[{need_order},{requirement_order}]"
                ),
            )
            raw_anchors = raw_requirement.get("retrieval_anchors")
            if (
                not isinstance(raw_anchors, Sequence)
                or isinstance(raw_anchors, (str, bytes))
                or not raw_anchors
            ):
                raise ValueError("retrieval_anchors must be a non-empty list")
            requirements.append(
                CoverageLedgerRequirement(
                    requirement_ref=raw_requirement.get("requirement_ref"),
                    description=raw_requirement.get("description"),
                    priority=_positive_int(
                        raw_requirement.get("priority"),
                        field_name="coverage requirement priority",
                    ),
                    retrieval_anchors=tuple(raw_anchors),
                )
            )
        needs.append(
            CoverageLedgerNeed(
                need_ref=raw_need.get("need_ref"),
                objective=raw_need.get("objective"),
                why_it_matters=raw_need.get("why_it_matters"),
                priority=_positive_int(
                    raw_need.get("priority"), field_name="coverage need priority"
                ),
                requirements=tuple(requirements),
            )
        )
    return CoverageLedger(
        topic_id=topic.topic_id,
        topic_sha256=topic.input_sha256,
        model=_text(model, field_name="coverage ledger model"),
        prompt_version=_text(
            prompt_version, field_name="coverage ledger prompt_version"
        ),
        information_needs=tuple(needs),
    )


def coverage_ledger_from_record(record: Mapping[str, Any]) -> CoverageLedger:
    if not isinstance(record, Mapping):
        raise ValueError("cached coverage ledger must be an object")
    _expect_keys(
        record,
        {
            "schema_version",
            "topic_id",
            "topic_sha256",
            "model",
            "prompt_version",
            "information_needs",
            "ledger_sha256",
        },
        label="cached coverage ledger",
    )
    topic = PlanningTopic(
        topic_id=record.get("topic_id"),
        title="cached coverage ledger",
        problem_statement="cached coverage ledger",
        background="cached coverage ledger",
    )
    payload = {"information_needs": record.get("information_needs")}
    ledger = parse_coverage_ledger(
        topic,
        payload,
        model=record.get("model"),
        prompt_version=record.get("prompt_version"),
    )
    object.__setattr__(ledger, "topic_sha256", record.get("topic_sha256"))
    if record.get("schema_version") != COVERAGE_LEDGER_SCHEMA_VERSION:
        raise ValueError("unsupported coverage ledger schema version")
    if record.get("ledger_sha256") != ledger.ledger_sha256:
        raise ValueError("cached coverage ledger fingerprint mismatch")
    return ledger


@dataclass(frozen=True, slots=True)
class PlanningRequirement:
    requirement_id: str
    description: str
    priority: int
    order: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "description", _text(self.description, field_name="requirement description")
        )
        _positive_int(self.priority, field_name="requirement priority")
        _positive_int(self.order, field_name="requirement order")
        if not isinstance(self.requirement_id, str) or not self.requirement_id.startswith(
            "requirement-"
        ):
            raise ValueError("requirement_id must be a locally derived requirement ID")


@dataclass(frozen=True, slots=True)
class PlanTask:
    task_id: str
    need_id: str
    query: str
    evidence_target: str
    priority: int
    order: int
    requirement_ids: tuple[str, ...]
    query_role: str
    depends_on: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "query", _text(self.query, field_name="query"))
        object.__setattr__(
            self,
            "evidence_target",
            _text(self.evidence_target, field_name="evidence_target"),
        )
        _positive_int(self.priority, field_name="task priority")
        _positive_int(self.order, field_name="task order")
        if self.query_role not in QUERY_ROLES:
            raise ValueError("task query_role is unsupported")
        if not self.requirement_ids or len(set(self.requirement_ids)) != len(
            self.requirement_ids
        ):
            raise ValueError("task requirement IDs must be non-empty and unique")
        if any(not value.startswith("requirement-") for value in self.requirement_ids):
            raise ValueError("task requirement IDs must be locally derived")
        if not isinstance(self.task_id, str) or not self.task_id.startswith("task-"):
            raise ValueError("task_id must be a locally derived task ID")
        if not isinstance(self.need_id, str) or not self.need_id.startswith("need-"):
            raise ValueError("task need_id must be a locally derived need ID")
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError("task dependencies must be unique")


@dataclass(frozen=True, slots=True)
class InformationNeed:
    need_id: str
    objective: str
    why_it_matters: str
    completion_guidance: str
    priority: int
    order: int
    task_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("objective", "why_it_matters", "completion_guidance"):
            object.__setattr__(self, name, _text(getattr(self, name), field_name=name))
        _positive_int(self.priority, field_name="need priority")
        _positive_int(self.order, field_name="need order")
        if not isinstance(self.need_id, str) or not self.need_id.startswith("need-"):
            raise ValueError("need_id must be a locally derived need ID")
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("need task IDs must be unique")


@dataclass(frozen=True, slots=True)
class StaticResearchPlan:
    """One immutable, validated plan sealed before any retrieval."""

    topic_id: str
    topic_sha256: str
    model: str
    prompt_version: str
    max_total_tasks: int | None
    planning_stage: Literal["draft", "audited"]
    draft_plan_sha256: str | None
    requirements: tuple[PlanningRequirement, ...]
    information_needs: tuple[InformationNeed, ...]
    tasks: tuple[PlanTask, ...]
    raw_need_count: int
    raw_task_count: int
    truncated_task_count: int
    schema_version: str = PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PLAN_SCHEMA_VERSION:
            raise ValueError("unsupported static plan schema version")
        for name in ("topic_id", "topic_sha256", "model", "prompt_version"):
            _text(getattr(self, name), field_name=name)
        if len(self.topic_sha256) != 64 or set(self.topic_sha256) - _HEX64:
            raise ValueError("topic_sha256 must be a lowercase SHA-256 digest")
        if self.max_total_tasks is not None:
            _positive_int(self.max_total_tasks, field_name="max_total_tasks")
        if self.planning_stage not in {"draft", "audited"}:
            raise ValueError("planning_stage must be draft or audited")
        if self.planning_stage == "draft" and self.draft_plan_sha256 is not None:
            raise ValueError("draft plan must not reference another draft")
        if self.planning_stage == "audited" and (
            not isinstance(self.draft_plan_sha256, str)
            or len(self.draft_plan_sha256) != 64
            or set(self.draft_plan_sha256) - _HEX64
        ):
            raise ValueError("audited plan must reference a draft SHA-256")
        for name in ("raw_need_count", "raw_task_count", "truncated_task_count"):
            _nonnegative_int(getattr(self, name), field_name=name)
        if self.raw_need_count != len(self.information_needs):
            raise ValueError("raw_need_count must equal validated information-needs count")
        if self.raw_task_count != len(self.tasks) + self.truncated_task_count:
            raise ValueError("raw_task_count does not match admitted and truncated tasks")
        if self.max_total_tasks is not None and len(self.tasks) > self.max_total_tasks:
            raise ValueError("plan exceeds max_total_tasks")
        requirement_ids = [requirement.requirement_id for requirement in self.requirements]
        need_ids = [need.need_id for need in self.information_needs]
        task_ids = [task.task_id for task in self.tasks]
        if (
            not requirement_ids
            or len(set(requirement_ids)) != len(requirement_ids)
            or len(set(need_ids)) != len(need_ids)
            or len(set(task_ids)) != len(task_ids)
        ):
            raise ValueError("plan IDs must be unique")
        requirement_set = set(requirement_ids)
        task_set = set(task_ids)
        for need in self.information_needs:
            if any(task_id not in task_set for task_id in need.task_ids):
                raise ValueError("need references a task outside the sealed plan")
        for task in self.tasks:
            if task.need_id not in set(need_ids):
                raise ValueError("task references an unknown need")
            if any(value not in requirement_set for value in task.requirement_ids):
                raise ValueError("task references an unknown requirement")
            if any(dep not in task_set for dep in task.depends_on):
                raise ValueError("task dependency is outside the sealed plan")
        covered = {value for task in self.tasks for value in task.requirement_ids}
        if covered != requirement_set:
            raise ValueError("every planning requirement must be covered by a task")

    @property
    def plan_sha256(self) -> str:
        return _sha256_json(static_plan_to_record(self, include_fingerprint=False))

    def as_record(self) -> dict[str, Any]:
        return static_plan_to_record(self)


@dataclass(frozen=True, slots=True)
class PlannerResponse:
    payload: Mapping[str, Any]
    model: str
    prompt_version: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    latency_seconds: float = 0.0
    finish_reason: str | None = None

    def __post_init__(self) -> None:
        _text(self.model, field_name="planner response model")
        _text(self.prompt_version, field_name="planner response prompt_version")
        for name in ("input_tokens", "output_tokens"):
            _nonnegative_int(getattr(self, name), field_name=name)
        if (
            isinstance(self.cost, bool)
            or not isinstance(self.cost, (int, float))
            or not math.isfinite(float(self.cost))
            or self.cost < 0
        ):
            raise ValueError("planner response cost must be finite and non-negative")
        if (
            isinstance(self.latency_seconds, bool)
            or not isinstance(self.latency_seconds, (int, float))
            or not math.isfinite(float(self.latency_seconds))
            or self.latency_seconds < 0
        ):
            raise ValueError("planner response latency must be finite and non-negative")
        if self.finish_reason is not None:
            _text(self.finish_reason, field_name="planner response finish_reason")


@dataclass(frozen=True, slots=True)
class PlannerStageMetrics:
    input_tokens: int
    output_tokens: int
    cost: float
    latency_seconds: float
    finish_reason: str

    def __post_init__(self) -> None:
        _nonnegative_int(self.input_tokens, field_name="stage input_tokens")
        _nonnegative_int(self.output_tokens, field_name="stage output_tokens")
        if (
            isinstance(self.cost, bool)
            or not isinstance(self.cost, (int, float))
            or not math.isfinite(float(self.cost))
            or self.cost < 0
        ):
            raise ValueError("stage cost must be finite and non-negative")
        if (
            isinstance(self.latency_seconds, bool)
            or not isinstance(self.latency_seconds, (int, float))
            or not math.isfinite(float(self.latency_seconds))
            or self.latency_seconds < 0
        ):
            raise ValueError("stage latency must be finite and non-negative")
        _text(self.finish_reason, field_name="stage finish_reason")

    @classmethod
    def from_response(cls, response: PlannerResponse) -> PlannerStageMetrics:
        if response.finish_reason is None:
            raise ValueError("planner response must report finish_reason")
        return cls(
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost=float(response.cost),
            latency_seconds=float(response.latency_seconds),
            finish_reason=response.finish_reason,
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": self.cost,
            "latency_seconds": self.latency_seconds,
            "finish_reason": self.finish_reason,
        }


def _stage_metrics_from_record(record: Mapping[str, Any]) -> PlannerStageMetrics:
    if not isinstance(record, Mapping):
        raise ValueError("cached stage metrics must be an object")
    _expect_keys(
        record,
        {
            "input_tokens",
            "output_tokens",
            "cost",
            "latency_seconds",
            "finish_reason",
        },
        label="cached stage metrics",
    )
    return PlannerStageMetrics(
        input_tokens=_nonnegative_int(
            record.get("input_tokens"), field_name="stage input_tokens"
        ),
        output_tokens=_nonnegative_int(
            record.get("output_tokens"), field_name="stage output_tokens"
        ),
        cost=float(record.get("cost")),
        latency_seconds=float(record.get("latency_seconds")),
        finish_reason=_text(
            record.get("finish_reason"), field_name="stage finish_reason"
        ),
    )


@dataclass(frozen=True, slots=True)
class CoverageLedgerCacheEntry:
    ledger: CoverageLedger
    metrics: PlannerStageMetrics

    def as_record(self) -> dict[str, Any]:
        return {"ledger": self.ledger.as_record(), "metrics": self.metrics.as_record()}


def coverage_ledger_cache_entry_from_record(
    record: Mapping[str, Any],
) -> CoverageLedgerCacheEntry:
    if not isinstance(record, Mapping):
        raise ValueError("cached coverage ledger entry must be an object")
    _expect_keys(record, {"ledger", "metrics"}, label="coverage ledger entry")
    return CoverageLedgerCacheEntry(
        ledger=coverage_ledger_from_record(record.get("ledger")),
        metrics=_stage_metrics_from_record(record.get("metrics")),
    )


@dataclass(frozen=True, slots=True)
class CoverageExpansionCacheEntry:
    plan: StaticResearchPlan
    metrics: PlannerStageMetrics

    def as_record(self) -> dict[str, Any]:
        return {"plan": self.plan.as_record(), "metrics": self.metrics.as_record()}


def coverage_expansion_cache_entry_from_record(
    record: Mapping[str, Any],
) -> CoverageExpansionCacheEntry:
    if not isinstance(record, Mapping):
        raise ValueError("cached coverage expansion entry must be an object")
    _expect_keys(record, {"plan", "metrics"}, label="coverage expansion entry")
    return CoverageExpansionCacheEntry(
        plan=static_plan_from_record(record.get("plan")),
        metrics=_stage_metrics_from_record(record.get("metrics")),
    )


class PlannerCallError(ValueError):
    """Sanitized provider-output failure with usage but no response content."""

    def __init__(
        self,
        detail: Literal[
            "output_truncated",
            "invalid_json",
            "missing_content",
            "invalid_response_envelope",
        ],
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: float = 0.0,
        latency_seconds: float = 0.0,
        finish_reason: str | None = None,
        content_shape: Literal[
            "starts_object", "fenced", "prose_prefix", "no_object"
        ]
        | None = None,
        parse_error_offset: int | None = None,
    ) -> None:
        super().__init__(detail)
        self.detail = detail
        self.input_tokens = _nonnegative_int(
            input_tokens, field_name="planner call input_tokens"
        )
        self.output_tokens = _nonnegative_int(
            output_tokens, field_name="planner call output_tokens"
        )
        self.cost = float(cost or 0.0)
        self.latency_seconds = float(latency_seconds)
        self.finish_reason = (
            None
            if finish_reason is None
            else _text(finish_reason, field_name="planner call finish_reason")
        )
        self.content_shape = content_shape
        self.parse_error_offset = (
            None
            if parse_error_offset is None
            else _nonnegative_int(
                parse_error_offset, field_name="parse_error_offset"
            )
        )


@dataclass(frozen=True, slots=True)
class PlannerOutcome:
    status: Literal["success", "failed"]
    plan: StaticResearchPlan | None
    cache_hit: bool
    model: str
    prompt_version: str
    input_tokens: int
    output_tokens: int
    cost: float
    latency_seconds: float
    draft_cache_hit: bool = False
    audit_cache_hit: bool = False
    failure_kind: str | None = None
    failure_stage: Literal["draft", "audit"] | None = None
    failure_detail: str | None = None
    validation_error: str | None = None
    finish_reason: str | None = None
    planner_path: str = "full_plan_v3"
    first_stage_input_tokens: int = 0
    first_stage_output_tokens: int = 0
    second_stage_input_tokens: int = 0
    second_stage_output_tokens: int = 0
    second_stage_finish_reason: str | None = None
    content_shape: str | None = None
    parse_error_offset: int | None = None
    fallback_used: bool = False
    fallback_reason: str | None = None


class PlannerModel(Protocol):
    def generate(
        self, topic: PlanningTopic, *, max_total_tasks: int | None
    ) -> PlannerResponse: ...


class PlannerAuditor(Protocol):
    def audit(
        self,
        topic: PlanningTopic,
        draft: StaticResearchPlan,
        *,
        max_total_tasks: int | None,
    ) -> PlannerResponse: ...


class CoveragePlannerModel(Protocol):
    def generate_coverage_ledger(
        self, topic: PlanningTopic
    ) -> PlannerResponse: ...

    def expand_coverage_ledger(
        self,
        topic: PlanningTopic,
        ledger: CoverageLedger,
        *,
        max_total_tasks: int | None,
    ) -> PlannerResponse: ...


class PlanArtifactCache(Protocol):
    """Validated plan cache seam; implementations must never cache failures."""

    def get(self, key: str) -> StaticResearchPlan | None: ...

    def put(self, key: str, plan: StaticResearchPlan) -> bool: ...


class CoverageLedgerArtifactCache(Protocol):
    """Validated cache seam for compact coverage ledgers."""

    def get(self, key: str) -> CoverageLedgerCacheEntry | None: ...

    def put(self, key: str, entry: CoverageLedgerCacheEntry) -> bool: ...


class CoverageExpansionArtifactCache(Protocol):
    """Validated cache seam for expanded plans and their provider metrics."""

    def get(self, key: str) -> CoverageExpansionCacheEntry | None: ...

    def put(self, key: str, entry: CoverageExpansionCacheEntry) -> bool: ...


class ResponseCachePlanArtifactCache:
    """Use the repository's typed response/artifact cache for sealed plans."""

    def __init__(
        self,
        store: ArtifactStoreLike,
        *,
        base_url: str,
        namespace: str | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {}
        if namespace is not None:
            kwargs["namespace"] = namespace
        self._cache = ResponseCache(store, **kwargs)
        self._base_url = base_url

    @staticmethod
    def _request(key: str) -> dict[str, str]:
        return {"plan_key": _text(key, field_name="plan cache key")}

    def get(self, key: str) -> StaticResearchPlan | None:
        entry = self._cache.get(
            base_url=self._base_url,
            operation=PLANNER_OPERATION,
            request=self._request(key),
            ttl_seconds=None,
            validator=lambda payload: static_plan_from_record(payload),
        )
        return None if entry is None else entry.value

    def put(self, key: str, plan: StaticResearchPlan) -> bool:
        if not isinstance(plan, StaticResearchPlan):
            raise TypeError("only validated static plans may be cached")
        stored = self._cache.put(
            base_url=self._base_url,
            operation=PLANNER_OPERATION,
            request=self._request(key),
            response=plan.as_record(),
        )
        return stored is not None


class ResponseCacheCoverageLedgerArtifactCache:
    """Persist validated coverage ledgers through the shared artifact store."""

    def __init__(
        self,
        store: ArtifactStoreLike,
        *,
        base_url: str,
        namespace: str | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {}
        if namespace is not None:
            kwargs["namespace"] = namespace
        self._cache = ResponseCache(store, **kwargs)
        self._base_url = base_url

    @staticmethod
    def _request(key: str) -> dict[str, str]:
        return {"ledger_key": _text(key, field_name="coverage ledger cache key")}

    def get(self, key: str) -> CoverageLedgerCacheEntry | None:
        entry = self._cache.get(
            base_url=self._base_url,
            operation=COVERAGE_LEDGER_OPERATION,
            request=self._request(key),
            ttl_seconds=None,
            validator=lambda payload: coverage_ledger_cache_entry_from_record(payload),
        )
        return None if entry is None else entry.value

    def put(self, key: str, entry: CoverageLedgerCacheEntry) -> bool:
        if not isinstance(entry, CoverageLedgerCacheEntry):
            raise TypeError("only validated coverage ledger entries may be cached")
        stored = self._cache.put(
            base_url=self._base_url,
            operation=COVERAGE_LEDGER_OPERATION,
            request=self._request(key),
            response=entry.as_record(),
        )
        return stored is not None


class ResponseCacheCoverageExpansionArtifactCache:
    """Persist final plans together with durable stage metrics."""

    def __init__(
        self,
        store: ArtifactStoreLike,
        *,
        base_url: str,
        namespace: str | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {}
        if namespace is not None:
            kwargs["namespace"] = namespace
        self._cache = ResponseCache(store, **kwargs)
        self._base_url = base_url

    @staticmethod
    def _request(key: str) -> dict[str, str]:
        return {"expansion_key": _text(key, field_name="expansion cache key")}

    def get(self, key: str) -> CoverageExpansionCacheEntry | None:
        entry = self._cache.get(
            base_url=self._base_url,
            operation=COVERAGE_EXPANSION_OPERATION,
            request=self._request(key),
            ttl_seconds=None,
            validator=lambda payload: coverage_expansion_cache_entry_from_record(
                payload
            ),
        )
        return None if entry is None else entry.value

    def put(self, key: str, entry: CoverageExpansionCacheEntry) -> bool:
        if not isinstance(entry, CoverageExpansionCacheEntry):
            raise TypeError("only validated coverage expansion entries may be cached")
        stored = self._cache.put(
            base_url=self._base_url,
            operation=COVERAGE_EXPANSION_OPERATION,
            request=self._request(key),
            response=entry.as_record(),
        )
        return stored is not None


def _parse_priority(value: Any, *, field_name: str) -> int:
    return _positive_int(value, field_name=field_name)


def parse_static_plan(
    topic: PlanningTopic,
    payload: Mapping[str, Any],
    *,
    model: str,
    prompt_version: str,
    max_total_tasks: int | None,
    planning_stage: Literal["draft", "audited"] = "draft",
    draft_plan_sha256: str | None = None,
) -> StaticResearchPlan:
    """Validate model JSON and deterministically seal its admitted task prefix."""

    if not isinstance(topic, PlanningTopic):
        raise TypeError("topic must be a PlanningTopic")
    if not isinstance(payload, Mapping):
        raise ValueError("planner output must be an object")
    _expect_keys(payload, {"requirements", "information_needs"}, label="plan")
    if max_total_tasks is not None:
        _positive_int(max_total_tasks, field_name="max_total_tasks")
    if planning_stage not in {"draft", "audited"}:
        raise ValueError("planning_stage must be draft or audited")
    if planning_stage == "draft" and draft_plan_sha256 is not None:
        raise ValueError("draft plan must not reference another draft")
    if planning_stage == "audited" and (
        not isinstance(draft_plan_sha256, str)
        or len(draft_plan_sha256) != 64
        or set(draft_plan_sha256) - _HEX64
    ):
        raise ValueError("audited plan must reference a draft SHA-256")

    raw_requirements = payload.get("requirements")
    if (
        not isinstance(raw_requirements, Sequence)
        or isinstance(raw_requirements, (str, bytes))
        or not raw_requirements
    ):
        raise ValueError("requirements must be a non-empty list")
    requirement_rows: list[dict[str, Any]] = []
    requirement_refs: set[str] = set()
    for requirement_order, raw_requirement in enumerate(raw_requirements, start=1):
        if not isinstance(raw_requirement, Mapping):
            raise ValueError("requirements entries must be objects")
        _expect_keys(
            raw_requirement,
            {"requirement_ref", "description", "priority"},
            label=f"requirements[{requirement_order}]",
        )
        requirement_ref = _text(
            raw_requirement.get("requirement_ref"), field_name="requirement_ref"
        )
        if requirement_ref in requirement_refs:
            raise ValueError("duplicate requirement_ref in static plan")
        requirement_refs.add(requirement_ref)
        description = _text(
            raw_requirement.get("description"), field_name="requirement description"
        )
        requirement_rows.append(
            {
                "requirement_ref": requirement_ref,
                "requirement_id": _stable_id(
                    "requirement",
                    {
                        "topic_id": topic.topic_id,
                        "order": requirement_order,
                        "description": description,
                    },
                ),
                "description": description,
                "priority": _parse_priority(
                    raw_requirement.get("priority"), field_name="requirement priority"
                ),
                "order": requirement_order,
            }
        )
    requirement_rows.sort(key=lambda row: (row["priority"], row["order"]))
    requirement_id_by_ref = {
        row["requirement_ref"]: row["requirement_id"] for row in requirement_rows
    }
    requirements = tuple(
        PlanningRequirement(
            requirement_id=row["requirement_id"],
            description=row["description"],
            priority=row["priority"],
            order=row["order"],
        )
        for row in requirement_rows
    )
    raw_needs = payload.get("information_needs")
    if not isinstance(raw_needs, Sequence) or isinstance(raw_needs, (str, bytes)) or not raw_needs:
        raise ValueError("information_needs must be a non-empty list")

    need_rows: list[dict[str, Any]] = []
    query_keys: set[str] = set()
    task_refs: set[str] = set()
    for need_order, raw_need in enumerate(raw_needs, start=1):
        if not isinstance(raw_need, Mapping):
            raise ValueError("information_needs entries must be objects")
        _expect_keys(
            raw_need,
            {
                "need_ref",
                "objective",
                "why_it_matters",
                "completion_guidance",
                "priority",
                "search_tasks",
            },
            label=f"information_needs[{need_order}]",
        )
        objective = _text(raw_need.get("objective"), field_name="objective")
        why = _text(raw_need.get("why_it_matters"), field_name="why_it_matters")
        completion = _text(
            raw_need.get("completion_guidance"), field_name="completion_guidance"
        )
        priority = _parse_priority(raw_need.get("priority"), field_name="need priority")
        raw_tasks = raw_need.get("search_tasks")
        if not isinstance(raw_tasks, Sequence) or isinstance(raw_tasks, (str, bytes)) or not raw_tasks:
            raise ValueError("each information need must have search_tasks")
        need_id = _stable_id(
            "need",
            {"topic_id": topic.topic_id, "order": need_order, "objective": objective},
        )
        task_rows: list[dict[str, Any]] = []
        for task_order, raw_task in enumerate(raw_tasks, start=1):
            if not isinstance(raw_task, Mapping):
                raise ValueError("search_tasks entries must be objects")
            _expect_keys(
                raw_task,
                {
                    "task_ref",
                    "query",
                    "evidence_target",
                    "priority",
                    "depends_on",
                    "requirement_refs",
                    "query_role",
                },
                label=f"search_tasks[{need_order},{task_order}]",
            )
            query = _text(raw_task.get("query"), field_name="query")
            query_key = query.casefold()
            if query_key in query_keys:
                raise ValueError("duplicate normalized query in static plan")
            query_keys.add(query_key)
            task_ref = raw_task.get("task_ref", f"need-{need_order}-task-{task_order}")
            task_ref = _text(task_ref, field_name="task_ref")
            if task_ref in task_refs:
                raise ValueError("duplicate task_ref in static plan")
            task_refs.add(task_ref)
            depends = raw_task.get("depends_on", [])
            if not isinstance(depends, Sequence) or isinstance(depends, (str, bytes)):
                raise ValueError("depends_on must be a list")
            depends_tuple = tuple(_text(value, field_name="depends_on") for value in depends)
            raw_requirement_refs = raw_task.get("requirement_refs")
            if (
                not isinstance(raw_requirement_refs, Sequence)
                or isinstance(raw_requirement_refs, (str, bytes))
                or not raw_requirement_refs
            ):
                raise ValueError("requirement_refs must be a non-empty list")
            task_requirement_refs = tuple(
                _text(value, field_name="requirement_ref")
                for value in raw_requirement_refs
            )
            if len(set(task_requirement_refs)) != len(task_requirement_refs):
                raise ValueError("task requirement_refs must be unique")
            if any(ref not in requirement_id_by_ref for ref in task_requirement_refs):
                raise ValueError("task references an unknown requirement_ref")
            query_role = _text(raw_task.get("query_role"), field_name="query_role")
            if query_role not in QUERY_ROLES:
                raise ValueError("task query_role is unsupported")
            task_rows.append(
                {
                    "task_ref": task_ref,
                    "query": query,
                    "evidence_target": _text(
                        raw_task.get("evidence_target"), field_name="evidence_target"
                    ),
                    "priority": _parse_priority(
                        raw_task.get("priority"), field_name="task priority"
                    ),
                    "order": task_order,
                    "depends_on_refs": depends_tuple,
                    "requirement_ids": tuple(
                        requirement_id_by_ref[ref] for ref in task_requirement_refs
                    ),
                    "query_role": query_role,
                    "need_id": need_id,
                }
            )
        task_rows.sort(key=lambda row: (row["priority"], row["order"]))
        need_rows.append(
            {
                "need_id": need_id,
                "objective": objective,
                "why_it_matters": why,
                "completion_guidance": completion,
                "priority": priority,
                "order": need_order,
                "task_rows": task_rows,
            }
        )

    need_rows.sort(key=lambda row: (row["priority"], row["order"]))
    all_task_rows = [row for need in need_rows for row in need["task_rows"]]
    task_row_by_ref = {row["task_ref"]: row for row in all_task_rows}
    row_index = {row["task_ref"]: index for index, row in enumerate(all_task_rows)}
    for row in all_task_rows:
        for dependency in row["depends_on_refs"]:
            if dependency not in task_row_by_ref:
                raise ValueError("task dependency references an unknown task_ref")
            if dependency == row["task_ref"] or row_index[dependency] >= row_index[row["task_ref"]]:
                raise ValueError("task dependencies must refer to earlier planned tasks")

    task_id_by_ref = {
        row["task_ref"]: _stable_id(
            "task",
            {
                "topic_id": topic.topic_id,
                "need_id": row["need_id"],
                "order": row["order"],
                "query": row["query"],
            },
        )
        for row in all_task_rows
    }
    tasks_by_need: dict[str, list[PlanTask]] = {need["need_id"]: [] for need in need_rows}
    for global_order, row in enumerate(all_task_rows, start=1):
        tasks_by_need[row["need_id"]].append(
            PlanTask(
                task_id=task_id_by_ref[row["task_ref"]],
                need_id=row["need_id"],
                query=row["query"],
                evidence_target=row["evidence_target"],
                priority=row["priority"],
                order=global_order,
                requirement_ids=row["requirement_ids"],
                query_role=row["query_role"],
                depends_on=tuple(task_id_by_ref[ref] for ref in row["depends_on_refs"]),
            )
        )

    admitted: list[PlanTask] = []
    selected_ids: set[str] = set()
    cursors = {need["need_id"]: 0 for need in need_rows}
    task_limit = len(all_task_rows) if max_total_tasks is None else max_total_tasks
    while len(admitted) < task_limit:
        progress = False
        for need in need_rows:
            need_id = need["need_id"]
            candidates = tasks_by_need[need_id]
            cursor = cursors[need_id]
            if cursor >= len(candidates):
                continue
            candidate = candidates[cursor]
            if not set(candidate.depends_on).issubset(selected_ids):
                continue
            admitted.append(candidate)
            selected_ids.add(candidate.task_id)
            cursors[need_id] = cursor + 1
            progress = True
            if len(admitted) >= task_limit:
                break
        if not progress:
            break

    admitted_by_need: dict[str, tuple[str, ...]] = {
        need["need_id"]: tuple(
            task.task_id for task in admitted if task.need_id == need["need_id"]
        )
        for need in need_rows
    }
    needs = tuple(
        InformationNeed(
            need_id=need["need_id"],
            objective=need["objective"],
            why_it_matters=need["why_it_matters"],
            completion_guidance=need["completion_guidance"],
            priority=need["priority"],
            order=need["order"],
            task_ids=admitted_by_need[need["need_id"]],
        )
        for need in need_rows
    )
    return StaticResearchPlan(
        topic_id=topic.topic_id,
        topic_sha256=topic.input_sha256,
        model=_text(model, field_name="model"),
        prompt_version=_text(prompt_version, field_name="prompt_version"),
        max_total_tasks=max_total_tasks,
        planning_stage=planning_stage,
        draft_plan_sha256=draft_plan_sha256,
        requirements=requirements,
        information_needs=needs,
        tasks=tuple(admitted),
        raw_need_count=len(need_rows),
        raw_task_count=len(all_task_rows),
        truncated_task_count=len(all_task_rows) - len(admitted),
    )


def parse_compact_query_expansion(
    topic: PlanningTopic,
    ledger: CoverageLedger,
    payload: Mapping[str, Any],
    *,
    model: str,
    prompt_version: str,
) -> StaticResearchPlan:
    """Build a complete final plan from compact per-requirement query sets."""

    if ledger.topic_sha256 != topic.input_sha256:
        raise ValueError("coverage ledger does not match topic")
    if not isinstance(payload, Mapping):
        raise ValueError("compact expansion output must be an object")
    raw_sets = payload.get("query_sets")
    if (
        not isinstance(raw_sets, Sequence)
        or isinstance(raw_sets, (str, bytes))
        or not raw_sets
    ):
        raise ValueError("compact query_sets must be a non-empty list")
    queries_by_requirement: dict[str, list[dict[str, Any]]] = {}
    for set_order, raw_set in enumerate(raw_sets, start=1):
        if not isinstance(raw_set, Mapping):
            raise ValueError("compact query_sets entries must be objects")
        requirement_ref = _text(
            raw_set.get("requirement_ref"), field_name="requirement_ref"
        )
        if requirement_ref in queries_by_requirement:
            raise ValueError("compact query_sets must have unique requirement_ref")
        raw_queries = raw_set.get("queries")
        if (
            not isinstance(raw_queries, Sequence)
            or isinstance(raw_queries, (str, bytes))
            or not raw_queries
        ):
            raise ValueError("each compact query set must contain queries")
        queries: list[dict[str, Any]] = []
        for query_order, raw_query in enumerate(raw_queries, start=1):
            if not isinstance(raw_query, Mapping):
                raise ValueError("compact queries entries must be objects")
            query = _text(raw_query.get("query"), field_name="query")
            role = _text(raw_query.get("query_role"), field_name="query_role")
            if role not in QUERY_ROLES:
                raise ValueError("task query_role is unsupported")
            queries.append(
                {
                    "query": query,
                    "query_role": role,
                    "anchor_dimension": _text(
                        raw_query.get("anchor_dimension"),
                        field_name="anchor_dimension",
                    ),
                }
            )
        queries_by_requirement[requirement_ref] = queries

    ledger_refs = {
        requirement.requirement_ref for requirement in ledger.requirements
    }
    if set(queries_by_requirement) != ledger_refs:
        raise ValueError("compact expansion requires exact ledger requirement coverage")

    requirement_rows: list[dict[str, Any]] = []
    for requirement_order, requirement in enumerate(ledger.requirements, start=1):
        requirement_rows.append(
            {
                "requirement_ref": requirement.requirement_ref,
                "requirement_id": _stable_id(
                    "requirement",
                    {
                        "topic_id": topic.topic_id,
                        "order": requirement_order,
                        "description": requirement.description,
                    },
                ),
                "description": requirement.description,
                "priority": requirement.priority,
                "order": requirement_order,
            }
        )
    requirement_rows.sort(key=lambda row: (row["priority"], row["order"]))
    requirement_id_by_ref = {
        row["requirement_ref"]: row["requirement_id"] for row in requirement_rows
    }
    requirements = tuple(
        PlanningRequirement(
            requirement_id=row["requirement_id"],
            description=row["description"],
            priority=row["priority"],
            order=row["order"],
        )
        for row in requirement_rows
    )

    # Provider extensions are deliberately projected away above. Exact normalized
    # query duplicates cannot produce new results from the forever-cached endpoint,
    # so retain one retrieval task and merge every requirement mapping into it.
    # Non-identical semantic variants remain separate tasks.
    query_row_by_key: dict[str, dict[str, Any]] = {}
    need_rows: list[dict[str, Any]] = []
    for need_order, need in enumerate(ledger.information_needs, start=1):
        need_id = _stable_id(
            "need",
            {
                "topic_id": topic.topic_id,
                "order": need_order,
                "objective": need.objective,
            },
        )
        task_rows: list[dict[str, Any]] = []
        for requirement in need.requirements:
            for query_row in queries_by_requirement[requirement.requirement_ref]:
                query_key = query_row["query"].casefold()
                existing = query_row_by_key.get(query_key)
                if existing is not None:
                    if requirement.requirement_ref not in existing["requirement_refs"]:
                        existing["requirement_refs"].append(
                            requirement.requirement_ref
                        )
                    anchor_key = query_row["anchor_dimension"].casefold()
                    if anchor_key not in existing["anchor_keys"]:
                        existing["anchor_keys"].add(anchor_key)
                        existing["anchor_dimensions"].append(
                            query_row["anchor_dimension"]
                        )
                    continue
                task_order = len(task_rows) + 1
                row = {
                    "query": query_row["query"],
                    "query_role": query_row["query_role"],
                    "anchor_dimensions": [query_row["anchor_dimension"]],
                    "anchor_keys": {query_row["anchor_dimension"].casefold()},
                    "requirement_refs": [requirement.requirement_ref],
                    "priority": task_order,
                    "order": task_order,
                    "need_id": need_id,
                }
                query_row_by_key[query_key] = row
                task_rows.append(row)
        need_rows.append(
            {
                "need_id": need_id,
                "objective": need.objective,
                "why_it_matters": need.why_it_matters,
                "completion_guidance": (
                    "Retrieve candidate evidence for every atomic requirement "
                    "in this information need."
                ),
                "priority": need.priority,
                "order": need_order,
                "task_rows": task_rows,
            }
        )

    need_rows.sort(key=lambda row: (row["priority"], row["order"]))
    all_task_rows = [row for need in need_rows for row in need["task_rows"]]
    task_id_by_query = {
        row["query"].casefold(): _stable_id(
            "task",
            {
                "topic_id": topic.topic_id,
                "need_id": row["need_id"],
                "order": row["order"],
                "query": row["query"],
            },
        )
        for row in all_task_rows
    }
    tasks_by_need: dict[str, list[PlanTask]] = {
        need["need_id"]: [] for need in need_rows
    }
    for global_order, row in enumerate(all_task_rows, start=1):
        tasks_by_need[row["need_id"]].append(
            PlanTask(
                task_id=task_id_by_query[row["query"].casefold()],
                need_id=row["need_id"],
                query=row["query"],
                evidence_target="; ".join(row["anchor_dimensions"]),
                priority=row["priority"],
                order=global_order,
                requirement_ids=tuple(
                    requirement_id_by_ref[ref]
                    for ref in row["requirement_refs"]
                ),
                query_role=row["query_role"],
                depends_on=(),
            )
        )

    admitted: list[PlanTask] = []
    cursors = {need["need_id"]: 0 for need in need_rows}
    while len(admitted) < len(all_task_rows):
        for need in need_rows:
            need_id = need["need_id"]
            cursor = cursors[need_id]
            candidates = tasks_by_need[need_id]
            if cursor < len(candidates):
                admitted.append(candidates[cursor])
                cursors[need_id] = cursor + 1

    admitted_by_need = {
        need["need_id"]: tuple(
            task.task_id for task in admitted if task.need_id == need["need_id"]
        )
        for need in need_rows
    }
    needs = tuple(
        InformationNeed(
            need_id=need["need_id"],
            objective=need["objective"],
            why_it_matters=need["why_it_matters"],
            completion_guidance=need["completion_guidance"],
            priority=need["priority"],
            order=need["order"],
            task_ids=admitted_by_need[need["need_id"]],
        )
        for need in need_rows
    )
    return StaticResearchPlan(
        topic_id=topic.topic_id,
        topic_sha256=topic.input_sha256,
        model=_text(model, field_name="model"),
        prompt_version=_text(prompt_version, field_name="prompt_version"),
        max_total_tasks=None,
        planning_stage="audited",
        draft_plan_sha256=ledger.ledger_sha256,
        requirements=requirements,
        information_needs=needs,
        tasks=tuple(admitted),
        raw_need_count=len(need_rows),
        raw_task_count=len(all_task_rows),
        truncated_task_count=0,
    )


def _ledger_fallback_query_role(anchor: str) -> str:
    normalized = anchor.casefold()
    if any(
        term in normalized
        for term in (
            "adverse",
            "contrary",
            "disputed",
            "false alarm",
            "hype",
            "limitation",
            "negative",
            "no effect",
            "reproducibility",
            "risk",
            "uncertainty",
        )
    ):
        return "contrary_evidence"
    if any(
        term in normalized
        for term in (
            "accuracy",
            "clinical",
            "evaluation",
            "magnitude",
            "measure",
            "outcome",
            "performance",
            "statistic",
            "time window",
            "trial",
        )
    ):
        return "measurable_outcomes"
    if any(
        term in normalized
        for term in (
            "academic",
            "government",
            "meta-analysis",
            "official",
            "primary source",
            "regulatory",
            "systematic review",
        )
    ):
        return "official_primary"
    if any(term in normalized for term in ("compare", "comparison", " versus ", " vs ")):
        return "comparison"
    return "precise_facts"


def build_ledger_fallback_plan(
    topic: PlanningTopic,
    ledger: CoverageLedger,
    *,
    model: str,
    expansion_prompt_version: str,
    validation_error: str,
    source_payload: Mapping[str, Any],
) -> StaticResearchPlan:
    """Seal a total, marked plan from a validated ledger after local expansion failure."""

    query_sets: list[dict[str, Any]] = []
    for requirement in ledger.requirements:
        queries: list[dict[str, str]] = [
            {
                "query": requirement.description,
                "query_role": "broad_discovery",
                "anchor_dimension": "validated requirement description",
            }
        ]
        queries.extend(
            {
                "query": f"{topic.title}: {anchor}",
                "query_role": _ledger_fallback_query_role(anchor),
                "anchor_dimension": anchor,
            }
            for anchor in requirement.retrieval_anchors
        )
        query_sets.append(
            {
                "requirement_ref": requirement.requirement_ref,
                "queries": queries,
            }
        )
    payload = {"query_sets": query_sets}
    source_hash = _sha256_json(source_payload)[:12]
    return parse_compact_query_expansion(
        topic,
        ledger,
        payload,
        model=model,
        prompt_version=(
            f"{expansion_prompt_version}+{COMPACT_EXPANSION_ASSEMBLER_VERSION}+"
            f"{LEDGER_FALLBACK_VERSION}-{validation_error}-{source_hash}"
        ),
    )


def _fallback_reason_from_prompt_version(prompt_version: str) -> str | None:
    marker = f"+{LEDGER_FALLBACK_VERSION}-"
    if marker not in prompt_version:
        return None
    suffix = prompt_version.split(marker, 1)[1]
    reason, separator, _source_hash = suffix.rpartition("-")
    return reason if separator and reason else "unknown_local_validation"


def static_plan_to_record(
    plan: StaticResearchPlan, *, include_fingerprint: bool = True
) -> dict[str, Any]:
    if not isinstance(plan, StaticResearchPlan):
        raise TypeError("plan must be a StaticResearchPlan")
    record: dict[str, Any] = {
        "schema_version": plan.schema_version,
        "topic_id": plan.topic_id,
        "topic_sha256": plan.topic_sha256,
        "model": plan.model,
        "prompt_version": plan.prompt_version,
        "max_total_tasks": plan.max_total_tasks,
        "planning_stage": plan.planning_stage,
        "draft_plan_sha256": plan.draft_plan_sha256,
        "requirements": [
            {
                "requirement_id": requirement.requirement_id,
                "description": requirement.description,
                "priority": requirement.priority,
                "order": requirement.order,
            }
            for requirement in plan.requirements
        ],
        "information_needs": [
            {
                "need_id": need.need_id,
                "objective": need.objective,
                "why_it_matters": need.why_it_matters,
                "completion_guidance": need.completion_guidance,
                "priority": need.priority,
                "order": need.order,
                "task_ids": list(need.task_ids),
            }
            for need in plan.information_needs
        ],
        "tasks": [
            {
                "task_id": task.task_id,
                "need_id": task.need_id,
                "query": task.query,
                "evidence_target": task.evidence_target,
                "priority": task.priority,
                "order": task.order,
                "requirement_ids": list(task.requirement_ids),
                "query_role": task.query_role,
                "depends_on": list(task.depends_on),
            }
            for task in plan.tasks
        ],
        "raw_need_count": plan.raw_need_count,
        "raw_task_count": plan.raw_task_count,
        "truncated_task_count": plan.truncated_task_count,
    }
    if include_fingerprint:
        record["plan_sha256"] = plan.plan_sha256
    return record


def static_plan_from_record(record: Mapping[str, Any]) -> StaticResearchPlan:
    if not isinstance(record, Mapping):
        raise ValueError("cached static plan must be an object")
    allowed = {
        "schema_version",
        "topic_id",
        "topic_sha256",
        "model",
        "prompt_version",
        "max_total_tasks",
        "planning_stage",
        "draft_plan_sha256",
        "requirements",
        "information_needs",
        "tasks",
        "raw_need_count",
        "raw_task_count",
        "truncated_task_count",
        "plan_sha256",
    }
    _expect_keys(record, allowed, label="cached plan")
    if record.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported static plan schema version")
    raw_requirements = record.get("requirements")
    raw_needs = record.get("information_needs")
    raw_tasks = record.get("tasks")
    if not isinstance(raw_requirements, Sequence) or isinstance(
        raw_requirements, (str, bytes)
    ):
        raise ValueError("cached requirements must be a list")
    if not isinstance(raw_needs, Sequence) or isinstance(raw_needs, (str, bytes)):
        raise ValueError("cached information_needs must be a list")
    if not isinstance(raw_tasks, Sequence) or isinstance(raw_tasks, (str, bytes)):
        raise ValueError("cached tasks must be a list")
    requirements_list: list[PlanningRequirement] = []
    for row in raw_requirements:
        if not isinstance(row, Mapping):
            raise ValueError("cached requirements entries must be objects")
        _expect_keys(
            row,
            {"requirement_id", "description", "priority", "order"},
            label="cached requirement",
        )
        requirements_list.append(
            PlanningRequirement(
                requirement_id=_text(
                    row.get("requirement_id"), field_name="requirement_id"
                ),
                description=_text(
                    row.get("description"), field_name="requirement description"
                ),
                priority=_positive_int(
                    row.get("priority"), field_name="requirement priority"
                ),
                order=_positive_int(row.get("order"), field_name="requirement order"),
            )
        )
    requirements = tuple(requirements_list)
    needs_list: list[InformationNeed] = []
    for row in raw_needs:
        if not isinstance(row, Mapping):
            raise ValueError("cached information_needs entries must be objects")
        _expect_keys(
            row,
            {
                "need_id",
                "objective",
                "why_it_matters",
                "completion_guidance",
                "priority",
                "order",
                "task_ids",
            },
            label="cached information need",
        )
        task_ids = row.get("task_ids")
        if not isinstance(task_ids, Sequence) or isinstance(task_ids, (str, bytes)):
            raise ValueError("cached need task_ids must be a list")
        needs_list.append(
            InformationNeed(
                need_id=_text(row.get("need_id"), field_name="need_id"),
                objective=_text(row.get("objective"), field_name="objective"),
                why_it_matters=_text(row.get("why_it_matters"), field_name="why_it_matters"),
                completion_guidance=_text(
                    row.get("completion_guidance"), field_name="completion_guidance"
                ),
                priority=_positive_int(row.get("priority"), field_name="need priority"),
                order=_positive_int(row.get("order"), field_name="need order"),
                task_ids=tuple(_text(value, field_name="task_id") for value in task_ids),
            )
        )
    needs = tuple(needs_list)
    if len(needs) != len(raw_needs):
        raise ValueError("cached information_needs entries must be objects")
    tasks_list: list[PlanTask] = []
    for row in raw_tasks:
        if not isinstance(row, Mapping):
            raise ValueError("cached tasks entries must be objects")
        _expect_keys(
            row,
            {
                "task_id",
                "need_id",
                "query",
                "evidence_target",
                "priority",
                "order",
                "requirement_ids",
                "query_role",
                "depends_on",
            },
            label="cached task",
        )
        depends_on = row.get("depends_on")
        requirement_ids = row.get("requirement_ids")
        if not isinstance(depends_on, Sequence) or isinstance(depends_on, (str, bytes)):
            raise ValueError("cached task depends_on must be a list")
        if not isinstance(requirement_ids, Sequence) or isinstance(
            requirement_ids, (str, bytes)
        ):
            raise ValueError("cached task requirement_ids must be a list")
        tasks_list.append(
            PlanTask(
                task_id=_text(row.get("task_id"), field_name="task_id"),
                need_id=_text(row.get("need_id"), field_name="need_id"),
                query=_text(row.get("query"), field_name="query"),
                evidence_target=_text(row.get("evidence_target"), field_name="evidence_target"),
                priority=_positive_int(row.get("priority"), field_name="task priority"),
                order=_positive_int(row.get("order"), field_name="task order"),
                requirement_ids=tuple(
                    _text(value, field_name="requirement_id")
                    for value in requirement_ids
                ),
                query_role=_text(row.get("query_role"), field_name="query_role"),
                depends_on=tuple(_text(value, field_name="depends_on") for value in depends_on),
            )
        )
    tasks = tuple(tasks_list)
    plan = StaticResearchPlan(
        topic_id=_text(record.get("topic_id"), field_name="topic_id"),
        topic_sha256=_text(record.get("topic_sha256"), field_name="topic_sha256"),
        model=_text(record.get("model"), field_name="model"),
        prompt_version=_text(record.get("prompt_version"), field_name="prompt_version"),
        max_total_tasks=(
            None
            if record.get("max_total_tasks") is None
            else _positive_int(record.get("max_total_tasks"), field_name="max_total_tasks")
        ),
        planning_stage=_text(
            record.get("planning_stage"), field_name="planning_stage"
        ),
        draft_plan_sha256=(
            None
            if record.get("draft_plan_sha256") is None
            else _text(record.get("draft_plan_sha256"), field_name="draft_plan_sha256")
        ),
        requirements=requirements,
        information_needs=needs,
        tasks=tasks,
        raw_need_count=_nonnegative_int(
            record.get("raw_need_count"), field_name="raw_need_count"
        ),
        raw_task_count=_nonnegative_int(
            record.get("raw_task_count"), field_name="raw_task_count"
        ),
        truncated_task_count=_nonnegative_int(
            record.get("truncated_task_count"), field_name="truncated_task_count"
        ),
    )
    if record.get("plan_sha256") != plan.plan_sha256:
        raise ValueError("cached static plan fingerprint is invalid")
    return plan


@dataclass(frozen=True, slots=True)
class AllocatedTask:
    allocation_index: int
    need_id: str
    task: PlanTask


@dataclass(frozen=True, slots=True)
class TaskAllocation:
    requested_count: int
    tasks: tuple[AllocatedTask, ...]
    shortfall: int
    complete: bool
    policy: str = "breadth_first_round_robin_v1"


def allocate_tasks(plan: StaticResearchPlan, requested_count: int) -> TaskAllocation:
    """Take a deterministic prefix of the plan's sealed round-robin order."""

    if not isinstance(plan, StaticResearchPlan):
        raise TypeError("plan must be a StaticResearchPlan")
    _positive_int(requested_count, field_name="requested_count")
    selected = tuple(
        AllocatedTask(index, task.need_id, task)
        for index, task in enumerate(plan.tasks[:requested_count], start=1)
    )
    shortfall = requested_count - len(selected)
    return TaskAllocation(
        requested_count=requested_count,
        tasks=selected,
        shortfall=shortfall,
        complete=shortfall == 0,
    )


class StaticPlanner:
    """Draft, audit, and seal one static plan before retrieval."""

    def __init__(
        self,
        model: PlannerModel,
        *,
        auditor: PlannerAuditor,
        cache: PlanArtifactCache | None = None,
        max_total_tasks: int | None = None,
        provider_base_url: str = "unknown",
        model_name: str | None = None,
        prompt_version: str | None = None,
        auditor_model_name: str | None = None,
        audit_prompt_version: str | None = None,
    ) -> None:
        if not callable(getattr(model, "generate", None)):
            raise TypeError("model must implement generate")
        if not callable(getattr(auditor, "audit", None)):
            raise TypeError("auditor must implement audit")
        self.model = model
        self.auditor = auditor
        self.cache = cache
        self.max_total_tasks = (
            None
            if max_total_tasks is None
            else _positive_int(max_total_tasks, field_name="max_total_tasks")
        )
        self.provider_base_url = _text(provider_base_url, field_name="provider_base_url")
        self.model_name = model_name or str(getattr(model, "model", "unknown"))
        self.prompt_version = prompt_version or str(
            getattr(model, "prompt_version", DEFAULT_PROMPT_VERSION)
        )
        self.auditor_model_name = auditor_model_name or str(
            getattr(auditor, "model", self.model_name)
        )
        self.audit_prompt_version = audit_prompt_version or str(
            getattr(auditor, "prompt_version", f"{self.prompt_version}-audit")
        )

    def cache_key(
        self,
        topic: PlanningTopic,
        *,
        planning_stage: Literal["draft", "audited"] = "draft",
        draft_plan_sha256: str | None = None,
    ) -> str:
        if not isinstance(topic, PlanningTopic):
            raise TypeError("topic must be a PlanningTopic")
        if planning_stage == "draft" and draft_plan_sha256 is not None:
            raise ValueError("draft cache identity must not include a draft SHA-256")
        if planning_stage == "audited" and (
            not isinstance(draft_plan_sha256, str) or len(draft_plan_sha256) != 64
        ):
            raise ValueError("audited cache identity requires a draft SHA-256")
        return _sha256_json(
            {
                "schema_version": PLAN_SCHEMA_VERSION,
                "operation": PLANNER_OPERATION,
                "planning_stage": planning_stage,
                "provider_base_url": self.provider_base_url,
                "model": (
                    self.model_name
                    if planning_stage == "draft"
                    else self.auditor_model_name
                ),
                "prompt_version": (
                    self.prompt_version
                    if planning_stage == "draft"
                    else self.audit_prompt_version
                ),
                "max_total_tasks": self.max_total_tasks,
                "draft_plan_sha256": draft_plan_sha256,
                "topic": topic.as_record(),
            }
        )

    def _cached_plan(self, key: str) -> StaticResearchPlan | None:
        if self.cache is None:
            return None
        try:
            return self.cache.get(key)
        except Exception:
            return None

    def _store_plan(self, key: str, plan: StaticResearchPlan) -> None:
        if self.cache is None:
            return
        try:
            self.cache.put(key, plan)
        except Exception:
            pass

    def _failed_outcome(
        self,
        *,
        failure_kind: str,
        model: str,
        prompt_version: str,
        responses: Sequence[PlannerResponse],
        draft_cache_hit: bool,
        failure_stage: Literal["draft", "audit"],
        failure_detail: str,
        call_error: PlannerCallError | None = None,
        finish_reason: str | None = None,
        validation_error: str | None = None,
    ) -> PlannerOutcome:
        call_input_tokens = 0 if call_error is None else call_error.input_tokens
        call_output_tokens = 0 if call_error is None else call_error.output_tokens
        call_cost = 0.0 if call_error is None else call_error.cost
        call_latency = 0.0 if call_error is None else call_error.latency_seconds
        return PlannerOutcome(
            status="failed",
            plan=None,
            cache_hit=False,
            model=model,
            prompt_version=prompt_version,
            input_tokens=(
                sum(response.input_tokens for response in responses)
                + call_input_tokens
            ),
            output_tokens=(
                sum(response.output_tokens for response in responses)
                + call_output_tokens
            ),
            cost=sum(float(response.cost) for response in responses) + call_cost,
            latency_seconds=(
                sum(float(response.latency_seconds) for response in responses)
                + call_latency
            ),
            draft_cache_hit=draft_cache_hit,
            audit_cache_hit=False,
            failure_kind=failure_kind,
            failure_stage=failure_stage,
            failure_detail=failure_detail,
            validation_error=validation_error,
            finish_reason=(
                call_error.finish_reason if call_error is not None else finish_reason
            ),
        )

    def plan_topic(self, topic: PlanningTopic) -> PlannerOutcome:
        responses: list[PlannerResponse] = []
        draft_key = self.cache_key(topic, planning_stage="draft")
        draft = self._cached_plan(draft_key)
        draft_cache_hit = draft is not None
        if draft is not None and (
            draft.planning_stage != "draft" or draft.topic_sha256 != topic.input_sha256
        ):
            draft = None
            draft_cache_hit = False
        if draft is None:
            response: PlannerResponse | None = None
            try:
                response = self.model.generate(
                    topic, max_total_tasks=self.max_total_tasks
                )
                if not isinstance(response, PlannerResponse):
                    raise ValueError("planner model returned an invalid response envelope")
                responses.append(response)
                draft = parse_static_plan(
                    topic,
                    response.payload,
                    model=response.model,
                    prompt_version=response.prompt_version,
                    max_total_tasks=self.max_total_tasks,
                    planning_stage="draft",
                )
            except PlannerCallError as error:
                return self._failed_outcome(
                    failure_kind="draft_invalid_output",
                    model=self.model_name,
                    prompt_version=self.prompt_version,
                    responses=responses,
                    draft_cache_hit=False,
                    failure_stage="draft",
                    failure_detail=error.detail,
                    call_error=error,
                )
            except ValueError as error:
                return self._failed_outcome(
                    failure_kind="draft_invalid_output",
                    model=self.model_name,
                    prompt_version=self.prompt_version,
                    responses=responses,
                    draft_cache_hit=False,
                    failure_stage="draft",
                    failure_detail="local_validation",
                    finish_reason=(
                        None if response is None else response.finish_reason
                    ),
                    validation_error=_validation_error_code(error),
                )
            except Exception:
                return self._failed_outcome(
                    failure_kind="draft_model_failure",
                    model=self.model_name,
                    prompt_version=self.prompt_version,
                    responses=responses,
                    draft_cache_hit=False,
                    failure_stage="draft",
                    failure_detail="model_failure",
                )
            self._store_plan(draft_key, draft)

        audit_key = self.cache_key(
            topic,
            planning_stage="audited",
            draft_plan_sha256=draft.plan_sha256,
        )
        audited = self._cached_plan(audit_key)
        audit_cache_hit = audited is not None
        if audited is not None and (
            audited.planning_stage != "audited"
            or audited.topic_sha256 != topic.input_sha256
            or audited.draft_plan_sha256 != draft.plan_sha256
        ):
            audited = None
            audit_cache_hit = False
        if audited is None:
            audit_response: PlannerResponse | None = None
            try:
                audit_response = self.auditor.audit(
                    topic,
                    draft,
                    max_total_tasks=self.max_total_tasks,
                )
                if not isinstance(audit_response, PlannerResponse):
                    raise ValueError("planner auditor returned an invalid response envelope")
                responses.append(audit_response)
                audited = parse_static_plan(
                    topic,
                    audit_response.payload,
                    model=audit_response.model,
                    prompt_version=audit_response.prompt_version,
                    max_total_tasks=self.max_total_tasks,
                    planning_stage="audited",
                    draft_plan_sha256=draft.plan_sha256,
                )
            except PlannerCallError as error:
                return self._failed_outcome(
                    failure_kind="audit_invalid_output",
                    model=self.auditor_model_name,
                    prompt_version=self.audit_prompt_version,
                    responses=responses,
                    draft_cache_hit=draft_cache_hit,
                    failure_stage="audit",
                    failure_detail=error.detail,
                    call_error=error,
                )
            except ValueError as error:
                return self._failed_outcome(
                    failure_kind="audit_invalid_output",
                    model=self.auditor_model_name,
                    prompt_version=self.audit_prompt_version,
                    responses=responses,
                    draft_cache_hit=draft_cache_hit,
                    failure_stage="audit",
                    failure_detail="local_validation",
                    finish_reason=(
                        None
                        if audit_response is None
                        else audit_response.finish_reason
                    ),
                    validation_error=_validation_error_code(error),
                )
            except Exception:
                return self._failed_outcome(
                    failure_kind="audit_model_failure",
                    model=self.auditor_model_name,
                    prompt_version=self.audit_prompt_version,
                    responses=responses,
                    draft_cache_hit=draft_cache_hit,
                    failure_stage="audit",
                    failure_detail="model_failure",
                )
            self._store_plan(audit_key, audited)

        return PlannerOutcome(
            status="success",
            plan=audited,
            cache_hit=audit_cache_hit,
            model=audited.model,
            prompt_version=audited.prompt_version,
            input_tokens=sum(response.input_tokens for response in responses),
            output_tokens=sum(response.output_tokens for response in responses),
            cost=sum(float(response.cost) for response in responses),
            latency_seconds=sum(float(response.latency_seconds) for response in responses),
            draft_cache_hit=draft_cache_hit,
            audit_cache_hit=audit_cache_hit,
        )


class AsymmetricStaticPlanner:
    """Extract a compact coverage ledger, then expand one strict final plan."""

    def __init__(
        self,
        model: CoveragePlannerModel,
        *,
        ledger_cache: CoverageLedgerArtifactCache | None = None,
        expansion_cache: CoverageExpansionArtifactCache | None = None,
        max_total_tasks: int | None = None,
        provider_base_url: str = "unknown",
        model_name: str | None = None,
        ledger_prompt_version: str | None = None,
        expansion_prompt_version: str | None = None,
        ledger_max_tokens: int | None = None,
        expansion_max_tokens: int | None = None,
        ledger_gate_max_output_tokens: int = 8_000,
        expansion_gate_max_output_tokens: int = 24_000,
    ) -> None:
        if not callable(getattr(model, "generate_coverage_ledger", None)):
            raise TypeError("model must implement generate_coverage_ledger")
        if not callable(getattr(model, "expand_coverage_ledger", None)):
            raise TypeError("model must implement expand_coverage_ledger")
        self.model = model
        self.ledger_cache = ledger_cache
        self.expansion_cache = expansion_cache
        if max_total_tasks is not None:
            raise ValueError("coverage-ledger richer path does not accept a task cap")
        self.max_total_tasks = None
        self.provider_base_url = _text(
            provider_base_url, field_name="provider_base_url"
        )
        self.model_name = model_name or str(getattr(model, "model", "unknown"))
        self.ledger_prompt_version = ledger_prompt_version or str(
            getattr(model, "ledger_prompt_version", DEFAULT_LEDGER_PROMPT_VERSION)
        )
        self.expansion_prompt_version = expansion_prompt_version or str(
            getattr(
                model,
                "expansion_prompt_version",
                DEFAULT_EXPANSION_PROMPT_VERSION,
            )
        )
        self.ledger_max_tokens = _positive_int(
            (
                getattr(model, "max_tokens", DEFAULT_PLANNER_MAX_TOKENS)
                if ledger_max_tokens is None
                else ledger_max_tokens
            ),
            field_name="ledger_max_tokens",
        )
        self.expansion_max_tokens = _positive_int(
            (
                getattr(model, "audit_max_tokens", DEFAULT_AUDIT_MAX_TOKENS)
                if expansion_max_tokens is None
                else expansion_max_tokens
            ),
            field_name="expansion_max_tokens",
        )
        self.ledger_gate_max_output_tokens = _positive_int(
            ledger_gate_max_output_tokens,
            field_name="ledger_gate_max_output_tokens",
        )
        if self.ledger_gate_max_output_tokens >= self.ledger_max_tokens:
            raise ValueError("ledger headroom gate must be below ledger_max_tokens")
        self.expansion_gate_max_output_tokens = _positive_int(
            expansion_gate_max_output_tokens,
            field_name="expansion_gate_max_output_tokens",
        )
        if self.expansion_gate_max_output_tokens >= self.expansion_max_tokens:
            raise ValueError(
                "expansion headroom gate must be below expansion_max_tokens"
            )

    def ledger_cache_key(self, topic: PlanningTopic) -> str:
        return _sha256_json(
            {
                "schema_version": COVERAGE_LEDGER_SCHEMA_VERSION,
                "operation": COVERAGE_LEDGER_OPERATION,
                "provider_base_url": self.provider_base_url,
                "model": self.model_name,
                "prompt_version": self.ledger_prompt_version,
                "max_tokens": self.ledger_max_tokens,
                "gate_max_output_tokens": self.ledger_gate_max_output_tokens,
                "topic": topic.as_record(),
            }
        )

    def plan_cache_key(self, topic: PlanningTopic, ledger: CoverageLedger) -> str:
        return _sha256_json(
            {
                "schema_version": PLAN_SCHEMA_VERSION,
                "operation": COVERAGE_EXPANSION_OPERATION,
                "provider_base_url": self.provider_base_url,
                "model": self.model_name,
                "prompt_version": self.expansion_prompt_version,
                "assembler_version": (
                    COMPACT_EXPANSION_ASSEMBLER_VERSION
                    if self.expansion_prompt_version
                    in COMPACT_EXPANSION_PROMPT_VERSIONS
                    else None
                ),
                "max_tokens": self.expansion_max_tokens,
                "gate_max_output_tokens": self.expansion_gate_max_output_tokens,
                "max_total_tasks": self.max_total_tasks,
                "coverage_ledger_sha256": ledger.ledger_sha256,
                "topic": topic.as_record(),
            }
        )

    def _failure(
        self,
        *,
        failure_kind: str,
        failure_stage: Literal["draft", "audit"],
        failure_detail: str,
        ledger_cache_hit: bool,
        ledger_response: PlannerResponse | None = None,
        expansion_response: PlannerResponse | None = None,
        ledger_metrics: PlannerStageMetrics | None = None,
        expansion_metrics: PlannerStageMetrics | None = None,
        call_error: PlannerCallError | None = None,
        validation_error: str | None = None,
    ) -> PlannerOutcome:
        responses = tuple(
            response
            for response in (ledger_response, expansion_response)
            if response is not None
        )
        call_input = 0 if call_error is None else call_error.input_tokens
        call_output = 0 if call_error is None else call_error.output_tokens
        call_cost = 0.0 if call_error is None else call_error.cost
        call_latency = 0.0 if call_error is None else call_error.latency_seconds
        first_input = 0 if ledger_metrics is None else ledger_metrics.input_tokens
        first_output = 0 if ledger_metrics is None else ledger_metrics.output_tokens
        second_input = (
            0 if expansion_metrics is None else expansion_metrics.input_tokens
        )
        second_output = (
            0 if expansion_metrics is None else expansion_metrics.output_tokens
        )
        if ledger_response is not None:
            first_input = ledger_response.input_tokens
            first_output = ledger_response.output_tokens
        if expansion_response is not None:
            second_input = expansion_response.input_tokens
            second_output = expansion_response.output_tokens
        if call_error is not None and failure_stage == "draft":
            first_input += call_input
            first_output += call_output
        if call_error is not None and failure_stage == "audit":
            second_input += call_input
            second_output += call_output
        finish_reason = (
            call_error.finish_reason
            if call_error is not None
            else (
                None
                if failure_stage == "draft" and ledger_response is None
                else (
                    ledger_response.finish_reason
                    if failure_stage == "draft"
                    else (
                        None
                        if expansion_response is None
                        else expansion_response.finish_reason
                    )
                )
            )
        )
        return PlannerOutcome(
            status="failed",
            plan=None,
            cache_hit=False,
            model=self.model_name,
            prompt_version=(
                self.ledger_prompt_version
                if failure_stage == "draft"
                else self.expansion_prompt_version
            ),
            input_tokens=sum(response.input_tokens for response in responses)
            + call_input,
            output_tokens=sum(response.output_tokens for response in responses)
            + call_output,
            cost=sum(float(response.cost) for response in responses) + call_cost,
            latency_seconds=sum(
                float(response.latency_seconds) for response in responses
            )
            + call_latency,
            draft_cache_hit=ledger_cache_hit,
            audit_cache_hit=False,
            failure_kind=failure_kind,
            failure_stage=failure_stage,
            failure_detail=failure_detail,
            validation_error=validation_error,
            finish_reason=finish_reason,
            planner_path="coverage_ledger_v1",
            first_stage_input_tokens=first_input,
            first_stage_output_tokens=first_output,
            second_stage_input_tokens=second_input,
            second_stage_output_tokens=second_output,
            second_stage_finish_reason=(
                call_error.finish_reason
                if call_error is not None and failure_stage == "audit"
                else (
                    expansion_response.finish_reason
                    if expansion_response is not None
                    else (
                        None
                        if expansion_metrics is None
                        else expansion_metrics.finish_reason
                    )
                )
            ),
            content_shape=(
                None if call_error is None else call_error.content_shape
            ),
            parse_error_offset=(
                None if call_error is None else call_error.parse_error_offset
            ),
        )

    def plan_topic(self, topic: PlanningTopic) -> PlannerOutcome:
        ledger_response: PlannerResponse | None = None
        expansion_response: PlannerResponse | None = None
        ledger_key = self.ledger_cache_key(topic)
        ledger_entry: CoverageLedgerCacheEntry | None = None
        if self.ledger_cache is not None:
            try:
                ledger_entry = self.ledger_cache.get(ledger_key)
            except Exception:
                return self._failure(
                    failure_kind="cache_failure",
                    failure_stage="draft",
                    failure_detail="ledger_cache_read_failure",
                    ledger_cache_hit=False,
                )
        ledger_cache_hit = ledger_entry is not None
        if ledger_entry is not None and (
            ledger_entry.ledger.topic_sha256 != topic.input_sha256
            or ledger_entry.metrics.finish_reason != "stop"
            or ledger_entry.metrics.output_tokens
            >= self.ledger_gate_max_output_tokens
        ):
            return self._failure(
                failure_kind="cache_failure",
                failure_stage="draft",
                failure_detail="ledger_cache_validation_failure",
                ledger_cache_hit=True,
                ledger_metrics=ledger_entry.metrics,
            )
        if ledger_entry is None:
            try:
                ledger_response = self.model.generate_coverage_ledger(topic)
                if not isinstance(ledger_response, PlannerResponse):
                    raise ValueError(
                        "coverage model returned an invalid response envelope"
                    )
                if ledger_response.finish_reason != "stop":
                    return self._failure(
                        failure_kind="draft_invalid_output",
                        failure_stage="draft",
                        failure_detail="unexpected_finish_reason",
                        ledger_cache_hit=False,
                        ledger_response=ledger_response,
                    )
                ledger_metrics = PlannerStageMetrics.from_response(
                    ledger_response
                )
                if (
                    ledger_metrics.output_tokens
                    >= self.ledger_gate_max_output_tokens
                ):
                    return self._failure(
                        failure_kind="draft_invalid_output",
                        failure_stage="draft",
                        failure_detail="output_headroom_exceeded",
                        ledger_cache_hit=False,
                        ledger_response=ledger_response,
                        ledger_metrics=ledger_metrics,
                    )
                ledger = parse_coverage_ledger(
                    topic,
                    ledger_response.payload,
                    model=ledger_response.model,
                    prompt_version=ledger_response.prompt_version,
                )
            except PlannerCallError as error:
                return self._failure(
                    failure_kind="draft_invalid_output",
                    failure_stage="draft",
                    failure_detail=error.detail,
                    ledger_cache_hit=False,
                    call_error=error,
                )
            except ValueError as error:
                return self._failure(
                    failure_kind="draft_invalid_output",
                    failure_stage="draft",
                    failure_detail="local_validation",
                    ledger_cache_hit=False,
                    ledger_response=ledger_response,
                    validation_error=_validation_error_code(error),
                )
            except Exception:
                return self._failure(
                    failure_kind="draft_model_failure",
                    failure_stage="draft",
                    failure_detail="model_failure",
                    ledger_cache_hit=False,
                )
            ledger_entry = CoverageLedgerCacheEntry(ledger, ledger_metrics)
            if self.ledger_cache is not None:
                try:
                    stored = self.ledger_cache.put(ledger_key, ledger_entry)
                except Exception:
                    stored = False
                if not stored:
                    return self._failure(
                        failure_kind="cache_failure",
                        failure_stage="draft",
                        failure_detail="ledger_cache_write_failure",
                        ledger_cache_hit=False,
                        ledger_response=ledger_response,
                        ledger_metrics=ledger_metrics,
                    )

        ledger = ledger_entry.ledger
        ledger_metrics = ledger_entry.metrics

        plan_key = self.plan_cache_key(topic, ledger)
        expansion_entry: CoverageExpansionCacheEntry | None = None
        if self.expansion_cache is not None:
            try:
                expansion_entry = self.expansion_cache.get(plan_key)
            except Exception:
                return self._failure(
                    failure_kind="cache_failure",
                    failure_stage="audit",
                    failure_detail="expansion_cache_read_failure",
                    ledger_cache_hit=ledger_cache_hit,
                    ledger_response=ledger_response,
                    ledger_metrics=ledger_metrics,
                )
        expansion_cache_hit = expansion_entry is not None
        if expansion_entry is not None and (
            expansion_entry.plan.planning_stage != "audited"
            or expansion_entry.plan.topic_sha256 != topic.input_sha256
            or expansion_entry.plan.draft_plan_sha256 != ledger.ledger_sha256
            or any(task.depends_on for task in expansion_entry.plan.tasks)
            or expansion_entry.metrics.finish_reason != "stop"
            or expansion_entry.metrics.output_tokens
            >= self.expansion_gate_max_output_tokens
        ):
            return self._failure(
                failure_kind="cache_failure",
                failure_stage="audit",
                failure_detail="expansion_cache_validation_failure",
                ledger_cache_hit=ledger_cache_hit,
                ledger_response=ledger_response,
                ledger_metrics=ledger_metrics,
                expansion_metrics=expansion_entry.metrics,
            )
        if expansion_entry is None:
            try:
                expansion_response = self.model.expand_coverage_ledger(
                    topic,
                    ledger,
                    max_total_tasks=self.max_total_tasks,
                )
                if not isinstance(expansion_response, PlannerResponse):
                    raise ValueError("expansion model returned an invalid response envelope")
                if expansion_response.finish_reason != "stop":
                    return self._failure(
                        failure_kind="audit_invalid_output",
                        failure_stage="audit",
                        failure_detail="unexpected_finish_reason",
                        ledger_cache_hit=ledger_cache_hit,
                        ledger_response=ledger_response,
                        expansion_response=expansion_response,
                        ledger_metrics=ledger_metrics,
                    )
                expansion_metrics = PlannerStageMetrics.from_response(
                    expansion_response
                )
                if (
                    expansion_metrics.output_tokens
                    >= self.expansion_gate_max_output_tokens
                ):
                    return self._failure(
                        failure_kind="audit_invalid_output",
                        failure_stage="audit",
                        failure_detail="output_headroom_exceeded",
                        ledger_cache_hit=ledger_cache_hit,
                        ledger_response=ledger_response,
                        expansion_response=expansion_response,
                        ledger_metrics=ledger_metrics,
                        expansion_metrics=expansion_metrics,
                    )
                plan = (
                    parse_compact_query_expansion(
                        topic,
                        ledger,
                        expansion_response.payload,
                        model=expansion_response.model,
                        prompt_version=(
                            f"{expansion_response.prompt_version}+"
                            f"{COMPACT_EXPANSION_ASSEMBLER_VERSION}"
                        ),
                    )
                    if self.expansion_prompt_version
                    in COMPACT_EXPANSION_PROMPT_VERSIONS
                    else parse_static_plan(
                        topic,
                        expansion_response.payload,
                        model=expansion_response.model,
                        prompt_version=expansion_response.prompt_version,
                        max_total_tasks=self.max_total_tasks,
                        planning_stage="audited",
                        draft_plan_sha256=ledger.ledger_sha256,
                    )
                )
                if any(task.depends_on for task in plan.tasks):
                    raise ValueError(
                        "coverage-ledger plan tasks must be dependency-free"
                    )
            except PlannerCallError as error:
                return self._failure(
                    failure_kind="audit_invalid_output",
                    failure_stage="audit",
                    failure_detail=error.detail,
                    ledger_cache_hit=ledger_cache_hit,
                    ledger_response=ledger_response,
                    ledger_metrics=ledger_metrics,
                    call_error=error,
                )
            except ValueError as error:
                validation_error = _validation_error_code(error)
                if (
                    self.expansion_prompt_version == "planner-static-expand-v3"
                    and expansion_response is not None
                ):
                    try:
                        plan = build_ledger_fallback_plan(
                            topic,
                            ledger,
                            model=expansion_response.model,
                            expansion_prompt_version=(
                                expansion_response.prompt_version
                            ),
                            validation_error=validation_error,
                            source_payload=expansion_response.payload,
                        )
                    except ValueError as fallback_error:
                        return self._failure(
                            failure_kind="audit_invalid_output",
                            failure_stage="audit",
                            failure_detail="fallback_local_validation",
                            ledger_cache_hit=ledger_cache_hit,
                            ledger_response=ledger_response,
                            expansion_response=expansion_response,
                            ledger_metrics=ledger_metrics,
                            validation_error=(
                                _validation_error_code(fallback_error)
                            ),
                        )
                else:
                    return self._failure(
                        failure_kind="audit_invalid_output",
                        failure_stage="audit",
                        failure_detail="local_validation",
                        ledger_cache_hit=ledger_cache_hit,
                        ledger_response=ledger_response,
                        expansion_response=expansion_response,
                        ledger_metrics=ledger_metrics,
                        validation_error=validation_error,
                    )
            except Exception:
                return self._failure(
                    failure_kind="audit_model_failure",
                    failure_stage="audit",
                    failure_detail="model_failure",
                    ledger_cache_hit=ledger_cache_hit,
                    ledger_response=ledger_response,
                    ledger_metrics=ledger_metrics,
                )
            expansion_entry = CoverageExpansionCacheEntry(plan, expansion_metrics)
            if self.expansion_cache is not None:
                try:
                    stored = self.expansion_cache.put(plan_key, expansion_entry)
                except Exception:
                    stored = False
                if not stored:
                    return self._failure(
                        failure_kind="cache_failure",
                        failure_stage="audit",
                        failure_detail="expansion_cache_write_failure",
                        ledger_cache_hit=ledger_cache_hit,
                        ledger_response=ledger_response,
                        expansion_response=expansion_response,
                        ledger_metrics=ledger_metrics,
                        expansion_metrics=expansion_metrics,
                    )

        plan = expansion_entry.plan
        expansion_metrics = expansion_entry.metrics
        fallback_reason = _fallback_reason_from_prompt_version(
            plan.prompt_version
        )

        responses = tuple(
            response
            for response in (ledger_response, expansion_response)
            if response is not None
        )
        return PlannerOutcome(
            status="success",
            plan=plan,
            cache_hit=expansion_cache_hit,
            model=plan.model,
            prompt_version=plan.prompt_version,
            input_tokens=sum(response.input_tokens for response in responses),
            output_tokens=sum(response.output_tokens for response in responses),
            cost=sum(float(response.cost) for response in responses),
            latency_seconds=sum(
                float(response.latency_seconds) for response in responses
            ),
            draft_cache_hit=ledger_cache_hit,
            audit_cache_hit=expansion_cache_hit,
            planner_path="coverage_ledger_v1",
            first_stage_input_tokens=(
                ledger_metrics.input_tokens
            ),
            first_stage_output_tokens=(
                ledger_metrics.output_tokens
            ),
            second_stage_input_tokens=(
                expansion_metrics.input_tokens
            ),
            second_stage_output_tokens=(
                expansion_metrics.output_tokens
            ),
            second_stage_finish_reason=(
                expansion_metrics.finish_reason
            ),
            fallback_used=fallback_reason is not None,
            fallback_reason=fallback_reason,
        )


COVERAGE_LEDGER_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "information_needs": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "need_ref": {"type": "string", "minLength": 1},
                    "objective": {"type": "string", "minLength": 1},
                    "why_it_matters": {"type": "string", "minLength": 1},
                    "priority": {"type": "integer", "minimum": 1},
                    "requirements": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "requirement_ref": {
                                    "type": "string",
                                    "minLength": 1,
                                },
                                "description": {
                                    "type": "string",
                                    "minLength": 1,
                                },
                                "priority": {"type": "integer", "minimum": 1},
                                "retrieval_anchors": {
                                    "type": "array",
                                    "minItems": 1,
                                    "maxItems": 5,
                                    "uniqueItems": True,
                                    "items": {"type": "string", "minLength": 1},
                                },
                            },
                            "required": [
                                "requirement_ref",
                                "description",
                                "priority",
                                "retrieval_anchors",
                            ],
                        },
                    },
                },
                "required": [
                    "need_ref",
                    "objective",
                    "why_it_matters",
                    "priority",
                    "requirements",
                ],
            },
        }
    },
    "required": ["information_needs"],
}


MINIMAL_COVERAGE_LEDGER_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "information_needs": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "need_ref": {"type": "string", "minLength": 1},
                    "objective": {"type": "string", "minLength": 1},
                    "priority": {"type": "integer", "minimum": 1},
                    "requirements": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "requirement_ref": {
                                    "type": "string",
                                    "minLength": 1,
                                },
                                "description": {
                                    "type": "string",
                                    "minLength": 1,
                                },
                                "priority": {"type": "integer", "minimum": 1},
                            },
                            "required": [
                                "requirement_ref",
                                "description",
                                "priority",
                            ],
                        },
                    },
                },
                "required": [
                    "need_ref",
                    "objective",
                    "priority",
                    "requirements",
                ],
            },
        }
    },
    "required": ["information_needs"],
}


def _expand_minimal_coverage_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Project a minimal provider ledger into the canonical local ledger shape."""

    if not isinstance(payload, Mapping):
        raise ValueError("minimal coverage ledger output must be an object")
    raw_needs = payload.get("information_needs")
    if (
        not isinstance(raw_needs, Sequence)
        or isinstance(raw_needs, (str, bytes))
        or not raw_needs
    ):
        raise ValueError("minimal coverage information_needs must be non-empty")
    needs: list[dict[str, Any]] = []
    for raw_need in raw_needs:
        if not isinstance(raw_need, Mapping):
            raise ValueError("minimal coverage needs must be objects")
        objective = _text(raw_need.get("objective"), field_name="objective")
        raw_requirements = raw_need.get("requirements")
        if (
            not isinstance(raw_requirements, Sequence)
            or isinstance(raw_requirements, (str, bytes))
            or not raw_requirements
        ):
            raise ValueError("minimal coverage requirements must be non-empty")
        requirements: list[dict[str, Any]] = []
        for raw_requirement in raw_requirements:
            if not isinstance(raw_requirement, Mapping):
                raise ValueError("minimal coverage requirements must be objects")
            description = _text(
                raw_requirement.get("description"),
                field_name="requirement description",
            )
            requirements.append(
                {
                    "requirement_ref": _text(
                        raw_requirement.get("requirement_ref"),
                        field_name="requirement_ref",
                    ),
                    "description": description,
                    "priority": _positive_int(
                        raw_requirement.get("priority"),
                        field_name="requirement priority",
                    ),
                    "retrieval_anchors": [description],
                }
            )
        needs.append(
            {
                "need_ref": _text(
                    raw_need.get("need_ref"), field_name="need_ref"
                ),
                "objective": objective,
                "why_it_matters": objective,
                "priority": _positive_int(
                    raw_need.get("priority"), field_name="need priority"
                ),
                "requirements": requirements,
            }
        )
    return {"information_needs": needs}


COMPACT_QUERY_EXPANSION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "query_sets": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "requirement_ref": {"type": "string", "minLength": 1},
                    "queries": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "query": {"type": "string", "minLength": 1},
                                "query_role": {
                                    "type": "string",
                                    "enum": sorted(QUERY_ROLES),
                                },
                                "anchor_dimension": {
                                    "type": "string",
                                    "minLength": 1,
                                },
                            },
                            "required": [
                                "query",
                                "query_role",
                                "anchor_dimension",
                            ],
                        },
                    },
                },
                "required": ["requirement_ref", "queries"],
            },
        }
    },
    "required": ["query_sets"],
}


STATIC_PLAN_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "requirements": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "requirement_ref": {"type": "string"},
                    "description": {"type": "string"},
                    "priority": {"type": "integer", "minimum": 1},
                },
                "required": ["requirement_ref", "description", "priority"],
            },
        },
        "information_needs": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "need_ref": {"type": "string"},
                    "objective": {"type": "string"},
                    "why_it_matters": {"type": "string"},
                    "completion_guidance": {"type": "string"},
                    "priority": {"type": "integer", "minimum": 1},
                    "search_tasks": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "task_ref": {"type": "string"},
                                "query": {"type": "string"},
                                "evidence_target": {"type": "string"},
                                "priority": {"type": "integer", "minimum": 1},
                                "depends_on": {"type": "array", "items": {"type": "string"}},
                                "requirement_refs": {
                                    "type": "array",
                                    "minItems": 1,
                                    "uniqueItems": True,
                                    "items": {"type": "string"},
                                },
                                "query_role": {
                                    "type": "string",
                                    "enum": sorted(QUERY_ROLES),
                                },
                            },
                            "required": [
                                "task_ref",
                                "query",
                                "evidence_target",
                                "priority",
                                "depends_on",
                                "requirement_refs",
                                "query_role",
                            ],
                        },
                    },
                },
                "required": [
                    "need_ref",
                    "objective",
                    "why_it_matters",
                    "completion_guidance",
                    "priority",
                    "search_tasks",
                ],
            },
        }
    },
    "required": ["requirements", "information_needs"],
}


def _decode_json_content(
    content: str, *, required_keys: Sequence[str]
) -> Any:
    """Decode JSON, tolerating only an in-memory fence or prose wrapper."""

    try:
        return json.loads(content)
    except json.JSONDecodeError as original_error:
        required = set(required_keys)
        decoder = json.JSONDecoder()
        for index, character in enumerate(content):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(content[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, Mapping) and required.issubset(value):
                return value
        raise original_error


def _content_shape(content: str) -> Literal[
    "starts_object", "fenced", "prose_prefix", "no_object"
]:
    stripped = content.lstrip()
    if stripped.startswith("{"):
        return "starts_object"
    if stripped.startswith("```"):
        return "fenced"
    if "{" in stripped:
        return "prose_prefix"
    return "no_object"


class OpenAIStaticPlannerModel:
    """Provider adapter using the repository-wide provider resolution only."""

    def __init__(
        self,
        client: Any,
        *,
        model: str = DEFAULT_PLANNER_MODEL,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
        audit_prompt_version: str = DEFAULT_AUDIT_PROMPT_VERSION,
        ledger_prompt_version: str = DEFAULT_LEDGER_PROMPT_VERSION,
        expansion_prompt_version: str = DEFAULT_EXPANSION_PROMPT_VERSION,
        max_tokens: int = DEFAULT_PLANNER_MAX_TOKENS,
        audit_max_tokens: int = DEFAULT_AUDIT_MAX_TOKENS,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not callable(getattr(client.chat.completions, "create", None)):
            raise TypeError("client must expose chat.completions.create")
        self.client = client
        self.model = _text(model, field_name="model")
        self.prompt_version = _text(prompt_version, field_name="prompt_version")
        self.audit_prompt_version = _text(
            audit_prompt_version, field_name="audit_prompt_version"
        )
        self.ledger_prompt_version = _text(
            ledger_prompt_version, field_name="ledger_prompt_version"
        )
        self.expansion_prompt_version = _text(
            expansion_prompt_version, field_name="expansion_prompt_version"
        )
        self.max_tokens = _positive_int(max_tokens, field_name="max_tokens")
        self.audit_max_tokens = _positive_int(
            audit_max_tokens, field_name="audit_max_tokens"
        )
        if timeout_seconds <= 0 or not math.isfinite(float(timeout_seconds)):
            raise ValueError("timeout_seconds must be finite and positive")
        self.timeout_seconds = float(timeout_seconds)

    @staticmethod
    def _limit_instruction(max_total_tasks: int | None) -> str:
        if max_total_tasks is None:
            return (
                "There is no artificial task-count cap. Include every distinct task "
                "needed for complete coverage, but do not add redundant variants."
            )
        return f"The local hard cap is {max_total_tasks} tasks."

    def _create(
        self,
        *,
        system: str,
        user_payload: Mapping[str, Any],
        prompt_version: str,
        schema_name: str,
        max_tokens: int,
        schema: Mapping[str, Any] = STATIC_PLAN_JSON_SCHEMA,
    ) -> PlannerResponse:
        started = time.perf_counter()
        user = json.dumps(user_payload, ensure_ascii=False, sort_keys=True)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            },
            timeout=self.timeout_seconds,
        )
        usage = response.usage if hasattr(response, "usage") else response.get("usage")
        input_tokens = getattr(usage, "prompt_tokens", 0) if usage is not None else 0
        output_tokens = getattr(usage, "completion_tokens", 0) if usage is not None else 0
        reported_cost = getattr(usage, "cost", 0.0) if usage is not None else 0.0
        choices = response.choices if hasattr(response, "choices") else response.get("choices")
        if not isinstance(choices, Sequence) or not choices:
            raise PlannerCallError(
                "invalid_response_envelope",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=reported_cost,
                latency_seconds=round(time.perf_counter() - started, 6),
            )
        choice = choices[0]
        finish_reason = (
            choice.finish_reason
            if hasattr(choice, "finish_reason")
            else choice.get("finish_reason")
        )
        if finish_reason == "length":
            raise PlannerCallError(
                "output_truncated",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=reported_cost,
                latency_seconds=round(time.perf_counter() - started, 6),
                finish_reason=finish_reason,
            )
        message = choice.message if hasattr(choice, "message") else choice.get("message")
        content = message.content if hasattr(message, "content") else message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise PlannerCallError(
                "output_truncated" if finish_reason == "length" else "missing_content",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=reported_cost,
                latency_seconds=round(time.perf_counter() - started, 6),
                finish_reason=finish_reason,
            )
        try:
            payload = _decode_json_content(
                content, required_keys=tuple(schema.get("required", ()))
            )
        except json.JSONDecodeError as error:
            raise PlannerCallError(
                "output_truncated" if finish_reason == "length" else "invalid_json",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=reported_cost,
                latency_seconds=round(time.perf_counter() - started, 6),
                finish_reason=finish_reason,
                content_shape=_content_shape(content),
                parse_error_offset=error.pos,
            ) from error
        return PlannerResponse(
            payload=payload,
            model=self.model,
            prompt_version=prompt_version,
            input_tokens=_nonnegative_int(input_tokens, field_name="input_tokens"),
            output_tokens=_nonnegative_int(output_tokens, field_name="output_tokens"),
            cost=float(reported_cost or 0.0),
            latency_seconds=round(time.perf_counter() - started, 6),
            finish_reason=finish_reason,
        )

    def generate(
        self, topic: PlanningTopic, *, max_total_tasks: int | None
    ) -> PlannerResponse:
        system = (
            "Create a draft static research plan for candidate recall before any "
            "retrieval. The downstream system retrieves candidate documents, extracts "
            "nuggets, and deduplicates later; do not optimize for a minimal query list "
            "or report-ready prose. First atomize "
            "every independently searchable request in the original topic into the "
            "requirements array; do not silently drop details from either the problem "
            "statement or background. Then group requirements into information needs "
            "and create complementary queries that open distinct retrieval directions. "
            "Useful directions may differ by atomic need, named entity, law or program, "
            "source family, measurement or outcome, jurisdiction or language, technical "
            "term, or evidence polarity. Semantic overlap is allowed when those anchors "
            "differ and may shift dense-retrieval candidates. Avoid pure paraphrases "
            "that retain the same anchors. Every requirement must "
            "be targeted by at least one task and every task must name its requirement "
            "refs and query role. Use broad discovery for scope, precise facts and "
            "official-primary queries for verifiable details, measurable-outcome "
            "queries for requested effects, comparisons where requested, and contrary "
            "evidence where conclusions may conflict, but do not create queries merely "
            "to fill a query-role category. Do not impose dates, locations, "
            "populations, or examples absent from the topic. When the topic asks for "
            "multiple countries or cases, retain a broad discovery query rather than "
            "substituting a few familiar examples. Distinguish requested outcomes from "
            "easier proxy measures. Exact duplicate query strings are invalid. Tasks must "
            "be useful without retrieval feedback. Do not write the report, cite "
            "documents, or include markdown. "
            + self._limit_instruction(max_total_tasks)
        )
        return self._create(
            system=system,
            user_payload={
                "topic": topic.as_record(),
                "required_output": (
                    "atomic requirements plus information needs and fully mapped "
                    "query variants"
                ),
            },
            prompt_version=self.prompt_version,
            schema_name="planner_static_draft_v3",
            max_tokens=self.max_tokens,
        )

    def generate_coverage_ledger(
        self, topic: PlanningTopic
    ) -> PlannerResponse:
        minimal = self.ledger_prompt_version == "planner-static-ledger-v3"
        final_contract = self.ledger_prompt_version == "planner-static-ledger-v4"
        if final_contract:
            system = (
                "Create the smallest complete set of atomic, independently "
                "answerable report requirements before retrieval. Read the complete "
                "title, problem statement, and requester background. The problem "
                "statement is authoritative for what the report must and must not "
                "contain. Use the title for disambiguation. Use the background to "
                "recover factual questions, comparisons, constraints, outcomes, "
                "audience needs, and decision-relevant context not already stated. "
                "Purely biographical or workflow context may affect priority and "
                "framing but must not become a standalone requirement. Merge "
                "restatements and semantic duplicates across the problem statement "
                "and background. Do not split one requirement merely by source type, "
                "query strategy, terminology, or evidence polarity. Preserve every "
                "distinct requested entity, comparison, qualifier, population, "
                "jurisdiction, outcome, timeframe, uncertainty, limitation, and "
                "evidence type. Requests such as if any, whether, balanced, sober "
                "assessment, or honest numbers license verification and potentially "
                "contrary evidence. Group the requirements into coherent information "
                "needs. Give each requirement one to five short, non-exhaustive "
                "candidate-shifting retrieval-anchor seeds. Do not invent dates, "
                "locations, populations, examples, or report sections. This is a "
                "coverage ledger, not a query plan: emit no search tasks, query roles, "
                "dependencies, report prose, citations, or markdown. Return exactly "
                "one schema-valid JSON object."
            )
        elif minimal:
            system = (
                "Create the smallest complete coverage ledger before retrieval. Nest "
                "every atomic, independently searchable requirement from the "
                "unabridged problem statement and background under a short "
                "information-need objective. Do not drop qualifiers, comparisons, "
                "requested outcomes, populations, jurisdictions, resource types, or "
                "contrary-evidence needs. Emit only stable references, short "
                "objectives, priorities, and atomic requirement descriptions. Do not "
                "emit reasons, anchors, query ideas, search tasks, query-role labels, "
                "dependencies, prose reports, citations, or markdown. Do not impose "
                "dates, locations, populations, or examples absent from the topic."
            )
        else:
            system = (
                "Create a compact coverage ledger before retrieval. Nest every atomic, "
                "independently searchable requirement from the unabridged problem "
                "statement and background under an appropriate information need. Do not "
                "drop qualifiers, comparisons, requested outcomes, populations, "
                "jurisdictions, resource types, or contrary-evidence needs. For each "
                "requirement, list short retrieval-anchor ideas that could move a dense "
                "retriever toward different candidates, such as named entities, official "
                "source families, measures, jurisdictions, languages, technical terms, "
                "or evidence polarity. This is a coverage ledger, not an executable query "
                "plan: do not emit search tasks, query-role labels, dependencies, prose "
                "reports, citations, or markdown. Do not impose dates, locations, "
                "populations, or examples absent from the topic."
            )
        user_payload = {
            "topic": topic.as_record(),
            "required_output": (
                "nested information needs and atomic requirements"
                if minimal
                else (
                    "nested information needs, atomic requirements, and "
                    "retrieval-anchor ideas"
                )
            ),
        }
        try:
            response = self._create(
                system=system,
                user_payload=user_payload,
                prompt_version=self.ledger_prompt_version,
                schema_name=(
                    "planner_static_coverage_ledger_v4"
                    if final_contract
                    else (
                        "planner_static_coverage_ledger_v3"
                        if minimal
                        else "planner_static_coverage_ledger_v1"
                    )
                ),
                max_tokens=self.max_tokens,
                schema=(
                    MINIMAL_COVERAGE_LEDGER_JSON_SCHEMA
                    if minimal
                    else COVERAGE_LEDGER_JSON_SCHEMA
                ),
            )
            if not minimal:
                return response
            return PlannerResponse(
                payload=_expand_minimal_coverage_payload(response.payload),
                model=response.model,
                prompt_version=response.prompt_version,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cost=response.cost,
                latency_seconds=response.latency_seconds,
                finish_reason=response.finish_reason,
            )
        except PlannerCallError as first_error:
            if not (
                self.ledger_prompt_version
                in {"planner-static-ledger-v2", "planner-static-ledger-v4"}
                and first_error.detail == "invalid_json"
            ):
                raise
            repair_system = (
                system
                + " The previous response had an invalid JSON envelope. Begin the "
                "response with {, emit exactly one schema-valid JSON object, and "
                "include no preamble, disclaimer, markdown fence, or trailing text."
            )
            try:
                repaired = self._create(
                    system=repair_system,
                    user_payload={
                        **user_payload,
                        "repair_reason": "invalid_json_envelope",
                    },
                    prompt_version=self.ledger_prompt_version,
                    schema_name=(
                        "planner_static_coverage_ledger_v4_repair"
                        if final_contract
                        else "planner_static_coverage_ledger_v2_repair"
                    ),
                    max_tokens=self.max_tokens,
                    schema=COVERAGE_LEDGER_JSON_SCHEMA,
                )
            except PlannerCallError as second_error:
                raise PlannerCallError(
                    second_error.detail,
                    input_tokens=(
                        first_error.input_tokens + second_error.input_tokens
                    ),
                    output_tokens=(
                        first_error.output_tokens + second_error.output_tokens
                    ),
                    cost=first_error.cost + second_error.cost,
                    latency_seconds=(
                        first_error.latency_seconds
                        + second_error.latency_seconds
                    ),
                    finish_reason=second_error.finish_reason,
                    content_shape=second_error.content_shape,
                    parse_error_offset=second_error.parse_error_offset,
                ) from second_error
            return PlannerResponse(
                payload=repaired.payload,
                model=repaired.model,
                prompt_version=repaired.prompt_version,
                input_tokens=first_error.input_tokens + repaired.input_tokens,
                output_tokens=first_error.output_tokens + repaired.output_tokens,
                cost=first_error.cost + repaired.cost,
                latency_seconds=(
                    first_error.latency_seconds + repaired.latency_seconds
                ),
                finish_reason=repaired.finish_reason,
            )

    def expand_coverage_ledger(
        self,
        topic: PlanningTopic,
        ledger: CoverageLedger,
        *,
        max_total_tasks: int | None,
    ) -> PlannerResponse:
        if ledger.topic_sha256 != topic.input_sha256:
            raise ValueError("coverage ledger does not match topic")
        limit = (
            "There is no artificial task-count cap. Include every "
            "candidate-expanding retrieval route justified by the topic."
            if max_total_tasks is None
            else f"The local hard cap is {max_total_tasks} tasks."
        )
        compact = (
            self.expansion_prompt_version in COMPACT_EXPANSION_PROMPT_VERSIONS
        )
        final_contract = self.expansion_prompt_version == "planner-static-expand-v3"
        system = (
            "Create the complete final static retrieval plan. Independently reread "
            "the unabridged original topic, then use the coverage ledger as a fallible "
            "aid: recover anything it omitted or grouped incorrectly. Maximize "
            "information-need coverage and marginal candidate yield for downstream "
            "nugget extraction. Preserve overlapping queries when distinct entities, "
            "official sources, measures, jurisdictions, languages, technical terms, "
            "or evidence polarity may shift dense-retrieval candidates. Discourage "
            "only pure paraphrases with unchanged anchors; exact duplicate query "
            "strings are invalid. Map every final atomic requirement to at least one "
            "task. Query-role labels describe a route and are not a checklist. Do not "
            "impose unsupported dates, locations, populations, or example cases, and "
            "do not replace requested outcomes with easier proxies. Every task is an "
            "independent static retrieval call: use an empty depends_on array for "
            "every task. Return only the complete final schema, with no report, "
            "citations, comments, or markdown. "
            + limit
        )
        if final_contract:
            system = (
                "Generate only query sets for the validated requirement ledger. "
                "Reread the complete original title, problem statement, and requester "
                "background. Emit each ledger requirement_ref exactly once with one "
                "or more queries. Begin each requirement with a faithful direct query "
                "that preserves its entities, qualifiers, and requested outcome. Add "
                "another query only when it introduces a plausible candidate-shifting "
                "anchor: a named entity or alias; technical or field-specific "
                "terminology; an official, primary, academic, regulatory, statistical, "
                "or other source family; a requested measure, outcome, statistic, "
                "trial, comparison, or case; a topic-licensed jurisdiction, language, "
                "population, or timeframe; or contrary, null, negative, disputed, "
                "limitation, or safety evidence. Semantic overlap is allowed when the "
                "anchor changes and may retrieve different documents. Avoid pure "
                "paraphrases with unchanged anchors and exact duplicate query strings. "
                "Generate source-language variants only when a named jurisdiction, "
                "entity, or terminology makes them plausibly candidate-expanding; do "
                "not translate every query mechanically. Do not impose unsupported "
                "restrictions or replace requested outcomes with easier proxies. Query "
                "roles are audit metadata, not a quota. There is no query-count quota "
                "or richer-arm cap: include every query with a plausible marginal "
                "candidate-retrieval benefit, then stop. Do not repeat requirement "
                "descriptions, information-need prose, priorities, IDs, dependencies, "
                "or mappings that local code constructs. Return exactly one "
                "schema-valid JSON object with no prose, report, citations, or "
                "markdown."
            )
        elif compact:
            system = (
                "Generate only query sets for the validated coverage ledger. Re-read "
                "the unabridged topic to make each query useful, but do not repeat "
                "requirement descriptions, information-need prose, priorities, task "
                "IDs, dependencies, or mappings that local code can construct. Emit "
                "each ledger requirement_ref exactly once with one or more queries. "
                "Every query must include a query role and a concise anchor dimension "
                "stating the distinct entity, source family, measure, jurisdiction, "
                "language, technical term, or evidence polarity intended to shift "
                "dense-retrieval candidates. Preserve overlapping routes when anchor "
                "dimensions differ; avoid pure paraphrases and exact duplicate query "
                "strings. Do not impose unsupported narrowing or replace requested "
                "outcomes with proxies. Include every candidate-expanding route "
                "justified by the topic; there is no task-count cap. Return only the "
                "compact schema with no prose, report, citations, or markdown."
            )
        return self._create(
            system=system,
            user_payload={
                "topic": topic.as_record(),
                "coverage_ledger": ledger.as_record(),
                "required_output": "complete dependency-free final retrieval plan",
            },
            prompt_version=self.expansion_prompt_version,
            schema_name=(
                "planner_static_expand_v3"
                if final_contract
                else (
                    "planner_static_expand_v2"
                    if compact
                    else "planner_static_expand_v1"
                )
            ),
            max_tokens=self.audit_max_tokens,
            schema=(
                COMPACT_QUERY_EXPANSION_JSON_SCHEMA
                if compact
                else STATIC_PLAN_JSON_SCHEMA
            ),
        )

    def audit(
        self,
        topic: PlanningTopic,
        draft: StaticResearchPlan,
        *,
        max_total_tasks: int | None,
    ) -> PlannerResponse:
        if draft.planning_stage != "draft":
            raise ValueError("planner auditor requires a draft plan")
        system = (
            "Act as an independent recall-oriented coverage auditor before retrieval. "
            "The downstream system extracts and deduplicates nuggets, so preserve "
            "overlapping queries whenever distinct entities, sources, measures, "
            "jurisdictions, terminology, or evidence polarity may expand candidates. "
            "Do not optimize for the smallest plan. Re-read the "
            "unabridged original topic and reconstruct its atomic requirements; never "
            "assume the draft requirement list is complete. Return a complete corrected "
            "replacement plan in the required schema, not comments or a patch. Recover "
            "omitted requirements, repair harmful unsupported temporal/geographic/population "
            "restrictions, restore broad scope when examples displaced it, eliminate "
            "only exact duplicate or unusable queries, and ensure requested measurable outcomes are "
            "not replaced by easier proxies. Preserve useful draft tasks where they "
            "remain valid. Every final requirement must be covered by at least one "
            "distinct task with an accurate query role. Do not inspect or anticipate "
            "retrieval results. Do not write the report, cite documents, or include "
            "markdown. "
            + self._limit_instruction(max_total_tasks)
        )
        return self._create(
            system=system,
            user_payload={
                "topic": topic.as_record(),
                "draft_plan": draft.as_record(),
                "required_output": "complete audited replacement plan",
            },
            prompt_version=self.audit_prompt_version,
            schema_name="planner_static_audit_v4",
            max_tokens=self.audit_max_tokens,
        )


@dataclass(frozen=True, slots=True)
class EvidencePassage:
    """A bounded passage with source/English and exact document lineage."""

    passage_id: str
    topic_id: str
    need_id: str
    task_id: str
    query: str
    document_id: str
    rank: int
    score: float
    source_language: str
    representation: Literal["canonical", "report_text"]
    text: str
    start_char: int
    end_char: int
    content_hash: str
    source_text_hash: str
    retrieval_object_key: str | None = None

    def __post_init__(self) -> None:
        _text(self.passage_id, field_name="passage_id")
        for name in ("topic_id", "need_id", "task_id", "query", "document_id", "source_language", "text"):
            _text(getattr(self, name), field_name=name)
        _positive_int(self.rank, field_name="passage rank")
        if not isinstance(self.score, (int, float)) or not math.isfinite(float(self.score)):
            raise ValueError("passage score must be finite")
        if self.representation not in {"canonical", "report_text"}:
            raise ValueError("invalid passage representation")
        if isinstance(self.start_char, bool) or not isinstance(self.start_char, int) or self.start_char < 0:
            raise ValueError("passage start_char must be non-negative")
        if isinstance(self.end_char, bool) or not isinstance(self.end_char, int) or self.end_char <= self.start_char:
            raise ValueError("passage end_char must follow start_char")
        if self.end_char - self.start_char != len(self.text):
            raise ValueError("passage offsets must cover exact text")
        for name in ("content_hash", "source_text_hash"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64 or set(value) - _HEX64:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if hashlib.sha256(self.text.encode("utf-8")).hexdigest() != self.content_hash:
            raise ValueError("passage content_hash does not match text")

    def as_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "passage_id": self.passage_id,
            "topic_id": self.topic_id,
            "information_need_id": self.need_id,
            "task_id": self.task_id,
            "query": self.query,
            "document_id": self.document_id,
            "rank": self.rank,
            "score": self.score,
            "source_language": self.source_language,
            "representation": self.representation,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "content_hash": self.content_hash,
            "source_text_hash": self.source_text_hash,
        }
        if self.retrieval_object_key is not None:
            record["retrieval_object_key"] = self.retrieval_object_key
        return record


class StaticPassageReader(Protocol):
    def read(
        self,
        retrieval: RetrievalResult,
        *,
        topic_id: str,
        need_id: str,
        task_id: str,
        query: str,
        max_passages: int,
    ) -> Sequence[EvidencePassage]: ...


def _retrieval_object_key(result: RetrievalResult) -> str | None:
    reference = result.retrieval_artifact
    if isinstance(reference, RetrievalArtifactReference):
        return reference.object_key
    return None


class TopRankedSentenceReader:
    """Small deterministic reader that emits bounded first-sentence evidence."""

    def __init__(
        self,
        *,
        representation: Literal["canonical", "report_text"] = "canonical",
        max_passage_chars: int = 1500,
    ) -> None:
        if representation not in {"canonical", "report_text"}:
            raise ValueError("representation must be canonical or report_text")
        self.representation = representation
        self.max_passage_chars = _positive_int(
            max_passage_chars, field_name="max_passage_chars"
        )

    def read(
        self,
        retrieval: RetrievalResult,
        *,
        topic_id: str,
        need_id: str,
        task_id: str,
        query: str,
        max_passages: int,
    ) -> Sequence[EvidencePassage]:
        _positive_int(max_passages, field_name="max_passages")
        if not isinstance(retrieval, RetrievalResult):
            raise TypeError("retrieval must be a RetrievalResult")
        object_key = _retrieval_object_key(retrieval)
        passages: list[EvidencePassage] = []
        for document in retrieval.documents:
            if len(passages) >= max_passages:
                break
            selected = document.text if self.representation == "canonical" else document.report_text
            if selected is None or not selected:
                continue
            match = _SENTENCE_END.match(selected)
            candidate = match.group(0) if match is not None else selected
            candidate = candidate.strip()
            if not candidate:
                continue
            start = selected.find(candidate)
            if start < 0:
                continue
            if len(candidate) > self.max_passage_chars:
                candidate = candidate[: self.max_passage_chars].rstrip()
            end = start + len(candidate)
            source_language = document.language or "unknown"
            content_hash = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
            source_hash = hashlib.sha256((document.text or "").encode("utf-8")).hexdigest()
            passage_id = _stable_id(
                "passage",
                {
                    "topic_id": topic_id,
                    "need_id": need_id,
                    "task_id": task_id,
                    "document_id": document.document_id,
                    "representation": self.representation,
                    "start_char": start,
                    "end_char": end,
                    "content_hash": content_hash,
                },
            )
            passages.append(
                EvidencePassage(
                    passage_id=passage_id,
                    topic_id=topic_id,
                    need_id=need_id,
                    task_id=task_id,
                    query=query,
                    document_id=document.document_id,
                    rank=document.rank,
                    score=document.score,
                    source_language=source_language,
                    representation=self.representation,
                    text=candidate,
                    start_char=start,
                    end_char=end,
                    content_hash=content_hash,
                    source_text_hash=source_hash,
                    retrieval_object_key=object_key,
                )
            )
        return tuple(passages)


class ChunkedPassageReader:
    """Adapt the repository's exact ``PassageChunker`` contract to evidence."""

    def __init__(
        self,
        chunker: Any,
        *,
        representation: Literal["canonical", "report_text"] = "canonical",
    ) -> None:
        if not callable(getattr(chunker, "chunk", None)):
            raise TypeError("chunker must expose a callable chunk method")
        if representation not in {"canonical", "report_text"}:
            raise ValueError("representation must be canonical or report_text")
        self.chunker = chunker
        self.representation = representation

    def read(
        self,
        retrieval: RetrievalResult,
        *,
        topic_id: str,
        need_id: str,
        task_id: str,
        query: str,
        max_passages: int,
    ) -> Sequence[EvidencePassage]:
        _positive_int(max_passages, field_name="max_passages")
        passages: list[EvidencePassage] = []
        object_key = _retrieval_object_key(retrieval)
        for document in retrieval.documents:
            if len(passages) >= max_passages:
                break
            canonical_text = document.text
            selected_text = (
                canonical_text
                if self.representation == "canonical"
                else document.report_text
            )
            if canonical_text is None or selected_text is None:
                continue
            language = (
                document.language
                if self.representation == "canonical"
                else "eng"
            )
            if not isinstance(language, str) or not language.strip():
                continue
            chunks = self.chunker.chunk(
                document_id=document.document_id,
                canonical_text=canonical_text,
                selected_text=selected_text,
                representation=self.representation,
                language=language,
            )
            for chunk in chunks:
                if len(passages) >= max_passages:
                    break
                if not isinstance(chunk, Passage):
                    raise ValueError("chunker returned a non-Passage value")
                content_hash = chunk.content_hash.hex()
                passage_id = _stable_id(
                    "passage",
                    {
                        "topic_id": topic_id,
                        "need_id": need_id,
                        "task_id": task_id,
                        "document_id": chunk.document_id,
                        "representation": chunk.representation,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        "content_hash": content_hash,
                    },
                )
                passages.append(
                    EvidencePassage(
                        passage_id=passage_id,
                        topic_id=topic_id,
                        need_id=need_id,
                        task_id=task_id,
                        query=query,
                        document_id=document.document_id,
                        rank=document.rank,
                        score=document.score,
                        source_language=document.language or "unknown",
                        representation=self.representation,
                        text=chunk.text,
                        start_char=chunk.start_char,
                        end_char=chunk.end_char,
                        content_hash=content_hash,
                        source_text_hash=hashlib.sha256(
                            canonical_text.encode("utf-8")
                        ).hexdigest(),
                        retrieval_object_key=object_key,
                    )
                )
        return tuple(passages)


class StaticRetriever(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        pipeline: str,
        top_k: int | None,
        max_content_fallbacks: int | None,
        fallback_on_hf_error: bool,
        query_context: Mapping[str, object] | None = None,
    ) -> RetrievalResult: ...

    def get_cached_retrieval(
        self,
        query: str,
        *,
        pipeline: str,
        top_k: int | None,
        max_content_fallbacks: int | None,
        fallback_on_hf_error: bool,
        query_context: Mapping[str, object] | None = None,
    ) -> RetrievalResult | None: ...


@dataclass(frozen=True, slots=True)
class StaticTaskResult:
    allocation_index: int
    need_id: str
    task_id: str
    query: str
    status: Literal["success", "failed"]
    passages: tuple[EvidencePassage, ...] = ()
    failure_kind: str | None = None

    def __post_init__(self) -> None:
        _positive_int(self.allocation_index, field_name="allocation_index")
        _text(self.need_id, field_name="need_id")
        _text(self.task_id, field_name="task_id")
        _text(self.query, field_name="query")
        if self.status not in {"success", "failed"}:
            raise ValueError("invalid static task status")
        if self.status == "success" and not self.passages:
            raise ValueError("successful static task must contain passages")
        if self.status == "failed" and self.passages:
            raise ValueError("failed static task must not contain passages")
        if self.status == "failed" and not self.failure_kind:
            raise ValueError("failed static task must have a failure_kind")

    def as_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "allocation_index": self.allocation_index,
            "information_need_id": self.need_id,
            "task_id": self.task_id,
            "query": self.query,
            "status": self.status,
            "passages": [passage.as_record() for passage in self.passages],
        }
        if self.failure_kind is not None:
            record["failure_kind"] = self.failure_kind
        return record


@dataclass(frozen=True, slots=True)
class StaticTopicEvidence:
    topic_id: str
    plan_sha256: str
    requested_task_count: int
    task_results: tuple[StaticTaskResult, ...]
    status: Literal["success", "partial", "failed"]
    failure_kind: str | None = None

    def __post_init__(self) -> None:
        _text(self.topic_id, field_name="topic_id")
        if len(self.plan_sha256) != 64 or set(self.plan_sha256) - _HEX64:
            raise ValueError("plan_sha256 must be a lowercase SHA-256 digest")
        _positive_int(self.requested_task_count, field_name="requested_task_count")
        allocations = [result.allocation_index for result in self.task_results]
        if allocations != list(range(1, len(allocations) + 1)):
            raise ValueError("task results must be in allocation order")
        successful = sum(result.status == "success" for result in self.task_results)
        expected = (
            "success"
            if successful == len(self.task_results) and self.task_results
            else "partial"
            if successful
            else "failed"
        )
        if self.status != expected and self.failure_kind != "plan_shortfall":
            raise ValueError("static topic status does not match task results")
        if self.status == "failed" and not self.failure_kind:
            raise ValueError("failed static topic must have a failure_kind")

    @property
    def passages(self) -> tuple[EvidencePassage, ...]:
        return tuple(
            passage
            for result in self.task_results
            for passage in result.passages
        )

    def as_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "schema_version": RUN_SCHEMA_VERSION,
            "topic_id": self.topic_id,
            "plan_sha256": self.plan_sha256,
            "requested_task_count": self.requested_task_count,
            "status": self.status,
            "task_results": [result.as_record() for result in self.task_results],
        }
        if self.failure_kind is not None:
            record["failure_kind"] = self.failure_kind
        return record


def evidence_passage_from_record(record: Mapping[str, Any]) -> EvidencePassage:
    if not isinstance(record, Mapping):
        raise ValueError("passage artifact must be an object")
    _expect_keys(
        record,
        {
            "passage_id",
            "topic_id",
            "information_need_id",
            "task_id",
            "query",
            "document_id",
            "rank",
            "score",
            "source_language",
            "representation",
            "text",
            "start_char",
            "end_char",
            "content_hash",
            "source_text_hash",
            "retrieval_object_key",
        },
        label="passage artifact",
    )
    retrieval_object_key = record.get("retrieval_object_key")
    if retrieval_object_key is not None:
        retrieval_object_key = _text(
            retrieval_object_key, field_name="retrieval_object_key"
        )
    return EvidencePassage(
        passage_id=_text(record.get("passage_id"), field_name="passage_id"),
        topic_id=_text(record.get("topic_id"), field_name="topic_id"),
        need_id=_text(record.get("information_need_id"), field_name="need_id"),
        task_id=_text(record.get("task_id"), field_name="task_id"),
        query=_text(record.get("query"), field_name="query"),
        document_id=_text(record.get("document_id"), field_name="document_id"),
        rank=_positive_int(record.get("rank"), field_name="passage rank"),
        score=record.get("score"),
        source_language=_text(
            record.get("source_language"), field_name="source_language"
        ),
        representation=record.get("representation"),
        text=_text(record.get("text"), field_name="text"),
        start_char=record.get("start_char"),
        end_char=record.get("end_char"),
        content_hash=_text(record.get("content_hash"), field_name="content_hash"),
        source_text_hash=_text(
            record.get("source_text_hash"), field_name="source_text_hash"
        ),
        retrieval_object_key=retrieval_object_key,
    )


def static_topic_evidence_from_record(
    record: Mapping[str, Any], *, plan: StaticResearchPlan | None = None
) -> StaticTopicEvidence:
    if not isinstance(record, Mapping):
        raise ValueError("static topic evidence must be an object")
    _expect_keys(
        record,
        {
            "schema_version",
            "topic_id",
            "plan_sha256",
            "requested_task_count",
            "status",
            "task_results",
            "failure_kind",
        },
        label="static topic evidence",
    )
    if record.get("schema_version") != RUN_SCHEMA_VERSION:
        raise ValueError("unsupported static run schema version")
    raw_results = record.get("task_results")
    if not isinstance(raw_results, Sequence) or isinstance(raw_results, (str, bytes)):
        raise ValueError("task_results must be a list")
    task_results: list[StaticTaskResult] = []
    plan_tasks = {} if plan is None else {task.task_id: task for task in plan.tasks}
    for raw_result in raw_results:
        if not isinstance(raw_result, Mapping):
            raise ValueError("task result must be an object")
        _expect_keys(
            raw_result,
            {
                "allocation_index",
                "information_need_id",
                "task_id",
                "query",
                "status",
                "passages",
                "failure_kind",
            },
            label="static task result",
        )
        raw_passages = raw_result.get("passages")
        if not isinstance(raw_passages, Sequence) or isinstance(raw_passages, (str, bytes)):
            raise ValueError("task result passages must be a list")
        passages = tuple(evidence_passage_from_record(value) for value in raw_passages)
        task_id = _text(raw_result.get("task_id"), field_name="task_id")
        query = _text(raw_result.get("query"), field_name="query")
        need_id = _text(
            raw_result.get("information_need_id"), field_name="information_need_id"
        )
        if plan is not None:
            task = plan_tasks.get(task_id)
            if task is None or task.query != query or task.need_id != need_id:
                raise ValueError("task result is not anchored to the supplied plan")
            if any(
                passage.topic_id != plan.topic_id
                or passage.task_id != task_id
                or passage.need_id != need_id
                or passage.query != query
                for passage in passages
            ):
                raise ValueError("passage is not anchored to its task")
        task_results.append(
            StaticTaskResult(
                allocation_index=_positive_int(
                    raw_result.get("allocation_index"), field_name="allocation_index"
                ),
                need_id=need_id,
                task_id=task_id,
                query=query,
                status=raw_result.get("status"),
                passages=passages,
                failure_kind=raw_result.get("failure_kind"),
            )
        )
    result = StaticTopicEvidence(
        topic_id=_text(record.get("topic_id"), field_name="topic_id"),
        plan_sha256=_text(record.get("plan_sha256"), field_name="plan_sha256"),
        requested_task_count=_positive_int(
            record.get("requested_task_count"), field_name="requested_task_count"
        ),
        task_results=tuple(task_results),
        status=record.get("status"),
        failure_kind=record.get("failure_kind"),
    )
    if plan is not None and (
        result.topic_id != plan.topic_id or result.plan_sha256 != plan.plan_sha256
    ):
        raise ValueError("static topic evidence has mismatched plan identity")
    return result


def execute_static_plan(
    plan: StaticResearchPlan,
    retriever: StaticRetriever,
    passage_reader: StaticPassageReader,
    *,
    requested_task_count: int,
    pipeline: str = "ragtime2",
    top_k: int = DEFAULT_TOP_K,
    max_passages_per_task: int = 8,
) -> StaticTopicEvidence:
    """Execute only precommitted tasks and merge results in allocation order."""

    if plan.planning_stage != "audited":
        raise ValueError("static plan must be audited before retrieval")
    allocation = allocate_tasks(plan, requested_task_count)
    if not allocation.complete:
        return StaticTopicEvidence(
            topic_id=plan.topic_id,
            plan_sha256=plan.plan_sha256,
            requested_task_count=requested_task_count,
            task_results=(),
            status="failed",
            failure_kind="plan_shortfall",
        )
    _positive_int(top_k, field_name="top_k")
    _positive_int(max_passages_per_task, field_name="max_passages_per_task")
    task_results: list[StaticTaskResult] = []
    task_status: dict[str, Literal["success", "failed"]] = {}
    for allocated in allocation.tasks:
        task = allocated.task
        if any(task_status.get(dependency) != "success" for dependency in task.depends_on):
            task_results.append(
                StaticTaskResult(
                    allocation_index=allocated.allocation_index,
                    need_id=allocated.need_id,
                    task_id=task.task_id,
                    query=task.query,
                    status="failed",
                    failure_kind="dependency_failed",
                )
            )
            task_status[task.task_id] = "failed"
            continue
        context = {
            "kind": "planner_static_v2",
            "plan_sha256": plan.plan_sha256,
            "information_need_id": allocated.need_id,
            "task_id": task.task_id,
            "allocation_index": allocated.allocation_index,
        }
        try:
            cached_lookup = getattr(retriever, "get_cached_retrieval", None)
            retrieval = (
                cached_lookup(
                    task.query,
                    pipeline=pipeline,
                    top_k=top_k,
                    max_content_fallbacks=0,
                    fallback_on_hf_error=False,
                    query_context=context,
                )
                if callable(cached_lookup)
                else None
            )
            if retrieval is None:
                retrieval = retriever.retrieve(
                    task.query,
                    pipeline=pipeline,
                    top_k=top_k,
                    max_content_fallbacks=0,
                    fallback_on_hf_error=False,
                    query_context=context,
                )
            if not isinstance(retrieval, RetrievalResult):
                raise ValueError("retriever returned an invalid result")
            if not retrieval.fully_hydrated or retrieval.hydration_failures:
                raise ValueError("retrieval_hydration_failure")
            passages = tuple(
                passage_reader.read(
                    retrieval,
                    topic_id=plan.topic_id,
                    need_id=allocated.need_id,
                    task_id=task.task_id,
                    query=task.query,
                    max_passages=max_passages_per_task,
                )
            )
            if not passages:
                raise ValueError("no_evidence")
            if any(not isinstance(passage, EvidencePassage) for passage in passages):
                raise ValueError("invalid_passage")
        except ValueError as error:
            kind = str(error) if str(error) in {"retrieval_hydration_failure", "no_evidence", "invalid_passage"} else "retrieval_failure"
            task_results.append(
                StaticTaskResult(
                    allocation_index=allocated.allocation_index,
                    need_id=allocated.need_id,
                    task_id=task.task_id,
                    query=task.query,
                    status="failed",
                    failure_kind=kind,
                )
            )
            task_status[task.task_id] = "failed"
            continue
        except Exception:
            task_results.append(
                StaticTaskResult(
                    allocation_index=allocated.allocation_index,
                    need_id=allocated.need_id,
                    task_id=task.task_id,
                    query=task.query,
                    status="failed",
                    failure_kind="retrieval_failure",
                )
            )
            task_status[task.task_id] = "failed"
            continue
        task_results.append(
            StaticTaskResult(
                allocation_index=allocated.allocation_index,
                need_id=allocated.need_id,
                task_id=task.task_id,
                query=task.query,
                status="success",
                passages=passages,
            )
        )
        task_status[task.task_id] = "success"
    task_results.sort(key=lambda result: result.allocation_index)
    successful = sum(result.status == "success" for result in task_results)
    status: Literal["success", "partial", "failed"] = (
        "success" if successful == len(task_results) else "partial" if successful else "failed"
    )
    return StaticTopicEvidence(
        topic_id=plan.topic_id,
        plan_sha256=plan.plan_sha256,
        requested_task_count=requested_task_count,
        task_results=tuple(task_results),
        status=status,
        failure_kind=(
            None
            if successful == len(task_results)
            else "task_failures"
            if successful
            else "all_tasks_failed"
        ),
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_planner_dry_run(
    topics: Sequence[PlanningTopic],
    planner: StaticPlanner | AsymmetricStaticPlanner,
    *,
    max_planner_calls: int = DEFAULT_MAX_PLANNER_CALLS,
    output_path: Path | None = None,
) -> tuple[dict[str, Any], ...]:
    """Run a bounded planner-only sample and write sanitized metrics if asked."""

    _positive_int(max_planner_calls, field_name="max_planner_calls")
    if len(topics) > max_planner_calls:
        raise ValueError("planner-only run exceeds max_planner_calls")
    records: list[dict[str, Any]] = []
    for topic in topics:
        outcome = planner.plan_topic(topic)
        record: dict[str, Any] = {
            "schema_version": RUN_SCHEMA_VERSION,
            "topic_id": topic.topic_id,
            "status": outcome.status,
            "cache_hit": outcome.cache_hit,
            "draft_cache_hit": outcome.draft_cache_hit,
            "audit_cache_hit": outcome.audit_cache_hit,
            "model": outcome.model,
            "prompt_version": outcome.prompt_version,
            "input_tokens": outcome.input_tokens,
            "output_tokens": outcome.output_tokens,
            "cost": outcome.cost,
            "latency_seconds": outcome.latency_seconds,
            "planner_path": outcome.planner_path,
            "first_stage_input_tokens": outcome.first_stage_input_tokens,
            "first_stage_output_tokens": outcome.first_stage_output_tokens,
            "second_stage_input_tokens": outcome.second_stage_input_tokens,
            "second_stage_output_tokens": outcome.second_stage_output_tokens,
            "second_stage_finish_reason": outcome.second_stage_finish_reason,
        }
        if outcome.plan is not None:
            record.update(
                {
                    "plan_sha256": outcome.plan.plan_sha256,
                    "planning_stage": outcome.plan.planning_stage,
                    "draft_plan_sha256": outcome.plan.draft_plan_sha256,
                    "requirement_count": len(outcome.plan.requirements),
                    "information_need_count": len(outcome.plan.information_needs),
                    "raw_task_count": outcome.plan.raw_task_count,
                    "admitted_task_count": len(outcome.plan.tasks),
                    "truncated_task_count": outcome.plan.truncated_task_count,
                    "task_counts_by_need": {
                        need.need_id: len(need.task_ids)
                        for need in outcome.plan.information_needs
                    },
                    "plan": outcome.plan.as_record(),
                }
            )
        if outcome.failure_kind is not None:
            record["failure_kind"] = outcome.failure_kind
        if outcome.failure_stage is not None:
            record["failure_stage"] = outcome.failure_stage
        if outcome.failure_detail is not None:
            record["failure_detail"] = outcome.failure_detail
        if outcome.validation_error is not None:
            record["validation_error"] = outcome.validation_error
        if outcome.finish_reason is not None:
            record["finish_reason"] = outcome.finish_reason
        if outcome.content_shape is not None:
            record["content_shape"] = outcome.content_shape
        if outcome.parse_error_offset is not None:
            record["parse_error_offset"] = outcome.parse_error_offset
        if outcome.fallback_used:
            record["fallback_used"] = True
            record["fallback_reason"] = outcome.fallback_reason
        records.append(record)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(f".{output_path.name}.tmp")
        temporary.write_text(
            "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )
        temporary.replace(output_path)
    return tuple(records)


def load_baseline_query_budgets(path: Path) -> dict[str, int]:
    """Read per-topic ``B_i`` from a completed Baseline C topic artifact."""

    budgets: dict[str, int] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"line {line_number}: blank baseline topic record")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_number}: invalid baseline topic JSON") from error
            if not isinstance(record, Mapping):
                raise ValueError(f"line {line_number}: baseline topic must be an object")
            topic_id = _text(record.get("topic_id"), field_name="topic_id")
            if topic_id in budgets:
                raise ValueError(f"duplicate baseline topic_id {topic_id}")
            if record.get("status") != "success":
                raise ValueError(f"baseline topic {topic_id} is not successful")
            results = record.get("query_results")
            if not isinstance(results, Sequence) or isinstance(results, (str, bytes)) or not results:
                raise ValueError(f"baseline topic {topic_id} has no query results")
            if any(
                not isinstance(result, Mapping) or result.get("status") != "success"
                for result in results
            ):
                raise ValueError(f"baseline topic {topic_id} has a failed query")
            budgets[topic_id] = len(results)
    if not budgets:
        raise ValueError("baseline topic artifact is empty")
    return budgets


def _atomic_jsonl_write(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def run_static_topics(
    topics: Sequence[PlanningTopic],
    planner: StaticPlanner | AsymmetricStaticPlanner,
    retriever: StaticRetriever,
    passage_reader: StaticPassageReader,
    *,
    budget_mode: Literal["matched_baseline_c_v1", "frontier_v1"],
    baseline_budgets: Mapping[str, int] | None = None,
    max_planner_calls: int = DEFAULT_MAX_PLANNER_CALLS,
    top_k: int = DEFAULT_TOP_K,
    max_passages_per_task: int = 8,
    output_path: Path | None = None,
) -> tuple[dict[str, Any], ...]:
    """Run topics serially and publish only sanitized, typed-derived records."""

    if budget_mode not in {"matched_baseline_c_v1", "frontier_v1"}:
        raise ValueError("unsupported static budget mode")
    _positive_int(max_planner_calls, field_name="max_planner_calls")
    if len(topics) > max_planner_calls:
        raise ValueError("static run exceeds max_planner_calls")
    if budget_mode == "matched_baseline_c_v1" and baseline_budgets is None:
        raise ValueError("matched baseline mode requires baseline_budgets")
    records: list[dict[str, Any]] = []
    seen_topics: set[str] = set()
    for topic in topics:
        if topic.topic_id in seen_topics:
            raise ValueError(f"duplicate topic_id {topic.topic_id}")
        seen_topics.add(topic.topic_id)
        outcome = planner.plan_topic(topic)
        record: dict[str, Any] = {
            "schema_version": RUN_SCHEMA_VERSION,
            "topic_id": topic.topic_id,
            "budget_mode": budget_mode,
            "planner_status": outcome.status,
            "planner_cache_hit": outcome.cache_hit,
            "planner_draft_cache_hit": outcome.draft_cache_hit,
            "planner_audit_cache_hit": outcome.audit_cache_hit,
            "planner_path": outcome.planner_path,
            "planner_model": outcome.model,
            "planner_prompt_version": outcome.prompt_version,
            "planner_input_tokens": outcome.input_tokens,
            "planner_output_tokens": outcome.output_tokens,
            "planner_cost": outcome.cost,
            "planner_latency_seconds": outcome.latency_seconds,
        }
        if outcome.fallback_used:
            record["planner_fallback_used"] = True
            record["planner_fallback_reason"] = outcome.fallback_reason
        if outcome.plan is None:
            record["status"] = "failed"
            record["failure_kind"] = outcome.failure_kind or "planner_failure"
            if outcome.failure_stage is not None:
                record["failure_stage"] = outcome.failure_stage
            if outcome.failure_detail is not None:
                record["failure_detail"] = outcome.failure_detail
            if outcome.validation_error is not None:
                record["validation_error"] = outcome.validation_error
            if outcome.finish_reason is not None:
                record["finish_reason"] = outcome.finish_reason
            records.append(record)
            continue
        plan = outcome.plan
        requested_count = (
            baseline_budgets[topic.topic_id]
            if budget_mode == "matched_baseline_c_v1"
            and baseline_budgets is not None
            and topic.topic_id in baseline_budgets
            else len(plan.tasks)
        )
        if budget_mode == "matched_baseline_c_v1" and (
            baseline_budgets is None or topic.topic_id not in baseline_budgets
        ):
            record["status"] = "failed"
            record["failure_kind"] = "missing_baseline_budget"
            record["plan"] = plan.as_record()
            records.append(record)
            continue
        evidence = execute_static_plan(
            plan,
            retriever,
            passage_reader,
            requested_task_count=requested_count,
            top_k=top_k,
            max_passages_per_task=max_passages_per_task,
        )
        record.update(
            {
                "status": evidence.status,
                "plan": plan.as_record(),
                "requested_task_count": requested_count,
                "evidence": evidence.as_record(),
            }
        )
        if evidence.failure_kind is not None:
            record["failure_kind"] = evidence.failure_kind
        records.append(record)
    if output_path is not None:
        _atomic_jsonl_write(output_path, records)
    return tuple(records)


def representative_topics(topics: Sequence[PlanningTopic], count: int) -> tuple[PlanningTopic, ...]:
    _positive_int(count, field_name="count")
    if not topics:
        raise ValueError("topics must not be empty")
    if count >= len(topics):
        return tuple(topics)
    if count == 1:
        return (topics[0],)
    indices = {round(index * (len(topics) - 1) / (count - 1)) for index in range(count)}
    return tuple(topics[index] for index in sorted(indices))


def select_topics_by_id(
    topics: Sequence[PlanningTopic], topic_ids: Sequence[str]
) -> tuple[PlanningTopic, ...]:
    """Select an explicit ordered topic set and fail before provider work."""

    requested = tuple(_text(value, field_name="topic_id") for value in topic_ids)
    if not requested:
        raise ValueError("at least one explicit topic_id is required")
    if len(set(requested)) != len(requested):
        raise ValueError("explicit topic_id values must be unique")
    by_id = {topic.topic_id: topic for topic in topics}
    unknown = [topic_id for topic_id in requested if topic_id not in by_id]
    if unknown:
        raise ValueError(f"unknown topic_id: {unknown[0]}")
    return tuple(by_id[topic_id] for topic_id in requested)


def load_planning_topics(path: Path) -> tuple[PlanningTopic, ...]:
    topics: list[PlanningTopic] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_number}: invalid topic JSON") from error
            if not isinstance(payload, Mapping):
                raise ValueError(f"line {line_number}: topic must be an object")
            topic_id = _text(payload.get("topic_id"), field_name="topic_id")
            if topic_id in seen:
                raise ValueError(f"duplicate topic_id {topic_id}")
            seen.add(topic_id)
            topics.append(
                PlanningTopic(
                    topic_id=topic_id,
                    title=_text(payload.get("title", topic_id), field_name="title"),
                    problem_statement=_text(
                        payload.get("problem_statement"), field_name="problem_statement"
                    ),
                    background=_text(payload.get("background"), field_name="background"),
                )
            )
    if not topics:
        raise ValueError("topic file is empty")
    return tuple(topics)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a bounded two-pass static-plan dry run.")
    parser.add_argument("--topics", type=Path, default=DEFAULT_TOPIC_PATH)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument(
        "--topic-id",
        action="append",
        dest="topic_ids",
        default=None,
        help="select an exact topic ID; repeat for an ordered set",
    )
    parser.add_argument("--max-planner-calls", type=int, default=DEFAULT_MAX_PLANNER_CALLS)
    parser.add_argument("--max-total-tasks", type=int, default=DEFAULT_MAX_TOTAL_TASKS)
    parser.add_argument("--model", default=DEFAULT_PLANNER_MODEL)
    parser.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION)
    parser.add_argument("--audit-prompt-version", default=DEFAULT_AUDIT_PROMPT_VERSION)
    parser.add_argument(
        "--planner-path",
        choices=("full_plan_v3", "coverage_ledger_v1"),
        default="full_plan_v3",
    )
    parser.add_argument(
        "--ledger-prompt-version", default=DEFAULT_LEDGER_PROMPT_VERSION
    )
    parser.add_argument(
        "--expansion-prompt-version", default=DEFAULT_EXPANSION_PROMPT_VERSION
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    load_dotenv()
    base_url, api_key = provider_settings()
    from .artifact_store import ArtifactStore

    store = ArtifactStore.from_env()
    cache = ResponseCachePlanArtifactCache(store, base_url=base_url)
    ledger_cache = ResponseCacheCoverageLedgerArtifactCache(
        store, base_url=base_url
    )
    expansion_cache = ResponseCacheCoverageExpansionArtifactCache(
        store, base_url=base_url
    )
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0, max_retries=0)
    model = OpenAIStaticPlannerModel(
        client,
        model=arguments.model,
        prompt_version=arguments.prompt_version,
        audit_prompt_version=arguments.audit_prompt_version,
        ledger_prompt_version=arguments.ledger_prompt_version,
        expansion_prompt_version=arguments.expansion_prompt_version,
        timeout_seconds=60.0,
    )
    planner: StaticPlanner | AsymmetricStaticPlanner
    if arguments.planner_path == "coverage_ledger_v1":
        planner = AsymmetricStaticPlanner(
            model,
            ledger_cache=ledger_cache,
            expansion_cache=expansion_cache,
            max_total_tasks=arguments.max_total_tasks,
            provider_base_url=base_url,
            model_name=arguments.model,
            ledger_prompt_version=arguments.ledger_prompt_version,
            expansion_prompt_version=arguments.expansion_prompt_version,
        )
    else:
        planner = StaticPlanner(
            model,
            auditor=model,
            cache=cache,
            max_total_tasks=arguments.max_total_tasks,
            provider_base_url=base_url,
            model_name=arguments.model,
            prompt_version=arguments.prompt_version,
            auditor_model_name=arguments.model,
            audit_prompt_version=arguments.audit_prompt_version,
        )
    all_topics = load_planning_topics(arguments.topics)
    topics = (
        select_topics_by_id(all_topics, arguments.topic_ids)
        if arguments.topic_ids is not None
        else representative_topics(all_topics, arguments.count)
    )
    if len(topics) > arguments.max_planner_calls:
        raise ValueError("planner-only run exceeds max-planner-calls")
    output = arguments.output
    if output is None:
        output = (
            default_project_cache_dir()
            / "experiments"
            / "planner-static-v2"
            / f"planner-only-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            / "records.jsonl"
        )
    records = run_planner_dry_run(topics, planner, output_path=output)
    print(json.dumps({"output": str(output), "records": len(records)}, sort_keys=True))
    return 0 if all(record["status"] == "success" for record in records) else 1


__all__ = [
    "AllocatedTask",
    "AsymmetricStaticPlanner",
    "CoverageLedger",
    "CoverageLedgerArtifactCache",
    "CoverageLedgerCacheEntry",
    "CoverageExpansionArtifactCache",
    "CoverageExpansionCacheEntry",
    "CoverageLedgerNeed",
    "CoverageLedgerRequirement",
    "DEFAULT_MAX_TOTAL_TASKS",
    "DEFAULT_MAX_PLANNER_CALLS",
    "DEFAULT_PLANNER_MODEL",
    "DEFAULT_AUDIT_PROMPT_VERSION",
    "EvidencePassage",
    "ChunkedPassageReader",
    "InformationNeed",
    "OpenAIStaticPlannerModel",
    "PlanArtifactCache",
    "PlannerAuditor",
    "PlanTask",
    "PlannerOutcome",
    "PlannerResponse",
    "PlanningTopic",
    "PlanningRequirement",
    "ResponseCachePlanArtifactCache",
    "ResponseCacheCoverageLedgerArtifactCache",
    "ResponseCacheCoverageExpansionArtifactCache",
    "RUN_SCHEMA_VERSION",
    "StaticPassageReader",
    "StaticResearchPlan",
    "StaticRetriever",
    "StaticPlanner",
    "StaticTaskResult",
    "StaticTopicEvidence",
    "TaskAllocation",
    "TopRankedSentenceReader",
    "allocate_tasks",
    "build_parser",
    "execute_static_plan",
    "coverage_ledger_from_record",
    "load_planning_topics",
    "load_baseline_query_budgets",
    "main",
    "parse_static_plan",
    "parse_coverage_ledger",
    "parse_compact_query_expansion",
    "representative_topics",
    "select_topics_by_id",
    "run_planner_dry_run",
    "run_static_topics",
    "static_plan_from_record",
    "static_plan_to_record",
    "static_topic_evidence_from_record",
    "evidence_passage_from_record",
]
