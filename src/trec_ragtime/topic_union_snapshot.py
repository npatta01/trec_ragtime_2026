"""Build immutable topic scoring snapshots from validated richer retrieval records."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import PipelineResult, RetrievalResult, RetrievedDocument, SearchHit
from .planner_static import StaticResearchPlan, static_plan_from_record
from .topic_union_scoring import InformationNeed, TopicRetrievalInput


class TopicSnapshotError(ValueError):
    """Raised when retrieval checkpoint lineage cannot form a safe snapshot."""


@dataclass(frozen=True, slots=True)
class SnapshotCandidate:
    document_id: str
    rank: int
    score: float


@dataclass(frozen=True, slots=True)
class SnapshotRetrieval:
    task_id: str
    query: str
    object_keys: tuple[str, ...]
    candidates: tuple[SnapshotCandidate, ...]

    @property
    def provenance_key(self) -> str:
        if len(self.object_keys) == 1:
            return self.object_keys[0]
        digest = hashlib.sha256("\n".join(self.object_keys).encode()).hexdigest()
        return f"snapshot-union/v1/{digest}/{self.task_id}"


@dataclass(frozen=True, slots=True)
class TopicSnapshot:
    topic_id: str
    plan_sha256: str
    needs: tuple[InformationNeed, ...]
    retrievals: tuple[SnapshotRetrieval, ...]
    input_object_keys: tuple[str, ...]

    @property
    def successful_query_count(self) -> int:
        return len(self.retrievals)


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TopicSnapshotError(f"invalid snapshot input {path}") from error


def _plans(path: Path) -> tuple[StaticResearchPlan, ...]:
    plans: list[StaticResearchPlan] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise TopicSnapshotError("unable to read frozen plans") from error
    for line in lines:
        if not line:
            raise TopicSnapshotError("blank frozen-plan line")
        record = json.loads(line)
        plan = static_plan_from_record(record["plan"])
        if record.get("topic_id") != plan.topic_id:
            raise TopicSnapshotError("frozen plan topic mismatch")
        plans.append(plan)
    if not plans or len({row.topic_id for row in plans}) != len(plans):
        raise TopicSnapshotError("frozen plans must have unique topics")
    return tuple(plans)


def _candidate(document_id: object, rank: object, score: object) -> SnapshotCandidate:
    if not isinstance(document_id, str) or not document_id:
        raise TopicSnapshotError("candidate document ID is invalid")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise TopicSnapshotError("candidate rank is invalid")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise TopicSnapshotError("candidate score is invalid")
    numeric = float(score)
    if not math.isfinite(numeric):
        raise TopicSnapshotError("candidate score is invalid")
    return SnapshotCandidate(document_id, rank, numeric)


def _breadth_retrievals(
    pool: dict[str, Any], plan: StaticResearchPlan
) -> tuple[SnapshotRetrieval, ...]:
    tasks = {row.task_id: row for row in plan.tasks}
    candidates = pool.get("candidates")
    if not isinstance(candidates, list) or pool.get("candidate_count") != len(candidates):
        raise TopicSnapshotError("breadth candidate count is invalid")
    grouped: dict[str, list[SnapshotCandidate]] = {}
    for row in candidates:
        task_id = row.get("task_id")
        if task_id not in tasks:
            raise TopicSnapshotError(f"unknown task {task_id!r} in breadth pool")
        grouped.setdefault(task_id, []).append(
            _candidate(row.get("document_id"), row.get("rank"), row.get("score"))
        )
    object_key = pool.get("retrieval_object_key")
    if not isinstance(object_key, str) or not object_key.startswith("retrieval/v1/"):
        raise TopicSnapshotError("breadth retrieval object key is invalid")
    rows: list[SnapshotRetrieval] = []
    for task in plan.tasks:
        if task.task_id not in grouped:
            continue
        if pool.get("query") != task.query:
            raise TopicSnapshotError("breadth query does not match frozen task")
        rows.append(
            SnapshotRetrieval(task.task_id, task.query, (object_key,), tuple(grouped[task.task_id]))
        )
    return tuple(rows)


def _fused_retrievals(
    pool: dict[str, Any], plan: StaticResearchPlan
) -> tuple[SnapshotRetrieval, ...]:
    tasks = {row.task_id: row for row in plan.tasks}
    object_keys = pool.get("source_retrieval_object_keys")
    if (
        not isinstance(object_keys, list)
        or not object_keys
        or any(not isinstance(key, str) or not key.startswith("retrieval/v1/") for key in object_keys)
    ):
        raise TopicSnapshotError("fused retrieval object keys are invalid")
    candidates = pool.get("candidates")
    if not isinstance(candidates, list) or pool.get("candidate_count") != len(candidates):
        raise TopicSnapshotError("fused candidate count is invalid")
    grouped: dict[str, list[SnapshotCandidate]] = {}
    for row in candidates:
        task_ids = row.get("task_ids")
        if not isinstance(task_ids, list) or not task_ids:
            raise TopicSnapshotError("fused candidate task IDs are invalid")
        candidate = _candidate(row.get("document_id"), row.get("best_rank"), row.get("max_score"))
        for task_id in task_ids:
            if task_id not in tasks:
                raise TopicSnapshotError(f"unknown task {task_id!r} in fused pool")
            grouped.setdefault(task_id, []).append(candidate)
    return tuple(
        SnapshotRetrieval(
            task.task_id,
            task.query,
            tuple(sorted(set(object_keys))),
            tuple(grouped[task.task_id]),
        )
        for task in plan.tasks
        if task.task_id in grouped
    )


def _checkpoint_retrieval(
    record: dict[str, Any], plan: StaticResearchPlan
) -> SnapshotRetrieval | None:
    if record.get("status") != "success":
        return None
    if record.get("schema_version") != "planner-static-full-richer-query-v1":
        raise TopicSnapshotError("unsupported full-run checkpoint schema")
    tasks = {row.task_id: row for row in plan.tasks}
    task_id = record.get("task_id")
    task = tasks.get(task_id)
    if task is None:
        raise TopicSnapshotError(f"unknown task {task_id!r} in full-run checkpoint")
    if record.get("query") != task.query:
        raise TopicSnapshotError("checkpoint query does not match frozen task")
    object_key = record.get("retrieval_object_key")
    if not isinstance(object_key, str) or not object_key.startswith("retrieval/v1/"):
        raise TopicSnapshotError("checkpoint retrieval object key is invalid")
    candidates = record.get("candidates")
    if not isinstance(candidates, list) or record.get("candidate_count") != len(candidates):
        raise TopicSnapshotError("checkpoint candidate count is invalid")
    return SnapshotRetrieval(
        task.task_id,
        task.query,
        (object_key,),
        tuple(
            _candidate(row.get("document_id"), row.get("rank"), row.get("score"))
            for row in candidates
        ),
    )


def build_snapshot(
    plans_path: Path, coverage_index: Path, checkpoint_dir: Path
) -> tuple[TopicSnapshot, ...]:
    """Capture all validated successful richer inputs visible at one instant."""

    plans = _plans(Path(plans_path))
    by_topic = {row.topic_id: row for row in plans}
    coverage: dict[str, dict[str, Any]] = {}
    for line in Path(coverage_index).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        topic_id = record.get("topic_id")
        if topic_id not in by_topic or topic_id in coverage:
            raise TopicSnapshotError("coverage index topic is invalid or duplicated")
        coverage[topic_id] = record

    snapshots: list[TopicSnapshot] = []
    for plan in plans:
        row = coverage.get(plan.topic_id)
        if row is None:
            raise TopicSnapshotError(f"coverage index lacks topic {plan.topic_id}")
        retrievals: dict[str, SnapshotRetrieval] = {}
        sources = row.get("supplement_sources")
        if not isinstance(sources, list) or not sources:
            raise TopicSnapshotError("coverage supplement sources are invalid")
        for source in sources:
            if not isinstance(source, dict) or not isinstance(source.get("path"), str):
                raise TopicSnapshotError("coverage source path is invalid")
            pool = _json(Path(source["path"]))
            if pool.get("topic_id") != plan.topic_id or pool.get("plan_sha256") != plan.plan_sha256:
                raise TopicSnapshotError("coverage pool lineage is invalid")
            schema = pool.get("schema_version")
            if schema == "planner-static-breadth-pool-v1":
                pool_rows = _breadth_retrievals(pool, plan)
            elif schema == "planner-static-candidate-pool-v1":
                pool_rows = _fused_retrievals(pool, plan)
            else:
                raise TopicSnapshotError("unsupported coverage pool schema")
            for item in pool_rows:
                if item.task_id in retrievals:
                    raise TopicSnapshotError("duplicate successful task input")
                retrievals[item.task_id] = item

        topic_checkpoints = Path(checkpoint_dir) / plan.topic_id
        if topic_checkpoints.exists():
            for path in sorted(topic_checkpoints.glob("*.json")):
                record = _json(path)
                if record.get("topic_id") != plan.topic_id or record.get("plan_sha256") != plan.plan_sha256:
                    raise TopicSnapshotError("checkpoint lineage is invalid")
                item = _checkpoint_retrieval(record, plan)
                if item is None:
                    continue
                if item.task_id in retrievals:
                    raise TopicSnapshotError("duplicate successful task input")
                retrievals[item.task_id] = item

        ordered = tuple(item for task in plan.tasks if (item := retrievals.get(task.task_id)) is not None)
        if not ordered:
            raise TopicSnapshotError(f"topic {plan.topic_id} has no successful retrieval")
        object_keys = tuple(sorted({key for item in ordered for key in item.object_keys}))
        snapshots.append(
            TopicSnapshot(
                topic_id=plan.topic_id,
                plan_sha256=plan.plan_sha256,
                needs=tuple(InformationNeed(row.need_id, row.objective) for row in plan.information_needs),
                retrievals=ordered,
                input_object_keys=object_keys,
            )
        )
    return tuple(snapshots)


def hydrate_snapshot_topic(snapshot: TopicSnapshot, corpus: object) -> tuple[TopicRetrievalInput, ...]:
    """Hydrate exact snapshot IDs from the local corpus without network retrieval."""

    enrich = getattr(corpus, "enrich_documents", None)
    if not callable(enrich):
        raise TopicSnapshotError("corpus must expose enrich_documents")
    ids = [candidate.document_id for row in snapshot.retrievals for candidate in row.candidates]
    hydration = enrich(ids)
    if hydration.failures or hydration.unresolved_ids or hydration.missing_translation_ids:
        raise TopicSnapshotError("snapshot hydration is incomplete")
    documents = {row.document_id: row for row in hydration.documents}
    inputs: list[TopicRetrievalInput] = []
    for row in snapshot.retrievals:
        hydrated: list[RetrievedDocument] = []
        hits: list[SearchHit] = []
        for candidate in row.candidates:
            source = documents[candidate.document_id]
            hits.append(SearchHit(candidate.rank, candidate.document_id, candidate.score))
            hydrated.append(
                RetrievedDocument(
                    rank=candidate.rank,
                    document_id=candidate.document_id,
                    score=candidate.score,
                    text=source.text,
                    url=source.url,
                    date=source.date,
                    language=source.language,
                    content_source=source.source,
                    content_local_cache_hit=source.local_cache_hit,
                    english_text=source.english_text,
                )
            )
        result = RetrievalResult(
            search_result=PipelineResult(
                query=row.query,
                pipeline="ragtime2",
                hits=tuple(hits),
                processed=True,
                local_cache_hit=True,
                remote_cached=None,
                server_timestamp=None,
                cache_stored_at=None,
            ),
            documents=tuple(hydrated),
        )
        inputs.append(
            TopicRetrievalInput(
                snapshot.topic_id, row.task_id, row.query, row.provenance_key, result
            )
        )
    return tuple(inputs)


__all__ = [
    "SnapshotCandidate",
    "SnapshotRetrieval",
    "TopicSnapshot",
    "TopicSnapshotError",
    "build_snapshot",
    "hydrate_snapshot_topic",
]
