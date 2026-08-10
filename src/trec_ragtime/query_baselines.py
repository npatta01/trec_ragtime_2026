"""Resumable organizer retrieval for the fixed query-plan baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any, Protocol

from .client import RagtimeError
from .huggingface import HuggingFaceError
from .local_corpus import LocalCorpusError
from .models import RetrievalResult, SearchHit
from .paths import default_project_cache_dir
from .retrieval_cache import RetrievalArtifactReference


PIPELINE = "ragtime2"
DEFAULT_TOP_K = 100
DEFAULT_RRF_K = 60
EXPECTED_TOPIC_IDS = tuple(str(topic_id) for topic_id in range(2000, 2103))
EXPECTED_QUERY_PLAN_SHA256 = (
    "83dacea6961986bcbd9aeb1423e8da321d80f0d185d39a875ef148f8f22185d7"
)
_BASELINE_NAMES = {"b": "baseline-b-single-query", "c": "baseline-c-facet-rrf"}


class ConcurrentRetrievalStopped(RuntimeError):
    """Raised after a concurrent wave records a failure and stops safely."""


@dataclass(frozen=True, slots=True)
class QueryPlan:
    """Literal pre-generated queries for one topic."""

    topic_id: str
    plan_status: str
    single_query: str
    facet_queries: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class QueryTask:
    """One organizer request waiting for a dedicated retriever slot."""

    ordinal: int
    topic_id: str
    query_index: int
    query: str
    attempt: int
    query_context: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _QueryTaskResult:
    """Checkpoint data plus a fresh retrieval for same-process assembly."""

    record: dict[str, Any]
    retrieval: RetrievalResult | None = None


class RetrieverLike(Protocol):
    """Minimal high-level retrieval boundary used by the runner."""

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

    @property
    def retrieval_corpus_identity(self) -> Mapping[str, Any]: ...

    @property
    def corpus_identity(self) -> Mapping[str, Any]: ...


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _non_blank(value: Any, *, field: str, line_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"line {line_number}: {field} must be a non-blank string")
    return value


def load_query_plans(path: Path) -> tuple[QueryPlan, ...]:
    """Load ordered query plans without rewriting any query text."""

    plan_path = Path(path)
    plans: list[QueryPlan] = []
    topic_ids: set[str] = set()
    try:
        handle = plan_path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"unable to read query plans {plan_path}: {exc}") from None
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"line {line_number}: blank line is not allowed")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"line {line_number}: invalid JSON ({exc.msg})"
                ) from None
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_number}: plan must be a JSON object")
            topic_id = _non_blank(
                payload.get("topic_id"), field="topic_id", line_number=line_number
            )
            if topic_id in topic_ids:
                raise ValueError(f"line {line_number}: duplicate topic_id {topic_id!r}")
            topic_ids.add(topic_id)
            plan_status = _non_blank(
                payload.get("plan_status"),
                field="plan_status",
                line_number=line_number,
            )
            single_query = _non_blank(
                payload.get("single_query"),
                field="single_query",
                line_number=line_number,
            )
            raw_facets = payload.get("facet_queries")
            if not isinstance(raw_facets, list) or not raw_facets:
                raise ValueError(
                    f"line {line_number}: facet_queries must be a non-empty list"
                )
            facets = tuple(
                _non_blank(
                    query,
                    field=f"facet_queries[{index}]",
                    line_number=line_number,
                )
                for index, query in enumerate(raw_facets)
            )
            plans.append(
                QueryPlan(
                    topic_id=topic_id,
                    plan_status=plan_status,
                    single_query=single_query,
                    facet_queries=facets,
                )
            )
    if not plans:
        raise ValueError(f"query plan file is empty: {plan_path}")
    return tuple(plans)


def fuse_ranked_hits(
    ranked_lists: Sequence[Sequence[SearchHit]],
    *,
    query_indices: Sequence[int] | None = None,
    rrf_k: int = DEFAULT_RRF_K,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """Fuse successful ranked lists with deterministic reciprocal-rank fusion."""

    if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k < 1:
        raise ValueError("rrf_k must be a positive integer")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if query_indices is None:
        query_indices = tuple(range(1, len(ranked_lists) + 1))
    if len(query_indices) != len(ranked_lists):
        raise ValueError("query_indices must match ranked_lists")
    accumulators: dict[str, dict[str, Any]] = {}
    for query_index, hits in zip(query_indices, ranked_lists, strict=True):
        seen_in_list: set[str] = set()
        for hit in hits:
            if hit.document_id in seen_in_list:
                continue
            seen_in_list.add(hit.document_id)
            entry = accumulators.setdefault(
                hit.document_id,
                {
                    "document_id": hit.document_id,
                    "fused_score": 0.0,
                    "best_rank": hit.rank,
                    "source_query_indices": [],
                    "best_score": hit.score,
                },
            )
            entry["fused_score"] += 1.0 / (rrf_k + hit.rank)
            entry["best_rank"] = min(entry["best_rank"], hit.rank)
            entry["best_score"] = max(entry["best_score"], hit.score)
            entry["source_query_indices"].append(query_index)
    ordered = sorted(
        accumulators.values(),
        key=lambda entry: (-entry["fused_score"], entry["document_id"]),
    )[:top_k]
    single_list = len(ranked_lists) == 1
    return [
        {
            "document_id": entry["document_id"],
            "fused_score": entry["fused_score"],
            "best_rank": entry["best_rank"],
            "source_query_indices": entry["source_query_indices"],
            "source_query_count": len(entry["source_query_indices"]),
            "rank": rank,
            "score": entry["best_score"] if single_list else None,
        }
        for rank, entry in enumerate(ordered, start=1)
    ]


def _read_latest_jsonl(path: Path, *, key_fields: tuple[str, ...]) -> dict[tuple[Any, ...], dict[str, Any]]:
    if not path.exists():
        return {}
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}: line {line_number} is blank")
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}: line {line_number} is not an object")
            latest[tuple(payload[field] for field in key_fields)] = payload
    return latest


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    _atomic_write(
        path,
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in records
        ),
    )


def _queries_for(plan: QueryPlan, baseline: str) -> tuple[str, ...]:
    return (plan.single_query,) if baseline == "b" else plan.facet_queries


def _query_context(
    *, baseline: str, plan_sha256: str, query_index: int
) -> dict[str, object]:
    if baseline == "b":
        construction = "problem_background_single_query_v1"
    elif baseline == "c":
        construction = "problem_background_facets_v1"
    else:  # pragma: no cover - validated by the public runner.
        raise ValueError("baseline must be 'b' or 'c'")
    return {
        "kind": f"baseline_{baseline}",
        "query_construction": construction,
        "query_plan_sha256": plan_sha256,
        "query_index": query_index,
    }


def _fresh_reference_record(result: RetrievalResult) -> dict[str, object] | None:
    """Serialize optional provenance from a typed fresh retrieval."""

    if not result.fully_hydrated or result.hydration_failures:
        raise ValueError(
            "successful baseline retrieval is not completely hydrated"
        )
    reference = result.retrieval_artifact
    if reference is None:
        return None
    if not isinstance(reference, RetrievalArtifactReference):
        raise ValueError(
            "successful baseline retrieval has an invalid RetrievalArtifactReference"
        )
    return reference.as_record()


def _checkpoint_has_reference(record: Mapping[str, Any]) -> bool:
    reference = record.get("retrieval_artifact")
    if not isinstance(reference, Mapping):
        return False
    return (
        isinstance(reference.get("object_key"), str)
        and isinstance(reference.get("request_digest"), str)
        and isinstance(reference.get("retriever_fingerprint"), str)
        and isinstance(reference.get("stored_at"), (int, float))
        and not isinstance(reference.get("stored_at"), bool)
        and isinstance(reference.get("cache_hit"), bool)
    )


def _ordered_query_records(
    plans: Sequence[QueryPlan],
    baseline: str,
    checkpoints: dict[tuple[Any, ...], dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        checkpoints[(plan.topic_id, query_index)]
        for plan in plans
        for query_index, _query in enumerate(_queries_for(plan, baseline), start=1)
        if (plan.topic_id, query_index) in checkpoints
    ]


def _query_record_from_result(
    *,
    topic_id: str,
    query_index: int,
    query: str,
    result: RetrievalResult,
    started_at: str,
    duration_seconds: float,
    attempt: int,
    top_k: int,
) -> dict[str, Any]:
    if result.query != query or result.pipeline != PIPELINE:
        raise ValueError("retrieval result does not match the query checkpoint")
    search = result.search_result
    hits = search.hits[:top_k]
    record: dict[str, Any] = {
        "topic_id": topic_id,
        "query_index": query_index,
        "query": query,
        "query_chars": len(query),
        "pipeline": PIPELINE,
        "started_at": started_at,
        "status": "success",
        "duration_seconds": round(duration_seconds, 3),
        "attempt": attempt,
        "local_cache_hit": search.local_cache_hit,
        "remote_cached": search.remote_cached,
        "hit_count": len(hits),
        "hits": [hit.as_record() for hit in hits],
    }
    retrieval_artifact = _fresh_reference_record(result)
    if retrieval_artifact is not None:
        record["retrieval_artifact"] = retrieval_artifact
    return record


def _run_query_task(
    task: QueryTask,
    *,
    retriever: RetrieverLike,
    top_k: int,
) -> _QueryTaskResult:
    started_at = _now()
    started = time.perf_counter()
    try:
        result = retriever.retrieve(
            task.query,
            pipeline=PIPELINE,
            top_k=top_k,
            max_content_fallbacks=0,
            fallback_on_hf_error=False,
            query_context=task.query_context,
        )
    except (RagtimeError, HuggingFaceError, LocalCorpusError, ValueError) as exc:
        return _QueryTaskResult(
            record={
                "topic_id": task.topic_id,
                "query_index": task.query_index,
                "query": task.query,
                "query_chars": len(task.query),
                "pipeline": PIPELINE,
                "started_at": started_at,
                "status": "error",
                "duration_seconds": round(time.perf_counter() - started, 3),
                "attempt": task.attempt,
                "error_type": type(exc).__name__,
                "error_detail": str(exc),
                "hits": [],
            }
        )
    try:
        record = _query_record_from_result(
            topic_id=task.topic_id,
            query_index=task.query_index,
            query=task.query,
            result=result,
            started_at=started_at,
            duration_seconds=time.perf_counter() - started,
            attempt=task.attempt,
            top_k=top_k,
        )
        return _QueryTaskResult(record=record, retrieval=result)
    except ValueError as exc:
        return _QueryTaskResult(
            record={
                "topic_id": task.topic_id,
                "query_index": task.query_index,
                "query": task.query,
                "query_chars": len(task.query),
                "pipeline": PIPELINE,
                "started_at": started_at,
                "status": "error",
                "duration_seconds": round(time.perf_counter() - started, 3),
                "attempt": task.attempt,
                "error_type": type(exc).__name__,
                "error_detail": str(exc),
                "hits": [],
            }
        )


def _query_records_for_topic(
    plan: QueryPlan,
    *,
    baseline: str,
    checkpoints: Mapping[tuple[Any, ...], dict[str, Any]],
) -> list[dict[str, Any]] | None:
    records: list[dict[str, Any]] = []
    for query_index, query in enumerate(_queries_for(plan, baseline), start=1):
        record = checkpoints.get((plan.topic_id, query_index))
        if record is None:
            for (stored_topic_id, stored_index), stale in checkpoints.items():
                if (
                    stored_topic_id == plan.topic_id
                    and stale.get("query") == query
                    and stored_index != query_index
                ):
                    raise ValueError(
                        f"checkpoint query_index must equal plan position {query_index}"
                    )
            return None
        if record.get("query") != query or record.get("pipeline") != PIPELINE:
            return None
        stored_index = record.get("query_index")
        if (
            isinstance(stored_index, bool)
            or not isinstance(stored_index, int)
            or stored_index < 1
            or stored_index != query_index
        ):
            raise ValueError(
                f"checkpoint query_index must equal plan position {query_index}"
            )
        records.append(record)
    return records


def _reference_matches(
    expected: Mapping[str, Any], actual: RetrievalArtifactReference
) -> bool:
    """Compare immutable identity provenance, allowing a warm-hit flag change."""

    stored_at = expected.get("stored_at")
    if isinstance(stored_at, bool) or not isinstance(stored_at, (int, float)):
        return False
    return (
        expected.get("object_key") == actual.object_key
        and expected.get("request_digest") == actual.request_digest_hex
        and expected.get("retriever_fingerprint") == actual.retriever_fingerprint_hex
        and float(stored_at) == actual.stored_at
    )


def _load_cached_query_result(
    record: Mapping[str, Any],
    *,
    baseline: str,
    plan_sha256: str,
    retriever: RetrieverLike,
    top_k: int,
    expected_query_index: int,
) -> RetrievalResult:
    if record.get("status") != "success" or not _checkpoint_has_reference(record):
        raise ValueError(
            "successful baseline query checkpoint is missing a retrieval artifact reference"
        )
    query = record.get("query")
    query_index = record.get("query_index")
    if (
        not isinstance(query, str)
        or isinstance(query_index, bool)
        or not isinstance(query_index, int)
        or query_index < 1
        or query_index != expected_query_index
    ):
        raise ValueError("baseline query checkpoint has an invalid query identity")
    cached = retriever.get_cached_retrieval(
        query,
        pipeline=PIPELINE,
        top_k=top_k,
        max_content_fallbacks=0,
        fallback_on_hf_error=False,
        query_context=_query_context(
            baseline=baseline,
            plan_sha256=plan_sha256,
            query_index=query_index,
        ),
    )
    if cached is None:
        raise ValueError(
            f"retrieval artifact is missing for topic={record.get('topic_id')} "
            f"query_index={query_index}"
        )
    reference = cached.retrieval_artifact
    if not isinstance(reference, RetrievalArtifactReference):
        raise ValueError("cache-only retrieval is missing its artifact reference")
    checkpoint_reference = record["retrieval_artifact"]
    assert isinstance(checkpoint_reference, Mapping)
    if not _reference_matches(checkpoint_reference, reference):
        raise ValueError(
            f"retrieval artifact reference mismatch for topic={record.get('topic_id')} "
            f"query_index={query_index}"
        )
    cached_hits = [hit.as_record() for hit in cached.search_result.hits[:top_k]]
    if cached_hits != record.get("hits"):
        raise ValueError("cache-only retrieval hits disagree with the query checkpoint")
    return cached


def _document_signature(document: Any) -> tuple[Any, ...]:
    """Canonical fields used to detect contradictory facet hydration."""

    return (
        document.text,
        document.english_text,
        document.report_text,
        document.language,
        document.content_source,
        document.url,
        document.date,
    )


def _assemble_candidates(
    candidates: list[dict[str, Any]],
    cached_results: Sequence[RetrievalResult],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    documents_by_id: dict[str, Any] = {}
    for result in cached_results:
        for document in result.documents:
            previous = documents_by_id.get(document.document_id)
            if previous is not None and _document_signature(previous) != _document_signature(document):
                raise ValueError(
                    "canonical document text disagreement for document_id="
                    f"{document.document_id} across cached facet retrievals"
                )
            documents_by_id[document.document_id] = document

    selected_ids = {candidate["document_id"] for candidate in candidates}
    missing_translation_ids: list[str] = []
    seen_missing_translation_ids: set[str] = set()
    for result in cached_results:
        for document_id in result.missing_translation_ids:
            if (
                document_id in selected_ids
                and document_id not in seen_missing_translation_ids
            ):
                seen_missing_translation_ids.add(document_id)
                missing_translation_ids.append(document_id)
    enriched: list[dict[str, Any]] = []
    for candidate in candidates:
        document = documents_by_id.get(candidate["document_id"])
        if document is None:
            raise ValueError(
                "cache-only retrieval is missing hydrated document_id="
                f"{candidate['document_id']}"
            )
        enriched.append(
            {
                **candidate,
                "hydrated": document.hydrated,
                "text": document.text,
                "report_text": document.report_text,
                "english_text": document.english_text,
                "language": document.language,
                "url": document.url,
                "date": document.date,
            }
        )
    return enriched, {
        "hydrated_count": len(enriched),
        "unresolved_ids": [],
        "missing_translation_ids": missing_translation_ids,
        "failure_count": 0,
    }


def _topic_record(
    plan: QueryPlan,
    *,
    baseline: str,
    query_records: Sequence[dict[str, Any]],
    plan_sha256: str,
    retriever: RetrieverLike,
    top_k: int,
    rrf_k: int,
    fresh_results: Mapping[tuple[str, int], RetrievalResult] | None = None,
) -> dict[str, Any]:
    successful = [record for record in query_records if record["status"] == "success"]
    current_results = fresh_results or {}
    cached_results: list[RetrievalResult] = []
    for query_index, record in enumerate(query_records, start=1):
        if record["status"] != "success":
            continue
        fresh = current_results.get((plan.topic_id, query_index))
        if fresh is not None:
            cached_results.append(fresh)
        else:
            cached_results.append(
                _load_cached_query_result(
                    record,
                    baseline=baseline,
                    plan_sha256=plan_sha256,
                    retriever=retriever,
                    top_k=top_k,
                    expected_query_index=query_index,
                )
            )
    ranked_lists = [
        tuple(SearchHit(**hit) for hit in record["hits"])
        for record in successful
    ]
    candidates = (
        fuse_ranked_hits(
            ranked_lists,
            query_indices=[record["query_index"] for record in successful],
            rrf_k=rrf_k,
            top_k=top_k,
        )
        if ranked_lists
        else []
    )
    if baseline == "c":
        for candidate in candidates:
            candidate["score"] = None
    candidates, hydration = _assemble_candidates(candidates, cached_results)
    if not successful:
        status = "error"
    elif len(successful) == len(query_records):
        status = "success"
    else:
        status = "partial"
    topic_query_results = []
    for record in query_records:
        topic_record = {key: value for key, value in record.items() if key != "topic_id"}
        topic_query_results.append(topic_record)
    return {
        "topic_id": plan.topic_id,
        "variant": _BASELINE_NAMES[baseline],
        "status": status,
        "plan_status": plan.plan_status,
        "queries": list(_queries_for(plan, baseline)),
        "query_results": topic_query_results,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "hydration": hydration,
        "created_at": _now(),
    }


def _manifest_counts(
    plans: Sequence[QueryPlan],
    checkpoints: dict[tuple[Any, ...], dict[str, Any]],
    topics: dict[tuple[Any, ...], dict[str, Any]],
) -> dict[str, int]:
    topic_records = list(topics.values())
    query_records = list(checkpoints.values())
    return {
        "topics_planned": len(plans),
        "topics_recorded": len(topic_records),
        "successful_topics": sum(record["status"] == "success" for record in topic_records),
        "partial_topics": sum(record["status"] == "partial" for record in topic_records),
        "failed_topics": sum(record["status"] == "error" for record in topic_records),
        "query_records": len(query_records),
        "successful_queries": sum(record["status"] == "success" for record in query_records),
        "failed_queries": sum(record["status"] == "error" for record in query_records),
        "candidates": sum(record["candidate_count"] for record in topic_records),
        "unresolved_documents": sum(
            len(record["hydration"]["unresolved_ids"]) for record in topic_records
        ),
        "missing_translation_documents": sum(
            len(record["hydration"]["missing_translation_ids"])
            for record in topic_records
        ),
    }


def _corpus_identity(retriever: RetrieverLike) -> dict[str, Any]:
    try:
        captured = getattr(retriever, "retrieval_corpus_identity", None)
        identity = dict(
            captured if isinstance(captured, Mapping) else retriever.corpus_identity
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "retriever must expose a valid immutable corpus_identity mapping"
        ) from exc
    required = {"backend", "path", "schema_version", "dataset", "revision"}
    missing = sorted(required.difference(identity))
    if missing:
        raise ValueError(
            "corpus_identity is missing required fields: " + ", ".join(missing)
        )
    try:
        json.dumps(identity, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("corpus_identity must be JSON serializable") from exc
    return identity


@contextmanager
def _baseline_root_lock(output_root: Path) -> Iterator[None]:
    """Permit only one baseline process to use organizer capacity at a time."""

    lock_path = output_root / ".query-baselines.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        try:
            import fcntl
        except ImportError:  # pragma: no cover - supported production host is POSIX.
            yield
        else:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(
                    f"another query baseline is already running under {output_root}"
                ) from None
            try:
                yield
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _hydration_error_topic_record(
    plan: QueryPlan,
    *,
    baseline: str,
    query_records: Sequence[dict[str, Any]],
    error: Exception,
) -> dict[str, Any]:
    return {
        "topic_id": plan.topic_id,
        "variant": _BASELINE_NAMES[baseline],
        "status": "error",
        "plan_status": plan.plan_status,
        "queries": list(_queries_for(plan, baseline)),
        "query_results": [
            {key: value for key, value in record.items() if key != "topic_id"}
            for record in query_records
        ],
        "candidate_count": 0,
        "candidates": [],
        "hydration": {
            "status": "error",
            "error_type": type(error).__name__,
            "error_detail": str(error),
            "hydrated_count": 0,
            "unresolved_ids": [],
            "missing_translation_ids": [],
            "failure_count": 1,
        },
        "created_at": _now(),
    }


def run_query_baseline(
    plans_path: Path,
    *,
    baseline: str,
    retriever: RetrieverLike,
    retrievers: Sequence[RetrieverLike] | None = None,
    concurrency: int = 1,
    output_root: Path,
    run_id: str,
    top_k: int = DEFAULT_TOP_K,
    rrf_k: int = DEFAULT_RRF_K,
    progress: Callable[[str], None] | None = None,
    expected_topic_ids: Sequence[str] | None = EXPECTED_TOPIC_IDS,
    expected_plan_sha256: str | None = EXPECTED_QUERY_PLAN_SHA256,
) -> Path:
    """Run or resume one exact query-plan baseline and return its artifact path."""

    if baseline not in _BASELINE_NAMES:
        raise ValueError("baseline must be 'b' or 'c'")
    if (
        not isinstance(run_id, str)
        or not run_id
        or run_id in {".", ".."}
        or Path(run_id).name != run_id
    ):
        raise ValueError("run_id must be one non-empty path component")
    if top_k != DEFAULT_TOP_K:
        raise ValueError(f"official baselines require top_k={DEFAULT_TOP_K}")
    if rrf_k != DEFAULT_RRF_K:
        raise ValueError(f"official baselines require rrf_k={DEFAULT_RRF_K}")
    if concurrency not in {1, 2, 4}:
        raise ValueError("concurrency must be one of 1, 2, or 4")

    worker_retrievers = (
        (retriever,) if retrievers is None else tuple(retrievers)
    )
    if len(worker_retrievers) < concurrency:
        raise ValueError("one independent retriever is required per worker")
    worker_retrievers = worker_retrievers[:concurrency]
    if len({id(worker) for worker in worker_retrievers}) != concurrency:
        raise ValueError("worker retrievers must be distinct instances")

    plans_path = Path(plans_path)
    plans = load_query_plans(plans_path)
    actual_topic_ids = tuple(plan.topic_id for plan in plans)
    if expected_topic_ids is not None and actual_topic_ids != tuple(expected_topic_ids):
        raise ValueError("query plan topic IDs do not match the expected 103-topic set")
    plan_sha256 = _sha256_file(plans_path)
    if (
        expected_plan_sha256 is not None
        and plan_sha256 != expected_plan_sha256
    ):
        raise ValueError(
            "query plan SHA-256 does not match the pinned baseline query plan"
        )
    corpus_identity = _corpus_identity(retriever)
    if any(
        _corpus_identity(worker) != corpus_identity for worker in worker_retrievers
    ):
        raise ValueError("all worker retrievers must use the same corpus identity")

    output_root = Path(output_root).expanduser().absolute()
    if output_root.is_symlink():
        raise ValueError("output_root must not be a symbolic link")
    output_root.mkdir(parents=True, exist_ok=True)
    output_root = output_root.resolve()
    with _baseline_root_lock(output_root):
        run_dir = output_root / run_id
        if run_dir.is_symlink():
            raise ValueError("run directory must not be a symbolic link")
        if run_dir.resolve(strict=False).parent != output_root:
            raise ValueError("run directory must be directly beneath output_root")
        run_dir.mkdir(mode=0o700, exist_ok=True)
        return _run_query_baseline_locked(
            plans_path=plans_path,
            plans=plans,
            plan_sha256=plan_sha256,
            corpus_identity=corpus_identity,
            baseline=baseline,
            retriever=retriever,
            worker_retrievers=worker_retrievers,
            concurrency=concurrency,
            run_dir=run_dir,
            run_id=run_id,
            top_k=top_k,
            rrf_k=rrf_k,
            progress=progress,
        )


def _run_query_baseline_locked(
    *,
    plans_path: Path,
    plans: Sequence[QueryPlan],
    plan_sha256: str,
    corpus_identity: dict[str, Any],
    baseline: str,
    retriever: RetrieverLike,
    worker_retrievers: Sequence[RetrieverLike],
    concurrency: int,
    run_dir: Path,
    run_id: str,
    top_k: int,
    rrf_k: int,
    progress: Callable[[str], None] | None,
) -> Path:
    """Execute with the shared baseline-root lock already held."""

    run_dir.mkdir(mode=0o700, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    queries_path = run_dir / "queries.jsonl"
    topics_path = run_dir / "topics.jsonl"

    prior_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else None
    )
    if prior_manifest is not None:
        expected = {
            "baseline": baseline,
            "pipeline": PIPELINE,
            "top_k": top_k,
            "rrf_k": rrf_k,
            "plan_sha256": plan_sha256,
            "corpus_identity": corpus_identity,
        }
        actual = {
            "baseline": prior_manifest.get("baseline"),
            "pipeline": prior_manifest.get("pipeline"),
            "top_k": prior_manifest.get("top_k"),
            "rrf_k": prior_manifest.get("rrf_k"),
            "plan_sha256": prior_manifest.get("query_plan", {}).get("sha256"),
            "corpus_identity": prior_manifest.get("corpus"),
        }
        if actual != expected:
            raise ValueError("existing run configuration does not match this request")

    checkpoints = _read_latest_jsonl(
        queries_path, key_fields=("topic_id", "query_index")
    )
    topics = _read_latest_jsonl(topics_path, key_fields=("topic_id",))
    created_at = prior_manifest.get("created_at") if prior_manifest else _now()
    fresh_results: dict[tuple[str, int], RetrievalResult] = {}

    def write_manifest(
        status: str, *, failure: dict[str, str] | None = None
    ) -> None:
        manifest = {
            "schema_version": "query_baseline_v1",
            "run_id": run_id,
            "baseline": baseline,
            "variant": _BASELINE_NAMES[baseline],
            "status": status,
            "created_at": created_at,
            "updated_at": _now(),
            "query_plan": {
                "path": str(plans_path.resolve()),
                "sha256": plan_sha256,
            },
            "corpus": corpus_identity,
            "query_construction": (
                "problem_background_single_query_v1"
                if baseline == "b"
                else "problem_background_facets_v1"
            ),
            "pipeline": PIPELINE,
            "top_k": top_k,
            "rrf_k": rrf_k,
            "transport_retries": 0,
            "concurrency": concurrency,
            "counts": _manifest_counts(plans, checkpoints, topics),
        }
        if failure is not None:
            manifest["failure"] = failure
        _write_json(manifest_path, manifest)

    def publish_topic(plan: QueryPlan) -> bool:
        try:
            query_records = _query_records_for_topic(
                plan,
                baseline=baseline,
                checkpoints=checkpoints,
            )
        except ValueError as exc:
            query_records = None
            topic = _hydration_error_topic_record(
                plan,
                baseline=baseline,
                query_records=[],
                error=exc,
            )
            topics[(plan.topic_id,)] = topic
            _write_jsonl(
                topics_path,
                [
                    topics[(ordered_plan.topic_id,)]
                    for ordered_plan in plans
                    if (ordered_plan.topic_id,) in topics
                ],
            )
            write_manifest("running")
            return True
        if query_records is None:
            return False
        try:
            topic = _topic_record(
                plan,
                baseline=baseline,
                query_records=query_records,
                plan_sha256=plan_sha256,
                retriever=retriever,
                top_k=top_k,
                rrf_k=rrf_k,
                fresh_results=fresh_results,
            )
        except (RagtimeError, HuggingFaceError, LocalCorpusError, ValueError) as exc:
            topic = _hydration_error_topic_record(
                plan,
                baseline=baseline,
                query_records=query_records,
                error=exc,
            )
        topics[(plan.topic_id,)] = topic
        _write_jsonl(
            topics_path,
            [
                topics[(ordered_plan.topic_id,)]
                for ordered_plan in plans
                if (ordered_plan.topic_id,) in topics
            ],
        )
        write_manifest("running")
        if progress is not None:
            counts = _manifest_counts(plans, checkpoints, topics)
            progress(
                f"{len(topics)}/{len(plans)} topic={plan.topic_id} "
                f"status={topic['status']} failed_queries={counts['failed_queries']}"
            )
        return True

    tasks: list[QueryTask] = []
    pending_keys: set[tuple[str, int]] = set()
    ordinal = 0
    for plan in plans:
        for query_index, query in enumerate(_queries_for(plan, baseline), start=1):
            key = (plan.topic_id, query_index)
            checkpoint = checkpoints.get(key)
            if checkpoint is None:
                malformed = any(
                    stored_topic_id == plan.topic_id
                    and stale.get("query") == query
                    and stored_index != query_index
                    for (stored_topic_id, stored_index), stale in checkpoints.items()
                )
                if malformed:
                    # Leave malformed checkpoints in place for explicit topic
                    # publication diagnostics; never silently retry them.
                    continue
            if checkpoint is not None:
                stored_index = checkpoint.get("query_index")
                if (
                    isinstance(stored_index, bool)
                    or not isinstance(stored_index, int)
                    or stored_index < 1
                    or stored_index != query_index
                ):
                    # Leave malformed checkpoints in place for explicit topic
                    # publication diagnostics; never silently retry them.
                    continue
            if (
                checkpoint is not None
                and checkpoint.get("status") == "success"
                and checkpoint.get("query") == query
                and checkpoint.get("pipeline") == PIPELINE
                and _checkpoint_has_reference(checkpoint)
            ):
                continue
            attempt = int(checkpoint.get("attempt", 0)) + 1 if checkpoint else 1
            tasks.append(
                QueryTask(
                    ordinal=ordinal,
                    topic_id=plan.topic_id,
                    query_index=query_index,
                    query=query,
                    attempt=attempt,
                    query_context=_query_context(
                        baseline=baseline,
                        plan_sha256=plan_sha256,
                        query_index=query_index,
                    ),
                )
            )
            pending_keys.add(key)
            ordinal += 1

    write_manifest("running")
    task_iterator = iter(tasks)
    executor = ThreadPoolExecutor(
        max_workers=concurrency,
        thread_name_prefix="ragtime-baseline",
    )
    stopped_on_failure = False
    total_queries = sum(len(_queries_for(plan, baseline)) for plan in plans)
    try:
        while True:
            wave = list(islice(task_iterator, concurrency))
            if not wave:
                break
            futures: dict[Future[_QueryTaskResult], QueryTask] = {
                executor.submit(
                    _run_query_task,
                    task,
                    retriever=worker,
                    top_k=top_k,
                ): task
                for worker, task in zip(
                    worker_retrievers[: len(wave)], wave, strict=True
                )
            }
            wave_failed = False
            while futures:
                done, _pending = wait(futures, return_when=FIRST_COMPLETED)
                for future in sorted(done, key=lambda item: futures[item].ordinal):
                    task = futures.pop(future)
                    task_result = future.result()
                    checkpoint = task_result.record
                    key = (task.topic_id, task.query_index)
                    checkpoints[key] = checkpoint
                    if task_result.retrieval is not None:
                        fresh_results[key] = task_result.retrieval
                    pending_keys.discard(key)
                    _write_jsonl(
                        queries_path,
                        _ordered_query_records(plans, baseline, checkpoints),
                    )
                    write_manifest("running")
                    wave_failed = wave_failed or checkpoint["status"] == "error"
                    if progress is not None:
                        counts = _manifest_counts(plans, checkpoints, topics)
                        progress(
                            f"queries={counts['query_records']}/{total_queries} "
                            f"topic={task.topic_id} query={task.query_index} "
                            f"status={checkpoint['status']} "
                            f"failed_queries={counts['failed_queries']}"
                        )
            if wave_failed and concurrency > 1:
                stopped_on_failure = True
                break
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    for plan in plans:
        plan_keys = {
            (plan.topic_id, query_index)
            for query_index, _query in enumerate(
                _queries_for(plan, baseline), start=1
            )
        }
        if plan_keys.isdisjoint(pending_keys):
            publish_topic(plan)

    if stopped_on_failure:
        write_manifest("stopped_on_failure")
        raise ConcurrentRetrievalStopped(
            "concurrent retrieval stopped after an explicit failed query; "
            "resume with lower concurrency"
        )

    write_manifest("complete")
    return run_dir


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer") from None
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def default_query_plan_path() -> Path:
    return (
        default_project_cache_dir()
        / "experiments"
        / "query-variants"
        / "problem-background-2026-08-03"
        / "query_plans.jsonl"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a resumable B or C RAGTIME query baseline."
    )
    parser.add_argument("baseline", choices=("b", "c"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--plans",
        type=Path,
        default=default_query_plan_path(),
        help="immutable query-plan JSONL (default: %(default)s)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_project_cache_dir() / "experiments" / "query-baselines",
    )
    parser.add_argument(
        "--top-k",
        type=_positive_int,
        choices=(DEFAULT_TOP_K,),
        default=DEFAULT_TOP_K,
    )
    parser.add_argument(
        "--rrf-k",
        type=_positive_int,
        choices=(DEFAULT_RRF_K,),
        default=DEFAULT_RRF_K,
    )
    parser.add_argument(
        "--concurrency",
        type=_positive_int,
        choices=(1, 2, 4),
        default=1,
        help="independent retrieval workers (default: %(default)s)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    from .retriever import RagtimeRetriever

    try:
        with ExitStack() as stack:
            retrievers = tuple(
                stack.enter_context(RagtimeRetriever.from_env(retries=0))
                for _index in range(arguments.concurrency)
            )
            run_dir = run_query_baseline(
                arguments.plans,
                baseline=arguments.baseline,
                retriever=retrievers[0],
                retrievers=retrievers,
                concurrency=arguments.concurrency,
                output_root=arguments.output_root,
                run_id=arguments.run_id,
                top_k=arguments.top_k,
                rrf_k=arguments.rrf_k,
                progress=lambda line: print(line, flush=True),
            )
    except ConcurrentRetrievalStopped as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2
    print(f"Baseline artifact: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "QueryPlan",
    "build_parser",
    "default_query_plan_path",
    "fuse_ranked_hits",
    "load_query_plans",
    "main",
    "run_query_baseline",
]
