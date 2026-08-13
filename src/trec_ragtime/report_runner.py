"""Resumable, parallel, manifest-last report generation over a topic set.

Structurally this is ``query_baselines`` for generation: a root lock, per-topic
checkpoints rewritten atomically in topic order, a wave scheduler, and a
manifest rewritten on every completion. The official JSONL is emitted once, at
the end, only from reports that individually passed both the domain gate in
``report_cache.validate_report`` and the wire gate in ``report_submission``.

Two deliberate differences from the retrieval baselines. A topic failure does
not abort the run: there is no shared organizer capacity to protect here, so the
run records the failure and continues. And the durable artifact cache is
mandatory rather than best effort -- silently dropping to no cache would destroy
resumability mid-run, which is exactly when it matters.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

from .artifact_store import ArtifactStore, ArtifactStoreError
from .evidence_bundle import (
    EvidenceBundle,
    EvidenceBundleError,
    build_evidence_bundle_from_handoff,
    build_evidence_bundles,
    load_evidence_run,
    sha256_file,
)
from .report_handoff import (
    HandoffSourceError,
    load_handoff_manifest,
    require_handoff_bundle,
)
from .paths import default_project_cache_dir, ensure_private_cache_dir
from .report_cache import (
    GenerationSettings,
    ReportArtifactCache,
    ReportRequestIdentity,
    TopicReport,
)
from .report_generation import (
    ChatCompletionsLike,
    ObservedCacheStore,
    ReportGenerationError,
    allocate_facet_budget,
    build_messages,
    generate_topic_report,
)
from .report_submission import (
    MAX_RUN_ID_LENGTH,
    SubmissionValidationError,
    submission_record,
    validate_submission_file,
    validate_run_desc,
    validate_run_id,
    validate_team_id,
)
from .topic_candidates import Topic, load_topics

REPORT_RUN_SCHEMA_VERSION = "report_generation_v1"
DEFAULT_TOPICS_PATH = "topics/topics.all.2026.v0625-fix.jsonl"
DEFAULT_API_KEY_ENVS = ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "LITELLM_PROXY_API_KEY")
DEFAULT_BASE_URL = "https://api.openai.com/v1"
ALLOWED_CONCURRENCY = (1, 2, 4, 8)


class ReportRunError(RuntimeError):
    """The run cannot proceed as configured."""


@dataclass(frozen=True, slots=True)
class ReportRunResult:
    """Outcome summary for one report generation run."""

    run_dir: Path
    status: str
    generated: int
    cached: int
    failed: int
    missing: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.status == "complete"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", text=True
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    _atomic_write(
        path,
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in records
        ),
    )


@contextmanager
def _report_root_lock(output_root: Path) -> Iterator[None]:
    """Permit only one report generation process per output root."""

    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".report-generation.lock"
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
                raise ReportRunError(
                    f"another report run is already active under {output_root}"
                ) from None
            try:
                yield
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _settings_record(settings: GenerationSettings) -> dict[str, Any]:
    """Sanitized settings: identity-bearing values only, never credentials."""

    return {
        "base_url": settings.base_url,
        "model": settings.model,
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "max_output_tokens": settings.max_output_tokens,
        "seed": settings.seed,
        "max_repairs": settings.max_repairs,
        "max_citations": settings.max_citations,
        "max_passages": settings.max_passages,
        "evidence_chars": settings.evidence_chars,
        "budget_ratio": settings.budget_ratio,
        "response_format_mode": settings.response_format_mode,
        "prompt_version": settings.prompt_version,
        "output_schema_version": settings.output_schema_version,
        "segmenter_version": settings.segmenter_version,
        "citation_policy_version": settings.citation_policy_version,
        "length_packing_version": settings.length_packing_version,
        "generator_behavior_version": settings.generator_behavior_version,
    }


def _checkpoint(report: TopicReport) -> dict[str, Any]:
    reference = report.report_artifact
    return {
        "topic_id": report.topic_id,
        "status": "success",
        "sentence_count": len(report.sentences),
        "normalized_character_count": report.normalized_character_count,
        "budgeted_character_count": report.budgeted_character_count,
        "character_limit": report.character_limit,
        "reference_count": len(report.references),
        "bundle_seal": report.bundle_seal.hex(),
        "object_key": reference.object_key if reference is not None else None,
        "cache_hit": bool(reference.cache_hit) if reference is not None else False,
        "model_id": report.diagnostics.model_id,
        "repair_attempts": report.diagnostics.repair_attempts,
        "dropped": sorted({item.reason for item in report.diagnostics.dropped}),
        "dropped_count": len(report.diagnostics.dropped),
        "usage": dict(report.diagnostics.usage or {}),
    }


def _lineage(report: TopicReport, bundle: EvidenceBundle) -> dict[str, Any]:
    return {
        "topic_id": report.topic_id,
        "bundle_seal": report.bundle_seal.hex(),
        "handles": {
            passage.handle: passage.document_id for passage in bundle.passages
        },
        "facets": {facet.facet_id: facet.source for facet in bundle.facets},
        "sentences": [
            {
                "index": item.index,
                "substantive": item.substantive,
                "facet_ids": list(item.facet_ids),
                "citations": [
                    {
                        "handle": citation.handle,
                        "document_id": citation.document_id,
                        "score": citation.score,
                    }
                    for citation in item.citations
                ],
            }
            for item in report.sentences
        ],
    }


@dataclass(frozen=True, slots=True)
class _TopicTask:
    ordinal: int
    topic: Topic
    bundle: EvidenceBundle
    request: ReportRequestIdentity


@dataclass(frozen=True, slots=True)
class _TopicOutcome:
    ordinal: int
    topic_id: str
    report: TopicReport | None
    error: str | None


def _run_topic(
    task: _TopicTask,
    *,
    completions: ChatCompletionsLike,
    cache: ReportArtifactCache,
    clock: Callable[[], float],
    refresh: bool,
    probe: Callable[[], bool] | None,
) -> _TopicOutcome:
    try:
        report = generate_topic_report(
            bundle=task.bundle,
            request=task.request,
            completions=completions,
            cache=cache,
            clock=clock,
            refresh=refresh,
            llm_cache_probe=probe,
        )
    except (ReportGenerationError, ArtifactStoreError, ValueError) as error:
        return _TopicOutcome(task.ordinal, task.topic.topic_id, None, f"{type(error).__name__}: {error}")
    return _TopicOutcome(task.ordinal, task.topic.topic_id, report, None)


def _handoff_topic_matches(handoff: Any, topic: Topic) -> bool:
    """Require the embedded official topic to match the catalog exactly."""

    record = handoff.topic
    return (
        record.topic_id == topic.topic_id
        and record.collection_id == topic.collection_id
        and record.title == topic.title
        and record.problem_statement == topic.problem_statement
        and record.background == topic.background
        and record.limit == topic.limit
    )


def _bundles_from_handoff(
    manifest_path: Path,
    *,
    topics: Sequence[Topic],
    store: Any,
    topics_sha256: str,
    evidence_chars: int,
) -> tuple[tuple[EvidenceBundle, ...], tuple[Topic, ...], tuple[str, ...], dict[str, Any]]:
    """Load one self-contained handoff bundle per topic, cache-only."""

    manifest = load_handoff_manifest(manifest_path)
    bundles: list[EvidenceBundle] = []
    available: list[Topic] = []
    missing: list[str] = []
    for topic in topics:
        entry = manifest.entry(topic.topic_id)
        if entry is None or not entry.available:
            missing.append(topic.topic_id)
            continue
        handoff = require_handoff_bundle(store, entry)
        if not _handoff_topic_matches(handoff, topic):
            raise ReportRunError(
                f"handoff topic record for {topic.topic_id} does not match the"
                " official topic catalog"
            )
        bundles.append(
            build_evidence_bundle_from_handoff(
                handoff, topics_sha256=topics_sha256, evidence_chars=evidence_chars
            )
        )
        available.append(topic)
    record = {
        "run_schema": "retrieval_handoff_v1",
        "manifest_path_name": manifest.path_name,
        "handoff_fingerprint": manifest.fingerprint,
        "topic_catalog_digest": manifest.topic_catalog_digest,
        "handoff_complete": manifest.complete,
    }
    return tuple(bundles), tuple(available), tuple(missing), record


def run_report_generation(
    *,
    input_dir: Path | str | None = None,
    handoff_manifest: Path | str | None = None,
    topics_path: Path | str,
    output_root: Path | str,
    team_id: str,
    run_id: str,
    run_desc: str,
    settings: GenerationSettings,
    completions: Sequence[ChatCompletionsLike],
    store: Any,
    topic_ids: Sequence[str] | None = None,
    concurrency: int = 1,
    clock: Callable[[], float] = time.time,
    refresh: bool = False,
    allow_partial: bool = False,
    dry_run: bool = False,
    progress: Callable[[str], None] | None = None,
    cache_probes: Sequence[Callable[[], bool] | None] | None = None,
) -> ReportRunResult:
    """Generate, validate, and export reports for every requested topic."""

    team_id = validate_team_id(team_id)
    run_id = validate_run_id(run_id)
    run_desc = validate_run_desc(run_desc)
    if concurrency not in ALLOWED_CONCURRENCY:
        raise ReportRunError(f"concurrency must be one of {list(ALLOWED_CONCURRENCY)}")
    workers = list(completions)
    if len(workers) < concurrency:
        raise ReportRunError(
            f"{concurrency} workers need {concurrency} completions clients, got {len(workers)}"
        )
    if len({id(worker) for worker in workers}) != len(workers):
        raise ReportRunError("each worker needs its own completions client")

    topics_path = Path(topics_path)
    input_dir = Path(input_dir) if input_dir is not None else None
    all_topics = load_topics(topics_path)
    if topic_ids is not None:
        wanted = set(topic_ids)
        unknown = wanted - {topic.topic_id for topic in all_topics}
        if unknown:
            raise ReportRunError(f"unknown topic ids: {sorted(unknown)}")
        topics = tuple(topic for topic in all_topics if topic.topic_id in wanted)
    else:
        topics = all_topics

    topics_sha256 = sha256_file(topics_path)
    if handoff_manifest is not None:
        try:
            bundles, available, missing, input_record = _bundles_from_handoff(
                Path(handoff_manifest),
                topics=topics,
                store=store,
                topics_sha256=topics_sha256,
                evidence_chars=settings.evidence_chars,
            )
        except (HandoffSourceError, EvidenceBundleError, ArtifactStoreError) as error:
            raise ReportRunError(str(error)) from None
        corpus_record: dict[str, Any] = {"dataset": None, "revision": None}
        if not available:
            raise ReportRunError(
                "the handoff manifest covers none of the requested topics"
            )
    else:
        if input_dir is None:
            raise ReportRunError("either input_dir or handoff_manifest is required")
        try:
            evidence_run = load_evidence_run(input_dir)
        except EvidenceBundleError as error:
            raise ReportRunError(str(error)) from None

        available = tuple(topic for topic in topics if topic.topic_id in evidence_run.topics)
        missing = tuple(
            topic.topic_id for topic in topics if topic.topic_id not in evidence_run.topics
        )
        if not available:
            raise ReportRunError(
                f"evidence run {evidence_run.run_id} covers none of the requested topics"
            )

        bundles = build_evidence_bundles(
            evidence_run,
            topics=available,
            topics_sha256=topics_sha256,
            max_passages=settings.max_passages,
            evidence_chars=settings.evidence_chars,
        )
        input_record = {
            "run_schema": evidence_run.run_schema,
            "run_id": evidence_run.run_id,
            "run_dir_name": evidence_run.run_dir_name,
            "manifest_sha256": evidence_run.manifest_sha256,
            "records_sha256": evidence_run.records_sha256,
            "rejected_candidates": evidence_run.rejected_candidates,
        }
        corpus_record = {
            "dataset": evidence_run.corpus_dataset,
            "revision": evidence_run.corpus_revision,
        }
    requests = tuple(
        ReportRequestIdentity.for_bundle(bundle, topic=topic, settings=settings)
        for bundle, topic in zip(bundles, available)
    )
    fingerprint = settings.fingerprint.hex()

    output_root = Path(output_root)
    with _report_root_lock(output_root):
        run_dir = ensure_private_cache_dir(output_root.resolve() / run_id)
        manifest_path = run_dir / "manifest.json"
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("generator_fingerprint") not in (None, fingerprint):
                raise ReportRunError(
                    f"run {run_id} already exists with different generation settings;"
                    " choose a new --run-id"
                )

        base_manifest: dict[str, Any] = {
            "schema_version": REPORT_RUN_SCHEMA_VERSION,
            "run_id": run_id,
            "team_id": team_id,
            "run_desc": run_desc,
            "generator_fingerprint": fingerprint,
            "settings": _settings_record(settings),
            "input": input_record,
            "topics": {
                "path_name": topics_path.name,
                "sha256": topics_sha256,
                "requested": len(topics),
                "with_evidence": len(available),
            },
            "corpus": corpus_record,
            "concurrency": concurrency,
        }

        def write_manifest(status: str, records: Sequence[Mapping[str, Any]]) -> None:
            payload = dict(base_manifest)
            payload["status"] = status
            payload["partial"] = status == "incomplete" and allow_partial
            payload["counts"] = {
                "topics_requested": len(topics),
                "topics_with_evidence": len(available),
                "topics_missing_evidence": len(missing),
                "reports": len(records),
                "cache_hits": sum(1 for item in records if item.get("cache_hit")),
                "failed": len(available) - len(records),
            }
            payload["missing_topics"] = list(missing)
            payload["reports"] = [
                {
                    "topic_id": item["topic_id"],
                    "object_key": item["object_key"],
                    "bundle_seal": item["bundle_seal"],
                    "sentence_count": item["sentence_count"],
                    "normalized_character_count": item["normalized_character_count"],
                    "cache_hit": item["cache_hit"],
                }
                for item in records
            ]
            _write_json(manifest_path, payload)

        if dry_run:
            previews = []
            for bundle, topic in zip(bundles, available):
                budgets = allocate_facet_budget(
                    bundle, limit=topic.limit, budget_ratio=settings.budget_ratio
                )
                messages = build_messages(bundle, budgets, settings)
                previews.append(
                    {
                        "topic_id": topic.topic_id,
                        "status": "dry_run",
                        "facets": len(bundle.facets),
                        "passages": len(bundle.passages),
                        "prompt_characters": sum(len(item["content"]) for item in messages),
                        "bundle_seal": bundle.seal,
                    }
                )
            _write_jsonl(run_dir / "reports.jsonl", previews)
            write_manifest("dry_run", [])
            return ReportRunResult(run_dir, "dry_run", 0, 0, 0, missing)

        cache = ReportArtifactCache(store, clock=clock)
        if cache_probes is None:
            probes: list[Callable[[], bool] | None] = [None] * len(workers)
        else:
            probes = list(cache_probes)
            if len(probes) < len(workers):
                raise ReportRunError("a cache probe is required for every worker")

        tasks = [
            _TopicTask(ordinal, topic, bundle, request)
            for ordinal, (topic, bundle, request) in enumerate(
                zip(available, bundles, requests)
            )
        ]
        reports: dict[str, TopicReport] = {}
        bundle_by_topic = {bundle.topic_id: bundle for bundle in bundles}
        failures: dict[str, str] = {}
        checkpoints: dict[str, dict[str, Any]] = {}

        def ordered(mapping: Mapping[str, Any]) -> list[Any]:
            return [mapping[topic.topic_id] for topic in available if topic.topic_id in mapping]

        write_manifest("running", [])
        executor = ThreadPoolExecutor(
            max_workers=concurrency, thread_name_prefix="ragtime-report"
        )
        try:
            task_iterator = iter(tasks)
            while True:
                wave = list(islice(task_iterator, concurrency))
                if not wave:
                    break
                futures: dict[Future[_TopicOutcome], _TopicTask] = {
                    executor.submit(
                        _run_topic,
                        task,
                        completions=worker,
                        cache=cache,
                        clock=clock,
                        refresh=refresh,
                        probe=probe,
                    ): task
                    for (worker, probe), task in zip(
                        list(zip(workers, probes))[: len(wave)], wave, strict=True
                    )
                }
                while futures:
                    done, _pending = wait(futures, return_when=FIRST_COMPLETED)
                    for future in sorted(done, key=lambda item: futures[item].ordinal):
                        task = futures.pop(future)
                        outcome = future.result()
                        if outcome.report is None:
                            failures[outcome.topic_id] = outcome.error or "unknown error"
                        else:
                            reports[outcome.topic_id] = outcome.report
                            checkpoints[outcome.topic_id] = _checkpoint(outcome.report)
                        _write_jsonl(run_dir / "reports.jsonl", ordered(checkpoints))
                        write_manifest("running", ordered(checkpoints))
                        if progress is not None:
                            progress(
                                f"topics={len(checkpoints)}/{len(tasks)} "
                                f"topic={outcome.topic_id} "
                                f"status={'error' if outcome.report is None else 'success'} "
                                f"failed={len(failures)}"
                            )
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

        _write_jsonl(
            run_dir / "lineage.jsonl",
            [
                _lineage(reports[topic.topic_id], bundle_by_topic[topic.topic_id])
                for topic in available
                if topic.topic_id in reports
            ],
        )
        _write_json(
            run_dir / "failures.json",
            {
                "failed_topics": sorted(failures),
                "missing_evidence_topics": list(missing),
                "details": failures,
            },
        )

        complete = not failures and not missing
        status = "complete" if complete else "incomplete"
        records = ordered(checkpoints)

        if complete or allow_partial:
            submission = [
                submission_record(
                    team_id=team_id,
                    topic_id=topic.topic_id,
                    run_id=run_id,
                    run_desc=run_desc,
                    sentences=reports[topic.topic_id].sentence_payloads(),
                )
                for topic in available
                if topic.topic_id in reports
            ]
            submission_path = run_dir / "submission.jsonl"
            _write_jsonl(submission_path, submission)
            emitted = {
                topic.topic_id: topic.limit
                for topic in available
                if topic.topic_id in reports
            }
            validation = validate_submission_file(
                submission_path, topics=emitted, team_id=team_id, run_id=run_id
            )
            if not validation.ok:
                submission_path.unlink(missing_ok=True)
                write_manifest("invalid", records)
                detail = "; ".join(item.render() for item in validation.violations[:5])
                raise ReportRunError(f"aggregate submission failed validation: {detail}")

        write_manifest(status, records)
        return ReportRunResult(
            run_dir=run_dir,
            status=status,
            generated=sum(1 for item in records if not item["cache_hit"]),
            cached=sum(1 for item in records if item["cache_hit"]),
            failed=len(failures),
            missing=missing,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragtime-report",
        description="Generate validated RAGTIME reports from a sealed evidence run.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Retrieval run directory")
    source.add_argument(
        "--handoff-manifest",
        help="Manifest from ragtime-retrieval-handoff; consumes one sealed bundle per topic",
    )
    parser.add_argument("--topics", default=DEFAULT_TOPICS_PATH, help="Official topics JSONL")
    parser.add_argument(
        "--team-id",
        default=os.getenv("RAGTIME_TEAM_ID"),
        required=os.getenv("RAGTIME_TEAM_ID") is None,
        help="Registered TREC team identifier (env RAGTIME_TEAM_ID)",
    )
    parser.add_argument(
        "--run-id",
        default=os.getenv("RAGTIME_RUN_ID"),
        required=os.getenv("RAGTIME_RUN_ID") is None,
        help=f"Run identifier, at most {MAX_RUN_ID_LENGTH} characters",
    )
    parser.add_argument("--run-desc", default=None, help="Run description for metadata")
    parser.add_argument(
        "--model",
        default=os.getenv("RAGTIME_REPORT_MODEL"),
        required=os.getenv("RAGTIME_REPORT_MODEL") is None,
        help="Generation model id (env RAGTIME_REPORT_MODEL)",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL") or os.getenv("OPENROUTER_BASE_URL") or DEFAULT_BASE_URL,
        help="OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--api-key-env", default=None, help="Environment variable holding the API key"
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-output-tokens", type=int, default=4000)
    parser.add_argument("--max-evidence", type=int, default=30, help="Passages per topic")
    parser.add_argument("--evidence-chars", type=int, default=0, help="0 keeps full text")
    parser.add_argument("--max-citations", type=int, default=3)
    parser.add_argument("--budget-ratio", type=float, default=0.9)
    parser.add_argument("--max-repairs", type=int, default=1)
    parser.add_argument(
        "--structured-output", choices=("json_schema", "json_object"), default="json_schema"
    )
    parser.add_argument("--concurrency", type=int, choices=ALLOWED_CONCURRENCY, default=4)
    parser.add_argument("--topic", action="append", dest="topics_subset", default=None)
    parser.add_argument("--output-root", default=None, help="Run output root")
    parser.add_argument("--refresh-llm", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _api_key(name: str | None) -> tuple[str, str]:
    names = (name,) if name else DEFAULT_API_KEY_ENVS
    for candidate in names:
        value = os.getenv(candidate or "")
        if value:
            return candidate, value
    raise ReportRunError(
        f"no API key found; set one of {list(names)} or pass --api-key-env"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_dir = Path(args.input) if args.input else None
    handoff_manifest = Path(args.handoff_manifest) if args.handoff_manifest else None
    topics_path = Path(args.topics)
    if input_dir is not None and not input_dir.is_dir():
        print(f"input run directory not found: {input_dir}", file=sys.stderr)
        return 2
    if handoff_manifest is not None and not handoff_manifest.is_file():
        print(f"handoff manifest not found: {handoff_manifest}", file=sys.stderr)
        return 2
    if not topics_path.is_file():
        print(f"topics file not found: {topics_path}", file=sys.stderr)
        return 2

    try:
        team_id = validate_team_id(args.team_id)
        run_id = validate_run_id(args.run_id)
        run_desc = validate_run_desc(
            args.run_desc or f"{args.model} report generation"
        )
    except SubmissionValidationError as error:
        print(str(error), file=sys.stderr)
        return 2

    output_root = Path(
        args.output_root or default_project_cache_dir() / "experiments" / "reports"
    )

    try:
        settings = GenerationSettings(
            base_url=args.base_url,
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_output_tokens=args.max_output_tokens,
            seed=args.seed,
            max_repairs=args.max_repairs,
            max_citations=args.max_citations,
            max_passages=args.max_evidence,
            evidence_chars=args.evidence_chars,
            budget_ratio=args.budget_ratio,
            response_format_mode=args.structured_output,
        )
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2

    concurrency = 1 if args.dry_run else args.concurrency
    with ExitStack() as stack:
        try:
            store = ArtifactStore.from_env()
        except ArtifactStoreError as error:
            print(
                "durable artifact cache is required for resumable generation: "
                f"{error}",
                file=sys.stderr,
            )
            return 2
        stack.callback(lambda: None)

        workers: list[Any] = []
        observers: list[ObservedCacheStore] = []
        if args.dry_run:
            workers = [object()]
        else:
            try:
                key_name, api_key = _api_key(args.api_key_env)
            except ReportRunError as error:
                print(str(error), file=sys.stderr)
                return 2
            if not args.quiet:
                print(f"using API key from {key_name}")
            from openai import OpenAI

            from .llm_cache import CachedChatCompletions

            for _ in range(concurrency):
                client = OpenAI(
                    base_url=args.base_url, api_key=api_key, max_retries=0
                )
                observed = ObservedCacheStore(store)
                workers.append(CachedChatCompletions(client, observed))
                observers.append(observed)

        try:
            result = run_report_generation(
                input_dir=input_dir,
                handoff_manifest=handoff_manifest,
                topics_path=topics_path,
                output_root=output_root,
                team_id=team_id,
                run_id=run_id,
                run_desc=run_desc,
                settings=settings,
                completions=workers,
                store=store,
                topic_ids=tuple(args.topics_subset) if args.topics_subset else None,
                concurrency=concurrency,
                refresh=args.refresh_llm,
                allow_partial=args.allow_partial,
                dry_run=args.dry_run,
                cache_probes=[
                    (lambda observer=observer: bool(observer.last_read_hit))
                    for observer in observers
                ]
                or None,
                progress=None if args.quiet else lambda line: print(line),
            )
        except ReportRunError as error:
            print(str(error), file=sys.stderr)
            return 4 if "validation" in str(error) else 2
        except (EvidenceBundleError, ArtifactStoreError, ValueError) as error:
            print(str(error), file=sys.stderr)
            return 2

    print(
        f"{result.status}: {result.generated} generated, {result.cached} cached, "
        f"{result.failed} failed, {len(result.missing)} without evidence"
    )
    print(f"run directory: {result.run_dir}")
    if result.status == "dry_run":
        return 0
    return 0 if result.ok else 3


__all__ = [
    "ALLOWED_CONCURRENCY",
    "DEFAULT_TOPICS_PATH",
    "REPORT_RUN_SCHEMA_VERSION",
    "ReportRunError",
    "ReportRunResult",
    "build_parser",
    "main",
    "run_report_generation",
]


if __name__ == "__main__":
    raise SystemExit(main())
