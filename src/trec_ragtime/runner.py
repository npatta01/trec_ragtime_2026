"""End-to-end baseline orchestration for RAGTIME report and MLIR runs."""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from tempfile import NamedTemporaryFile

from .baseline import EvidenceSentence, ExtractiveReportGenerator
from .fusion import RetrievalQuery
from .multiretriever import MultiQueryRetrievalResult, MultiQueryRetriever
from .query_planning import QueryPlanner
from .retriever import RagtimeRetriever
from .submission import (
    MLIRRun,
    ReportMetadata,
    ReportSubmission,
    validate_mlir_runs,
    validate_report_run,
    write_mlir_run,
    write_report_jsonl,
)
from .topics import TopicRequest


ProgressCallback = Callable[[int, int, str], None]


def validate_baseline_parameters(
    *,
    generator: ExtractiveReportGenerator,
    team_id: str,
    run_id: str,
    run_desc: str,
    rrf_k: float,
    max_documents: int,
    hydrate_top_k: int,
    refresh_search: bool,
    refresh_documents: bool,
) -> float:
    """Validate run controls without opening the corpus or search client."""

    if not isinstance(generator, ExtractiveReportGenerator):
        raise TypeError("generator must be an ExtractiveReportGenerator")
    ReportMetadata(
        team_id=team_id,
        topic_id="validation",
        run_id=run_id,
        run_desc=run_desc,
    )
    if isinstance(rrf_k, bool) or not isinstance(rrf_k, (int, float)):
        raise ValueError("rrf_k must be a positive finite number")
    try:
        numeric_rrf_k = float(rrf_k)
    except OverflowError as error:
        raise ValueError("rrf_k must be a positive finite number") from error
    if not math.isfinite(numeric_rrf_k) or numeric_rrf_k <= 0:
        raise ValueError("rrf_k must be a positive finite number")
    if (
        isinstance(max_documents, bool)
        or not isinstance(max_documents, int)
        or not 1 <= max_documents <= 1000
    ):
        raise ValueError("max_documents must be an integer from 1 to 1000")
    if (
        isinstance(hydrate_top_k, bool)
        or not isinstance(hydrate_top_k, int)
        or not 1 <= hydrate_top_k <= max_documents
    ):
        raise ValueError(
            "hydrate_top_k must be a positive integer no larger than "
            "max_documents"
        )
    if hydrate_top_k != generator.max_documents:
        raise ValueError(
            "hydrate_top_k must equal the generator's max_documents so "
            "every hydrated document is eligible evidence"
        )
    for name, value in (
        ("refresh_search", refresh_search),
        ("refresh_documents", refresh_documents),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean")
    return numeric_rrf_k


@dataclass(frozen=True, slots=True)
class TopicRunArtifact:
    """Auditable intermediate state for one report request."""

    topic: TopicRequest
    queries: tuple[RetrievalQuery, ...]
    retrieval: MultiQueryRetrievalResult
    evidence: tuple[EvidenceSentence, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.topic, TopicRequest):
            raise TypeError("topic must be a TopicRequest")
        if not isinstance(self.retrieval, MultiQueryRetrievalResult):
            raise TypeError("retrieval must be a MultiQueryRetrievalResult")
        try:
            queries = tuple(self.queries)
            evidence = tuple(self.evidence)
        except TypeError as error:
            raise TypeError("queries and evidence must be iterable") from error
        if not queries or any(
            not isinstance(query, RetrievalQuery) for query in queries
        ):
            raise TypeError("queries must contain RetrievalQuery values")
        if any(not isinstance(item, EvidenceSentence) for item in evidence):
            raise TypeError("evidence must contain EvidenceSentence values")
        if queries != self.retrieval.fused_ranking.queries:
            raise ValueError("artifact queries do not match fused retrieval queries")
        retrieved_ids = {item.document_id for item in self.retrieval.documents}
        if any(item.document_id not in retrieved_ids for item in evidence):
            raise ValueError("artifact evidence refers to an unhydrated document")
        object.__setattr__(self, "queries", queries)
        object.__setattr__(self, "evidence", evidence)

    def as_record(self) -> dict[str, object]:
        return {
            "topic_id": self.topic.topic_id,
            "queries": [
                {
                    "query_id": query.query_id,
                    "text": query.text,
                    "pipeline": query.pipeline,
                    "language": query.language,
                    "weight": query.weight,
                }
                for query in self.queries
            ],
            "fused_hits": [
                {
                    "rank": hit.rank,
                    "document_id": hit.document_id,
                    "score": hit.score,
                    "provenance": [
                        {
                            "query_id": item.query_id,
                            "pipeline": item.pipeline,
                            "language": item.language,
                            "source_rank": item.source_rank,
                            "source_score": item.source_score,
                            "query_weight": item.query_weight,
                            "contribution": item.contribution,
                        }
                        for item in hit.provenance
                    ],
                }
                for hit in self.retrieval.fused_ranking.hits
            ],
            "hydrated_hit_count": self.retrieval.hydrated_hit_count,
            "unresolved_ids": list(self.retrieval.unresolved_ids),
            "missing_translation_ids": list(
                self.retrieval.missing_translation_ids
            ),
            "evidence": [item.as_record() for item in self.evidence],
        }


@dataclass(frozen=True, slots=True)
class BaselineRunResult:
    """Complete paired report-generation and MLIR outputs."""

    topics: tuple[TopicRequest, ...]
    reports: tuple[ReportSubmission, ...]
    mlir_runs: tuple[MLIRRun, ...]
    artifacts: tuple[TopicRunArtifact, ...]

    def __post_init__(self) -> None:
        try:
            topics = tuple(self.topics)
            reports = tuple(self.reports)
            mlir_runs = tuple(self.mlir_runs)
            artifacts = tuple(self.artifacts)
        except TypeError as error:
            raise TypeError("run-result collections must be iterable") from error
        if any(not isinstance(topic, TopicRequest) for topic in topics):
            raise TypeError("topics must contain TopicRequest values")
        if any(not isinstance(report, ReportSubmission) for report in reports):
            raise TypeError("reports must contain ReportSubmission values")
        if any(not isinstance(run, MLIRRun) for run in mlir_runs):
            raise TypeError("mlir_runs must contain MLIRRun values")
        if any(not isinstance(item, TopicRunArtifact) for item in artifacts):
            raise TypeError("artifacts must contain TopicRunArtifact values")

        topic_ids = tuple(topic.topic_id for topic in topics)
        if not topic_ids or len(topic_ids) != len(set(topic_ids)):
            raise ValueError("topics must be non-empty and unique")
        if tuple(report.metadata.topic_id for report in reports) != topic_ids:
            raise ValueError("reports do not match topics in source order")
        if tuple(run.topic_id for run in mlir_runs) != topic_ids:
            raise ValueError("MLIR runs do not match topics in source order")
        if tuple(item.topic.topic_id for item in artifacts) != topic_ids:
            raise ValueError("artifacts do not match topics in source order")
        if any(
            report.metadata.run_id != mlir.run_id
            for report, mlir in zip(reports, mlir_runs, strict=True)
        ):
            raise ValueError("paired report and MLIR run IDs do not match")
        if any(
            not response.citations
            for report in reports
            for response in report.responses
        ):
            raise ValueError("baseline reports must cite every response sentence")
        if any(not report.responses for report in reports):
            raise ValueError("baseline reports must contain at least one sentence")
        object.__setattr__(self, "topics", topics)
        object.__setattr__(self, "reports", reports)
        object.__setattr__(self, "mlir_runs", mlir_runs)
        object.__setattr__(self, "artifacts", artifacts)

    def write(
        self,
        *,
        report_path: str | PathLike[str],
        mlir_path: str | PathLike[str],
        artifact_path: str | PathLike[str] | None = None,
    ) -> None:
        """Validate complete topic coverage before writing any run output."""

        output_paths = [Path(report_path).expanduser(), Path(mlir_path).expanduser()]
        if artifact_path is not None:
            output_paths.append(Path(artifact_path).expanduser())
        resolved_paths = [path.resolve() for path in output_paths]
        if len(resolved_paths) != len(set(resolved_paths)):
            raise ValueError("report, MLIR, and artifact output paths must be distinct")
        for path in output_paths:
            if path.exists() and path.is_dir():
                raise ValueError(f"output path points to a directory: {path}")
            if not path.parent.is_dir():
                raise ValueError(
                    f"output parent directory does not exist: {path.parent}"
                )

        topic_limits = {topic.topic_id: topic.limit for topic in self.topics}
        topic_ids = tuple(topic_limits)
        validate_report_run(self.reports, topic_limits=topic_limits)
        validate_mlir_runs(self.mlir_runs, expected_topic_ids=topic_ids)

        artifact_payload: str | None = None
        if artifact_path is not None:
            lines = [
                json.dumps(
                    artifact.as_record(),
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                for artifact in self.artifacts
            ]
            artifact_payload = "\n".join(lines) + "\n"

        # Finish every potentially failing serialization/write against an
        # adjacent temporary file before replacing any requested destination.
        staged: list[tuple[Path, Path]] = []
        try:
            report_destination, mlir_destination = output_paths[:2]
            report_stage = _reserve_staging_path(report_destination)
            staged.append((report_stage, report_destination))
            write_report_jsonl(
                report_stage,
                self.reports,
                topic_limits=topic_limits,
            )
            mlir_stage = _reserve_staging_path(mlir_destination)
            staged.append((mlir_stage, mlir_destination))
            write_mlir_run(
                mlir_stage,
                self.mlir_runs,
                expected_topic_ids=topic_ids,
            )
            if artifact_payload is not None:
                artifact_destination = output_paths[2]
                artifact_stage = _reserve_staging_path(artifact_destination)
                staged.append((artifact_stage, artifact_destination))
                artifact_stage.write_text(artifact_payload, encoding="utf-8")

            _publish_staged_outputs(staged)
        finally:
            for stage, _ in staged:
                _cleanup_temporary(stage)


def _reserve_staging_path(destination: Path) -> Path:
    """Reserve a closed temporary file adjacent to one output destination."""

    with NamedTemporaryFile(
        mode="wb",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as handle:
        return Path(handle.name)


def _publish_staged_outputs(staged: list[tuple[Path, Path]]) -> None:
    """Publish an output set with best-effort rollback on replacement failure."""

    reserved_backups: list[Path] = []
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    publication_succeeded = False
    try:
        for _, destination in staged:
            if destination.exists() or destination.is_symlink():
                backup = _reserve_staging_path(destination)
                reserved_backups.append(backup)
                destination.replace(backup)
                backups.append((backup, destination))

        for stage, destination in staged:
            stage.replace(destination)
            published.append(destination)
        publication_succeeded = True
    except BaseException as error:
        rollback_errors: list[OSError] = []
        for destination in reversed(published):
            try:
                destination.unlink(missing_ok=True)
            except OSError as rollback_error:
                rollback_errors.append(rollback_error)
        for backup, destination in reversed(backups):
            try:
                backup.replace(destination)
            except OSError as rollback_error:
                rollback_errors.append(rollback_error)
        if rollback_errors:
            note = (
                "one or more output destinations could not be restored after "
                "publication failed"
            )
            retained_backups = [
                str(backup)
                for backup, _ in backups
                if backup.exists() or backup.is_symlink()
            ]
            if retained_backups:
                note += "; recovery backups retained at: " + ", ".join(
                    retained_backups
                )
            error.add_note(note)
        raise
    finally:
        backups_with_data = {backup for backup, _ in backups}
        for backup in reserved_backups:
            if publication_succeeded or backup not in backups_with_data:
                _cleanup_temporary(backup)


def _cleanup_temporary(path: Path) -> None:
    """Remove a staging file without masking publication or rollback results."""

    try:
        path.unlink(missing_ok=True)
    except OSError as error:
        try:
            warnings.warn(
                f"unable to remove temporary run output {path}: "
                f"{error.__class__.__name__}",
                RuntimeWarning,
                stacklevel=2,
            )
        except Exception:
            # Warning filters or hooks must not replace the publication result.
            pass


@dataclass(frozen=True, slots=True)
class BaselineRunner:
    """Run the same query, fusion, hydration, and generation policy per topic."""

    retriever: RagtimeRetriever
    planner: QueryPlanner
    generator: ExtractiveReportGenerator
    team_id: str
    run_id: str
    run_desc: str
    rrf_k: float = 60.0
    max_documents: int = 1000
    hydrate_top_k: int = 30
    refresh_search: bool = False
    refresh_documents: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.retriever, RagtimeRetriever):
            raise TypeError("retriever must be a RagtimeRetriever")
        if not hasattr(self.planner, "plan") or not callable(self.planner.plan):
            raise TypeError("planner must provide a callable plan method")
        numeric_rrf_k = validate_baseline_parameters(
            generator=self.generator,
            team_id=self.team_id,
            run_id=self.run_id,
            run_desc=self.run_desc,
            rrf_k=self.rrf_k,
            max_documents=self.max_documents,
            hydrate_top_k=self.hydrate_top_k,
            refresh_search=self.refresh_search,
            refresh_documents=self.refresh_documents,
        )
        object.__setattr__(self, "rrf_k", numeric_rrf_k)

    def run(
        self,
        topics: Iterable[TopicRequest],
        *,
        progress: ProgressCallback | None = None,
    ) -> BaselineRunResult:
        topic_values = tuple(topics)
        if not topic_values:
            raise ValueError("topics must not be empty")
        if any(not isinstance(topic, TopicRequest) for topic in topic_values):
            raise TypeError("topics must contain TopicRequest values")
        if progress is not None and not callable(progress):
            raise TypeError("progress must be callable or None")

        topic_ids = tuple(topic.topic_id for topic in topic_values)
        if len(topic_ids) != len(set(topic_ids)):
            raise ValueError("topics must contain unique topic IDs")

        # Validate the complete plan before the first participant-API request.
        planned_topics: list[tuple[TopicRequest, tuple[RetrievalQuery, ...]]] = []
        for topic in topic_values:
            raw_queries = self.planner.plan(topic)
            if isinstance(raw_queries, (str, bytes)):
                raise TypeError(
                    f"query plan for topic {topic.topic_id!r} must be an iterable"
                )
            try:
                queries = tuple(raw_queries)
            except TypeError as error:
                raise TypeError(
                    f"query plan for topic {topic.topic_id!r} must be an iterable"
                ) from error
            if not queries or any(
                not isinstance(query, RetrievalQuery) for query in queries
            ):
                raise TypeError(
                    f"query plan for topic {topic.topic_id!r} must contain "
                    "RetrievalQuery values"
                )
            query_ids = tuple(query.query_id for query in queries)
            if len(query_ids) != len(set(query_ids)):
                raise ValueError(
                    f"query plan for topic {topic.topic_id!r} contains "
                    "duplicate query IDs"
                )
            planned_topics.append((topic, queries))

        multi_retriever = MultiQueryRetriever(self.retriever)
        reports: list[ReportSubmission] = []
        mlir_runs: list[MLIRRun] = []
        artifacts: list[TopicRunArtifact] = []
        total = len(topic_values)

        for index, (topic, queries) in enumerate(planned_topics, start=1):
            retrieval = multi_retriever.retrieve(
                queries,
                rrf_k=self.rrf_k,
                max_documents=self.max_documents,
                hydrate_top_k=self.hydrate_top_k,
                refresh_search=self.refresh_search,
                refresh_documents=self.refresh_documents,
            )
            retrieval.require_report_ready()
            generated = self.generator.generate(
                topic,
                retrieval,
                team_id=self.team_id,
                run_id=self.run_id,
                run_desc=self.run_desc,
            )
            mlir = MLIRRun.from_scores(
                topic_id=topic.topic_id,
                run_id=self.run_id,
                scores=(
                    (hit.document_id, hit.score)
                    for hit in retrieval.fused_ranking.hits
                ),
                max_hits=self.max_documents,
            )
            reports.append(generated.report)
            mlir_runs.append(mlir)
            artifacts.append(
                TopicRunArtifact(
                    topic=topic,
                    queries=queries,
                    retrieval=retrieval,
                    evidence=generated.evidence,
                )
            )
            if progress is not None:
                progress(index, total, topic.topic_id)

        return BaselineRunResult(
            topics=topic_values,
            reports=tuple(reports),
            mlir_runs=tuple(mlir_runs),
            artifacts=tuple(artifacts),
        )


__all__ = [
    "BaselineRunResult",
    "BaselineRunner",
    "ProgressCallback",
    "TopicRunArtifact",
    "validate_baseline_parameters",
]
