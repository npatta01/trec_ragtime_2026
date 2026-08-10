from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trec_ragtime import RagtimeRetriever
from trec_ragtime.baseline import ExtractiveReportGenerator
from trec_ragtime.models import (
    HuggingFaceDocument,
    HydrationResult,
    PipelineResult,
    SearchHit,
)
from trec_ragtime.query_planning import CompositeQueryPlanner, StaticQueryPlanner
from trec_ragtime.runner import BaselineRunResult, BaselineRunner, TopicRunArtifact
from trec_ragtime.submission import MLIRRun, ReportSentence, ReportSubmission
from trec_ragtime.topics import TopicRequest


def topic(topic_id: str, subject: str) -> TopicRequest:
    return TopicRequest(
        topic_id=topic_id,
        collection_id="ragtime/2/all",
        title=f"{subject} policy",
        background=f"A researcher is studying {subject} outcomes.",
        problem_statement=(
            f"Explain when the {subject} policy changed and its measured effects."
        ),
        limit=5000,
    )


class FakeClient:
    cache_path = None

    def __init__(self, documents_by_query: dict[str, tuple[str, ...]]) -> None:
        self.documents_by_query = documents_by_query
        self.calls: list[tuple[str, str, bool]] = []

    def search(self, query: str, *, pipeline: str, refresh: bool) -> PipelineResult:
        self.calls.append((query, pipeline, refresh))
        document_ids = self.documents_by_query[query]
        return PipelineResult(
            query=query,
            pipeline=pipeline,
            hits=tuple(
                SearchHit(rank=rank, document_id=document_id, score=1.0 / rank)
                for rank, document_id in enumerate(document_ids, start=1)
            ),
            processed=True,
            local_cache_hit=False,
            remote_cached=False,
            server_timestamp=None,
            cache_stored_at=None,
        )

    def close(self) -> None:
        pass


class FakeCorpus:
    backend_name = "fake"

    def __init__(self, documents: dict[str, HuggingFaceDocument]) -> None:
        self.documents = documents
        self.calls: list[tuple[str, ...]] = []

    def enrich_documents(self, document_ids, *, refresh: bool = False):
        requested = tuple(document_ids)
        self.calls.append(requested)
        return HydrationResult(
            documents=tuple(
                self.documents[document_id]
                for document_id in requested
                if document_id in self.documents
            ),
            unresolved_ids=tuple(
                document_id
                for document_id in requested
                if document_id not in self.documents
            ),
            failures=(),
        )

    get_many = enrich_documents

    def close(self) -> None:
        pass


def corpus_document(document_id: str, subject: str) -> HuggingFaceDocument:
    return HuggingFaceDocument(
        dataset="trec-ragtime/ragtime2",
        config="default",
        split="eng",
        document_id=document_id,
        text=(
            f"The {subject} policy changed in 2023, and a national study measured "
            f"improved {subject} outcomes after implementation."
        ),
        url=f"https://example.test/{document_id}",
        date="2024-01-01",
        local_cache_hit=None,
        cache_stored_at=None,
        source="test",
    )


class BaselineRunnerTests(unittest.TestCase):
    def build_runner(self, topics: tuple[TopicRequest, ...]) -> BaselineRunner:
        planner = CompositeQueryPlanner("ragtime2%10")
        plans = {request.topic_id: planner.plan(request)[0] for request in topics}
        client = FakeClient(
            {query.text: (f"doc-{topic_id}",) for topic_id, query in plans.items()}
        )
        corpus = FakeCorpus(
            {
                f"doc-{request.topic_id}": corpus_document(
                    f"doc-{request.topic_id}", request.title.split()[0]
                )
                for request in topics
            }
        )
        return BaselineRunner(
            retriever=RagtimeRetriever(client, corpus),
            planner=planner,
            generator=ExtractiveReportGenerator(
                max_documents=10,
                min_sentence_characters=20,
            ),
            team_id="team",
            run_id="baseline",
            run_desc="extractive multilingual baseline",
            max_documents=10,
            hydrate_top_k=10,
        )

    def test_runs_every_topic_and_builds_paired_outputs_and_audit(self):
        topics = (topic("2001", "alpha"), topic("2000", "beta"))
        progress: list[tuple[int, int, str]] = []

        result = self.build_runner(topics).run(
            topics,
            progress=lambda index, total, topic_id: progress.append(
                (index, total, topic_id)
            ),
        )

        self.assertEqual(
            [report.metadata.topic_id for report in result.reports],
            ["2001", "2000"],
        )
        self.assertEqual([run.topic_id for run in result.mlir_runs], ["2001", "2000"])
        self.assertEqual(progress, [(1, 2, "2001"), (2, 2, "2000")])
        self.assertTrue(all(artifact.evidence for artifact in result.artifacts))
        self.assertTrue(
            all(report.responses[0].citations for report in result.reports)
        )

    def test_writes_complete_deterministic_report_mlir_and_artifact_files(self):
        topics = (topic("2001", "alpha"), topic("2000", "beta"))
        result = self.build_runner(topics).run(topics)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "reports.jsonl"
            mlir_path = root / "mlir.run"
            artifact_path = root / "audit.jsonl"
            result.write(
                report_path=report_path,
                mlir_path=mlir_path,
                artifact_path=artifact_path,
            )

            reports = [
                json.loads(line)
                for line in report_path.read_text(encoding="utf-8").splitlines()
            ]
            mlir_lines = mlir_path.read_text(encoding="utf-8").splitlines()
            artifacts = [
                json.loads(line)
                for line in artifact_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(
            [value["metadata"]["topic_id"] for value in reports],
            ["2000", "2001"],
        )
        self.assertEqual([line.split()[0] for line in mlir_lines], ["2000", "2001"])
        self.assertEqual([value["topic_id"] for value in artifacts], ["2001", "2000"])
        self.assertIn("provenance", artifacts[0]["fused_hits"][0])

    def test_runner_configuration_fails_before_search(self):
        request = topic("2000", "alpha")
        runner = self.build_runner((request,))

        for changes in (
            {"rrf_k": True},
            {"rrf_k": 0},
            {"max_documents": 1001},
            {"hydrate_top_k": 0},
            {"hydrate_top_k": 11},
            {"refresh_search": 1},
        ):
            values = {
                "retriever": runner.retriever,
                "planner": runner.planner,
                "generator": runner.generator,
                "team_id": runner.team_id,
                "run_id": runner.run_id,
                "run_desc": runner.run_desc,
                "rrf_k": runner.rrf_k,
                "max_documents": runner.max_documents,
                "hydrate_top_k": runner.hydrate_top_k,
                "refresh_search": runner.refresh_search,
                "refresh_documents": runner.refresh_documents,
            }
            values.update(changes)
            with self.subTest(changes=changes):
                with self.assertRaises((TypeError, ValueError)):
                    BaselineRunner(**values)

    def test_result_requires_matching_run_ids_and_cited_sentences(self):
        topics = (topic("2000", "alpha"),)
        result = self.build_runner(topics).run(topics)
        mismatched_mlir = MLIRRun.from_scores(
            topic_id="2000",
            run_id="different",
            scores={"doc-2000": 1.0},
        )
        with self.assertRaisesRegex(ValueError, "run IDs do not match"):
            BaselineRunResult(
                topics=result.topics,
                reports=result.reports,
                mlir_runs=(mismatched_mlir,),
                artifacts=result.artifacts,
            )

        uncited_report = ReportSubmission(
            metadata=result.reports[0].metadata,
            responses=(ReportSentence("Uncited sentence.", {}),),
            references=(),
        )
        with self.assertRaisesRegex(ValueError, "cite every response sentence"):
            BaselineRunResult(
                topics=result.topics,
                reports=(uncited_report,),
                mlir_runs=result.mlir_runs,
                artifacts=result.artifacts,
            )

        empty_report = ReportSubmission.build(
            metadata=result.reports[0].metadata,
            responses=(),
        )
        with self.assertRaisesRegex(ValueError, "at least one sentence"):
            BaselineRunResult(
                topics=result.topics,
                reports=(empty_report,),
                mlir_runs=result.mlir_runs,
                artifacts=result.artifacts,
            )

    def test_result_and_artifact_copy_caller_owned_collections(self):
        topics = (topic("2000", "alpha"),)
        result = self.build_runner(topics).run(topics)
        source_artifact = result.artifacts[0]
        mutable_queries = list(source_artifact.queries)
        mutable_evidence = list(source_artifact.evidence)
        artifact = TopicRunArtifact(
            topic=source_artifact.topic,
            queries=mutable_queries,  # type: ignore[arg-type]
            retrieval=source_artifact.retrieval,
            evidence=mutable_evidence,  # type: ignore[arg-type]
        )
        mutable_topics = list(result.topics)
        mutable_reports = list(result.reports)
        mutable_mlir = list(result.mlir_runs)
        mutable_artifacts = [artifact]
        copied = BaselineRunResult(
            topics=mutable_topics,  # type: ignore[arg-type]
            reports=mutable_reports,  # type: ignore[arg-type]
            mlir_runs=mutable_mlir,  # type: ignore[arg-type]
            artifacts=mutable_artifacts,  # type: ignore[arg-type]
        )

        mutable_queries.clear()
        mutable_evidence.clear()
        mutable_topics.clear()
        mutable_reports.clear()
        mutable_mlir.clear()
        mutable_artifacts.clear()

        self.assertEqual(len(artifact.queries), 1)
        self.assertTrue(artifact.evidence)
        self.assertEqual(len(copied.topics), 1)
        self.assertEqual(len(copied.reports), 1)
        self.assertEqual(len(copied.mlir_runs), 1)
        self.assertEqual(len(copied.artifacts), 1)

    def test_duplicate_topics_fail_before_search(self):
        request = topic("2000", "alpha")
        runner = self.build_runner((request, request))

        with self.assertRaisesRegex(ValueError, "unique topic IDs"):
            runner.run((request, request))

        self.assertEqual(runner.retriever.client.calls, [])

    def test_missing_static_plan_fails_before_any_search(self):
        topics = (topic("2000", "alpha"), topic("2001", "beta"))
        configured = self.build_runner(topics)
        planner = StaticQueryPlanner(
            {"2000": CompositeQueryPlanner().plan(topics[0])}
        )
        runner = BaselineRunner(
            retriever=configured.retriever,
            planner=planner,
            generator=configured.generator,
            team_id=configured.team_id,
            run_id=configured.run_id,
            run_desc=configured.run_desc,
            max_documents=configured.max_documents,
            hydrate_top_k=configured.hydrate_top_k,
        )

        with self.assertRaisesRegex(ValueError, "no precomputed query plan"):
            runner.run(topics)

        self.assertEqual(runner.retriever.client.calls, [])

    def test_refuses_to_overwrite_one_output_with_another(self):
        topics = (topic("2000", "alpha"),)
        result = self.build_runner(topics).run(topics)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "same-output"

            with self.assertRaisesRegex(ValueError, "must be distinct"):
                result.write(report_path=path, mlir_path=path)

            self.assertFalse(path.exists())

    def test_stages_all_outputs_before_replacing_destinations(self):
        topics = (topic("2000", "alpha"),)
        result = self.build_runner(topics).run(topics)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "reports.jsonl"
            mlir_path = root / "mlir.run"
            report_path.write_text("old report", encoding="utf-8")
            mlir_path.write_text("old mlir", encoding="utf-8")

            with patch(
                "trec_ragtime.runner.write_mlir_run",
                side_effect=OSError("simulated staging failure"),
            ):
                with self.assertRaisesRegex(OSError, "staging failure"):
                    result.write(report_path=report_path, mlir_path=mlir_path)

            self.assertEqual(report_path.read_text(encoding="utf-8"), "old report")
            self.assertEqual(mlir_path.read_text(encoding="utf-8"), "old mlir")
            self.assertEqual(list(root.glob(".*.tmp")), [])

    def test_publication_failure_restores_the_previous_output_set(self):
        topics = (topic("2000", "alpha"),)
        result = self.build_runner(topics).run(topics)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "reports.jsonl"
            mlir_path = root / "mlir.run"
            report_path.write_text("old report", encoding="utf-8")
            mlir_path.write_text("old mlir", encoding="utf-8")
            original_replace = Path.replace
            replace_count = 0

            def fail_on_second_publication(source: Path, target: Path) -> Path:
                nonlocal replace_count
                replace_count += 1
                if replace_count == 6:
                    raise OSError("simulated publication failure")
                return original_replace(source, target)

            with patch.object(Path, "replace", new=fail_on_second_publication):
                with self.assertRaisesRegex(OSError, "publication failure"):
                    result.write(report_path=report_path, mlir_path=mlir_path)

            self.assertEqual(report_path.read_text(encoding="utf-8"), "old report")
            self.assertEqual(mlir_path.read_text(encoding="utf-8"), "old mlir")
            self.assertEqual(list(root.glob(".*.tmp")), [])

    def test_failed_restore_preserves_the_recovery_backup(self):
        topics = (topic("2000", "alpha"),)
        result = self.build_runner(topics).run(topics)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "reports.jsonl"
            mlir_path = root / "mlir.run"
            report_path.write_text("old report", encoding="utf-8")
            mlir_path.write_text("old mlir", encoding="utf-8")
            original_replace = Path.replace
            replace_count = 0

            def fail_publication_and_one_restore(
                source: Path,
                target: Path,
            ) -> Path:
                nonlocal replace_count
                replace_count += 1
                if replace_count == 6:
                    raise OSError("simulated publication failure")
                if replace_count == 7:
                    raise OSError("simulated restore failure")
                return original_replace(source, target)

            with patch.object(
                Path,
                "replace",
                new=fail_publication_and_one_restore,
            ):
                with self.assertRaisesRegex(OSError, "publication failure") as caught:
                    result.write(report_path=report_path, mlir_path=mlir_path)

            self.assertEqual(report_path.read_text(encoding="utf-8"), "old report")
            self.assertFalse(mlir_path.exists())
            recovery_files = list(root.glob(".mlir.run.*.tmp"))
            self.assertEqual(len(recovery_files), 1)
            self.assertEqual(
                recovery_files[0].read_text(encoding="utf-8"),
                "old mlir",
            )
            self.assertTrue(
                any(
                    "could not be restored" in note
                    for note in caught.exception.__notes__
                )
            )


if __name__ == "__main__":
    unittest.main()
