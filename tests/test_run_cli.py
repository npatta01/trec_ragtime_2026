from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from trec_ragtime.run_cli import _format_error, main
from trec_ragtime.topics import TopicRequest


class ContextRetriever:
    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        pass


class RunCliTests(unittest.TestCase):
    def test_error_format_includes_recovery_notes(self) -> None:
        error = OSError("publication failed")
        error.add_note("recovery backup retained at C:\\safe\\backup.tmp")

        message = _format_error(error)

        self.assertIn("error: publication failed", message)
        self.assertIn("note: recovery backup retained", message)

    @patch("trec_ragtime.run_cli.BaselineRunner")
    @patch("trec_ragtime.run_cli.RagtimeRetriever.from_env")
    @patch("trec_ragtime.run_cli.load_topic_requests")
    def test_baseline_cli_wires_outputs_without_live_requests(
        self,
        load_topics: Mock,
        from_env: Mock,
        runner_class: Mock,
    ) -> None:
        request = TopicRequest(
            topic_id="2000",
            collection_id="ragtime/2/all",
            title="Title",
            background="Background",
            problem_statement="Problem statement",
            limit=5000,
        )
        load_topics.return_value = (request,)
        from_env.return_value = ContextRetriever()
        result = Mock()
        runner_class.return_value.run.return_value = result

        exit_code = main(
            [
                "--topics",
                "topics.jsonl",
                "--reports",
                "reports.jsonl",
                "--mlir",
                "mlir.run",
                "--artifacts",
                "audit.jsonl",
                "--team-id",
                "team",
                "--run-id",
                "baseline",
                "--local-index-path",
                "cache.duckdb",
                "--facets",
                "--max-documents",
                "50",
                "--hydrate-top-k",
                "25",
                "--evidence-documents",
                "25",
            ]
        )

        self.assertEqual(exit_code, 0)
        load_topics.assert_called_once_with("topics.jsonl")
        from_env.assert_called_once_with(
            local_index_path=Path("cache.duckdb").resolve()
        )
        runner_class.return_value.run.assert_called_once()
        result.write.assert_called_once_with(
            report_path="reports.jsonl",
            mlir_path="mlir.run",
            artifact_path="audit.jsonl",
        )
        kwargs = runner_class.call_args.kwargs
        self.assertEqual(kwargs["max_documents"], 50)
        self.assertEqual(kwargs["hydrate_top_k"], 25)
        self.assertEqual(kwargs["generator"].max_documents, 25)
        self.assertEqual(kwargs["planner"].max_facets, 8)

    @patch("trec_ragtime.run_cli.RagtimeRetriever.from_env")
    @patch("trec_ragtime.run_cli.load_query_plans")
    @patch("trec_ragtime.run_cli.load_topic_requests")
    def test_precomputed_plan_is_loaded_before_opening_retriever(
        self,
        load_topics: Mock,
        load_plans: Mock,
        from_env: Mock,
    ) -> None:
        load_topics.side_effect = ValueError("invalid topics")

        with self.assertRaises(SystemExit) as raised:
            main(
                [
                    "--topics",
                    "bad.jsonl",
                    "--reports",
                    "reports.jsonl",
                    "--mlir",
                    "mlir.run",
                    "--team-id",
                    "team",
                    "--run-id",
                    "baseline",
                    "--query-plan",
                    "plans.jsonl",
                ]
            )

        self.assertEqual(raised.exception.code, 2)
        load_plans.assert_not_called()
        from_env.assert_not_called()

    @patch("trec_ragtime.run_cli.RagtimeRetriever.from_env")
    @patch("trec_ragtime.run_cli.load_topic_requests")
    def test_output_collision_fails_before_opening_retriever(
        self,
        load_topics: Mock,
        from_env: Mock,
    ) -> None:
        load_topics.return_value = (
            TopicRequest(
                topic_id="2000",
                collection_id="ragtime/2/all",
                title="Title",
                background="Background",
                problem_statement="Problem statement",
                limit=5000,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            topic_path = Path(directory) / "topics.jsonl"
            topic_path.write_text("fixture", encoding="utf-8")

            with self.assertRaises(SystemExit) as raised:
                main(
                    [
                        "--topics",
                        str(topic_path),
                        "--reports",
                        str(topic_path),
                        "--mlir",
                        str(Path(directory) / "mlir.run"),
                        "--team-id",
                        "team",
                        "--run-id",
                        "baseline",
                    ]
                )

        self.assertEqual(raised.exception.code, 2)
        from_env.assert_not_called()

    @patch("trec_ragtime.run_cli.RagtimeRetriever.from_env")
    @patch("trec_ragtime.run_cli.load_topic_requests")
    def test_invalid_depths_fail_before_opening_retriever(
        self,
        load_topics: Mock,
        from_env: Mock,
    ) -> None:
        load_topics.return_value = (
            TopicRequest(
                topic_id="2000",
                collection_id="ragtime/2/all",
                title="Title",
                background="Background",
                problem_statement="Problem statement",
                limit=5000,
            ),
        )

        with self.assertRaises(SystemExit) as raised:
            main(
                [
                    "--topics",
                    "topics.jsonl",
                    "--reports",
                    "reports.jsonl",
                    "--mlir",
                    "mlir.run",
                    "--team-id",
                    "team",
                    "--run-id",
                    "baseline",
                    "--hydrate-top-k",
                    "10",
                    "--evidence-documents",
                    "20",
                ]
            )

        self.assertEqual(raised.exception.code, 2)
        from_env.assert_not_called()


if __name__ == "__main__":
    unittest.main()
