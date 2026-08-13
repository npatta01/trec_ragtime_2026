from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

from trec_ragtime.report_cache import GenerationSettings
from trec_ragtime.report_runner import (
    REPORT_RUN_SCHEMA_VERSION,
    ReportRunError,
    build_parser,
    main,
    run_report_generation,
)
from trec_ragtime.report_submission import validate_submission_file
from tests.test_report_cache import MemoryArtifactStore
from tests.test_report_generation import ScriptedCompletions, completion, draft, line

TEAM_ID = "PLACEHOLDER-TEAM"
RUN_ID = "test-run-1"


def good_draft() -> Any:
    return completion(
        draft(
            line("The ban took effect in 2019.", ["S1"]),
            line("Youth vaping prevalence fell sharply.", ["S2"], ["F2"]),
        )
    )


def bad_draft() -> Any:
    return completion(draft(line("An invented claim.", ["S9"])))


class EndlessCompletions:
    """Returns the same response for every request, without a script limit."""

    def __init__(self, factory: Any) -> None:
        self.factory = factory
        self.calls = 0

    def create(self, *, refresh: bool = False, **request: Any) -> Any:
        self.calls += 1
        return self.factory()


class TripwireCompletions:
    def create(self, *, refresh: bool = False, **request: Any) -> Any:
        raise AssertionError("provider called")


class RunnerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)
        self.store = MemoryArtifactStore()
        self.output_root = self.root / "runs"

    def write_topics(self, *topic_ids: str, limit: int = 5000) -> Path:
        path = self.root / "topics.jsonl"
        path.write_text(
            "\n".join(
                json.dumps(
                    {
                        "topic_id": topic_id,
                        "collection_id": "ragtime/2/all",
                        "title": f"Topic {topic_id}",
                        "problem_statement": f"A statement for {topic_id}.",
                        "background": "A legislative aide needs this.",
                        "limit": limit,
                    }
                )
                for topic_id in topic_ids
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def write_run(self, *topic_ids: str, passages: int = 3) -> Path:
        run_dir = self.root / "baseline-c-test"
        run_dir.mkdir(exist_ok=True)
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "query_baseline_v1",
                    "run_id": "baseline-c-test",
                    "status": "complete",
                    "corpus": {"dataset": "trec-ragtime/ragtime2", "revision": "c99bf3e"},
                }
            ),
            encoding="utf-8",
        )
        records = []
        for topic_id in topic_ids:
            records.append(
                {
                    "topic_id": topic_id,
                    "status": "success",
                    "queries": ["facet one", "facet two"],
                    "candidates": [
                        {
                            "document_id": f"{topic_id}-doc-{index}",
                            "fused_score": 1.0 / index,
                            "rank": index,
                            "source_query_indices": [1] if index % 2 else [2],
                            "hydrated": True,
                            "text": f"Source text {index}.",
                            "report_text": f"English evidence body {index} for {topic_id}.",
                            "english_text": None,
                            "language": "eng",
                            "url": f"https://example.test/{index}",
                            "date": "2022-09-21T06:34:00.000Z",
                        }
                        for index in range(1, passages + 1)
                    ],
                }
            )
        (run_dir / "topics.jsonl").write_text(
            "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
        )
        return run_dir

    def settings(self, **overrides: Any) -> GenerationSettings:
        values: dict[str, Any] = {
            "base_url": "https://api.example.test/v1",
            "model": "test-model",
            "temperature": 0.0,
            "top_p": None,
            "max_output_tokens": 4000,
            "seed": 7,
            "max_repairs": 0,
            "max_citations": 3,
            "max_passages": 3,
            "evidence_chars": 0,
            "budget_ratio": 0.9,
        }
        values.update(overrides)
        return GenerationSettings(**values)

    def generate(self, fixture_topics: tuple[str, ...] = ("2000", "2001"), **overrides: Any):
        completions = overrides.pop("completions", None)
        if completions is None:
            completions = [EndlessCompletions(good_draft)]
        input_dir = overrides.pop("input_dir", None)
        if input_dir is None:
            input_dir = self.write_run(*fixture_topics)
        topics_path = overrides.pop("topics_path", None)
        if topics_path is None:
            topics_path = self.write_topics(*fixture_topics)
        return run_report_generation(
            input_dir=input_dir,
            topics_path=topics_path,
            output_root=overrides.pop("output_root", self.output_root),
            team_id=TEAM_ID,
            run_id=overrides.pop("run_id", RUN_ID),
            run_desc="offline test run",
            settings=overrides.pop("settings", self.settings()),
            completions=completions,
            store=overrides.pop("store", self.store),
            clock=overrides.pop("clock", lambda: 1_700_000_000.0),
            **overrides,
        )


class RunTests(RunnerTestCase):
    def test_complete_run_writes_a_valid_submission(self) -> None:
        result = self.generate()
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.generated, 2)
        self.assertEqual(result.failed, 0)

        submission = result.run_dir / "submission.jsonl"
        self.assertTrue(submission.is_file())
        report = validate_submission_file(
            submission, topics={"2000": 5000, "2001": 5000}, team_id=TEAM_ID, run_id=RUN_ID
        )
        self.assertTrue(report.ok, report.violations)
        self.assertEqual(report.topic_ids, ("2000", "2001"))

    def test_submission_records_match_the_official_shape(self) -> None:
        result = self.generate(("2000",))
        record = json.loads((result.run_dir / "submission.jsonl").read_text().strip())
        self.assertEqual(set(record), {"metadata", "responses", "references"})
        self.assertEqual(record["metadata"]["team_id"], TEAM_ID)
        self.assertEqual(record["metadata"]["topic_id"], "2000")
        self.assertEqual(record["references"], ["2000-doc-1", "2000-doc-2"])

    def test_run_directory_carries_every_artifact(self) -> None:
        result = self.generate()
        for name in ("reports.jsonl", "lineage.jsonl", "submission.jsonl", "manifest.json"):
            with self.subTest(name=name):
                self.assertTrue((result.run_dir / name).is_file())

    def test_manifest_is_sanitized(self) -> None:
        result = self.generate()
        raw = (result.run_dir / "manifest.json").read_text()
        manifest = json.loads(raw)
        self.assertEqual(manifest["schema_version"], REPORT_RUN_SCHEMA_VERSION)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["team_id"], TEAM_ID)
        self.assertEqual(len(manifest["generator_fingerprint"]), 64)
        self.assertNotIn("English evidence body", raw)
        self.assertNotIn("api_key", raw.lower())
        self.assertNotIn("REPORT REQUEST", raw)

    def test_lineage_maps_sentences_to_handles_and_documents(self) -> None:
        result = self.generate(("2000",))
        lineage = json.loads((result.run_dir / "lineage.jsonl").read_text().strip())
        self.assertEqual(lineage["topic_id"], "2000")
        self.assertEqual(lineage["handles"]["S1"], "2000-doc-1")
        first = lineage["sentences"][0]
        self.assertEqual(first["citations"][0]["handle"], "S1")
        self.assertEqual(first["citations"][0]["document_id"], "2000-doc-1")

    def test_topics_are_emitted_once_in_topic_file_order(self) -> None:
        result = self.generate(("2000", "2001"))
        lines = (result.run_dir / "submission.jsonl").read_text().strip().split("\n")
        topic_ids = [json.loads(line)["metadata"]["topic_id"] for line in lines]
        self.assertEqual(topic_ids, ["2000", "2001"])
        self.assertEqual(len(topic_ids), len(set(topic_ids)))

    def test_topic_subset_limits_the_run(self) -> None:
        result = self.generate(("2000", "2001"), topic_ids=("2001",))
        lines = (result.run_dir / "submission.jsonl").read_text().strip().split("\n")
        self.assertEqual([json.loads(line)["metadata"]["topic_id"] for line in lines], ["2001"])
        self.assertEqual(result.status, "complete")

    def test_warm_rerun_uses_the_cache_without_calling_the_provider(self) -> None:
        self.generate()
        result = self.generate(run_id="test-run-2", completions=[TripwireCompletions()])
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.cached, 2)
        self.assertEqual(result.generated, 0)

    def test_warm_rerun_reproduces_identical_submission_bytes(self) -> None:
        first = self.generate()
        original = (first.run_dir / "submission.jsonl").read_bytes()
        second = self.generate(completions=[TripwireCompletions()])
        self.assertEqual((second.run_dir / "submission.jsonl").read_bytes(), original)

    def test_retagging_a_run_reuses_cached_reports(self) -> None:
        self.generate()
        # A different run_id must not invalidate generated reports.
        result = self.generate(run_id="retagged", completions=[TripwireCompletions()])
        self.assertEqual(result.cached, 2)

    def test_failed_topic_leaves_the_run_incomplete_without_a_submission(self) -> None:
        result = self.generate(completions=[EndlessCompletions(bad_draft)], allow_partial=False)
        self.assertEqual(result.status, "incomplete")
        self.assertEqual(result.failed, 2)
        self.assertFalse((result.run_dir / "submission.jsonl").exists())
        manifest = json.loads((result.run_dir / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "incomplete")

    def test_failures_file_records_reason_codes(self) -> None:
        result = self.generate(completions=[EndlessCompletions(bad_draft)])
        failures = json.loads((result.run_dir / "failures.json").read_text())
        self.assertEqual(sorted(failures["failed_topics"]), ["2000", "2001"])
        self.assertTrue(failures["details"]["2000"])

    def test_one_topic_failure_does_not_abort_the_run(self) -> None:
        class Mixed:
            def __init__(self) -> None:
                self.calls = 0

            def create(self, *, refresh: bool = False, **request: Any) -> Any:
                self.calls += 1
                body = "".join(item["content"] for item in request["messages"])
                return bad_draft() if "2000" in body else good_draft()

        result = self.generate(completions=[Mixed()])
        self.assertEqual(result.failed, 1)
        self.assertEqual(result.generated, 1)
        self.assertEqual(result.status, "incomplete")

    def test_allow_partial_writes_a_marked_submission(self) -> None:
        class Mixed:
            def create(self, *, refresh: bool = False, **request: Any) -> Any:
                body = "".join(item["content"] for item in request["messages"])
                return bad_draft() if "2000" in body else good_draft()

        result = self.generate(completions=[Mixed()], allow_partial=True)
        self.assertEqual(result.status, "incomplete")
        submission = result.run_dir / "submission.jsonl"
        self.assertTrue(submission.is_file())
        lines = submission.read_text().strip().split("\n")
        self.assertEqual([json.loads(line)["metadata"]["topic_id"] for line in lines], ["2001"])
        manifest = json.loads((result.run_dir / "manifest.json").read_text())
        self.assertTrue(manifest["partial"])

    def test_concurrency_produces_the_same_deterministic_output(self) -> None:
        serial = self.generate(("2000", "2001"))
        parallel = self.generate(
            ("2000", "2001"),
            output_root=self.root / "runs-parallel",
            concurrency=2,
            completions=[EndlessCompletions(good_draft), EndlessCompletions(good_draft)],
        )
        self.assertEqual(
            (serial.run_dir / "submission.jsonl").read_bytes(),
            (parallel.run_dir / "submission.jsonl").read_bytes(),
        )

    def test_every_topic_runs_when_topics_outnumber_workers(self) -> None:
        # The pool must refill a freed worker without dropping queued topics.
        # Pulling from the task iterator before checking for a free slot
        # silently discarded a topic whenever the pool was saturated.
        ids = tuple(str(2000 + n) for n in range(9))
        result = self.generate(
            ids,
            concurrency=2,
            completions=[EndlessCompletions(good_draft), EndlessCompletions(good_draft)],
        )
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.generated + result.cached, len(ids))
        lines = (result.run_dir / "submission.jsonl").read_text().strip().split("\n")
        self.assertEqual(
            [json.loads(line)["metadata"]["topic_id"] for line in lines], list(ids)
        )

    def test_too_few_completions_for_the_concurrency_is_rejected(self) -> None:
        with self.assertRaises(ReportRunError):
            self.generate(concurrency=2, completions=[EndlessCompletions(good_draft)])

    def test_reusing_a_run_id_with_different_settings_is_rejected(self) -> None:
        self.generate()
        with self.assertRaises(ReportRunError):
            self.generate(settings=self.settings(model="other-model"))

    def test_reusing_a_run_id_with_the_same_settings_resumes(self) -> None:
        self.generate()
        result = self.generate(completions=[TripwireCompletions()])
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.cached, 2)

    def test_missing_topic_evidence_is_reported_not_crashed(self) -> None:
        result = self.generate(
            input_dir=self.write_run("2000"), topics_path=self.write_topics("2000", "2001")
        )
        self.assertEqual(result.status, "incomplete")
        self.assertIn("2001", result.missing)

    def test_dry_run_makes_no_provider_call_and_writes_no_submission(self) -> None:
        result = self.generate(completions=[TripwireCompletions()], dry_run=True)
        self.assertEqual(result.status, "dry_run")
        self.assertFalse((result.run_dir / "submission.jsonl").exists())

    def test_checkpoints_are_written_in_topic_order(self) -> None:
        result = self.generate(("2000", "2001"))
        lines = (result.run_dir / "reports.jsonl").read_text().strip().split("\n")
        self.assertEqual([json.loads(line)["topic_id"] for line in lines], ["2000", "2001"])
        self.assertEqual(json.loads(lines[0])["status"], "success")

    def test_run_directory_is_owner_only(self) -> None:
        import stat

        result = self.generate()
        mode = stat.S_IMODE(result.run_dir.stat().st_mode)
        self.assertEqual(mode & 0o077, 0)

    def test_no_credentials_reach_the_run_directory(self) -> None:
        result = self.generate()
        for path in result.run_dir.iterdir():
            with self.subTest(path=path.name):
                body = path.read_text().lower()
                for needle in ("api_key", "apikey", "authorization", "bearer", "secret"):
                    self.assertNotIn(needle, body)


class CliTests(RunnerTestCase):
    def run_cli(self, *argv: str) -> tuple[int, str, str]:
        out, err = StringIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_parser_requires_team_and_run_identity(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--input", "x", "--model", "m"])

    def test_over_long_run_id_is_a_usage_error(self) -> None:
        run_dir = self.write_run("2000")
        topics = self.write_topics("2000")
        code, _, err = self.run_cli(
            "--input", str(run_dir),
            "--topics", str(topics),
            "--team-id", TEAM_ID,
            "--run-id", "a" * 26,
            "--model", "test-model",
            "--output-root", str(self.output_root),
        )
        self.assertEqual(code, 2)
        self.assertIn("run_id", err)

    def test_blank_team_id_is_a_usage_error(self) -> None:
        run_dir = self.write_run("2000")
        topics = self.write_topics("2000")
        code, _, err = self.run_cli(
            "--input", str(run_dir),
            "--topics", str(topics),
            "--team-id", "   ",
            "--run-id", RUN_ID,
            "--model", "test-model",
            "--output-root", str(self.output_root),
        )
        self.assertEqual(code, 2)
        self.assertIn("team_id", err)

    def test_missing_input_directory_is_a_usage_error(self) -> None:
        topics = self.write_topics("2000")
        code, _, err = self.run_cli(
            "--input", str(self.root / "absent"),
            "--topics", str(topics),
            "--team-id", TEAM_ID,
            "--run-id", RUN_ID,
            "--model", "test-model",
            "--output-root", str(self.output_root),
        )
        self.assertEqual(code, 2)
        self.assertTrue(err.strip())


if __name__ == "__main__":
    unittest.main()
