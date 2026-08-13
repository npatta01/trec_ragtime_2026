from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from trec_ragtime.submission_validation import (
    AUTOJUDGE_MODULE,
    AUTOJUDGE_REQUIREMENT,
    DEFAULT_SPEC,
    DEFAULT_TOPICS_PATH,
    SubmissionValidationError,
    autojudge_runner,
    build_check_command,
    build_parser,
    main,
    request_record,
    run_check,
    write_requests_file,
)
from trec_ragtime.topic_candidates import Topic, load_topics


def _topic(**overrides: object) -> Topic:
    fields: dict[str, object] = {
        "topic_id": "2000",
        "collection_id": "ragtime/2/all",
        "problem_statement": "a problem statement",
        "background": "a background",
        "limit": 5000,
        "title": "A title",
    }
    fields.update(overrides)
    return Topic(**fields)  # type: ignore[arg-type]


class RequestTranslationTests(unittest.TestCase):
    def test_request_record_renames_the_fields_autojudge_requires(self) -> None:
        record = request_record(_topic())
        self.assertEqual(record["request_id"], "2000")
        self.assertEqual(record["collection_ids"], ["ragtime/2/all"])
        self.assertEqual(record["title"], "A title")
        self.assertNotIn("topic_id", record)
        self.assertNotIn("collection_id", record)

    def test_request_record_preserves_the_per_topic_length_limit(self) -> None:
        self.assertEqual(request_record(_topic(limit=3000))["limit"], 3000)

    def test_request_record_falls_back_to_the_topic_id_for_a_missing_title(self) -> None:
        record = request_record(_topic(title=None))
        self.assertEqual(record["title"], "2000")

    def test_request_record_rejects_a_non_topic(self) -> None:
        with self.assertRaises(TypeError):
            request_record({"topic_id": "2000"})  # type: ignore[arg-type]

    def test_write_requests_file_preserves_source_order(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            path = write_requests_file(
                [_topic(topic_id="2001"), _topic(topic_id="2000")],
                Path(scratch) / "nested" / "requests.jsonl",
            )
            ids = [
                json.loads(line)["request_id"]
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual(ids, ["2001", "2000"])


class OfficialTopicsTranslationTests(unittest.TestCase):
    """The official topics file is the input this translation exists for."""

    def test_official_topics_translate_to_loadable_requests(self) -> None:
        topics_path = Path(__file__).resolve().parents[1] / DEFAULT_TOPICS_PATH
        if not topics_path.is_file():
            self.skipTest(f"official topics file is absent: {topics_path}")
        topics = load_topics(topics_path)
        with tempfile.TemporaryDirectory() as scratch:
            path = write_requests_file(topics, Path(scratch) / "requests.jsonl")
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual(len(records), len(topics))
        for record in records:
            self.assertIsInstance(record["request_id"], str)
            self.assertIsInstance(record["title"], str)
            self.assertIsInstance(record["limit"], int)


class CommandConstructionTests(unittest.TestCase):
    def test_check_command_places_every_option_before_the_paths(self) -> None:
        command = build_check_command(
            ["python", "-m", AUTOJUDGE_MODULE],
            [Path("run-a.jsonl"), Path("run-b.jsonl")],
            requests_path=Path("requests.jsonl"),
            strict=True,
            suppress_warnings=True,
        )
        self.assertEqual(
            command,
            [
                "python",
                "-m",
                AUTOJUDGE_MODULE,
                "check",
                "--spec",
                DEFAULT_SPEC,
                "--topics",
                "requests.jsonl",
                "--strict",
                "--suppress-warnings",
                "run-a.jsonl",
                "run-b.jsonl",
            ],
        )

    def test_check_command_omits_absent_spec_and_topics(self) -> None:
        command = build_check_command(
            ["python", "-m", AUTOJUDGE_MODULE],
            [Path("run.jsonl")],
            spec=None,
            requests_path=None,
        )
        self.assertEqual(command, ["python", "-m", AUTOJUDGE_MODULE, "check", "run.jsonl"])

    def test_check_command_requires_a_submission(self) -> None:
        with self.assertRaises(ValueError):
            build_check_command(["python"], [])

    def test_runner_uses_the_current_interpreter_when_importable(self) -> None:
        runner = autojudge_runner()
        if runner[0] == sys.executable:
            self.assertEqual(runner, [sys.executable, "-m", AUTOJUDGE_MODULE])
        else:
            self.assertEqual(runner[0:4], [runner[0], "run", "--isolated", "--no-project"])
            self.assertIn(AUTOJUDGE_REQUIREMENT, runner)
            self.assertEqual(runner[-3:], ["python", "-m", AUTOJUDGE_MODULE])


class RunCheckTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.scratch = Path(self.temporary_directory.name)
        self.topics_path = self.scratch / "topics.jsonl"
        self.topics_path.write_text(
            json.dumps(
                {
                    "topic_id": "2000",
                    "collection_id": "ragtime/2/all",
                    "title": "A title",
                    "problem_statement": "a problem statement",
                    "background": "a background",
                    "limit": 5000,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.submission_path = self.scratch / "submission.jsonl"
        self.submission_path.write_text("{}\n", encoding="utf-8")

    def test_run_check_reports_a_missing_submission_before_running(self) -> None:
        with self.assertRaises(SubmissionValidationError):
            run_check(
                [self.scratch / "absent.jsonl"],
                topics_path=self.topics_path,
                echo=False,
            )

    def test_run_check_propagates_the_checker_exit_code(self) -> None:
        recorded: dict[str, object] = {}

        def fake_run(command, check=False):  # type: ignore[no-untyped-def]
            recorded["command"] = list(command)
            recorded["requests"] = Path(command[command.index("--topics") + 1]).read_text(
                encoding="utf-8"
            )
            return subprocess.CompletedProcess(command, 255)

        original = subprocess.run
        subprocess.run = fake_run  # type: ignore[assignment]
        self.addCleanup(lambda: setattr(subprocess, "run", original))
        code = run_check(
            [self.submission_path],
            topics_path=self.topics_path,
            echo=False,
        )
        self.assertEqual(code, 255)
        self.assertIn("--spec", recorded["command"])  # type: ignore[operator]
        self.assertEqual(
            json.loads(str(recorded["requests"]).splitlines()[0])["request_id"], "2000"
        )

    def test_run_check_can_keep_the_translated_requests_file(self) -> None:
        requests_out = self.scratch / "kept-requests.jsonl"

        def fake_run(command, check=False):  # type: ignore[no-untyped-def]
            return subprocess.CompletedProcess(command, 0)

        original = subprocess.run
        subprocess.run = fake_run  # type: ignore[assignment]
        self.addCleanup(lambda: setattr(subprocess, "run", original))
        code = run_check(
            [self.submission_path],
            topics_path=self.topics_path,
            requests_out=requests_out,
            echo=False,
        )
        self.assertEqual(code, 0)
        self.assertTrue(requests_out.is_file())

    def test_run_check_without_topics_omits_the_cross_check(self) -> None:
        recorded: dict[str, object] = {}

        def fake_run(command, check=False):  # type: ignore[no-untyped-def]
            recorded["command"] = list(command)
            return subprocess.CompletedProcess(command, 0)

        original = subprocess.run
        subprocess.run = fake_run  # type: ignore[assignment]
        self.addCleanup(lambda: setattr(subprocess, "run", original))
        run_check([self.submission_path], topics_path=None, echo=False)
        self.assertNotIn("--topics", recorded["command"])  # type: ignore[operator]


class ParserTests(unittest.TestCase):
    def test_parser_defaults_to_the_official_topics_and_spec(self) -> None:
        arguments = build_parser().parse_args(["run.jsonl"])
        self.assertEqual(arguments.topics, DEFAULT_TOPICS_PATH)
        self.assertEqual(arguments.spec, DEFAULT_SPEC)
        self.assertEqual(arguments.submissions, [Path("run.jsonl")])

    def test_no_topics_clears_the_topic_cross_check(self) -> None:
        arguments = build_parser().parse_args(["--no-topics", "run.jsonl"])
        self.assertIsNone(arguments.topics)

    def test_main_reports_a_missing_submission_as_exit_two(self) -> None:
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(["--no-topics", "definitely-absent.jsonl"])
        self.assertEqual(code, 2)
        self.assertIn("submission file not found", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
