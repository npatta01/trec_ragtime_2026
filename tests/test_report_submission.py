from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from trec_ragtime.report_submission import (
    MAX_RUN_ID_LENGTH,
    SUBMISSION_FORMAT_VERSION,
    SubmissionValidationError,
    main,
    submission_record,
    validate_run_desc,
    validate_run_id,
    validate_submission_file,
    validate_submission_record,
    validate_team_id,
)

TEAM_ID = "PLACEHOLDER-TEAM"
RUN_ID = "placeholder-run-1"
RUN_DESC = "offline fixture"


def sentence(text: str, citations: dict[str, float] | None = None) -> dict[str, object]:
    return {"text": text, "citations": dict(citations or {})}


def good_record(topic_id: str = "2000") -> dict[str, object]:
    return submission_record(
        team_id=TEAM_ID,
        topic_id=topic_id,
        run_id=RUN_ID,
        run_desc=RUN_DESC,
        sentences=[
            sentence("The ban took effect in 2019.", {"doc-a": 100.0}),
            sentence("Youth vaping fell sharply.", {"doc-b": 100.0, "doc-a": 90.0}),
        ],
    )


def write_topics(directory: Path, topic_ids: tuple[str, ...], limit: int = 5000) -> Path:
    path = directory / "topics.jsonl"
    lines = [
        json.dumps(
            {
                "topic_id": topic_id,
                "collection_id": "ragtime/2/all",
                "title": f"Topic {topic_id}",
                "problem_statement": "A statement.",
                "background": "Some background.",
                "limit": limit,
            }
        )
        for topic_id in topic_ids
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_submission(directory: Path, records: list[dict[str, object]]) -> Path:
    path = directory / "submission.jsonl"
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


class MetadataFieldTests(unittest.TestCase):
    def test_format_version_is_declared(self) -> None:
        self.assertTrue(SUBMISSION_FORMAT_VERSION.strip())
        self.assertEqual(MAX_RUN_ID_LENGTH, 20)

    def test_run_id_at_the_limit_is_accepted(self) -> None:
        run_id = "a" * MAX_RUN_ID_LENGTH
        self.assertEqual(validate_run_id(run_id), run_id)

    def test_run_id_over_the_limit_is_rejected(self) -> None:
        with self.assertRaises(SubmissionValidationError):
            validate_run_id("a" * (MAX_RUN_ID_LENGTH + 1))

    def test_run_id_length_is_measured_after_nfkc(self) -> None:
        # Each ligature expands to two characters, pushing 11 codepoints to 22.
        with self.assertRaises(SubmissionValidationError):
            validate_run_id("ﬁ" * 11)

    def test_blank_and_whitespace_bearing_identifiers_are_rejected(self) -> None:
        for value in ("", "   ", "run id", "run\tid", "run\nid"):
            with self.subTest(value=repr(value)):
                with self.assertRaises(SubmissionValidationError):
                    validate_run_id(value)
                with self.assertRaises(SubmissionValidationError):
                    validate_team_id(value)

    def test_non_string_identifiers_are_rejected(self) -> None:
        for value in (None, 7, b"team"):
            with self.subTest(value=value):
                with self.assertRaises(SubmissionValidationError):
                    validate_team_id(value)  # type: ignore[arg-type]

    def test_run_desc_allows_spaces_but_not_control_characters(self) -> None:
        self.assertEqual(validate_run_desc("a useful run"), "a useful run")
        with self.assertRaises(SubmissionValidationError):
            validate_run_desc("bad\ndesc")


class SubmissionRecordTests(unittest.TestCase):
    def test_record_matches_the_official_shape(self) -> None:
        record = good_record()
        self.assertEqual(set(record), {"metadata", "responses", "references"})
        self.assertEqual(
            set(record["metadata"]),  # type: ignore[arg-type]
            {"team_id", "topic_id", "run_id", "run_desc"},
        )
        self.assertEqual(
            record["responses"][0],  # type: ignore[index]
            {"text": "The ban took effect in 2019.", "citations": {"doc-a": 100.0}},
        )

    def test_references_are_the_sorted_citation_union(self) -> None:
        self.assertEqual(good_record()["references"], ["doc-a", "doc-b"])

    def test_responses_preserve_report_order(self) -> None:
        texts = [item["text"] for item in good_record()["responses"]]  # type: ignore[index]
        self.assertEqual(
            texts,
            ["The ban took effect in 2019.", "Youth vaping fell sharply."],
        )

    def test_uncited_sentence_keeps_an_empty_citation_map(self) -> None:
        record = submission_record(
            team_id=TEAM_ID,
            topic_id="2000",
            run_id=RUN_ID,
            run_desc=RUN_DESC,
            sentences=[
                sentence("This report covers three themes."),
                sentence("The ban took effect in 2019.", {"doc-a": 100.0}),
            ],
        )
        self.assertEqual(record["responses"][0]["citations"], {})  # type: ignore[index]
        self.assertEqual(record["references"], ["doc-a"])

    def test_record_round_trips_through_json(self) -> None:
        record = good_record()
        self.assertEqual(json.loads(json.dumps(record)), record)

    def test_empty_sentence_list_is_rejected(self) -> None:
        with self.assertRaises(SubmissionValidationError):
            submission_record(
                team_id=TEAM_ID,
                topic_id="2000",
                run_id=RUN_ID,
                run_desc=RUN_DESC,
                sentences=[],
            )


class RecordValidationTests(unittest.TestCase):
    def validate(self, record: object, *, limit: int = 5000) -> list[str]:
        return [item.code for item in validate_submission_record(record, character_limit=limit)]

    def test_good_record_has_no_violations(self) -> None:
        self.assertEqual(self.validate(good_record()), [])

    def test_missing_and_extra_top_level_keys_are_rejected(self) -> None:
        record = good_record()
        del record["references"]
        self.assertIn("top_level_keys", self.validate(record))
        record = good_record()
        record["extra"] = 1
        self.assertIn("top_level_keys", self.validate(record))

    def test_metadata_key_set_is_exact(self) -> None:
        record = good_record()
        record["metadata"]["extra"] = "x"  # type: ignore[index]
        self.assertIn("metadata_keys", self.validate(record))

    def test_over_length_run_id_is_reported(self) -> None:
        record = good_record()
        record["metadata"]["run_id"] = "a" * 21  # type: ignore[index]
        self.assertIn("run_id", self.validate(record))

    def test_empty_responses_are_rejected(self) -> None:
        record = good_record()
        record["responses"] = []
        self.assertIn("responses_empty", self.validate(record))

    def test_multi_sentence_response_is_rejected(self) -> None:
        record = good_record()
        record["responses"][0]["text"] = "One claim. Another claim."  # type: ignore[index]
        self.assertIn("not_single_sentence", self.validate(record))

    def test_unnormalized_text_is_rejected(self) -> None:
        record = good_record()
        record["responses"][0]["text"] = "The  ban   took effect."  # type: ignore[index]
        self.assertIn("text_not_normalized", self.validate(record))

    def test_non_finite_and_boolean_scores_are_rejected(self) -> None:
        for score in (float("inf"), float("nan"), True):
            with self.subTest(score=score):
                record = good_record()
                record["responses"][0]["citations"] = {"doc-a": score}  # type: ignore[index]
                self.assertIn("citation_score", self.validate(record))

    def test_duplicate_scores_within_a_sentence_are_rejected(self) -> None:
        record = good_record()
        record["responses"][1]["citations"] = {"doc-a": 90.0, "doc-b": 90.0}  # type: ignore[index]
        self.assertIn("citation_score_tie", self.validate(record))

    def test_blank_citation_key_is_rejected(self) -> None:
        record = good_record()
        record["responses"][0]["citations"] = {"": 100.0}  # type: ignore[index]
        self.assertIn("citation_key", self.validate(record))

    def test_references_superset_is_rejected(self) -> None:
        record = good_record()
        record["references"] = ["doc-a", "doc-b", "doc-c"]
        self.assertIn("references_union", self.validate(record))

    def test_references_subset_is_rejected(self) -> None:
        record = good_record()
        record["references"] = ["doc-a"]
        self.assertIn("references_union", self.validate(record))

    def test_duplicate_reference_entries_are_rejected(self) -> None:
        record = good_record()
        record["references"] = ["doc-a", "doc-a", "doc-b"]
        self.assertIn("references_duplicate", self.validate(record))

    def test_character_limit_is_enforced_on_normalized_text(self) -> None:
        record = submission_record(
            team_id=TEAM_ID,
            topic_id="2000",
            run_id=RUN_ID,
            run_desc=RUN_DESC,
            sentences=[sentence("A" * 40 + ".", {"doc-a": 100.0})],
        )
        self.assertEqual(self.validate(record, limit=41), [])
        self.assertIn("character_limit", self.validate(record, limit=40))

    def test_character_limit_counts_nfkc_expansion(self) -> None:
        record = submission_record(
            team_id=TEAM_ID,
            topic_id="2000",
            run_id=RUN_ID,
            run_desc=RUN_DESC,
            sentences=[sentence("ﬁ" * 5 + ".", {"doc-a": 100.0})],
        )
        # Five ligatures normalize to ten characters, plus the period. Text that
        # needs normalizing is itself a violation, but the limit must still be
        # measured against the expanded count rather than the raw codepoints.
        self.assertEqual(self.validate(record, limit=11), ["text_not_normalized"])
        self.assertEqual(
            sorted(self.validate(record, limit=10)),
            ["character_limit", "text_not_normalized"],
        )

    def test_limit_counts_composition_across_the_sentence_join(self) -> None:
        record = submission_record(
            team_id=TEAM_ID,
            topic_id="2000",
            run_id=RUN_ID,
            run_desc=RUN_DESC,
            sentences=[
                sentence("Prevalence fell.", {"doc-a": 100.0}),
                sentence("Enforcement began.", {"doc-b": 100.0}),
            ],
        )
        total = len("Prevalence fell.") + len("Enforcement began.")
        self.assertEqual(self.validate(record, limit=total), [])
        self.assertIn("character_limit", self.validate(record, limit=total - 1))

    def test_non_object_record_is_rejected(self) -> None:
        for value in ([], "x", 3, None):
            with self.subTest(value=value):
                self.assertIn("record_type", self.validate(value))


class SubmissionFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)

    def topics(self, *topic_ids: str) -> dict[str, int]:
        return {topic_id: 5000 for topic_id in topic_ids}

    def test_complete_file_validates(self) -> None:
        path = write_submission(self.root, [good_record("2000"), good_record("2001")])
        report = validate_submission_file(path, topics=self.topics("2000", "2001"))
        self.assertTrue(report.ok, report.violations)
        self.assertEqual(report.record_count, 2)
        self.assertEqual(report.topic_ids, ("2000", "2001"))

    def test_missing_topic_is_reported(self) -> None:
        path = write_submission(self.root, [good_record("2000")])
        report = validate_submission_file(path, topics=self.topics("2000", "2001"))
        self.assertIn("topic_missing", {item.code for item in report.violations})

    def test_unknown_topic_is_reported(self) -> None:
        path = write_submission(self.root, [good_record("9999")])
        report = validate_submission_file(path, topics=self.topics("2000"))
        self.assertIn("topic_unknown", {item.code for item in report.violations})

    def test_duplicate_topic_is_reported(self) -> None:
        path = write_submission(self.root, [good_record("2000"), good_record("2000")])
        report = validate_submission_file(path, topics=self.topics("2000"))
        self.assertIn("topic_duplicate", {item.code for item in report.violations})

    def test_strict_order_flags_out_of_order_topics(self) -> None:
        path = write_submission(self.root, [good_record("2001"), good_record("2000")])
        topics = self.topics("2000", "2001")
        self.assertTrue(validate_submission_file(path, topics=topics).ok)
        report = validate_submission_file(path, topics=topics, strict_order=True)
        self.assertIn("topic_order", {item.code for item in report.violations})

    def test_inconsistent_metadata_across_records_is_reported(self) -> None:
        second = good_record("2001")
        second["metadata"]["run_id"] = "another-run"  # type: ignore[index]
        path = write_submission(self.root, [good_record("2000"), second])
        report = validate_submission_file(path, topics=self.topics("2000", "2001"))
        self.assertIn("metadata_inconsistent", {item.code for item in report.violations})

    def test_expected_team_and_run_are_enforced_when_supplied(self) -> None:
        path = write_submission(self.root, [good_record("2000")])
        topics = self.topics("2000")
        self.assertTrue(validate_submission_file(path, topics=topics).ok)
        report = validate_submission_file(path, topics=topics, team_id="OTHER")
        self.assertIn("team_id_mismatch", {item.code for item in report.violations})
        report = validate_submission_file(path, topics=topics, run_id="other-run")
        self.assertIn("run_id_mismatch", {item.code for item in report.violations})

    def test_malformed_json_line_is_reported(self) -> None:
        path = self.root / "submission.jsonl"
        path.write_text('{"metadata": }\n', encoding="utf-8")
        report = validate_submission_file(path, topics=self.topics("2000"))
        self.assertIn("json_decode", {item.code for item in report.violations})

    def test_blank_line_is_reported(self) -> None:
        path = self.root / "submission.jsonl"
        path.write_text(json.dumps(good_record("2000")) + "\n\n", encoding="utf-8")
        report = validate_submission_file(path, topics=self.topics("2000"))
        self.assertIn("blank_line", {item.code for item in report.violations})

    def test_empty_file_is_reported(self) -> None:
        path = self.root / "submission.jsonl"
        path.write_text("", encoding="utf-8")
        report = validate_submission_file(path, topics=self.topics("2000"))
        self.assertFalse(report.ok)

    def test_violations_carry_line_numbers(self) -> None:
        bad = good_record("2001")
        bad["references"] = []
        path = write_submission(self.root, [good_record("2000"), bad])
        report = validate_submission_file(path, topics=self.topics("2000", "2001"))
        union = [item for item in report.violations if item.code == "references_union"]
        self.assertEqual([item.line for item in union], [2])
        self.assertEqual([item.topic_id for item in union], ["2001"])


class ValidateCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)

    def run_cli(self, *argv: str) -> tuple[int, str, str]:
        out, err = StringIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_valid_file_exits_zero(self) -> None:
        topics = write_topics(self.root, ("2000", "2001"))
        path = write_submission(self.root, [good_record("2000"), good_record("2001")])
        code, out, _ = self.run_cli(str(path), "--topics", str(topics))
        self.assertEqual(code, 0, out)
        self.assertIn("2 report", out)

    def test_invalid_file_exits_one_and_prints_violations(self) -> None:
        topics = write_topics(self.root, ("2000",))
        record = good_record("2000")
        record["references"] = []
        path = write_submission(self.root, [record])
        code, out, _ = self.run_cli(str(path), "--topics", str(topics))
        self.assertEqual(code, 1)
        self.assertIn("references_union", out)

    def test_missing_file_exits_two(self) -> None:
        topics = write_topics(self.root, ("2000",))
        code, _, err = self.run_cli(str(self.root / "absent.jsonl"), "--topics", str(topics))
        self.assertEqual(code, 2)
        self.assertTrue(err.strip())

    def test_topic_limits_come_from_the_topic_file(self) -> None:
        topics = write_topics(self.root, ("2000",), limit=10)
        path = write_submission(self.root, [good_record("2000")])
        code, out, _ = self.run_cli(str(path), "--topics", str(topics))
        self.assertEqual(code, 1)
        self.assertIn("character_limit", out)

    def test_strict_order_flag_is_wired(self) -> None:
        topics = write_topics(self.root, ("2000", "2001"))
        path = write_submission(self.root, [good_record("2001"), good_record("2000")])
        self.assertEqual(self.run_cli(str(path), "--topics", str(topics))[0], 0)
        code, out, _ = self.run_cli(str(path), "--topics", str(topics), "--strict-order")
        self.assertEqual(code, 1)
        self.assertIn("topic_order", out)


if __name__ == "__main__":
    unittest.main()


class RagtimeSpecRunIdTests(unittest.TestCase):
    """run_id rules follow the ragtime26 AutoJudge spec, not a looser guess."""

    def test_limit_is_twenty_characters(self) -> None:
        self.assertEqual(MAX_RUN_ID_LENGTH, 20)
        self.assertEqual(validate_run_id("a" * 20), "a" * 20)
        with self.assertRaises(SubmissionValidationError):
            validate_run_id("a" * 21)

    def test_allowed_characters(self) -> None:
        for value in ("np-handoff-1", "run_1.a", "ABC-123"):
            with self.subTest(value=value):
                self.assertEqual(validate_run_id(value), value)

    def test_rejected_characters(self) -> None:
        for value in (".leading", "has/slash", "has:colon", "has+plus"):
            with self.subTest(value=value):
                with self.assertRaises(SubmissionValidationError):
                    validate_run_id(value)

    def test_the_run_id_actually_used_is_within_spec(self) -> None:
        self.assertEqual(validate_run_id("np-handoff-1"), "np-handoff-1")
