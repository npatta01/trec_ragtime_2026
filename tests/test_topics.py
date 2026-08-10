from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from trec_ragtime.topics import TopicFileError, TopicRequest, load_topic_requests


def topic_record(topic_id: str = "2000", **overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "topic_id": topic_id,
        "collection_id": "ragtime/2/all",
        "title": "A title",
        "problem_statement": "A problem",
        "background": "Some background",
        "limit": 5000,
    }
    record.update(overrides)
    return record


class TopicRequestTests(unittest.TestCase):
    def test_valid_topic_is_immutable_and_preserves_extra_fields(self) -> None:
        record = topic_record(
            audience="researchers",
            options={"languages": ["eng", "spa"]},
        )

        topic = TopicRequest.from_mapping(record)

        self.assertEqual(topic.topic_id, "2000")
        self.assertEqual(topic.extra["audience"], "researchers")
        self.assertEqual(
            topic.extra["options"],
            {"languages": ["eng", "spa"]},
        )
        self.assertNotIn("request_id", topic.as_record())
        with self.assertRaises(FrozenInstanceError):
            topic.title = "Changed"  # type: ignore[misc]
        with self.assertRaises(TypeError):
            topic.extra["audience"] = "changed"  # type: ignore[index]

    def test_request_id_is_accepted_and_canonicalized(self) -> None:
        record = topic_record()
        record["request_id"] = record.pop("topic_id")

        topic = TopicRequest.from_mapping(record)

        self.assertEqual(topic.topic_id, "2000")
        self.assertEqual(topic.as_record()["topic_id"], "2000")
        self.assertNotIn("request_id", topic.extra)

    def test_equal_id_aliases_are_accepted_but_conflicts_are_rejected(self) -> None:
        same = topic_record(request_id="2000")
        self.assertEqual(TopicRequest.from_mapping(same).topic_id, "2000")

        conflict = topic_record(request_id="different")
        with self.assertRaisesRegex(ValueError, "conflict"):
            TopicRequest.from_mapping(conflict)

    def test_collection_must_be_the_official_2026_collection(self) -> None:
        with self.assertRaisesRegex(ValueError, "collection_id"):
            TopicRequest.from_mapping(topic_record(collection_id="ragtime2"))

    def test_topic_id_must_not_have_surrounding_whitespace(self) -> None:
        with self.assertRaisesRegex(ValueError, "leading or trailing whitespace"):
            TopicRequest.from_mapping(topic_record(topic_id=" 2000 "))

    def test_required_text_fields_must_be_nonempty_strings(self) -> None:
        for field_name in (
            "topic_id",
            "title",
            "problem_statement",
            "background",
        ):
            for invalid in (None, "", "   ", 12):
                with self.subTest(field_name=field_name, invalid=invalid):
                    with self.assertRaisesRegex(ValueError, field_name):
                        TopicRequest.from_mapping(
                            topic_record(**{field_name: invalid})
                        )

    def test_all_required_fields_must_be_present(self) -> None:
        for field_name in (
            "topic_id",
            "collection_id",
            "title",
            "problem_statement",
            "background",
            "limit",
        ):
            with self.subTest(field_name=field_name):
                record = topic_record()
                del record[field_name]
                with self.assertRaisesRegex(ValueError, "missing required field"):
                    TopicRequest.from_mapping(record)

    def test_limit_must_be_a_positive_non_boolean_integer(self) -> None:
        for invalid in (True, False, 0, -1, 1.5, "5000"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    TopicRequest.from_mapping(topic_record(limit=invalid))


class TopicLoaderTests(unittest.TestCase):
    def write_lines(self, directory: str, *lines: str) -> Path:
        path = Path(directory) / "topics.jsonl"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def test_loads_in_file_order_and_uses_physical_line_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = topic_record("2002", z_extra=2, a_extra=1)
            second = topic_record("2001")
            path = self.write_lines(
                directory,
                json.dumps(first),
                "",
                json.dumps(second),
            )

            topics = load_topic_requests(path)

            self.assertIsInstance(topics, tuple)
            self.assertEqual([topic.topic_id for topic in topics], ["2002", "2001"])
            self.assertEqual(list(topics[0].extra), ["a_extra", "z_extra"])

    def test_missing_file_uses_the_topic_loader_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.jsonl"

            with self.assertRaisesRegex(TopicFileError, "unable to open topic file"):
                load_topic_requests(path)

    def test_invalid_json_error_is_line_numbered_and_sanitized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            secret_payload = '{"bearer_token": "do-not-repeat"'
            path = self.write_lines(
                directory,
                json.dumps(topic_record("2000")),
                secret_payload,
            )

            with self.assertRaises(TopicFileError) as caught:
                load_topic_requests(path)

            self.assertIn("line 2: invalid JSON", str(caught.exception))
            self.assertNotIn("do-not-repeat", str(caught.exception))

    def test_nonobject_record_is_rejected_with_its_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_lines(directory, "", "[]")

            with self.assertRaisesRegex(
                TopicFileError,
                "line 2: topic must be a JSON object",
            ):
                load_topic_requests(path)

    def test_topic_validation_error_includes_its_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_lines(
                directory,
                json.dumps(topic_record("2000")),
                json.dumps(topic_record("2001", title="")),
            )

            with self.assertRaisesRegex(
                TopicFileError,
                "line 2: invalid topic: title must be a nonempty string",
            ):
                load_topic_requests(path)

    def test_duplicate_topic_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_lines(
                directory,
                json.dumps(topic_record("2000")),
                "",
                json.dumps(topic_record("2000")),
            )

            with self.assertRaisesRegex(
                TopicFileError,
                r"line 3: duplicate topic_id \(first defined on line 1\)",
            ):
                load_topic_requests(path)

    def test_empty_or_whitespace_only_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_lines(directory, "", "   ", "\t")

            with self.assertRaisesRegex(TopicFileError, "topic file is empty"):
                load_topic_requests(path)

    def test_nonfinite_json_number_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_lines(
                directory,
                json.dumps(topic_record("2000")).replace("5000", "NaN"),
            )

            with self.assertRaisesRegex(TopicFileError, "line 1: invalid JSON"):
                load_topic_requests(path)


if __name__ == "__main__":
    unittest.main()
