from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from trec_ragtime.topic_candidates import (
    Topic,
    build_topic_query,
    load_topics,
    pipeline_for_top_k,
    query_hash,
    write_control_a_artifact,
)


class TopicHelpersTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.topic_file = Path(self.temporary_directory.name) / "topics.jsonl"

    def write_topics(self, records: list[object]) -> None:
        self.topic_file.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )

    def test_load_topics_preserves_source_order_and_validates_fields(self) -> None:
        self.write_topics(
            [
                {
                    "topic_id": "2000",
                    "collection_id": "ragtime/2/all",
                    "problem_statement": "first problem",
                    "background": "first background",
                    "limit": 5000,
                },
                {
                    "topic_id": "2001",
                    "collection_id": "ragtime/2/all",
                    "problem_statement": "second problem",
                    "background": "second background",
                    "limit": 5000,
                },
            ]
        )

        topics = load_topics(self.topic_file)

        self.assertEqual([topic.topic_id for topic in topics], ["2000", "2001"])
        self.assertEqual(topics[0].limit, 5000)

    def test_build_topic_query_labels_problem_and_background(self) -> None:
        topic = Topic("1", "ragtime/2/all", "problem", "background", 5000)

        self.assertEqual(
            build_topic_query(topic), "Problem: problem\n\nBackground: background"
        )

    def test_pipeline_for_top_k_translates_depth_to_organizer_protocol(self) -> None:
        self.assertEqual(pipeline_for_top_k(100), "ragtime2%100")

        with self.assertRaises(ValueError):
            pipeline_for_top_k(0)

    def test_query_hash_is_sha256_of_exact_query(self) -> None:
        query = "Problem: café"

        self.assertEqual(
            query_hash(query), hashlib.sha256(query.encode("utf-8")).hexdigest()
        )

    def test_load_topics_rejects_malformed_records(self) -> None:
        cases = [
            ("not json\n", "invalid JSON"),
            (
                json.dumps(
                    {
                        "topic_id": "1",
                        "collection_id": "ragtime/2/all",
                        "problem_statement": "problem",
                        "background": "background",
                    }
                )
                + "\n",
                "missing field 'limit'",
            ),
            (
                json.dumps(
                    {
                        "topic_id": "1",
                        "collection_id": "ragtime/2/all",
                        "problem_statement": "problem",
                        "background": "background",
                        "limit": 0,
                    }
                )
                + "\n",
                "limit",
            ),
            (
                json.dumps(
                    {
                        "topic_id": "1",
                        "collection_id": "ragtime/2/all",
                        "problem_statement": "",
                        "background": "background",
                        "limit": 5000,
                    }
                )
                + "\n",
                "problem_statement",
            ),
        ]

        for contents, message in cases:
            with self.subTest(message=message):
                self.topic_file.write_text(contents, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_topics(self.topic_file)

    def test_load_topics_rejects_duplicate_topic_ids(self) -> None:
        record = {
            "topic_id": "1",
            "collection_id": "ragtime/2/all",
            "problem_statement": "problem",
            "background": "background",
            "limit": 5000,
        }
        self.write_topics([record, record])

        with self.assertRaisesRegex(ValueError, "duplicate topic_id"):
            load_topics(self.topic_file)


class ControlAArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.output_root = Path(self.temporary_directory.name) / "control-a"

    def test_writer_creates_stable_manifest_and_jsonl(self) -> None:
        run_dir = self.output_root / "run-1"
        manifest = {"z": "last", "a": "first"}
        records = [
            {
                "topic_id": "1",
                "text": "canonical source text",
                "report_text": "English report text",
                "hydrated": True,
                "missing_translation": False,
            }
        ]

        write_control_a_artifact(run_dir, manifest=manifest, records=records)

        self.assertEqual(
            json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")),
            manifest,
        )
        candidate_lines = (run_dir / "candidates.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        self.assertEqual(len(candidate_lines), 1)
        self.assertEqual(json.loads(candidate_lines[0]), records[0])
        self.assertTrue(candidate_lines[0].startswith('{"hydrated":true'))
        self.assertEqual(list(run_dir.glob(".control-a-*")), [])

    def test_writer_refuses_to_overwrite_existing_run(self) -> None:
        run_dir = self.output_root / "run-1"
        run_dir.mkdir(parents=True)

        with self.assertRaises(FileExistsError):
            write_control_a_artifact(run_dir, manifest={}, records=[])

        self.assertEqual(list(run_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
