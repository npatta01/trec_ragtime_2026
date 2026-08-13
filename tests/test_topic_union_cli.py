from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trec_ragtime.topic_union_cli import select_topics, write_dry_run
from trec_ragtime.topic_union_scoring import InformationNeed
from trec_ragtime.topic_union_snapshot import SnapshotCandidate, SnapshotRetrieval, TopicSnapshot


def topic(topic_id: str, queries: int, *, sealed_queries: int | None = None) -> TopicSnapshot:
    retrievals = tuple(
        SnapshotRetrieval(
            task_id=f"task-{index}",
            query=f"secret query {index}",
            object_keys=(f"retrieval/v1/{topic_id}/{index}",),
            candidates=(SnapshotCandidate(f"doc-{index}", 1, 0.5),),
        )
        for index in range(queries)
    )
    return TopicSnapshot(
        topic_id=topic_id,
        plan_sha256="a" * 64,
        needs=(InformationNeed("need-1", "secret need text"),),
        retrievals=retrievals,
        input_object_keys=tuple(row.object_keys[0] for row in retrievals),
        sealed_query_count=queries if sealed_queries is None else sealed_queries,
    )


class TopicUnionCliTests(unittest.TestCase):
    def test_topic_limit_prefers_most_successful_queries_then_topic_id(self) -> None:
        selected = select_topics((topic("2002", 2), topic("2001", 3), topic("2000", 3)), None, 2)
        self.assertEqual([row.topic_id for row in selected], ["2000", "2001"])

    def test_explicit_topic_selection_is_exact(self) -> None:
        selected = select_topics((topic("2000", 1), topic("2001", 4)), ("2000",), None)
        self.assertEqual([row.topic_id for row in selected], ["2000"])
        with self.assertRaisesRegex(ValueError, "unknown topics"):
            select_topics((topic("2000", 1),), ("missing",), None)

    def test_complete_only_excludes_partially_retrieved_topics(self) -> None:
        selected = select_topics(
            (topic("2000", 2, sealed_queries=3), topic("2001", 4)),
            None,
            None,
            complete_only=True,
        )
        self.assertEqual([row.topic_id for row in selected], ["2001"])

    def test_dry_run_is_owner_only_and_contains_no_queries_or_need_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "wave"
            write_dry_run(output, (topic("2000", 2),))
            payload = (output / "snapshot.json").read_text()
            self.assertNotIn("secret query", payload)
            self.assertNotIn("secret need text", payload)
            record = json.loads(payload)
            self.assertEqual(record["topics"][0]["successful_query_count"], 2)
            self.assertEqual((output.stat().st_mode & 0o777), 0o700)
            self.assertEqual(((output / "snapshot.json").stat().st_mode & 0o777), 0o600)


if __name__ == "__main__":
    unittest.main()
