from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trec_ragtime.planner_static import allocate_tasks, static_plan_from_record
from trec_ragtime.topic_union_snapshot import TopicSnapshotError, build_snapshot


ROOT = Path(
    "/home/npatta01/data/competitions/trec_ragtime_2026/.git/"
    "ragtime-cache/experiments/planner-static-final-v1"
)
FROZEN_PLANS = ROOT / "dry-runs/all-103-2026-08-12/all-103-final-plans.jsonl"


class TopicUnionSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.record = json.loads(FROZEN_PLANS.read_text().splitlines()[0])
        self.plan = static_plan_from_record(self.record["plan"])
        self.allocated = allocate_tasks(self.plan, len(self.plan.tasks)).tasks

    def write_json(self, path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n")

    def test_snapshot_combines_breadth_and_success_checkpoints_but_excludes_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plans = root / "plans.jsonl"
            plans.write_text(json.dumps(self.record) + "\n")
            first, second, third = self.allocated[:3]
            pool = root / "pool.json"
            self.write_json(
                pool,
                {
                    "schema_version": "planner-static-breadth-pool-v1",
                    "topic_id": self.plan.topic_id,
                    "plan_sha256": self.plan.plan_sha256,
                    "query": first.task.query,
                    "retrieval_object_key": "retrieval/v1/breadth",
                    "candidate_count": 1,
                    "candidates": [
                        {
                            "document_id": "doc-a", "rank": 1, "score": 0.4,
                            "task_id": first.task.task_id,
                            "requirement_ids": list(first.task.requirement_ids),
                        }
                    ],
                },
            )
            coverage = root / "coverage.jsonl"
            coverage.write_text(
                json.dumps(
                    {
                        "topic_id": self.plan.topic_id,
                        "supplement_sources": [{"path": str(pool)}],
                    }
                )
                + "\n"
            )
            checkpoints = root / "checkpoints" / self.plan.topic_id
            self.write_json(
                checkpoints / "0002-success.json",
                {
                    "schema_version": "planner-static-full-richer-query-v1",
                    "status": "success",
                    "topic_id": self.plan.topic_id,
                    "plan_sha256": self.plan.plan_sha256,
                    "task_id": second.task.task_id,
                    "query": second.task.query,
                    "retrieval_object_key": "retrieval/v1/full",
                    "candidate_count": 1,
                    "candidates": [{"document_id": "doc-b", "rank": 1, "score": 0.8}],
                },
            )
            self.write_json(
                checkpoints / "0003-error.json",
                {
                    "schema_version": "planner-static-full-richer-query-v1",
                    "status": "error",
                    "topic_id": self.plan.topic_id,
                    "plan_sha256": self.plan.plan_sha256,
                    "task_id": third.task.task_id,
                    "query": third.task.query,
                },
            )

            snapshot = build_snapshot(plans, coverage, root / "checkpoints")

            self.assertEqual(len(snapshot), 1)
            topic = snapshot[0]
            self.assertEqual(topic.topic_id, self.plan.topic_id)
            self.assertEqual(topic.successful_query_count, 2)
            self.assertEqual([row.task_id for row in topic.retrievals], [first.task.task_id, second.task.task_id])
            self.assertEqual(topic.input_object_keys, ("retrieval/v1/breadth", "retrieval/v1/full"))
            self.assertEqual(tuple(row.need_id for row in topic.needs), tuple(row.need_id for row in self.plan.information_needs))

    def test_unknown_task_in_breadth_pool_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plans = root / "plans.jsonl"
            plans.write_text(json.dumps(self.record) + "\n")
            pool = root / "pool.json"
            self.write_json(
                pool,
                {
                    "schema_version": "planner-static-breadth-pool-v1",
                    "topic_id": self.plan.topic_id,
                    "plan_sha256": self.plan.plan_sha256,
                    "query": "unknown",
                    "retrieval_object_key": "retrieval/v1/x",
                    "candidate_count": 1,
                    "candidates": [{"document_id": "doc", "rank": 1, "score": 0.1, "task_id": "task-unknown"}],
                },
            )
            coverage = root / "coverage.jsonl"
            coverage.write_text(json.dumps({"topic_id": self.plan.topic_id, "supplement_sources": [{"path": str(pool)}]}) + "\n")

            with self.assertRaisesRegex(TopicSnapshotError, "unknown task"):
                build_snapshot(plans, coverage, root / "checkpoints")


if __name__ == "__main__":
    unittest.main()
