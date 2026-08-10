from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trec_ragtime.query_planning import (
    CompositeQueryPlanner,
    FacetQueryPlanner,
    QueryPlanError,
    StaticQueryPlanner,
    load_query_plans,
)
from trec_ragtime.topics import TopicRequest


def topic() -> TopicRequest:
    return TopicRequest(
        topic_id="2000",
        collection_id="ragtime/2/all",
        title="Example topic",
        background="The requester needs a decision-ready overview.",
        problem_statement=(
            "Explain when the policy changed and who was affected. "
            "Compare measured outcomes in different countries; include limitations."
        ),
        limit=5000,
    )


class QueryPlanningTests(unittest.TestCase):
    def test_composite_query_uses_all_request_fields(self):
        queries = CompositeQueryPlanner("ragtime2%100").plan(topic())

        self.assertEqual(len(queries), 1)
        query = queries[0]
        self.assertEqual(query.query_id, "2000:full")
        self.assertEqual(query.pipeline, "ragtime2%100")
        self.assertEqual(query.language, "eng")
        self.assertIn("Example topic", query.text)
        self.assertIn("decision-ready", query.text)
        self.assertIn("policy changed", query.text)

    def test_facet_query_plan_is_deterministic_and_bounded(self):
        planner = FacetQueryPlanner(
            pipeline="ragtime2%80",
            max_facets=2,
            min_facet_characters=10,
        )

        first = planner.plan(topic())
        second = planner.plan(topic())

        self.assertEqual(first, second)
        self.assertEqual([query.query_id for query in first], [
            "2000:full",
            "2000:facet:1",
            "2000:facet:2",
        ])
        self.assertTrue(all(query.pipeline == "ragtime2%80" for query in first))
        self.assertTrue(all(query.language == "eng" for query in first))

    def test_facet_splitter_preserves_acronyms_and_honors_no_full_query(self):
        request = TopicRequest(
            topic_id="2000",
            collection_id="ragtime/2/all",
            title="Travel",
            background="A travel researcher needs evidence.",
            problem_statement=(
                "Explain why U.S. pilgrims changed their plans. Compare outcomes."
            ),
            limit=5000,
        )
        queries = FacetQueryPlanner(
            include_full_query=False,
            min_facet_characters=1000,
        ).plan(request)

        self.assertEqual(len(queries), 1)
        self.assertEqual(queries[0].query_id, "2000:facet:1")
        self.assertIn("U.S. pilgrims", queries[0].text)

    def test_acronym_at_sentence_end_does_not_hide_the_next_facet(self):
        request = TopicRequest(
            topic_id="2000",
            collection_id="ragtime/2/all",
            title="Operations",
            background="A researcher needs evidence.",
            problem_statement=(
                "The agency operates in the U.S. Compare outcomes across regions."
            ),
            limit=5000,
        )

        queries = FacetQueryPlanner(
            include_full_query=False,
            min_facet_characters=10,
        ).plan(request)

        self.assertEqual(len(queries), 2)
        self.assertIn("operates in the U.S.", queries[0].text)
        self.assertIn("Compare outcomes", queries[1].text)

    def test_facet_splitter_repairs_titles_months_and_uppercase_acronyms(self):
        request = TopicRequest(
            topic_id="2000",
            collection_id="ragtime/2/all",
            title="Review",
            background="A researcher needs evidence.",
            problem_statement=(
                "Dr. Smith reviewed the U.S. Department policy. "
                "Compare outcomes reported on Jan. 5."
            ),
            limit=5000,
        )

        queries = FacetQueryPlanner(
            include_full_query=False,
            min_facet_characters=10,
        ).plan(request)

        self.assertEqual(len(queries), 2)
        self.assertIn("Dr. Smith", queries[0].text)
        self.assertIn("U.S. Department", queries[0].text)
        self.assertIn("Jan. 5", queries[1].text)

    def test_facet_splitter_keeps_true_sentence_and_paragraph_boundaries(self):
        request = TopicRequest(
            topic_id="2000",
            collection_id="ragtime/2/all",
            title="Review",
            background="A researcher needs evidence.",
            problem_statement=(
                "The policy covered taxes, sales, etc.\n\n"
                "The court upheld the law. "
                "The agency operates in the U.S. "
                "Congress opened hearings on the law."
            ),
            limit=5000,
        )

        queries = FacetQueryPlanner(
            include_full_query=False,
            min_facet_characters=10,
        ).plan(request)

        self.assertEqual(len(queries), 4)
        self.assertTrue(queries[0].text.endswith("sales, etc."))
        self.assertTrue(queries[1].text.endswith("upheld the law."))
        self.assertTrue(queries[2].text.endswith("in the U.S."))
        self.assertTrue(queries[3].text.endswith("on the law."))

    def test_facet_splitter_treats_a_spaced_dash_as_a_boundary(self):
        request = TopicRequest(
            topic_id="2000",
            collection_id="ragtime/2/all",
            title="Review",
            background="A researcher needs evidence.",
            problem_statement=(
                "Assess implementation costs — Compare measurable outcomes."
            ),
            limit=5000,
        )

        queries = FacetQueryPlanner(
            include_full_query=False,
            min_facet_characters=10,
        ).plan(request)

        self.assertEqual(len(queries), 2)
        self.assertTrue(queries[0].text.endswith("implementation costs"))
        self.assertTrue(queries[1].text.endswith("measurable outcomes."))

    def test_static_planner_rejects_missing_topic(self):
        baseline = CompositeQueryPlanner().plan(topic())
        planner = StaticQueryPlanner({"other": baseline})

        with self.assertRaisesRegex(QueryPlanError, "no precomputed query plan"):
            planner.plan(topic())

    def test_loads_multilingual_precomputed_plan_and_alias(self):
        records = [
            {
                "request_id": "2000",
                "queries": [
                    {
                        "query_id": "2000:spa:1",
                        "text": "cambio de política",
                        "pipeline": "spa",
                        "language": "spa",
                        "weight": 1.5,
                    },
                    {
                        "query_id": "2000:zho:1",
                        "text": "政策变化",
                        "pipeline": "zho",
                        "language": "zho",
                    },
                ],
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plans.jsonl"
            path.write_text(
                "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
                encoding="utf-8",
            )

            planner = load_query_plans(path)
            queries = planner.plan(topic())

        self.assertEqual([query.language for query in queries], ["spa", "zho"])
        self.assertEqual(queries[0].pipeline, "spa")
        self.assertEqual(queries[0].weight, 1.5)
        self.assertEqual(queries[0].text, "cambio de política")
        self.assertEqual(queries[1].text, "政策变化")

    def test_query_plan_errors_do_not_echo_payload_text(self):
        secret_marker = "DO-NOT-ECHO-QUERY"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plans.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "topic_id": "2000",
                        "queries": [
                            {"query_id": "q1", "text": secret_marker, "weight": True}
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(QueryPlanError) as raised:
                load_query_plans(path)

        self.assertNotIn(secret_marker, str(raised.exception))

    def test_rejects_duplicate_topics_and_query_ids(self):
        duplicate_queries = {
            "topic_id": "2000",
            "queries": [
                {"query_id": "same", "text": "one"},
                {"query_id": "same", "text": "two"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plans.jsonl"
            path.write_text(json.dumps(duplicate_queries), encoding="utf-8")

            with self.assertRaisesRegex(QueryPlanError, "duplicate query IDs"):
                load_query_plans(path)

        duplicate_topics = [
            {"topic_id": "2000", "queries": [{"query_id": "q1", "text": "one"}]},
            {"topic_id": "2000", "queries": [{"query_id": "q2", "text": "two"}]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plans.jsonl"
            path.write_text(
                "\n".join(json.dumps(record) for record in duplicate_topics),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(QueryPlanError, "repeats topic"):
                load_query_plans(path)


if __name__ == "__main__":
    unittest.main()
