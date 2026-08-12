"""Topic selection, facet numbering, and cache-only retrieval for the CLI."""

from __future__ import annotations

import hashlib
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from trec_ragtime.nuggets_cli import (
    PlannedFacet,
    build_parser,
    plan_queries,
    prepare_topic,
    select_plans,
)
from trec_ragtime.query_baselines import QueryPlan


def plan(topic_id: str = "2000") -> QueryPlan:
    return QueryPlan(
        topic_id=topic_id,
        plan_status="ok",
        single_query="grain exports",
        facet_queries=("port closures", "insurance withdrawal"),
    )


class RecordingRetriever:
    """Serves cached retrievals for named queries only, like the real one."""

    def __init__(self, available: dict[str, str]) -> None:
        self.available = available
        self.calls: list[str] = []
        self.contexts: list[Any] = []

    def get_cached_retrieval(self, query: str, **kwargs: Any) -> Any:
        self.calls.append(query)
        self.contexts.append(kwargs.get("query_context"))
        assert kwargs.get("max_content_fallbacks") == 0
        assert kwargs.get("fallback_on_hf_error") is False
        document_id = self.available.get(query)
        if document_id is None:
            return None
        return SimpleNamespace(
            query=query,
            retrieval_artifact=SimpleNamespace(
                object_key=f"retrieval/v1/aa/bb/{document_id}.mpk",
                request_digest=hashlib.sha256(document_id.encode()).digest(),
            ),
            documents=(),
        )


def fake_reranking(document_id: str) -> Any:
    return SimpleNamespace(
        query="q",
        representation="report_text",
        documents=(
            SimpleNamespace(
                document=SimpleNamespace(
                    document_id=document_id,
                    language="eng",
                    text="hello passage text",
                    report_text="hello passage text",
                ),
                score=0.5,
                evidence=(
                    SimpleNamespace(
                        document_id=document_id,
                        start_char=0,
                        end_char=5,
                        score=0.5,
                        content_hash=hashlib.sha256(b"hello").digest(),
                    ),
                ),
                reranked_rank=1,
                status="scored",
            ),
        ),
    )


class PlanQueriesTestCase(unittest.TestCase):
    def test_facet_zero_is_the_single_baseline_b_query(self) -> None:
        self.assertEqual(
            plan_queries(plan()),
            (
                PlannedFacet(0, "grain exports", "b", 1),
                PlannedFacet(1, "port closures", "c", 1),
                PlannedFacet(2, "insurance withdrawal", "c", 2),
            ),
        )

    def test_retrieval_query_indices_are_one_based(self) -> None:
        """The baseline runner numbers from 1; a 0-based guess misses every artifact."""

        self.assertEqual(
            [(f.baseline, f.query_index) for f in plan_queries(plan())],
            [("b", 1), ("c", 1), ("c", 2)],
        )


class SelectPlansTestCase(unittest.TestCase):
    plans = (plan("2000"), plan("2001"))

    def test_returns_everything_when_unfiltered(self) -> None:
        self.assertEqual(select_plans(self.plans, None), self.plans)

    def test_filters_to_the_requested_topics(self) -> None:
        self.assertEqual(
            [p.topic_id for p in select_plans(self.plans, ["2001"])], ["2001"]
        )

    def test_rejects_a_topic_the_plan_does_not_have(self) -> None:
        with self.assertRaisesRegex(ValueError, "2099"):
            select_plans(self.plans, ["2000", "2099"])


class PrepareTopicTestCase(unittest.TestCase):
    """The facet-numbering invariant lives here and nowhere else."""

    def _prepare(self, available: dict[str, str]) -> Any:
        """Stub the reranking stage; this test is about facet bookkeeping."""

        retriever = RecordingRetriever(available)
        counter = {"n": 0}

        def fake_rerank(retrieval: Any, **_: Any) -> Any:
            counter["n"] += 1
            return fake_reranking(f"doc-{counter['n']}")

        with (
            patch("trec_ragtime.passages.PassageChunker") as chunker,
            patch("trec_ragtime.passage_scoring.CrossEncoderPairLengthFunction"),
            patch("trec_ragtime.passage_score_cache.PassageScoreCache") as score_cache,
            patch("trec_ragtime.passage_score_cache.CachedPassageScorer"),
            patch("trec_ragtime.reranking.rerank_retrieval", fake_rerank),
        ):
            chunker.return_value.fingerprint = "chunker-v1"
            score_cache.return_value.fingerprint = "scorer-v1"
            return retriever, prepare_topic(
                plan(),
                narrative="I need a report on grain exports.",
                plan_sha256="0" * 64,
                retriever=retriever,
                store=object(),
                scorer=object(),
                scorer_config=object(),
                representation="report_text",
                top_k=100,
            )

    def test_keeps_plan_numbering_when_a_middle_facet_is_missing(self) -> None:
        # Facet 1 has no cached retrieval; facet 2 must stay facet 2.
        _, work = self._prepare(
            {"grain exports": "a", "insurance withdrawal": "c"}
        )
        self.assertEqual([facet.facet_index for facet in work.facets], [0, 2])
        self.assertEqual([row.facet_index for row in work.inputs], [0, 2])
        self.assertEqual(
            [facet.query for facet in work.facets],
            ["grain exports", "insurance withdrawal"],
        )

    def test_facets_and_inputs_agree_pairwise(self) -> None:
        _, work = self._prepare(
            {"grain exports": "a", "port closures": "b", "insurance withdrawal": "c"}
        )
        self.assertEqual(
            {(f.facet_index, f.query) for f in work.facets},
            {(row.facet_index, row.query) for row in work.inputs},
        )

    def test_reports_a_topic_with_no_cached_retrieval(self) -> None:
        retriever, work = self._prepare({})
        self.assertIsNone(work)
        # Every planned facet was looked up, and none became a live request.
        self.assertEqual(len(retriever.calls), 3)

    def test_carries_the_retrieval_artifact_reference(self) -> None:
        _, work = self._prepare({"grain exports": "a"})
        self.assertEqual(work.inputs[0].object_key, "retrieval/v1/aa/bb/a.mpk")
        self.assertEqual(work.topic_id, "2000")
        self.assertIn("report_text", work.passage_source_fingerprint)

    def test_the_nuggetizer_query_is_the_narrative_not_the_search_query(self) -> None:
        _, work = self._prepare({"grain exports": "a"})
        self.assertEqual(work.query, "I need a report on grain exports.")

    def test_looks_retrievals_up_with_the_baseline_identity(self) -> None:
        """A lookup that omits this context misses every stored artifact."""

        retriever, _ = self._prepare({"grain exports": "a"})
        first = retriever.contexts[0]
        self.assertEqual(first["kind"], "baseline_b")
        self.assertEqual(first["query_index"], 1)
        self.assertEqual(
            [c["kind"] for c in retriever.contexts],
            ["baseline_b", "baseline_c", "baseline_c"],
        )
        self.assertEqual([c["query_index"] for c in retriever.contexts], [1, 1, 2])


class ParserTestCase(unittest.TestCase):
    def test_requires_a_model(self) -> None:
        with self.assertRaises(SystemExit):
            build_parser().parse_args([])

    def test_defaults_to_the_english_representation(self) -> None:
        arguments = build_parser().parse_args(["--model", "m"])
        self.assertEqual(arguments.representation, "report_text")
        self.assertIsNone(arguments.topic_ids)


if __name__ == "__main__":
    unittest.main()
