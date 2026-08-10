from __future__ import annotations

import math
import unittest
from dataclasses import FrozenInstanceError

from trec_ragtime.fusion import (
    FusedHit,
    QueryRanking,
    RetrievalQuery,
    fuse_rankings,
    weighted_reciprocal_rank_fusion,
)
from trec_ragtime.models import PipelineResult, SearchHit


def pipeline_result(
    query: str,
    *hits: SearchHit,
    pipeline: str = "ragtime2",
) -> PipelineResult:
    return PipelineResult(
        query=query,
        pipeline=pipeline,
        hits=tuple(hits),
        processed=True,
        local_cache_hit=False,
        remote_cached=False,
        server_timestamp=None,
        cache_stored_at=None,
    )


class RetrievalQueryTests(unittest.TestCase):
    def test_query_is_immutable_and_normalizes_numeric_weight(self):
        query = RetrievalQuery(
            query_id="q-es",
            text="texto de consulta",
            pipeline="spa",
            weight=2,
            language="spa",
        )

        self.assertEqual(query.weight, 2.0)
        with self.assertRaises(FrozenInstanceError):
            query.weight = 3.0  # type: ignore[misc]

    def test_query_rejects_empty_identifiers_text_pipeline_and_language(self):
        cases = (
            {"query_id": " ", "text": "query"},
            {"query_id": "q1", "text": " "},
            {"query_id": "q1", "text": "query", "pipeline": ""},
            {"query_id": "q1", "text": "query", "language": "\t"},
            {"query_id": " q1", "text": "query"},
            {"query_id": "q1", "text": "query", "pipeline": "spa "},
            {"query_id": "q1", "text": "query", "language": " spa"},
        )
        for values in cases:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    RetrievalQuery(**values)

    def test_query_weight_must_be_positive_and_finite(self):
        for weight in (0, -1, math.inf, -math.inf, math.nan, True, "1"):
            with self.subTest(weight=weight):
                with self.assertRaisesRegex(ValueError, "weight"):
                    RetrievalQuery(  # type: ignore[arg-type]
                        "q1",
                        "query",
                        weight=weight,
                    )


class QueryRankingTests(unittest.TestCase):
    def test_result_must_match_query_text_and_pipeline(self):
        query = RetrievalQuery("q1", "planned", pipeline="eng")

        with self.assertRaisesRegex(ValueError, "result query"):
            QueryRanking(query, pipeline_result("different", pipeline=query.pipeline))
        with self.assertRaisesRegex(ValueError, "result pipeline"):
            QueryRanking(query, pipeline_result(query.text, pipeline="ragtime2"))

    def test_matching_result_is_accepted(self):
        query = RetrievalQuery("q1", "planned", pipeline="eng")
        result = pipeline_result(query.text, pipeline=query.pipeline)

        self.assertEqual(QueryRanking(query, result).result, result)


class WeightedReciprocalRankFusionTests(unittest.TestCase):
    def test_fuses_by_weighted_rank_and_retains_ordered_provenance(self):
        first = RetrievalQuery("q1", "first", weight=2.0, language="eng")
        second = RetrievalQuery("q2", "second", weight=1.0, language="spa")
        rankings = (
            QueryRanking(
                first,
                pipeline_result(
                    first.text,
                    SearchHit(rank=1, document_id="doc-b", score=-1000.0),
                    SearchHit(rank=2, document_id="doc-shared", score=0.01),
                ),
            ),
            QueryRanking(
                second,
                pipeline_result(
                    second.text,
                    SearchHit(rank=1, document_id="doc-shared", score=9999.0),
                    SearchHit(rank=2, document_id="doc-c", score=10000.0),
                ),
            ),
        )

        fused = weighted_reciprocal_rank_fusion(rankings, k=10)

        self.assertEqual(fused.query_ids, ("q1", "q2"))
        self.assertEqual(
            [hit.document_id for hit in fused.hits],
            ["doc-shared", "doc-b", "doc-c"],
        )
        self.assertAlmostEqual(fused.hits[0].score, (2.0 / 12.0) + (1.0 / 11.0))
        self.assertAlmostEqual(fused.hits[1].score, 2.0 / 11.0)
        self.assertAlmostEqual(fused.hits[2].score, 1.0 / 12.0)
        self.assertEqual(
            [item.query_id for item in fused.hits[0].provenance],
            ["q1", "q2"],
        )
        self.assertEqual(
            [item.source_rank for item in fused.hits[0].provenance],
            [2, 1],
        )
        self.assertEqual(
            [item.source_score for item in fused.hits[0].provenance],
            [0.01, 9999.0],
        )

    def test_equal_fused_scores_are_ordered_by_document_id(self):
        first = RetrievalQuery("q1", "first")
        second = RetrievalQuery("q2", "second")
        rankings = (
            QueryRanking(
                first,
                pipeline_result(
                    first.text,
                    SearchHit(rank=1, document_id="doc-z", score=-100.0),
                    SearchHit(rank=2, document_id="doc-a", score=500.0),
                ),
            ),
            QueryRanking(
                second,
                pipeline_result(
                    second.text,
                    SearchHit(rank=1, document_id="doc-a", score=-999.0),
                    SearchHit(rank=2, document_id="doc-z", score=1000.0),
                ),
            ),
        )

        fused = weighted_reciprocal_rank_fusion(rankings)

        self.assertEqual(
            [hit.document_id for hit in fused.hits],
            ["doc-a", "doc-z"],
        )
        self.assertEqual(fused.hits[0].score, fused.hits[1].score)

    def test_duplicate_document_within_query_uses_best_source_rank_once(self):
        query = RetrievalQuery("q1", "query", weight=3.0)
        ranking = QueryRanking(
            query,
            pipeline_result(
                query.text,
                SearchHit(rank=3, document_id="duplicate", score=10000.0),
                SearchHit(rank=2, document_id="other", score=0.0),
                SearchHit(rank=1, document_id="duplicate", score=-10000.0),
            ),
        )

        fused = weighted_reciprocal_rank_fusion((ranking,), k=5)

        duplicate = fused.hits[0]
        self.assertEqual(duplicate.document_id, "duplicate")
        self.assertAlmostEqual(duplicate.score, 3.0 / 6.0)
        self.assertEqual(len(duplicate.provenance), 1)
        self.assertEqual(duplicate.provenance[0].source_rank, 1)
        self.assertEqual(duplicate.provenance[0].source_score, -10000.0)

    def test_max_docs_truncates_and_reranks(self):
        query = RetrievalQuery("q1", "query")
        ranking = QueryRanking(
            query,
            pipeline_result(
                query.text,
                SearchHit(rank=1, document_id="doc-1", score=1.0),
                SearchHit(rank=2, document_id="doc-2", score=0.9),
                SearchHit(rank=3, document_id="doc-3", score=0.8),
            ),
        )

        fused = weighted_reciprocal_rank_fusion((ranking,), max_docs=2)

        self.assertEqual(fused.max_docs, 2)
        self.assertEqual([hit.document_id for hit in fused.hits], ["doc-1", "doc-2"])
        self.assertEqual([hit.rank for hit in fused.hits], [1, 2])

    def test_duplicate_query_ids_are_rejected(self):
        first = RetrievalQuery("same", "first")
        second = RetrievalQuery("same", "second")
        rankings = (
            QueryRanking(first, pipeline_result(first.text)),
            QueryRanking(second, pipeline_result(second.text)),
        )

        with self.assertRaisesRegex(ValueError, "query IDs must be unique"):
            weighted_reciprocal_rank_fusion(rankings)

    def test_k_must_be_positive_and_finite(self):
        query = RetrievalQuery("q1", "query")
        rankings = (QueryRanking(query, pipeline_result(query.text)),)
        for k in (0, -1, math.inf, -math.inf, math.nan, True, "60"):
            with self.subTest(k=k):
                with self.assertRaisesRegex(ValueError, "k"):
                    weighted_reciprocal_rank_fusion(  # type: ignore[arg-type]
                        rankings,
                        k=k,
                    )

    def test_max_docs_must_be_an_integer_from_one_through_one_thousand(self):
        query = RetrievalQuery("q1", "query")
        rankings = (QueryRanking(query, pipeline_result(query.text)),)
        for max_docs in (0, -1, 1001, True, 1.5, "10"):
            with self.subTest(max_docs=max_docs):
                with self.assertRaisesRegex(ValueError, "max_docs"):
                    weighted_reciprocal_rank_fusion(  # type: ignore[arg-type]
                        rankings,
                        max_docs=max_docs,
                    )

    def test_empty_rankings_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            weighted_reciprocal_rank_fusion(())

    def test_source_hit_rank_and_score_are_validated(self):
        query = RetrievalQuery("q1", "query")
        invalid_hits = (
            SearchHit(rank=0, document_id="doc", score=1.0),
            SearchHit(rank=1, document_id="doc", score=math.nan),
        )
        for hit in invalid_hits:
            with self.subTest(hit=hit):
                ranking = QueryRanking(query, pipeline_result(query.text, hit))
                with self.assertRaises(ValueError):
                    weighted_reciprocal_rank_fusion((ranking,))

    def test_concise_alias_returns_the_same_immutable_result(self):
        query = RetrievalQuery("q1", "query")
        ranking = QueryRanking(
            query,
            pipeline_result(
                query.text,
                SearchHit(rank=1, document_id="doc", score=1.0),
            ),
        )

        explicit = weighted_reciprocal_rank_fusion((ranking,))
        concise = fuse_rankings((ranking,))

        self.assertEqual(concise, explicit)
        self.assertIsInstance(concise.hits[0], FusedHit)
        with self.assertRaises(FrozenInstanceError):
            concise.hits[0].score = 0.0  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
