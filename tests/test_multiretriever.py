from __future__ import annotations

import unittest

from trec_ragtime import RagtimeRetriever
from trec_ragtime.fusion import RetrievalQuery
from trec_ragtime.models import (
    HuggingFaceDocument,
    HydrationFailure,
    HydrationResult,
    PipelineResult,
    SearchHit,
)
from trec_ragtime.multiretriever import (
    MultiQueryRetrievalError,
    MultiQueryRetriever,
)


def pipeline_result(
    query: str,
    pipeline: str,
    document_ids: list[str],
) -> PipelineResult:
    return PipelineResult(
        query=query,
        pipeline=pipeline,
        hits=tuple(
            SearchHit(
                rank=rank,
                document_id=document_id,
                score=float(len(document_ids) - rank + 1),
            )
            for rank, document_id in enumerate(document_ids, start=1)
        ),
        processed=True,
        local_cache_hit=False,
        remote_cached=False,
        server_timestamp=None,
        cache_stored_at=None,
    )


def document(
    document_id: str,
    *,
    split: str = "eng",
    english_text: str | None = None,
) -> HuggingFaceDocument:
    return HuggingFaceDocument(
        dataset="trec-ragtime/ragtime2",
        config="default",
        split=split,
        document_id=document_id,
        text=f"source {document_id}",
        url=None,
        date=None,
        local_cache_hit=None,
        cache_stored_at=None,
        source="test",
        english_text=english_text,
    )


class FakeClient:
    def __init__(self, results: dict[tuple[str, str], PipelineResult]) -> None:
        self.results = results
        self.calls: list[dict[str, object]] = []
        self.cache_path = None

    def search(self, query: str, *, pipeline: str, refresh: bool) -> PipelineResult:
        self.calls.append(
            {"query": query, "pipeline": pipeline, "refresh": refresh}
        )
        return self.results[(query, pipeline)]

    def close(self) -> None:
        pass


class FakeCorpus:
    backend_name = "fake"

    def __init__(self, hydration: HydrationResult) -> None:
        self.hydration = hydration
        self.calls: list[dict[str, object]] = []

    def enrich_documents(
        self,
        document_ids,
        *,
        refresh: bool = False,
    ) -> HydrationResult:
        self.calls.append(
            {"document_ids": tuple(document_ids), "refresh": refresh}
        )
        return self.hydration

    get_many = enrich_documents

    def close(self) -> None:
        pass


class MultiQueryRetrieverTests(unittest.TestCase):
    def test_fuses_all_searches_then_hydrates_once_in_fused_order(self):
        first = RetrievalQuery("q1", "first", pipeline="ragtime2%100")
        second = RetrievalQuery("q2", "second", pipeline="spa", language="spa")
        client = FakeClient(
            {
                (first.text, first.pipeline): pipeline_result(
                    first.text, first.pipeline, ["doc-a", "doc-b"]
                ),
                (second.text, second.pipeline): pipeline_result(
                    second.text, second.pipeline, ["doc-b", "doc-c"]
                ),
            }
        )
        corpus = FakeCorpus(
            HydrationResult(
                documents=(document("doc-c"), document("doc-b"), document("doc-a")),
                unresolved_ids=(),
                failures=(),
            )
        )
        retriever = MultiQueryRetriever(RagtimeRetriever(client, corpus))

        result = retriever.retrieve(
            [first, second],
            max_documents=100,
            refresh_search=True,
            refresh_documents=True,
        )

        self.assertEqual(
            [hit.document_id for hit in result.fused_ranking.hits],
            ["doc-b", "doc-a", "doc-c"],
        )
        self.assertEqual(
            [item.document_id for item in result.documents],
            ["doc-b", "doc-a", "doc-c"],
        )
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(
            corpus.calls,
            [
                {
                    "document_ids": ("doc-b", "doc-a", "doc-c"),
                    "refresh": True,
                }
            ],
        )
        self.assertTrue(result.fully_report_ready)

        with self.assertRaisesRegex(ValueError, "fused-prefix order"):
            type(result)(
                rankings=result.rankings,
                fused_ranking=result.fused_ranking,
                documents=tuple(reversed(result.documents)),
                hydrated_hit_count=result.hydrated_hit_count,
            )
        with self.assertRaisesRegex(ValueError, "fused ranking queries"):
            type(result)(
                rankings=tuple(reversed(result.rankings)),
                fused_ranking=result.fused_ranking,
                documents=result.documents,
                hydrated_hit_count=result.hydrated_hit_count,
            )

    def test_hydration_prefix_does_not_truncate_submitted_ranking(self):
        query = RetrievalQuery("q1", "first")
        client = FakeClient(
            {
                (query.text, query.pipeline): pipeline_result(
                    query.text,
                    query.pipeline,
                    ["doc-a", "doc-b", "doc-c"],
                )
            }
        )
        corpus = FakeCorpus(
            HydrationResult(
                documents=(document("doc-a"), document("doc-b")),
                unresolved_ids=(),
                failures=(),
            )
        )

        result = MultiQueryRetriever(RagtimeRetriever(client, corpus)).retrieve(
            [query],
            max_documents=3,
            hydrate_top_k=2,
        )

        self.assertEqual(len(result.fused_ranking.hits), 3)
        self.assertEqual(result.hydrated_hit_count, 2)
        self.assertEqual([item.document_id for item in result.documents], [
            "doc-a",
            "doc-b",
        ])

    def test_missing_translation_and_failures_block_generation(self):
        query = RetrievalQuery("q1", "first")
        client = FakeClient(
            {
                (query.text, query.pipeline): pipeline_result(
                    query.text, query.pipeline, ["doc-spa", "doc-missing"]
                )
            }
        )
        corpus = FakeCorpus(
            HydrationResult(
                documents=(document("doc-spa", split="spa"),),
                unresolved_ids=("doc-missing",),
                failures=(HydrationFailure(split="spa", message="lookup failed"),),
                missing_translation_ids=("doc-spa",),
            )
        )

        result = MultiQueryRetriever(RagtimeRetriever(client, corpus)).retrieve([query])

        self.assertEqual(result.unresolved_ids, ("doc-missing",))
        self.assertEqual(result.missing_translation_ids, ("doc-spa",))
        self.assertFalse(result.fully_report_ready)
        with self.assertRaisesRegex(
            MultiQueryRetrievalError,
            "unresolved document IDs.*missing English translations.*hydration failures",
        ):
            result.require_report_ready()

    def test_empty_search_result_is_not_report_ready(self):
        query = RetrievalQuery("q1", "first")
        client = FakeClient(
            {
                (query.text, query.pipeline): pipeline_result(
                    query.text, query.pipeline, []
                )
            }
        )
        corpus = FakeCorpus(HydrationResult((), (), ()))

        result = MultiQueryRetriever(RagtimeRetriever(client, corpus)).retrieve([query])

        self.assertEqual(result.documents, ())
        self.assertFalse(result.fully_report_ready)
        with self.assertRaisesRegex(MultiQueryRetrievalError, "report-ready text"):
            result.require_report_ready()

    def test_controls_are_strictly_validated(self):
        query = RetrievalQuery("q1", "first")
        result = pipeline_result(query.text, query.pipeline, [])
        retriever = MultiQueryRetriever(
            RagtimeRetriever(
                FakeClient({(query.text, query.pipeline): result}),
                FakeCorpus(HydrationResult((), (), ())),
            )
        )

        for options in (
            {"rrf_k": 0},
            {"rrf_k": float("nan")},
            {"max_documents": True},
            {"max_documents": 1001},
            {"hydrate_top_k": 0},
            {"hydrate_top_k": True},
            {"refresh_search": 1},
            {"refresh_documents": "yes"},
        ):
            with self.subTest(options=options):
                with self.assertRaises((TypeError, ValueError)):
                    retriever.retrieve([query], **options)

        duplicate = RetrievalQuery("q1", "another query")
        with self.assertRaisesRegex(ValueError, "query IDs must be unique"):
            retriever.retrieve([query, duplicate])
        self.assertEqual(retriever.retriever.client.calls, [])


if __name__ == "__main__":
    unittest.main()
