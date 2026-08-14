from __future__ import annotations

import importlib
import unittest

from trec_ragtime.models import RetrievedDocument
from trec_ragtime.reranking import (
    PassageEvidence,
    RerankedDocument,
    RerankingResult,
)


def reranking(query: str, document_id: str, score: float) -> RerankingResult:
    text = f"evidence for {document_id}"
    document = RetrievedDocument(
        rank=1,
        document_id=document_id,
        score=0.5,
        text=text,
        url=None,
        date=None,
        language="eng",
        content_source="local_corpus",
        content_local_cache_hit=True,
    )
    evidence = PassageEvidence(
        document_id=document_id,
        start_char=0,
        end_char=len(text),
        score=score,
        content_hash=__import__("hashlib").sha256(text.encode()).digest(),
    )
    return RerankingResult(
        query=query,
        representation="report_text",
        documents=(RerankedDocument(document, score, (evidence,), 1, "scored"),),
    )


class StaticNuggetWorkTests(unittest.TestCase):
    def test_assembles_existing_reranking_results_without_changing_query_facets(self) -> None:
        try:
            module = importlib.import_module("trec_ragtime.static_nugget_work")
        except ModuleNotFoundError:
            module = None
        self.assertIsNotNone(module, "static_nugget_work module is missing")

        rows = (
            module.StaticRerankingInput(
                facet_index=0,
                query="query one",
                object_key="retrieval/v1/a",
                request_digest=b"a" * 32,
                source_fingerprint="source-one",
                result=reranking("query one", "doc-a", 0.9),
            ),
            module.StaticRerankingInput(
                facet_index=1,
                query="query two",
                object_key="retrieval/v1/b",
                request_digest=b"b" * 32,
                source_fingerprint="source-two",
                result=reranking("query two", "doc-b", 0.8),
            ),
        )

        work = module.build_static_topic_work(
            topic_id="2000",
            narrative="Full topic narrative",
            representation="report_text",
            rows=rows,
        )

        self.assertEqual(work.topic_id, "2000")
        self.assertEqual(work.query, "Full topic narrative")
        self.assertEqual(
            [(row.facet_index, row.query) for row in work.facets],
            [(0, "query one"), (1, "query two")],
        )
        self.assertEqual(
            [(row.facet_index, row.object_key) for row in work.inputs],
            [(0, "retrieval/v1/a"), (1, "retrieval/v1/b")],
        )
        self.assertEqual([row.document_id for row in work.facets[0].candidates], ["doc-a"])
        self.assertEqual([row.document_id for row in work.facets[1].candidates], ["doc-b"])
        self.assertTrue(work.passage_source_fingerprint.startswith("static-plan-reranking-v1:"))


if __name__ == "__main__":
    unittest.main()
