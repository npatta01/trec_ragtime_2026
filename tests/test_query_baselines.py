from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trec_ragtime.client import RagtimeApiError
from trec_ragtime.models import (
    HuggingFaceDocument,
    HydrationResult,
    PipelineResult,
    RetrievalResult,
    RetrievedDocument,
    SearchHit,
)
from trec_ragtime.query_baselines import (
    PIPELINE,
    build_parser,
    fuse_ranked_hits,
    load_query_plans,
    run_query_baseline,
)


def _search_result(query: str, document_ids: list[str]) -> PipelineResult:
    return PipelineResult(
        query=query,
        pipeline="ragtime2",
        hits=tuple(
            SearchHit(rank=rank, document_id=document_id, score=1.0 / rank)
            for rank, document_id in enumerate(document_ids, start=1)
        ),
        processed=True,
        local_cache_hit=False,
        remote_cached=False,
        server_timestamp=None,
        cache_stored_at=None,
    )


def _retrieval_result(query: str, document_ids: list[str]) -> RetrievalResult:
    search = _search_result(query, document_ids)
    return RetrievalResult(
        search_result=search,
        documents=tuple(
            RetrievedDocument(
                rank=hit.rank,
                document_id=hit.document_id,
                score=hit.score,
                text=f"text {hit.document_id}",
                url=f"https://example.test/{hit.document_id}",
                date="2026-01-01",
                language="eng",
                content_source="local_corpus",
                content_local_cache_hit=True,
            )
            for hit in search.hits
        ),
    )


class FakeRetriever:
    def __init__(self, responses: dict[str, RetrievalResult | Exception]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str, int | None, int | None, bool]] = []

    def retrieve(
        self,
        query: str,
        *,
        pipeline: str,
        top_k: int | None,
        max_content_fallbacks: int | None,
        fallback_on_hf_error: bool,
    ) -> RetrievalResult:
        self.calls.append(
            (
                query,
                pipeline,
                top_k,
                max_content_fallbacks,
                fallback_on_hf_error,
            )
        )
        response = self.responses[query]
        if isinstance(response, Exception):
            raise response
        return response

    def enrich_documents(self, document_ids: list[str]) -> HydrationResult:
        return HydrationResult(
            documents=tuple(
                HuggingFaceDocument(
                    dataset="trec-ragtime/ragtime2",
                    config="default",
                    split="eng",
                    document_id=document_id,
                    text=f"text {document_id}",
                    url=f"https://example.test/{document_id}",
                    date="2026-01-01",
                    local_cache_hit=True,
                    cache_stored_at=None,
                )
                for document_id in document_ids
            ),
            unresolved_ids=(),
            failures=(),
        )


class QueryBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.plans_path = self.root / "query_plans.jsonl"
        plans = [
            {
                "topic_id": "2000",
                "plan_status": "model_generated",
                "single_query": "single zero",
                "facet_queries": ["facet zero a", "facet zero b"],
            },
            {
                "topic_id": "2001",
                "plan_status": "model_generated",
                "single_query": "single one",
                "facet_queries": ["facet one a", "facet one b"],
            },
        ]
        self.plans_path.write_text(
            "".join(json.dumps(plan) + "\n" for plan in plans),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_load_query_plans_preserves_literal_queries(self) -> None:
        plans = load_query_plans(self.plans_path)

        self.assertEqual([plan.topic_id for plan in plans], ["2000", "2001"])
        self.assertEqual(plans[0].single_query, "single zero")
        self.assertEqual(plans[0].facet_queries, ("facet zero a", "facet zero b"))

    def test_cli_defaults_use_alias_and_user_facing_depth(self) -> None:
        arguments = build_parser().parse_args(["b", "--run-id", "baseline-b"])

        self.assertEqual(arguments.top_k, 100)
        self.assertEqual(arguments.rrf_k, 60)
        self.assertEqual(PIPELINE, "ragtime2")

    def test_fuse_ranked_hits_uses_rrf_60_and_deterministic_ties(self) -> None:
        fused = fuse_ranked_hits(
            [
                _search_result("a", ["shared", "only-a"]).hits,
                _search_result("b", ["shared", "only-b"]).hits,
            ],
            rrf_k=60,
            top_k=100,
        )

        self.assertEqual([candidate["document_id"] for candidate in fused], ["shared", "only-a", "only-b"])
        self.assertAlmostEqual(fused[0]["fused_score"], 2 / 61)
        self.assertEqual(fused[0]["source_query_indices"], [1, 2])
        self.assertEqual(fused[0]["best_rank"], 1)
        self.assertEqual(fused[1]["rank"], 2)
        self.assertEqual(fused[2]["rank"], 3)

    def test_baseline_b_resumes_success_and_retries_failure(self) -> None:
        first_retriever = FakeRetriever(
            {
                "single zero": _retrieval_result("single zero", ["doc-a", "doc-b"]),
                "single one": RagtimeApiError("RAGTIME /pipeline request failed: HTTP 504"),
            }
        )

        run_dir = run_query_baseline(
            self.plans_path,
            baseline="b",
            retriever=first_retriever,
            output_root=self.root / "runs",
            run_id="baseline-b",
            top_k=100,
        )

        self.assertEqual(
            first_retriever.calls,
            [
                ("single zero", "ragtime2", 100, 0, False),
                ("single one", "ragtime2", 100, 0, False),
            ],
        )
        first_topics = {
            row["topic_id"]: row
            for row in map(json.loads, (run_dir / "topics.jsonl").read_text().splitlines())
        }
        self.assertEqual(first_topics["2000"]["status"], "success")
        self.assertEqual(first_topics["2000"]["candidate_count"], 2)
        self.assertEqual(first_topics["2001"]["status"], "error")

        second_retriever = FakeRetriever(
            {"single one": _retrieval_result("single one", ["doc-c"])}
        )
        run_query_baseline(
            self.plans_path,
            baseline="b",
            retriever=second_retriever,
            output_root=self.root / "runs",
            run_id="baseline-b",
            top_k=100,
        )

        self.assertEqual(
            second_retriever.calls,
            [("single one", "ragtime2", 100, 0, False)],
        )
        resumed_topics = {
            row["topic_id"]: row
            for row in map(json.loads, (run_dir / "topics.jsonl").read_text().splitlines())
        }
        self.assertEqual(resumed_topics["2001"]["status"], "success")
        manifest = json.loads((run_dir / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["counts"]["successful_topics"], 2)
        self.assertEqual(manifest["counts"]["failed_queries"], 0)

    def test_baseline_c_labels_partial_and_error_topics(self) -> None:
        retriever = FakeRetriever(
            {
                "facet zero a": RagtimeApiError("RAGTIME /pipeline request failed: HTTP 504"),
                "facet zero b": _retrieval_result("facet zero b", ["shared", "doc-a"]),
                "facet one a": RagtimeApiError("RAGTIME /pipeline request failed: HTTP 504"),
                "facet one b": RagtimeApiError("RAGTIME /pipeline request failed: HTTP 504"),
            }
        )

        run_dir = run_query_baseline(
            self.plans_path,
            baseline="c",
            retriever=retriever,
            output_root=self.root / "runs",
            run_id="baseline-c",
            top_k=100,
            rrf_k=60,
        )

        topics = {
            row["topic_id"]: row
            for row in map(json.loads, (run_dir / "topics.jsonl").read_text().splitlines())
        }
        self.assertEqual(topics["2000"]["status"], "partial")
        self.assertEqual(topics["2000"]["candidate_count"], 2)
        self.assertEqual(topics["2001"]["status"], "error")
        self.assertEqual(topics["2001"]["candidate_count"], 0)
        self.assertEqual(topics["2000"]["candidates"][0]["fused_score"], 1 / 61)
        self.assertEqual(topics["2000"]["candidates"][0]["source_query_indices"], [2])
        self.assertIsNone(topics["2000"]["candidates"][0]["score"])
        self.assertEqual(topics["2000"]["query_results"][0]["status"], "error")


if __name__ == "__main__":
    unittest.main()
