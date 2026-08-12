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


class RerankDocumentCapTestCase(unittest.TestCase):
    """A smoke-test escape hatch. Real runs rerank everything retrieved, because
    truncating by organizer rank removes the documents reranking exists to
    promote."""

    def _retrieval(self, count: int) -> Any:
        from trec_ragtime.models import (
            PipelineResult, RetrievalResult, RetrievedDocument, SearchHit)
        documents = tuple(
            RetrievedDocument(
                rank=i, document_id=f"doc-{i}", score=1.0 - i / 100,
                text=f"text {i}", url=None, date=None, language="eng",
                content_source="fixture", content_local_cache_hit=True,
                english_text=f"text {i}",
            )
            for i in range(1, count + 1)
        )
        return RetrievalResult(
            search_result=PipelineResult(
                query="q", pipeline="ragtime2",
                hits=tuple(SearchHit(d.rank, d.document_id, d.score) for d in documents),
                processed=True, local_cache_hit=True, remote_cached=True,
                server_timestamp=None, cache_stored_at=None,
            ),
            documents=documents,
        )

    def test_keeps_the_highest_ranked_documents(self) -> None:
        from trec_ragtime.nuggets_cli import _head

        capped = _head(self._retrieval(100), 20)
        self.assertEqual(len(capped.documents), 20)
        self.assertEqual([d.rank for d in capped.documents], list(range(1, 21)))

    def test_hits_stay_consistent_with_the_kept_documents(self) -> None:
        from trec_ragtime.nuggets_cli import _head

        capped = _head(self._retrieval(100), 20)
        self.assertEqual(
            {h.document_id for h in capped.search_result.hits},
            {d.document_id for d in capped.documents},
        )

    def test_a_cap_above_the_result_size_changes_nothing(self) -> None:
        from trec_ragtime.nuggets_cli import _head

        self.assertEqual(len(_head(self._retrieval(5), 20).documents), 5)


class ScorerTransportTestCase(unittest.TestCase):
    def test_local_is_the_default_and_honours_device(self) -> None:
        from contextlib import ExitStack

        from trec_ragtime.nuggets_cli import _open_scorer

        arguments = build_parser().parse_args(["--model", "m", "--device", "cuda"])
        with ExitStack() as stack:
            config, scorer = _open_scorer(arguments, stack)
        self.assertEqual(config.effective_device, "cuda")
        self.assertEqual(type(scorer).__name__, "LocalCrossEncoderScorer")

    def test_rocm_is_normalised_to_the_torch_device(self) -> None:
        from contextlib import ExitStack

        from trec_ragtime.nuggets_cli import _open_scorer

        arguments = build_parser().parse_args(["--model", "m", "--device", "rocm"])
        with ExitStack() as stack:
            config, _ = _open_scorer(arguments, stack)
        self.assertEqual(config.effective_device, "cuda")

    def test_flash_without_an_endpoint_is_refused(self) -> None:
        from contextlib import ExitStack

        from trec_ragtime.nuggets_cli import _open_scorer

        arguments = build_parser().parse_args(["--model", "m", "--scorer", "flash"])
        with ExitStack() as stack:
            with self.assertRaisesRegex(ValueError, "flash-endpoint-id"):
                _open_scorer(arguments, stack)

    def test_flash_scales_to_a_range_starting_at_zero(self) -> None:
        """workers is the throughput bound; the minimum must stay 0 to stop billing."""

        from unittest.mock import patch

        from trec_ragtime.nuggets_cli import _open_scorer
        from contextlib import ExitStack

        arguments = build_parser().parse_args(
            ["--model", "m", "--scorer", "flash", "--flash-endpoint-id", "ep-1",
             "--flash-workers", "8"]
        )
        seen = {}

        class _Config:
            def __init__(self, **kwargs: Any) -> None:
                seen.update(kwargs)
                self.effective_model_config = object()

        with (
            patch("trec_ragtime.flash_passage_scorer.RunpodFlashConfig", _Config),
            patch("trec_ragtime.flash_passage_scorer.FlashEndpointBatchRunner"),
            patch("trec_ragtime.flash_passage_scorer.RunpodFlashPassageScorer"),
            ExitStack() as stack,
        ):
            _open_scorer(arguments, stack)
        self.assertEqual(seen["workers"], (0, 8))
        self.assertEqual(seen["concurrency"], 8)

    def test_flash_rejects_a_worker_count_below_one(self) -> None:
        from contextlib import ExitStack

        from trec_ragtime.nuggets_cli import _open_scorer

        arguments = build_parser().parse_args(
            ["--model", "m", "--scorer", "flash", "--flash-endpoint-id", "ep-1",
             "--flash-workers", "0"]
        )
        with ExitStack() as stack:
            with self.assertRaisesRegex(ValueError, "flash-workers"):
                _open_scorer(arguments, stack)

    def test_flash_exits_two_rather_than_traceback(self) -> None:
        from trec_ragtime.nuggets_cli import main

        self.assertEqual(main(["--model", "m", "--scorer", "flash"]), 2)


class ParserTestCase(unittest.TestCase):
    def test_requires_a_model(self) -> None:
        with self.assertRaises(SystemExit):
            build_parser().parse_args([])

    def test_defaults_to_the_english_representation(self) -> None:
        arguments = build_parser().parse_args(["--model", "m"])
        self.assertEqual(arguments.representation, "report_text")
        self.assertIsNone(arguments.topic_ids)
        # Uncapped by default: the cap is an explicit cost decision.
        self.assertIsNone(arguments.rerank_documents)


if __name__ == "__main__":
    unittest.main()


class ScorerIdentityTestCase(unittest.TestCase):
    """The device is part of the scorer fingerprint, so it selects the cache."""

    def test_device_changes_which_scores_are_reachable(self) -> None:
        from trec_ragtime.passage_scoring import LocalCrossEncoderConfig

        gpu = LocalCrossEncoderConfig(device="cuda")
        cpu = LocalCrossEncoderConfig(device="cpu")
        # Scores computed on a GPU are invisible to a CPU-identity run, which
        # would silently recompute all of them instead of reusing the work.
        self.assertNotEqual(gpu.fingerprint, cpu.fingerprint)

    def test_the_remote_batch_size_matches_the_cli_default(self) -> None:
        """A batch-size drift would isolate the cache just as a device would."""

        from trec_ragtime.passage_scoring import DEFAULT_BATCH_SIZE, LocalCrossEncoderConfig

        self.assertEqual(DEFAULT_BATCH_SIZE, 32)
        self.assertEqual(
            LocalCrossEncoderConfig(device="cuda", batch_size=32).fingerprint,
            LocalCrossEncoderConfig(device="cuda").fingerprint,
        )

    def test_a_gpu_identity_scorer_builds_without_a_gpu(self) -> None:
        """Consuming GPU scores locally must not require CUDA to be present."""

        from trec_ragtime.passage_scoring import LocalCrossEncoderConfig, LocalCrossEncoderScorer

        scorer = LocalCrossEncoderScorer(LocalCrossEncoderConfig(device="cuda"))
        self.assertIsNone(scorer._model)
