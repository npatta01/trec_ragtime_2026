from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from trec_ragtime.client import RagtimeApiError
from trec_ragtime.local_corpus import LocalCorpusReadError
from trec_ragtime.models import (
    HuggingFaceDocument,
    HydrationResult,
    PipelineResult,
    RetrievalResult,
    RetrievedDocument,
    SearchHit,
)
from trec_ragtime.retrieval_cache import (
    RetrievalArtifactReference,
    retrieval_object_key,
)
from trec_ragtime.query_baselines import (
    PIPELINE,
    _atomic_write,
    build_parser,
    default_query_plan_path,
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


def _retrieval_result(
    query: str,
    document_ids: list[str],
    *,
    missing_translation_ids: tuple[str, ...] = (),
) -> RetrievalResult:
    search = _search_result(query, document_ids)
    request_digest = hashlib.sha256(query.encode("utf-8")).digest()
    fingerprint = hashlib.sha256(b"baseline-test-retriever").digest()
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
        missing_translation_ids=missing_translation_ids,
        retrieval_artifact=RetrievalArtifactReference(
            object_key=retrieval_object_key(fingerprint, request_digest),
            request_digest=request_digest,
            retriever_fingerprint=fingerprint,
            stored_at=123.0,
            cache_hit=False,
        ),
    )


class FakeRetriever:
    def __init__(
        self,
        responses: dict[str, RetrievalResult | Exception],
        *,
        corpus_revision: str = "test-revision",
        hydration_error: Exception | None = None,
    ) -> None:
        self.responses = responses
        self.corpus_revision = corpus_revision
        self.hydration_error = hydration_error
        self.calls: list[tuple[str, str, int | None, int | None, bool]] = []
        self.cached_calls: list[tuple[str, str, int | None, int | None, bool]] = []
        self.contexts: list[dict[str, object] | None] = []

    @property
    def corpus_identity(self) -> dict[str, object]:
        return {
            "backend": "local",
            "path": "/tmp/test-corpus.duckdb",
            "schema_version": 2,
            "dataset": "trec-ragtime/ragtime2",
            "config": "default",
            "revision": self.corpus_revision,
            "document_count": 10,
            "translation_count": 5,
        }

    @property
    def retrieval_corpus_identity(self) -> dict[str, object]:
        return self.corpus_identity

    def build_retrieval_identity(
        self,
        query: str,
        *,
        pipeline: str,
        top_k: int | None,
        max_content_fallbacks: int | None,
        fallback_on_hf_error: bool,
        query_context: dict[str, object] | None = None,
    ) -> SimpleNamespace:
        request_digest = hashlib.sha256(query.encode("utf-8")).digest()
        fingerprint = hashlib.sha256(b"baseline-test-retriever").digest()
        return SimpleNamespace(
            object_key=retrieval_object_key(fingerprint, request_digest),
            request_digest=request_digest,
            retriever_fingerprint=fingerprint,
        )

    def retrieve(
        self,
        query: str,
        *,
        pipeline: str,
        top_k: int | None,
        max_content_fallbacks: int | None,
        fallback_on_hf_error: bool,
        query_context: dict[str, object] | None = None,
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
        self.contexts.append(query_context)
        response = self.responses[query]
        if isinstance(response, Exception):
            raise response
        return response

    def get_cached_retrieval(
        self,
        query: str,
        *,
        pipeline: str,
        top_k: int | None,
        max_content_fallbacks: int | None,
        fallback_on_hf_error: bool,
        query_context: dict[str, object] | None = None,
    ) -> RetrievalResult | None:
        self.cached_calls.append(
            (
                query,
                pipeline,
                top_k,
                max_content_fallbacks,
                fallback_on_hf_error,
            )
        )
        response = self.responses.get(query)
        if isinstance(response, RetrievalResult):
            return response
        return None

    def enrich_documents(self, document_ids: list[str]) -> HydrationResult:
        if self.hydration_error is not None:
            raise self.hydration_error
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


class ConcurrentFakeRetriever(FakeRetriever):
    def __init__(
        self,
        responses: dict[str, RetrievalResult | Exception],
        *,
        tracker: dict[str, object],
        delays: dict[str, float] | None = None,
        blockers: dict[str, threading.Event] | None = None,
        completed: dict[str, threading.Event] | None = None,
        hydration_blocker: threading.Event | None = None,
    ) -> None:
        super().__init__(responses)
        self.tracker = tracker
        self.delays = delays or {}
        self.blockers = blockers or {}
        self.completed = completed or {}
        self.hydration_blocker = hydration_blocker
        self.active = 0

    def retrieve(
        self,
        query: str,
        *,
        pipeline: str,
        top_k: int | None,
        max_content_fallbacks: int | None,
        fallback_on_hf_error: bool,
        query_context: dict[str, object] | None = None,
    ) -> RetrievalResult:
        lock = self.tracker["lock"]
        assert isinstance(lock, type(threading.Lock()))
        with lock:
            self.active += 1
            self.tracker["active"] = int(self.tracker["active"]) + 1
            self.tracker["maximum"] = max(
                int(self.tracker["maximum"]), int(self.tracker["active"])
            )
            if self.active > 1:
                self.tracker["retriever_overlap"] = True
        try:
            blocker = self.blockers.get(query)
            if blocker is not None:
                blocker.wait(timeout=5)
            delay = self.delays.get(query, 0.0)
            if delay:
                time.sleep(delay)
            return super().retrieve(
                query,
                pipeline=pipeline,
                top_k=top_k,
                max_content_fallbacks=max_content_fallbacks,
                fallback_on_hf_error=fallback_on_hf_error,
                query_context=query_context,
            )
        finally:
            completed = self.completed.get(query)
            if completed is not None:
                completed.set()
            with lock:
                self.active -= 1
                self.tracker["active"] = int(self.tracker["active"]) - 1

    def enrich_documents(self, document_ids: list[str]) -> HydrationResult:
        if self.hydration_blocker is not None:
            self.hydration_blocker.wait(timeout=5)
        return super().enrich_documents(document_ids)


class CacheOnlyFakeRetriever(FakeRetriever):
    """A warm-run fake that can only serve exact retrieval artifacts."""

    def retrieve(self, *args: object, **kwargs: object) -> RetrievalResult:
        raise AssertionError("warm topic assembly must not call retrieve")

    def enrich_documents(self, document_ids: list[str]) -> HydrationResult:
        raise AssertionError("warm topic assembly must not call enrich_documents")


class MetadataRaisingCacheOnlyRetriever(CacheOnlyFakeRetriever):
    @property
    def corpus_identity(self) -> dict[str, object]:
        raise AssertionError("warm resume must not reopen corpus metadata")

    @property
    def retrieval_corpus_identity(self) -> dict[str, object]:
        return {
            "backend": "local",
            "path": "/tmp/test-corpus.duckdb",
            "schema_version": 2,
            "dataset": "trec-ragtime/ragtime2",
            "config": "default",
            "revision": self.corpus_revision,
            "document_count": 10,
            "translation_count": 5,
        }


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
        self.expected_topic_ids = ("2000", "2001")

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
        self.assertEqual(arguments.concurrency, 1)
        self.assertEqual(arguments.plans, default_query_plan_path())
        self.assertEqual(
            arguments.output_root,
            default_query_plan_path().parents[2] / "query-baselines",
        )
        self.assertEqual(PIPELINE, "ragtime2")

    def test_cli_rejects_nonstandard_depth_and_rrf(self) -> None:
        with self.assertRaises(SystemExit):
            build_parser().parse_args(
                ["b", "--run-id", "baseline-b", "--top-k", "7"]
            )
        with self.assertRaises(SystemExit):
            build_parser().parse_args(
                ["c", "--run-id", "baseline-c", "--rrf-k", "3"]
            )
        with self.assertRaises(SystemExit):
            build_parser().parse_args(
                ["c", "--run-id", "baseline-c", "--concurrency", "8"]
            )

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
            expected_topic_ids=self.expected_topic_ids,
            expected_plan_sha256=None,
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
            {
                "single zero": _retrieval_result("single zero", ["doc-a", "doc-b"]),
                "single one": _retrieval_result("single one", ["doc-c"]),
            }
        )
        idle_retriever = FakeRetriever({})
        run_query_baseline(
            self.plans_path,
            baseline="b",
            retriever=second_retriever,
            retrievers=(second_retriever, idle_retriever),
            concurrency=2,
            output_root=self.root / "runs",
            run_id="baseline-b",
            top_k=100,
            expected_topic_ids=self.expected_topic_ids,
            expected_plan_sha256=None,
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
        self.assertEqual(manifest["concurrency"], 2)
        self.assertEqual(manifest["counts"]["successful_topics"], 2)
        self.assertEqual(manifest["counts"]["failed_queries"], 0)
        self.assertEqual(
            first_retriever.contexts,
            [
                {
                    "kind": "baseline_b",
                    "query_construction": "problem_background_single_query_v1",
                    "query_plan_sha256": hashlib.sha256(
                        self.plans_path.read_bytes()
                    ).hexdigest(),
                    "query_index": 1,
                },
                {
                    "kind": "baseline_b",
                    "query_construction": "problem_background_single_query_v1",
                    "query_plan_sha256": hashlib.sha256(
                        self.plans_path.read_bytes()
                    ).hexdigest(),
                    "query_index": 1,
                },
            ],
        )

    def test_baseline_b_keeps_complete_uncached_result_in_memory(self) -> None:
        responses = {
            "single zero": _retrieval_result("single zero", ["doc-a"]),
            "single one": _retrieval_result("single one", ["doc-b"]),
        }
        for query in responses:
            result = responses[query]
            responses[query] = RetrievalResult(
                search_result=result.search_result,
                documents=result.documents,
                missing_translation_ids=result.missing_translation_ids,
            )
        retriever = FakeRetriever(responses)

        run_dir = run_query_baseline(
            self.plans_path,
            baseline="b",
            retriever=retriever,
            output_root=self.root / "runs",
            run_id="uncached-in-memory",
            expected_topic_ids=self.expected_topic_ids,
            expected_plan_sha256=None,
        )

        topics = {
            row["topic_id"]: row
            for row in map(json.loads, (run_dir / "topics.jsonl").read_text().splitlines())
        }
        self.assertEqual([topics[topic]["status"] for topic in self.expected_topic_ids], ["success", "success"])
        self.assertEqual([topics[topic]["candidate_count"] for topic in self.expected_topic_ids], [1, 1])
        self.assertEqual(retriever.cached_calls, [])
        queries = [
            json.loads(line)
            for line in (run_dir / "queries.jsonl").read_text().splitlines()
        ]
        self.assertTrue(all("retrieval_artifact" not in row for row in queries))

    def test_resume_recomputes_successful_checkpoint_without_reference(self) -> None:
        uncached = {
            query: RetrievalResult(
                search_result=result.search_result,
                documents=result.documents,
                missing_translation_ids=result.missing_translation_ids,
            )
            for query, result in {
                "single zero": _retrieval_result("single zero", ["doc-a"]),
                "single one": _retrieval_result("single one", ["doc-b"]),
            }.items()
        }
        first = FakeRetriever(uncached)
        first_run = run_query_baseline(
            self.plans_path,
            baseline="b",
            retriever=first,
            output_root=self.root / "runs",
            run_id="recompute-no-reference",
            expected_topic_ids=self.expected_topic_ids,
            expected_plan_sha256=None,
        )
        first_topics = [
            json.loads(line)
            for line in (first_run / "topics.jsonl").read_text().splitlines()
        ]
        self.assertEqual(
            [topic["status"] for topic in first_topics], ["success", "success"]
        )

        second = FakeRetriever(uncached)
        run_query_baseline(
            self.plans_path,
            baseline="b",
            retriever=second,
            output_root=self.root / "runs",
            run_id="recompute-no-reference",
            expected_topic_ids=self.expected_topic_ids,
            expected_plan_sha256=None,
        )

        self.assertEqual([call[0] for call in second.calls], ["single zero", "single one"])
        self.assertEqual(second.cached_calls, [])

    def test_baseline_c_keeps_complete_uncached_facets_in_memory(self) -> None:
        responses = {
            query: RetrievalResult(
                search_result=result.search_result,
                documents=result.documents,
                missing_translation_ids=result.missing_translation_ids,
            )
            for query, result in {
                "facet zero a": _retrieval_result("facet zero a", ["doc-a"]),
                "facet zero b": _retrieval_result("facet zero b", ["doc-b"]),
                "facet one a": _retrieval_result("facet one a", ["doc-c"]),
                "facet one b": _retrieval_result("facet one b", ["doc-d"]),
            }.items()
        }
        retriever = FakeRetriever(responses)

        run_dir = run_query_baseline(
            self.plans_path,
            baseline="c",
            retriever=retriever,
            output_root=self.root / "runs",
            run_id="uncached-facets",
            expected_topic_ids=self.expected_topic_ids,
            expected_plan_sha256=None,
        )

        topics = [
            json.loads(line)
            for line in (run_dir / "topics.jsonl").read_text().splitlines()
        ]
        self.assertEqual([topic["status"] for topic in topics], ["success", "success"])
        self.assertEqual([topic["candidate_count"] for topic in topics], [2, 2])
        self.assertEqual(retriever.cached_calls, [])

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
            expected_topic_ids=self.expected_topic_ids,
            expected_plan_sha256=None,
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

    def test_baseline_b_preserves_only_first_100_organizer_hits(self) -> None:
        document_ids = [f"doc-{index:03d}" for index in range(1, 102)]
        retriever = FakeRetriever(
            {
                "single zero": _retrieval_result("single zero", document_ids),
                "single one": _retrieval_result("single one", ["other"]),
            }
        )

        run_dir = run_query_baseline(
            self.plans_path,
            baseline="b",
            retriever=retriever,
            output_root=self.root / "runs",
            run_id="depth-test",
            expected_topic_ids=self.expected_topic_ids,
            expected_plan_sha256=None,
        )

        queries = [
            json.loads(line)
            for line in (run_dir / "queries.jsonl").read_text().splitlines()
        ]
        first_query = queries[0]
        self.assertEqual(first_query["hit_count"], 100)
        self.assertEqual(first_query["hits"][-1]["document_id"], "doc-100")
        topics = [
            json.loads(line)
            for line in (run_dir / "topics.jsonl").read_text().splitlines()
        ]
        first_topic = topics[0]
        self.assertEqual(first_topic["candidate_count"], 100)
        self.assertEqual(first_topic["candidates"][0]["score"], 1.0)
        self.assertNotIn(
            "doc-101",
            {candidate["document_id"] for candidate in first_topic["candidates"]},
        )

    def test_run_id_parent_escape_is_rejected_before_output_write(self) -> None:
        output_root = self.root / "does-not-exist"
        retriever = FakeRetriever({})

        with self.assertRaisesRegex(ValueError, "run_id"):
            run_query_baseline(
                self.plans_path,
                baseline="b",
                retriever=retriever,
                output_root=output_root,
                run_id="..",
                expected_topic_ids=self.expected_topic_ids,
                expected_plan_sha256=None,
            )

        self.assertFalse(output_root.exists())
        self.assertEqual(retriever.calls, [])

    def test_wrong_topic_set_is_rejected_before_retrieval(self) -> None:
        retriever = FakeRetriever({})

        with self.assertRaisesRegex(ValueError, "topic IDs"):
            run_query_baseline(
                self.plans_path,
                baseline="b",
                retriever=retriever,
                output_root=self.root / "runs",
                run_id="wrong-topics",
                expected_plan_sha256=None,
            )

        self.assertEqual(retriever.calls, [])

    def test_hydration_error_is_explicit_and_resumable(self) -> None:
        retriever = FakeRetriever(
            {
                "single zero": _retrieval_result("single zero", ["doc-a"]),
                "single one": _retrieval_result("single one", ["doc-b"]),
            },
            hydration_error=LocalCorpusReadError("local corpus read failed"),
        )

        run_dir = run_query_baseline(
            self.plans_path,
            baseline="b",
            retriever=retriever,
            output_root=self.root / "runs",
            run_id="hydration-error",
            expected_topic_ids=self.expected_topic_ids,
            expected_plan_sha256=None,
        )

        topics = [
            json.loads(line)
            for line in (run_dir / "topics.jsonl").read_text().splitlines()
        ]
        self.assertEqual([topic["status"] for topic in topics], ["success", "success"])
        self.assertEqual(topics[0]["hydration"]["failure_count"], 0)
        self.assertEqual(topics[0]["candidate_count"], 1)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["counts"]["failed_topics"], 0)

    def test_corpus_identity_change_rejects_resume_without_calls(self) -> None:
        first = FakeRetriever(
            {
                "single zero": _retrieval_result("single zero", ["doc-a"]),
                "single one": _retrieval_result("single one", ["doc-b"]),
            },
            corpus_revision="revision-one",
        )
        run_query_baseline(
            self.plans_path,
            baseline="b",
            retriever=first,
            output_root=self.root / "runs",
            run_id="corpus-identity",
            expected_topic_ids=self.expected_topic_ids,
            expected_plan_sha256=None,
        )
        second = FakeRetriever({}, corpus_revision="revision-two")

        with self.assertRaisesRegex(ValueError, "configuration"):
            run_query_baseline(
                self.plans_path,
                baseline="b",
                retriever=second,
                output_root=self.root / "runs",
                run_id="corpus-identity",
                expected_topic_ids=self.expected_topic_ids,
                expected_plan_sha256=None,
            )

        self.assertEqual(second.calls, [])

    @unittest.skipUnless(os.name == "posix", "requires POSIX flock")
    def test_root_lock_rejects_second_runner_before_retrieval(self) -> None:
        import fcntl

        output_root = self.root / "runs"
        output_root.mkdir()
        lock_path = output_root / ".query-baselines.lock"
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        retriever = FakeRetriever({})
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "already running"):
                run_query_baseline(
                    self.plans_path,
                    baseline="b",
                    retriever=retriever,
                    output_root=output_root,
                    run_id="locked",
                    expected_topic_ids=self.expected_topic_ids,
                    expected_plan_sha256=None,
                )
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

        self.assertEqual(retriever.calls, [])

    def test_atomic_write_preserves_old_file_when_replace_fails(self) -> None:
        destination = self.root / "artifact.jsonl"
        destination.write_text("old\n", encoding="utf-8")

        with patch(
            "trec_ragtime.query_baselines.os.replace",
            side_effect=OSError("interrupted replacement"),
        ):
            with self.assertRaises(OSError):
                _atomic_write(destination, "new\n")

        self.assertEqual(destination.read_text(encoding="utf-8"), "old\n")

    def test_four_workers_overlap_without_sharing_a_retriever(self) -> None:
        plans = [
            {
                "topic_id": str(topic_id),
                "plan_status": "model_generated",
                "single_query": f"query {topic_id}",
                "facet_queries": [f"facet {topic_id}"],
            }
            for topic_id in range(2000, 2008)
        ]
        plans_path = self.root / "eight_plans.jsonl"
        plans_path.write_text(
            "".join(json.dumps(plan) + "\n" for plan in plans),
            encoding="utf-8",
        )
        tracker: dict[str, object] = {
            "lock": threading.Lock(),
            "active": 0,
            "maximum": 0,
            "retriever_overlap": False,
        }
        responses = {
            plan["single_query"]: _retrieval_result(
                plan["single_query"], [f"doc-{plan['topic_id']}"]
            )
            for plan in plans
        }
        delays = {
            plan["single_query"]: 0.08 - index * 0.005
            for index, plan in enumerate(plans)
        }
        retrievers = tuple(
            ConcurrentFakeRetriever(responses, tracker=tracker, delays=delays)
            for _ in range(4)
        )

        run_dir = run_query_baseline(
            plans_path,
            baseline="b",
            retriever=retrievers[0],
            retrievers=retrievers,
            concurrency=4,
            output_root=self.root / "runs",
            run_id="concurrent-four",
            expected_topic_ids=tuple(plan["topic_id"] for plan in plans),
            expected_plan_sha256=None,
        )

        self.assertEqual(tracker["maximum"], 4)
        self.assertFalse(tracker["retriever_overlap"])
        queries = [
            json.loads(line)
            for line in (run_dir / "queries.jsonl").read_text().split("\n")
            if line
        ]
        topics = [
            json.loads(line)
            for line in (run_dir / "topics.jsonl").read_text().split("\n")
            if line
        ]
        expected_order = [plan["topic_id"] for plan in plans]
        self.assertEqual([row["topic_id"] for row in queries], expected_order)
        self.assertEqual([row["topic_id"] for row in topics], expected_order)

    def test_completed_query_is_checkpointed_while_peer_is_blocked(self) -> None:
        tracker: dict[str, object] = {
            "lock": threading.Lock(),
            "active": 0,
            "maximum": 0,
            "retriever_overlap": False,
        }
        release_slow = threading.Event()
        fast_completed = threading.Event()
        responses = {
            "single zero": _retrieval_result("single zero", ["doc-a"]),
            "single one": _retrieval_result("single one", ["doc-b"]),
        }
        retrievers = (
            ConcurrentFakeRetriever(
                responses,
                tracker=tracker,
                completed={"single zero": fast_completed},
            ),
            ConcurrentFakeRetriever(
                responses,
                tracker=tracker,
                blockers={"single one": release_slow},
            ),
        )
        errors: list[BaseException] = []

        def run() -> None:
            try:
                run_query_baseline(
                    self.plans_path,
                    baseline="b",
                    retriever=retrievers[0],
                    retrievers=retrievers,
                    concurrency=2,
                    output_root=self.root / "runs",
                    run_id="immediate-checkpoint",
                    expected_topic_ids=self.expected_topic_ids,
                    expected_plan_sha256=None,
                )
            except BaseException as exc:  # pragma: no cover - surfaced below.
                errors.append(exc)

        runner = threading.Thread(target=run)
        runner.start()
        self.assertTrue(fast_completed.wait(timeout=2))
        queries_path = (
            self.root / "runs" / "immediate-checkpoint" / "queries.jsonl"
        )
        deadline = time.monotonic() + 2
        checkpointed = False
        while time.monotonic() < deadline:
            if queries_path.exists():
                rows = [
                    json.loads(line)
                    for line in queries_path.read_text().split("\n")
                    if line
                ]
                if [row["topic_id"] for row in rows] == ["2000"]:
                    checkpointed = True
                    break
            time.sleep(0.01)
        self.assertTrue(checkpointed)
        self.assertTrue(runner.is_alive())
        release_slow.set()
        runner.join(timeout=5)

        self.assertFalse(runner.is_alive())
        self.assertEqual(errors, [])

    def test_all_completed_queries_checkpoint_before_topic_hydration(self) -> None:
        tracker: dict[str, object] = {
            "lock": threading.Lock(),
            "active": 0,
            "maximum": 0,
            "retriever_overlap": False,
        }
        release_hydration = threading.Event()
        zero_completed = threading.Event()
        one_completed = threading.Event()
        responses = {
            "single zero": _retrieval_result("single zero", ["doc-a"]),
            "single one": _retrieval_result("single one", ["doc-b"]),
        }
        retrievers = (
            ConcurrentFakeRetriever(
                responses,
                tracker=tracker,
                completed={"single zero": zero_completed},
                hydration_blocker=release_hydration,
            ),
            ConcurrentFakeRetriever(
                responses,
                tracker=tracker,
                completed={"single one": one_completed},
            ),
        )
        errors: list[BaseException] = []

        def run() -> None:
            try:
                run_query_baseline(
                    self.plans_path,
                    baseline="b",
                    retriever=retrievers[0],
                    retrievers=retrievers,
                    concurrency=2,
                    output_root=self.root / "runs",
                    run_id="checkpoint-before-hydration",
                    expected_topic_ids=self.expected_topic_ids,
                    expected_plan_sha256=None,
                )
            except BaseException as exc:  # pragma: no cover - surfaced below.
                errors.append(exc)

        runner = threading.Thread(target=run)
        runner.start()
        try:
            self.assertTrue(zero_completed.wait(timeout=2))
            self.assertTrue(one_completed.wait(timeout=2))
            queries_path = (
                self.root
                / "runs"
                / "checkpoint-before-hydration"
                / "queries.jsonl"
            )
            deadline = time.monotonic() + 2
            checkpoint_count = 0
            while time.monotonic() < deadline:
                if queries_path.exists():
                    checkpoint_count = len(
                        [line for line in queries_path.read_text().split("\n") if line]
                    )
                    if checkpoint_count == 2:
                        break
                time.sleep(0.01)
        finally:
            release_hydration.set()
            runner.join(timeout=5)

        self.assertEqual(checkpoint_count, 2)
        self.assertEqual(errors, [])

    def test_resume_rebuilds_stale_topic_from_successful_checkpoint(self) -> None:
        first = FakeRetriever(
            {
                "single zero": _retrieval_result("single zero", ["doc-a"]),
                "single one": _retrieval_result("single one", ["doc-b"]),
            }
        )
        run_dir = run_query_baseline(
            self.plans_path,
            baseline="b",
            retriever=first,
            output_root=self.root / "runs",
            run_id="stale-topic",
            expected_topic_ids=self.expected_topic_ids,
            expected_plan_sha256=None,
        )
        topics = [
            json.loads(line)
            for line in (run_dir / "topics.jsonl").read_text().split("\n")
            if line
        ]
        topics[0] = {
            **topics[0],
            "status": "error",
            "candidate_count": 0,
            "candidates": [],
        }
        (run_dir / "topics.jsonl").write_text(
            "".join(json.dumps(topic) + "\n" for topic in topics),
            encoding="utf-8",
        )
        resumed = FakeRetriever(
            {
                "single zero": _retrieval_result("single zero", ["doc-a"]),
                "single one": _retrieval_result("single one", ["doc-b"]),
            }
        )

        run_query_baseline(
            self.plans_path,
            baseline="b",
            retriever=resumed,
            output_root=self.root / "runs",
            run_id="stale-topic",
            expected_topic_ids=self.expected_topic_ids,
            expected_plan_sha256=None,
        )

        rebuilt = [
            json.loads(line)
            for line in (run_dir / "topics.jsonl").read_text().split("\n")
            if line
        ]
        self.assertEqual(rebuilt[0]["status"], "success")
        self.assertEqual(rebuilt[0]["candidate_count"], 1)
        self.assertEqual(resumed.calls, [])

    def test_warm_topic_reconstruction_is_cache_only(self) -> None:
        cold = FakeRetriever(
            {
                "single zero": _retrieval_result("single zero", ["doc-a"]),
                "single one": _retrieval_result("single one", ["doc-b"]),
            }
        )
        run_dir = run_query_baseline(
            self.plans_path,
            baseline="b",
            retriever=cold,
            output_root=self.root / "runs",
            run_id="warm-cache-only",
            expected_topic_ids=self.expected_topic_ids,
            expected_plan_sha256=None,
        )
        warm = CacheOnlyFakeRetriever(
            {
                "single zero": _retrieval_result("single zero", ["doc-a"]),
                "single one": _retrieval_result("single one", ["doc-b"]),
            }
        )

        run_query_baseline(
            self.plans_path,
            baseline="b",
            retriever=warm,
            output_root=self.root / "runs",
            run_id="warm-cache-only",
            expected_topic_ids=self.expected_topic_ids,
            expected_plan_sha256=None,
        )

        self.assertEqual(warm.calls, [])
        self.assertEqual(
            [call[0] for call in warm.cached_calls], ["single zero", "single one"]
        )
        topics = [
            json.loads(line)
            for line in (run_dir / "topics.jsonl").read_text().splitlines()
        ]
        self.assertEqual([topic["status"] for topic in topics], ["success", "success"])
        self.assertTrue(all("retrieval_artifact" in row for row in topics[0]["query_results"]))

    def test_warm_resume_uses_captured_identity_without_corpus_metadata(self) -> None:
        cold = FakeRetriever(
            {
                "single zero": _retrieval_result("single zero", ["doc-a"]),
                "single one": _retrieval_result("single one", ["doc-b"]),
            }
        )
        run_dir = run_query_baseline(
            self.plans_path,
            baseline="b",
            retriever=cold,
            output_root=self.root / "runs",
            run_id="warm-no-metadata",
            expected_topic_ids=self.expected_topic_ids,
            expected_plan_sha256=None,
        )
        warm = MetadataRaisingCacheOnlyRetriever(
            {
                "single zero": _retrieval_result("single zero", ["doc-a"]),
                "single one": _retrieval_result("single one", ["doc-b"]),
            }
        )

        run_query_baseline(
            self.plans_path,
            baseline="b",
            retriever=warm,
            output_root=self.root / "runs",
            run_id="warm-no-metadata",
            expected_topic_ids=self.expected_topic_ids,
            expected_plan_sha256=None,
        )

        topics = [
            json.loads(line)
            for line in (run_dir / "topics.jsonl").read_text().splitlines()
        ]
        self.assertEqual([topic["status"] for topic in topics], ["success", "success"])
        self.assertEqual(warm.calls, [])

    def test_wrong_but_self_consistent_reference_is_rejected_before_checkpoint(self) -> None:
        wrong = _retrieval_result("different query", ["doc-a"])
        retriever = FakeRetriever(
            {
                "single zero": wrong,
                "single one": _retrieval_result("single one", ["doc-b"]),
            }
        )

        run_dir = run_query_baseline(
            self.plans_path,
            baseline="b",
            retriever=retriever,
            output_root=self.root / "runs",
            run_id="wrong-checkpoint-reference",
            expected_topic_ids=self.expected_topic_ids,
            expected_plan_sha256=None,
        )

        queries = [
            json.loads(line)
            for line in (run_dir / "queries.jsonl").read_text().splitlines()
        ]
        first = next(row for row in queries if row["topic_id"] == "2000")
        self.assertEqual(first["status"], "error")
        self.assertNotIn("retrieval_artifact", first)

    def test_checkpoint_query_index_must_match_plan_position(self) -> None:
        retriever = FakeRetriever(
            {
                "single zero": _retrieval_result("single zero", ["doc-a"]),
                "single one": _retrieval_result("single one", ["doc-b"]),
            }
        )
        run_dir = run_query_baseline(
            self.plans_path,
            baseline="b",
            retriever=retriever,
            output_root=self.root / "runs",
            run_id="bad-query-index",
            expected_topic_ids=self.expected_topic_ids,
            expected_plan_sha256=None,
        )
        queries = [
            json.loads(line)
            for line in (run_dir / "queries.jsonl").read_text().splitlines()
        ]
        queries[0]["query_index"] = 0
        (run_dir / "queries.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in queries),
            encoding="utf-8",
        )

        run_query_baseline(
            self.plans_path,
            baseline="b",
            retriever=FakeRetriever(
                {
                    "single zero": _retrieval_result("single zero", ["doc-a"]),
                    "single one": _retrieval_result("single one", ["doc-b"]),
                }
            ),
            output_root=self.root / "runs",
            run_id="bad-query-index",
            expected_topic_ids=self.expected_topic_ids,
            expected_plan_sha256=None,
        )
        topics = [
            json.loads(line)
            for line in (run_dir / "topics.jsonl").read_text().splitlines()
        ]
        self.assertEqual(topics[0]["status"], "error")
        self.assertIn("query_index", topics[0]["hydration"]["error_detail"])

    def test_c_missing_translation_order_is_first_seen(self) -> None:
        responses: dict[str, RetrievalResult | Exception] = {
            "facet zero a": _retrieval_result(
                "facet zero a", ["doc-b"], missing_translation_ids=("doc-b",)
            ),
            "facet zero b": _retrieval_result(
                "facet zero b", ["doc-a"], missing_translation_ids=("doc-a",)
            ),
            "facet one a": RagtimeApiError("failure"),
            "facet one b": RagtimeApiError("failure"),
        }
        run_dir = run_query_baseline(
            self.plans_path,
            baseline="c",
            retriever=FakeRetriever(responses),
            output_root=self.root / "runs",
            run_id="translation-order",
            expected_topic_ids=self.expected_topic_ids,
            expected_plan_sha256=None,
        )
        topics = {
            row["topic_id"]: row
            for row in map(json.loads, (run_dir / "topics.jsonl").read_text().splitlines())
        }
        self.assertEqual(
            topics["2000"]["hydration"]["missing_translation_ids"],
            ["doc-b", "doc-a"],
        )

    def test_c_same_id_text_disagreement_is_explicit(self) -> None:
        first = _retrieval_result("facet zero a", ["shared"])
        second_document = RetrievedDocument(
            rank=1,
            document_id="shared",
            score=1.0,
            text="a different canonical text",
            url="https://example.test/shared",
            date="2026-01-01",
            language="eng",
            content_source="local_corpus",
            content_local_cache_hit=True,
        )
        # Give the independently-built second result a valid lightweight ref.
        digest = hashlib.sha256(b"facet zero b").digest()
        fingerprint = hashlib.sha256(b"baseline-test-retriever").digest()
        second = RetrievalResult(
            search_result=_search_result("facet zero b", ["shared"]),
            documents=(second_document,),
            retrieval_artifact=RetrievalArtifactReference(
                object_key=retrieval_object_key(fingerprint, digest),
                request_digest=digest,
                retriever_fingerprint=fingerprint,
                stored_at=123.0,
                cache_hit=False,
            ),
        )
        retriever = FakeRetriever(
            {
                "facet zero a": first,
                "facet zero b": second,
                "facet one a": _retrieval_result("facet one a", ["doc-c"]),
                "facet one b": _retrieval_result("facet one b", ["doc-d"]),
            }
        )

        run_dir = run_query_baseline(
            self.plans_path,
            baseline="c",
            retriever=retriever,
            output_root=self.root / "runs",
            run_id="c-text-disagreement",
            expected_topic_ids=self.expected_topic_ids,
            expected_plan_sha256=None,
        )

        topics = {
            row["topic_id"]: row
            for row in map(json.loads, (run_dir / "topics.jsonl").read_text().splitlines())
        }
        self.assertEqual(topics["2000"]["status"], "error")
        self.assertIn("text disagreement", topics["2000"]["hydration"]["error_detail"])

    def test_concurrent_failure_stops_scheduling_new_queries(self) -> None:
        plans = [
            {
                "topic_id": str(topic_id),
                "plan_status": "model_generated",
                "single_query": f"query {topic_id}",
                "facet_queries": [f"facet {topic_id}"],
            }
            for topic_id in range(2000, 2006)
        ]
        plans_path = self.root / "six_plans.jsonl"
        plans_path.write_text(
            "".join(json.dumps(plan) + "\n" for plan in plans),
            encoding="utf-8",
        )
        tracker: dict[str, object] = {
            "lock": threading.Lock(),
            "active": 0,
            "maximum": 0,
            "retriever_overlap": False,
        }
        responses: dict[str, RetrievalResult | Exception] = {
            plan["single_query"]: _retrieval_result(
                plan["single_query"], [f"doc-{plan['topic_id']}"]
            )
            for plan in plans
        }
        responses["query 2000"] = RagtimeApiError(
            "RAGTIME /pipeline request failed: HTTP 504"
        )
        delays = {plan["single_query"]: 0.1 for plan in plans[1:]}
        retrievers = tuple(
            ConcurrentFakeRetriever(responses, tracker=tracker, delays=delays)
            for _ in range(4)
        )

        with self.assertRaisesRegex(RuntimeError, "concurrent retrieval stopped"):
            run_query_baseline(
                plans_path,
                baseline="b",
                retriever=retrievers[0],
                retrievers=retrievers,
                concurrency=4,
                output_root=self.root / "runs",
                run_id="stop-on-error",
                expected_topic_ids=tuple(plan["topic_id"] for plan in plans),
                expected_plan_sha256=None,
            )

        total_calls = sum(len(worker.calls) for worker in retrievers)
        self.assertEqual(total_calls, 4)
        manifest = json.loads(
            (
                self.root
                / "runs"
                / "stop-on-error"
                / "manifest.json"
            ).read_text()
        )
        self.assertEqual(manifest["status"], "stopped_on_failure")
        self.assertEqual(manifest["counts"]["failed_queries"], 1)


if __name__ == "__main__":
    unittest.main()
