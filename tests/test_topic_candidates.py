from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

from trec_ragtime import (
    PipelineResult,
    RagtimeApiError,
    RetrievalResult,
    RetrievedDocument,
    SearchHit,
)
from trec_ragtime.retrieval_cache import RetrievalArtifactReference, retrieval_object_key
from trec_ragtime.topic_candidates import (
    Topic,
    build_topic_query,
    build_parser,
    load_topics,
    main,
    pipeline_for_top_k,
    query_hash,
    run_control_a,
    write_control_a_artifact,
)


class TopicHelpersTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.topic_file = Path(self.temporary_directory.name) / "topics.jsonl"

    def write_topics(self, records: list[object]) -> None:
        self.topic_file.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )

    def test_load_topics_preserves_source_order_and_validates_fields(self) -> None:
        self.write_topics(
            [
                {
                    "topic_id": "2000",
                    "collection_id": "ragtime/2/all",
                    "title": "First topic",
                    "problem_statement": "first problem",
                    "background": "first background",
                    "limit": 5000,
                },
                {
                    "topic_id": "2001",
                    "collection_id": "ragtime/2/all",
                    "problem_statement": "second problem",
                    "background": "second background",
                    "limit": 5000,
                },
            ]
        )

        topics = load_topics(self.topic_file)

        self.assertEqual([topic.topic_id for topic in topics], ["2000", "2001"])
        self.assertEqual(topics[0].title, "First topic")
        self.assertIsNone(topics[1].title)
        self.assertEqual(topics[0].limit, 5000)

    def test_build_topic_query_uses_problem_statement_only(self) -> None:
        topic = Topic("1", "ragtime/2/all", "problem", "background", 5000)

        self.assertEqual(build_topic_query(topic), "problem")

    def test_pipeline_for_top_k_translates_depth_to_organizer_protocol(self) -> None:
        self.assertEqual(pipeline_for_top_k(100), "ragtime2")
        self.assertEqual(pipeline_for_top_k(24), "ragtime2")

        with self.assertRaises(ValueError):
            pipeline_for_top_k(0)
        with self.assertRaises(ValueError):
            pipeline_for_top_k(101)

    def test_query_hash_is_sha256_of_exact_query(self) -> None:
        query = "Problem: café"

        self.assertEqual(
            query_hash(query), hashlib.sha256(query.encode("utf-8")).hexdigest()
        )

    def test_load_topics_rejects_malformed_records(self) -> None:
        cases = [
            ("not json\n", "invalid JSON"),
            (
                json.dumps(
                    {
                        "topic_id": "1",
                        "collection_id": "ragtime/2/all",
                        "problem_statement": "problem",
                        "background": "background",
                    }
                )
                + "\n",
                "missing field 'limit'",
            ),
            (
                json.dumps(
                    {
                        "topic_id": "1",
                        "collection_id": "ragtime/2/all",
                        "problem_statement": "problem",
                        "background": "background",
                        "limit": 0,
                    }
                )
                + "\n",
                "limit",
            ),
            (
                json.dumps(
                    {
                        "topic_id": "1",
                        "collection_id": "ragtime/2/all",
                        "problem_statement": "",
                        "background": "background",
                        "limit": 5000,
                    }
                )
                + "\n",
                "problem_statement",
            ),
        ]

        for contents, message in cases:
            with self.subTest(message=message):
                self.topic_file.write_text(contents, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_topics(self.topic_file)

    def test_load_topics_rejects_duplicate_topic_ids(self) -> None:
        record = {
            "topic_id": "1",
            "collection_id": "ragtime/2/all",
            "problem_statement": "problem",
            "background": "background",
            "limit": 5000,
        }
        self.write_topics([record, record])

        with self.assertRaisesRegex(ValueError, "duplicate topic_id"):
            load_topics(self.topic_file)


class ControlAArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.output_root = Path(self.temporary_directory.name) / "control-a"

    def test_writer_creates_stable_manifest_and_jsonl(self) -> None:
        run_dir = self.output_root / "run-1"
        manifest = {"z": "last", "a": "first"}
        records = [
            {
                "topic_id": "1",
                "text": "canonical source text",
                "report_text": "English report text",
                "hydrated": True,
                "missing_translation": False,
            }
        ]

        write_control_a_artifact(run_dir, manifest=manifest, records=records)

        self.assertEqual(
            json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")),
            manifest,
        )
        candidate_lines = (run_dir / "candidates.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        self.assertEqual(len(candidate_lines), 1)
        self.assertEqual(json.loads(candidate_lines[0]), records[0])
        self.assertTrue(candidate_lines[0].startswith('{"hydrated":true'))
        self.assertEqual(list(run_dir.glob(".control-a-*")), [])

    def test_writer_refuses_to_overwrite_existing_run(self) -> None:
        run_dir = self.output_root / "run-1"
        run_dir.mkdir(parents=True)

        with self.assertRaises(FileExistsError):
            write_control_a_artifact(run_dir, manifest={}, records=[])

        self.assertEqual(list(run_dir.iterdir()), [])


class FakeRetriever:
    def __init__(self, results: list[object]) -> None:
        self.results = iter(results)
        self.calls: list[dict[str, object]] = []
        self.query_contexts: list[dict[str, object] | None] = []
        self.identity_calls: list[str] = []
        self.cache_configuration = {
            "namespace": "ragtime2-2026-v1",
            "bucket_id": "owner/ragtime-cache",
            "prefix": "responses",
            "local_dir": "/shared/ragtime-cache/artifacts-v1",
        }
        self.corpus_path = Path("/shared/ragtime-cache/ragtime2-corpus-v2.duckdb")
        self.corpus = SimpleNamespace(
            metadata=SimpleNamespace(revision="corpus-revision-1")
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
            {
                "query": query,
                "pipeline": pipeline,
                "top_k": top_k,
                "max_content_fallbacks": max_content_fallbacks,
                "fallback_on_hf_error": fallback_on_hf_error,
            }
        )
        self.query_contexts.append(query_context)
        result = next(self.results)
        if isinstance(result, BaseException):
            raise result
        return result  # type: ignore[return-value]

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
        self.identity_calls.append(query)
        request_digest = hashlib.sha256(query.encode("utf-8")).digest()
        fingerprint = hashlib.sha256(b"control-a-test-retriever").digest()
        return SimpleNamespace(
            object_key=retrieval_object_key(fingerprint, request_digest),
            request_digest=request_digest,
            retriever_fingerprint=fingerprint,
        )


def make_retrieval_result(
    query: str,
    documents: tuple[RetrievedDocument, ...],
    *,
    missing_translation_ids: tuple[str, ...] = (),
    with_reference: bool = True,
    result_query: str | None = None,
    pipeline: str = "ragtime2",
) -> RetrievalResult:
    request_digest = hashlib.sha256(query.encode("utf-8")).digest()
    fingerprint = hashlib.sha256(b"control-a-test-retriever").digest()
    return RetrievalResult(
        search_result=PipelineResult(
            query=query if result_query is None else result_query,
            pipeline=pipeline,
            hits=tuple(
                SearchHit(
                    rank=document.rank,
                    document_id=document.document_id,
                    score=document.score,
                )
                for document in documents
            ),
            processed=True,
            local_cache_hit=True,
            remote_cached=True,
            server_timestamp=123.0,
            cache_stored_at=123.0,
        ),
        documents=documents,
        missing_translation_ids=missing_translation_ids,
        retrieval_artifact=(
            RetrievalArtifactReference(
                object_key=retrieval_object_key(fingerprint, request_digest),
                request_digest=request_digest,
                retriever_fingerprint=fingerprint,
                stored_at=123.0,
                cache_hit=False,
            )
            if with_reference
            else None
        ),
    )


class ControlARunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        root = Path(self.temporary_directory.name)
        self.topic_file = root / "topics.jsonl"
        self.topic_file.write_text(
            "\n".join(
                json.dumps(record)
                for record in (
                    {
                        "topic_id": "1",
                        "collection_id": "ragtime/2/all",
                        "problem_statement": "first problem",
                        "background": "first background",
                        "limit": 5000,
                    },
                    {
                        "topic_id": "2",
                        "collection_id": "ragtime/2/all",
                        "problem_statement": "second problem",
                        "background": "second background",
                        "limit": 5000,
                    },
                )
            )
            + "\n",
            encoding="utf-8",
        )
        self.output_root = root / "artifacts"

    def test_runner_calls_organizer_and_writes_auditable_candidates(self) -> None:
        unresolved = RetrievedDocument(
            rank=2,
            document_id="doc-unresolved",
            score=0.2,
            text=None,
            url=None,
            date=None,
            language=None,
            content_source=None,
            content_local_cache_hit=None,
        )
        translated = RetrievedDocument(
            rank=1,
            document_id="doc-translation",
            score=0.9,
            text="texto fuente",
            url="https://example.test/doc",
            date="2025-01-01",
            language="spa",
            content_source="huggingface",
            content_local_cache_hit=True,
        )
        first_query = "first problem"
        second_query = "second problem"
        retriever = FakeRetriever(
            [
                make_retrieval_result(first_query, (unresolved,)),
                make_retrieval_result(
                    second_query,
                    (translated,),
                    missing_translation_ids=("doc-translation",),
                ),
            ]
        )

        run_dir = run_control_a(
            self.topic_file,
            retriever=retriever,
            output_root=self.output_root,
            run_id="control-a-test",
            top_k=100,
            git_revision="git-revision-1",
            created_at="2026-08-02T00:00:00+00:00",
        )

        self.assertEqual(run_dir, self.output_root / "control-a-test")
        self.assertEqual(
            [call["pipeline"] for call in retriever.calls],
            ["ragtime2", "ragtime2"],
        )
        self.assertEqual([call["top_k"] for call in retriever.calls], [100, 100])
        self.assertEqual(
            [call["max_content_fallbacks"] for call in retriever.calls], [0, 0]
        )
        self.assertEqual(
            [call["fallback_on_hf_error"] for call in retriever.calls],
            [False, False],
        )

        manifest = json.loads(
            (run_dir / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["requested_top_k"], 100)
        self.assertEqual(manifest["pipeline"], "ragtime2")
        self.assertEqual(manifest["query_construction"], "problem_only_v1")
        self.assertEqual(manifest["retrieval"]["top_k"], 100)
        self.assertEqual(
            manifest["api_cache"],
            {
                "bucket_id": "owner/ragtime-cache",
                "local_dir": "/shared/ragtime-cache/artifacts-v1",
                "namespace": "ragtime2-2026-v1",
                "prefix": "responses",
            },
        )
        self.assertNotIn("api_cache_path", manifest)
        self.assertEqual(manifest["git_revision"], "git-revision-1")
        self.assertEqual(manifest["corpus"]["revision"], "corpus-revision-1")
        self.assertEqual(manifest["counts"]["topics"], 2)
        self.assertEqual(manifest["counts"]["candidates"], 2)
        self.assertEqual(
            [
                {
                    key: value
                    for key, value in status.items()
                    if key != "retrieval_artifact"
                }
                for status in manifest["topic_status"]
            ],
            [
                {"missing_translation_ids": [], "topic_id": "1", "unresolved_ids": ["doc-unresolved"]},
                {"missing_translation_ids": ["doc-translation"], "topic_id": "2", "unresolved_ids": []},
            ],
        )

        candidate_lines = (run_dir / "candidates.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        candidates = [json.loads(line) for line in candidate_lines]
        self.assertEqual([candidate["topic_id"] for candidate in candidates], ["1", "2"])
        self.assertEqual(candidates[0]["document_id"], "doc-unresolved")
        self.assertFalse(candidates[0]["hydrated"])
        self.assertFalse(candidates[0]["report_ready"])
        self.assertTrue(candidates[1]["missing_translation"])
        self.assertEqual(candidates[1]["text"], "texto fuente")
        self.assertIsNone(candidates[1]["report_text"])
        self.assertEqual(candidates[0]["query_hash"], query_hash(first_query))
        self.assertEqual(candidates[1]["query_hash"], query_hash(second_query))
        self.assertEqual(
            manifest["topic_source"]["sha256"],
            hashlib.sha256(self.topic_file.read_bytes()).hexdigest(),
        )
        self.assertNotIn("bearer_token", json.dumps(manifest).lower())

    def test_runner_uses_stable_context_and_persists_one_reference_per_success(self) -> None:
        first = RetrievedDocument(
            rank=1,
            document_id="doc-first",
            score=1.0,
            text="first text",
            url=None,
            date=None,
            language="eng",
            content_source="local_corpus",
            content_local_cache_hit=True,
        )
        retriever = FakeRetriever(
            [
                make_retrieval_result("first problem", (first,)),
                RagtimeApiError("HTTP 504"),
            ]
        )

        run_dir = run_control_a(
            self.topic_file,
            retriever=retriever,
            output_root=self.output_root,
            run_id="context-and-references",
        )

        self.assertEqual(
            retriever.query_contexts,
            [
                {"kind": "control_a", "query_construction": "problem_only_v1"},
                {"kind": "control_a", "query_construction": "problem_only_v1"},
            ],
        )
        manifest = json.loads((run_dir / "manifest.json").read_text())
        statuses = manifest["topic_status"]
        self.assertIn("retrieval_artifact", statuses[0])
        self.assertNotIn("retrieval_artifact", statuses[1])
        self.assertEqual(
            statuses[0]["retrieval_artifact"]["cache_hit"], False
        )

    def test_complete_result_without_reference_is_a_success_and_omits_reference(self) -> None:
        first = RetrievedDocument(
            rank=1,
            document_id="doc-first",
            score=1.0,
            text="first text",
            url=None,
            date=None,
            language="eng",
            content_source="local_corpus",
            content_local_cache_hit=True,
        )
        retriever = FakeRetriever(
            [
                make_retrieval_result(
                    "first problem", (first,), with_reference=False
                ),
                make_retrieval_result("second problem", (), with_reference=False),
            ]
        )

        run_dir = run_control_a(
            self.topic_file,
            retriever=retriever,
            output_root=self.output_root,
            run_id="uncached-complete",
        )

        manifest = json.loads((run_dir / "manifest.json").read_text())
        self.assertEqual(manifest["counts"]["failed_topics"], 0)
        self.assertNotIn("retrieval_artifact", manifest["topic_status"][0])
        self.assertNotIn("retrieval_artifact", manifest["topic_status"][1])
        self.assertEqual(manifest["counts"]["candidates"], 1)
        self.assertEqual(retriever.identity_calls, [])

    def test_reference_less_result_with_wrong_query_is_recorded_as_failure(self) -> None:
        document = RetrievedDocument(
            rank=1,
            document_id="doc-wrong-query",
            score=1.0,
            text="wrong query text",
            url=None,
            date=None,
            language="eng",
            content_source="local_corpus",
            content_local_cache_hit=True,
        )
        retriever = FakeRetriever(
            [
                make_retrieval_result(
                    "first problem",
                    (document,),
                    result_query="different problem",
                    with_reference=False,
                ),
                make_retrieval_result("second problem", (), with_reference=False),
            ]
        )

        run_dir = run_control_a(
            self.topic_file,
            retriever=retriever,
            output_root=self.output_root,
            run_id="wrong-query-no-reference",
        )

        manifest = json.loads((run_dir / "manifest.json").read_text())
        self.assertEqual(manifest["counts"]["failed_topics"], 1)
        self.assertEqual(manifest["counts"]["candidates"], 0)
        self.assertEqual(manifest["topic_status"][0]["error"]["type"], "ValueError")

    def test_reference_less_result_with_wrong_pipeline_is_recorded_as_failure(self) -> None:
        retriever = FakeRetriever(
            [
                make_retrieval_result(
                    "first problem",
                    (),
                    pipeline="wrong-pipeline",
                    with_reference=False,
                ),
                make_retrieval_result("second problem", (), with_reference=False),
            ]
        )

        run_dir = run_control_a(
            self.topic_file,
            retriever=retriever,
            output_root=self.output_root,
            run_id="wrong-pipeline-no-reference",
        )

        manifest = json.loads((run_dir / "manifest.json").read_text())
        self.assertEqual(manifest["counts"]["failed_topics"], 1)
        self.assertEqual(manifest["counts"]["candidates"], 0)
        self.assertEqual(manifest["topic_status"][0]["error"]["type"], "ValueError")

    def test_incomplete_result_keeps_candidates_and_counts_without_reference(self) -> None:
        unresolved = RetrievedDocument(
            rank=1,
            document_id="missing-doc",
            score=0.2,
            text=None,
            url=None,
            date=None,
            language=None,
            content_source=None,
            content_local_cache_hit=None,
        )
        incomplete = RetrievalResult(
            search_result=PipelineResult(
                query="first problem",
                pipeline="ragtime2",
                hits=(SearchHit(rank=1, document_id="missing-doc", score=0.2),),
                processed=True,
                local_cache_hit=False,
                remote_cached=False,
                server_timestamp=None,
                cache_stored_at=None,
            ),
            documents=(unresolved,),
        )
        retriever = FakeRetriever(
            [incomplete, make_retrieval_result("second problem", ())]
        )

        run_dir = run_control_a(
            self.topic_file,
            retriever=retriever,
            output_root=self.output_root,
            run_id="incomplete-preserved",
        )

        manifest = json.loads((run_dir / "manifest.json").read_text())
        self.assertEqual(manifest["counts"]["failed_topics"], 0)
        self.assertEqual(manifest["counts"]["candidates"], 1)
        self.assertEqual(manifest["counts"]["unresolved_documents"], 1)
        self.assertNotIn("retrieval_artifact", manifest["topic_status"][0])
        candidates = [
            json.loads(line)
            for line in (run_dir / "candidates.jsonl").read_text().splitlines()
        ]
        self.assertEqual(candidates[0]["document_id"], "missing-doc")
        self.assertFalse(candidates[0]["hydrated"])

    def test_runner_refuses_to_overwrite_existing_run(self) -> None:
        run_dir = self.output_root / "existing"
        run_dir.mkdir(parents=True)
        retriever = FakeRetriever([])

        with self.assertRaises(FileExistsError):
            run_control_a(
                self.topic_file,
                retriever=retriever,
                output_root=self.output_root,
                run_id="existing",
            )

    def test_runner_records_retrieval_failure_and_continues(self) -> None:
        translated = RetrievedDocument(
            rank=1,
            document_id="doc-success",
            score=0.9,
            text="source text",
            url=None,
            date=None,
            language="eng",
            content_source="huggingface",
            content_local_cache_hit=True,
        )
        retriever = FakeRetriever(
            [
                RagtimeApiError("HTTP 504"),
                make_retrieval_result("second problem", (translated,)),
            ]
        )

        run_dir = run_control_a(
            self.topic_file,
            retriever=retriever,
            output_root=self.output_root,
            run_id="control-a-with-failure",
            top_k=100,
        )

        manifest = json.loads(
            (run_dir / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["counts"]["topics"], 2)
        self.assertEqual(manifest["counts"]["candidates"], 1)
        self.assertEqual(manifest["counts"]["failed_topics"], 1)
        statuses_without_refs = [
            {
                key: value
                for key, value in status.items()
                if key != "retrieval_artifact"
            }
            for status in manifest["topic_status"]
        ]
        self.assertEqual(
            statuses_without_refs,
            [
                {
                    "error": {"type": "RagtimeApiError"},
                    "missing_translation_ids": [],
                    "topic_id": "1",
                    "unresolved_ids": [],
                },
                {
                    "missing_translation_ids": [],
                    "topic_id": "2",
                    "unresolved_ids": [],
                },
            ],
        )


class ControlACommandLineTests(unittest.TestCase):
    def test_help_is_available_without_constructing_a_retriever(self) -> None:
        stdout = StringIO()

        with redirect_stdout(stdout), self.assertRaises(SystemExit) as raised:
            build_parser().parse_args(["--help"])

        self.assertEqual(raised.exception.code, 0)
        self.assertIn("--top-k", stdout.getvalue())
        self.assertIn("--run-id", stdout.getvalue())

    def test_run_id_is_required(self) -> None:
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as raised:
                build_parser().parse_args([])

        self.assertEqual(raised.exception.code, 2)

    def test_parser_defaults_to_corrected_official_topics(self) -> None:
        arguments = build_parser().parse_args(["--run-id", "run-1"])

        self.assertEqual(arguments.top_k, 100)
        self.assertEqual(
            arguments.topics, Path("topics/topics.all.2026.v0625-fix.jsonl")
        )
        topics = load_topics(arguments.topics)
        self.assertEqual(len(topics), 103)
        self.assertEqual(topics[0].title, "Anti-vaping Legislation")

    def test_parser_accepts_custom_paths_and_depth(self) -> None:
        arguments = build_parser().parse_args(
            [
                "--run-id",
                "run-1",
                "--top-k",
                "24",
                "--topics",
                "custom/topics.jsonl",
                "--output-root",
                "/tmp/control-a",
            ]
        )

        self.assertEqual(arguments.top_k, 24)
        self.assertEqual(arguments.topics, Path("custom/topics.jsonl"))
        self.assertEqual(arguments.output_root, Path("/tmp/control-a"))

    def test_parser_rejects_non_positive_top_k(self) -> None:
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as raised:
                build_parser().parse_args(["--run-id", "run-1", "--top-k", "0"])

        self.assertEqual(raised.exception.code, 2)

    def test_main_help_does_not_construct_a_retriever(self) -> None:
        stdout = StringIO()

        with redirect_stdout(stdout):
            self.assertEqual(main(["--help"]), 0)

        self.assertIn("Control-A", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
