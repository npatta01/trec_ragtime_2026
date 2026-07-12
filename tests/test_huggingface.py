from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from collections import deque
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import requests

from trec_ragtime import (
    Document,
    HuggingFaceConfigurationError,
    HuggingFaceCorpus,
    HuggingFaceDocument,
    HuggingFaceHydrationError,
    HydrationFailure,
    HydrationResult,
    PipelineResult,
    RagtimeApiError,
    RagtimeClient,
    RagtimeRetriever,
    SearchHit,
)


class FakeResponse:
    def __init__(self, payload, *, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            response = requests.Response()
            response.status_code = self.status_code
            raise requests.HTTPError(response=response)

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class RecordingGetSession:
    def __init__(self, *responses):
        self.responses = deque(responses)
        self.calls = []
        self.closed = False

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if not self.responses:
            raise AssertionError("unexpected Hugging Face request")
        return self.responses.popleft()

    def close(self):
        self.closed = True


class RecordingPostSession:
    def __init__(self, *responses):
        self.responses = deque(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if not self.responses:
            raise AssertionError("unexpected RAGTIME request")
        return self.responses.popleft()


def document_row(
    document_id,
    text,
    *,
    url=None,
    date=None,
    truncated_cells=None,
):
    row = {"id": document_id, "text": text}
    if url is not None:
        row["url"] = url
    if date is not None:
        row["date"] = date
    return {
        "row_idx": 0,
        "row": row,
        "truncated_cells": [] if truncated_cells is None else truncated_cells,
    }


def filter_payload(*rows, partial=False):
    return {
        "partial": partial,
        "rows": list(rows),
        "num_rows_total": len(rows),
    }


def hf_document(
    document_id,
    text,
    *,
    split="eng",
    local_cache_hit=False,
    english_text=None,
):
    return HuggingFaceDocument(
        dataset="trec-ragtime/ragtime2",
        config="default",
        split=split,
        document_id=document_id,
        text=text,
        url=f"https://source.example/{document_id}",
        date="2026-01-02",
        local_cache_hit=local_cache_hit,
        cache_stored_at=123.0,
        english_text=english_text,
    )


def private_document(document_id, text, *, local_cache_hit=False):
    return Document(
        collection="ragtime2",
        document_id=document_id,
        text=text,
        local_cache_hit=local_cache_hit,
        remote_cached=False,
        server_timestamp=456.0,
        cache_stored_at=789.0,
    )


def pipeline_result(document_ids, *, query="query", pipeline="ragtime2%20"):
    hits = tuple(
        SearchHit(
            rank=index,
            document_id=document_id,
            score=1.0 - (index / 10),
        )
        for index, document_id in enumerate(document_ids, start=1)
    )
    return PipelineResult(
        query=query,
        pipeline=pipeline,
        hits=hits,
        processed=True,
        local_cache_hit=True,
        remote_cached=False,
        server_timestamp=123.0,
        cache_stored_at=122.0,
    )


class FakeRagtimeClient:
    def __init__(self, search_result, *, contents=None):
        self.search_result = search_result
        self.contents = dict(contents or {})
        self.search_calls = []
        self.content_calls = []
        self.closed = False
        self.cache_path = Path("/tmp/fake-ragtime-cache.sqlite3")

    def search(self, query, *, pipeline, refresh):
        self.search_calls.append(
            {"query": query, "pipeline": pipeline, "refresh": refresh}
        )
        return self.search_result

    def get_content(self, document_id, *, refresh):
        self.content_calls.append({"document_id": document_id, "refresh": refresh})
        value = self.contents[document_id]
        if isinstance(value, Exception):
            raise value
        return value

    def close(self):
        self.closed = True


class FakeCorpus:
    backend_name = "fake"

    def __init__(self, result):
        self.result = result
        self.calls = []
        self.closed = False

    def get_many(self, document_ids, *, refresh):
        self.calls.append({"document_ids": tuple(document_ids), "refresh": refresh})
        return self.result

    def enrich_documents(self, document_ids, *, refresh):
        return self.get_many(document_ids, refresh=refresh)

    def close(self):
        self.closed = True


class HuggingFaceCorpusTests(unittest.TestCase):
    def test_backend_is_deprecated(self):
        with self.assertWarnsRegex(DeprecationWarning, "LocalCorpusIndex"):
            corpus = HuggingFaceCorpus(use_cache=False)
        corpus.close()

    def test_batches_or_filters_escapes_ids_and_preserves_requested_order(self):
        session = RecordingGetSession(
            FakeResponse(filter_payload(document_row("doc'a", "A"))),
            FakeResponse(filter_payload(document_row("doc-b", "B"))),
            FakeResponse(filter_payload()),
            FakeResponse(filter_payload(document_row("doc-c", "C"))),
        )
        corpus = HuggingFaceCorpus(
            splits=("eng", "spa"),
            batch_size=2,
            session=session,
            use_cache=False,
        )

        result = corpus.get_many(["doc-b", "doc'a", "doc-c", "doc-b"])

        self.assertEqual(
            [document.document_id for document in result.documents],
            ["doc-b", "doc'a", "doc-c"],
        )
        self.assertEqual(
            [document.split for document in result.documents],
            ["spa", "eng", "spa"],
        )
        self.assertEqual(result.unresolved_ids, ())
        self.assertEqual(result.failures, ())
        self.assertEqual(result.missing_translation_ids, ("doc-b", "doc-c"))
        self.assertEqual(len(session.calls), 4)
        expected_filters = (
            "\"id\"='doc-b' OR \"id\"='doc''a'",
            "\"id\"='doc-b'",
            "\"id\"='doc-c'",
            "\"id\"='doc-c'",
        )
        expected_splits = ("eng", "spa", "eng", "spa")
        for call, expected_filter, expected_split in zip(
            session.calls,
            expected_filters,
            expected_splits,
            strict=True,
        ):
            url, arguments = call
            self.assertEqual(url, "https://datasets-server.huggingface.co/filter")
            self.assertEqual(arguments["headers"]["Accept"], "application/json")
            self.assertIsNone(arguments["headers"]["Authorization"])
            self.assertTrue(callable(arguments["auth"]))
            self.assertEqual(arguments["params"]["where"], expected_filter)
            self.assertEqual(arguments["params"]["split"], expected_split)
            self.assertEqual(arguments["params"]["length"], 100)

    def test_per_document_cache_is_reused_across_corpus_instances(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "responses.sqlite3"
            first_session = RecordingGetSession(
                FakeResponse(
                    filter_payload(
                        document_row("doc-1", "one"),
                        document_row("doc-2", "two"),
                    )
                )
            )
            first = HuggingFaceCorpus(
                splits=("eng",),
                cache_path=cache_path,
                session=first_session,
            )
            first_result = first.get_many(["doc-2", "doc-1"])

            second_session = RecordingGetSession()
            second = HuggingFaceCorpus(
                splits=("eng",),
                cache_path=cache_path,
                session=second_session,
            )
            second_result = second.get_many(["doc-1", "doc-2"])

            self.assertEqual(len(first_session.calls), 1)
            self.assertEqual(second_session.calls, [])
            self.assertTrue(
                all(document.local_cache_hit for document in second_result.documents)
            )
            self.assertEqual(
                [document.document_id for document in first_result.documents],
                ["doc-2", "doc-1"],
            )
            self.assertEqual(
                [document.document_id for document in second_result.documents],
                ["doc-1", "doc-2"],
            )
            with closing(sqlite3.connect(cache_path)) as connection:
                count = connection.execute(
                    "SELECT COUNT(*) FROM responses "
                    "WHERE operation = 'huggingface_document'"
                ).fetchone()[0]
            self.assertEqual(count, 2)

    def test_private_token_is_never_sent_or_written_to_hf_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache" / "responses.sqlite3"
            token = "unique-private-ragtime-token"
            session = RecordingGetSession(
                FakeResponse(filter_payload(document_row("doc-1", "safe text")))
            )
            with patch.dict(
                os.environ,
                {"RAGTIME_BEARER_TOKEN": token},
                clear=False,
            ):
                result = HuggingFaceCorpus(
                    splits=("eng",),
                    cache_path=cache_path,
                    session=session,
                ).get_many(["doc-1"])

            self.assertEqual(len(result.documents), 1)
            self.assertEqual(len(session.calls), 1)
            headers = session.calls[0][1]["headers"]
            self.assertIsNone(headers["Authorization"])
            anonymous_auth = session.calls[0][1]["auth"]
            self.assertTrue(callable(anonymous_auth))
            prepared = requests.Request(
                "GET",
                "https://datasets-server.huggingface.co/filter",
                headers={"Authorization": f"Bearer {token}"},
            ).prepare()
            anonymous_auth(prepared)
            self.assertNotIn("Authorization", prepared.headers)
            self.assertNotIn(token, repr(session.calls))
            for cache_file in cache_path.parent.iterdir():
                if cache_file.is_file():
                    self.assertNotIn(token.encode(), cache_file.read_bytes())

    def test_malformed_partial_and_truncated_payloads_are_not_cached(self):
        invalid_payloads = {
            "malformed": {"partial": False, "rows": "not-a-list"},
            "missing-count": {
                "partial": False,
                "rows": [document_row("doc-1", "unsafe uncounted text")],
            },
            "inconsistent-count": {
                "partial": False,
                "rows": [document_row("doc-1", "unsafe miscounted text")],
                "num_rows_total": 2,
            },
            "partial": filter_payload(
                document_row("doc-1", "unsafe partial text"),
                partial=True,
            ),
            "missing-truncated-cells": {
                "partial": False,
                "num_rows_total": 1,
                "rows": [
                    {
                        "row_idx": 0,
                        "row": {"id": "doc-1", "text": "unsafe unknown text"},
                    }
                ],
            },
            "truncated": filter_payload(
                document_row(
                    "doc-1",
                    "unsafe truncated text",
                    truncated_cells=["text"],
                )
            ),
        }
        for label, payload in invalid_payloads.items():
            with (
                self.subTest(payload=label),
                tempfile.TemporaryDirectory() as directory,
            ):
                cache_path = Path(directory) / "responses.sqlite3"
                invalid_result = HuggingFaceCorpus(
                    splits=("eng",),
                    cache_path=cache_path,
                    session=RecordingGetSession(FakeResponse(payload)),
                ).get_many(["doc-1"])

                self.assertEqual(invalid_result.documents, ())
                self.assertEqual(invalid_result.unresolved_ids, ("doc-1",))
                self.assertEqual(len(invalid_result.failures), 1)
                with closing(sqlite3.connect(cache_path)) as connection:
                    count = connection.execute(
                        "SELECT COUNT(*) FROM responses"
                    ).fetchone()[0]
                self.assertEqual(count, 0)

                valid_session = RecordingGetSession(
                    FakeResponse(filter_payload(document_row("doc-1", "valid")))
                )
                valid_result = HuggingFaceCorpus(
                    splits=("eng",),
                    cache_path=cache_path,
                    session=valid_session,
                ).get_many(["doc-1"])
                self.assertEqual(len(valid_session.calls), 1)
                self.assertFalse(valid_result.documents[0].local_cache_hit)
                self.assertEqual(valid_result.documents[0].text, "valid")

    def test_credentialed_sessions_are_rejected_before_any_request(self):
        credential_variants = {
            "authorization-header": ("headers", {"Authorization": "Bearer secret"}),
            "lowercase-authorization": (
                "headers",
                {"authorization": "Bearer secret"},
            ),
            "session-auth": ("auth", ("username", "password")),
        }
        for label, (attribute, value) in credential_variants.items():
            with self.subTest(credential=label):
                session = RecordingGetSession()
                setattr(session, attribute, value)
                corpus = HuggingFaceCorpus(
                    splits=("eng",),
                    session=session,
                    use_cache=False,
                )

                with self.assertRaises(HuggingFaceConfigurationError):
                    corpus.get_many(["doc-1"])

                self.assertEqual(session.calls, [])

    def test_session_cookies_are_cleared_before_request(self):
        session = RecordingGetSession(
            FakeResponse(filter_payload(document_row("doc-1", "text")))
        )
        session.cookies = ["secret-cookie"]
        result = HuggingFaceCorpus(
            splits=("eng",),
            session=session,
            use_cache=False,
        ).get_many(["doc-1"])

        self.assertEqual(session.cookies, [])
        self.assertEqual(len(session.calls), 1)
        self.assertEqual(result.documents[0].document_id, "doc-1")

    def test_boolean_options_are_strict(self):
        with self.assertRaises(HuggingFaceConfigurationError):
            HuggingFaceCorpus(use_cache=1)
        with self.assertRaises(HuggingFaceConfigurationError):
            HuggingFaceCorpus(retries=True)

        session = RecordingGetSession()
        corpus = HuggingFaceCorpus(
            splits=("eng",),
            session=session,
            use_cache=False,
        )
        with self.assertRaisesRegex(ValueError, "refresh must be a boolean"):
            corpus.get_many(["doc-1"], refresh=1)
        self.assertEqual(session.calls, [])

    def test_rate_limit_stops_all_remaining_splits_and_batches(self):
        session = RecordingGetSession(FakeResponse({}, status_code=429))
        corpus = HuggingFaceCorpus(
            splits=("eng", "spa", "rus", "zho"),
            batch_size=1,
            session=session,
            use_cache=False,
        )

        result = corpus.get_many(["doc-1", "doc-2", "doc-3"])

        self.assertEqual(len(session.calls), 1)
        self.assertEqual(result.documents, ())
        self.assertEqual(result.unresolved_ids, ("doc-1", "doc-2", "doc-3"))
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0].split, "eng")
        self.assertIn("HTTP 429", result.failures[0].message)

    def test_split_failure_is_reported_even_when_later_split_finds_document(self):
        session = RecordingGetSession(
            FakeResponse({}, status_code=500),
            FakeResponse(filter_payload(document_row("doc-1", "texto"))),
        )
        result = HuggingFaceCorpus(
            splits=("eng", "spa"),
            session=session,
            use_cache=False,
        ).get_many(["doc-1"])

        self.assertEqual(result.unresolved_ids, ())
        self.assertEqual(result.documents[0].split, "spa")
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0].split, "eng")


class RagtimeRetrieverTests(unittest.TestCase):
    def test_enriches_every_api_result_without_a_twenty_document_cap(self):
        document_ids = [f"doc-{index:03d}" for index in range(137)]
        search = pipeline_result(document_ids, pipeline="ragtime2%137")
        corpus = FakeCorpus(
            HydrationResult(
                documents=tuple(
                    hf_document(document_id, f"text {document_id}")
                    for document_id in document_ids
                ),
                unresolved_ids=(),
                failures=(),
            )
        )

        result = RagtimeRetriever(FakeRagtimeClient(search), corpus).retrieve(
            "query",
            pipeline="ragtime2%137",
            top_k=None,
        )

        self.assertEqual(len(result.documents), 137)
        self.assertEqual(
            [document.document_id for document in result.documents],
            document_ids,
        )
        self.assertEqual(corpus.calls[0]["document_ids"], tuple(document_ids))

    def test_public_enrich_documents_delegates_arbitrary_ids(self):
        corpus = FakeCorpus(
            HydrationResult(
                documents=(hf_document("doc-2", "two"),),
                unresolved_ids=("missing",),
                failures=(),
            )
        )
        retriever = RagtimeRetriever(
            FakeRagtimeClient(pipeline_result([])),
            corpus,
        )

        result = retriever.enrich_documents(
            ["doc-2", "missing", "doc-2"],
            refresh_documents=True,
        )

        self.assertEqual(result.documents[0].document_id, "doc-2")
        self.assertEqual(result.unresolved_ids, ("missing",))
        self.assertEqual(retriever.document_backend, "fake")
        self.assertEqual(
            corpus.calls,
            [
                {
                    "document_ids": ("doc-2", "missing", "doc-2"),
                    "refresh": True,
                }
            ],
        )

    def test_rank_and_score_are_preserved_when_hydration_order_differs(self):
        search = pipeline_result(["doc-high", "doc-mid", "doc-low"])
        client = FakeRagtimeClient(search)
        corpus = FakeCorpus(
            HydrationResult(
                documents=(
                    hf_document(
                        "doc-mid",
                        "medio",
                        split="spa",
                        english_text="middle",
                    ),
                    hf_document(
                        "doc-high",
                        "highest",
                        split="eng",
                        local_cache_hit=True,
                    ),
                ),
                unresolved_ids=(),
                failures=(),
            )
        )

        result = RagtimeRetriever(client, corpus).retrieve(
            "query",
            top_k=2,
            refresh_search=True,
            refresh_documents=True,
        )

        self.assertIs(result.search_result, search)
        self.assertEqual(
            [document.document_id for document in result.documents],
            ["doc-high", "doc-mid"],
        )
        self.assertEqual([document.rank for document in result.documents], [1, 2])
        self.assertEqual(
            [document.score for document in result.documents],
            [search.hits[0].score, search.hits[1].score],
        )
        self.assertEqual(
            [document.text for document in result.documents],
            ["highest", "medio"],
        )
        self.assertEqual(
            [document.report_text for document in result.documents],
            ["highest", "middle"],
        )
        self.assertEqual(
            [document.language for document in result.documents],
            ["eng", "spa"],
        )
        self.assertEqual(
            [document.content_source for document in result.documents],
            ["huggingface", "huggingface"],
        )
        self.assertEqual(
            client.search_calls,
            [
                {
                    "query": "query",
                    "pipeline": "ragtime2%20",
                    "refresh": True,
                }
            ],
        )
        self.assertEqual(
            corpus.calls,
            [
                {
                    "document_ids": ("doc-high", "doc-mid"),
                    "refresh": True,
                }
            ],
        )
        self.assertEqual(client.content_calls, [])

    def test_unresolved_documents_use_cached_private_content_by_default(self):
        search = pipeline_result(["doc-1", "doc-2"])
        client = FakeRagtimeClient(
            search,
            contents={
                "doc-1": private_document("doc-1", "one"),
                "doc-2": private_document("doc-2", "two"),
            },
        )
        corpus = FakeCorpus(
            HydrationResult(
                documents=(),
                unresolved_ids=("doc-1", "doc-2"),
                failures=(),
            )
        )

        result = RagtimeRetriever(client, corpus).retrieve("query")

        self.assertEqual(
            client.content_calls,
            [
                {"document_id": "doc-1", "refresh": False},
                {"document_id": "doc-2", "refresh": False},
            ],
        )
        self.assertEqual(result.unresolved_ids, ())
        self.assertTrue(result.fully_hydrated)
        self.assertEqual(result.missing_translation_ids, ("doc-1", "doc-2"))
        self.assertFalse(result.fully_report_ready)
        self.assertTrue(
            all(
                document.content_source == "ragtime_api"
                for document in result.documents
            )
        )

    def test_automatic_content_fallback_can_be_disabled(self):
        search = pipeline_result(["doc-1"])
        client = FakeRagtimeClient(
            search,
            contents={"doc-1": private_document("doc-1", "one")},
        )
        corpus = FakeCorpus(
            HydrationResult(
                documents=(),
                unresolved_ids=("doc-1",),
                failures=(),
            )
        )

        result = RagtimeRetriever(client, corpus).retrieve(
            "query",
            max_content_fallbacks=0,
        )

        self.assertEqual(client.content_calls, [])
        self.assertEqual(result.unresolved_ids, ("doc-1",))

    def test_automatic_content_fallback_has_a_five_document_circuit_bound(self):
        document_ids = [f"doc-{number}" for number in range(7)]
        client = FakeRagtimeClient(
            pipeline_result(document_ids),
            contents={
                document_id: private_document(document_id, document_id)
                for document_id in document_ids
            },
        )
        corpus = FakeCorpus(
            HydrationResult(
                documents=(),
                unresolved_ids=tuple(document_ids),
                failures=(),
            )
        )

        result = RagtimeRetriever(client, corpus).retrieve("query")

        self.assertEqual(len(client.content_calls), 5)
        self.assertEqual(result.unresolved_ids, ("doc-5", "doc-6"))

    def test_true_miss_fallback_stops_after_first_content_error(self):
        document_ids = ["doc-1", "doc-2", "doc-3"]
        client = FakeRagtimeClient(
            pipeline_result(document_ids),
            contents={
                "doc-1": private_document("doc-1", "one"),
                "doc-2": RagtimeApiError("RAGTIME /content failed: HTTP 503"),
                "doc-3": private_document("doc-3", "three"),
            },
        )
        corpus = FakeCorpus(
            HydrationResult(
                documents=(),
                unresolved_ids=tuple(document_ids),
                failures=(),
            )
        )

        result = RagtimeRetriever(client, corpus).retrieve(
            "query",
            max_content_fallbacks=3,
        )

        self.assertEqual(
            [call["document_id"] for call in client.content_calls],
            ["doc-1", "doc-2"],
        )
        self.assertEqual(result.unresolved_ids, ("doc-2", "doc-3"))

    def test_unlimited_true_miss_fallback_hydrates_every_result(self):
        document_ids = [f"doc-{number}" for number in range(7)]
        client = FakeRagtimeClient(
            pipeline_result(document_ids),
            contents={
                document_id: private_document(document_id, document_id)
                for document_id in document_ids
            },
        )
        corpus = FakeCorpus(
            HydrationResult(
                documents=(),
                unresolved_ids=tuple(document_ids),
                failures=(),
            )
        )

        result = RagtimeRetriever(client, corpus).retrieve(
            "query",
            max_content_fallbacks=None,
        )

        self.assertEqual(len(client.content_calls), len(document_ids))
        self.assertEqual(result.unresolved_ids, ())

    def test_automatic_fallback_reuses_the_shared_content_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache" / "api.sqlite3"
            pipeline_payload = {
                "pipeline": "ragtime2%20",
                "processed": True,
                "query": "query",
                "scores": {"doc-1": 1.0},
                "timestamp": 123.0,
            }
            content_payload = {
                "cached": False,
                "collection": "ragtime2",
                "id": "doc-1",
                "text": "endpoint text",
            }
            first_session = RecordingPostSession(
                FakeResponse(pipeline_payload),
                FakeResponse(content_payload),
            )
            missing = HydrationResult(
                documents=(),
                unresolved_ids=("doc-1",),
                failures=(),
            )
            with RagtimeRetriever(
                RagtimeClient(
                    "https://api.example",
                    "secret",
                    cache_path=cache_path,
                    session=first_session,
                ),
                FakeCorpus(missing),
            ) as first:
                first_result = first.retrieve("query")

            second_session = RecordingPostSession()
            with RagtimeRetriever(
                RagtimeClient(
                    "https://api.example",
                    "different-secret",
                    cache_path=cache_path,
                    session=second_session,
                ),
                FakeCorpus(missing),
            ) as second:
                second_result = second.retrieve("query")

            self.assertEqual(len(first_session.calls), 2)
            self.assertEqual(second_session.calls, [])
            self.assertFalse(first_result.documents[0].content_local_cache_hit)
            self.assertTrue(second_result.documents[0].content_local_cache_hit)

    def test_missing_report_text_ids_preserve_search_rank_across_sources(self):
        client = FakeRagtimeClient(
            pipeline_result(["doc-api", "doc-hf"]),
            contents={"doc-api": private_document("doc-api", "unknown language")},
        )
        corpus = FakeCorpus(
            HydrationResult(
                documents=(hf_document("doc-hf", "texto", split="spa"),),
                unresolved_ids=("doc-api",),
                failures=(),
                missing_translation_ids=("doc-hf",),
            )
        )

        result = RagtimeRetriever(client, corpus).retrieve("query")

        self.assertEqual(
            result.missing_translation_ids,
            ("doc-api", "doc-hf"),
        )
        self.assertTrue(result.fully_hydrated)
        self.assertFalse(result.fully_report_ready)
        self.assertTrue(
            all(document.report_text is None for document in result.documents)
        )

    def test_content_fallback_is_explicit_bounded_and_rank_ordered(self):
        search = pipeline_result(["doc-1", "doc-2", "doc-3", "doc-4"])
        client = FakeRagtimeClient(
            search,
            contents={
                "doc-1": private_document("doc-1", "one", local_cache_hit=True),
                "doc-2": private_document("doc-2", "two"),
                "doc-3": private_document("doc-3", "three"),
            },
        )
        corpus = FakeCorpus(
            HydrationResult(
                documents=(hf_document("doc-4", "four", split="zho"),),
                unresolved_ids=("doc-3", "doc-2", "doc-1"),
                failures=(),
            )
        )

        result = RagtimeRetriever(client, corpus).retrieve(
            "query",
            max_content_fallbacks=2,
            refresh_documents=True,
        )

        self.assertEqual(
            client.content_calls,
            [
                {"document_id": "doc-1", "refresh": True},
                {"document_id": "doc-2", "refresh": True},
            ],
        )
        self.assertEqual(
            [document.document_id for document in result.documents],
            ["doc-1", "doc-2", "doc-3", "doc-4"],
        )
        self.assertEqual(
            [document.content_source for document in result.documents],
            ["ragtime_api", "ragtime_api", None, "huggingface"],
        )
        self.assertEqual(result.unresolved_ids, ("doc-3",))

    def test_hf_failure_with_unresolved_ids_raises_before_private_fallback(self):
        search = pipeline_result(["doc-1", "doc-2"])
        client = FakeRagtimeClient(
            search,
            contents={
                "doc-1": private_document("doc-1", "one"),
                "doc-2": private_document("doc-2", "two"),
            },
        )
        failure = HydrationFailure(
            split="eng",
            message="Dataset Viewer /filter failed: HTTP 500",
        )
        corpus = FakeCorpus(
            HydrationResult(
                documents=(),
                unresolved_ids=("doc-1", "doc-2"),
                failures=(failure,),
            )
        )

        with self.assertRaises(HuggingFaceHydrationError) as caught:
            RagtimeRetriever(client, corpus).retrieve(
                "query",
                max_content_fallbacks=2,
            )

        self.assertEqual(caught.exception.failures, (failure,))
        self.assertEqual(client.content_calls, [])

    def test_resolved_hits_are_returned_with_nonfatal_split_failure(self):
        search = pipeline_result(["doc-1"])
        failure = HydrationFailure(
            split="eng",
            message="Dataset Viewer /filter failed: HTTP 500",
        )
        client = FakeRagtimeClient(search)
        corpus = FakeCorpus(
            HydrationResult(
                documents=(hf_document("doc-1", "texto", split="spa"),),
                unresolved_ids=(),
                failures=(failure,),
            )
        )

        result = RagtimeRetriever(client, corpus).retrieve("query")

        self.assertTrue(result.fully_hydrated)
        self.assertEqual(result.documents[0].text, "texto")
        self.assertEqual(result.hydration_failures, (failure,))
        self.assertEqual(client.content_calls, [])

    def test_opted_in_error_fallback_honors_cap_and_records_content_failure(self):
        search = pipeline_result(["doc-1", "doc-2", "doc-3"])
        client = FakeRagtimeClient(
            search,
            contents={
                "doc-1": private_document("doc-1", "one"),
                "doc-2": RagtimeApiError("RAGTIME /content request failed: HTTP 503"),
                "doc-3": private_document("doc-3", "three"),
            },
        )
        hf_failure = HydrationFailure(
            split="eng",
            message="Dataset Viewer /filter failed: HTTP 429",
        )
        corpus = FakeCorpus(
            HydrationResult(
                documents=(),
                unresolved_ids=("doc-1", "doc-2", "doc-3"),
                failures=(hf_failure,),
            )
        )

        result = RagtimeRetriever(client, corpus).retrieve(
            "query",
            max_content_fallbacks=3,
            fallback_on_hf_error=True,
        )

        self.assertEqual(
            [call["document_id"] for call in client.content_calls],
            ["doc-1", "doc-2"],
        )
        self.assertEqual(
            [document.content_source for document in result.documents],
            ["ragtime_api", None, None],
        )
        self.assertEqual(result.unresolved_ids, ("doc-2", "doc-3"))
        self.assertEqual(result.hydration_failures[0], hf_failure)
        self.assertEqual(result.hydration_failures[1].split, "ragtime2/content")
        self.assertIn(
            "HTTP 503",
            result.hydration_failures[1].message,
        )

    def test_boolean_controls_are_rejected_before_search(self):
        invalid_options = (
            {"refresh_search": 1},
            {"refresh_documents": 0},
            {"fallback_on_hf_error": "yes"},
        )
        for options in invalid_options:
            with self.subTest(options=options):
                client = FakeRagtimeClient(pipeline_result(["doc-1"]))
                corpus = FakeCorpus(
                    HydrationResult(documents=(), unresolved_ids=(), failures=())
                )
                with self.assertRaisesRegex(ValueError, "must be a boolean"):
                    RagtimeRetriever(client, corpus).retrieve("query", **options)
                self.assertEqual(client.search_calls, [])

    def test_unlimited_fallback_is_rejected_for_hydration_errors(self):
        client = FakeRagtimeClient(pipeline_result(["doc-1"]))
        corpus = FakeCorpus(
            HydrationResult(documents=(), unresolved_ids=(), failures=())
        )

        with self.assertRaisesRegex(ValueError, "requires a finite"):
            RagtimeRetriever(client, corpus).retrieve(
                "query",
                max_content_fallbacks=None,
                fallback_on_hf_error=True,
            )

        self.assertEqual(client.search_calls, [])


if __name__ == "__main__":
    unittest.main()
