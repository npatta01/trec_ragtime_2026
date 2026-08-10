from __future__ import annotations

import os
import unittest
from collections import deque
from typing import Any, Callable
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
    ResponseCache,
    SearchHit,
)
from trec_ragtime.artifact_store import ArtifactValidationError


class MemoryArtifactStore:
    def __init__(self) -> None:
        self.local: dict[str, bytes] = {}
        self.remote: dict[str, bytes] = {}
        self.read_calls: list[str] = []
        self.write_calls: list[tuple[str, bytes]] = []
        self.repaired_local_candidates = 0
        self.read_error: BaseException | None = None
        self.write_error: BaseException | None = None

    def read_validated(
        self, object_key: str, decoder: Callable[[bytes], Any]
    ) -> Any | None:
        self.read_calls.append(object_key)
        if self.read_error is not None:
            raise self.read_error
        payload = self.local.get(object_key)
        if payload is not None:
            try:
                return decoder(payload)
            except ArtifactValidationError:
                self.repaired_local_candidates += 1
                del self.local[object_key]
        payload = self.remote.get(object_key)
        if payload is None:
            return None
        value = decoder(payload)
        self.local[object_key] = payload
        return value

    def write(self, object_key: str, payload: bytes) -> None:
        if self.write_error is not None:
            raise self.write_error
        self.write_calls.append((object_key, payload))
        self.local[object_key] = payload
        self.remote[object_key] = payload


class FakeResponse:
    def __init__(self, payload: Any, *, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            response = requests.Response()
            response.status_code = self.status_code
            raise requests.HTTPError(response=response)

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class RecordingGetSession:
    def __init__(self, *responses: FakeResponse):
        self.responses = deque(responses)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((url, kwargs))
        if not self.responses:
            raise AssertionError("unexpected Hugging Face request")
        return self.responses.popleft()

    def close(self) -> None:
        self.closed = True


class RecordingPostSession:
    def __init__(self, *responses: FakeResponse):
        self.responses = deque(responses)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((url, kwargs))
        if not self.responses:
            raise AssertionError("unexpected RAGTIME request")
        return self.responses.popleft()

    def close(self) -> None:
        self.closed = True


def document_row(
    document_id: str,
    text: str,
    *,
    url: str | None = None,
    date: str | None = None,
    truncated_cells: list[str] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {"id": document_id, "text": text}
    if url is not None:
        row["url"] = url
    if date is not None:
        row["date"] = date
    return {
        "row_idx": 0,
        "row": row,
        "truncated_cells": [] if truncated_cells is None else truncated_cells,
    }


def filter_payload(*rows: dict[str, Any], partial: bool = False) -> dict[str, Any]:
    return {"partial": partial, "rows": list(rows), "num_rows_total": len(rows)}


def make_cache() -> tuple[ResponseCache, MemoryArtifactStore]:
    store = MemoryArtifactStore()
    return ResponseCache(store), store


def hf_document(
    document_id: str,
    text: str,
    *,
    split: str = "eng",
    local_cache_hit: bool = False,
    english_text: str | None = None,
) -> HuggingFaceDocument:
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


def private_document(
    document_id: str, text: str, *, local_cache_hit: bool = False
) -> Document:
    return Document(
        collection="ragtime2",
        document_id=document_id,
        text=text,
        local_cache_hit=local_cache_hit,
        remote_cached=False,
        server_timestamp=456.0,
        cache_stored_at=789.0,
    )


def pipeline_result(
    document_ids: list[str], *, query: str = "query", pipeline: str = "ragtime2"
) -> PipelineResult:
    hits = tuple(
        SearchHit(rank=index, document_id=document_id, score=1.0 - (index / 10))
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
    def __init__(self, search_result: PipelineResult, *, contents: dict[str, Any] | None = None):
        self.search_result = search_result
        self.contents = dict(contents or {})
        self.search_calls: list[dict[str, Any]] = []
        self.content_calls: list[dict[str, Any]] = []
        self.closed = False
        self.artifact_store = None
        self.cache_configuration = None

    def search(self, query: str, *, pipeline: str, refresh: bool) -> PipelineResult:
        self.search_calls.append({"query": query, "pipeline": pipeline, "refresh": refresh})
        return self.search_result

    def get_content(self, document_id: str, *, refresh: bool) -> Document:
        self.content_calls.append({"document_id": document_id, "refresh": refresh})
        value = self.contents[document_id]
        if isinstance(value, Exception):
            raise value
        return value

    def close(self) -> None:
        self.closed = True


class FakeCorpus:
    backend_name = "fake"

    def __init__(self, result: HydrationResult):
        self.result = result
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    def get_many(self, document_ids: Any, *, refresh: bool) -> HydrationResult:
        self.calls.append({"document_ids": tuple(document_ids), "refresh": refresh})
        return self.result

    def enrich_documents(self, document_ids: Any, *, refresh: bool) -> HydrationResult:
        return self.get_many(document_ids, refresh=refresh)

    def close(self) -> None:
        self.closed = True


class HuggingFaceCorpusTests(unittest.TestCase):
    def test_backend_is_deprecated(self) -> None:
        with self.assertWarnsRegex(DeprecationWarning, "LocalCorpusIndex"):
            corpus = HuggingFaceCorpus(use_cache=False)
        corpus.close()

    def test_batches_filters_escapes_ids_and_preserves_requested_order(self) -> None:
        session = RecordingGetSession(
            FakeResponse(filter_payload(document_row("doc'a", "A"))),
            FakeResponse(filter_payload(document_row("doc-b", "B"))),
            FakeResponse(filter_payload()),
            FakeResponse(filter_payload(document_row("doc-c", "C"))),
        )
        corpus = HuggingFaceCorpus(
            splits=("eng", "spa"), batch_size=2, session=session, use_cache=False
        )
        result = corpus.get_many(["doc-b", "doc'a", "doc-c", "doc-b"])
        self.assertEqual(
            [document.document_id for document in result.documents],
            ["doc-b", "doc'a", "doc-c"],
        )
        self.assertEqual([document.split for document in result.documents], ["spa", "eng", "spa"])
        self.assertEqual(result.unresolved_ids, ())
        self.assertEqual(result.missing_translation_ids, ("doc-b", "doc-c"))
        self.assertEqual(len(session.calls), 4)
        expected_filters = (
            '"id"=\'doc-b\' OR "id"=\'doc\'\'a\'',
            '"id"=\'doc-b\'',
            '"id"=\'doc-c\'',
            '"id"=\'doc-c\'',
        )
        for call, expected_filter, expected_split in zip(
            session.calls, expected_filters, ("eng", "spa", "eng", "spa"), strict=True
        ):
            url, arguments = call
            self.assertEqual(url, "https://datasets-server.huggingface.co/filter")
            self.assertEqual(arguments["headers"]["Accept"], "application/json")
            self.assertIsNone(arguments["headers"]["Authorization"])
            self.assertTrue(callable(arguments["auth"]))
            self.assertEqual(arguments["params"]["where"], expected_filter)
            self.assertEqual(arguments["params"]["split"], expected_split)
            self.assertEqual(arguments["params"]["length"], 100)

    def test_per_document_response_cache_is_reused_without_viewer_requests(self) -> None:
        cache, store = make_cache()
        first_session = RecordingGetSession(
            FakeResponse(filter_payload(document_row("doc-1", "one"), document_row("doc-2", "two")))
        )
        first = HuggingFaceCorpus(splits=("eng",), cache=cache, session=first_session)
        first_result = first.get_many(["doc-2", "doc-1"])
        second_session = RecordingGetSession()
        second = HuggingFaceCorpus(splits=("eng",), cache=cache, session=second_session)
        second_result = second.get_many(["doc-1", "doc-2"])
        self.assertEqual(len(first_session.calls), 1)
        self.assertEqual(second_session.calls, [])
        self.assertTrue(all(document.local_cache_hit for document in second_result.documents))
        self.assertEqual([document.document_id for document in first_result.documents], ["doc-2", "doc-1"])
        self.assertEqual([document.document_id for document in second_result.documents], ["doc-1", "doc-2"])
        self.assertEqual(len(store.write_calls), 2)

    def test_private_token_is_never_sent_or_written_to_hf_cache(self) -> None:
        cache, store = make_cache()
        token = "unique-private-ragtime-token"
        session = RecordingGetSession(FakeResponse(filter_payload(document_row("doc-1", "safe text"))))
        with patch.dict(os.environ, {"RAGTIME_BEARER_TOKEN": token}, clear=False):
            result = HuggingFaceCorpus(splits=("eng",), cache=cache, session=session).get_many(["doc-1"])
        self.assertEqual(len(result.documents), 1)
        headers = session.calls[0][1]["headers"]
        self.assertIsNone(headers["Authorization"])
        prepared = requests.Request(
            "GET", "https://datasets-server.huggingface.co/filter",
            headers={"Authorization": f"Bearer {token}"},
        ).prepare()
        session.calls[0][1]["auth"](prepared)
        self.assertNotIn("Authorization", prepared.headers)
        self.assertNotIn(token, repr(store.write_calls))

    def test_malformed_viewer_payload_never_writes(self) -> None:
        invalid_payloads = (
            {"partial": False, "rows": "not-a-list"},
            {"partial": True, "rows": [], "num_rows_total": 0},
            filter_payload(document_row("doc-1", "unsafe", truncated_cells=["text"])),
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                cache, store = make_cache()
                result = HuggingFaceCorpus(
                    splits=("eng",), cache=cache,
                    session=RecordingGetSession(FakeResponse(payload)),
                ).get_many(["doc-1"])
                self.assertEqual(result.documents, ())
                self.assertEqual(result.unresolved_ids, ("doc-1",))
                self.assertEqual(store.write_calls, [])

    def test_cache_read_and_write_failures_are_fatal(self) -> None:
        cache, store = make_cache()
        store.read_error = OSError("cache read failed")
        with self.assertRaises(OSError):
            HuggingFaceCorpus(splits=("eng",), cache=cache, session=RecordingGetSession()).get_many(["doc-1"])

        cache, store = make_cache()
        store.write_error = OSError("cache write failed")
        with self.assertRaises(OSError):
            HuggingFaceCorpus(
                splits=("eng",), cache=cache,
                session=RecordingGetSession(FakeResponse(filter_payload(document_row("doc-1", "text")))),
            ).get_many(["doc-1"])

    def test_malformed_l1_repairs_once_and_corrupt_l2_is_fatal(self) -> None:
        cache, store = make_cache()
        request = {
            "dataset": "trec-ragtime/ragtime2", "config": "default",
            "splits": ["eng"], "variant": "original", "id": "doc-1",
        }
        cache.put(
            base_url="https://datasets-server.huggingface.co",
            operation="huggingface_document", request=request,
            response={"dataset": "trec-ragtime/ragtime2", "config": "default", "split": "eng", "id": "doc-1", "text": "old"},
        )
        key = cache.object_key(
            base_url="https://datasets-server.huggingface.co",
            operation="huggingface_document", request=request,
        )
        store.local[key] = b"malformed"
        result = HuggingFaceCorpus(
            splits=("eng",), cache=cache, session=RecordingGetSession()
        ).get_many(["doc-1"])
        self.assertEqual(result.documents[0].text, "old")
        self.assertEqual(store.repaired_local_candidates, 1)

        store.local.clear()
        store.remote[key] = b"corrupt-l2"
        with self.assertRaises(ArtifactValidationError):
            HuggingFaceCorpus(splits=("eng",), cache=cache, session=RecordingGetSession()).get_many(["doc-1"])

    def test_credentialed_sessions_are_rejected_before_any_request(self) -> None:
        for attribute, value in (
            ("headers", {"Authorization": "Bearer secret"}),
            ("headers", {"authorization": "Bearer secret"}),
            ("auth", ("username", "password")),
        ):
            session = RecordingGetSession()
            setattr(session, attribute, value)
            corpus = HuggingFaceCorpus(splits=("eng",), session=session, use_cache=False)
            with self.assertRaises(HuggingFaceConfigurationError):
                corpus.get_many(["doc-1"])
            self.assertEqual(session.calls, [])

    def test_session_cookies_are_cleared_before_request(self) -> None:
        session = RecordingGetSession(FakeResponse(filter_payload(document_row("doc-1", "text"))))
        session.cookies = ["secret-cookie"]
        result = HuggingFaceCorpus(splits=("eng",), session=session, use_cache=False).get_many(["doc-1"])
        self.assertEqual(session.cookies, [])
        self.assertEqual(result.documents[0].document_id, "doc-1")

    def test_boolean_options_are_strict(self) -> None:
        with self.assertRaises(HuggingFaceConfigurationError):
            HuggingFaceCorpus(use_cache=1)
        with self.assertRaises(HuggingFaceConfigurationError):
            HuggingFaceCorpus(retries=True)
        corpus = HuggingFaceCorpus(splits=("eng",), session=RecordingGetSession(), use_cache=False)
        with self.assertRaisesRegex(ValueError, "refresh must be a boolean"):
            corpus.get_many(["doc-1"], refresh=1)

    def test_rate_limit_stops_all_remaining_splits_and_batches(self) -> None:
        session = RecordingGetSession(FakeResponse({}, status_code=429))
        result = HuggingFaceCorpus(
            splits=("eng", "spa", "rus", "zho"), batch_size=1,
            session=session, use_cache=False,
        ).get_many(["doc-1", "doc-2", "doc-3"])
        self.assertEqual(len(session.calls), 1)
        self.assertEqual(result.unresolved_ids, ("doc-1", "doc-2", "doc-3"))
        self.assertEqual(result.failures[0].split, "eng")

    def test_split_failure_is_reported_even_when_later_split_finds_document(self) -> None:
        session = RecordingGetSession(
            FakeResponse({}, status_code=500),
            FakeResponse(filter_payload(document_row("doc-1", "texto"))),
        )
        result = HuggingFaceCorpus(
            splits=("eng", "spa"), session=session, use_cache=False
        ).get_many(["doc-1"])
        self.assertEqual(result.unresolved_ids, ())
        self.assertEqual(result.documents[0].split, "spa")
        self.assertEqual(result.failures[0].split, "eng")


class RagtimeRetrieverTests(unittest.TestCase):
    def test_enriches_every_api_result_without_a_twenty_document_cap(self) -> None:
        document_ids = [f"doc-{index:03d}" for index in range(137)]
        corpus = FakeCorpus(
            HydrationResult(
                documents=tuple(hf_document(document_id, f"text {document_id}") for document_id in document_ids),
                unresolved_ids=(), failures=(),
            )
        )
        result = RagtimeRetriever(
            FakeRagtimeClient(pipeline_result(document_ids, pipeline="ragtime2%137")), corpus
        ).retrieve("query", pipeline="ragtime2%137", top_k=None)
        self.assertEqual(len(result.documents), 137)
        self.assertEqual([document.document_id for document in result.documents], document_ids)
        self.assertEqual(corpus.calls[0]["document_ids"], tuple(document_ids))

    def test_public_enrich_documents_delegates_arbitrary_ids(self) -> None:
        corpus = FakeCorpus(HydrationResult(documents=(hf_document("doc-2", "two"),), unresolved_ids=("missing",), failures=()))
        retriever = RagtimeRetriever(FakeRagtimeClient(pipeline_result([])), corpus)
        result = retriever.enrich_documents(["doc-2", "missing", "doc-2"], refresh_documents=True)
        self.assertEqual(result.documents[0].document_id, "doc-2")
        self.assertEqual(result.unresolved_ids, ("missing",))
        self.assertEqual(retriever.document_backend, "fake")
        self.assertEqual(corpus.calls[0]["document_ids"], ("doc-2", "missing", "doc-2"))

    def test_rank_score_and_report_text_are_preserved_when_hydration_order_differs(self) -> None:
        search = pipeline_result(["doc-high", "doc-mid", "doc-low"])
        corpus = FakeCorpus(
            HydrationResult(
                documents=(
                    hf_document("doc-mid", "medio", split="spa", english_text="middle"),
                    hf_document("doc-high", "highest", local_cache_hit=True),
                ), unresolved_ids=(), failures=(),
            )
        )
        result = RagtimeRetriever(FakeRagtimeClient(search), corpus).retrieve(
            "query", top_k=2, refresh_search=True, refresh_documents=True
        )
        self.assertIs(result.search_result, search)
        self.assertEqual([document.document_id for document in result.documents], ["doc-high", "doc-mid"])
        self.assertEqual([document.rank for document in result.documents], [1, 2])
        self.assertEqual([document.report_text for document in result.documents], ["highest", "middle"])

    def test_unresolved_documents_use_bounded_private_content_fallback(self) -> None:
        document_ids = [f"doc-{index}" for index in range(7)]
        client = FakeRagtimeClient(
            pipeline_result(document_ids),
            contents={document_id: private_document(document_id, document_id) for document_id in document_ids},
        )
        corpus = FakeCorpus(HydrationResult(documents=(), unresolved_ids=tuple(document_ids), failures=()))
        result = RagtimeRetriever(client, corpus).retrieve("query")
        self.assertEqual(len(client.content_calls), 5)
        self.assertEqual(result.unresolved_ids, ("doc-5", "doc-6"))

    def test_true_miss_fallback_stops_after_first_content_error(self) -> None:
        client = FakeRagtimeClient(
            pipeline_result(["doc-1", "doc-2", "doc-3"]),
            contents={"doc-1": private_document("doc-1", "one"), "doc-2": RagtimeApiError("HTTP 503"), "doc-3": private_document("doc-3", "three")},
        )
        corpus = FakeCorpus(HydrationResult(documents=(), unresolved_ids=("doc-1", "doc-2", "doc-3"), failures=()))
        result = RagtimeRetriever(client, corpus).retrieve("query", max_content_fallbacks=3)
        self.assertEqual([call["document_id"] for call in client.content_calls], ["doc-1", "doc-2"])
        self.assertEqual(result.unresolved_ids, ("doc-2", "doc-3"))

    def test_missing_report_text_ids_preserve_search_rank_across_sources(self) -> None:
        client = FakeRagtimeClient(
            pipeline_result(["doc-api", "doc-hf"]),
            contents={"doc-api": private_document("doc-api", "unknown language")},
        )
        corpus = FakeCorpus(
            HydrationResult(
                documents=(hf_document("doc-hf", "texto", split="spa"),),
                unresolved_ids=("doc-api",), failures=(), missing_translation_ids=("doc-hf",),
            )
        )
        result = RagtimeRetriever(client, corpus).retrieve("query")
        self.assertEqual(result.missing_translation_ids, ("doc-api", "doc-hf"))
        self.assertTrue(result.fully_hydrated)
        self.assertFalse(result.fully_report_ready)

    def test_hf_failure_with_unresolved_ids_raises_before_private_fallback(self) -> None:
        client = FakeRagtimeClient(
            pipeline_result(["doc-1", "doc-2"]),
            contents={"doc-1": private_document("doc-1", "one"), "doc-2": private_document("doc-2", "two")},
        )
        failure = HydrationFailure(split="eng", message="Dataset Viewer unavailable")
        corpus = FakeCorpus(HydrationResult(documents=(), unresolved_ids=("doc-1", "doc-2"), failures=(failure,)))
        with self.assertRaises(HuggingFaceHydrationError):
            RagtimeRetriever(client, corpus).retrieve("query")
        self.assertEqual(client.content_calls, [])

    def test_unbounded_fallback_requires_explicit_finite_error_fallback_bound(self) -> None:
        with self.assertRaises(ValueError):
            RagtimeRetriever(
                FakeRagtimeClient(pipeline_result(["doc-1"])),
                FakeCorpus(HydrationResult(documents=(), unresolved_ids=("doc-1",), failures=())),
            ).retrieve("query", fallback_on_hf_error=True, max_content_fallbacks=None)


if __name__ == "__main__":
    unittest.main()
