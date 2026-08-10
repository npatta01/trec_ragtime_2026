from __future__ import annotations

import os
import tempfile
import unittest
from collections import deque
from pathlib import Path
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
    LocalCorpusConfigurationError,
    LocalCorpusIndex,
    PipelineResult,
    RagtimeApiError,
    RagtimeClient,
    RagtimeRetriever,
    ResponseCache,
    RetrievalArtifactCache,
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
        self.base_url = "https://private.example"
        self.raise_on_search = False
        self.raise_on_content = False

    def search(self, query: str, *, pipeline: str, refresh: bool) -> PipelineResult:
        if self.raise_on_search:
            raise AssertionError("retrieval cache hit must not call search")
        self.search_calls.append({"query": query, "pipeline": pipeline, "refresh": refresh})
        return self.search_result

    def get_content(self, document_id: str, *, refresh: bool) -> Document:
        if self.raise_on_content:
            raise AssertionError("retrieval cache hit must not call content")
        self.content_calls.append({"document_id": document_id, "refresh": refresh})
        value = self.contents[document_id]
        if isinstance(value, Exception):
            raise value
        return value

    def close(self) -> None:
        self.closed = True


class FakeCorpus:
    backend_name = "fake"

    def __init__(
        self,
        result: HydrationResult,
        *,
        corpus_revision: str = "fake-revision",
        corpus_generation: int = 0,
    ):
        self.result = result
        self.calls: list[dict[str, Any]] = []
        self.closed = False
        self.raise_on_access = False
        self.corpus_generation = corpus_generation
        self._corpus_identity = {
            "backend": "fake",
            "schema_version": 1,
            "dataset": "trec-ragtime/ragtime2",
            "config": "default",
            "revision": corpus_revision,
            "document_count": 1,
            "shard_count": 1,
            "splits": ["eng", "spa", "rus", "zho"],
            "translation_config": "mt_docs",
            "translation_count": 0,
            "translation_shard_count": 0,
            "translation_splits": [],
        }
        self.raise_on_metadata = False

    @property
    def corpus_identity(self) -> dict[str, Any]:
        if self.raise_on_metadata:
            raise AssertionError("warm retrieval hit must not reopen corpus metadata")
        return self._corpus_identity

    def get_many(self, document_ids: Any, *, refresh: bool) -> HydrationResult:
        if self.raise_on_access:
            raise AssertionError("retrieval cache hit must not hydrate documents")
        self.calls.append({"document_ids": tuple(document_ids), "refresh": refresh})
        return self.result

    def enrich_documents(self, document_ids: Any, *, refresh: bool) -> HydrationResult:
        return self.get_many(document_ids, refresh=refresh)

    def close(self) -> None:
        self.closed = True


class LocalCorpusProbe(LocalCorpusIndex):
    """Minimal validated-index stand-in whose publication file can be replaced."""

    def __init__(self, path: Path, result: HydrationResult):
        self.path = path
        self.result = result
        self.calls = 0

    def enrich_documents(
        self, document_ids: Any, *, refresh: bool
    ) -> HydrationResult:
        self.calls += 1
        return self.result

    def close(self) -> None:
        return None


class HuggingFaceCorpusTests(unittest.TestCase):
    def test_backend_is_deprecated(self) -> None:
        with self.assertWarnsRegex(DeprecationWarning, "LocalCorpusIndex"):
            corpus = HuggingFaceCorpus(use_cache=False)
        corpus.close()

    def test_injected_store_metadata_is_rejected_without_echoing_values(self) -> None:
        cache, store = make_cache()
        store.bucket_id = "owner/private?token=do-not-echo"
        store.prefix = "https://private.example/body=do-not-echo"
        store.local_dir = "owner/credential=do-not-echo"

        with self.assertWarns(DeprecationWarning):
            corpus = HuggingFaceCorpus(
                splits=("eng",),
                cache=cache,
                session=RecordingGetSession(),
            )
        with self.assertRaises(HuggingFaceConfigurationError) as raised:
            _ = corpus.cache_configuration
        self.assertNotIn("do-not-echo", str(raised.exception))
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

    def test_legacy_settings_are_rejected_for_cache_opt_out_and_injected_cache(self) -> None:
        for variable in ("RAGTIME_CACHE_PATH", "RAGTIME_CACHE_DIR"):
            with self.subTest(variable=variable), patch.dict(
                os.environ,
                {
                    "RAGTIME_CACHE_PATH": "",
                    "RAGTIME_CACHE_DIR": "",
                    variable: "/tmp/legacy-cache",
                },
                clear=False,
            ):
                opt_out_session = RecordingGetSession()
                with self.assertRaises(HuggingFaceConfigurationError):
                    HuggingFaceCorpus(
                        splits=("eng",),
                        use_cache=False,
                        session=opt_out_session,
                    )
                self.assertEqual(opt_out_session.calls, [])

                cache, _ = make_cache()
                injected_session = RecordingGetSession()
                with self.assertRaises(HuggingFaceConfigurationError):
                    HuggingFaceCorpus(
                        splits=("eng",),
                        cache=cache,
                        session=injected_session,
                    )
                self.assertEqual(injected_session.calls, [])

    def test_cache_opt_out_does_not_require_artifact_bucket_configuration(self) -> None:
        with patch.dict(
            os.environ,
            {
                "RAGTIME_CACHE_PATH": "",
                "RAGTIME_CACHE_DIR": "",
                "RAGTIME_HF_BUCKET_ID": "",
            },
            clear=False,
        ), patch("trec_ragtime.huggingface.ArtifactStore.from_env") as from_env:
            HuggingFaceCorpus(
                splits=("eng",),
                use_cache=False,
                session=RecordingGetSession(),
            )
        from_env.assert_not_called()

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
    def _cached_retriever(
        self,
        store: MemoryArtifactStore,
        *,
        text: str = "cached text",
        document_result: HydrationResult | None = None,
        corpus_revision: str = "fake-revision",
        corpus_generation: int = 0,
    ) -> tuple[RagtimeRetriever, FakeRagtimeClient, FakeCorpus]:
        client = FakeRagtimeClient(pipeline_result(["doc-1"]))
        corpus = FakeCorpus(
            document_result
            if document_result is not None
            else HydrationResult(
                documents=(hf_document("doc-1", text),),
                unresolved_ids=(),
                failures=(),
            ),
            corpus_revision=corpus_revision,
            corpus_generation=corpus_generation,
        )
        return RagtimeRetriever(
            client,
            corpus,
            retrieval_cache=RetrievalArtifactCache(store, clock=lambda: 100.0),
        ), client, corpus

    def test_l1_hit_does_not_call_search_or_corpus(self) -> None:
        store = MemoryArtifactStore()
        first, _, _ = self._cached_retriever(store, text="first")
        first_result = first.retrieve("query")
        self.assertFalse(first_result.retrieval_artifact.cache_hit)

        second, client, corpus = self._cached_retriever(store, text="wrong")
        client.raise_on_search = True
        corpus.raise_on_access = True
        result = second.retrieve("query")

        self.assertTrue(result.retrieval_artifact.cache_hit)
        self.assertEqual(result.documents[0].text, "first")
        self.assertEqual(client.search_calls, [])
        self.assertEqual(corpus.calls, [])

    def test_l2_hit_does_not_call_search_or_corpus_and_populates_l1(self) -> None:
        store = MemoryArtifactStore()
        first, _, _ = self._cached_retriever(store, text="from l2")
        first_result = first.retrieve("query")
        object_key = first_result.retrieval_artifact.object_key
        store.local.clear()
        self.assertIn(object_key, store.remote)

        second, client, corpus = self._cached_retriever(store, text="wrong")
        client.raise_on_search = True
        corpus.raise_on_access = True
        result = second.retrieve("query")

        self.assertTrue(result.retrieval_artifact.cache_hit)
        self.assertEqual(result.documents[0].text, "from l2")
        self.assertIn(object_key, store.local)
        self.assertEqual(client.search_calls, [])
        self.assertEqual(corpus.calls, [])

    def test_miss_computes_once_validates_writes_and_returns_without_reread(self) -> None:
        store = MemoryArtifactStore()
        retriever, client, corpus = self._cached_retriever(store)

        result = retriever.retrieve("query")

        self.assertFalse(result.retrieval_artifact.cache_hit)
        self.assertEqual(len(store.read_calls), 1)
        self.assertEqual(len(store.write_calls), 1)
        self.assertEqual(len(client.search_calls), 1)
        self.assertEqual(len(corpus.calls), 1)

    def test_unresolved_or_hydration_failure_result_is_never_written(self) -> None:
        for hydration in (
            HydrationResult(
                documents=(), unresolved_ids=("doc-1",), failures=()
            ),
            HydrationResult(
                documents=(),
                unresolved_ids=("doc-1",),
                failures=(HydrationFailure(split="eng", message="temporary"),),
            ),
        ):
            with self.subTest(hydration=hydration):
                store = MemoryArtifactStore()
                retriever, _, _ = self._cached_retriever(store, document_result=hydration)
                result = retriever.retrieve(
                    "query",
                    fallback_on_hf_error=bool(hydration.failures),
                    max_content_fallbacks=0,
                )
                self.assertFalse(result.fully_hydrated)
                self.assertEqual(store.write_calls, [])

    def test_missing_translation_with_source_text_is_cacheable(self) -> None:
        store = MemoryArtifactStore()
        retriever, _, _ = self._cached_retriever(
            store,
            document_result=HydrationResult(
                documents=(hf_document("doc-1", "texto", split="spa"),),
                unresolved_ids=(),
                failures=(),
                missing_translation_ids=("doc-1",),
            ),
        )
        first = retriever.retrieve("query")
        self.assertTrue(first.fully_hydrated)
        self.assertFalse(first.fully_report_ready)
        self.assertEqual(len(store.write_calls), 1)

        second, client, corpus = self._cached_retriever(store, text="wrong")
        client.raise_on_search = True
        corpus.raise_on_access = True
        cached = second.retrieve("query")
        self.assertTrue(cached.retrieval_artifact.cache_hit)
        self.assertEqual(cached.documents[0].text, "texto")

    def test_refresh_bypasses_read_and_overwrites_only_after_complete_success(self) -> None:
        store = MemoryArtifactStore()
        first, _, _ = self._cached_retriever(store, text="old")
        first.retrieve("query")

        refreshed, client, corpus = self._cached_retriever(store, text="new")
        result = refreshed.retrieve("query", refresh=True)

        self.assertFalse(result.retrieval_artifact.cache_hit)
        self.assertEqual(len(store.read_calls), 1)
        self.assertEqual(len(store.write_calls), 2)
        self.assertTrue(client.search_calls[-1]["refresh"])
        self.assertTrue(corpus.calls[-1]["refresh"])

        check, check_client, check_corpus = self._cached_retriever(store, text="wrong")
        check_client.raise_on_search = True
        check_corpus.raise_on_access = True
        self.assertEqual(check.retrieve("query").documents[0].text, "new")

    def test_failed_refresh_preserves_existing_retrieval_object(self) -> None:
        store = MemoryArtifactStore()
        first, _, _ = self._cached_retriever(store, text="old")
        first.retrieve("query")
        failed, _, _ = self._cached_retriever(
            store,
            text="not persisted",
            document_result=HydrationResult(
                documents=(), unresolved_ids=("doc-1",), failures=()
            ),
        )
        result = failed.retrieve("query", refresh=True, max_content_fallbacks=0)
        self.assertFalse(result.fully_hydrated)
        self.assertEqual(len(store.write_calls), 1)

        check, check_client, check_corpus = self._cached_retriever(store, text="wrong")
        check_client.raise_on_search = True
        check_corpus.raise_on_access = True
        cached = check.retrieve("query")
        self.assertTrue(cached.retrieval_artifact.cache_hit)
        self.assertEqual(cached.documents[0].text, "old")

    def test_query_context_changes_identity_without_changing_query_bytes(self) -> None:
        store = MemoryArtifactStore()
        first, first_client, _ = self._cached_retriever(store, text="first")
        first_result = first.retrieve(
            "query", query_context={"kind": "control_a"}
        )
        second, second_client, _ = self._cached_retriever(store, text="second")
        second_result = second.retrieve(
            "query", query_context={"kind": "baseline_b"}
        )

        self.assertNotEqual(
            first_result.retrieval_artifact.object_key,
            second_result.retrieval_artifact.object_key,
        )
        self.assertEqual(first_client.search_calls[0]["query"], "query")
        self.assertEqual(second_client.search_calls[0]["query"], "query")
        self.assertEqual(len(store.write_calls), 2)

    def test_retrieval_artifact_store_failures_propagate(self) -> None:
        store = MemoryArtifactStore()
        store.read_error = OSError("retrieval artifact read failed")
        retriever, client, corpus = self._cached_retriever(store)
        with self.assertRaisesRegex(OSError, "retrieval artifact read failed"):
            retriever.retrieve("query")
        self.assertEqual(client.search_calls, [])
        self.assertEqual(corpus.calls, [])

        store = MemoryArtifactStore()
        store.write_error = OSError("retrieval artifact write failed")
        retriever, client, corpus = self._cached_retriever(store)
        with self.assertRaisesRegex(OSError, "retrieval artifact write failed"):
            retriever.retrieve("query")
        self.assertEqual(len(client.search_calls), 1)
        self.assertEqual(len(corpus.calls), 1)
        self.assertEqual(store.write_calls, [])

    def test_from_env_shares_clients_configured_artifact_store(self) -> None:
        store = MemoryArtifactStore()
        client = FakeRagtimeClient(pipeline_result([]))
        client.artifact_store = store
        corpus = FakeCorpus(
            HydrationResult(documents=(), unresolved_ids=(), failures=())
        )
        with tempfile.TemporaryDirectory() as temporary:
            index_path = Path(temporary) / "corpus.duckdb"
            index_path.touch()
            with patch(
                "trec_ragtime.retriever.RagtimeClient.from_env",
                return_value=client,
            ), patch(
                "trec_ragtime.retriever.LocalCorpusIndex",
                return_value=corpus,
            ), patch.object(
                RagtimeRetriever,
                "_capture_retrieval_corpus_identity",
                return_value=corpus.corpus_identity,
            ), patch.object(
                RagtimeRetriever,
                "_capture_corpus_publication_token",
                return_value=("custom", 0),
            ):
                retriever = RagtimeRetriever.from_env(local_index_path=index_path)
        self.assertIsNotNone(retriever.retrieval_cache)
        assert retriever.retrieval_cache is not None
        self.assertIs(retriever.retrieval_cache.store, store)

    def test_direct_constructor_without_cache_preserves_legacy_retrieval(self) -> None:
        client = FakeRagtimeClient(pipeline_result(["doc-1"]))
        del client.base_url
        corpus = FakeCorpus(
            HydrationResult(
                documents=(hf_document("doc-1", "legacy text"),),
                unresolved_ids=(),
                failures=(),
            )
        )
        del corpus._corpus_identity
        retriever = RagtimeRetriever(client, corpus)

        result = retriever.retrieve("query")

        self.assertEqual(result.documents[0].text, "legacy text")
        self.assertEqual(len(client.search_calls), 1)
        self.assertEqual(len(corpus.calls), 1)

    def test_warm_hits_reuse_captured_corpus_identity_without_metadata_access(self) -> None:
        store = MemoryArtifactStore()
        retriever, client, corpus = self._cached_retriever(store, text="captured")
        first = retriever.retrieve("query")
        corpus.raise_on_metadata = True
        client.raise_on_search = True
        corpus.raise_on_access = True

        second = retriever.retrieve("query")

        self.assertTrue(second.retrieval_artifact.cache_hit)
        self.assertEqual(second.documents[0].text, "captured")
        self.assertEqual(first.retrieval_artifact.object_key, second.retrieval_artifact.object_key)

    def test_corpus_generation_change_fails_before_hit_and_requires_new_retriever(self) -> None:
        store = MemoryArtifactStore()
        retriever, client, corpus = self._cached_retriever(
            store,
            text="old corpus",
            corpus_revision="old-revision",
            corpus_generation=0,
        )
        first = retriever.retrieve("query")
        corpus._corpus_identity["revision"] = "new-revision"
        corpus.corpus_generation = 1
        client.raise_on_search = True

        with self.assertRaisesRegex(
            LocalCorpusConfigurationError, "create a new RagtimeRetriever"
        ):
            retriever.retrieve("query")
        self.assertEqual(len(store.read_calls), 1)
        self.assertEqual(len(store.write_calls), 1)

        with self.assertRaisesRegex(
            LocalCorpusConfigurationError, "create a new RagtimeRetriever"
        ):
            retriever.retrieve("query", refresh=True)
        self.assertEqual(len(store.read_calls), 1)
        self.assertEqual(len(store.write_calls), 1)

        replacement, replacement_client, replacement_corpus = self._cached_retriever(
            store,
            text="new corpus",
            corpus_revision="new-revision",
            corpus_generation=1,
        )
        result = replacement.retrieve("query")
        self.assertFalse(result.retrieval_artifact.cache_hit)
        self.assertNotEqual(
            first.retrieval_artifact.object_key,
            result.retrieval_artifact.object_key,
        )
        self.assertEqual(result.documents[0].text, "new corpus")
        self.assertEqual(len(replacement_client.search_calls), 1)
        self.assertEqual(len(replacement_corpus.calls), 1)
        self.assertEqual(len(store.write_calls), 2)

    def test_replaced_local_corpus_file_fails_before_cache_hit(self) -> None:
        store = MemoryArtifactStore()
        old_identity = FakeCorpus(
            HydrationResult(documents=(), unresolved_ids=(), failures=())
        ).corpus_identity
        new_identity = {**old_identity, "revision": "new-revision"}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "corpus.duckdb"
            path.write_bytes(b"old publication")
            old_corpus = LocalCorpusProbe(
                path,
                HydrationResult(
                    documents=(hf_document("doc-1", "old corpus"),),
                    unresolved_ids=(),
                    failures=(),
                ),
            )
            old_client = FakeRagtimeClient(pipeline_result(["doc-1"]))
            with patch.object(
                RagtimeRetriever,
                "_capture_retrieval_corpus_identity",
                return_value=old_identity,
            ):
                old = RagtimeRetriever(
                    old_client,
                    old_corpus,
                    retrieval_cache=RetrievalArtifactCache(store, clock=lambda: 100.0),
                )
            first = old.retrieve("query")

            replacement_path = Path(temporary) / "replacement.duckdb"
            replacement_path.write_bytes(b"new publication with a different size")
            os.replace(replacement_path, path)
            old_client.raise_on_search = True
            with self.assertRaisesRegex(
                LocalCorpusConfigurationError, "create a new RagtimeRetriever"
            ):
                old.retrieve("query")
            with self.assertRaisesRegex(
                LocalCorpusConfigurationError, "create a new RagtimeRetriever"
            ):
                old.retrieve("query", refresh=True)
            self.assertEqual(len(store.read_calls), 1)
            self.assertEqual(len(store.write_calls), 1)

            new_corpus = LocalCorpusProbe(
                path,
                HydrationResult(
                    documents=(hf_document("doc-1", "new corpus"),),
                    unresolved_ids=(),
                    failures=(),
                ),
            )
            new_client = FakeRagtimeClient(pipeline_result(["doc-1"]))
            with patch.object(
                RagtimeRetriever,
                "_capture_retrieval_corpus_identity",
                return_value=new_identity,
            ):
                new = RagtimeRetriever(
                    new_client,
                    new_corpus,
                    retrieval_cache=RetrievalArtifactCache(store, clock=lambda: 100.0),
                )
            result = new.retrieve("query")

        self.assertFalse(result.retrieval_artifact.cache_hit)
        self.assertEqual(result.documents[0].text, "new corpus")
        self.assertNotEqual(
            first.retrieval_artifact.object_key,
            result.retrieval_artifact.object_key,
        )
        self.assertEqual(len(store.write_calls), 2)

    def test_custom_generation_change_during_cache_get_fails_closed_after_read(self) -> None:
        store = MemoryArtifactStore()
        retriever, client, corpus = self._cached_retriever(store, text="old corpus")
        retriever.retrieve("query")
        cache = retriever.retrieval_cache
        assert cache is not None
        original_get = cache.get

        def raced_get(identity: Any) -> Any:
            cached = original_get(identity)
            corpus.corpus_generation = 1
            return cached

        client.raise_on_search = True
        corpus.raise_on_access = True
        with patch.object(cache, "get", side_effect=raced_get):
            with self.assertRaisesRegex(
                LocalCorpusConfigurationError, "create a new RagtimeRetriever"
            ):
                retriever.retrieve("query")
        self.assertEqual(len(store.read_calls), 2)

        with self.assertRaisesRegex(
            LocalCorpusConfigurationError, "create a new RagtimeRetriever"
        ):
            retriever.retrieve("query")
        self.assertEqual(len(store.read_calls), 2)

    def test_custom_generation_change_during_compute_fails_before_write(self) -> None:
        store = MemoryArtifactStore()
        retriever, _, corpus = self._cached_retriever(store, text="old corpus")
        original_enrich = corpus.enrich_documents

        def raced_enrich(document_ids: Any, *, refresh: bool) -> HydrationResult:
            hydration = original_enrich(document_ids, refresh=refresh)
            corpus.corpus_generation = 1
            return hydration

        with patch.object(corpus, "enrich_documents", side_effect=raced_enrich):
            with self.assertRaisesRegex(
                LocalCorpusConfigurationError, "create a new RagtimeRetriever"
            ):
                retriever.retrieve("query")
        self.assertEqual(store.write_calls, [])

    def test_custom_generation_change_during_cache_put_fails_after_old_result_write(self) -> None:
        store = MemoryArtifactStore()
        retriever, _, corpus = self._cached_retriever(store, text="old corpus")
        identity = retriever.build_retrieval_identity("query")
        cache = retriever.retrieval_cache
        assert cache is not None
        original_put = cache.put

        def raced_put(request: Any, result: Any) -> Any:
            corpus.raise_on_metadata = True
            corpus.raise_on_access = True
            stored = original_put(request, result)
            corpus.corpus_generation = 1
            return stored

        with patch.object(cache, "put", side_effect=raced_put):
            with self.assertRaisesRegex(
                LocalCorpusConfigurationError, "create a new RagtimeRetriever"
            ):
                retriever.retrieve("query")
        self.assertEqual(len(store.write_calls), 1)

        stored = cache.get(identity)
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertTrue(stored.retrieval_artifact.cache_hit)
        self.assertEqual(stored.documents[0].text, "old corpus")
        with self.assertRaisesRegex(
            LocalCorpusConfigurationError, "create a new RagtimeRetriever"
        ):
            retriever.retrieve("query")

    def test_replaced_local_corpus_file_during_cache_get_fails_closed_after_read(self) -> None:
        store = MemoryArtifactStore()
        old_identity = FakeCorpus(
            HydrationResult(documents=(), unresolved_ids=(), failures=())
        ).corpus_identity
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "corpus.duckdb"
            path.write_bytes(b"old publication")
            corpus = LocalCorpusProbe(
                path,
                HydrationResult(
                    documents=(hf_document("doc-1", "old corpus"),),
                    unresolved_ids=(),
                    failures=(),
                ),
            )
            client = FakeRagtimeClient(pipeline_result(["doc-1"]))
            with patch.object(
                RagtimeRetriever,
                "_capture_retrieval_corpus_identity",
                return_value=old_identity,
            ):
                retriever = RagtimeRetriever(
                    client,
                    corpus,
                    retrieval_cache=RetrievalArtifactCache(store, clock=lambda: 100.0),
                )
            retriever.retrieve("query")
            replacement = Path(temporary) / "replacement.duckdb"
            replacement.write_bytes(b"new publication")
            cache = retriever.retrieval_cache
            assert cache is not None
            original_get = cache.get

            def raced_get(identity: Any) -> Any:
                cached = original_get(identity)
                os.replace(replacement, path)
                return cached

            client.raise_on_search = True
            with patch.object(cache, "get", side_effect=raced_get):
                with self.assertRaisesRegex(
                    LocalCorpusConfigurationError, "create a new RagtimeRetriever"
                ):
                    retriever.retrieve("query")
            self.assertEqual(len(store.read_calls), 2)

            with self.assertRaisesRegex(
                LocalCorpusConfigurationError, "create a new RagtimeRetriever"
            ):
                retriever.retrieve("query")
            self.assertEqual(len(store.read_calls), 2)

    def test_replaced_local_corpus_file_during_cache_put_fails_after_old_result_write(self) -> None:
        store = MemoryArtifactStore()
        old_identity = FakeCorpus(
            HydrationResult(documents=(), unresolved_ids=(), failures=())
        ).corpus_identity
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "corpus.duckdb"
            path.write_bytes(b"old publication")
            replacement = Path(temporary) / "replacement.duckdb"
            replacement.write_bytes(b"new publication")
            corpus = LocalCorpusProbe(
                path,
                HydrationResult(
                    documents=(hf_document("doc-1", "old corpus"),),
                    unresolved_ids=(),
                    failures=(),
                ),
            )
            client = FakeRagtimeClient(pipeline_result(["doc-1"]))
            with patch.object(
                RagtimeRetriever,
                "_capture_retrieval_corpus_identity",
                return_value=old_identity,
            ):
                retriever = RagtimeRetriever(
                    client,
                    corpus,
                    retrieval_cache=RetrievalArtifactCache(store, clock=lambda: 100.0),
                )
            identity = retriever.build_retrieval_identity("query")
            cache = retriever.retrieval_cache
            assert cache is not None
            original_put = cache.put

            def raced_put(request: Any, result: Any) -> Any:
                stored = original_put(request, result)
                os.replace(replacement, path)
                return stored

            with patch.object(cache, "put", side_effect=raced_put):
                with self.assertRaisesRegex(
                    LocalCorpusConfigurationError, "create a new RagtimeRetriever"
                ):
                    retriever.retrieve("query")
            self.assertEqual(len(store.write_calls), 1)

            stored = cache.get(identity)
            self.assertIsNotNone(stored)
            assert stored is not None
            self.assertTrue(stored.retrieval_artifact.cache_hit)
            self.assertEqual(stored.documents[0].text, "old corpus")
            with self.assertRaisesRegex(
                LocalCorpusConfigurationError, "create a new RagtimeRetriever"
            ):
                retriever.retrieve("query")

    def test_refresh_refreshes_private_content_fallback(self) -> None:
        store = MemoryArtifactStore()
        client = FakeRagtimeClient(
            pipeline_result(["doc-1"]),
            contents={"doc-1": private_document("doc-1", "fresh fallback")},
        )
        corpus = FakeCorpus(
            HydrationResult(documents=(), unresolved_ids=("doc-1",), failures=())
        )
        retriever = RagtimeRetriever(
            client,
            corpus,
            retrieval_cache=RetrievalArtifactCache(store, clock=lambda: 100.0),
        )

        result = retriever.retrieve("query", refresh=True, max_content_fallbacks=1)

        self.assertEqual(result.documents[0].text, "fresh fallback")
        self.assertEqual(client.content_calls[0]["refresh"], True)

        old_client, old_corpus = (
            FakeRagtimeClient(pipeline_result(["doc-1"])),
            FakeCorpus(
                HydrationResult(
                    documents=(hf_document("doc-1", "should not replace"),),
                    unresolved_ids=(),
                    failures=(),
                )
            ),
        )
        old = RagtimeRetriever(
            old_client,
            old_corpus,
            retrieval_cache=RetrievalArtifactCache(store, clock=lambda: 100.0),
        )
        old.retrieve("query", refresh=True)
        failed_client = FakeRagtimeClient(
            pipeline_result(["doc-1"]),
            contents={"doc-1": RagtimeApiError("fallback unavailable")},
        )
        failed_corpus = FakeCorpus(
            HydrationResult(documents=(), unresolved_ids=("doc-1",), failures=())
        )
        failed = RagtimeRetriever(
            failed_client,
            failed_corpus,
            retrieval_cache=RetrievalArtifactCache(store, clock=lambda: 100.0),
        )
        incomplete = failed.retrieve("query", refresh=True, max_content_fallbacks=1)
        self.assertFalse(incomplete.fully_hydrated)
        self.assertEqual(failed_client.content_calls[0]["refresh"], True)

        check, check_client, check_corpus = self._cached_retriever(store, text="wrong")
        check_client.raise_on_search = True
        check_corpus.raise_on_access = True
        cached = check.retrieve("query")
        self.assertTrue(cached.retrieval_artifact.cache_hit)
        self.assertEqual(cached.documents[0].text, "should not replace")

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
