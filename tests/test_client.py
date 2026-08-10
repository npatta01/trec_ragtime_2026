from __future__ import annotations

import json
import os
import unittest
import warnings
from collections import deque
from typing import Any, Callable
from unittest.mock import Mock, patch

import requests
from requests_ratelimiter import LimiterMixin

from trec_ragtime import (
    RagtimeApiError,
    RagtimeClient,
    RagtimeConfigurationError,
    RagtimeProtocolError,
)
from trec_ragtime.artifact_store import ArtifactValidationError
from trec_ragtime.cache import ResponseCache


class MemoryArtifactStore:
    """In-memory L1/L2 store with the reviewed ArtifactStore contract."""

    def __init__(self) -> None:
        self.local: dict[str, bytes] = {}
        self.remote: dict[str, bytes] = {}
        self.read_calls: list[str] = []
        self.write_calls: list[tuple[str, bytes]] = []
        self.repaired_local_candidates = 0
        self.read_error: BaseException | None = None
        self.read_result = True
        self.write_error: BaseException | None = None
        self.write_result = True

    def read_validated(
        self, object_key: str, decoder: Callable[[bytes], Any]
    ) -> Any | None:
        self.read_calls.append(object_key)
        if self.read_error is not None:
            raise self.read_error
        if not self.read_result:
            return None
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

    def write(self, object_key: str, payload: bytes) -> bool:
        if self.write_error is not None:
            raise self.write_error
        self.write_calls.append((object_key, payload))
        if not self.write_result:
            return False
        self.local[object_key] = payload
        self.remote[object_key] = payload
        return True


class FakeResponse:
    def __init__(self, payload: Any, *, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.headers: dict[str, str] = {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            response = requests.Response()
            response.status_code = self.status_code
            raise requests.HTTPError(response=response)

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class RecordingSession:
    def __init__(self, *responses: FakeResponse):
        self.responses = deque(responses)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((url, kwargs))
        if not self.responses:
            raise AssertionError("unexpected upstream request")
        return self.responses.popleft()

    def close(self) -> None:
        self.closed = True


class RaisingSession(RecordingSession):
    def __init__(self, error: BaseException):
        super().__init__()
        self.error = error

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((url, kwargs))
        raise self.error


def pipeline_payload(
    scores: dict[str, float] | None = None,
    *,
    query: str = "query",
    pipeline: str = "ragtime2",
) -> dict[str, Any]:
    return {
        "pipeline": pipeline,
        "processed": True,
        "query": query,
        "scores": {"doc-b": 0.2, "doc-c": 0.9, "doc-a": 0.9}
        if scores is None
        else scores,
        "timestamp": 123.0,
    }


def content_payload(document_id: str = "doc-1", text: str = "document text") -> dict[str, Any]:
    return {
        "cached": False,
        "collection": "ragtime2",
        "id": document_id,
        "text": text,
    }


def make_cache() -> tuple[ResponseCache, MemoryArtifactStore]:
    store = MemoryArtifactStore()
    return ResponseCache(store), store


class RagtimeClientTests(unittest.TestCase):
    def make_client(
        self,
        session: RecordingSession,
        *,
        cache: ResponseCache | None = None,
        artifact_store: MemoryArtifactStore | None = None,
        **kwargs: Any,
    ) -> RagtimeClient:
        return RagtimeClient(
            "https://api.example",
            "secret",
            session=session,
            cache=cache,
            artifact_store=artifact_store,
            **kwargs,
        )

    def test_default_session_proactively_rate_limits_private_requests(self) -> None:
        client = RagtimeClient("https://api.example", "secret", use_cache=False)
        try:
            self.assertIsInstance(client._session, LimiterMixin)
            self.assertEqual(client.requests_per_second, 1.0)
        finally:
            client.close()

    def test_each_retry_acquires_a_rate_limit_token(self) -> None:
        class SequenceAdapter(requests.adapters.BaseAdapter):
            def __init__(self) -> None:
                self.statuses = deque((429, 200))

            def send(self, request: requests.PreparedRequest, **kwargs: Any) -> requests.Response:
                response = requests.Response()
                response.status_code = self.statuses.popleft()
                response.request = request
                response.headers["Content-Type"] = "application/json"
                response._content = json.dumps(pipeline_payload()).encode()
                return response

            def close(self) -> None:
                pass

        client = RagtimeClient(
            "https://api.example", "secret", use_cache=False, retries=1
        )
        adapter = SequenceAdapter()
        client._session.mount("https://", adapter)
        acquire = client._session.limiter.try_acquire
        client._session.limiter.try_acquire = Mock(wraps=acquire)
        try:
            with patch("trec_ragtime.client.time.sleep"):
                result = client.search("query")
            self.assertEqual(result.hits[0].document_id, "doc-a")
            self.assertEqual(client._session.limiter.try_acquire.call_count, 2)
        finally:
            client.close()

    def test_search_posts_exact_request_and_canonical_bearer_header(self) -> None:
        session = RecordingSession(FakeResponse(pipeline_payload()))
        client = self.make_client(session, use_cache=False)
        result = client.search("query")

        self.assertEqual(session.calls[0][0], "https://api.example/pipeline")
        self.assertEqual(
            session.calls[0][1]["json"], {"pipeline": "ragtime2", "query": "query"}
        )
        self.assertEqual(session.calls[0][1]["headers"]["Authorization"], "Bearer secret")
        self.assertFalse(result.local_cache_hit)

    def test_search_sorts_scores_with_document_id_tiebreak(self) -> None:
        session = RecordingSession(FakeResponse(pipeline_payload()))
        result = self.make_client(session, use_cache=False).search("query")
        self.assertEqual(
            [hit.document_id for hit in result.hits], ["doc-a", "doc-c", "doc-b"]
        )
        self.assertEqual([hit.rank for hit in result.hits], [1, 2, 3])

    def test_cache_is_reused_across_client_instances_and_tokens(self) -> None:
        cache, store = make_cache()
        first = RagtimeClient(
            "https://api.example", "first-token", cache=cache,
            session=RecordingSession(FakeResponse(pipeline_payload())),
        )
        first_result = first.search("query")
        second_session = RecordingSession()
        second = RagtimeClient(
            "https://api.example", "second-token", cache=cache, session=second_session
        )
        second_result = second.search("query")

        self.assertFalse(first_result.local_cache_hit)
        self.assertTrue(second_result.local_cache_hit)
        self.assertEqual(second_result.hits, first_result.hits)
        self.assertEqual(second_session.calls, [])
        self.assertIs(store, cache.store)

    def test_refresh_replaces_a_cached_response(self) -> None:
        cache, _ = make_cache()
        self.make_client(
            RecordingSession(FakeResponse(pipeline_payload({"old": 0.1}))), cache=cache
        ).search("query")
        refreshed = self.make_client(
            RecordingSession(FakeResponse(pipeline_payload({"new": 0.8}))), cache=cache
        ).search("query", refresh=True)
        cached = self.make_client(RecordingSession(), cache=cache).search("query")

        self.assertFalse(refreshed.local_cache_hit)
        self.assertEqual(refreshed.hits[0].document_id, "new")
        self.assertTrue(cached.local_cache_hit)
        self.assertEqual(cached.hits[0].document_id, "new")

    def test_failed_refresh_preserves_previous_cached_response_and_does_not_write(self) -> None:
        cache, store = make_cache()
        self.make_client(
            RecordingSession(FakeResponse(pipeline_payload({"previous": 0.5}))),
            cache=cache,
        ).search("query")
        writes_before = len(store.write_calls)
        refreshing = self.make_client(
            RecordingSession(FakeResponse({}, status_code=503)), cache=cache
        )
        with self.assertRaises(RagtimeApiError):
            refreshing.search("query", refresh=True)
        cached = self.make_client(RecordingSession(), cache=cache).search("query")
        self.assertTrue(cached.local_cache_hit)
        self.assertEqual(cached.hits[0].document_id, "previous")
        self.assertEqual(len(store.write_calls), writes_before)

    def test_content_uses_ragtime2_collection_and_is_cached(self) -> None:
        cache, _ = make_cache()
        session = RecordingSession(FakeResponse(content_payload()))
        first = self.make_client(session, cache=cache).get_content("doc-1")
        second = self.make_client(RecordingSession(), cache=cache).get_content("doc-1")
        self.assertEqual(session.calls[0][1]["json"], {"collection": "ragtime2", "id": "doc-1"})
        self.assertEqual(first.text, "document text")
        self.assertTrue(second.local_cache_hit)

    def test_transport_status_json_and_protocol_failures_never_write(self) -> None:
        scenarios = (
            (requests.ConnectionError("offline"), RagtimeApiError),
            (FakeResponse({}, status_code=401), RagtimeApiError),
            (FakeResponse(ValueError("bad json")), RagtimeProtocolError),
            (FakeResponse({"processed": True}), RagtimeProtocolError),
        )
        for response, error_type in scenarios:
            with self.subTest(error_type=error_type):
                cache, store = make_cache()
                session = (
                    RecordingSession(response)
                    if isinstance(response, FakeResponse)
                    else RaisingSession(response)
                )
                with self.assertRaises(error_type):
                    self.make_client(session, cache=cache, retries=0).search("query")
                self.assertEqual(store.write_calls, [])

    def test_write_failure_keeps_fully_validated_live_result(self) -> None:
        cache, store = make_cache()
        store.write_result = False
        result = self.make_client(
            RecordingSession(FakeResponse(pipeline_payload())), cache=cache
        ).search("query")
        self.assertFalse(result.local_cache_hit)
        self.assertIsNone(result.cache_stored_at)
        self.assertEqual(len(store.write_calls), 1)

    def test_read_failure_is_a_miss_and_live_request_still_runs(self) -> None:
        cache, store = make_cache()
        store.read_result = False
        session = RecordingSession(FakeResponse(pipeline_payload()))
        result = self.make_client(session, cache=cache).search("query")
        self.assertFalse(result.local_cache_hit)
        self.assertEqual(len(session.calls), 1)

    def test_thrown_cache_read_failure_is_a_miss_and_live_request_still_runs(self) -> None:
        cache, store = make_cache()
        store.read_error = OSError("cache unavailable")
        session = RecordingSession(FakeResponse(pipeline_payload()))

        result = self.make_client(session, cache=cache).search("query")

        self.assertFalse(result.local_cache_hit)
        self.assertEqual(len(session.calls), 1)

    def test_thrown_cache_write_failure_keeps_fully_validated_live_result(self) -> None:
        cache, store = make_cache()
        store.write_error = OSError("disk full")

        result = self.make_client(
            RecordingSession(FakeResponse(pipeline_payload())), cache=cache
        ).search("query")

        self.assertFalse(result.local_cache_hit)
        self.assertIsNone(result.cache_stored_at)

    def test_malformed_l1_repairs_once_before_one_upstream_request(self) -> None:
        cache, store = make_cache()
        request = {"pipeline": "ragtime2", "query": "query"}
        cache.put(
            base_url="https://api.example", operation="pipeline", request=request,
            response=pipeline_payload(),
        )
        key = cache.object_key(base_url="https://api.example", operation="pipeline", request=request)
        store.local[key] = b"malformed"
        repaired = self.make_client(RecordingSession(), cache=cache).search("query")
        self.assertTrue(repaired.local_cache_hit)
        self.assertEqual(store.repaired_local_candidates, 1)
        self.assertEqual(len(store.write_calls), 1)

    def test_corrupt_l2_is_fatal_and_never_reaches_participant(self) -> None:
        cache, store = make_cache()
        request = {"pipeline": "ragtime2", "query": "query"}
        key = cache.object_key(base_url="https://api.example", operation="pipeline", request=request)
        store.remote[key] = b"corrupt-l2"
        with self.assertRaises(ArtifactValidationError):
            self.make_client(RecordingSession(), cache=cache).search("query")
        self.assertEqual(store.write_calls, [])

    def test_malformed_live_payload_does_not_write(self) -> None:
        cache, store = make_cache()
        with self.assertRaises(RagtimeProtocolError):
            self.make_client(
                RecordingSession(FakeResponse({"processed": True})), cache=cache
            ).search("query")
        self.assertEqual(store.write_calls, [])

    def test_artifact_store_is_constructed_only_when_caching_is_enabled(self) -> None:
        with patch.dict(
            os.environ,
            {
                "RAGTIME_BASE_URL": "https://dummy.example",
                "RAGTIME_BEARER_TOKEN": "dummy-token",
                "RAGTIME_CACHE_PATH": "",
                "RAGTIME_CACHE_DIR": "",
            },
            clear=True,
        ), patch("trec_ragtime.client.ArtifactStore.from_env") as from_env:
            RagtimeClient.from_env(use_cache=False, session=RecordingSession())
        from_env.assert_not_called()

    def test_from_env_rejects_nonempty_legacy_cache_settings(self) -> None:
        with patch.dict(
            os.environ,
            {
                "RAGTIME_BASE_URL": "https://api.example",
                "RAGTIME_BEARER_TOKEN": "secret",
                "RAGTIME_CACHE_PATH": "/tmp/legacy.sqlite3",
                "RAGTIME_CACHE_DIR": "",
            },
            clear=False,
        ), self.assertRaises(RagtimeConfigurationError):
            RagtimeClient.from_env(session=RecordingSession())

    def test_from_env_rejects_legacy_settings_for_cache_opt_out_and_injected_cache(self) -> None:
        cache, _ = make_cache()
        for variable in ("RAGTIME_CACHE_PATH", "RAGTIME_CACHE_DIR"):
            with self.subTest(variable=variable), patch.dict(
                os.environ,
                {
                    "RAGTIME_BASE_URL": "https://api.example",
                    "RAGTIME_BEARER_TOKEN": "secret",
                    "RAGTIME_CACHE_PATH": "",
                    "RAGTIME_CACHE_DIR": "",
                    variable: "/tmp/legacy-cache",
                },
                clear=False,
            ):
                with self.assertRaises(RagtimeConfigurationError):
                    RagtimeClient.from_env(
                        use_cache=False,
                        session=RecordingSession(),
                    )
                with self.assertRaises(RagtimeConfigurationError):
                    RagtimeClient.from_env(
                        cache=cache,
                        session=RecordingSession(),
                    )

    def test_cache_opt_out_does_not_require_artifact_bucket_configuration(self) -> None:
        with patch.dict(
            os.environ,
            {
                "RAGTIME_BASE_URL": "https://api.example",
                "RAGTIME_BEARER_TOKEN": "secret",
                "RAGTIME_CACHE_PATH": "",
                "RAGTIME_CACHE_DIR": "",
                "RAGTIME_HF_BUCKET_ID": "",
            },
            clear=False,
        ), patch("trec_ragtime.client.ArtifactStore.from_env") as from_env:
            RagtimeClient.from_env(use_cache=False, session=RecordingSession())
        from_env.assert_not_called()

    def test_missing_artifact_cache_configuration_disables_cache_with_one_warning(self) -> None:
        with patch.dict(
            os.environ,
            {
                "RAGTIME_BASE_URL": "https://api.example",
                "RAGTIME_BEARER_TOKEN": "secret",
                "RAGTIME_HF_BUCKET_ID": "",
                "RAGTIME_CACHE_PATH": "",
                "RAGTIME_CACHE_DIR": "",
            },
            clear=True,
        ), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            client = RagtimeClient.from_env(session=RecordingSession())
        self.assertIsNone(client.cache)
        self.assertIsNone(client.artifact_store)
        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, RuntimeWarning)
        self.assertNotIn("secret", str(caught[0].message))
        client.close()

    def test_missing_bucket_does_not_hide_invalid_cache_environment_settings(self) -> None:
        settings = (
            {"RAGTIME_CACHE_NAMESPACE": ""},
            {"RAGTIME_ARTIFACT_PREFIX": "/invalid"},
            {"RAGTIME_ARTIFACT_CACHE_DIR": "relative-cache"},
        )
        for setting in settings:
            with self.subTest(setting=setting), patch.dict(
                os.environ,
                {
                    "RAGTIME_BASE_URL": "https://api.example",
                    "RAGTIME_BEARER_TOKEN": "secret",
                    "RAGTIME_HF_BUCKET_ID": "",
                    **setting,
                },
                clear=True,
            ):
                with self.assertRaises(RagtimeConfigurationError):
                    RagtimeClient.from_env(session=RecordingSession())

    def test_package_root_does_not_export_cache_implementation_surface(self) -> None:
        import trec_ragtime

        cache_exports = (
            "CACHE_SCHEMA_VERSION",
            "DEFAULT_CACHE_NAMESPACE",
            "ArtifactStore",
            "ArtifactStoreConfigurationError",
            "ArtifactStoreError",
            "ArtifactStoreRemoteError",
            "ArtifactStoreUnavailableError",
            "ArtifactValidationError",
            "ArtifactStoreLike",
            "PROJECT_CACHE_DIR_ENV",
            "default_project_cache_dir",
            "ensure_private_cache_dir",
            "CacheEntry",
            "ResponseCache",
            "ResponseEnvelope",
            "RetrievalArtifactCache",
            "base_url_digest",
            "canonical_request_bytes",
            "canonicalize_json",
            "request_digest",
            "response_object_key",
        )
        for name in cache_exports:
            with self.subTest(name=name):
                self.assertFalse(hasattr(trec_ragtime, name))

    def test_public_client_exposes_artifact_store_and_sanitized_configuration(self) -> None:
        cache, store = make_cache()
        client = RagtimeClient("https://api.example", "secret", cache=cache, use_cache=True)
        self.assertIs(client.artifact_store, store)
        self.assertFalse(hasattr(client, "cache_path"))

    def test_injected_store_metadata_is_rejected_without_echoing_values(self) -> None:
        cache, store = make_cache()
        store.bucket_id = "owner/private?token=do-not-echo"
        store.prefix = "https://private.example/body=do-not-echo"
        store.local_dir = "owner/credential=do-not-echo"
        client = RagtimeClient("https://api.example", "secret", cache=cache, use_cache=True)

        with self.assertRaises(RagtimeConfigurationError) as raised:
            _ = client.cache_configuration

        self.assertNotIn("do-not-echo", str(raised.exception))

    def test_constructor_rejects_invalid_cache_ttls(self) -> None:
        for value in (-1, float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaises(RagtimeConfigurationError):
                RagtimeClient(
                    "https://api.example", "secret", use_cache=False,
                    search_cache_ttl_seconds=value,
                )

    def test_constructor_rejects_invalid_request_rates(self) -> None:
        for value in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaisesRegex(
                RagtimeConfigurationError, "requests_per_second"
            ):
                RagtimeClient(
                    "https://api.example", "secret", use_cache=False,
                    requests_per_second=value,
                )

    def test_cache_control_flags_require_booleans(self) -> None:
        with self.assertRaisesRegex(RagtimeConfigurationError, "use_cache"):
            RagtimeClient("https://api.example", "secret", use_cache="false")
        client = RagtimeClient(
            "https://api.example", "secret", use_cache=False,
            session=RecordingSession(),
        )
        with self.assertRaisesRegex(ValueError, "refresh"):
            client.search("query", refresh="false")
        with self.assertRaisesRegex(ValueError, "refresh"):
            client.get_content("doc-1", refresh="false")

    def test_configuration_rejects_insecure_or_malformed_credentials(self) -> None:
        with self.assertRaises(RagtimeConfigurationError):
            RagtimeClient("http://api.example", "secret", use_cache=False)
        with self.assertRaises(RagtimeConfigurationError):
            RagtimeClient("https://api.example", "secret\nheader", use_cache=False)
        for base_url in (
            "https://user:password@api.example",
            "https://api.example?query=yes",
            "https://api.example#fragment",
        ):
            with self.subTest(base_url=base_url), self.assertRaises(RagtimeConfigurationError):
                RagtimeClient(base_url, "secret", use_cache=False)


if __name__ == "__main__":
    unittest.main()
