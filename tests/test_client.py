from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import requests

from trec_ragtime import (
    RagtimeApiError,
    RagtimeClient,
    RagtimeConfigurationError,
    RagtimeProtocolError,
    SQLiteResponseCache,
    default_cache_path,
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


class RecordingSession:
    def __init__(self, *responses):
        self.responses = deque(responses)
        self.calls = []
        self.closed = False

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if not self.responses:
            raise AssertionError("unexpected upstream request")
        return self.responses.popleft()

    def close(self):
        self.closed = True


def pipeline_payload(scores=None, *, query="query", pipeline="ragtime2%20"):
    return {
        "pipeline": pipeline,
        "processed": True,
        "query": query,
        "scores": {"doc-b": 0.2, "doc-c": 0.9, "doc-a": 0.9}
        if scores is None
        else scores,
        "timestamp": 123.0,
    }


class RagtimeClientTests(unittest.TestCase):
    def test_search_posts_exact_request_and_canonical_bearer_header(self):
        for token in ("secret", " Bearer secret ", "bearer secret", "BEARER secret"):
            with self.subTest(token=token):
                session = RecordingSession(FakeResponse(pipeline_payload()))
                client = RagtimeClient(
                    "https://api.example/",
                    token,
                    session=session,
                    use_cache=False,
                )
                result = client.search("query")

                self.assertEqual(session.calls[0][0], "https://api.example/pipeline")
                self.assertEqual(
                    session.calls[0][1]["json"],
                    {"pipeline": "ragtime2%20", "query": "query"},
                )
                self.assertEqual(
                    session.calls[0][1]["headers"]["Authorization"],
                    "Bearer secret",
                )
                self.assertFalse(result.local_cache_hit)

    def test_search_sorts_scores_with_document_id_tiebreak(self):
        session = RecordingSession(FakeResponse(pipeline_payload()))
        client = RagtimeClient(
            "https://api.example", "secret", session=session, use_cache=False
        )

        result = client.search("query")

        self.assertEqual(
            [hit.document_id for hit in result.hits], ["doc-a", "doc-c", "doc-b"]
        )
        self.assertEqual([hit.rank for hit in result.hits], [1, 2, 3])

    def test_cache_is_reused_across_client_instances_and_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "shared" / "responses.sqlite3"
            first_session = RecordingSession(FakeResponse(pipeline_payload()))
            first = RagtimeClient(
                "https://api.example/",
                "first-token",
                cache_path=cache_path,
                session=first_session,
            )
            first_result = first.search("query")

            second_session = RecordingSession()
            second = RagtimeClient(
                "https://api.example",
                "second-token",
                cache_path=cache_path,
                session=second_session,
            )
            second_result = second.search("query")

            self.assertFalse(first_result.local_cache_hit)
            self.assertTrue(second_result.local_cache_hit)
            self.assertEqual(second_result.hits, first_result.hits)
            self.assertEqual(second_session.calls, [])

    def test_default_api_cache_entries_never_expire(self):
        with tempfile.TemporaryDirectory() as directory:
            now = [100.0]
            cache = SQLiteResponseCache(
                Path(directory) / "responses.sqlite3",
                clock=lambda: now[0],
            )
            content_payload = {
                "cached": False,
                "collection": "ragtime2",
                "id": "doc-1",
                "text": "document text",
            }
            first_session = RecordingSession(
                FakeResponse(pipeline_payload({"doc-1": 1.0})),
                FakeResponse(content_payload),
            )
            first = RagtimeClient(
                "https://api.example",
                "first-token",
                cache=cache,
                session=first_session,
            )
            first.search("query")
            first.get_content("doc-1")

            now[0] += 100 * 365 * 24 * 60 * 60
            second_session = RecordingSession()
            second = RagtimeClient(
                "https://api.example",
                "second-token",
                cache=cache,
                session=second_session,
            )
            cached_search = second.search("query")
            cached_document = second.get_content("doc-1")

            self.assertIsNone(second.search_cache_ttl_seconds)
            self.assertIsNone(second.content_cache_ttl_seconds)
            self.assertTrue(cached_search.local_cache_hit)
            self.assertTrue(cached_document.local_cache_hit)
            self.assertEqual(second_session.calls, [])

    def test_refresh_replaces_a_cached_response(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "responses.sqlite3"
            first = RagtimeClient(
                "https://api.example",
                "secret",
                cache_path=cache_path,
                session=RecordingSession(FakeResponse(pipeline_payload({"old": 0.1}))),
            )
            first.search("query")

            refresh_session = RecordingSession(
                FakeResponse(pipeline_payload({"new": 0.8}))
            )
            refreshed = RagtimeClient(
                "https://api.example",
                "secret",
                cache_path=cache_path,
                session=refresh_session,
            ).search("query", refresh=True)

            cached = RagtimeClient(
                "https://api.example",
                "secret",
                cache_path=cache_path,
                session=RecordingSession(),
            ).search("query")

            self.assertFalse(refreshed.local_cache_hit)
            self.assertEqual(refreshed.hits[0].document_id, "new")
            self.assertTrue(cached.local_cache_hit)
            self.assertEqual(cached.hits[0].document_id, "new")

    def test_failed_refresh_preserves_previous_cached_response(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "responses.sqlite3"
            RagtimeClient(
                "https://api.example",
                "secret",
                cache_path=cache_path,
                session=RecordingSession(
                    FakeResponse(pipeline_payload({"previous": 0.5}))
                ),
            ).search("query")

            refreshing = RagtimeClient(
                "https://api.example",
                "secret",
                cache_path=cache_path,
                session=RecordingSession(FakeResponse({}, status_code=503)),
            )
            with self.assertRaises(RagtimeApiError):
                refreshing.search("query", refresh=True)

            cached = RagtimeClient(
                "https://api.example",
                "secret",
                cache_path=cache_path,
                session=RecordingSession(),
            ).search("query")
            self.assertTrue(cached.local_cache_hit)
            self.assertEqual(cached.hits[0].document_id, "previous")

    def test_content_uses_ragtime2_collection_and_is_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "responses.sqlite3"
            payload = {
                "cached": False,
                "collection": "ragtime2",
                "id": "doc-1",
                "text": "document text",
            }
            session = RecordingSession(FakeResponse(payload))
            first = RagtimeClient(
                "https://api.example",
                "secret",
                cache_path=cache_path,
                session=session,
            ).get_content("doc-1")
            second = RagtimeClient(
                "https://api.example",
                "secret",
                cache_path=cache_path,
                session=RecordingSession(),
            ).get_content("doc-1")

            self.assertEqual(
                session.calls[0][1]["json"],
                {"collection": "ragtime2", "id": "doc-1"},
            )
            self.assertEqual(first.text, "document text")
            self.assertTrue(second.local_cache_hit)

    def test_malformed_payload_is_not_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "responses.sqlite3"
            malformed = RagtimeClient(
                "https://api.example",
                "secret",
                cache_path=cache_path,
                session=RecordingSession(FakeResponse({"processed": True})),
            )
            with self.assertRaises(RagtimeProtocolError):
                malformed.search("query")

            valid_session = RecordingSession(FakeResponse(pipeline_payload()))
            valid = RagtimeClient(
                "https://api.example",
                "secret",
                cache_path=cache_path,
                session=valid_session,
            ).search("query")
            self.assertFalse(valid.local_cache_hit)
            self.assertEqual(len(valid_session.calls), 1)

    def test_malformed_cached_object_is_evicted_and_refetched_once(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "responses.sqlite3"
            cache = SQLiteResponseCache(cache_path)
            request = {"pipeline": "ragtime2%20", "query": "query"}
            cache.put(
                base_url="https://api.example",
                operation="pipeline",
                request=request,
                response={},
            )
            session = RecordingSession(FakeResponse(pipeline_payload()))
            repaired = RagtimeClient(
                "https://api.example",
                "secret",
                cache=cache,
                session=session,
            ).search("query")

            self.assertFalse(repaired.local_cache_hit)
            self.assertEqual(len(session.calls), 1)
            cached = RagtimeClient(
                "https://api.example",
                "secret",
                cache=cache,
                session=RecordingSession(),
            ).search("query")
            self.assertTrue(cached.local_cache_hit)

    def test_http_error_is_sanitized_and_not_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "responses.sqlite3"
            client = RagtimeClient(
                "https://api.example",
                "do-not-leak",
                cache_path=cache_path,
                session=RecordingSession(FakeResponse({}, status_code=401)),
            )
            with self.assertRaisesRegex(RagtimeApiError, "HTTP 401") as caught:
                client.search("query")
            self.assertNotIn("do-not-leak", str(caught.exception))

            connection = sqlite3.connect(cache_path)
            try:
                count = connection.execute("SELECT COUNT(*) FROM responses").fetchone()[
                    0
                ]
            finally:
                connection.close()
            self.assertEqual(count, 0)

    def test_token_is_not_written_to_cache_files(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "private" / "responses.sqlite3"
            token = "unique-secret-token"
            client = RagtimeClient(
                "https://api.example",
                token,
                cache_path=cache_path,
                session=RecordingSession(FakeResponse(pipeline_payload())),
            )
            client.search("query")

            for cache_file in cache_path.parent.iterdir():
                if cache_file.is_file():
                    self.assertNotIn(token.encode(), cache_file.read_bytes())
            self.assertEqual(cache_path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(cache_path.parent.stat().st_mode & 0o777, 0o700)

    def test_cache_ttl_expires_without_deleting_previous_value(self):
        with tempfile.TemporaryDirectory() as directory:
            now = [100.0]
            cache = SQLiteResponseCache(
                Path(directory) / "responses.sqlite3",
                clock=lambda: now[0],
            )
            request = {"pipeline": "ragtime2%20", "query": "query"}
            response = pipeline_payload()
            cache.put(
                base_url="https://api.example",
                operation="pipeline",
                request=request,
                response=response,
            )

            now[0] = 109.9
            self.assertIsNotNone(
                cache.get(
                    base_url="https://api.example",
                    operation="pipeline",
                    request=request,
                    ttl_seconds=10,
                )
            )
            now[0] = 110.0
            self.assertIsNone(
                cache.get(
                    base_url="https://api.example",
                    operation="pipeline",
                    request=request,
                    ttl_seconds=10,
                )
            )
            self.assertIsNotNone(
                cache.get(
                    base_url="https://api.example",
                    operation="pipeline",
                    request=request,
                    ttl_seconds=None,
                )
            )

    def test_sqlite_cache_accepts_concurrent_writers(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "shared" / "responses.sqlite3"
            SQLiteResponseCache(cache_path)

            def write_response(index):
                cache = SQLiteResponseCache(cache_path)
                cache.put(
                    base_url="https://api.example",
                    operation="pipeline",
                    request={"pipeline": "ragtime2%20", "query": f"query-{index}"},
                    response=pipeline_payload(query=f"query-{index}"),
                )

            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(write_response, range(40)))

            connection = sqlite3.connect(cache_path)
            try:
                count = connection.execute("SELECT COUNT(*) FROM responses").fetchone()[
                    0
                ]
            finally:
                connection.close()
            self.assertEqual(count, 40)

    def test_default_cache_path_is_independent_of_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worktree_a = root / "worktree-a"
            worktree_b = root / "worktree-b"
            worktree_a.mkdir()
            worktree_b.mkdir()
            with (
                patch.dict(
                    os.environ,
                    {
                        "RAGTIME_PROJECT_CACHE_DIR": str(
                            root / "cache" / "trec-ragtime-2026"
                        ),
                        "XDG_CACHE_HOME": "",
                    },
                    clear=False,
                ),
                patch.dict(
                    os.environ,
                    {"RAGTIME_CACHE_PATH": "", "RAGTIME_CACHE_DIR": ""},
                ),
            ):
                original_cwd = Path.cwd()
                try:
                    os.chdir(worktree_a)
                    path_a = default_cache_path()
                    os.chdir(worktree_b)
                    path_b = default_cache_path()
                finally:
                    os.chdir(original_cwd)

            self.assertEqual(path_a, path_b)
            self.assertEqual(
                path_a, root / "cache" / "trec-ragtime-2026" / "api-v1.sqlite3"
            )

    def test_relative_cache_paths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "absolute path"):
            SQLiteResponseCache("relative.sqlite3")
        for variable in (
            "RAGTIME_CACHE_PATH",
            "RAGTIME_CACHE_DIR",
            "RAGTIME_PROJECT_CACHE_DIR",
        ):
            with (
                self.subTest(variable=variable),
                patch.dict(
                    os.environ,
                    {
                        "RAGTIME_CACHE_PATH": "",
                        "RAGTIME_CACHE_DIR": "",
                        "RAGTIME_PROJECT_CACHE_DIR": "",
                        "XDG_CACHE_HOME": "",
                        variable: "relative-cache",
                    },
                ),
            ):
                with self.assertRaisesRegex(ValueError, "absolute path"):
                    default_cache_path()

    def test_preexisting_permissive_cache_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_directory = Path(directory) / "shared"
            cache_directory.mkdir(mode=0o755)
            cache_directory.chmod(0o755)
            with self.assertRaisesRegex(PermissionError, "user-only"):
                SQLiteResponseCache(cache_directory / "responses.sqlite3")

    def test_constructor_rejects_invalid_cache_ttls(self):
        for value in (-1, float("nan"), float("inf"), True):
            with (
                self.subTest(value=value),
                self.assertRaises(RagtimeConfigurationError),
            ):
                RagtimeClient(
                    "https://api.example",
                    "secret",
                    use_cache=False,
                    search_cache_ttl_seconds=value,
                )

        RagtimeClient(
            "https://api.example",
            "secret",
            use_cache=False,
            search_cache_ttl_seconds=0,
            content_cache_ttl_seconds=None,
        )

    def test_cache_control_flags_require_booleans(self):
        with self.assertRaisesRegex(RagtimeConfigurationError, "use_cache"):
            RagtimeClient(
                "https://api.example",
                "secret",
                use_cache="false",
            )

        client = RagtimeClient(
            "https://api.example",
            "secret",
            use_cache=False,
            session=RecordingSession(),
        )
        with self.assertRaisesRegex(ValueError, "refresh"):
            client.search("query", refresh="false")
        with self.assertRaisesRegex(ValueError, "refresh"):
            client.get_content("doc-1", refresh="false")

    def test_configuration_rejects_insecure_or_malformed_credentials(self):
        with self.assertRaises(RagtimeConfigurationError):
            RagtimeClient("http://api.example", "secret", use_cache=False)
        with self.assertRaises(RagtimeConfigurationError):
            RagtimeClient("https://api.example", "secret\nheader", use_cache=False)
        for base_url in (
            "https://user:password@api.example",
            "https://api.example?query=yes",
            "https://api.example#fragment",
        ):
            with (
                self.subTest(base_url=base_url),
                self.assertRaises(RagtimeConfigurationError),
            ):
                RagtimeClient(base_url, "secret", use_cache=False)


if __name__ == "__main__":
    unittest.main()
