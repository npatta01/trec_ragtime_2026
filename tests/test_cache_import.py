from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import stat
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from trec_ragtime.cache import ResponseCache
from trec_ragtime.cache_import import CacheImportError, import_sqlite_cache


class MemoryArtifactStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.local: dict[str, bytes] = {}
        self.reads: list[str] = []
        self.read_sources: list[str] = []
        self.write_result = True
        self.keep_remote_on_write = False
        self.bucket_id = "owner/private-cache"
        self.prefix = "migration-v1"

    def read_validated(self, object_key: str, decoder: Any) -> Any | None:
        self.reads.append(object_key)
        encoded = self.local.get(object_key)
        source = "local"
        if encoded is None:
            encoded = self.objects.get(object_key)
            source = "remote"
        if encoded is None:
            return None
        self.read_sources.append(source)
        return decoder(encoded)

    def write(self, object_key: str, payload: bytes) -> bool:
        if self.write_result is not True:
            return False
        if not self.keep_remote_on_write or object_key not in self.objects:
            self.objects[object_key] = payload
        self.local[object_key] = payload
        return True

    def evict_local(self, object_key: str) -> None:
        self.local.pop(object_key, None)


def pipeline_row(
    *,
    cache_key: str = "pipeline-key",
    schema_version: int = 1,
    namespace: str = "ragtime2-2026-v1",
    base_url: str = "https://ragtime.example.test",
    operation: str = "pipeline",
    request: dict[str, Any] | None = None,
    response: dict[str, Any] | None = None,
    stored_at: float = 1_700_000_000.0,
) -> tuple[Any, ...]:
    return (
        cache_key,
        schema_version,
        namespace,
        base_url,
        operation,
        json.dumps(request or {"pipeline": "ragtime2%100", "query": "safe query"}),
        json.dumps(
            response
            or {
                "query": "safe query",
                "pipeline": "ragtime2%100",
                "scores": {"doc-1": 0.9, "doc-2": 0.8},
            }
        ),
        stored_at,
    )


def content_row(
    *,
    cache_key: str = "content-key",
    request: dict[str, Any] | None = None,
    response: dict[str, Any] | None = None,
    stored_at: float = 1_700_000_000.0,
) -> tuple[Any, ...]:
    return (
        cache_key,
        1,
        "ragtime2-2026-v1",
        "https://ragtime.example.test",
        "content",
        json.dumps(request or {"collection": "ragtime2", "id": "doc-1"}),
        json.dumps(
            response
            or {"collection": "ragtime2", "id": "doc-1", "text": "source text"}
        ),
        stored_at,
    )


def huggingface_row(
    *, cache_key: str = "hf-key", stored_at: float, response: dict[str, Any] | None = None
) -> tuple[Any, ...]:
    request = {
        "config": "default",
        "dataset": "trec-ragtime/ragtime2",
        "id": "doc-1",
        "splits": ["eng", "spa", "rus", "zho"],
        "variant": "original",
    }
    payload = response or {
        "config": "default",
        "dataset": "trec-ragtime/ragtime2",
        "date": None,
        "id": "doc-1",
        "split": "eng",
        "text": "canonical document text",
        "url": "https://example.test/doc-1",
    }
    return (
        cache_key,
        1,
        "ragtime2-2026-v1",
        "https://datasets-server.huggingface.co",
        "huggingface_document",
        json.dumps(request),
        json.dumps(payload),
        stored_at,
    )


class CacheImportTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.source = self.root / "legacy.sqlite3"
        self.manifest = self.root / "import-manifest.json"
        self._create_source([])

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _create_source(self, rows: list[tuple[Any, ...]]) -> None:
        connection = sqlite3.connect(self.source)
        try:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE responses (
                        cache_key TEXT PRIMARY KEY,
                        schema_version INTEGER NOT NULL,
                        namespace TEXT NOT NULL,
                        base_url TEXT NOT NULL,
                        operation TEXT NOT NULL,
                        request_json TEXT NOT NULL,
                        response_json TEXT NOT NULL,
                        stored_at REAL NOT NULL
                    )
                    """
                )
                connection.executemany(
                    "INSERT INTO responses VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows
                )
        finally:
            connection.close()

    def _cache(self) -> tuple[ResponseCache, MemoryArtifactStore]:
        store = MemoryArtifactStore()
        return ResponseCache(store, clock=time.time), store

    def test_import_filters_rows_reuses_validators_and_writes_sanitized_manifest(self) -> None:
        now = time.time()
        rows = [
            pipeline_row(),
            content_row(cache_key="content-valid"),
            huggingface_row(stored_at=now - 1),
            # Unsupported schema, namespace, and operation are independently counted.
            pipeline_row(cache_key="schema", schema_version=2),
            pipeline_row(cache_key="namespace", namespace="codex-private"),
            pipeline_row(cache_key="operation", operation="topics"),
            # Endpoint/domain validation and finite timestamp checks reject these rows.
            pipeline_row(
                cache_key="bad-echo",
                response={"query": "different", "scores": {"doc-1": 0.9}},
            ),
            content_row(
                cache_key="bad-document",
                response={"collection": "ragtime2", "id": "doc-1", "text": 7},
            ),
            huggingface_row(
                cache_key="hf-expired",
                stored_at=now - (30 * 24 * 60 * 60 + 1),
            ),
            pipeline_row(cache_key="non-finite", stored_at=math.inf),
            pipeline_row(
                cache_key="non-finite-score",
                response={
                    "query": "safe query",
                    "pipeline": "ragtime2%100",
                    "scores": {"doc-1": math.inf},
                },
            ),
            pipeline_row(
                cache_key="secret",
                response={
                    "query": "safe query",
                    "scores": {"doc-1": 0.9},
                    "bearer_token": "secret-token-value",
                },
            ),
        ]
        self.source.unlink()
        self._create_source(rows)
        before = self.source.read_bytes()
        source_hash = hashlib.sha256(before).hexdigest()
        cache, store = self._cache()

        summary = import_sqlite_cache(self.source, cache, self.manifest)

        self.assertEqual(summary.source_sha256, source_hash)
        self.assertEqual(summary.source_size, len(before))
        self.assertEqual(summary.imported["pipeline"], 1)
        self.assertEqual(summary.imported["content"], 1)
        self.assertEqual(summary.imported["huggingface_document"], 1)
        self.assertEqual(summary.verified, summary.imported)
        self.assertEqual(
            summary.skipped,
            {
                "pipeline": 6,
                "content": 1,
                "huggingface_document": 1,
                "unsupported": 1,
            },
        )
        self.assertEqual(summary.skipped_reasons, {"row_rejected": 9})
        self.assertEqual(self.source.read_bytes(), before)
        self.assertGreaterEqual(len(store.reads), 3)
        self.assertEqual(store.read_sources, ["remote", "remote", "remote"])

        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        self.assertEqual(manifest["source"]["sha256"], source_hash)
        self.assertEqual(manifest["source"]["size"], len(before))
        self.assertEqual(
            manifest["destination"],
            {
                "bucket_id": "owner/private-cache",
                "namespace": "ragtime2-2026-v1",
                "prefix": "migration-v1",
            },
        )
        self.assertEqual(manifest["counts"]["pipeline"]["imported"], 1)
        self.assertEqual(manifest["counts"]["pipeline"]["skipped"], 6)
        self.assertEqual(manifest["counts"]["huggingface_document"]["verified"], 1)
        self.assertEqual(manifest["skipped_reasons"], {"row_rejected": 9})
        manifest_text = self.manifest.read_text(encoding="utf-8")
        for forbidden in (
            "source text",
            "safe query",
            "query that must not be copied",
            "secret-token-value",
            "https://example.test/doc-1",
        ):
            self.assertNotIn(forbidden, manifest_text)
        self.assertEqual(stat.S_IMODE(self.manifest.stat().st_mode), 0o600)

    def test_import_preserves_stored_at_and_is_idempotent_last_writer_wins(self) -> None:
        stored_at = 1_700_000_123.5
        self.source.unlink()
        self._create_source([pipeline_row(stored_at=stored_at)])
        cache, _ = self._cache()

        first = import_sqlite_cache(self.source, cache, self.manifest)
        second = import_sqlite_cache(self.source, cache, self.manifest)

        self.assertEqual(first.imported, second.imported)
        request = {"pipeline": "ragtime2%100", "query": "safe query"}
        entry = cache.get(
            base_url="https://ragtime.example.test",
            operation="pipeline",
            request=request,
            ttl_seconds=None,
            validator=lambda payload: payload,
        )
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry.stored_at, stored_at)

    def test_l2_failure_after_l1_eviction_fails_independent_verification(self) -> None:
        self.source.unlink()
        self._create_source([pipeline_row()])

        class RemoteDisappearsStore(MemoryArtifactStore):
            def evict_local(self, object_key: str) -> None:
                super().evict_local(object_key)
                self.objects.clear()

        store = RemoteDisappearsStore()
        cache = ResponseCache(store, clock=time.time)
        before = self.source.read_bytes()

        with self.assertRaises(CacheImportError):
            import_sqlite_cache(self.source, cache, self.manifest)

        self.assertEqual(self.source.read_bytes(), before)
        self.assertFalse(self.manifest.exists())
        self.assertEqual(store.read_sources, [])

    def test_write_false_fails_before_independent_verification(self) -> None:
        self.source.unlink()
        self._create_source([pipeline_row()])
        cache, store = self._cache()
        request = {"pipeline": "ragtime2%100", "query": "safe query"}
        response = {
            "query": "safe query",
            "pipeline": "ragtime2%100",
            "scores": {"doc-1": 0.9, "doc-2": 0.8},
        }
        cache.put(
            base_url="https://ragtime.example.test",
            operation="pipeline",
            request=request,
            response=response,
            stored_at=1_700_000_000.0,
        )
        before = self.source.read_bytes()
        store.write_result = False

        with self.assertRaises(CacheImportError):
            import_sqlite_cache(self.source, cache, self.manifest)

        self.assertEqual(self.source.read_bytes(), before)
        self.assertFalse(self.manifest.exists())
        self.assertEqual(store.read_sources, [])
        self.assertEqual(store.local, store.objects)

    def test_independent_verification_rejects_mismatched_artifact(self) -> None:
        self.source.unlink()
        self._create_source([pipeline_row()])
        cache, store = self._cache()
        request = {"pipeline": "ragtime2%100", "query": "safe query"}
        stale_response = {
            "query": "safe query",
            "pipeline": "ragtime2%100",
            "scores": {"doc-1": 0.1, "doc-2": 0.2},
        }
        cache.put(
            base_url="https://ragtime.example.test",
            operation="pipeline",
            request=request,
            response=stale_response,
            stored_at=1_700_000_000.0,
        )
        store.keep_remote_on_write = True
        before = self.source.read_bytes()

        with self.assertRaises(CacheImportError):
            import_sqlite_cache(self.source, cache, self.manifest)

        self.assertEqual(self.source.read_bytes(), before)
        self.assertFalse(self.manifest.exists())
        self.assertEqual(store.read_sources, ["remote"])

    def test_verification_uses_source_timestamp_not_adapter_return_value(self) -> None:
        self.source.unlink()
        self._create_source([pipeline_row()])
        cache, _store = self._cache()
        original_put = cache.put

        def put_with_different_success(**kwargs: Any) -> float | None:
            result = original_put(**kwargs)
            if result is None:
                return None
            return result + 1.0

        before = self.source.read_bytes()
        with patch.object(cache, "put", side_effect=put_with_different_success):
            summary = import_sqlite_cache(self.source, cache, self.manifest)

        self.assertEqual(summary.imported["pipeline"], 1)
        self.assertEqual(summary.verified["pipeline"], 1)
        self.assertEqual(self.source.read_bytes(), before)

    def test_source_is_opened_read_only_and_missing_source_does_not_get_created(self) -> None:
        self.source.unlink()
        with self.assertRaises(FileNotFoundError):
            import_sqlite_cache(self.source, self._cache()[0], self.manifest)
        self.assertFalse(self.source.exists())

        self._create_source([pipeline_row()])
        self.source.chmod(0o444)
        before = self.source.read_bytes()
        import_sqlite_cache(self.source, self._cache()[0], self.manifest)
        self.assertEqual(self.source.read_bytes(), before)

    def test_manifest_source_collision_is_rejected_before_reads_or_writes(self) -> None:
        for collision in ("same-path", "hardlink", "symlink"):
            with self.subTest(collision=collision):
                self.source.unlink()
                self._create_source([pipeline_row()])
                if collision == "same-path":
                    manifest_path = self.source
                elif collision == "hardlink":
                    manifest_path = self.root / "manifest-hardlink.json"
                    manifest_path.hardlink_to(self.source)
                else:
                    manifest_path = self.root / "manifest-symlink.json"
                    manifest_path.symlink_to(self.source)
                before = self.source.read_bytes()
                cache, store = self._cache()

                with self.assertRaises(CacheImportError) as raised:
                    import_sqlite_cache(self.source, cache, manifest_path)

                self.assertNotIn(str(self.source), str(raised.exception))
                self.assertEqual(self.source.read_bytes(), before)
                self.assertEqual(store.objects, {})

    def test_noncanonical_namespace_override_is_rejected_without_writes_or_source_mutation(self) -> None:
        self.source.unlink()
        self._create_source([pipeline_row(namespace="codex-private")])
        store = MemoryArtifactStore()
        cache = ResponseCache(store, namespace="codex-private", clock=time.time)
        before = self.source.read_bytes()

        with self.assertRaises(ValueError):
            import_sqlite_cache(
                self.source,
                cache,
                self.manifest,
                canonical_namespace="codex-private",
            )

        self.assertEqual(store.objects, {})
        self.assertEqual(self.source.read_bytes(), before)
        self.assertFalse(self.manifest.exists())

    def test_unsafe_destination_metadata_is_rejected_without_leaking_values(self) -> None:
        bad_metadata = (
            "owner/api_key=sensitive",
            "owner/credential=sensitive",
            "owner/private?query=sensitive",
            "owner/private#fragment",
            "https://owner/private?token=secret",
        )
        for bad_value in bad_metadata:
            with self.subTest(bad_value=bad_value):
                cache, store = self._cache()
                store.bucket_id = bad_value
                before = self.source.read_bytes()

                with self.assertRaises(CacheImportError) as raised:
                    import_sqlite_cache(self.source, cache, self.manifest)

                self.assertNotIn(bad_value, str(raised.exception))
                self.assertEqual(store.objects, {})
                self.assertEqual(self.source.read_bytes(), before)
                self.assertFalse(self.manifest.exists())

        for bad_value in (
            "api_key=sensitive",
            "credential=sensitive",
            "query=sensitive",
            '{"body":"private body"}',
            "https://private/path?token=secret",
        ):
            with self.subTest(bad_value=bad_value):
                cache, store = self._cache()
                store.prefix = bad_value
                before = self.source.read_bytes()

                with self.assertRaises(CacheImportError) as raised:
                    import_sqlite_cache(self.source, cache, self.manifest)

                self.assertNotIn(bad_value, str(raised.exception))
                self.assertEqual(store.objects, {})
                self.assertEqual(self.source.read_bytes(), before)
                self.assertFalse(self.manifest.exists())


if __name__ == "__main__":
    unittest.main()
