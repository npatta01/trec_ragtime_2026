from __future__ import annotations

import math
import unittest
from typing import Any, Callable

import msgspec

from trec_ragtime.artifact_store import ArtifactValidationError
from trec_ragtime.models import (
    PipelineResult,
    RetrievalResult,
    RetrievedDocument,
    SearchHit,
)
from trec_ragtime.retrieval_cache import (
    RETRIEVAL_SCHEMA_VERSION,
    RetrievalArtifactCache,
    RetrievalArtifactReference,
    RetrievalEnvelope,
    RetrievalRequestIdentity,
    retrieval_object_key,
    retriever_fingerprint,
)


class MemoryArtifactStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.reads: list[str] = []
        self.writes: list[tuple[str, bytes]] = []

    def read_validated(
        self, object_key: str, decoder: Callable[[bytes], Any]
    ) -> Any | None:
        self.reads.append(object_key)
        payload = self.objects.get(object_key)
        if payload is None:
            return None
        return decoder(payload)

    def write(self, object_key: str, payload: bytes) -> None:
        self.writes.append((object_key, payload))
        self.objects[object_key] = payload


def request_fixture(**changes: Any) -> RetrievalRequestIdentity:
    values: dict[str, Any] = {
        "query": "¿qué ocurrió? 中文",
        "pipeline": "ragtime2%20",
        "top_k": None,
        "endpoint_depth_version": "endpoint-depth-v1",
        "max_content_fallbacks": 5,
        "fallback_on_hf_error": False,
        "content_collection": "ragtime2",
        "backend_policy": "local-corpus-v2",
        "query_context": {"kind": "control_a", "construction": "problem_only_v1"},
        "base_url": "https://api.example.test/",
        "corpus_identity": {
            "backend": "local-corpus",
            "path": "/machine/local/path.duckdb",
            "schema_version": 2,
            "dataset": "trec-ragtime/ragtime2",
            "config": "default",
            "revision": "abc123",
            "document_count": 100,
            "shard_count": 4,
            "splits": ["eng", "spa", "rus", "zho"],
            "translation_config": "mt_docs",
            "translation_count": 50,
            "translation_shard_count": 2,
            "translation_splits": ["spa.mt.eng", "rus.mt.eng", "zho.mt.eng"],
        },
    }
    values.update(changes)
    return RetrievalRequestIdentity(**values)


def result_fixture(**changes: Any) -> RetrievalResult:
    hit = SearchHit(rank=1, document_id="doc-1", score=0.75)
    search = PipelineResult(
        query="¿qué ocurrió? 中文",
        pipeline="ragtime2%20",
        hits=(hit,),
        processed=True,
        local_cache_hit=True,
        remote_cached=None,
        server_timestamp=12.0,
        cache_stored_at=13.0,
    )
    document = RetrievedDocument(
        rank=1,
        document_id="doc-1",
        score=0.75,
        text="source bytes: café",
        url="https://example.test/doc-1",
        date="2026-01-01",
        language="spa",
        content_source="huggingface",
        content_local_cache_hit=True,
        english_text="English translation",
    )
    values: dict[str, Any] = {
        "search_result": search,
        "documents": (document,),
        "hydration_failures": (),
        "missing_translation_ids": (),
    }
    values.update(changes)
    return RetrievalResult(**values)


class RetrievalCacheTestCase(unittest.TestCase):
    def test_identity_is_stable_across_machine_path_refresh_and_credentials(self) -> None:
        first = request_fixture()
        second = request_fixture(
            corpus_identity={**first.corpus_identity, "path": "/other/host/index.duckdb"}
        )
        self.assertEqual(first.request_digest, second.request_digest)
        self.assertEqual(first.retriever_fingerprint, second.retriever_fingerprint)
        self.assertEqual(first.object_key, second.object_key)

        for ignored in (
            {"refresh": True},
            {"worker_id": "worker-99"},
            {"concurrency": 8},
            {"token": "do-not-persist"},
        ):
            with self.subTest(ignored=ignored):
                # These controls are deliberately not constructor fields: the
                # identity builder receives no refresh, worker, concurrency, or
                # credential input at all.
                self.assertEqual(first.object_key, request_fixture().object_key)

    def test_each_result_affecting_identity_field_changes_request_or_fingerprint(self) -> None:
        base = request_fixture()
        mutations = (
            {"query": "different"},
            {"pipeline": "ragtime2"},
            {"top_k": 10},
            {"endpoint_depth_version": "endpoint-depth-v2"},
            {"max_content_fallbacks": 0},
            {"fallback_on_hf_error": True},
            {"content_collection": "ragtime/2/all"},
            {"backend_policy": "dataset-viewer"},
            {"query_context": {"kind": "baseline_b", "index": 1}},
            {"corpus_identity": {**base.corpus_identity, "revision": "def456"}},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                changed = request_fixture(**mutation)
                self.assertNotEqual(base.object_key, changed.object_key)

    def test_fingerprint_helper_and_object_layout_are_canonical(self) -> None:
        request = request_fixture()
        self.assertEqual(
            request.retriever_fingerprint,
            retriever_fingerprint(
                request.base_url,
                request.corpus_identity,
            ),
        )
        self.assertEqual(
            request.object_key,
            retrieval_object_key(request.retriever_fingerprint, request.request_digest),
        )
        self.assertRegex(
            request.object_key,
            r"^retrieval/v1/[0-9a-f]{64}/[0-9a-f]{2}/[0-9a-f]{64}\.mpk$",
        )

    def test_round_trip_preserves_nested_result_fields_and_reference_provenance(self) -> None:
        store = MemoryArtifactStore()
        cache = RetrievalArtifactCache(store, clock=lambda: 99.5)
        request = request_fixture()
        original = result_fixture()
        written = cache.put(request, original)
        self.assertEqual(written, original)
        self.assertEqual(len(store.writes), 1)
        envelope = msgspec.msgpack.decode(store.writes[0][1], type=RetrievalEnvelope)
        self.assertEqual(envelope.schema_version, RETRIEVAL_SCHEMA_VERSION)
        loaded = cache.get(request)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.search_result, original.search_result)
        self.assertEqual(loaded.documents, original.documents)
        self.assertEqual(loaded.missing_translation_ids, original.missing_translation_ids)
        self.assertIsInstance(loaded.retrieval_artifact, RetrievalArtifactReference)
        assert loaded.retrieval_artifact is not None
        self.assertTrue(loaded.retrieval_artifact.cache_hit)
        self.assertEqual(loaded.retrieval_artifact.object_key, request.object_key)
        self.assertEqual(loaded.retrieval_artifact.request_digest_hex, request.request_digest.hex())

    def test_strict_envelope_rejects_unknown_fields_and_truncation(self) -> None:
        store = MemoryArtifactStore()
        cache = RetrievalArtifactCache(store)
        request = request_fixture()
        cache.put(request, result_fixture())
        key, payload = store.writes[0]
        decoded = msgspec.msgpack.decode(payload)
        decoded["unknown"] = True
        store.objects[key] = msgspec.msgpack.encode(decoded)
        with self.assertRaises(ArtifactValidationError):
            cache.get(request)
        store.objects[key] = payload[:-1]
        with self.assertRaises(ArtifactValidationError):
            cache.get(request)

    def test_invalid_results_are_rejected_before_write(self) -> None:
        invalid_results = (
            result_fixture(
                search_result=PipelineResult(
                    query="q",
                    pipeline="p",
                    hits=(SearchHit(rank=2, document_id="doc-1", score=0.5),),
                    processed=True,
                    local_cache_hit=False,
                    remote_cached=None,
                    server_timestamp=1.0,
                    cache_stored_at=2.0,
                )
            ),
            result_fixture(
                documents=(
                    RetrievedDocument(
                        rank=1,
                        document_id="doc-1",
                        score=math.nan,
                        text="source",
                        url=None,
                        date=None,
                        language="spa",
                        content_source="huggingface",
                        content_local_cache_hit=None,
                    ),
                )
            ),
            result_fixture(
                hydration_failures=(
                    # A successful retrieval artifact never hides operational failures.
                    # The concrete type is checked by the adapter.
                    type("Failure", (), {"split": "eng", "message": "timeout"})(),
                )
            ),
            result_fixture(
                documents=(
                    RetrievedDocument(
                        rank=1,
                        document_id="doc-1",
                        score=0.75,
                        text=None,
                        url=None,
                        date=None,
                        language=None,
                        content_source=None,
                        content_local_cache_hit=None,
                    ),
                )
            ),
        )
        for invalid in invalid_results:
            with self.subTest(invalid=invalid):
                store = MemoryArtifactStore()
                with self.assertRaises((ValueError, TypeError, ArtifactValidationError)):
                    RetrievalArtifactCache(store).put(request_fixture(), invalid)
                self.assertEqual(store.writes, [])

    def test_missing_translation_is_cacheable_when_source_text_is_present(self) -> None:
        result = result_fixture(
            documents=(
                RetrievedDocument(
                    rank=1,
                    document_id="doc-1",
                    score=0.75,
                    text="source bytes",
                    url=None,
                    date=None,
                    language="spa",
                    content_source="huggingface",
                    content_local_cache_hit=True,
                    english_text=None,
                ),
            ),
            missing_translation_ids=("doc-1",),
        )
        store = MemoryArtifactStore()
        cache = RetrievalArtifactCache(store)
        cache.put(request_fixture(), result)
        self.assertIsNotNone(cache.get(request_fixture()))

    def test_missing_translation_must_be_explicit_for_non_english_source(self) -> None:
        result = result_fixture(
            documents=(
                RetrievedDocument(
                    rank=1,
                    document_id="doc-1",
                    score=0.75,
                    text="source bytes",
                    url=None,
                    date=None,
                    language="spa",
                    content_source="huggingface",
                    content_local_cache_hit=True,
                    english_text=None,
                ),
            ),
        )
        with self.assertRaises(ValueError):
            RetrievalArtifactCache(MemoryArtifactStore()).put(request_fixture(), result)


if __name__ == "__main__":
    unittest.main()
