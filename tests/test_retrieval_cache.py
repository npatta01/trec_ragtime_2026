from __future__ import annotations

import math
import dataclasses
import json
import pickle
import unittest
from typing import Any, Callable

import msgspec
import trec_ragtime as public_api

from trec_ragtime.artifact_store import ArtifactValidationError
from trec_ragtime.cache import ResponseCache, base_url_digest
from trec_ragtime.models import (
    HydrationFailure,
    PipelineResult,
    RetrievalResult,
    RetrievedDocument,
    SearchHit,
)
from trec_ragtime.retriever import RagtimeRetriever
from trec_ragtime.retrieval_cache import (
    RETRIEVAL_SCHEMA_VERSION,
    RetrievalArtifactCache,
    RetrievalArtifactReference,
    RetrievalEnvelope,
    RetrievalRequestIdentity,
    RetrievalRequestWire,
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
    def test_retriever_identity_builder_and_cache_only_lookup_are_pure(self) -> None:
        class NoWorkClient:
            base_url = "https://api.example.test"

            def __init__(self) -> None:
                self.search_calls = 0

            def search(self, *args: Any, **kwargs: Any) -> None:
                self.search_calls += 1
                raise AssertionError("cache-only lookup called organizer search")

        class NoWorkCorpus:
            backend_name = "local"
            corpus_identity = request_fixture().corpus_identity
            corpus_generation = 0

            def __init__(self) -> None:
                self.calls = 0

            def enrich_documents(self, *args: Any, **kwargs: Any) -> None:
                self.calls += 1
                raise AssertionError("cache-only lookup hydrated corpus documents")

            def close(self) -> None:
                return None

        client = NoWorkClient()
        corpus = NoWorkCorpus()
        retriever = RagtimeRetriever(client, corpus, retrieval_cache=RetrievalArtifactCache(MemoryArtifactStore()))
        first = retriever.build_retrieval_identity(
            "¿qué ocurrió? 中文",
            pipeline="ragtime2%20",
            query_context={"kind": "control_a"},
        )
        second = retriever.build_retrieval_identity(
            "¿qué ocurrió? 中文",
            pipeline="ragtime2%20",
            query_context={"kind": "baseline_b"},
        )
        self.assertNotEqual(first.object_key, second.object_key)
        self.assertIsNone(
            retriever.get_cached_retrieval(
                "¿qué ocurrió? 中文",
                pipeline="ragtime2%20",
                query_context={"kind": "control_a"},
            )
        )
        self.assertEqual(client.search_calls, 0)
        self.assertEqual(corpus.calls, 0)

    def test_non_integer_or_boolean_schema_versions_are_rejected_everywhere(self) -> None:
        for version in (True, 1.0, "1"):
            with self.subTest(version=version):
                with self.assertRaises((TypeError, ValueError)):
                    request_fixture(schema_version=version)
                with self.assertRaises((TypeError, ValueError)):
                    retriever_fingerprint(
                        request_fixture().base_url,
                        request_fixture().corpus_identity,
                        schema_version=version,
                    )

    def test_non_integer_or_boolean_hit_ranks_are_rejected_before_write(self) -> None:
        store = MemoryArtifactStore()
        search = PipelineResult(
            query="¿qué ocurrió? 中文",
            pipeline="ragtime2%20",
            hits=(SearchHit(rank=1.0, document_id="doc-1", score=0.75),),
            processed=True,
            local_cache_hit=True,
            remote_cached=None,
            server_timestamp=12.0,
            cache_stored_at=13.0,
        )
        result = result_fixture(search_result=search)
        with self.assertRaises((TypeError, ValueError)):
            RetrievalArtifactCache(store).put(request_fixture(), result)
        self.assertEqual(store.writes, [])

    def test_duplicate_top_level_and_nested_map_keys_are_rejected(self) -> None:
        store = MemoryArtifactStore()
        cache = RetrievalArtifactCache(store)
        request = request_fixture()
        cache.put(request, result_fixture())
        key, payload = store.writes[0]

        # A fixmap with one extra schema_version pair is accepted by a normal
        # last-wins decoder, but must fail canonical re-encoding.
        self.assertEqual(payload[0], 0x85)
        top_level_duplicate = (
            bytes([0x86])
            + payload[1:]
            + msgspec.msgpack.encode("schema_version")
            + msgspec.msgpack.encode(1)
        )
        store.objects[key] = top_level_duplicate
        with self.assertRaises(ArtifactValidationError):
            cache.get(request)

        cache.put(request, result_fixture())
        key, payload = store.writes[-1]
        # Add two entries with the same key inside the nested ``extra`` map.
        original = b"\xa5extra\x80"
        replacement = b"\xa5extra\x82\xa1a\x01\xa1a\x02"
        self.assertIn(original, payload)
        store.objects[key] = payload.replace(original, replacement, 1)
        with self.assertRaises(ArtifactValidationError):
            cache.get(request)

    def test_identity_nested_context_and_backend_policy_are_deeply_immutable(self) -> None:
        request = request_fixture(
            query_context={"nested": {"ordered": ["a", "b"]}},
            backend_policy={"policy": {"strict": True}},
        )
        before = (request.request_digest, request.retriever_fingerprint, request.object_key)
        with self.assertRaises(TypeError):
            request.query_context["nested"]["ordered"] += ("c",)
        with self.assertRaises(TypeError):
            request.backend_policy["policy"]["strict"] = False
        self.assertEqual(before, (request.request_digest, request.retriever_fingerprint, request.object_key))

    def test_identity_mapping_blocks_normal_mutators_and_source_aliases(self) -> None:
        source_context = {"nested": {"ordered": ["a"]}}
        source_backend = {"policy": {"strict": True}}
        request = request_fixture(
            query_context=source_context,
            backend_policy=source_backend,
        )
        before = (request.request_digest, request.retriever_fingerprint, request.object_key)
        source_context["nested"]["ordered"].append("caller-mutation")
        source_backend["policy"]["strict"] = False
        self.assertEqual(before, (request.request_digest, request.retriever_fingerprint, request.object_key))

        mapping = request.query_context
        mutators = (
            lambda: mapping.__init__({"changed": 1}),
            lambda: mapping.__setitem__("changed", 1),
            lambda: mapping.__delitem__("nested"),
            lambda: mapping.clear(),
            lambda: mapping.pop("nested"),
            lambda: mapping.popitem(),
            lambda: mapping.setdefault("changed", 1),
            lambda: mapping.update({"changed": 1}),
            lambda: mapping.__ior__({"changed": 1}),
        )
        for mutate in mutators:
            with self.subTest(mutator=mutate):
                with self.assertRaises(TypeError):
                    mutate()
        self.assertEqual(before, (request.request_digest, request.retriever_fingerprint, request.object_key))

    def test_identity_fields_remain_serializable_across_public_boundaries(self) -> None:
        request = request_fixture(
            query_context={"nested": {"ordered": ["a", "b"]}},
            backend_policy={"policy": {"strict": True}},
        )
        before = (request.request_digest, request.retriever_fingerprint, request.object_key)
        json.dumps(request.query_context)
        json.dumps(request.backend_policy)
        msgspec.msgpack.encode(request.query_context)
        msgspec.msgpack.encode(request.backend_policy)
        record = dataclasses.asdict(request)
        json.dumps(record)
        restored = pickle.loads(pickle.dumps(request))
        self.assertEqual(before, (restored.request_digest, restored.retriever_fingerprint, restored.object_key))
        with self.assertRaises(TypeError):
            restored.query_context["nested"]["ordered"] += ("c",)

    def test_invalid_language_and_provenance_combinations_are_rejected(self) -> None:
        invalid_language = result_fixture(
            documents=(
                RetrievedDocument(
                    rank=1,
                    document_id="doc-1",
                    score=0.75,
                    text="source bytes",
                    url=None,
                    date=None,
                    language="martian",
                    content_source="huggingface",
                    content_local_cache_hit=True,
                    english_text="English translation",
                ),
            )
        )
        invalid_source = result_fixture(
            documents=(
                RetrievedDocument(
                    rank=1,
                    document_id="doc-1",
                    score=0.75,
                    text="source bytes",
                    url=None,
                    date=None,
                    language="spa",
                    content_source="foreign_backend",
                    content_local_cache_hit=True,
                    english_text="English translation",
                ),
            )
        )
        for invalid in (invalid_language, invalid_source):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    RetrievalArtifactCache(MemoryArtifactStore()).put(request_fixture(), invalid)

        # The private /content fallback has source text but no corpus language
        # or translation metadata and is a valid cacheable retrieval status.
        fallback = result_fixture(
            documents=(
                RetrievedDocument(
                    rank=1,
                    document_id="doc-1",
                    score=0.75,
                    text="source bytes",
                    url=None,
                    date=None,
                    language=None,
                    content_source="ragtime_api",
                    content_local_cache_hit=True,
                    english_text=None,
                ),
            ),
            missing_translation_ids=("doc-1",),
        )
        cache = RetrievalArtifactCache(MemoryArtifactStore())
        cache.put(request_fixture(), fallback)

    def test_request_wire_is_publicly_exported(self) -> None:
        self.assertIs(public_api.RetrievalRequestWire, RetrievalRequestWire)

    def test_base_url_validation_matches_response_cache_identity(self) -> None:
        response_cache = ResponseCache(MemoryArtifactStore())
        response_request = {"query": "¿qué ocurrió? 中文", "pipeline": "ragtime2%20"}
        invalid_base_urls = (
            "not-a-url",
            "ftp://api.example.test",
            "https://",
            "https://api.example.test?",
            "https://api.example.test#",
            "https://api.example.test:",
            "https://api.example.test:/path",
            "https://:443/path",
            "https://api.example.test:invalid/path",
            "https://[::1/path",
            "https://api.example.test /path",
        )
        for base_url in invalid_base_urls:
            with self.subTest(base_url=base_url):
                with self.assertRaises(ValueError):
                    response_cache.object_key(
                        base_url=base_url,
                        operation="pipeline",
                        request=response_request,
                    )
                with self.assertRaises(ValueError):
                    request_fixture(base_url=base_url)

        request = request_fixture(base_url="https://api.example.test/")
        self.assertEqual(request.base_url_digest, base_url_digest("https://api.example.test"))

        path = "https://api.example.test/api/v1"
        path_request = request_fixture(base_url=f"{path}///")
        self.assertEqual(path_request.base_url, path)
        self.assertEqual(path_request.base_url_digest, base_url_digest(path))
        self.assertEqual(
            response_cache.object_key(
                base_url=f"{path}///",
                operation="pipeline",
                request=response_request,
            ),
            response_cache.object_key(
                base_url=path,
                operation="pipeline",
                request=response_request,
            ),
        )

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

    def test_nonempty_operational_hydration_failure_is_rejected(self) -> None:
        result = result_fixture(
            hydration_failures=(
                # A complete retrieval artifact never hides operational failures.
                # This is a valid failure value, so completeness is the rejection.
                HydrationFailure(split="eng", message="timeout"),
            )
        )
        with self.assertRaises(ValueError):
            RetrievalArtifactCache(MemoryArtifactStore()).put(request_fixture(), result)

    def test_nonempty_malformed_hydration_failure_is_a_type_error(self) -> None:
        result = result_fixture(hydration_failures=("not-a-hydration-failure",))
        with self.assertRaises(TypeError):
            RetrievalArtifactCache(MemoryArtifactStore()).put(request_fixture(), result)

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
