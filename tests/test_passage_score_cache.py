from __future__ import annotations

import hashlib
import math
import multiprocessing
import subprocess
import sys
import threading
import unittest
from collections.abc import Callable, Mapping, Sequence
from unittest import mock

import msgspec
import trec_ragtime.passage_score_cache as passage_score_cache_module

from trec_ragtime.artifact_store import (
    ArtifactStoreRemoteError,
    ArtifactValidationError,
)
from trec_ragtime.passage_score_cache import (
    CacheOnlyPassageScorer,
    CachedPassageScorer,
    PassageScoreArtifactValidationError,
    PassageScoreCache,
    decode_score_map,
    encode_score_map,
    query_hash,
    score_object_key,
)
from trec_ragtime.passage_scoring import LocalCrossEncoderConfig
from trec_ragtime.passages import Passage, sha256_bytes


class FakeArtifactStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.read_calls: list[str] = []
        self.read_validated_calls: list[str] = []
        self.write_calls: list[tuple[str, bytes]] = []

    def read(self, key: str) -> bytes | None:
        self.read_calls.append(key)
        return self.objects.get(key)

    def read_validated(
        self,
        key: str,
        decoder: Callable[[bytes], object],
    ) -> object | None:
        self.read_validated_calls.append(key)
        payload = self.objects.get(key)
        return None if payload is None else decoder(payload)

    def write(self, key: str, payload: bytes) -> bool:
        self.write_calls.append((key, payload))
        self.objects[key] = payload
        return True

    def evict_local(self, key: str) -> None:
        self.objects.pop(key, None)


class FailingArtifactStore(FakeArtifactStore):
    def __init__(
        self,
        *,
        read_error: BaseException | None = None,
        write_error: BaseException | None = None,
    ) -> None:
        super().__init__()
        self.read_error = read_error
        self.write_error = write_error

    def read_validated(
        self,
        key: str,
        decoder: Callable[[bytes], object],
    ) -> object | None:
        self.read_validated_calls.append(key)
        if self.read_error is not None:
            raise self.read_error
        payload = self.objects.get(key)
        return None if payload is None else decoder(payload)

    def write(self, key: str, payload: bytes) -> bool:
        if self.write_error is not None:
            raise self.write_error
        return super().write(key, payload)


class UnsuccessfulWriteArtifactStore(FakeArtifactStore):
    def write(self, key: str, payload: bytes) -> bool:
        self.write_calls.append((key, payload))
        return False


class TruthyNonBooleanWriteArtifactStore(FakeArtifactStore):
    def write(self, key: str, payload: bytes) -> object:
        self.write_calls.append((key, payload))
        return 1


class RecordingScorer:
    def __init__(
        self,
        fingerprint: str = "model-v1",
        *,
        failure: BaseException | None = None,
        response: Mapping[bytes, object] | None = None,
    ) -> None:
        self.fingerprint = fingerprint
        self.failure = failure
        self.response = response
        self.calls: list[tuple[str, tuple[bytes, ...]]] = []

    def score(self, query: str, passages: Sequence[Passage]) -> Mapping[bytes, object]:
        self.calls.append((query, tuple(passage.content_hash for passage in passages)))
        if self.failure is not None:
            raise self.failure
        if self.response is not None:
            return self.response
        return {
            passage.content_hash: float(index + 1)
            for index, passage in enumerate(passages)
        }


class BarrierScorer(RecordingScorer):
    def __init__(
        self,
        barrier: threading.Barrier,
        response: Mapping[bytes, object],
    ) -> None:
        super().__init__(response=response)
        self._barrier = barrier

    def score(self, query: str, passages: Sequence[Passage]) -> Mapping[bytes, object]:
        response = super().score(query, passages)
        try:
            self._barrier.wait(timeout=2.0)
        except threading.BrokenBarrierError:
            pass
        return response


class BlockingScorer(RecordingScorer):
    def __init__(self, blocked_queries: Sequence[str]) -> None:
        super().__init__()
        self.entered = {query: threading.Event() for query in blocked_queries}
        self.release = {query: threading.Event() for query in blocked_queries}
        self.second_call_entered = threading.Event()
        self._call_count_lock = threading.Lock()
        self._call_counts: dict[str, int] = {}

    def score(self, query: str, passages: Sequence[Passage]) -> Mapping[bytes, object]:
        response = super().score(query, passages)
        with self._call_count_lock:
            call_count = self._call_counts.get(query, 0) + 1
            self._call_counts[query] = call_count
            if call_count == 2:
                self.second_call_entered.set()
        self.entered[query].set()
        if not self.release[query].wait(timeout=5.0):
            raise AssertionError(f"scoring was not released for {query!r}")
        return response


class ProcessBarrierScorer(RecordingScorer):
    def __init__(self, barrier: object) -> None:
        super().__init__()
        self._barrier = barrier

    def score(self, query: str, passages: Sequence[Passage]) -> Mapping[bytes, object]:
        response = super().score(query, passages)
        getattr(self._barrier, "wait")(timeout=5.0)
        return response


class ConcurrentFakeArtifactStore(FakeArtifactStore):
    def __init__(
        self,
        objects: object,
        update_lock: object,
        write_barrier: object,
    ) -> None:
        super().__init__()
        self.objects = objects  # type: ignore[assignment]
        self._update_lock = update_lock
        self._write_barrier = write_barrier

    def write(self, key: str, payload: bytes) -> bool:
        getattr(self._write_barrier, "wait")(timeout=5.0)
        return super().write(key, payload)

    def atomic_update(
        self,
        key: str,
        update: Callable[[bytes | None], bytes],
    ) -> bytes:
        with self._update_lock:  # type: ignore[operator]
            current = self.objects.get(key)
            payload = update(current)
            if not isinstance(payload, bytes):
                raise TypeError("atomic update must return bytes")
            self.objects[key] = payload
            return payload


def _score_in_process(
    store: ConcurrentFakeArtifactStore,
    scorer_barrier: object,
    passage: Passage,
) -> None:
    _cache(store, ProcessBarrierScorer(scorer_barrier)).score(
        "shared query", (passage,)
    )


def _passage(
    text: str,
    *,
    document_id: str = "doc",
    representation: str = "canonical",
) -> Passage:
    return Passage(
        document_id=document_id,
        representation=representation,  # type: ignore[arg-type]
        language="eng",
        text=text,
        start_char=0,
        end_char=len(text),
        boundary="sentence",
        content_hash=sha256_bytes(text),
    )


def _report_passage(text: str, *, document_id: str = "doc") -> Passage:
    return Passage(
        document_id=document_id,
        representation="report_text",
        language="eng",
        text=text,
        start_char=0,
        end_char=len(text),
        boundary="sentence",
        content_hash=sha256_bytes(text),
    )


def _cache(
    store: FakeArtifactStore,
    scorer: RecordingScorer,
    *,
    chunker_fingerprint: str = "chunker-v1",
    representation: str = "canonical",
) -> PassageScoreCache:
    return PassageScoreCache(
        store,
        scorer,
        chunker_fingerprint=chunker_fingerprint,
        representation=representation,
    )


def _legacy_v2_fingerprint(config: LocalCrossEncoderConfig) -> str:
    material = msgspec.msgpack.encode(
        (
            "local-cross-encoder-fingerprint-v1",
            config.schema_version,
            config.model_name,
            config.revision,
            config.tokenizer_name,
            config.tokenizer_revision,
            config.sentence_transformers_version,
            config.transformers_version,
            config.tokenizers_version,
            config.torch_version,
            config.effective_device,
            config.effective_precision,
            config.batch_size,
            config.max_length,
            config.query_format,
            config.query_policy,
        )
    )
    return "local-cross-encoder-v2/" + hashlib.sha256(material).hexdigest()


class PassageScoreCacheTests(unittest.TestCase):
    def test_cache_only_scorer_has_identity_and_refuses_compute(self) -> None:
        scorer = CacheOnlyPassageScorer("model-v1")
        self.assertEqual(scorer.fingerprint, "model-v1")
        with self.assertRaisesRegex(
            PassageScoreArtifactValidationError, "cached passage scores required"
        ):
            scorer.score("secret query", ())

    def test_key_and_map_bytes_are_stable_across_subprocesses(self) -> None:
        code = """
from trec_ragtime.passage_score_cache import encode_score_map, score_object_key
scores = {b'b' * 32: 2, b'a' * 32: 1}
print(score_object_key('exact query', 'model-v1', 'chunker-v1', 'canonical'))
print(encode_score_map(scores).hex())
"""
        expected = subprocess.check_output([sys.executable, "-c", code], text=True)
        actual = subprocess.check_output([sys.executable, "-c", code], text=True)

        self.assertEqual(actual, expected)
        self.assertEqual(expected.count("passage-scores/v1/"), 1)
        self.assertNotIn("exact query", expected)

    def test_query_hash_uses_exact_utf8_bytes(self) -> None:
        self.assertNotEqual(query_hash("café"), query_hash("cafe\u0301"))
        self.assertNotEqual(
            score_object_key("query", "model-v1", "chunker-v1", "canonical"),
            score_object_key("query ", "model-v1", "chunker-v1", "canonical"),
        )

    def test_score_identity_uses_schema_constant_without_rotating_current_v1_key(
        self,
    ) -> None:
        key_v1 = score_object_key("exact query", "model-v1", "chunker-v1", "canonical")

        with mock.patch.object(
            passage_score_cache_module,
            "CACHE_SCHEMA_VERSION",
            2,
        ):
            key_v2 = score_object_key(
                "exact query", "model-v1", "chunker-v1", "canonical"
            )

        self.assertEqual(passage_score_cache_module.CACHE_SCHEMA_VERSION, 1)
        self.assertEqual(
            key_v1,
            "passage-scores/v1/"
            "bc1c48d73ed13a576ceffd98f086462649e21723071ce688e631def66ecdd89e/"
            "38/382a3aa4e4212f03dfe6584350f40bc3772f5c3544e10373375ee6c70083e5db.mpk",
        )
        self.assertTrue(key_v1.startswith("passage-scores/v1/"))
        self.assertNotEqual(key_v1, key_v2)

    def test_scorer_chunker_and_representation_isolate_objects(self) -> None:
        keys = {
            score_object_key("q", "model-v1", "chunker-v1", "canonical"),
            score_object_key("q", "model-v2", "chunker-v1", "canonical"),
            score_object_key("q", "model-v1", "chunker-v2", "canonical"),
            score_object_key("q", "model-v1", "chunker-v1", "report_text"),
        }
        self.assertEqual(len(keys), 4)

    def test_corrected_mixedbread_policy_rotates_durable_object_identity(self) -> None:
        policy = LocalCrossEncoderConfig(
            model_name="mixedbread-ai/mxbai-rerank-base-v2",
            revision="3ea9d4dffa7d12a4f366be8e275c349de9fc9865",
            tokenizer_name="mixedbread-ai/mxbai-rerank-base-v2",
            tokenizer_revision="3ea9d4dffa7d12a4f366be8e275c349de9fc9865",
        )
        corrected = _cache(
            FakeArtifactStore(),
            RecordingScorer(fingerprint=policy.fingerprint),
        ).object_key("exact query")
        legacy = score_object_key(
            "exact query",
            _legacy_v2_fingerprint(policy),
            "chunker-v1",
            "canonical",
        )

        self.assertNotEqual(corrected, legacy)

    def test_map_codec_normalizes_scores_and_orders_binary_keys(self) -> None:
        first = b"a" * 32
        second = b"b" * 32

        payload = encode_score_map({second: 2, first: 1.5})

        self.assertEqual(decode_score_map(payload), {first: 1.5, second: 2.0})
        self.assertEqual(payload, encode_score_map({first: 1.5, second: 2.0}))

    def test_empty_request_does_not_read_score_or_write_an_empty_artifact(self) -> None:
        store = FakeArtifactStore()
        scorer = RecordingScorer()

        self.assertEqual(_cache(store, scorer).score("query", ()), {})
        self.assertEqual(store.read_validated_calls, [])
        self.assertEqual(store.write_calls, [])
        self.assertEqual(scorer.calls, [])

    def test_requested_passage_identity_is_validated_before_storage(self) -> None:
        store = FakeArtifactStore()
        scorer = RecordingScorer()

        with self.assertRaises(PassageScoreArtifactValidationError):
            _cache(store, scorer).score("q", (_report_passage("x"),))

        self.assertEqual(store.read_validated_calls, [])
        self.assertEqual(scorer.calls, [])

    def test_cold_miss_scores_unique_passages_and_writes_one_complete_map(self) -> None:
        store = FakeArtifactStore()
        scorer = RecordingScorer()
        first = _passage("first", document_id="first")
        duplicate = _passage("first", document_id="duplicate")
        second = _passage("second", document_id="second")

        result = _cache(store, scorer).score("query", (first, duplicate, second))

        self.assertEqual(result, {first.content_hash: 1.0, second.content_hash: 2.0})
        self.assertEqual(
            scorer.calls, [("query", (first.content_hash, second.content_hash))]
        )
        self.assertEqual(len(store.read_validated_calls), 1)
        self.assertEqual(store.read_calls, [])
        self.assertEqual(len(store.write_calls), 1)
        self.assertEqual(
            decode_score_map(store.write_calls[0][1]),
            {first.content_hash: 1.0, second.content_hash: 2.0},
        )

    def test_complete_warm_hit_returns_only_requested_hashes(self) -> None:
        store = FakeArtifactStore()
        first = _passage("first", document_id="first")
        second = _passage("second", document_id="second")
        _cache(store, RecordingScorer()).score("query", (first, second))
        store.write_calls.clear()
        warm_scorer = RecordingScorer(failure=AssertionError("warm hit scored"))

        result = _cache(store, warm_scorer).score("query", (second,))

        self.assertEqual(result, {second.content_hash: 2.0})
        self.assertEqual(warm_scorer.calls, [])
        self.assertEqual(store.write_calls, [])

    def test_missing_only_scoring_rewrites_the_complete_union(self) -> None:
        store = FakeArtifactStore()
        scorer = RecordingScorer()
        first = _passage("first", document_id="first")
        second = _passage("second", document_id="second")
        third = _passage("third", document_id="third")
        cache = _cache(store, scorer)

        cache.score("query", (first, second))
        first_payload = store.write_calls[0][1]
        store.write_calls.clear()
        result = cache.score("query", (second, third))

        self.assertEqual(result, {second.content_hash: 2.0, third.content_hash: 1.0})
        self.assertEqual(
            scorer.calls,
            [
                ("query", (first.content_hash, second.content_hash)),
                ("query", (third.content_hash,)),
            ],
        )
        self.assertEqual(len(store.write_calls), 1)
        self.assertNotEqual(first_payload, store.write_calls[0][1])
        self.assertEqual(
            decode_score_map(store.write_calls[0][1]),
            {
                first.content_hash: 1.0,
                second.content_hash: 2.0,
                third.content_hash: 1.0,
            },
        )

    def test_different_query_keys_can_score_concurrently(self) -> None:
        store = FakeArtifactStore()
        first = _passage("first", document_id="first")
        second = _passage("second", document_id="second")
        scorer = BlockingScorer(("query-a", "query-b"))
        cache_a = _cache(store, scorer)
        cache_b = _cache(store, scorer)
        results: dict[str, dict[bytes, float]] = {}
        errors: list[BaseException] = []

        def score(
            name: str,
            cache: PassageScoreCache,
            query: str,
            passage: Passage,
        ) -> None:
            try:
                results[name] = cache.score(query, (passage,))
            except BaseException as error:
                errors.append(error)

        threads = [
            threading.Thread(target=score, args=("a", cache_a, "query-a", first)),
            threading.Thread(target=score, args=("b", cache_b, "query-b", second)),
        ]
        for thread in threads:
            thread.start()
        try:
            self.assertTrue(scorer.entered["query-a"].wait(timeout=2.0))
            self.assertTrue(scorer.entered["query-b"].wait(timeout=2.0))
        finally:
            for release in scorer.release.values():
                release.set()
            for thread in threads:
                thread.join(timeout=3.0)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(
            results,
            {"a": {first.content_hash: 1.0}, "b": {second.content_hash: 1.0}},
        )

    def test_same_query_key_serializes_and_preserves_the_union(self) -> None:
        store = FakeArtifactStore()
        first = _passage("first", document_id="first")
        second = _passage("second", document_id="second")
        scorer = BlockingScorer(("same query",))
        cache_a = _cache(store, scorer)
        cache_b = _cache(store, scorer)
        results: dict[str, dict[bytes, float]] = {}
        errors: list[BaseException] = []

        def score(
            name: str,
            cache: PassageScoreCache,
            passage: Passage,
        ) -> None:
            try:
                results[name] = cache.score("same query", (passage,))
            except BaseException as error:
                errors.append(error)

        first_thread = threading.Thread(target=score, args=("first", cache_a, first))
        second_thread = threading.Thread(target=score, args=("second", cache_b, second))
        first_thread.start()
        self.assertTrue(scorer.entered["same query"].wait(timeout=2.0))
        second_thread.start()
        try:
            self.assertFalse(scorer.second_call_entered.wait(timeout=0.5))
        finally:
            scorer.release["same query"].set()
            first_thread.join(timeout=3.0)
            second_thread.join(timeout=3.0)

        self.assertTrue(
            all(not thread.is_alive() for thread in (first_thread, second_thread))
        )
        self.assertEqual(errors, [])
        self.assertEqual(
            results,
            {
                "first": {first.content_hash: 1.0},
                "second": {second.content_hash: 1.0},
            },
        )
        self.assertEqual(
            decode_score_map(store.objects[cache_a.object_key("same query")]),
            {first.content_hash: 1.0, second.content_hash: 1.0},
        )

    def test_atomic_store_merge_preserves_disjoint_updates_across_processes(
        self,
    ) -> None:
        context = multiprocessing.get_context("spawn")
        with context.Manager() as manager:
            store = ConcurrentFakeArtifactStore(
                manager.dict(),
                context.Lock(),
                context.Barrier(2),
            )
            first = _passage("first", document_id="first")
            second = _passage("second", document_id="second")
            scorer_barrier = context.Barrier(2)
            processes = [
                context.Process(
                    target=_score_in_process,
                    args=(store, scorer_barrier, first),
                ),
                context.Process(
                    target=_score_in_process,
                    args=(store, scorer_barrier, second),
                ),
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=10.0)

            self.assertTrue(all(not process.is_alive() for process in processes))
            self.assertEqual(
                [process.exitcode for process in processes],
                [0, 0],
            )
            key = _cache(store, RecordingScorer()).object_key("shared query")
            self.assertEqual(
                decode_score_map(store.objects[key]),
                {first.content_hash: 1.0, second.content_hash: 1.0},
            )

    def test_concurrent_extensions_keep_complete_union_and_warm_reads_do_not_rescore(
        self,
    ) -> None:
        store = FakeArtifactStore()
        h0 = _passage("h0", document_id="h0")
        h1 = _passage("h1", document_id="h1")
        h2 = _passage("h2", document_id="h2")
        _cache(
            store,
            RecordingScorer(response={h0.content_hash: 10.0}),
        ).score("query", (h0,))
        store.write_calls.clear()

        scorer_barrier = threading.Barrier(2)
        cache_a = _cache(
            store,
            BarrierScorer(scorer_barrier, {h1.content_hash: 11.0}),
        )
        cache_b = _cache(
            store,
            BarrierScorer(scorer_barrier, {h2.content_hash: 12.0}),
        )
        start_barrier = threading.Barrier(3)
        errors: list[BaseException] = []

        def extend(cache: PassageScoreCache, passage: Passage) -> None:
            try:
                start_barrier.wait(timeout=2.0)
                cache.score("query", (passage,))
            except BaseException as error:
                errors.append(error)

        threads = [
            threading.Thread(target=extend, args=(cache_a, h1)),
            threading.Thread(target=extend, args=(cache_b, h2)),
        ]
        for thread in threads:
            thread.start()
        start_barrier.wait(timeout=2.0)
        for thread in threads:
            thread.join(timeout=3.0)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        key = cache_a.object_key("query")
        self.assertEqual(
            decode_score_map(store.objects[key]),
            {
                h0.content_hash: 10.0,
                h1.content_hash: 11.0,
                h2.content_hash: 12.0,
            },
        )

        warm_scorer = RecordingScorer(failure=AssertionError("warm hit rescored"))
        warm_cache = _cache(store, warm_scorer)
        self.assertEqual(
            warm_cache.score("query", (h0, h1, h2)),
            {
                h0.content_hash: 10.0,
                h1.content_hash: 11.0,
                h2.content_hash: 12.0,
            },
        )
        self.assertEqual(warm_scorer.calls, [])

    def test_cached_passage_scorer_delegates_to_one_cache(self) -> None:
        store = FakeArtifactStore()
        scorer = RecordingScorer()
        cache = _cache(store, scorer)
        cached = CachedPassageScorer(cache)
        passage = _passage("passage")

        self.assertEqual(cached.fingerprint, scorer.fingerprint)
        self.assertEqual(cached.cache, cache)
        self.assertEqual(cached.score("q", (passage,)), {passage.content_hash: 1.0})

    def test_old_positional_identity_constructor_is_removed(self) -> None:
        with self.assertRaises(TypeError):
            PassageScoreCache(
                FakeArtifactStore(),
                RecordingScorer(),
                "chunker-v1",  # type: ignore[call-arg]
                "canonical",  # type: ignore[call-arg]
                ArtifactValidationError,  # type: ignore[call-arg]
            )

    def test_malformed_truncated_non_map_nonfinite_and_wrong_key_artifacts_fail(
        self,
    ) -> None:
        passage = _passage("first")
        key = score_object_key("query", "model-v1", "chunker-v1", "canonical")
        valid = msgspec.msgpack.encode({passage.content_hash: 1.0})
        cases = (
            b"not msgpack",
            valid[:-1],
            msgspec.msgpack.encode([passage.content_hash, 1.0]),
            msgspec.msgpack.encode({passage.content_hash: math.nan}),
            msgspec.msgpack.encode({b"wrong": 1.0}),
            msgspec.msgpack.encode({}),
        )

        for raw in cases:
            with self.subTest(raw_prefix=raw[:12]):
                store = FakeArtifactStore()
                store.objects[key] = raw
                scorer = RecordingScorer()

                with self.assertRaises(ArtifactValidationError):
                    _cache(store, scorer).score("query", (passage,))
                self.assertEqual(scorer.calls, [])
                self.assertEqual(store.write_calls, [])

    def test_direct_codec_rejects_non_bytes_non_map_and_bad_values(self) -> None:
        with self.assertRaises(PassageScoreArtifactValidationError):
            decode_score_map(bytearray())  # type: ignore[arg-type]
        with self.assertRaises(PassageScoreArtifactValidationError):
            decode_score_map(msgspec.msgpack.encode({b"x" * 32: math.inf}))
        with self.assertRaises(PassageScoreArtifactValidationError):
            encode_score_map({b"x" * 31: 1.0})  # type: ignore[dict-item]
        with self.assertRaises(PassageScoreArtifactValidationError):
            encode_score_map({b"x" * 32: "not a score"})  # type: ignore[dict-item]

    def test_storage_failures_propagate_without_becoming_misses(self) -> None:
        passage = _passage("first")
        read_store = FailingArtifactStore(read_error=RuntimeError("read failed"))
        scorer = RecordingScorer()

        with self.assertRaisesRegex(RuntimeError, "read failed"):
            _cache(read_store, scorer).score("query", (passage,))
        self.assertEqual(scorer.calls, [])
        self.assertEqual(read_store.write_calls, [])

        write_store = FailingArtifactStore(write_error=RuntimeError("write failed"))
        with self.assertRaisesRegex(RuntimeError, "write failed"):
            _cache(write_store, RecordingScorer()).score("query", (passage,))
        self.assertEqual(write_store.write_calls, [])

    def test_unsuccessful_write_does_not_return_scores_as_cached(self) -> None:
        passage = _passage("first")
        store = UnsuccessfulWriteArtifactStore()
        scorer = RecordingScorer()

        with self.assertRaises(ArtifactStoreRemoteError):
            _cache(store, scorer).score("query", (passage,))

        self.assertEqual(scorer.calls, [("query", (passage.content_hash,))])
        self.assertEqual(len(store.write_calls), 1)
        self.assertEqual(store.objects, {})

    def test_truthy_non_boolean_write_result_is_fatal(self) -> None:
        passage = _passage("first")
        store = TruthyNonBooleanWriteArtifactStore()

        with self.assertRaises(ArtifactStoreRemoteError):
            _cache(store, RecordingScorer()).score("query", (passage,))

        self.assertEqual(len(store.write_calls), 1)
        self.assertEqual(store.objects, {})

    def test_scorer_failure_or_partial_coverage_never_writes(self) -> None:
        first = _passage("first")
        second = _passage("second", document_id="second")

        failed_store = FakeArtifactStore()
        with self.assertRaisesRegex(RuntimeError, "model failed"):
            _cache(
                failed_store,
                RecordingScorer(failure=RuntimeError("model failed")),
            ).score("query", (first, second))
        self.assertEqual(failed_store.write_calls, [])

        partial_store = FakeArtifactStore()
        partial_scorer = RecordingScorer(response={first.content_hash: 1.0})
        with self.assertRaises(ValueError):
            _cache(partial_store, partial_scorer).score("query", (first, second))
        self.assertEqual(partial_store.write_calls, [])


if __name__ == "__main__":
    unittest.main()
