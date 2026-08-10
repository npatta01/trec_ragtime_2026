from __future__ import annotations

import hashlib
import math
import subprocess
import sys
import unittest
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import msgspec

from trec_ragtime.passages import Passage, sha256_bytes
from trec_ragtime.passage_score_cache import (
    CachedPassageScorer,
    PassageScoreArtifactValidationError,
    PassageScoreCache,
    PassageScoreEnvelope,
    decode_score_envelope,
    effective_score_fingerprint,
    encode_score_envelope,
    query_hash,
    score_object_key,
)


class FakeArtifactStore:
    class ArtifactValidationError(Exception):
        pass

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.read_calls: list[str] = []
        self.read_validated_calls: list[str] = []
        self.write_calls: list[tuple[str, bytes]] = []
        self.read_error: BaseException | None = None
        self.write_error: BaseException | None = None

    def read(self, key: str) -> bytes | None:
        self.read_calls.append(key)
        return self.objects.get(key)

    def read_validated(
        self,
        key: str,
        decoder: Callable[[bytes], Any],
    ) -> Any | None:
        self.read_validated_calls.append(key)
        if self.read_error is not None:
            raise self.read_error
        raw = self.objects.get(key)
        return None if raw is None else decoder(raw)

    def write(self, key: str, payload: bytes) -> None:
        if self.write_error is not None:
            raise self.write_error
        self.write_calls.append((key, payload))
        self.objects[key] = payload


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
        return {passage.content_hash: float(index + 1) for index, passage in enumerate(passages)}


def _passage(
    text: str,
    *,
    document_id: str = "doc",
    chunker_fingerprint: str = "chunker-v1",
    representation: str = "canonical",
) -> Passage:
    return Passage(
        document_id=document_id,
        source_content_hash=sha256_bytes(text),
        representation=representation,  # type: ignore[arg-type]
        representation_hash=sha256_bytes(text),
        language="eng",
        chunk_index=0,
        passage_id=sha256_bytes(f"passage:{document_id}:{text}"),
        text=text,
        start_char=0,
        end_char=len(text),
        boundary="paragraph_packed",
        previous_passage_id=None,
        next_passage_id=None,
        content_hash=sha256_bytes(text),
        chunker_fingerprint=chunker_fingerprint,
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
        chunker_fingerprint,
        representation,
        FakeArtifactStore.ArtifactValidationError,
    )


def _coverage_digest(coverage: Sequence[bytes]) -> bytes:
    return hashlib.sha256(msgspec.msgpack.encode(tuple(coverage))).digest()


def _raw_envelope(
    query: str,
    *,
    scorer_fingerprint: str = "model-v1",
    chunker_fingerprint: str = "chunker-v1",
    representation: str = "canonical",
    schema_version: int = 1,
    coverage: Sequence[bytes] = (),
    scores: Mapping[bytes, object] | None = None,
    extra: Mapping[str, object] | None = None,
) -> bytes:
    sorted_coverage = tuple(sorted(coverage))
    fields: dict[str, object] = {
        "schema_version": schema_version,
        "scorer_fingerprint": scorer_fingerprint,
        "chunker_fingerprint": chunker_fingerprint,
        "representation": representation,
        "score_fingerprint": effective_score_fingerprint(
            scorer_fingerprint,
            chunker_fingerprint,
            representation,
            schema_version=schema_version,
        ),
        "query_hash": query_hash(query),
        "coverage": sorted_coverage,
        "coverage_digest": _coverage_digest(sorted_coverage),
        "scores": dict(scores or {}),
    }
    if extra:
        fields.update(extra)
    return msgspec.msgpack.encode(fields)


class PassageScoreCacheTests(unittest.TestCase):
    def test_key_and_encoded_bytes_are_stable_across_subprocesses(self) -> None:
        code = """
import msgspec
from trec_ragtime.passage_score_cache import (
    PassageScoreEnvelope,
    effective_score_fingerprint,
    encode_score_envelope,
    query_hash,
    score_object_key,
)
coverage = (b'a' * 32, b'b' * 32)
envelope = PassageScoreEnvelope(
    schema_version=1,
    scorer_fingerprint='model-v1',
    chunker_fingerprint='chunker-v1',
    representation='canonical',
    score_fingerprint=effective_score_fingerprint('model-v1', 'chunker-v1', 'canonical'),
    query_hash=query_hash('exact query'),
    coverage=coverage,
    coverage_digest=__import__('hashlib').sha256(msgspec.msgpack.encode(coverage)).digest(),
    scores={b'b' * 32: 2.0, b'a' * 32: 1.0},
)
print(score_object_key('exact query', 'model-v1', 'chunker-v1', 'canonical'))
print(encode_score_envelope(envelope).hex())
"""
        expected = subprocess.check_output([sys.executable, "-c", code], text=True)
        actual = subprocess.check_output([sys.executable, "-c", code], text=True)
        self.assertEqual(actual, expected)
        self.assertEqual(expected.count("passage-scores/v1/"), 1)

    def test_query_hash_uses_exact_utf8_bytes_and_only_query_path_identity(self) -> None:
        self.assertNotEqual(query_hash("café"), query_hash("cafe\u0301"))
        self.assertNotEqual(
            score_object_key("query", "model-v1", "chunker-v1", "canonical"),
            score_object_key("query ", "model-v1", "chunker-v1", "canonical"),
        )
        self.assertEqual(
            score_object_key("query", "model-v1", "chunker-v1", "canonical"),
            score_object_key("query", "model-v1", "chunker-v1", "canonical"),
        )

    def test_scorer_chunker_and_representation_fingerprints_isolate_objects(self) -> None:
        variants = {
            score_object_key("q", "model-v1", "chunker-v1", "canonical"),
            score_object_key("q", "model-v2", "chunker-v1", "canonical"),
            score_object_key("q", "model-v1", "chunker-v2", "canonical"),
            score_object_key("q", "model-v1", "chunker-v1", "report_text"),
        }
        self.assertEqual(len(variants), 4)
        self.assertEqual(
            effective_score_fingerprint("model-v1", "chunker-v1", "canonical"),
            effective_score_fingerprint("model-v1", "chunker-v1", "canonical"),
        )

    def test_requested_passage_identity_is_validated_before_storage(self) -> None:
        store = FakeArtifactStore()
        scorer = RecordingScorer()
        cache = _cache(store, scorer)

        with self.assertRaises(PassageScoreArtifactValidationError):
            cache.score("q", (_passage("x", chunker_fingerprint="other"),))
        with self.assertRaises(PassageScoreArtifactValidationError):
            cache.score("q", (_passage("x", representation="report_text"),))
        self.assertEqual(store.read_validated_calls, [])
        self.assertEqual(scorer.calls, [])

    def test_cold_miss_scores_all_unique_passages_and_writes_once(self) -> None:
        store = FakeArtifactStore()
        scorer = RecordingScorer()
        cache = _cache(store, scorer)
        first = _passage("first", document_id="first")
        duplicate = _passage("first", document_id="duplicate")
        second = _passage("second", document_id="second")

        result = cache.score("query", (first, duplicate, second))

        self.assertEqual(result, {first.content_hash: 1.0, second.content_hash: 2.0})
        self.assertEqual(scorer.calls, [("query", (first.content_hash, second.content_hash))])
        self.assertEqual(len(store.write_calls), 1)
        self.assertEqual(store.read_calls, [])

        envelope = decode_score_envelope(store.write_calls[0][1])
        self.assertEqual(envelope.coverage, tuple(sorted((first.content_hash, second.content_hash))))
        self.assertEqual(list(envelope.scores), list(sorted(envelope.scores)))
        self.assertEqual(set(envelope.scores), set(envelope.coverage))

    def test_complete_warm_hit_returns_only_request_keys_without_scorer_or_write(self) -> None:
        store = FakeArtifactStore()
        first_scorer = RecordingScorer()
        first_cache = _cache(store, first_scorer)
        first = _passage("first", document_id="first")
        second = _passage("second", document_id="second")
        first_cache.score("query", (first, second))
        store.write_calls.clear()

        warm_scorer = RecordingScorer(failure=AssertionError("warm hit called scorer"))
        result = _cache(store, warm_scorer).score("query", (second,))

        self.assertEqual(result, {second.content_hash: 2.0})
        self.assertEqual(warm_scorer.calls, [])
        self.assertEqual(store.write_calls, [])

    def test_overlapping_query_sets_reuse_one_object_and_rewrite_complete_union(self) -> None:
        store = FakeArtifactStore()
        scorer = RecordingScorer()
        cache = _cache(store, scorer)
        first = _passage("first", document_id="first")
        second = _passage("second", document_id="second")
        third = _passage("third", document_id="third")

        first_result = cache.score("query", (second, first))
        first_write = store.write_calls[0][1]
        store.write_calls.clear()
        second_result = cache.score("query", (third, second))

        self.assertEqual(first_result, {first.content_hash: 2.0, second.content_hash: 1.0})
        self.assertEqual(second_result, {second.content_hash: 1.0, third.content_hash: 1.0})
        self.assertEqual(
            scorer.calls,
            [
                ("query", (second.content_hash, first.content_hash)),
                ("query", (third.content_hash,)),
            ],
        )
        self.assertEqual(len(store.write_calls), 1)
        union = decode_score_envelope(store.write_calls[0][1])
        self.assertEqual(union.coverage, tuple(sorted((first.content_hash, second.content_hash, third.content_hash))))
        self.assertEqual(
            store.write_calls[0][1],
            encode_score_envelope(union),
        )
        self.assertNotEqual(first_write, store.write_calls[0][1])

    def test_cached_passage_scorer_exposes_scorer_contract(self) -> None:
        store = FakeArtifactStore()
        scorer = RecordingScorer()
        cache = _cache(store, scorer)
        cached = CachedPassageScorer(scorer, cache)
        passage = _passage("passage")

        self.assertEqual(cached.fingerprint, scorer.fingerprint)
        self.assertEqual(cached.score("q", (passage,)), {passage.content_hash: 1.0})

    def test_decode_rejects_strict_unknown_fields(self) -> None:
        raw = _raw_envelope("query", coverage=(b"a" * 32,), scores={b"a" * 32: 1.0}, extra={"unknown": 1})

        with self.assertRaises(PassageScoreArtifactValidationError):
            decode_score_envelope(raw)

    def test_malformed_truncated_and_wrong_schema_objects_are_rejected_without_write(self) -> None:
        first = _passage("first")
        cases = (
            b"not msgpack",
            _raw_envelope("query", coverage=(first.content_hash,), scores={first.content_hash: 1.0})[:-1],
            _raw_envelope("query", schema_version=2, coverage=(first.content_hash,), scores={first.content_hash: 1.0}),
        )
        for raw in cases:
            with self.subTest(raw_prefix=raw[:12]):
                store = FakeArtifactStore()
                key = score_object_key("query", "model-v1", "chunker-v1", "canonical")
                store.objects[key] = raw
                scorer = RecordingScorer()
                cache = _cache(store, scorer)

                with self.assertRaises(FakeArtifactStore.ArtifactValidationError):
                    cache.score("query", (first,))
                self.assertEqual(scorer.calls, [])
                self.assertEqual(store.write_calls, [])

    def test_wrong_query_and_wrong_fingerprint_objects_are_rejected(self) -> None:
        first = _passage("first")
        key = score_object_key("query", "model-v1", "chunker-v1", "canonical")
        cases = (
            _raw_envelope("different", coverage=(first.content_hash,), scores={first.content_hash: 1.0}),
            _raw_envelope("query", scorer_fingerprint="wrong", coverage=(first.content_hash,), scores={first.content_hash: 1.0}),
            _raw_envelope("query", extra={"scorer_fingerprint": "bad\n"}, coverage=(first.content_hash,), scores={first.content_hash: 1.0}),
            _raw_envelope("query", chunker_fingerprint="wrong", coverage=(first.content_hash,), scores={first.content_hash: 1.0}),
            _raw_envelope("query", representation="report_text", coverage=(first.content_hash,), scores={first.content_hash: 1.0}),
        )
        for raw in cases:
            with self.subTest(raw_prefix=raw[:12]):
                store = FakeArtifactStore()
                store.objects[key] = raw
                scorer = RecordingScorer()
                with self.assertRaises(FakeArtifactStore.ArtifactValidationError):
                    _cache(store, scorer).score("query", (first,))
                self.assertEqual(scorer.calls, [])
                self.assertEqual(store.write_calls, [])

    def test_nonfinite_and_incomplete_coverage_objects_are_rejected(self) -> None:
        first = _passage("first")
        second = _passage("second", document_id="second")
        key = score_object_key("query", "model-v1", "chunker-v1", "canonical")
        cases = (
            _raw_envelope("query", coverage=(first.content_hash,), scores={first.content_hash: math.nan}),
            _raw_envelope("query", coverage=(first.content_hash, second.content_hash), scores={first.content_hash: 1.0}),
            _raw_envelope("query", coverage=(first.content_hash,), scores={first.content_hash: 1.0, b"x" * 32: 2.0}),
        )
        for raw in cases:
            with self.subTest(raw_prefix=raw[:12]):
                store = FakeArtifactStore()
                store.objects[key] = raw
                scorer = RecordingScorer()
                with self.assertRaises(FakeArtifactStore.ArtifactValidationError):
                    _cache(store, scorer).score("query", (first, second))
                self.assertEqual(scorer.calls, [])
                self.assertEqual(store.write_calls, [])

    def test_decoder_wraps_domain_validation_with_the_exact_injected_store_exception(self) -> None:
        with self.assertRaises(PassageScoreArtifactValidationError):
            decode_score_envelope(b"bad")

        store = FakeArtifactStore()
        first = _passage("first")
        key = score_object_key("query", "model-v1", "chunker-v1", "canonical")
        store.objects[key] = b"bad"
        with self.assertRaises(FakeArtifactStore.ArtifactValidationError) as raised:
            _cache(store, RecordingScorer()).score("query", (first,))
        self.assertIsInstance(raised.exception.__cause__, PassageScoreArtifactValidationError)
        self.assertIsNot(raised.exception.__class__, PassageScoreArtifactValidationError)

    def test_storage_errors_propagate_and_are_never_treated_as_misses(self) -> None:
        first = _passage("first")
        store = FakeArtifactStore()
        store.read_error = RuntimeError("read failed")
        scorer = RecordingScorer()
        with self.assertRaisesRegex(RuntimeError, "read failed"):
            _cache(store, scorer).score("query", (first,))
        self.assertEqual(scorer.calls, [])
        self.assertEqual(store.write_calls, [])

        store = FakeArtifactStore()
        store.write_error = RuntimeError("write failed")
        with self.assertRaisesRegex(RuntimeError, "write failed"):
            _cache(store, RecordingScorer()).score("query", (first,))
        self.assertEqual(len(store.write_calls), 0)

    def test_scorer_failure_or_partial_response_never_writes(self) -> None:
        first = _passage("first")
        second = _passage("second", document_id="second")
        failing_store = FakeArtifactStore()
        with self.assertRaisesRegex(RuntimeError, "model failed"):
            _cache(
                failing_store,
                RecordingScorer(failure=RuntimeError("model failed")),
            ).score("query", (first, second))
        self.assertEqual(failing_store.write_calls, [])

        partial_store = FakeArtifactStore()
        partial_scorer = RecordingScorer(response={first.content_hash: 1.0})
        with self.assertRaises(ValueError):
            _cache(partial_store, partial_scorer).score("query", (first, second))
        self.assertEqual(partial_store.write_calls, [])


class PassageScoreEnvelopeTests(unittest.TestCase):
    def test_envelope_is_frozen_and_round_trips_canonically(self) -> None:
        coverage = (b"a" * 32, b"b" * 32)
        envelope = PassageScoreEnvelope(
            schema_version=1,
            scorer_fingerprint="model-v1",
            chunker_fingerprint="chunker-v1",
            representation="canonical",
            score_fingerprint=effective_score_fingerprint("model-v1", "chunker-v1", "canonical"),
            query_hash=query_hash("query"),
            coverage=coverage,
            coverage_digest=_coverage_digest(coverage),
            scores={coverage[1]: 2.0, coverage[0]: 1.0},
        )
        with self.assertRaises((AttributeError, TypeError)):
            envelope.schema_version = 2  # type: ignore[misc]
        decoded = decode_score_envelope(encode_score_envelope(envelope))
        self.assertEqual(decoded.coverage, coverage)
        self.assertEqual(list(decoded.scores), list(sorted(decoded.scores)))
        self.assertEqual(encode_score_envelope(envelope), encode_score_envelope(decoded))


if __name__ == "__main__":
    unittest.main()
