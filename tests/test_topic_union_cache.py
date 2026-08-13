from __future__ import annotations

import unittest

from trec_ragtime.topic_union_cache import (
    TopicScoreArtifactCache,
    TopicScoreFingerprint,
    TopicScoreRequest,
)
from trec_ragtime.topic_union_scoring import (
    InformationNeed,
    RetrievalSource,
    ScoredTopicDocument,
    TopicScoreResult,
)


class MemoryStore:
    def __init__(self) -> None:
        self.rows: dict[str, bytes] = {}

    def read_validated(self, key: str, decoder):
        payload = self.rows.get(key)
        return None if payload is None else decoder(payload)

    def write(self, key: str, payload: bytes) -> bool:
        self.rows[key] = payload
        return True

    def evict_local(self, key: str) -> None:
        del key


def fingerprint(**changes: str) -> TopicScoreFingerprint:
    values = {
        "scorer_fingerprint": "scorer-v1",
        "chunker_fingerprint": "chunker-v1",
        "representation": "report_text",
    }
    values.update(changes)
    return TopicScoreFingerprint(**values)


def request(**changes) -> TopicScoreRequest:
    values = {
        "topic_id": "2000",
        "plan_sha256": "a" * 64,
        "needs": (InformationNeed("N1", "need one"),),
        "input_object_keys": ("retrieval/v1/a",),
    }
    values.update(changes)
    return TopicScoreRequest(**values)


def result() -> TopicScoreResult:
    source = RetrievalSource("T1", "query", "retrieval/v1/a", 1, 0.4)
    return TopicScoreResult(
        topic_id="2000",
        representation="report_text",
        needs=(InformationNeed("N1", "need one"),),
        input_object_keys=("retrieval/v1/a",),
        unique_document_count=1,
        passage_count=1,
        documents=(
            ScoredTopicDocument("doc-a", 0.8, "N1", 0, 5, b"x" * 32, (source,)),
        ),
        scorer_fingerprint="scorer-v1",
        chunker_fingerprint="chunker-v1",
    )


class TopicUnionCacheTests(unittest.TestCase):
    def test_identity_changes_for_result_affecting_inputs(self) -> None:
        cache = TopicScoreArtifactCache(MemoryStore())
        baseline = cache.object_key(fingerprint(), request())
        self.assertNotEqual(baseline, cache.object_key(fingerprint(scorer_fingerprint="s2"), request()))
        self.assertNotEqual(baseline, cache.object_key(fingerprint(chunker_fingerprint="c2"), request()))
        self.assertNotEqual(baseline, cache.object_key(fingerprint(representation="canonical"), request()))
        self.assertNotEqual(
            baseline,
            cache.object_key(
                fingerprint(), request(needs=(InformationNeed("N1", "changed"),))
            ),
        )
        self.assertNotEqual(
            baseline,
            cache.object_key(
                fingerprint(), request(input_object_keys=("retrieval/v1/b",))
            ),
        )

    def test_complete_result_round_trips_canonically(self) -> None:
        store = MemoryStore()
        cache = TopicScoreArtifactCache(store)
        item = result()
        self.assertTrue(cache.put(fingerprint(), request(), item))
        self.assertEqual(cache.get(fingerprint(), request()), item)

    def test_result_that_does_not_match_request_is_rejected(self) -> None:
        cache = TopicScoreArtifactCache(MemoryStore())
        with self.assertRaisesRegex(ValueError, "does not match"):
            cache.put(
                fingerprint(), request(topic_id="2001"), result()
            )


if __name__ == "__main__":
    unittest.main()
