from __future__ import annotations

import hashlib
import unittest

import msgspec

from trec_ragtime.retrieval_handoff_cache import (
    HandoffFingerprint,
    RetrievalHandoffArtifactCache,
    handoff_object_key,
    request_for_bundle,
)

from tests.test_nugget_cache import MemoryArtifactStore
from tests.test_retrieval_handoff import RetrievalHandoffTests


def bundle_fixture():
    fixture = RetrievalHandoffTests("test_projection_is_self_contained_and_selected_only")
    fixture.setUp()
    return fixture.project()


class RetrievalHandoffCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = MemoryArtifactStore()
        self.cache = RetrievalHandoffArtifactCache(self.store)
        self.fingerprint = HandoffFingerprint()
        self.bundle = bundle_fixture()
        self.request = request_for_bundle(
            self.bundle, topic_catalog_digest=hashlib.sha256(b"topics").digest()
        )

    def test_object_key_is_namespaced_and_sharded(self) -> None:
        key = handoff_object_key(b"a" * 32, b"b" * 32)
        self.assertEqual(
            key,
            f"retrieval-handoff/v1/{(b'a' * 32).hex()}/62/{(b'b' * 32).hex()}.mpk",
        )

    def test_put_evicts_and_independently_rereads(self) -> None:
        durable = self.cache.put(self.fingerprint, self.request, self.bundle)
        key = self.cache.object_key(self.fingerprint, self.request)
        self.assertEqual(durable, self.bundle)
        self.assertEqual(self.store.evictions, [key])
        self.assertEqual(self.store.reads, [key])
        self.assertEqual(self.cache.get(self.fingerprint, self.request), self.bundle)

    def test_fingerprint_and_request_changes_do_not_hit(self) -> None:
        self.cache.put(self.fingerprint, self.request, self.bundle)
        changed_fp = msgspec.structs.replace(
            self.fingerprint, validation_version="strict-source-slice-v2"
        )
        changed_request = msgspec.structs.replace(
            self.request, topic_catalog_digest=hashlib.sha256(b"other").digest()
        )
        self.assertIsNone(self.cache.get(changed_fp, self.request))
        self.assertIsNone(self.cache.get(self.fingerprint, changed_request))

    def test_tampered_payload_is_never_returned(self) -> None:
        self.cache.put(self.fingerprint, self.request, self.bundle)
        key = self.cache.object_key(self.fingerprint, self.request)
        self.store.objects[key] = self.store.objects[key][:-1]
        self.assertIsNone(self.cache.get(self.fingerprint, self.request))

    def test_bundle_request_disagreement_is_rejected_before_write(self) -> None:
        wrong = msgspec.structs.replace(
            self.request, selected_evidence_digest=b"x" * 32
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.cache.put(self.fingerprint, wrong, self.bundle)
        self.assertEqual(self.store.writes, [])


if __name__ == "__main__":
    unittest.main()
