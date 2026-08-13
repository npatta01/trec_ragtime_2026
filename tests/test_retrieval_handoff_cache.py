from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import msgspec

from trec_ragtime.artifact_store import (
    ArtifactStore,
    ArtifactValidationError,
    ArtifactStoreRemoteError,
)
from trec_ragtime.retrieval_handoff_cache import (
    HandoffEnvelope,
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


class StrictMemoryArtifactStore(MemoryArtifactStore):
    """In-memory store whose strict reader propagates decoder failures."""

    def read_validated_strict(self, object_key, decoder):
        payload = self.read(object_key)
        if payload is None:
            return None
        return decoder(payload)


class RefusingEvictionArtifactStore(ArtifactStore):
    """Artifact store test double that retains a stale L1 candidate."""

    def __init__(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        super().__init__(
            bucket_id="owner/bucket",
            local_dir=Path(self._tempdir.name) / "artifacts",
        )
        self.objects: dict[str, bytes] = {}
        self.writes: list[str] = []

    def write(self, object_key: str, payload: bytes) -> bool:
        self.writes.append(object_key)
        self.objects[object_key] = payload
        self.local_candidate(object_key).write_bytes(payload)
        return True

    def read_validated_strict(self, object_key, decoder):
        candidate = self.local_candidate(object_key)
        if not candidate.exists():
            return None
        return decoder(candidate.read_bytes())

    def _best_effort_remove_candidate(self, candidate: Path) -> bool:
        return False

    def close(self) -> None:
        self._tempdir.cleanup()


class RetrievalHandoffCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = StrictMemoryArtifactStore()
        self.cache = RetrievalHandoffArtifactCache(self.store)
        self.fingerprint = HandoffFingerprint()
        self.producer_identity_digest = hashlib.sha256(b"producer-v1").digest()
        self.bundle = bundle_fixture()
        self.request = request_for_bundle(
            self.bundle,
            topic_catalog_digest=hashlib.sha256(b"topics").digest(),
            producer_identity_digest=self.producer_identity_digest,
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

    def test_put_fails_when_l1_eviction_refuses_and_does_not_reread_stale_l1(self) -> None:
        store = RefusingEvictionArtifactStore()
        self.addCleanup(store.close)
        cache = RetrievalHandoffArtifactCache(store)

        with self.assertRaises(ArtifactStoreRemoteError):
            cache.put(self.fingerprint, self.request, self.bundle)

        self.assertEqual(
            store.writes, [cache.object_key(self.fingerprint, self.request)]
        )

    def test_reads_a_manifest_object_key_without_reconstructing_upstream_inputs(self) -> None:
        self.cache.put(self.fingerprint, self.request, self.bundle)
        key = self.cache.object_key(self.fingerprint, self.request)

        self.assertEqual(
            self.cache.get_by_object_key(
                self.fingerprint,
                key,
                topic_catalog_digest=self.request.topic_catalog_digest,
                producer_identity_digest=self.producer_identity_digest,
            ),
            self.bundle,
        )

    def test_object_key_read_rejects_a_different_topic_catalog(self) -> None:
        self.cache.put(self.fingerprint, self.request, self.bundle)
        key = self.cache.object_key(self.fingerprint, self.request)

        self.assertIsNone(
            self.cache.get_by_object_key(
                self.fingerprint,
                key,
                topic_catalog_digest=hashlib.sha256(b"other topics").digest(),
                producer_identity_digest=self.producer_identity_digest,
            )
        )

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
        changed_producer = msgspec.structs.replace(
            self.request,
            producer_identity_digest=hashlib.sha256(b"producer-v2").digest(),
        )
        self.assertIsNone(self.cache.get(self.fingerprint, changed_producer))

    def test_tampered_payload_is_fatal(self) -> None:
        self.cache.put(self.fingerprint, self.request, self.bundle)
        key = self.cache.object_key(self.fingerprint, self.request)
        self.store.objects[key] = self.store.objects[key][:-1]
        with self.assertRaises(ArtifactValidationError):
            self.cache.get(self.fingerprint, self.request)

    def test_object_key_must_be_under_the_requested_fingerprint(self) -> None:
        self.cache.put(self.fingerprint, self.request, self.bundle)
        key = self.cache.object_key(self.fingerprint, self.request)
        foreign = key.replace(self.fingerprint.digest.hex(), "f" * 64, 1)
        with self.assertRaises(ArtifactValidationError):
            self.cache.get_by_object_key(
                self.fingerprint,
                foreign,
                topic_catalog_digest=self.request.topic_catalog_digest,
                producer_identity_digest=self.producer_identity_digest,
            )

    def test_object_key_request_digest_must_match_the_envelope(self) -> None:
        self.cache.put(self.fingerprint, self.request, self.bundle)
        original_key = self.cache.object_key(self.fingerprint, self.request)
        other_digest = hashlib.sha256(b"other request").digest()
        foreign_key = handoff_object_key(self.fingerprint.digest, other_digest)
        self.store.objects[foreign_key] = self.store.objects[original_key]
        with self.assertRaises(ArtifactValidationError):
            self.cache.get_by_object_key(
                self.fingerprint,
                foreign_key,
                topic_catalog_digest=self.request.topic_catalog_digest,
                producer_identity_digest=self.producer_identity_digest,
            )

    def test_object_key_read_rejects_a_tampered_envelope_identity(self) -> None:
        self.cache.put(self.fingerprint, self.request, self.bundle)
        key = self.cache.object_key(self.fingerprint, self.request)
        envelope = msgspec.msgpack.decode(
            self.store.objects[key], type=HandoffEnvelope
        )
        self.store.objects[key] = msgspec.msgpack.encode(
            msgspec.structs.replace(envelope, request_digest=b"x" * 32)
        )
        with self.assertRaises(ArtifactValidationError):
            self.cache.get_by_object_key(
                self.fingerprint,
                key,
                topic_catalog_digest=self.request.topic_catalog_digest,
                producer_identity_digest=self.producer_identity_digest,
            )

    def test_object_key_read_treats_malformed_payload_as_fatal(self) -> None:
        self.cache.put(self.fingerprint, self.request, self.bundle)
        key = self.cache.object_key(self.fingerprint, self.request)
        self.store.objects[key] = b"not a handoff envelope"
        with self.assertRaises(ArtifactValidationError):
            self.cache.get_by_object_key(
                self.fingerprint,
                key,
                topic_catalog_digest=self.request.topic_catalog_digest,
                producer_identity_digest=self.producer_identity_digest,
            )

    def test_object_key_read_accepts_a_topic_producer_mapping(self) -> None:
        self.cache.put(self.fingerprint, self.request, self.bundle)
        key = self.cache.object_key(self.fingerprint, self.request)
        self.assertEqual(
            self.cache.get_by_object_key(
                self.fingerprint,
                key,
                topic_catalog_digest=self.request.topic_catalog_digest,
                producer_identity_digests={
                    self.bundle.topic.topic_id: self.producer_identity_digest
                },
            ),
            self.bundle,
        )

    def test_object_key_read_missing_topic_mapping_is_a_miss(self) -> None:
        self.cache.put(self.fingerprint, self.request, self.bundle)
        key = self.cache.object_key(self.fingerprint, self.request)
        self.assertIsNone(
            self.cache.get_by_object_key(
                self.fingerprint,
                key,
                topic_catalog_digest=self.request.topic_catalog_digest,
                producer_identity_digests={"other-topic": self.producer_identity_digest},
            )
        )

    def test_object_key_read_with_a_different_producer_is_a_miss(self) -> None:
        self.cache.put(self.fingerprint, self.request, self.bundle)
        key = self.cache.object_key(self.fingerprint, self.request)
        self.assertIsNone(
            self.cache.get_by_object_key(
                self.fingerprint,
                key,
                topic_catalog_digest=self.request.topic_catalog_digest,
                producer_identity_digest=hashlib.sha256(b"producer-v2").digest(),
            )
        )

    def test_object_key_read_requires_exactly_one_producer_identity(self) -> None:
        self.cache.put(self.fingerprint, self.request, self.bundle)
        key = self.cache.object_key(self.fingerprint, self.request)
        with self.assertRaises(ValueError):
            self.cache.get_by_object_key(
                self.fingerprint,
                key,
                topic_catalog_digest=self.request.topic_catalog_digest,
            )
        with self.assertRaises(ValueError):
            self.cache.get_by_object_key(
                self.fingerprint,
                key,
                topic_catalog_digest=self.request.topic_catalog_digest,
                producer_identity_digest=self.producer_identity_digest,
                producer_identity_digests={
                    self.bundle.topic.topic_id: self.producer_identity_digest
                },
            )

    def test_object_key_read_rejects_a_malformed_producer_digest(self) -> None:
        self.cache.put(self.fingerprint, self.request, self.bundle)
        key = self.cache.object_key(self.fingerprint, self.request)
        with self.assertRaises(ArtifactValidationError):
            self.cache.get_by_object_key(
                self.fingerprint,
                key,
                topic_catalog_digest=self.request.topic_catalog_digest,
                producer_identity_digest=b"short",
            )

    def test_bundle_request_disagreement_is_rejected_before_write(self) -> None:
        wrong = msgspec.structs.replace(
            self.request, selected_evidence_digest=b"x" * 32
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.cache.put(self.fingerprint, wrong, self.bundle)
        self.assertEqual(self.store.writes, [])


if __name__ == "__main__":
    unittest.main()
