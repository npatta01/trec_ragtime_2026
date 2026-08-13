from __future__ import annotations

import hashlib
import json
import stat
import tempfile
import unittest
from pathlib import Path

import msgspec

from trec_ragtime.retrieval_handoff_cache import HandoffFingerprint, RetrievalHandoffArtifactCache
from trec_ragtime.retrieval_handoff import _expected_cluster_id, _expected_nugget_id
from trec_ragtime.retrieval_handoff_runner import (
    HandoffWork,
    run_topic,
    run_topics,
    write_handoff_manifest,
)

from tests.test_retrieval_handoff_cache import (
    StrictMemoryArtifactStore,
    bundle_fixture,
)


class RetrievalHandoffRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = StrictMemoryArtifactStore()
        self.cache = RetrievalHandoffArtifactCache(self.store)
        self.fingerprint = HandoffFingerprint()
        self.digest = hashlib.sha256(b"topics").digest()
        self.producer_identity_digest = hashlib.sha256(b"producer-v1").digest()
        self.bundle = bundle_fixture()
        self.work = HandoffWork(
            "2000", self.digest, self.producer_identity_digest, self.bundle
        )

    def test_cold_write_then_warm_resume(self) -> None:
        cold = run_topic(self.work, fingerprint=self.fingerprint, cache=self.cache)
        warm = run_topic(self.work, fingerprint=self.fingerprint, cache=self.cache)
        self.assertEqual(cold.status, "written")
        self.assertEqual(warm.status, "cached")
        self.assertEqual(warm.object_key, cold.object_key)
        self.assertEqual(len(self.store.writes), 1)
        self.assertEqual(
            cold.producer_identity_digest, self.producer_identity_digest.hex()
        )
        self.assertEqual(
            warm.producer_identity_digest, self.producer_identity_digest.hex()
        )

    def test_parallel_results_preserve_input_order(self) -> None:
        second_topic = msgspec.structs.replace(self.bundle.topic, topic_id="2001")
        second_nuggets = tuple(
            msgspec.structs.replace(
                nugget,
                nugget_id=_expected_nugget_id(
                    "2001", nugget.claim, tuple(link.alias for link in nugget.support)
                ),
                cluster_id=_expected_cluster_id("2001", nugget.variants),
            )
            for nugget in self.bundle.nuggets
        )
        second_bundle = msgspec.structs.replace(
            self.bundle, topic=second_topic, nuggets=second_nuggets
        )
        outcomes = run_topics(
            (
                HandoffWork(
                    "2001", self.digest, self.producer_identity_digest, second_bundle
                ),
                self.work,
            ),
            fingerprint=self.fingerprint,
            cache=self.cache,
            concurrency=2,
        )
        self.assertEqual(tuple(row.topic_id for row in outcomes), ("2001", "2000"))
        self.assertTrue(all(row.status == "written" for row in outcomes))

    def test_manifest_is_owner_only_complete_and_sanitized(self) -> None:
        outcome = run_topic(self.work, fingerprint=self.fingerprint, cache=self.cache)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_handoff_manifest(
                path,
                (outcome,),
                topic_catalog_digest=self.digest,
                fingerprint=self.fingerprint,
            )
            payload = json.loads(path.read_text())
            self.assertTrue(payload["complete"])
            self.assertEqual(payload["evidence_count"], 1)
            self.assertEqual(
                payload["topics"][0]["producer_identity_digest"],
                self.producer_identity_digest.hex(),
            )
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertNotIn("grain passage", path.read_text())

    def test_duplicate_work_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            run_topics(
                (self.work, self.work),
                fingerprint=self.fingerprint,
                cache=self.cache,
            )


if __name__ == "__main__":
    unittest.main()
