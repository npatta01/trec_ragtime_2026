from __future__ import annotations

import hashlib
import json
import stat
import tempfile
import unittest
from pathlib import Path

import msgspec

from trec_ragtime.retrieval_handoff_cache import HandoffFingerprint, RetrievalHandoffArtifactCache
from trec_ragtime.retrieval_handoff_runner import (
    HandoffWork,
    run_topic,
    run_topics,
    write_handoff_manifest,
)

from tests.test_nugget_cache import MemoryArtifactStore
from tests.test_retrieval_handoff_cache import bundle_fixture


class RetrievalHandoffRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = MemoryArtifactStore()
        self.cache = RetrievalHandoffArtifactCache(self.store)
        self.fingerprint = HandoffFingerprint()
        self.digest = hashlib.sha256(b"topics").digest()
        self.bundle = bundle_fixture()
        self.work = HandoffWork("2000", self.digest, self.bundle)

    def test_cold_write_then_warm_resume(self) -> None:
        cold = run_topic(self.work, fingerprint=self.fingerprint, cache=self.cache)
        warm = run_topic(self.work, fingerprint=self.fingerprint, cache=self.cache)
        self.assertEqual(cold.status, "written")
        self.assertEqual(warm.status, "cached")
        self.assertEqual(warm.object_key, cold.object_key)
        self.assertEqual(len(self.store.writes), 1)

    def test_parallel_results_preserve_input_order(self) -> None:
        second_topic = msgspec.structs.replace(self.bundle.topic, topic_id="2001")
        second_bundle = msgspec.structs.replace(self.bundle, topic=second_topic)
        outcomes = run_topics(
            (HandoffWork("2001", self.digest, second_bundle), self.work),
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
