from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trec_ragtime.artifact_store import ArtifactStore
from trec_ragtime.retrieval_handoff_cache import (
    HandoffFingerprint,
    RetrievalHandoffArtifactCache,
)
from trec_ragtime.retrieval_handoff_cli import main


@unittest.skipUnless(
    os.environ.get("RAGTIME_LIVE_HANDOFF") == "1",
    "set RAGTIME_LIVE_HANDOFF=1 for private artifact verification",
)
class RetrievalHandoffLiveTests(unittest.TestCase):
    def test_all_official_topics_are_complete_and_warm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "handoff.json"
            with (
                patch(
                    "trec_ragtime.retrieval_handoff_cli.RagtimeRetriever.from_env",
                    side_effect=AssertionError("warm verification opened retrieval"),
                ),
                patch(
                    "trec_ragtime.retrieval_handoff_cli._cache_only_scorer",
                    side_effect=AssertionError("warm verification opened scoring"),
                ),
                patch(
                    "trec_ragtime.retrieval_handoff_cli._nugget_candidates",
                    side_effect=AssertionError("warm verification opened nuggets"),
                ),
            ):
                code = main(
                    [
                        "--model",
                        os.environ.get(
                            "RAGTIME_HANDOFF_MODEL", "openai/gpt-4.1-mini"
                        ),
                        "--device",
                        "cuda",
                        "--retriever-fingerprint-sha256",
                        "487a18b1c3ebfdb4c967e6fea15c6a522d91f5bb002841d1ea03cb766c4ee1e7",
                        "--topic-plan-sha256",
                        "2066=e71f54e5ca59eadaa59073536f3c12f551d763c19993ba7c858624af861e3b01",
                        "--manifest",
                        str(manifest_path),
                    ]
                )
            self.assertEqual(code, 0)
            manifest = json.loads(manifest_path.read_text())
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["topic_count"], 103)
            self.assertEqual(
                {row["status"] for row in manifest["topics"]}, {"cached"}
            )

            cache = RetrievalHandoffArtifactCache(ArtifactStore.from_env())
            catalog_digest = bytes.fromhex(manifest["topic_catalog_digest"])
            sample = (
                manifest["topics"][0],
                manifest["topics"][len(manifest["topics"]) // 2],
                manifest["topics"][-1],
            )
            for row in sample:
                key = row["object_key"]
                cache.store.evict_local(key)
                bundle = cache.get_by_object_key(
                    HandoffFingerprint(),
                    key,
                    topic_catalog_digest=catalog_digest,
                    producer_identity_digest=bytes.fromhex(
                        row["producer_identity_digest"]
                    ),
                )
                self.assertIsNotNone(bundle)
                self.assertEqual(bundle.topic.topic_id, row["topic_id"])


if __name__ == "__main__":
    unittest.main()
