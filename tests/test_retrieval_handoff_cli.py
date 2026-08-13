from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from trec_ragtime.nugget_cache import NuggetArtifactCache
from trec_ragtime.passage_score_cache import PassageScoreArtifactValidationError
from trec_ragtime.retrieval_handoff_cli import (
    _cache_only_scorer,
    _discover_cached_handoffs,
    _nugget_candidates,
    _shared_tokenizer_loader,
    _topic_plan_digests,
    build_parser,
    main,
)
from trec_ragtime.retrieval_handoff_cache import (
    HandoffFingerprint,
    RetrievalHandoffArtifactCache,
    request_for_bundle,
)

from tests.test_nugget_cache import (
    MemoryArtifactStore,
    build_bundle,
    fingerprint as nugget_fingerprint,
    request as nugget_request,
)
from tests.test_retrieval_handoff_cache import bundle_fixture


class ListingMemoryArtifactStore(MemoryArtifactStore):
    def list_keys(self, prefix: str) -> tuple[str, ...]:
        return tuple(sorted(key for key in self.objects if key.startswith(prefix)))


class RetrievalHandoffCliTests(unittest.TestCase):
    def test_cold_discovery_reads_sealed_nuggets_from_durable_keys(self) -> None:
        store = ListingMemoryArtifactStore()
        bundle = build_bundle()
        fingerprint = nugget_fingerprint()
        NuggetArtifactCache(store, clock=lambda: 1.0).put(
            fingerprint, nugget_request(bundle), bundle
        )

        found = _nugget_candidates(store, fingerprint.digest)

        self.assertEqual(found[bundle.topic_id], (bundle,))

    def test_cold_identity_reconstruction_loads_the_tokenizer_once(self) -> None:
        sentinel = object()
        config = SimpleNamespace()
        with patch(
            "trec_ragtime.passage_scoring._default_tokenizer_loader",
            return_value=sentinel,
        ) as load:
            loader = _shared_tokenizer_loader()
            self.assertIs(loader(config), sentinel)
            self.assertIs(loader(config), sentinel)
        self.assertEqual(load.call_count, 1)

    def test_warm_discovery_reads_handoff_objects_without_upstream_loaders(self) -> None:
        store = ListingMemoryArtifactStore()
        cache = RetrievalHandoffArtifactCache(store)
        fingerprint = HandoffFingerprint()
        bundle = bundle_fixture()
        topic_catalog_digest = hashlib.sha256(b"topics").digest()
        request = request_for_bundle(
            bundle, topic_catalog_digest=topic_catalog_digest
        )
        cache.put(fingerprint, request, bundle)

        found = _discover_cached_handoffs(
            store,
            cache,
            fingerprint,
            topic_catalog_digest=topic_catalog_digest,
            topic_ids=(bundle.topic.topic_id,),
        )

        self.assertEqual(found[bundle.topic.topic_id][0], bundle)
        self.assertEqual(
            found[bundle.topic.topic_id][1], cache.object_key(fingerprint, request)
        )

    def test_main_warm_path_never_constructs_upstream_retriever(self) -> None:
        store = ListingMemoryArtifactStore()
        cache = RetrievalHandoffArtifactCache(store)
        fingerprint = HandoffFingerprint()
        bundle = bundle_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            topics = root / "topics.jsonl"
            topics.write_text('{"fixture":"topics"}\n')
            topic_catalog_digest = hashlib.sha256(topics.read_bytes()).digest()
            request = request_for_bundle(
                bundle, topic_catalog_digest=topic_catalog_digest
            )
            cache.put(fingerprint, request, bundle)
            manifest = root / "manifest.json"

            with (
                patch(
                    "trec_ragtime.retrieval_handoff_cli.load_query_plans",
                    return_value=(object(),),
                ),
                patch(
                    "trec_ragtime.retrieval_handoff_cli.select_plans",
                    return_value=(type("Plan", (), {"topic_id": "2000"})(),),
                ),
                patch(
                    "trec_ragtime.retrieval_handoff_cli.load_topics",
                    return_value=(bundle.topic,),
                ),
                patch(
                    "trec_ragtime.retrieval_handoff_cli.ArtifactStore.from_env",
                    return_value=store,
                ),
                patch(
                    "trec_ragtime.retrieval_handoff_cli.build_fingerprint",
                    return_value=SimpleNamespace(
                        digest=bundle.nugget_source.fingerprint_digest
                    ),
                ),
                patch(
                    "trec_ragtime.retrieval_handoff_cli.RagtimeRetriever.from_env",
                    side_effect=AssertionError("warm handoff must not open retriever"),
                ),
                patch(
                    "trec_ragtime.retrieval_handoff_cli._cache_only_scorer",
                    side_effect=AssertionError("warm handoff must not open scorer"),
                ),
                patch(
                    "trec_ragtime.retrieval_handoff_cli._nugget_candidates",
                    side_effect=AssertionError("warm handoff must not scan nuggets"),
                ),
            ):
                code = main(
                    [
                        "--model",
                        "provider-model",
                        "--topics",
                        str(topics),
                        "--manifest",
                        str(manifest),
                    ]
                )

            self.assertEqual(code, 0)
            self.assertTrue(manifest.exists())

    def test_local_scorer_identity_is_reconstructed_without_model_loading(self) -> None:
        arguments = build_parser().parse_args(["--model", "provider-model"])
        config, scorer = _cache_only_scorer(arguments)
        self.assertEqual(config.fingerprint, scorer.fingerprint)
        with self.assertRaisesRegex(
            PassageScoreArtifactValidationError, "cached passage scores required"
        ):
            scorer.score("query", ())

    def test_parser_defaults_to_report_text(self) -> None:
        arguments = build_parser().parse_args(["--model", "provider-model"])
        self.assertEqual(arguments.representation, "report_text")
        self.assertEqual(arguments.scorer, "local")

    def test_per_topic_plan_digest_override_is_strict(self) -> None:
        digest = "a" * 64
        self.assertEqual(_topic_plan_digests([f"2066={digest}"]), {"2066": digest})
        for bad in ("2066", "2066=ABC", "2066=" + "g" * 64):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                _topic_plan_digests([bad])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _topic_plan_digests([f"2066={digest}", f"2066={digest}"])


if __name__ == "__main__":
    unittest.main()
