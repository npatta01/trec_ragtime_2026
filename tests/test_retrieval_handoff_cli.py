from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import msgspec

from trec_ragtime.artifact_store import ArtifactValidationError
from trec_ragtime.nugget_cache import NuggetArtifactCache
from trec_ragtime.passage_score_cache import PassageScoreArtifactValidationError
from trec_ragtime.query_baselines import QueryPlan
from trec_ragtime.retrieval_handoff_cli import (
    _cache_only_scorer,
    _discover_cached_handoffs,
    _nugget_candidates,
    _prepare_source_identity,
    _producer_identity,
    _producer_scorer_config,
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
from trec_ragtime.topic_candidates import Topic

from tests.test_nugget_cache import (
    MemoryArtifactStore,
    build_bundle,
    fingerprint as nugget_fingerprint,
    request as nugget_request,
)
from tests.test_retrieval_handoff_cache import bundle_fixture
from tests.test_retrieval_handoff import RetrievalHandoffTests


class ListingMemoryArtifactStore(MemoryArtifactStore):
    def list_keys(self, prefix: str) -> tuple[str, ...]:
        return tuple(sorted(key for key in self.objects if key.startswith(prefix)))


class RetrievalHandoffCliTests(unittest.TestCase):
    @staticmethod
    def _fixture():
        fixture = RetrievalHandoffTests()
        fixture.setUp()
        return fixture

    @staticmethod
    def _plan() -> QueryPlan:
        return QueryPlan("2000", "complete", "grain exports", ())

    @staticmethod
    def _producer_digest(
        plan: QueryPlan,
        nugget_fingerprint_digest: bytes,
        *extra_arguments: str,
    ) -> bytes:
        arguments = build_parser().parse_args(
            ["--model", "provider-model", *extra_arguments]
        )
        return _producer_identity(
            plan,
            plan_sha256=arguments.plan_sha256,
            arguments=arguments,
            scorer_config=_producer_scorer_config(arguments),
            nugget_fingerprint_digest=nugget_fingerprint_digest,
        ).digest

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
        producer_identity_digest = b"p" * 32
        request = request_for_bundle(
            bundle,
            topic_catalog_digest=topic_catalog_digest,
            producer_identity_digest=producer_identity_digest,
        )
        cache.put(fingerprint, request, bundle)

        found = _discover_cached_handoffs(
            store,
            cache,
            fingerprint,
            topic_catalog_digest=topic_catalog_digest,
            expected_topics={bundle.topic.topic_id: bundle.topic},
            producer_identity_digests={
                bundle.topic.topic_id: producer_identity_digest
            },
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
        plan = self._plan()
        official_topic = Topic(
            bundle.topic.topic_id,
            bundle.topic.collection_id,
            bundle.topic.problem_statement,
            bundle.topic.background,
            bundle.topic.limit,
            bundle.topic.title,
        )
        producer_identity_digest = self._producer_digest(
            plan, bundle.nugget_source.fingerprint_digest
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            topics = root / "topics.jsonl"
            topics.write_text('{"fixture":"topics"}\n')
            topic_catalog_digest = hashlib.sha256(topics.read_bytes()).digest()
            request = request_for_bundle(
                bundle,
                topic_catalog_digest=topic_catalog_digest,
                producer_identity_digest=producer_identity_digest,
            )
            cache.put(fingerprint, request, bundle)
            manifest = root / "manifest.json"

            with (
                patch(
                    "trec_ragtime.retrieval_handoff_cli.load_query_plans",
                    return_value=(plan,),
                ),
                patch(
                    "trec_ragtime.retrieval_handoff_cli.select_plans",
                    return_value=(plan,),
                ),
                patch(
                    "trec_ragtime.retrieval_handoff_cli.load_topics",
                    return_value=(official_topic,),
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
                patch(
                    "trec_ragtime.passage_scoring._default_tokenizer_loader",
                    side_effect=AssertionError("warm handoff must not load tokenizer"),
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

    def test_main_cold_path_builds_from_durable_nugget_and_cached_retrieval(
        self,
    ) -> None:
        fixture = self._fixture()
        fixture.topic = replace(
            fixture.topic, problem_statement=fixture.request.query
        )
        store = ListingMemoryArtifactStore()
        fingerprint = nugget_fingerprint()
        NuggetArtifactCache(store, clock=lambda: 1.0).put(
            fingerprint, fixture.request, fixture.nuggets
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            topics = root / "topics.jsonl"
            topics.write_text('{"fixture":"topics"}\n')
            manifest = root / "manifest.json"
            plan = self._plan()

            with (
                patch(
                    "trec_ragtime.retrieval_handoff_cli.load_query_plans",
                    return_value=(plan,),
                ),
                patch(
                    "trec_ragtime.retrieval_handoff_cli.select_plans",
                    return_value=(plan,),
                ),
                patch(
                    "trec_ragtime.retrieval_handoff_cli.load_topics",
                    return_value=(fixture.topic,),
                ),
                patch(
                    "trec_ragtime.retrieval_handoff_cli.ArtifactStore.from_env",
                    return_value=store,
                ),
                patch(
                    "trec_ragtime.retrieval_handoff_cli.build_fingerprint",
                    return_value=fingerprint,
                ),
                patch(
                    "trec_ragtime.retrieval_handoff_cli._cache_only_scorer",
                    return_value=(object(), object()),
                ),
                patch(
                    "trec_ragtime.retrieval_handoff_cli.RagtimeRetriever.from_env",
                    return_value=nullcontext(object()),
                ),
                patch(
                    "trec_ragtime.retrieval_handoff_cli._prepare_source_identity",
                    return_value=(
                        fixture.request.inputs,
                        fixture.request.passage_source_fingerprint,
                        fixture.retrievals,
                    ),
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

            payload = json.loads(manifest.read_text())
            self.assertEqual(code, 0)
            self.assertTrue(payload["complete"])
            self.assertEqual(payload["topics"][0]["status"], "written")
            self.assertEqual(len(store.writes), 2)

    def test_main_cold_setup_failure_is_manifested_without_opening_retrieval(
        self,
    ) -> None:
        fixture = self._fixture()
        fixture.topic = replace(
            fixture.topic, problem_statement=fixture.request.query
        )
        store = ListingMemoryArtifactStore()
        fingerprint = nugget_fingerprint()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            topics = root / "topics.jsonl"
            topics.write_text('{"fixture":"topics"}\n')
            manifest = root / "manifest.json"
            plan = self._plan()

            with (
                patch(
                    "trec_ragtime.retrieval_handoff_cli.load_query_plans",
                    return_value=(plan,),
                ),
                patch(
                    "trec_ragtime.retrieval_handoff_cli.select_plans",
                    return_value=(plan,),
                ),
                patch(
                    "trec_ragtime.retrieval_handoff_cli.load_topics",
                    return_value=(fixture.topic,),
                ),
                patch(
                    "trec_ragtime.retrieval_handoff_cli.ArtifactStore.from_env",
                    return_value=store,
                ),
                patch(
                    "trec_ragtime.retrieval_handoff_cli.build_fingerprint",
                    return_value=fingerprint,
                ),
                patch(
                    "trec_ragtime.retrieval_handoff_cli._cache_only_scorer",
                    side_effect=ArtifactValidationError("missing scorer"),
                ),
                patch(
                    "trec_ragtime.retrieval_handoff_cli.RagtimeRetriever.from_env",
                    side_effect=AssertionError("failed setup must not open retrieval"),
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

            payload = json.loads(manifest.read_text())
            self.assertEqual(code, 1)
            self.assertFalse(payload["complete"])
            self.assertEqual(payload["topics"][0]["status"], "failed")
            self.assertEqual(
                payload["topics"][0]["reason"], "ArtifactValidationError"
            )

    def test_warm_discovery_selects_exact_producer_when_two_versions_exist(
        self,
    ) -> None:
        store = ListingMemoryArtifactStore()
        cache = RetrievalHandoffArtifactCache(store)
        fingerprint = HandoffFingerprint()
        bundle = bundle_fixture()
        catalog = hashlib.sha256(b"topics").digest()
        current = b"a" * 32
        stale = b"b" * 32
        current_request = request_for_bundle(
            bundle,
            topic_catalog_digest=catalog,
            producer_identity_digest=current,
        )
        cache.put(fingerprint, current_request, bundle)
        cache.put(
            fingerprint,
            request_for_bundle(
                bundle,
                topic_catalog_digest=catalog,
                producer_identity_digest=stale,
            ),
            bundle,
        )

        found = _discover_cached_handoffs(
            store,
            cache,
            fingerprint,
            topic_catalog_digest=catalog,
            expected_topics={bundle.topic.topic_id: bundle.topic},
            producer_identity_digests={bundle.topic.topic_id: current},
        )

        self.assertEqual(
            found[bundle.topic.topic_id][1],
            cache.object_key(fingerprint, current_request),
        )

    def test_warm_discovery_rejects_topic_record_drift(self) -> None:
        store = ListingMemoryArtifactStore()
        cache = RetrievalHandoffArtifactCache(store)
        fingerprint = HandoffFingerprint()
        expected = bundle_fixture()
        changed = msgspec.structs.replace(
            expected,
            topic=msgspec.structs.replace(expected.topic, title="different title"),
        )
        catalog = hashlib.sha256(b"topics").digest()
        producer = b"p" * 32
        cache.put(
            fingerprint,
            request_for_bundle(
                changed,
                topic_catalog_digest=catalog,
                producer_identity_digest=producer,
            ),
            changed,
        )

        with self.assertRaisesRegex(ArtifactValidationError, "topic"):
            _discover_cached_handoffs(
                store,
                cache,
                fingerprint,
                topic_catalog_digest=catalog,
                expected_topics={expected.topic.topic_id: expected.topic},
                producer_identity_digests={expected.topic.topic_id: producer},
            )

    def test_main_changed_producer_inputs_never_reuse_a_stale_handoff(self) -> None:
        bundle = bundle_fixture()
        plan = self._plan()
        official_topic = Topic(
            bundle.topic.topic_id,
            bundle.topic.collection_id,
            bundle.topic.problem_statement,
            bundle.topic.background,
            bundle.topic.limit,
            bundle.topic.title,
        )
        variants = (
            ("plan", ("--plan-sha256", "a" * 64)),
            ("top_k", ("--top-k", "99")),
            ("scorer", ("--device", "cuda")),
            (
                "retriever",
                ("--retriever-fingerprint-sha256", "a" * 64),
            ),
        )
        for name, extra in variants:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                store = ListingMemoryArtifactStore()
                cache = RetrievalHandoffArtifactCache(store)
                root = Path(directory)
                topics = root / "topics.jsonl"
                topics.write_text('{"fixture":"topics"}\n')
                catalog = hashlib.sha256(topics.read_bytes()).digest()
                base_producer = self._producer_digest(
                    plan, bundle.nugget_source.fingerprint_digest
                )
                cache.put(
                    HandoffFingerprint(),
                    request_for_bundle(
                        bundle,
                        topic_catalog_digest=catalog,
                        producer_identity_digest=base_producer,
                    ),
                    bundle,
                )
                manifest = root / "manifest.json"

                with (
                    patch(
                        "trec_ragtime.retrieval_handoff_cli.load_query_plans",
                        return_value=(plan,),
                    ),
                    patch(
                        "trec_ragtime.retrieval_handoff_cli.select_plans",
                        return_value=(plan,),
                    ),
                    patch(
                        "trec_ragtime.retrieval_handoff_cli.load_topics",
                        return_value=(official_topic,),
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
                        "trec_ragtime.retrieval_handoff_cli._cache_only_scorer",
                        side_effect=ArtifactValidationError("cold path reached"),
                    ) as cold,
                ):
                    code = main(
                        [
                            "--model",
                            "provider-model",
                            "--topics",
                            str(topics),
                            "--manifest",
                            str(manifest),
                            *extra,
                        ]
                    )

                self.assertEqual(code, 1)
                cold.assert_called_once()
                payload = json.loads(manifest.read_text())
                self.assertEqual(payload["topics"][0]["status"], "failed")

    def test_producer_identity_changes_with_chunker_runtime(self) -> None:
        plan = self._plan()
        nugget_digest = b"n" * 32
        with patch(
            "trec_ragtime.passages._runtime_version", return_value="runtime-a"
        ):
            first = self._producer_digest(plan, nugget_digest)
        with patch(
            "trec_ragtime.passages._runtime_version", return_value="runtime-b"
        ):
            second = self._producer_digest(plan, nugget_digest)

        self.assertNotEqual(first, second)

    def test_cold_source_identity_rejects_wrong_retriever_fingerprint(self) -> None:
        fixture = self._fixture()
        retrieval = replace(
            fixture.retrievals[0],
            retrieval_artifact=SimpleNamespace(
                object_key=fixture.request.inputs[0].object_key,
                request_digest=fixture.request.inputs[0].request_digest,
                retriever_fingerprint=b"x" * 32,
            ),
        )
        retriever = SimpleNamespace(
            get_cached_retrieval=lambda *args, **kwargs: retrieval
        )
        arguments = build_parser().parse_args(["--model", "provider-model"])
        scorer_config = _producer_scorer_config(arguments)

        with self.assertRaisesRegex(ValueError, "retriever fingerprint"):
            _prepare_source_identity(
                self._plan(),
                plan_sha256=arguments.plan_sha256,
                retriever=retriever,
                scorer=SimpleNamespace(fingerprint=scorer_config.fingerprint),
                scorer_config=scorer_config,
                representation="report_text",
                top_k=arguments.top_k,
                rerank_documents=arguments.rerank_documents,
                expected_retriever_fingerprint=b"r" * 32,
                tokenizer_loader=lambda _: (_ for _ in ()).throw(
                    AssertionError("wrong retriever must fail before tokenizer load")
                ),
            )

    def test_main_retriever_setup_failure_is_manifested(self) -> None:
        fixture = self._fixture()
        fixture.topic = replace(
            fixture.topic, problem_statement=fixture.request.query
        )
        store = ListingMemoryArtifactStore()
        plan = self._plan()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            topics = root / "topics.jsonl"
            topics.write_text('{"fixture":"topics"}\n')
            manifest = root / "manifest.json"
            with (
                patch(
                    "trec_ragtime.retrieval_handoff_cli.load_query_plans",
                    return_value=(plan,),
                ),
                patch(
                    "trec_ragtime.retrieval_handoff_cli.select_plans",
                    return_value=(plan,),
                ),
                patch(
                    "trec_ragtime.retrieval_handoff_cli.load_topics",
                    return_value=(fixture.topic,),
                ),
                patch(
                    "trec_ragtime.retrieval_handoff_cli.ArtifactStore.from_env",
                    return_value=store,
                ),
                patch(
                    "trec_ragtime.retrieval_handoff_cli.build_fingerprint",
                    return_value=nugget_fingerprint(),
                ),
                patch(
                    "trec_ragtime.retrieval_handoff_cli._cache_only_scorer",
                    return_value=(object(), object()),
                ),
                patch(
                    "trec_ragtime.retrieval_handoff_cli.RagtimeRetriever.from_env",
                    side_effect=RuntimeError("corpus unavailable"),
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

            payload = json.loads(manifest.read_text())
            self.assertEqual(code, 1)
            self.assertEqual(payload["topics"][0]["status"], "failed")
            self.assertEqual(payload["topics"][0]["reason"], "RuntimeError")

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
