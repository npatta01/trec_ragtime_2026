from __future__ import annotations

import re
import unittest
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import msgspec

from trec_ragtime.artifact_store import ArtifactStoreRemoteError, ArtifactValidationError
from trec_ragtime.evidence_bundle import (
    EvidenceBundle,
    EvidenceBundleSource,
    EvidenceFacet,
    EvidencePassage,
    topic_digest,
)
from trec_ragtime.report_cache import (
    REPORT_OBJECT_KEY_PREFIX,
    REPORT_SCHEMA_VERSION,
    GenerationSettings,
    ReportArtifactCache,
    ReportArtifactReference,
    ReportCitation,
    ReportDiagnostics,
    ReportRequestIdentity,
    ReportValidationError,
    ReportSentence,
    TopicReport,
    generator_fingerprint,
    report_object_key,
)
from trec_ragtime.topic_candidates import Topic


class MemoryArtifactStore:
    """Small in-memory store double with explicit cache operation ledgers."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.read_validated_calls: list[str] = []
        self.write_calls: list[tuple[str, bytes]] = []
        self.evict_local_calls: list[str] = []
        self.read_error: BaseException | None = None
        self.write_error: BaseException | None = None

    def read(self, object_key: str) -> bytes | None:
        return self.objects.get(object_key)

    def read_validated(self, object_key: str, decoder: Callable[[bytes], Any]) -> Any | None:
        self.read_validated_calls.append(object_key)
        if self.read_error is not None:
            raise self.read_error
        payload = self.objects.get(object_key)
        if payload is None:
            return None
        try:
            return decoder(payload)
        except ArtifactValidationError:
            self.objects.pop(object_key, None)
            return None

    def read_validated_strict(
        self, object_key: str, decoder: Callable[[bytes], Any]
    ) -> Any | None:
        """Decode durable bytes, raising rather than masking an invalid artifact."""

        self.read_validated_calls.append(object_key)
        if self.read_error is not None:
            raise self.read_error
        payload = self.objects.get(object_key)
        if payload is None:
            return None
        return decoder(payload)

    def write(self, object_key: str, payload: bytes) -> bool:
        if self.write_error is not None:
            raise self.write_error
        self.write_calls.append((object_key, payload))
        self.objects[object_key] = payload
        return True

    def evict_local(self, object_key: str) -> None:
        self.evict_local_calls.append(object_key)
        self.objects.pop(object_key, None)


def topic() -> Topic:
    return Topic(
        topic_id="2000",
        collection_id="ragtime/2/all",
        problem_statement="A statement about vaping law.",
        background="A legislative aide needs this.",
        limit=5000,
        title="Topic 2000",
    )


def bundle(**overrides: Any) -> EvidenceBundle:
    values: dict[str, Any] = {
        "topic_id": "2000",
        "collection_id": "ragtime/2/all",
        "title": "Topic 2000",
        "problem_statement": "A statement about vaping law.",
        "background": "A legislative aide needs this.",
        "character_limit": 5000,
        "max_passages": 2,
        "evidence_chars": 0,
        "facets": (
            EvidenceFacet(facet_id="F1", text="dates and countries", source="query_baseline_facet_v1"),
        ),
        "passages": tuple(
            EvidencePassage.create(
                handle=handle,
                document_id=document_id,
                report_text=f"Evidence text for {document_id}.",
                language="eng",
                url=None,
                date=None,
                rank=rank,
                retrieval_score=1.0 / rank,
                facet_ids=("F1",),
            )
            for handle, document_id, rank in (("S1", "doc-a", 1), ("S2", "doc-b", 2))
        ),
        "source": EvidenceBundleSource(
            run_schema="query_baseline_v1",
            run_id="baseline-c-test",
            run_dir_name="baseline-c-test",
            manifest_sha256="a" * 64,
            records_sha256="b" * 64,
            topics_sha256="c" * 64,
        ),
    }
    values.update(overrides)
    return EvidenceBundle.create(**values)


def settings(**overrides: Any) -> GenerationSettings:
    values: dict[str, Any] = {
        "base_url": "https://api.example.test/v1",
        "model": "test-model",
        "temperature": 0.0,
        "top_p": None,
        "max_output_tokens": 4000,
        "seed": 7,
        "max_repairs": 1,
        "max_citations": 3,
        "max_passages": 2,
        "evidence_chars": 0,
        "budget_ratio": 0.9,
        "response_format_mode": "json_schema",
    }
    values.update(overrides)
    return GenerationSettings(**values)


def identity(**overrides: Any) -> ReportRequestIdentity:
    return ReportRequestIdentity.for_bundle(
        overrides.pop("bundle", bundle()),
        topic=overrides.pop("topic", topic()),
        settings=overrides.pop("settings", settings()),
    )


def sentence(
    index: int,
    text: str,
    citations: tuple[tuple[str, str, float], ...] = (),
    *,
    substantive: bool = True,
) -> ReportSentence:
    return ReportSentence(
        index=index,
        text=text,
        substantive=substantive,
        citations=tuple(
            ReportCitation(document_id=document_id, handle=handle, score=score)
            for document_id, handle, score in citations
        ),
        facet_ids=("F1",),
    )


def diagnostics(**overrides: Any) -> ReportDiagnostics:
    values: dict[str, Any] = {
        "model_id": "test-model",
        "repair_attempts": 0,
        "latency_seconds": 1.5,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "dropped": (),
        "llm_cache_hit": False,
    }
    values.update(overrides)
    return ReportDiagnostics(**values)


def report(request: ReportRequestIdentity | None = None, **overrides: Any) -> TopicReport:
    request = request or identity()
    sentences = overrides.pop(
        "sentences",
        (
            sentence(1, "The ban took effect in 2019.", (("doc-a", "S1", 100.0),)),
            sentence(
                2,
                "Youth vaping fell sharply.",
                (("doc-b", "S2", 100.0), ("doc-a", "S1", 90.0)),
            ),
        ),
    )
    return TopicReport.create(
        request=request,
        sentences=sentences,
        created_at=overrides.pop("created_at", "2026-08-10T00:00:00+00:00"),
        diagnostics=overrides.pop("diagnostics", diagnostics()),
        **overrides,
    )


class FingerprintTests(unittest.TestCase):
    def test_fingerprint_is_32_bytes_and_deterministic(self) -> None:
        self.assertEqual(len(generator_fingerprint(settings())), 32)
        self.assertEqual(generator_fingerprint(settings()), generator_fingerprint(settings()))

    def test_every_declared_component_changes_the_fingerprint(self) -> None:
        baseline = generator_fingerprint(settings())
        cases = {
            "base_url": "https://other.example.test/v1",
            "model": "other-model",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_output_tokens": 2000,
            "seed": 8,
            "max_repairs": 2,
            "max_citations": 2,
            "max_passages": 5,
            "evidence_chars": 1200,
            "budget_ratio": 0.8,
            "response_format_mode": "json_object",
            "prompt_version": "report-prompt-v2",
            "output_schema_version": "report-sentences-v2",
            "segmenter_version": "pysbd-0.3.4-en-v1",
            "citation_policy_version": "citation-rank-v9",
            "length_packing_version": "whole-sentence-pack-v2",
            "generator_behavior_version": "report-generator-extractive-v1",
        }
        for field, value in cases.items():
            with self.subTest(field=field):
                self.assertNotEqual(generator_fingerprint(settings(**{field: value})), baseline)

    def test_trailing_slash_in_base_url_is_normalized(self) -> None:
        self.assertEqual(
            generator_fingerprint(settings(base_url="https://api.example.test/v1/")),
            generator_fingerprint(settings(base_url="https://api.example.test/v1")),
        )

    def test_invalid_settings_are_rejected(self) -> None:
        cases = (
            {"model": ""},
            {"temperature": float("nan")},
            {"max_output_tokens": 0},
            {"max_citations": 0},
            {"max_passages": 0},
            {"evidence_chars": -1},
            {"budget_ratio": 0.0},
            {"budget_ratio": 1.5},
            {"max_repairs": -1},
            {"response_format_mode": "yaml"},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    settings(**overrides)


class IdentityTests(unittest.TestCase):
    def test_object_key_layout(self) -> None:
        request = identity()
        pattern = re.compile(
            rf"{REPORT_OBJECT_KEY_PREFIX}/[0-9a-f]{{64}}/([0-9a-f]{{2}})/([0-9a-f]{{64}})\.mpk"
        )
        match = pattern.fullmatch(request.object_key)
        self.assertIsNotNone(match, request.object_key)
        assert match is not None
        self.assertTrue(match.group(2).startswith(match.group(1)))

    def test_object_key_matches_the_standalone_builder(self) -> None:
        request = identity()
        self.assertEqual(
            request.object_key,
            report_object_key(request.generator_fingerprint, request.request_digest),
        )

    def test_request_digest_is_deterministic(self) -> None:
        self.assertEqual(identity().request_digest, identity().request_digest)

    def test_bundle_seal_changes_the_request_digest(self) -> None:
        other = bundle(evidence_chars=500)
        self.assertNotEqual(
            identity(bundle=other).request_digest, identity().request_digest
        )

    def test_topic_change_changes_the_request_digest(self) -> None:
        other_topic = replace(topic(), problem_statement="Different statement.")
        other = bundle(problem_statement="Different statement.")
        self.assertNotEqual(
            ReportRequestIdentity.for_bundle(
                other, topic=other_topic, settings=settings()
            ).request_digest,
            identity().request_digest,
        )

    def test_settings_change_the_fingerprint_half_only(self) -> None:
        other = identity(settings=settings(model="other-model"))
        self.assertEqual(other.request_digest, identity().request_digest)
        self.assertNotEqual(other.generator_fingerprint, identity().generator_fingerprint)
        self.assertNotEqual(other.object_key, identity().object_key)

    def test_topic_digest_matches_the_bundle_topic(self) -> None:
        self.assertEqual(identity().topic_digest, topic_digest(topic()))

    def test_bundle_and_topic_must_agree(self) -> None:
        with self.assertRaises(ValueError):
            ReportRequestIdentity.for_bundle(
                bundle(), topic=replace(topic(), topic_id="2001"), settings=settings()
            )

    def test_bundle_and_topic_limit_must_agree(self) -> None:
        with self.assertRaises(ValueError):
            ReportRequestIdentity.for_bundle(
                bundle(), topic=replace(topic(), limit=4000), settings=settings()
            )

    def test_reference_rejects_a_key_that_does_not_match_its_digests(self) -> None:
        request = identity()
        with self.assertRaises(ValueError):
            ReportArtifactReference(
                object_key="reports/v1/deadbeef/aa/bb.mpk",
                request_digest=request.request_digest,
                generator_fingerprint=request.generator_fingerprint,
                stored_at=1.0,
                cache_hit=True,
            )


class TopicReportTests(unittest.TestCase):
    def test_create_derives_references_and_counts(self) -> None:
        item = report()
        self.assertEqual(item.references, ("doc-a", "doc-b"))
        expected = len("The ban took effect in 2019.") + len("Youth vaping fell sharply.")
        self.assertEqual(item.normalized_character_count, expected)
        self.assertEqual(item.budgeted_character_count, expected + 1)
        self.assertEqual(item.schema_version, REPORT_SCHEMA_VERSION)

    def test_create_binds_identity(self) -> None:
        request = identity()
        item = report(request)
        self.assertEqual(item.bundle_seal, request.bundle_seal)
        self.assertEqual(item.topic_digest, request.topic_digest)
        self.assertEqual(item.generator_fingerprint, request.generator_fingerprint)
        self.assertEqual(item.character_limit, 5000)


class CacheRoundTripTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = MemoryArtifactStore()
        self.cache = ReportArtifactCache(self.store, clock=lambda: 1_700_000_000.0)
        self.bundle = bundle()
        self.request = identity(bundle=self.bundle)

    def test_miss_returns_none_without_writing(self) -> None:
        self.assertIsNone(self.cache.get(self.request, bundle=self.bundle))
        self.assertEqual(self.store.write_calls, [])

    def test_put_then_get_round_trips(self) -> None:
        stored = self.cache.put(self.request, report(self.request), bundle=self.bundle)
        self.assertIsNotNone(stored.report_artifact)
        self.assertFalse(stored.report_artifact.cache_hit)

        loaded = self.cache.get(self.request, bundle=self.bundle)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertTrue(loaded.report_artifact.cache_hit)
        self.assertEqual(loaded.references, stored.references)
        self.assertEqual(
            [item.text for item in loaded.sentences], [item.text for item in stored.sentences]
        )
        self.assertEqual(
            [(c.document_id, c.handle, c.score) for c in loaded.sentences[1].citations],
            [("doc-b", "S2", 100.0), ("doc-a", "S1", 90.0)],
        )
        self.assertEqual(loaded.diagnostics.model_id, "test-model")
        self.assertEqual(loaded.created_at, "2026-08-10T00:00:00+00:00")

    def test_a_fresh_cache_over_the_same_store_replays(self) -> None:
        self.cache.put(self.request, report(self.request), bundle=self.bundle)
        elsewhere = ReportArtifactCache(self.store, clock=lambda: 0.0)
        loaded = elsewhere.get(self.request, bundle=self.bundle)
        self.assertIsNotNone(loaded)

    def test_object_key_has_no_machine_specific_component(self) -> None:
        self.assertNotIn("/tmp", self.request.object_key)
        self.assertTrue(self.request.object_key.startswith(f"{REPORT_OBJECT_KEY_PREFIX}/"))

    def test_a_different_fingerprint_does_not_read_the_same_artifact(self) -> None:
        self.cache.put(self.request, report(self.request), bundle=self.bundle)
        other = identity(bundle=self.bundle, settings=settings(model="other-model"))
        self.assertIsNone(self.cache.get(other, bundle=self.bundle))

    def test_tampered_payload_is_treated_as_a_miss(self) -> None:
        self.cache.put(self.request, report(self.request), bundle=self.bundle)
        self.store.objects[self.request.object_key] = msgspec.msgpack.encode({"nope": 1})
        self.assertIsNone(self.cache.get(self.request, bundle=self.bundle))

    def test_non_canonical_encoding_is_rejected(self) -> None:
        self.cache.put(self.request, report(self.request), bundle=self.bundle)
        payload = self.store.objects[self.request.object_key]
        decoded = msgspec.msgpack.decode(payload)
        decoded["stored_at"] = float(decoded["stored_at"])
        # Re-encoding a dict rather than the pinned Struct changes field order.
        self.store.objects[self.request.object_key] = msgspec.msgpack.encode(
            dict(reversed(list(decoded.items())))
        )
        self.assertIsNone(self.cache.get(self.request, bundle=self.bundle))

    def test_store_write_failure_returns_a_report_without_a_reference(self) -> None:
        self.store.write_error = ArtifactStoreRemoteError("offline")
        stored = self.cache.put(self.request, report(self.request), bundle=self.bundle)
        self.assertIsNone(stored.report_artifact)

    def test_store_read_failure_is_a_miss(self) -> None:
        self.store.read_error = ArtifactStoreRemoteError("offline")
        self.assertIsNone(self.cache.get(self.request, bundle=self.bundle))

    def test_bundle_must_match_the_request_seal(self) -> None:
        with self.assertRaises(ReportValidationError):
            self.cache.put(self.request, report(self.request), bundle=bundle(evidence_chars=500))


class ReportValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = MemoryArtifactStore()
        self.cache = ReportArtifactCache(self.store, clock=lambda: 1.0)
        self.bundle = bundle()
        self.request = identity(bundle=self.bundle)

    def assert_not_cached(self, **overrides: Any) -> None:
        with self.assertRaises(ReportValidationError):
            self.cache.put(self.request, report(self.request, **overrides), bundle=self.bundle)
        self.assertEqual(self.store.write_calls, [])

    def test_empty_sentence_list_is_not_cacheable(self) -> None:
        self.assert_not_cached(sentences=())

    def test_non_contiguous_indices_are_not_cacheable(self) -> None:
        self.assert_not_cached(
            sentences=(sentence(1, "A claim.", (("doc-a", "S1", 100.0),)), sentence(3, "B claim.", (("doc-b", "S2", 100.0),)))
        )

    def test_unnormalized_text_is_not_cacheable(self) -> None:
        self.assert_not_cached(
            sentences=(sentence(1, "A  spaced   claim.", (("doc-a", "S1", 100.0),)),)
        )

    def test_multi_sentence_text_is_not_cacheable(self) -> None:
        self.assert_not_cached(
            sentences=(sentence(1, "One claim. Two claims.", (("doc-a", "S1", 100.0),)),)
        )

    def test_duplicate_text_is_not_cacheable(self) -> None:
        self.assert_not_cached(
            sentences=(
                sentence(1, "A claim.", (("doc-a", "S1", 100.0),)),
                sentence(2, "A claim.", (("doc-b", "S2", 100.0),)),
            )
        )

    def test_substantive_sentence_without_citation_is_not_cacheable(self) -> None:
        self.assert_not_cached(sentences=(sentence(1, "A claim.", ()),))

    def test_non_substantive_sentence_with_citation_is_not_cacheable(self) -> None:
        self.assert_not_cached(
            sentences=(
                sentence(1, "A framing line.", (("doc-a", "S1", 100.0),), substantive=False),
                sentence(2, "A claim.", (("doc-b", "S2", 100.0),)),
            )
        )

    def test_non_substantive_sentence_without_citation_is_allowed(self) -> None:
        stored = self.cache.put(
            self.request,
            report(
                self.request,
                sentences=(
                    sentence(1, "This report covers two themes.", (), substantive=False),
                    sentence(2, "The ban took effect.", (("doc-a", "S1", 100.0),)),
                ),
            ),
            bundle=self.bundle,
        )
        self.assertEqual(stored.references, ("doc-a",))

    def test_citation_outside_the_allowlist_is_not_cacheable(self) -> None:
        self.assert_not_cached(
            sentences=(sentence(1, "A claim.", (("doc-z", "S9", 100.0),)),)
        )

    def test_handle_document_mismatch_is_not_cacheable(self) -> None:
        # S1 resolves to doc-a, so pairing it with doc-b must fail closed.
        self.assert_not_cached(
            sentences=(sentence(1, "A claim.", (("doc-b", "S1", 100.0),)),)
        )

    def test_duplicate_document_within_a_sentence_is_not_cacheable(self) -> None:
        self.assert_not_cached(
            sentences=(
                sentence(1, "A claim.", (("doc-a", "S1", 100.0), ("doc-a", "S1", 90.0))),
            )
        )

    def test_non_finite_and_boolean_scores_are_not_cacheable(self) -> None:
        for score in (float("inf"), float("nan")):
            with self.subTest(score=score):
                self.assert_not_cached(
                    sentences=(sentence(1, "A claim.", (("doc-a", "S1", score),)),)
                )

    def test_non_descending_scores_are_not_cacheable(self) -> None:
        self.assert_not_cached(
            sentences=(
                sentence(1, "A claim.", (("doc-a", "S1", 90.0), ("doc-b", "S2", 100.0))),
            )
        )

    def test_tied_scores_are_not_cacheable(self) -> None:
        self.assert_not_cached(
            sentences=(
                sentence(1, "A claim.", (("doc-a", "S1", 100.0), ("doc-b", "S2", 100.0))),
            )
        )

    def test_more_citations_than_the_cap_is_not_cacheable(self) -> None:
        narrow = identity(bundle=self.bundle, settings=settings(max_citations=1))
        with self.assertRaises(ReportValidationError):
            self.cache.put(
                narrow,
                report(
                    narrow,
                    sentences=(
                        sentence(
                            1, "A claim.", (("doc-a", "S1", 100.0), ("doc-b", "S2", 90.0))
                        ),
                    ),
                ),
                bundle=self.bundle,
            )

    def test_over_limit_report_is_not_cacheable(self) -> None:
        small_topic = replace(topic(), limit=10)
        small_bundle = bundle(character_limit=10)
        request = ReportRequestIdentity.for_bundle(
            small_bundle, topic=small_topic, settings=settings()
        )
        with self.assertRaises(ReportValidationError):
            self.cache.put(
                request,
                report(
                    request,
                    sentences=(
                        sentence(1, "A very long claim indeed.", (("doc-a", "S1", 100.0),)),
                    ),
                ),
                bundle=small_bundle,
            )
        self.assertEqual(self.store.write_calls, [])

    def test_blank_model_identity_is_not_cacheable(self) -> None:
        self.assert_not_cached(diagnostics=diagnostics(model_id=""))

    def test_identity_mismatch_is_not_cacheable(self) -> None:
        other = identity(bundle=self.bundle, settings=settings(model="other-model"))
        with self.assertRaises(ReportValidationError):
            self.cache.put(self.request, report(other), bundle=self.bundle)

    def test_dropped_diagnostics_record_hashes_not_text(self) -> None:
        from trec_ragtime.report_cache import DroppedSentence

        secret = "An unsupported claim that must never be stored."
        stored = self.cache.put(
            self.request,
            report(
                self.request,
                diagnostics=diagnostics(
                    dropped=(
                        DroppedSentence.create(index=3, reason="missing_citation", text=secret),
                    )
                ),
            ),
            bundle=self.bundle,
        )
        self.assertEqual(stored.diagnostics.dropped[0].character_count, len(secret))
        self.assertEqual(len(stored.diagnostics.dropped[0].text_sha256), 64)
        payload = self.store.objects[self.request.object_key]
        self.assertNotIn(b"unsupported claim", payload)

    def test_unknown_drop_reason_is_rejected(self) -> None:
        from trec_ragtime.report_cache import DroppedSentence

        with self.assertRaises(ValueError):
            DroppedSentence.create(index=1, reason="because", text="x")

    def test_artifacts_do_not_contain_evidence_text(self) -> None:
        self.cache.put(self.request, report(self.request), bundle=self.bundle)
        payload = self.store.objects[self.request.object_key]
        self.assertNotIn(b"Evidence text for doc-a", payload)
        self.assertNotIn(b"A statement about vaping law", payload)


if __name__ == "__main__":
    unittest.main()
