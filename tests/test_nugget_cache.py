"""Identity, isolation, and fail-closed reads for the nugget artifact cache."""

from __future__ import annotations

import hashlib
import unittest
from collections.abc import Callable
from typing import Any

import msgspec

from trec_ragtime.artifact_store import ArtifactValidationError
from trec_ragtime.nugget_cache import (
    NUGGET_OBJECT_KEY_PREFIX,
    NuggetArtifactCache,
    NuggetEnvelope,
    NuggetFingerprint,
    NuggetInput,
    NuggetRequest,
    evidence_digest,
    nugget_object_key,
)
from trec_ragtime.nuggets import (
    FacetEvidence,
    NuggetClusterer,
    NuggetError,
    normalize_claim,
    run_topic_nuggets,
)

from tests.test_nuggets import ScriptedBackend, candidate, fixed_embedder


class MemoryArtifactStore:
    """In-memory ``ArtifactStoreLike`` mirroring the retrieval cache fixture."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.reads: list[str] = []
        self.writes: list[str] = []
        self.evictions: list[str] = []
        self.write_result: bool = True

    def read(self, object_key: str) -> bytes | None:
        self.reads.append(object_key)
        return self.objects.get(object_key)

    def read_validated(self, object_key: str, decoder: Callable[[bytes], Any]) -> Any:
        payload = self.read(object_key)
        if payload is None:
            return None
        try:
            return decoder(payload)
        except ArtifactValidationError:
            self.evict_local(object_key)
            return None

    def write(self, object_key: str, payload: bytes) -> bool:
        self.writes.append(object_key)
        if self.write_result:
            self.objects[object_key] = payload
        return self.write_result

    def evict_local(self, object_key: str) -> None:
        self.evictions.append(object_key)


FACETS = [
    FacetEvidence(
        facet_index=0,
        query="grain exports",
        candidates=(
            candidate("grain passage", document_id="doc-a", rank=1, score=0.9),
            candidate("port passage", document_id="doc-b", rank=2, score=0.5),
        ),
    )
]


def build_bundle() -> Any:
    backend = ScriptedBackend(
        claims=(("grain exports fell", "vital"),),
        supported={"grain exports fell": ("grain passage",)},
    )
    return run_topic_nuggets(
        topic_id="2000",
        query="grain exports",
        representation="report_text",
        facets=FACETS,
        backend=backend,
        clusterer=NuggetClusterer(
            embed=fixed_embedder({normalize_claim("grain exports fell"): [1.0]})
        ),
    )


def fingerprint(**changes: Any) -> NuggetFingerprint:
    base: dict[str, Any] = {
        "schema_version": 1,
        "behavior_version": "nuggets-v1",
        "package": {"name": "nuggetizer", "version": "0.0.5"},
        "nuggetizer_config": {"model": "test-model", "max_nuggets": 20},
        "provider_base_url_digest": hashlib.sha256(b"https://provider/v1").digest(),
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "similarity_threshold": 0.92,
        "guard_version": "numeric-negation-guard-v1",
        "evidence_budget": 24,
        "max_nuggets": 20,
        "output_limits": {
            "max_claim_characters": 400,
            "max_claim_words": 30,
            "max_completion_tokens": 2048,
        },
    }
    base.update(changes)
    return NuggetFingerprint(**base)


def request(bundle: Any, **changes: Any) -> NuggetRequest:
    base: dict[str, Any] = {
        "topic_id": "2000",
        "query": "grain exports",
        "representation": "report_text",
        "inputs": (
            NuggetInput(
                facet_index=0,
                query="grain exports",
                object_key="retrieval/v1/aa/bb/cc.mpk",
                request_digest=hashlib.sha256(b"retrieval-request").digest(),
            ),
        ),
        "passage_source_fingerprint": "langchain-spacy-v1/deadbeef",
        "evidence_digest": evidence_digest(bundle.evidence),
    }
    base.update(changes)
    return NuggetRequest(**base)


class ObjectKeyTestCase(unittest.TestCase):
    def test_shards_by_request_digest_prefix(self) -> None:
        fingerprint_bytes = hashlib.sha256(b"fp").digest()
        request_bytes = hashlib.sha256(b"rq").digest()
        key = nugget_object_key(fingerprint_bytes, request_bytes)
        self.assertEqual(
            key,
            f"{NUGGET_OBJECT_KEY_PREFIX}/{fingerprint_bytes.hex()}/"
            f"{request_bytes.hex()[:2]}/{request_bytes.hex()}.mpk",
        )

    def test_rejects_a_digest_of_the_wrong_size(self) -> None:
        with self.assertRaises(ValueError):
            nugget_object_key(b"short", hashlib.sha256(b"rq").digest())


class RoundTripTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.store = MemoryArtifactStore()
        self.cache = NuggetArtifactCache(self.store, clock=lambda: 1234.5)
        self.bundle = build_bundle()
        self.fingerprint = fingerprint()
        self.request = request(self.bundle)

    def test_writes_then_reads_the_same_bundle(self) -> None:
        self.assertIs(self.cache.get(self.fingerprint, self.request), None)
        self.assertTrue(self.cache.put(self.fingerprint, self.request, self.bundle))
        self.assertEqual(self.cache.get(self.fingerprint, self.request), self.bundle)

    def test_reports_a_refused_write(self) -> None:
        self.store.write_result = False
        self.assertFalse(self.cache.put(self.fingerprint, self.request, self.bundle))
        self.assertIs(self.cache.get(self.fingerprint, self.request), None)

    def test_stored_bytes_are_stable_across_runs(self) -> None:
        self.cache.put(self.fingerprint, self.request, self.bundle)
        first = dict(self.store.objects)
        second_store = MemoryArtifactStore()
        NuggetArtifactCache(second_store, clock=lambda: 1234.5).put(
            self.fingerprint, request(build_bundle()), build_bundle()
        )
        self.assertEqual(first, second_store.objects)

    def test_rejects_a_non_finite_clock(self) -> None:
        cache = NuggetArtifactCache(self.store, clock=lambda: float("inf"))
        with self.assertRaises(ValueError):
            cache.put(self.fingerprint, self.request, self.bundle)

    def test_stores_no_passage_text_or_provider_material(self) -> None:
        """Offsets and hashes prove the source; the corpus keeps the text."""

        self.cache.put(self.fingerprint, self.request, self.bundle)
        stored = b"".join(self.store.objects.values())
        for leak in (b"grain passage", b"port passage", b"OPENAI", b"sk-", b"Bearer"):
            self.assertNotIn(leak, stored)
        # The claim and the lineage that proves it are present.
        self.assertIn(b"grain exports fell", stored)
        self.assertIn(self.bundle.evidence[0].content_hash, stored)


class IsolationTestCase(unittest.TestCase):
    """Any result-affecting change must retire the previous object."""

    def setUp(self) -> None:
        self.store = MemoryArtifactStore()
        self.cache = NuggetArtifactCache(self.store, clock=lambda: 1.0)
        self.bundle = build_bundle()
        self.fingerprint = fingerprint()
        self.request = request(self.bundle)
        self.cache.put(self.fingerprint, self.request, self.bundle)

    def test_a_changed_fingerprint_never_hits(self) -> None:
        variants = {
            "package": {"name": "nuggetizer", "version": "0.0.6"},
            "nuggetizer_config": {"model": "other-model", "max_nuggets": 20},
            "provider_base_url_digest": hashlib.sha256(b"https://other/v1").digest(),
            "embedding_revision": "0" * 40,
            "similarity_threshold": 0.85,
            "guard_version": "guard-v2",
            "evidence_budget": 12,
            "output_limits": {
                "max_claim_characters": 400,
                "max_claim_words": 12,
                "max_completion_tokens": 2048,
            },
        }
        for field, value in variants.items():
            with self.subTest(field=field):
                other = fingerprint(**{field: value})
                self.assertNotEqual(other.digest, self.fingerprint.digest)
                self.assertIs(self.cache.get(other, self.request), None)

    def test_a_changed_request_never_hits(self) -> None:
        variants = {
            "topic_id": "2001",
            "query": "port closures",
            "representation": "canonical",
            "passage_source_fingerprint": "langchain-spacy-v1/feedface",
            "evidence_digest": hashlib.sha256(b"other-evidence").digest(),
            "inputs": (
                NuggetInput(
                    facet_index=0,
                    query="grain exports",
                    object_key="retrieval/v1/aa/bb/other.mpk",
                    request_digest=hashlib.sha256(b"retrieval-request").digest(),
                ),
            ),
        }
        for field, value in variants.items():
            with self.subTest(field=field):
                other = request(self.bundle, **{field: value})
                self.assertNotEqual(other.digest, self.request.digest)
                self.assertIs(self.cache.get(self.fingerprint, other), None)


class CorruptionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.store = MemoryArtifactStore()
        self.cache = NuggetArtifactCache(self.store, clock=lambda: 1.0)
        self.bundle = build_bundle()
        self.fingerprint = fingerprint()
        self.request = request(self.bundle)
        self.cache.put(self.fingerprint, self.request, self.bundle)
        self.key = self.cache.object_key(self.fingerprint, self.request)

    def _expect_repaired_miss(self) -> None:
        self.assertIs(self.cache.get(self.fingerprint, self.request), None)
        self.assertIn(self.key, self.store.evictions)

    def test_truncated_bytes_are_a_miss(self) -> None:
        self.store.objects[self.key] = self.store.objects[self.key][:-4]
        self._expect_repaired_miss()

    def test_foreign_bytes_are_a_miss(self) -> None:
        self.store.objects[self.key] = b"not msgpack at all"
        self._expect_repaired_miss()

    def test_an_envelope_written_under_another_identity_is_a_miss(self) -> None:
        envelope = msgspec.msgpack.decode(
            self.store.objects[self.key], type=NuggetEnvelope
        )
        self.store.objects[self.key] = msgspec.msgpack.encode(
            msgspec.structs.replace(
                envelope, fingerprint=hashlib.sha256(b"elsewhere").digest()
            )
        )
        self._expect_repaired_miss()

    def test_tampered_evidence_is_a_miss(self) -> None:
        envelope = msgspec.msgpack.decode(
            self.store.objects[self.key], type=NuggetEnvelope
        )
        evidence = envelope.bundle.evidence
        altered = msgspec.structs.replace(evidence[0], document_id="doc-forged")
        self.store.objects[self.key] = msgspec.msgpack.encode(
            msgspec.structs.replace(
                envelope,
                bundle=msgspec.structs.replace(
                    envelope.bundle, evidence=(altered,) + evidence[1:]
                ),
            )
        )
        self._expect_repaired_miss()

    def test_a_non_canonical_encoding_is_a_miss(self) -> None:
        envelope = msgspec.msgpack.decode(
            self.store.objects[self.key], type=NuggetEnvelope
        )
        payload = msgspec.msgpack.decode(self.store.objects[self.key])
        # Re-encode with a reordered mapping so the bytes differ but decode.
        self.store.objects[self.key] = msgspec.msgpack.encode(
            {key: payload[key] for key in reversed(list(payload))}
        )
        self.assertIsNotNone(envelope)
        self._expect_repaired_miss()


class PutValidationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.cache = NuggetArtifactCache(MemoryArtifactStore(), clock=lambda: 1.0)
        self.bundle = build_bundle()

    def test_refuses_a_bundle_for_another_topic(self) -> None:
        other = request(self.bundle, topic_id="2001")
        with self.assertRaisesRegex(NuggetError, "different topic"):
            self.cache.put(fingerprint(), other, self.bundle)

    def test_refuses_a_bundle_answering_another_query(self) -> None:
        other = request(self.bundle, query="port closures")
        with self.assertRaisesRegex(NuggetError, "different query"):
            self.cache.put(fingerprint(), other, self.bundle)

    def test_refuses_a_bundle_whose_evidence_was_swapped(self) -> None:
        other = request(self.bundle, evidence_digest=hashlib.sha256(b"other").digest())
        with self.assertRaisesRegex(NuggetError, "requested alias table"):
            self.cache.put(fingerprint(), other, self.bundle)

    def test_refuses_a_bundle_that_contradicts_the_selection_policy(self) -> None:
        with self.assertRaisesRegex(NuggetError, "selection fingerprint"):
            self.cache.put(
                fingerprint(evidence_budget=99), request(self.bundle), self.bundle
            )

    def test_refuses_a_bundle_covering_other_facets(self) -> None:
        other = request(
            self.bundle,
            inputs=(
                NuggetInput(
                    facet_index=3,
                    query="other",
                    object_key="retrieval/v1/aa/bb/cc.mpk",
                    request_digest=hashlib.sha256(b"retrieval-request").digest(),
                ),
            ),
        )
        with self.assertRaisesRegex(NuggetError, "different facet set"):
            self.cache.put(fingerprint(), other, self.bundle)


if __name__ == "__main__":
    unittest.main()


class FingerprintCoverageTestCase(unittest.TestCase):
    """Every result-affecting constant is fingerprinted, or denied on purpose.

    Three separate commits fixed "a result-affecting input is missing from the
    artifact identity". A field-by-field isolation test cannot catch the next
    one, because a newly added constant is simply absent from every list. This
    fails closed instead: a new tunable constant must be either covered or
    explicitly excused.
    """

    # Constants whose value cannot change a stored bundle, with the reason.
    EXCUSED = {
        "trec_ragtime.nuggets": {
            "SCHEMA_VERSION": "fingerprinted directly",
            "BEHAVIOR_VERSION": "fingerprinted directly",
            "EMBEDDING_MODEL": "fingerprinted directly",
            "EMBEDDING_REVISION": "fingerprinted directly",
            "GUARD_VERSION": "fingerprinted directly",
            "DEFAULT_EVIDENCE_BUDGET": "a default for the fingerprinted budget",
            "DEFAULT_MAX_NUGGETS": "a default for the fingerprinted maximum",
            "DEFAULT_SIMILARITY_THRESHOLD": "a default for the fingerprinted threshold",
            "MAX_CLAIM_CHARACTERS": "fingerprinted via output_limits",
            "MAX_CLAIM_WORDS": "fingerprinted via output_limits",
            "SUPPORT_LABELS": "changing it changes behaviour, so bump BEHAVIOR_VERSION",
        },
        "trec_ragtime.nuggetizer_adapter": {
            "PACKAGE_NAME": "identifies the package already fingerprinted by version",
            "MAX_COMPLETION_TOKENS": "fingerprinted via output_limits",
            "DEFAULT_MAX_CALLS": "a default for the fingerprinted nuggetizer_config",
            "REQUEST_TIMEOUT_SECONDS": "a timeout aborts a call, it never alters a result",
        },
    }

    def test_no_result_affecting_constant_escapes_the_fingerprint(self) -> None:
        import importlib

        fingerprint = globals()['fingerprint']()
        encoded = msgspec.msgpack.encode(fingerprint)
        for module_name, excused in self.EXCUSED.items():
            module = importlib.import_module(module_name)
            declared = {
                name
                for name in vars(module)
                if name.isupper() and not name.startswith("_")
            }
            uncovered = declared - set(excused)
            self.assertEqual(
                uncovered,
                set(),
                f"{module_name} declares {sorted(uncovered)}; add each to "
                "NuggetFingerprint or excuse it here with a reason",
            )
            for name in ("MAX_CLAIM_CHARACTERS", "MAX_CLAIM_WORDS"):
                if name in declared:
                    self.assertIn(name.lower(), dict(fingerprint.output_limits))
        self.assertIn(b"output_limits", encoded)
