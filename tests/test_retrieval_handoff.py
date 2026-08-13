from __future__ import annotations

import hashlib
import unittest
from types import SimpleNamespace

import msgspec

from trec_ragtime.models import PipelineResult, RetrievalResult, RetrievedDocument, SearchHit
from trec_ragtime.nuggets import NuggetSupport
from trec_ragtime.nugget_cache import nugget_object_key
from trec_ragtime.retrieval_handoff import (
    HandoffError,
    HandoffNuggetSource,
    check_retrieval_handoff,
    handoff_bundle_digest,
    project_retrieval_handoff,
)
from trec_ragtime.topic_candidates import Topic

from tests.test_nugget_cache import build_bundle, fingerprint, request


class RetrievalHandoffTests(unittest.TestCase):
    def setUp(self) -> None:
        self.nuggets = build_bundle()
        self.request = request(self.nuggets)
        self.topic = Topic(
            "2000", "ragtime/2/all", "problem", "background", 5000, "title"
        )
        hits = (
            SearchHit(1, "doc-a", 0.9),
            SearchHit(2, "doc-b", 0.5),
        )
        documents = (
            RetrievedDocument(1, "doc-a", 0.9, "grain passage", None, None, "eng", "fixture", True),
            RetrievedDocument(2, "doc-b", 0.5, "port passage", None, None, "eng", "fixture", True),
        )
        result = RetrievalResult(
            PipelineResult("grain exports", "ragtime2%20", hits, True, True, None, None, None),
            documents,
            retrieval_artifact=SimpleNamespace(
                object_key=self.request.inputs[0].object_key,
                request_digest=self.request.inputs[0].request_digest,
            ),
        )
        self.retrievals = {0: result}
        source_fingerprint = fingerprint().digest
        self.source = HandoffNuggetSource(
            object_key=nugget_object_key(source_fingerprint, self.request.digest),
            fingerprint_digest=source_fingerprint,
            request_digest=self.request.digest,
            bundle_digest=hashlib.sha256(msgspec.msgpack.encode(self.nuggets)).digest(),
        )

    def project(self, nuggets=None):
        selected = self.nuggets if nuggets is None else nuggets
        return project_retrieval_handoff(
            topic=self.topic,
            nugget_bundle=selected,
            nugget_request=request(selected),
            nugget_source=msgspec.structs.replace(
                self.source,
                request_digest=request(selected).digest,
                bundle_digest=hashlib.sha256(msgspec.msgpack.encode(selected)).digest(),
            ),
            retrievals=self.retrievals,
        )

    def test_projection_is_self_contained_and_selected_only(self) -> None:
        bundle = self.project()
        self.assertEqual(tuple(row.alias for row in bundle.evidence), ("e001",))
        self.assertEqual(bundle.evidence[0].text, "grain passage")
        self.assertEqual(bundle.counts.omitted_evidence, 1)
        self.assertTrue(bundle.nuggets[0].support[0].citation_ready)
        check_retrieval_handoff(bundle)
        self.assertEqual(len(handoff_bundle_digest(bundle)), 32)

    def test_partial_support_is_retained_but_not_citation_ready(self) -> None:
        nugget = self.nuggets.nuggets[0]
        partial = msgspec.structs.replace(
            nugget,
            support=(NuggetSupport("e001", "partial_support", 0),),
        )
        changed = msgspec.structs.replace(self.nuggets, nuggets=(partial,))
        bundle = self.project(changed)
        self.assertFalse(bundle.nuggets[0].support[0].citation_ready)
        self.assertEqual(bundle.counts.partial_support, 1)

    def test_offset_or_hash_disagreement_fails_closed(self) -> None:
        bad = msgspec.structs.replace(
            self.nuggets.evidence[0], content_hash=b"x" * 32
        )
        changed = msgspec.structs.replace(
            self.nuggets, evidence=(bad, *self.nuggets.evidence[1:])
        )
        with self.assertRaises(ValueError):
            self.project(changed)

    def test_projection_rejects_a_request_that_does_not_pin_bundle_evidence(self) -> None:
        wrong_request = msgspec.structs.replace(
            self.request, evidence_digest=hashlib.sha256(b"other evidence").digest()
        )
        wrong_source = msgspec.structs.replace(
            self.source,
            object_key=nugget_object_key(
                self.source.fingerprint_digest, wrong_request.digest
            ),
            request_digest=wrong_request.digest,
        )

        with self.assertRaisesRegex(HandoffError, "evidence"):
            project_retrieval_handoff(
                topic=self.topic,
                nugget_bundle=self.nuggets,
                nugget_request=wrong_request,
                nugget_source=wrong_source,
                retrievals=self.retrievals,
            )

    def test_self_contained_validator_rejects_negative_variant_index(self) -> None:
        bundle = self.project()
        link = msgspec.structs.replace(
            bundle.nuggets[0].support[0], variant_index=-1
        )
        nugget = msgspec.structs.replace(bundle.nuggets[0], support=(link,))
        with self.assertRaisesRegex(HandoffError, "support"):
            check_retrieval_handoff(
                msgspec.structs.replace(bundle, nuggets=(nugget,))
            )

    def test_self_contained_validator_rejects_invalid_stable_ids(self) -> None:
        bundle = self.project()
        nugget = msgspec.structs.replace(
            bundle.nuggets[0],
            nugget_id="nug1_" + "0" * 24,
            cluster_id="nugc1_" + "0" * 24,
        )
        with self.assertRaisesRegex(HandoffError, "nugget"):
            check_retrieval_handoff(
                msgspec.structs.replace(bundle, nuggets=(nugget,))
            )

    def test_self_contained_validator_rejects_offset_length_mismatch(self) -> None:
        bundle = self.project()
        evidence = msgspec.structs.replace(bundle.evidence[0], end_char=999)
        with self.assertRaisesRegex(HandoffError, "evidence"):
            check_retrieval_handoff(
                msgspec.structs.replace(bundle, evidence=(evidence,))
            )

    def test_self_contained_validator_rejects_schema_type_corruption(self) -> None:
        bundle = self.project()
        topic = msgspec.structs.replace(bundle.topic, limit=True)
        with self.assertRaisesRegex(HandoffError, "schema"):
            check_retrieval_handoff(msgspec.structs.replace(bundle, topic=topic))

    def test_self_contained_validator_rejects_unknown_support_assignment(self) -> None:
        bundle = self.project()
        link = msgspec.structs.replace(
            bundle.nuggets[0].support[0], assignment="unknown", citation_ready=False
        )
        nugget = msgspec.structs.replace(bundle.nuggets[0], support=(link,))
        with self.assertRaisesRegex(HandoffError, "schema"):
            check_retrieval_handoff(msgspec.structs.replace(bundle, nuggets=(nugget,)))

    def test_self_contained_validator_rejects_counts_below_selected_evidence(self) -> None:
        bundle = self.project()
        counts = msgspec.structs.replace(
            bundle.counts, offered_evidence=0, omitted_evidence=-1
        )
        with self.assertRaisesRegex(HandoffError, "counts"):
            check_retrieval_handoff(msgspec.structs.replace(bundle, counts=counts))

    def test_self_contained_validator_rejects_facet_metadata_drift(self) -> None:
        bundle = self.project()
        facet = msgspec.structs.replace(bundle.facets[0], query="different query")
        with self.assertRaisesRegex(HandoffError, "facet"):
            check_retrieval_handoff(msgspec.structs.replace(bundle, facets=(facet,)))

    def test_projection_rejects_non_english_source_evidence(self) -> None:
        evidence = msgspec.structs.replace(self.nuggets.evidence[0], language="spa")
        changed = msgspec.structs.replace(
            self.nuggets, evidence=(evidence, *self.nuggets.evidence[1:])
        )
        changed_request = request(changed)
        changed_source = msgspec.structs.replace(
            self.source,
            object_key=nugget_object_key(
                self.source.fingerprint_digest, changed_request.digest
            ),
            request_digest=changed_request.digest,
            bundle_digest=hashlib.sha256(msgspec.msgpack.encode(changed)).digest(),
        )
        with self.assertRaisesRegex(HandoffError, "language"):
            project_retrieval_handoff(
                topic=self.topic,
                nugget_bundle=changed,
                nugget_request=changed_request,
                nugget_source=changed_source,
                retrievals=self.retrievals,
            )

    def test_projection_rejects_facet_metadata_disagreement(self) -> None:
        facet = msgspec.structs.replace(self.nuggets.facets[0], query="different query")
        changed = msgspec.structs.replace(self.nuggets, facets=(facet,))
        with self.assertRaisesRegex(HandoffError, "facet"):
            self.project(changed)

    def test_retrieval_identity_disagreement_fails_closed(self) -> None:
        result = self.retrievals[0]
        self.retrievals[0] = RetrievalResult(
            result.search_result,
            result.documents,
            retrieval_artifact=SimpleNamespace(
                object_key="wrong", request_digest=self.request.inputs[0].request_digest
            ),
        )
        with self.assertRaisesRegex(HandoffError, "identity"):
            self.project()

    def test_bundle_wire_bytes_are_stable(self) -> None:
        first = msgspec.msgpack.encode(self.project())
        decoded = msgspec.msgpack.decode(first, type=type(self.project()))
        self.assertEqual(msgspec.msgpack.encode(decoded), first)


if __name__ == "__main__":
    unittest.main()
