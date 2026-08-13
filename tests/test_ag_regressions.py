"""Regressions found by adversarially verifying the architecture against source."""

from __future__ import annotations

import unittest
from typing import Any

import msgspec

from trec_ragtime.artifact_store import ArtifactStoreRemoteError
from trec_ragtime.evidence_bundle import build_evidence_bundle_from_handoff
from trec_ragtime.report_generation import SentenceCandidate, rank_citations
from trec_ragtime.report_handoff import HandoffSourceError, HandoffTopicEntry, require_handoff_bundle
from trec_ragtime.retrieval_handoff import HandoffSupport
from tests.test_evidence_handoff import TOPICS_DIGEST, evidence, handoff, nugget
from tests.test_report_cache import MemoryArtifactStore


def bundle_with_ranks(*ranks: int, citation_ready: bool = True):
    rows = tuple(
        msgspec.structs.replace(evidence(f"e00{i}", i), document_rank=rank)
        for i, rank in enumerate(ranks, start=1)
    )
    support = tuple(
        HandoffSupport(
            alias=f"e00{i}", assignment="support", variant_index=0, citation_ready=citation_ready
        )
        for i in range(1, len(ranks) + 1)
    )
    nuggets = (nugget("a", "A claim under test.", support),)
    return build_evidence_bundle_from_handoff(
        handoff(evidence_rows=rows, nuggets=nuggets), topics_sha256=TOPICS_DIGEST
    )


class CitationScoreDescentTests(unittest.TestCase):
    """Scores must strictly descend for ANY retrieval rank, not just small ones.

    Selected handoff evidence keeps its original corpus document_rank, which is
    unbounded relative to the number of selected passages, so a rank-derived
    tie-break must be normalized by the observed rank span rather than by the
    passage count.
    """

    def scores(self, bundle, handles):
        candidate = SentenceCandidate(
            text="A claim under test.", handles=handles, facet_ids=("F1",), substantive=True
        )
        return [c.score for c in rank_citations(candidate, bundle, max_citations=3)]

    def test_descends_with_a_high_document_rank(self) -> None:
        bundle = bundle_with_ranks(1, 87)
        for handles in (("e001", "e002"), ("e002", "e001")):
            with self.subTest(handles=handles):
                scores = self.scores(bundle, handles)
                self.assertEqual(scores, sorted(scores, reverse=True), scores)
                self.assertEqual(len(set(scores)), len(scores))

    def test_descends_with_extreme_document_ranks(self) -> None:
        bundle = bundle_with_ranks(3, 970, 9999)
        scores = self.scores(bundle, ("e003", "e001", "e002"))
        self.assertEqual(scores, sorted(scores, reverse=True), scores)
        self.assertEqual(len(set(scores)), len(scores))

    def test_all_scores_stay_finite_and_positive(self) -> None:
        import math

        scores = self.scores(bundle_with_ranks(1, 5000), ("e002", "e001"))
        self.assertTrue(all(math.isfinite(s) for s in scores), scores)
        self.assertTrue(all(s > 0 for s in scores), scores)

    def test_better_retrieval_rank_still_breaks_ties_within_a_position(self) -> None:
        # Same model position for both, so only rank distinguishes them.
        bundle = bundle_with_ranks(1, 50)
        first = self.scores(bundle, ("e001",))[0]
        second = self.scores(bundle, ("e002",))[0]
        self.assertGreater(first, second)


class PartialSupportFacetTests(unittest.TestCase):
    """A facet nothing can cite must not be presented as a coverage target."""

    def only_partial(self):
        rows = (evidence("e001", 1), evidence("e002", 2, facet_index=1))
        nuggets = (
            nugget(
                "a",
                "A fully attested claim.",
                (HandoffSupport(alias="e001", assignment="support", variant_index=0, citation_ready=True),),
            ),
            nugget(
                "b",
                "A claim only partially attested.",
                (HandoffSupport(alias="e002", assignment="partial_support", variant_index=0, citation_ready=False),),
            ),
        )
        return build_evidence_bundle_from_handoff(
            handoff(evidence_rows=rows, nuggets=nuggets), topics_sha256=TOPICS_DIGEST
        )

    def test_partial_only_facet_is_not_a_coverage_target(self) -> None:
        bundle = self.only_partial()
        self.assertEqual([f.text for f in bundle.facets], ["A fully attested claim."])

    def test_every_retained_facet_has_at_least_one_citable_passage(self) -> None:
        bundle = self.only_partial()
        citable_facets = {
            facet_id
            for passage in bundle.passages
            if passage.citable
            for facet_id in passage.facet_ids
        }
        self.assertEqual(set(bundle.facet_ids), citable_facets)

    def test_partial_evidence_survives_as_context(self) -> None:
        bundle = self.only_partial()
        self.assertIn("e002", bundle.handle_allowlist)
        self.assertFalse(bundle.passage("e002").citable)

    def test_a_partial_link_to_a_retained_facet_keeps_its_label(self) -> None:
        rows = (evidence("e001", 1), evidence("e002", 2, facet_index=1))
        nuggets = (
            nugget(
                "a",
                "A claim.",
                (
                    HandoffSupport(alias="e001", assignment="support", variant_index=0, citation_ready=True),
                    HandoffSupport(alias="e002", assignment="partial_support", variant_index=0, citation_ready=False),
                ),
            ),
        )
        bundle = build_evidence_bundle_from_handoff(
            handoff(evidence_rows=rows, nuggets=nuggets), topics_sha256=TOPICS_DIGEST
        )
        link = bundle.passage("e002").support[0]
        self.assertEqual(link.assignment, "partial_support")
        self.assertFalse(link.citation_ready)


class HandoffFailureReportingTests(unittest.TestCase):
    def test_an_invalid_artifact_is_not_reported_as_absent(self) -> None:
        class RetainingStore(MemoryArtifactStore):
            """Models a real remote: a rejected payload stays durably stored."""

            def read_validated(self, object_key: str, decoder: Any) -> Any:
                payload = self.objects.get(object_key)
                if payload is None:
                    return None
                try:
                    return decoder(payload)
                except Exception:
                    return None

        store = RetainingStore()
        key = "retrieval-handoff/v1/" + "a" * 64 + "/bb/" + "b" * 64 + ".mpk"
        store.objects[key] = b"\x00 not a valid envelope"
        entry = HandoffTopicEntry("2000", "written", key)
        with self.assertRaises(HandoffSourceError) as caught:
            require_handoff_bundle(store, entry)
        self.assertIn("invalid", str(caught.exception).lower())
        self.assertNotIn("absent", str(caught.exception).lower())

    def test_a_genuinely_missing_artifact_is_reported_as_absent(self) -> None:
        entry = HandoffTopicEntry("2000", "written", "retrieval-handoff/v1/missing.mpk")
        with self.assertRaises(HandoffSourceError) as caught:
            require_handoff_bundle(MemoryArtifactStore(), entry)
        self.assertIn("absent", str(caught.exception).lower())

    def test_a_remote_store_failure_is_surfaced_not_swallowed(self) -> None:
        store = MemoryArtifactStore()
        store.read_error = ArtifactStoreRemoteError("bucket throttled")
        entry = HandoffTopicEntry("2000", "written", "retrieval-handoff/v1/k.mpk")
        with self.assertRaises((HandoffSourceError, ArtifactStoreRemoteError)):
            require_handoff_bundle(store, entry)


if __name__ == "__main__":
    unittest.main()
