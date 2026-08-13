from __future__ import annotations

import unittest

import msgspec

from trec_ragtime.evidence_bundle import (
    EvidenceBundleError,
    build_evidence_bundle_from_handoff,
)
from trec_ragtime.retrieval_handoff import (
    BEHAVIOR_VERSION,
    HandoffCounts,
    HandoffEvidence,
    HandoffFacet,
    HandoffNugget,
    HandoffNuggetSource,
    HandoffRetrievalInput,
    HandoffSupport,
    HandoffTopic,
    RetrievalHandoffBundle,
)

TOPICS_DIGEST = "c" * 64


def evidence(alias: str, index: int, *, facet_index: int = 0) -> HandoffEvidence:
    text = f"English passage {index} about the ban and its measured outcomes."
    import hashlib

    return HandoffEvidence(
        alias=alias,
        document_id=f"doc-{index}",
        text=text,
        representation="report_text",
        language="eng",
        start_char=0,
        end_char=len(text),
        content_hash=hashlib.sha256(text.encode("utf-8")).digest(),
        passage_score=1.0 / index,
        document_rank=index,
        facet_index=facet_index,
        facet_query=f"facet query {facet_index}",
        retrieval_input_index=facet_index,
    )


def nugget(
    suffix: str,
    claim: str,
    support: tuple[HandoffSupport, ...],
    *,
    importance: str = "okay",
) -> HandoffNugget:
    return HandoffNugget(
        nugget_id=f"nug1_{suffix * 24}"[:29],
        cluster_id=f"nugc1_{suffix * 24}"[:30],
        claim=claim,
        variants=(claim,),
        importance=importance,
        support=support,
    )


def handoff(
    *,
    evidence_rows: tuple[HandoffEvidence, ...] | None = None,
    nuggets: tuple[HandoffNugget, ...] | None = None,
    limit: int = 5000,
) -> RetrievalHandoffBundle:
    if evidence_rows is None:
        evidence_rows = (evidence("e001", 1), evidence("e002", 2, facet_index=1))
    if nuggets is None:
        nuggets = (
        nugget("a", "The ban took effect in 2019.",
               (HandoffSupport(alias="e001", assignment="support", variant_index=0, citation_ready=True),),
               importance="vital"),
        nugget("b", "Youth vaping prevalence fell.",
               (HandoffSupport(alias="e002", assignment="support", variant_index=0, citation_ready=True),)),
        )
    aliases = {row.alias for row in evidence_rows}
    full = sum(1 for n in nuggets for s in n.support if s.assignment == "support")
    partial = sum(1 for n in nuggets for s in n.support if s.assignment == "partial_support")
    ready = sum(1 for n in nuggets for s in n.support if s.citation_ready)
    return RetrievalHandoffBundle(
        schema_version=1,
        behavior_version=BEHAVIOR_VERSION,
        representation="report_text",
        topic=HandoffTopic(
            topic_id="2000",
            collection_id="ragtime/2/all",
            title="Anti-vaping legislation",
            problem_statement="A statement about vaping law.",
            background="A legislative aide needs this.",
            limit=limit,
        ),
        retrieval_inputs=(
            HandoffRetrievalInput(facet_index=0, query="facet query 0", object_key="retrieval/v1/a", request_digest=b"\x01" * 32),
            HandoffRetrievalInput(facet_index=1, query="facet query 1", object_key="retrieval/v1/b", request_digest=b"\x02" * 32),
        ),
        nugget_source=HandoffNuggetSource(
            object_key="nuggets/v1/x",
            fingerprint_digest=b"\x03" * 32,
            request_digest=b"\x04" * 32,
            bundle_digest=b"\x05" * 32,
        ),
        evidence=evidence_rows,
        nuggets=nuggets,
        facets=(
            HandoffFacet(facet_index=0, query="facet query 0", evidence_count=1, nugget_count=1),
            HandoffFacet(facet_index=1, query="facet query 1", evidence_count=1, nugget_count=1),
        ),
        counts=HandoffCounts(
            offered_evidence=len(aliases),
            selected_evidence=len(evidence_rows),
            omitted_evidence=0,
            nuggets=len(nuggets),
            full_support=full,
            partial_support=partial,
            citation_ready=ready,
            rejected=0,
        ),
    )


class ProjectionTests(unittest.TestCase):
    def project(self, **kwargs):
        return build_evidence_bundle_from_handoff(
            kwargs.pop("bundle", handoff()), topics_sha256=TOPICS_DIGEST, **kwargs
        )

    def test_topic_record_is_carried_verbatim(self) -> None:
        item = self.project()
        self.assertEqual(item.topic_id, "2000")
        self.assertEqual(item.collection_id, "ragtime/2/all")
        self.assertEqual(item.title, "Anti-vaping legislation")
        self.assertEqual(item.problem_statement, "A statement about vaping law.")
        self.assertEqual(item.background, "A legislative aide needs this.")

    def test_character_limit_comes_from_the_handoff_topic(self) -> None:
        self.assertEqual(self.project(bundle=handoff(limit=4200)).character_limit, 4200)

    def test_nuggets_become_facets_with_real_importance(self) -> None:
        item = self.project()
        self.assertEqual([f.text for f in item.facets],
                         ["The ban took effect in 2019.", "Youth vaping prevalence fell."])
        self.assertEqual([f.importance for f in item.facets], ["vital", "okay"])
        self.assertEqual({f.source for f in item.facets}, {"nugget_bundle_v1"})

    def test_upstream_aliases_are_preserved_not_renumbered(self) -> None:
        item = self.project()
        self.assertEqual(item.handles, ("e001", "e002"))
        self.assertEqual(item.resolve_handle("e001"), "doc-1")

    def test_evidence_text_is_carried_and_hashes_match(self) -> None:
        item = self.project()
        passage = item.passage("e001")
        self.assertTrue(passage.report_text.startswith("English passage 1"))
        source = handoff().evidence[0]
        self.assertEqual(passage.text_sha256, source.content_hash.hex())

    def test_support_links_carry_assignment_and_variant_index(self) -> None:
        passage = self.project().passage("e001")
        self.assertEqual(len(passage.support), 1)
        self.assertEqual(passage.support[0].assignment, "support")
        self.assertEqual(passage.support[0].variant_index, 0)
        self.assertTrue(passage.support[0].citation_ready)

    def test_nugget_source_digest_is_recorded(self) -> None:
        self.assertEqual(self.project().source.nugget_bundle_sha256, (b"\x05" * 32).hex())
        self.assertEqual(self.project().source.run_schema, "retrieval_handoff_v1")

    def test_seal_is_deterministic_and_content_sensitive(self) -> None:
        self.assertEqual(self.project().seal, self.project().seal)
        self.assertNotEqual(self.project(bundle=handoff(limit=4000)).seal, self.project().seal)


class CitationReadinessTests(unittest.TestCase):
    def partial_bundle(self) -> RetrievalHandoffBundle:
        rows = (evidence("e001", 1), evidence("e002", 2, facet_index=1))
        nuggets = (
            nugget("a", "The ban took effect in 2019.",
                   (HandoffSupport(alias="e001", assignment="support", variant_index=0, citation_ready=True),
                    HandoffSupport(alias="e002", assignment="partial_support", variant_index=0, citation_ready=False))),
        )
        return handoff(evidence_rows=rows, nuggets=nuggets)

    def test_partial_support_evidence_is_present_but_not_citable(self) -> None:
        item = build_evidence_bundle_from_handoff(self.partial_bundle(), topics_sha256=TOPICS_DIGEST)
        self.assertEqual(item.handles, ("e001", "e002"))
        self.assertTrue(item.passage("e001").citable)
        self.assertFalse(item.passage("e002").citable)

    def test_only_citable_handles_are_offered_for_citation(self) -> None:
        item = build_evidence_bundle_from_handoff(self.partial_bundle(), topics_sha256=TOPICS_DIGEST)
        self.assertEqual(item.citable_handles, ("e001",))
        self.assertEqual(item.handle_allowlist, frozenset({"e001", "e002"}))

    def test_partial_support_link_is_retained_with_its_label(self) -> None:
        item = build_evidence_bundle_from_handoff(self.partial_bundle(), topics_sha256=TOPICS_DIGEST)
        link = item.passage("e002").support[0]
        self.assertEqual(link.assignment, "partial_support")
        self.assertFalse(link.citation_ready)

    def test_non_citable_passage_earns_no_facet_credit(self) -> None:
        item = build_evidence_bundle_from_handoff(self.partial_bundle(), topics_sha256=TOPICS_DIGEST)
        self.assertEqual(item.passage("e002").facet_ids, ())
        self.assertEqual(item.passage("e001").facet_ids, ("F1",))

    def test_a_topic_with_no_citable_evidence_fails_closed(self) -> None:
        rows = (evidence("e001", 1),)
        nuggets = (
            nugget("a", "A partially attested claim.",
                   (HandoffSupport(alias="e001", assignment="partial_support", variant_index=0, citation_ready=False),)),
        )
        with self.assertRaises(EvidenceBundleError):
            build_evidence_bundle_from_handoff(
                handoff(evidence_rows=rows, nuggets=nuggets), topics_sha256=TOPICS_DIGEST
            )


class HandoffValidationTests(unittest.TestCase):
    def test_invalid_handoff_is_rejected(self) -> None:
        with self.assertRaises(EvidenceBundleError):
            build_evidence_bundle_from_handoff(object(), topics_sha256=TOPICS_DIGEST)

    def test_support_link_to_an_unknown_alias_fails_closed(self) -> None:
        rows = (evidence("e001", 1),)
        nuggets = (
            nugget("a", "A claim.",
                   (HandoffSupport(alias="e999", assignment="support", variant_index=0, citation_ready=True),)),
        )
        with self.assertRaises(EvidenceBundleError):
            build_evidence_bundle_from_handoff(
                handoff(evidence_rows=rows, nuggets=nuggets), topics_sha256=TOPICS_DIGEST
            )

    def test_evidence_text_not_matching_its_hash_fails_closed(self) -> None:
        rows = list(handoff().evidence)
        rows[0] = msgspec.structs.replace(rows[0], text="Swapped text.")
        with self.assertRaises(EvidenceBundleError):
            build_evidence_bundle_from_handoff(
                handoff(evidence_rows=tuple(rows)), topics_sha256=TOPICS_DIGEST
            )

    def test_empty_nugget_set_fails_closed(self) -> None:
        with self.assertRaises(EvidenceBundleError):
            build_evidence_bundle_from_handoff(
                handoff(nuggets=()), topics_sha256=TOPICS_DIGEST
            )


if __name__ == "__main__":
    unittest.main()
