from __future__ import annotations

import hashlib
import unittest
from collections.abc import Mapping, Sequence

import msgspec

from trec_ragtime.evidence_bundle import (
    EvidenceBundleError,
    build_evidence_bundle_from_handoff,
)
from trec_ragtime.nugget_cache import nugget_object_key
from trec_ragtime.nuggets import SCHEMA_VERSION as NUGGET_SCHEMA_VERSION, normalize_claim
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
TOPIC_ID = "2000"
NUGGET_FINGERPRINT = b"\x03" * 32
NUGGET_REQUEST = b"\x04" * 32
NUGGET_BUNDLE_DIGEST = b"\x05" * 32

# One nugget as a test cares about it: the claim, how important it is, and the
# evidence it cites as ``(alias, assignment)``.
ClaimSpec = tuple[str, str, Sequence[tuple[str, str]]]


def facet_query(facet_index: int) -> str:
    return f"facet query {facet_index}"


def evidence(
    alias: str,
    index: int,
    *,
    facet_index: int = 0,
    document_rank: int | None = None,
) -> HandoffEvidence:
    """Build one selected evidence row whose offsets and hash agree with its text."""

    text = f"English passage {index} about the ban and its measured outcomes."
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
        document_rank=index if document_rank is None else document_rank,
        facet_index=facet_index,
        facet_query=facet_query(facet_index),
        retrieval_input_index=facet_index,
    )


def support(alias: str, assignment: str = "support", *, variant_index: int = 0) -> HandoffSupport:
    """Link a nugget to evidence.

    check_retrieval_handoff requires citation_ready to equal
    ``assignment == "support"`` exactly, so it is derived rather than passed in.
    """

    return HandoffSupport(
        alias=alias,
        assignment=assignment,
        variant_index=variant_index,
        citation_ready=assignment == "support",
    )


def nugget(
    suffix: str,
    claim: str,
    support_links: Sequence[HandoffSupport],
    *,
    importance: str = "okay",
    topic_id: str = TOPIC_ID,
) -> HandoffNugget:
    """Build a nugget whose IDs match what the handoff validator recomputes.

    check_retrieval_handoff derives nugget_id and cluster_id from the topic,
    the normalized claim, and the cited aliases, so a fixture with invented
    identifiers is rejected as broken lineage. Variants are normalized claims,
    because the validator requires every variant to equal normalize_claim of
    itself and requires the normalized claim to appear among them.
    """

    del suffix  # A readability label for the caller; identity comes from the claim.
    links = tuple(sorted(support_links, key=lambda link: link.alias))
    aliases = tuple(link.alias for link in links)
    variants = (normalize_claim(claim),)
    nugget_id = "nug1_" + hashlib.sha256(
        msgspec.msgpack.encode(
            {
                "schema_version": NUGGET_SCHEMA_VERSION,
                "topic_id": topic_id,
                "claim": normalize_claim(claim),
                "aliases": sorted(aliases),
            }
        )
    ).hexdigest()[:24]
    cluster_id = "nugc1_" + hashlib.sha256(
        msgspec.msgpack.encode(
            {
                "schema_version": NUGGET_SCHEMA_VERSION,
                "topic_id": topic_id,
                "variants": sorted(variants),
            }
        )
    ).hexdigest()[:24]
    return HandoffNugget(
        nugget_id=nugget_id,
        cluster_id=cluster_id,
        claim=claim,
        variants=variants,
        importance=importance,
        support=links,
    )


def handoff(
    *,
    evidence_rows: tuple[HandoffEvidence, ...] | None = None,
    nuggets: tuple[HandoffNugget, ...] | None = None,
    limit: int = 5000,
) -> RetrievalHandoffBundle:
    """Assemble a bundle whose derived parts agree with its evidence and nuggets.

    check_retrieval_handoff recomputes the retrieval inputs, the facet coverage,
    the nugget order and every count from the bundle contents, so this fixture
    derives all of them instead of hand-writing them. A caller supplies only the
    evidence and nuggets a test is actually about; anything else holds by
    construction, and a defect a test injects on purpose survives untouched.
    """

    if evidence_rows is None:
        evidence_rows = (evidence("e001", 1), evidence("e002", 2, facet_index=1))
    if nuggets is None:
        nuggets = (
            nugget(
                "a",
                "The ban took effect in 2019.",
                (support("e001"),),
                importance="vital",
            ),
            nugget("b", "Youth vaping prevalence fell.", (support("e002"),)),
        )
    evidence_rows = tuple(sorted(evidence_rows, key=lambda row: row.alias))
    scores = {row.alias: row.passage_score for row in evidence_rows}
    # The validator demands vital before okay, then the strongest cited passage,
    # then the nugget id. A missing alias is a defect the validator reports on
    # its own, so ordering must not raise before it gets there.
    nuggets = tuple(
        sorted(
            nuggets,
            key=lambda row: (
                0 if row.importance == "vital" else 1,
                -max((scores.get(link.alias, 0.0) for link in row.support), default=0.0),
                row.nugget_id,
            ),
        )
    )

    # Retrieval inputs cover facet 0..max contiguously so that every row's
    # retrieval_input_index equals its facet_index, as the validator checks.
    facet_indices = tuple(
        range(max((row.facet_index for row in evidence_rows), default=0) + 1)
    )
    retrieval_inputs = tuple(
        HandoffRetrievalInput(
            facet_index=index,
            query=facet_query(index),
            object_key=f"retrieval/v1/{index}",
            request_digest=bytes([index + 1]) * 32,
        )
        for index in facet_indices
    )
    selected: dict[int, set[str]] = {
        index: {row.alias for row in evidence_rows if row.facet_index == index}
        for index in facet_indices
    }
    facets = tuple(
        HandoffFacet(
            facet_index=index,
            query=facet_query(index),
            evidence_count=len(selected[index]),
            nugget_count=sum(
                1
                for row in nuggets
                if any(link.alias in selected[index] for link in row.support)
            ),
        )
        for index in facet_indices
    )

    offered = sum(facet.evidence_count for facet in facets)
    full = sum(1 for n in nuggets for s in n.support if s.assignment == "support")
    partial = sum(
        1 for n in nuggets for s in n.support if s.assignment == "partial_support"
    )
    return RetrievalHandoffBundle(
        schema_version=1,
        behavior_version=BEHAVIOR_VERSION,
        representation="report_text",
        topic=HandoffTopic(
            topic_id=TOPIC_ID,
            collection_id="ragtime/2/all",
            title="Anti-vaping legislation",
            problem_statement="A statement about vaping law.",
            background="A legislative aide needs this.",
            limit=limit,
        ),
        retrieval_inputs=retrieval_inputs,
        nugget_source=HandoffNuggetSource(
            object_key=nugget_object_key(NUGGET_FINGERPRINT, NUGGET_REQUEST),
            fingerprint_digest=NUGGET_FINGERPRINT,
            request_digest=NUGGET_REQUEST,
            bundle_digest=NUGGET_BUNDLE_DIGEST,
        ),
        evidence=evidence_rows,
        nuggets=nuggets,
        facets=facets,
        counts=HandoffCounts(
            offered_evidence=offered,
            selected_evidence=len(evidence_rows),
            omitted_evidence=offered - len(evidence_rows),
            nuggets=len(nuggets),
            full_support=full,
            partial_support=partial,
            citation_ready=full,
            rejected=0,
        ),
    )


def described_handoff(
    *claims: ClaimSpec,
    limit: int = 5000,
    ranks: Mapping[str, int] | None = None,
) -> RetrievalHandoffBundle:
    """Build a valid bundle from a plain description of the nuggets under test.

    Each claim is ``(text, importance, [(alias, assignment), ...])``. Evidence
    rows are created for exactly the aliases the claims cite, because the
    validator requires the evidence table to be the union of referenced
    aliases -- no unreferenced rows and no dangling links. ``ranks`` overrides
    the corpus document_rank of individual aliases.
    """

    ranks = dict(ranks or {})
    aliases = sorted(
        {alias for _, _, links in claims for alias, _ in links},
        key=lambda alias: int(alias[1:]),
    )
    rows = tuple(
        evidence(
            alias,
            int(alias[1:]),
            facet_index=position % 2,
            document_rank=ranks.get(alias),
        )
        for position, alias in enumerate(aliases)
    )
    nuggets = tuple(
        nugget(
            f"n{position}",
            text,
            tuple(support(alias, assignment) for alias, assignment in links),
            importance=importance,
        )
        for position, (text, importance, links) in enumerate(claims)
    )
    return handoff(evidence_rows=rows, nuggets=nuggets, limit=limit)


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
        return described_handoff(
            (
                "The ban took effect in 2019.",
                "okay",
                [("e001", "support"), ("e002", "partial_support")],
            )
        )

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
        bundle = described_handoff(
            ("A partially attested claim.", "okay", [("e001", "partial_support")])
        )
        with self.assertRaises(EvidenceBundleError):
            build_evidence_bundle_from_handoff(bundle, topics_sha256=TOPICS_DIGEST)


class HandoffValidationTests(unittest.TestCase):
    def test_invalid_handoff_is_rejected(self) -> None:
        with self.assertRaises(EvidenceBundleError):
            build_evidence_bundle_from_handoff(object(), topics_sha256=TOPICS_DIGEST)

    def test_support_link_to_an_unknown_alias_fails_closed(self) -> None:
        # Everything else about this bundle is consistent, so the only defect
        # left is the support link naming evidence that was never selected.
        rows = (evidence("e001", 1),)
        nuggets = (nugget("a", "A claim.", (support("e999"),)),)
        with self.assertRaises(EvidenceBundleError):
            build_evidence_bundle_from_handoff(
                handoff(evidence_rows=rows, nuggets=nuggets), topics_sha256=TOPICS_DIGEST
            )

    def test_evidence_text_not_matching_its_hash_fails_closed(self) -> None:
        rows = list(handoff().evidence)
        # Same length as the original, so the offsets stay valid and the content
        # hash is the single invariant this bundle breaks.
        swapped = rows[0].text.replace("English", "Swapped")
        self.assertEqual(len(swapped), len(rows[0].text))
        rows[0] = msgspec.structs.replace(rows[0], text=swapped)
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
