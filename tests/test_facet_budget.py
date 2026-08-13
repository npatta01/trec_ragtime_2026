"""The facet budget is the planning decision, computed without a planning call."""

from __future__ import annotations

import unittest

from trec_ragtime.evidence_bundle import (
    EvidenceBundle,
    EvidenceBundleSource,
    EvidenceFacet,
    EvidencePassage,
    EvidenceSupport,
)
from trec_ragtime.report_cache import BUDGET_POLICY_VERSION, generator_fingerprint
from trec_ragtime.report_generation import allocate_facet_budget
from tests.test_report_cache import settings


def facet(index: int, importance: str = "okay") -> EvidenceFacet:
    return EvidenceFacet(
        facet_id=f"F{index}",
        text=f"claim number {index}",
        source="nugget_bundle_v1",
        importance=importance,
    )


def passage(index: int, facet_ids: tuple[str, ...], *, citable: bool = True) -> EvidencePassage:
    support = tuple(
        EvidenceSupport(
            facet_id=facet_id,
            assignment="support" if citable else "partial_support",
            variant_index=0,
            citation_ready=citable,
        )
        for facet_id in facet_ids
    )
    return EvidencePassage.create(
        handle=f"e{index:03d}",
        document_id=f"doc-{index}",
        report_text=f"Evidence body {index}.",
        language="eng",
        url=None,
        date=None,
        rank=index,
        retrieval_score=1.0 / index,
        facet_ids=facet_ids if citable else (),
        support=support,
        citable=citable,
    )


def bundle(facets, passages, *, limit: int = 5000) -> EvidenceBundle:
    return EvidenceBundle.create(
        topic_id="2000",
        collection_id="ragtime/2/all",
        title="Topic 2000",
        problem_statement="A statement.",
        background="Background.",
        character_limit=limit,
        max_passages=max(len(passages), 1),
        evidence_chars=0,
        facets=tuple(facets),
        passages=tuple(passages),
        source=EvidenceBundleSource(
            run_schema="retrieval_handoff_v1",
            run_id="handoff-v1",
            run_dir_name="nuggets/v1/x",
            manifest_sha256="a" * 64,
            records_sha256="b" * 64,
            topics_sha256="c" * 64,
        ),
    )


def allocate(item, limit=5000, ratio=0.9):
    return allocate_facet_budget(item, limit=limit, budget_ratio=ratio)


class WeightingTests(unittest.TestCase):
    def test_more_citable_evidence_earns_more_budget(self) -> None:
        item = bundle(
            [facet(1), facet(2)],
            [passage(1, ("F1",)), passage(2, ("F1",)), passage(3, ("F2",))],
        )
        budgets = {b.facet_id: b.characters for b in allocate(item)}
        self.assertGreater(budgets["F1"], budgets["F2"])

    def test_self_declared_importance_does_not_move_the_budget(self) -> None:
        # Measured over all 103 topics, "vital" does not predict evidence depth
        # (median 2 full-support links either way) and survives as a citable
        # facet LESS often than "okay" (85.8% vs 96.0%). It is also our label,
        # not the assessors'. Only citable evidence earns budget.
        item = bundle(
            [facet(1, "vital"), facet(2, "okay")],
            [passage(1, ("F1",)), passage(2, ("F2",))],
        )
        budgets = {b.facet_id: b.characters for b in allocate(item)}
        self.assertEqual(budgets["F1"], budgets["F2"])

    def test_evidence_still_outranks_importance(self) -> None:
        item = bundle(
            [facet(1, "vital"), facet(2, "okay")],
            [passage(1, ("F1",)), passage(2, ("F2",)), passage(3, ("F2",))],
        )
        budgets = {b.facet_id: b.characters for b in allocate(item)}
        self.assertGreater(budgets["F2"], budgets["F1"])

    def test_context_only_evidence_earns_no_budget_weight(self) -> None:
        # e003 is partial-support for F2, so it must not lift F2's allocation.
        item = bundle(
            [facet(1), facet(2)],
            [passage(1, ("F1",)), passage(2, ("F2",)), passage(3, ("F2",), citable=False)],
        )
        budgets = {b.facet_id: b.characters for b in allocate(item)}
        self.assertEqual(budgets["F1"], budgets["F2"])

    def test_equal_inputs_allocate_equally(self) -> None:
        item = bundle(
            [facet(1), facet(2)], [passage(1, ("F1",)), passage(2, ("F2",))]
        )
        values = [b.characters for b in allocate(item)]
        self.assertEqual(max(values) - min(values), 0)


class ApportionmentTests(unittest.TestCase):
    def item(self):
        return bundle(
            [facet(1, "vital"), facet(2), facet(3)],
            [
                passage(1, ("F1",)),
                passage(2, ("F1",)),
                passage(3, ("F2",)),
                passage(4, ("F3",)),
            ],
        )

    def test_allocation_sums_exactly_to_the_budget(self) -> None:
        for limit, ratio in ((5000, 0.9), (5000, 1.0), (997, 0.7), (13, 1.0)):
            with self.subTest(limit=limit, ratio=ratio):
                budgets = allocate(self.item(), limit=limit, ratio=ratio)
                expected = max(len(self.item().facets), int(limit * ratio))
                self.assertEqual(sum(b.characters for b in budgets), expected)

    def test_every_facet_receives_at_least_one_character(self) -> None:
        for limit in (5, 13, 5000):
            with self.subTest(limit=limit):
                budgets = allocate(self.item(), limit=limit, ratio=1.0)
                self.assertTrue(all(b.characters >= 1 for b in budgets), budgets)

    def test_order_follows_the_bundle_facets(self) -> None:
        budgets = allocate(self.item())
        self.assertEqual([b.facet_id for b in budgets], ["F1", "F2", "F3"])

    def test_allocation_is_deterministic(self) -> None:
        first = [b.characters for b in allocate(self.item())]
        second = [b.characters for b in allocate(self.item())]
        self.assertEqual(first, second)

    def test_remainder_goes_to_earlier_facets_on_a_tie(self) -> None:
        item = bundle(
            [facet(1), facet(2), facet(3)],
            [passage(1, ("F1",)), passage(2, ("F2",)), passage(3, ("F3",))],
        )
        budgets = [b.characters for b in allocate(item, limit=10, ratio=1.0)]
        self.assertEqual(sum(budgets), 10)
        self.assertEqual(budgets, sorted(budgets, reverse=True))

    def test_budget_carries_the_facet_text(self) -> None:
        budgets = allocate(self.item())
        self.assertEqual(budgets[0].text, "claim number 1")


class FingerprintTests(unittest.TestCase):
    def test_budget_policy_version_is_declared(self) -> None:
        self.assertTrue(BUDGET_POLICY_VERSION.strip())

    def test_budget_policy_version_participates_in_the_fingerprint(self) -> None:
        self.assertNotEqual(
            generator_fingerprint(settings(budget_policy_version="facet-budget-v9")),
            generator_fingerprint(settings()),
        )


if __name__ == "__main__":
    unittest.main()
