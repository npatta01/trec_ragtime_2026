"""Evidence linkage, clustering, and fail-closed validation for nuggets."""

from __future__ import annotations

import hashlib
import unittest
from collections.abc import Sequence
from typing import Any

import msgspec

from trec_ragtime.nuggets import (
    EvidenceCandidate,
    FacetEvidence,
    NuggetBundle,
    NuggetClusterer,
    NuggetError,
    NuggetSupport,
    build_evidence_table,
    check_bundle,
    normalize_claim,
    run_topic_nuggets,
)


def candidate(text: str, *, document_id: str = "doc-1", rank: int = 1, score: float = 1.0,
              start: int = 0) -> EvidenceCandidate:
    return EvidenceCandidate(
        document_id=document_id,
        representation="report_text",
        language="eng",
        start_char=start,
        end_char=start + len(text),
        content_hash=hashlib.sha256(text.encode("utf-8")).digest(),
        passage_score=score,
        document_rank=rank,
        source_text=text,
    )


def facet(index: int, *texts: str, **kwargs: Any) -> FacetEvidence:
    return FacetEvidence(
        facet_index=index,
        query=f"facet query {index}",
        candidates=tuple(
            candidate(text, document_id=f"doc-{index}-{position}", rank=position + 1,
                      score=1.0 - position / 10, **kwargs)
            for position, text in enumerate(texts)
        ),
    )


class ScriptedBackend:
    """A nuggetizer stand-in with a fixed claim set and support matrix."""

    def __init__(self, claims: Sequence[tuple[str, str]], supported: dict[str, Sequence[str]]):
        self._claims = tuple(claims)
        self._supported = supported
        self.assign_calls: list[str] = []

    def create(self, *, topic_id: str, query: str, passages: Sequence[tuple[str, str]]):
        self.passages = tuple(passages)
        return self._claims

    def assign(self, *, query: str, passage_text: str, nuggets: Sequence[tuple[str, str]]):
        self.assign_calls.append(passage_text)
        return tuple(
            "support" if passage_text in self._supported.get(claim, ()) else "not_support"
            for claim, _ in nuggets
        )


def fixed_embedder(vectors: dict[str, Sequence[float]]):
    def embed(texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [vectors[text] for text in texts]

    return embed


class EvidenceTableTestCase(unittest.TestCase):
    def test_assigns_sequential_aliases_in_selection_order(self) -> None:
        rows = build_evidence_table([facet(0, "alpha", "beta")], budget=2)
        self.assertEqual([row.evidence.alias for row in rows], ["e001", "e002"])
        self.assertEqual([row.text for row in rows], ["alpha", "beta"])

    def test_round_robins_so_one_facet_cannot_take_the_budget(self) -> None:
        rows = build_evidence_table(
            [facet(0, "a0", "a1", "a2"), facet(1, "b0", "b1")], budget=4
        )
        self.assertEqual([row.evidence.facet_index for row in rows], [0, 1, 0, 1])
        self.assertEqual([row.text for row in rows], ["a0", "b0", "a1", "b1"])

    def test_drains_remaining_facets_once_others_are_exhausted(self) -> None:
        rows = build_evidence_table([facet(0, "a0", "a1", "a2"), facet(1, "b0")], budget=4)
        self.assertEqual([row.text for row in rows], ["a0", "b0", "a1", "a2"])

    def test_stops_at_the_budget(self) -> None:
        rows = build_evidence_table([facet(0, "a0", "a1", "a2")], budget=2)
        self.assertEqual(len(rows), 2)

    def test_drops_a_passage_two_facets_both_retrieved(self) -> None:
        shared = candidate("shared", document_id="doc-9", rank=1)
        facets = [
            FacetEvidence(facet_index=0, query="q0", candidates=(shared,)),
            FacetEvidence(facet_index=1, query="q1", candidates=(shared,)),
        ]
        rows = build_evidence_table(facets, budget=4)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].evidence.facet_index, 0)

    def test_rejects_an_unusable_budget(self) -> None:
        for budget in (0, -1, True):
            with self.subTest(budget=budget):
                with self.assertRaises(NuggetError):
                    build_evidence_table([facet(0, "a")], budget=budget)


class ClustererTestCase(unittest.TestCase):
    def test_merges_exact_duplicates_without_embedding(self) -> None:
        clusterer = NuggetClusterer(embed=fixed_embedder({}))
        self.assertEqual(clusterer.cluster(["Grain fell", "grain   FELL"]), ((0, 1),))

    def test_merges_paraphrases_above_the_threshold(self) -> None:
        claims = ["grain exports fell", "grain shipments declined"]
        clusterer = NuggetClusterer(
            embed=fixed_embedder(
                {normalize_claim(claims[0]): [1.0, 0.0], normalize_claim(claims[1]): [0.97, 0.24]}
            ),
            threshold=0.92,
        )
        self.assertEqual(clusterer.cluster(claims), ((0, 1),))

    def test_keeps_dissimilar_claims_apart(self) -> None:
        claims = ["grain exports fell", "ports reopened"]
        clusterer = NuggetClusterer(
            embed=fixed_embedder(
                {normalize_claim(claims[0]): [1.0, 0.0], normalize_claim(claims[1]): [0.0, 1.0]}
            )
        )
        self.assertEqual(clusterer.cluster(claims), ((0,), (1,)))

    def test_numeric_guard_blocks_a_merge_the_threshold_would_allow(self) -> None:
        claims = ["exports fell 30%", "exports fell 13%"]
        clusterer = NuggetClusterer(
            embed=fixed_embedder(
                {normalize_claim(claims[0]): [1.0, 0.0], normalize_claim(claims[1]): [1.0, 0.0]}
            )
        )
        self.assertEqual(clusterer.cluster(claims), ((0,), (1,)))

    def test_negation_guard_blocks_a_merge_the_threshold_would_allow(self) -> None:
        claims = ["the port reopened", "the port did not reopen"]
        clusterer = NuggetClusterer(
            embed=fixed_embedder(
                {normalize_claim(claims[0]): [1.0, 0.0], normalize_claim(claims[1]): [1.0, 0.0]}
            )
        )
        self.assertEqual(clusterer.cluster(claims), ((0,), (1,)))

    def test_rejects_a_mismatched_embedder(self) -> None:
        clusterer = NuggetClusterer(embed=lambda texts: [[1.0, 0.0]])
        with self.assertRaisesRegex(NuggetError, "wrong number of vectors"):
            clusterer.cluster(["a", "b"])


class StageTestCase(unittest.TestCase):
    facets = [facet(0, "grain passage", "port passage"), facet(1, "tourism passage")]

    def _clusterer(self, **vectors: Sequence[float]) -> NuggetClusterer:
        return NuggetClusterer(
            embed=fixed_embedder({normalize_claim(k): v for k, v in vectors.items()})
        )

    def test_links_each_claim_to_the_passages_that_support_it(self) -> None:
        backend = ScriptedBackend(
            claims=(("grain exports fell", "vital"), ("ports closed", "okay")),
            supported={
                "grain exports fell": ("grain passage", "tourism passage"),
                "ports closed": ("port passage",),
            },
        )
        bundle = run_topic_nuggets(
            topic_id="2000",
            query="grain",
            representation="report_text",
            facets=self.facets,
            backend=backend,
            clusterer=self._clusterer(
                **{"grain exports fell": [1.0, 0.0], "ports closed": [0.0, 1.0]}
            ),
        )
        links = {
            nugget.claim: sorted(link.alias for link in nugget.support)
            for nugget in bundle.nuggets
        }
        aliases = {row.text: row.evidence.alias for row in
                   build_evidence_table(self.facets, budget=24)}
        self.assertEqual(
            links["grain exports fell"],
            sorted([aliases["grain passage"], aliases["tourism passage"]]),
        )
        self.assertEqual(links["ports closed"], [aliases["port passage"]])
        # One assign call per evidence passage, no more.
        self.assertEqual(len(backend.assign_calls), len(bundle.evidence))

    def test_quarantines_a_claim_no_passage_supports(self) -> None:
        backend = ScriptedBackend(
            claims=(("grain exports fell", "vital"), ("invented fact", "okay")),
            supported={"grain exports fell": ("grain passage",)},
        )
        bundle = run_topic_nuggets(
            topic_id="2000", query="grain", representation="report_text",
            facets=self.facets, backend=backend,
            clusterer=self._clusterer(**{"grain exports fell": [1.0, 0.0]}),
        )
        self.assertEqual([n.claim for n in bundle.nuggets], ["grain exports fell"])
        self.assertEqual(
            {r.reason: r.count for r in bundle.rejected}, {"unsupported": 1}
        )

    def test_quarantines_a_mixed_claim_instead_of_splitting_it(self) -> None:
        backend = ScriptedBackend(
            claims=(
                ("grain exports fell", "vital"),
                ("Grain fell. Ports also closed for a week.", "okay"),
            ),
            supported={
                "grain exports fell": ("grain passage",),
                "Grain fell. Ports also closed for a week.": ("port passage",),
            },
        )
        bundle = run_topic_nuggets(
            topic_id="2000", query="grain", representation="report_text",
            facets=self.facets, backend=backend,
            clusterer=self._clusterer(**{"grain exports fell": [1.0, 0.0]}),
        )
        self.assertEqual([n.claim for n in bundle.nuggets], ["grain exports fell"])
        self.assertEqual({r.reason: r.count for r in bundle.rejected}, {"not_atomic": 1})

    def test_merged_paraphrases_keep_every_attesting_source(self) -> None:
        backend = ScriptedBackend(
            claims=(("grain exports fell", "okay"), ("grain shipments declined", "vital")),
            supported={
                "grain exports fell": ("grain passage",),
                "grain shipments declined": ("port passage", "tourism passage"),
            },
        )
        bundle = run_topic_nuggets(
            topic_id="2000", query="grain", representation="report_text",
            facets=self.facets, backend=backend,
            clusterer=self._clusterer(
                **{"grain exports fell": [1.0, 0.0], "grain shipments declined": [0.99, 0.14]}
            ),
        )
        self.assertEqual(len(bundle.nuggets), 1)
        merged = bundle.nuggets[0]
        self.assertEqual(len(merged.support), 3)
        # The stronger importance wins, and both wordings are recorded.
        self.assertEqual(merged.importance, "vital")
        self.assertEqual(
            merged.variants,
            ("grain exports fell", "grain shipments declined"),
        )

    def test_merging_prefers_full_support_over_partial_for_one_alias(self) -> None:
        """Two paraphrases judged differently against the same passage merge up."""

        labels = {"grain exports fell": "partial_support", "grain shipments declined": "support"}

        class SplitLabelBackend(ScriptedBackend):
            def assign(self, *, query, passage_text, nuggets):
                self.assign_calls.append(passage_text)
                return tuple(
                    labels[claim] if passage_text == "grain passage" else "not_support"
                    for claim, _ in nuggets
                )

        backend = SplitLabelBackend(
            claims=(("grain exports fell", "okay"), ("grain shipments declined", "okay")),
            supported={},
        )
        bundle = run_topic_nuggets(
            topic_id="2000", query="grain", representation="report_text",
            facets=self.facets, backend=backend,
            clusterer=self._clusterer(
                **{"grain exports fell": [1.0, 0.0], "grain shipments declined": [0.99, 0.14]}
            ),
        )
        self.assertEqual(len(bundle.nuggets), 1)
        support = bundle.nuggets[0].support
        self.assertEqual(len(support), 1)
        self.assertEqual(support[0].assignment, "support")

    def test_retains_a_claim_attested_only_partially(self) -> None:
        """partial_support keeps a claim, and the weaker label is preserved."""

        class PartialOnly(ScriptedBackend):
            def assign(self, *, query, passage_text, nuggets):
                self.assign_calls.append(passage_text)
                return tuple(
                    "partial_support" if passage_text == "grain passage" else "not_support"
                    for _ in nuggets
                )

        backend = PartialOnly(claims=(("grain exports fell", "vital"),), supported={})
        bundle = run_topic_nuggets(
            topic_id="2000", query="grain", representation="report_text",
            facets=self.facets, backend=backend,
            clusterer=self._clusterer(**{"grain exports fell": [1.0, 0.0]}),
        )
        self.assertEqual(len(bundle.nuggets), 1)
        self.assertEqual(
            [link.assignment for link in bundle.nuggets[0].support], ["partial_support"]
        )
        self.assertEqual(bundle.rejected, ())

    def test_keeps_claims_with_decimals_and_abbreviations(self) -> None:
        """A numeric or abbreviated claim must not be quarantined as non-atomic."""

        claims = (
            ("exports fell 3.5% in the quarter", "vital"),
            ("U.S. buyers cancelled two cargoes", "okay"),
        )
        backend = ScriptedBackend(
            claims=claims, supported={claim: ("grain passage",) for claim, _ in claims}
        )
        bundle = run_topic_nuggets(
            topic_id="2000", query="grain", representation="report_text",
            facets=self.facets, backend=backend,
            clusterer=self._clusterer(
                **{
                    "exports fell 3.5% in the quarter": [1.0, 0.0],
                    "U.S. buyers cancelled two cargoes": [0.0, 1.0],
                }
            ),
        )
        self.assertEqual(len(bundle.nuggets), 2)
        self.assertEqual(bundle.rejected, ())

    def test_fails_the_topic_when_nothing_is_supported(self) -> None:
        backend = ScriptedBackend(claims=(("invented", "okay"),), supported={})
        with self.assertRaisesRegex(NuggetError, "supported by known evidence"):
            run_topic_nuggets(
                topic_id="2000", query="grain", representation="report_text",
                facets=self.facets, backend=backend,
                clusterer=self._clusterer(**{"invented": [1.0, 0.0]}),
            )

    def test_fails_the_topic_when_no_claim_is_atomic(self) -> None:
        backend = ScriptedBackend(claims=((" ".join(["word"] * 80), "okay"),), supported={})
        with self.assertRaisesRegex(NuggetError, "no atomic claims"):
            run_topic_nuggets(
                topic_id="2000", query="grain", representation="report_text",
                facets=self.facets, backend=backend, clusterer=self._clusterer(),
            )

    def test_fails_the_topic_without_evidence(self) -> None:
        backend = ScriptedBackend(claims=(("a", "okay"),), supported={})
        with self.assertRaisesRegex(NuggetError, "no ranked passages"):
            run_topic_nuggets(
                topic_id="2000", query="grain", representation="report_text",
                facets=[], backend=backend, clusterer=self._clusterer(),
            )

    def test_is_byte_identical_across_runs(self) -> None:
        def build() -> NuggetBundle:
            backend = ScriptedBackend(
                claims=(("grain exports fell", "vital"), ("ports closed", "okay")),
                supported={
                    "grain exports fell": ("grain passage",),
                    "ports closed": ("port passage", "tourism passage"),
                },
            )
            return run_topic_nuggets(
                topic_id="2000", query="grain", representation="report_text",
                facets=self.facets, backend=backend,
                clusterer=self._clusterer(
                    **{"grain exports fell": [1.0, 0.0], "ports closed": [0.0, 1.0]}
                ),
            )

        first, second = build(), build()
        self.assertEqual(msgspec.msgpack.encode(first), msgspec.msgpack.encode(second))
        self.assertEqual(
            [n.nugget_id for n in first.nuggets], [n.nugget_id for n in second.nuggets]
        )

    def test_orders_vital_claims_first(self) -> None:
        backend = ScriptedBackend(
            claims=(("ports closed", "okay"), ("grain exports fell", "vital")),
            supported={
                "ports closed": ("port passage",),
                "grain exports fell": ("grain passage",),
            },
        )
        bundle = run_topic_nuggets(
            topic_id="2000", query="grain", representation="report_text",
            facets=self.facets, backend=backend,
            clusterer=self._clusterer(
                **{"grain exports fell": [1.0, 0.0], "ports closed": [0.0, 1.0]}
            ),
        )
        self.assertEqual([n.importance for n in bundle.nuggets], ["vital", "okay"])

    def test_records_facet_coverage(self) -> None:
        backend = ScriptedBackend(
            claims=(("grain exports fell", "vital"),),
            supported={"grain exports fell": ("grain passage",)},
        )
        bundle = run_topic_nuggets(
            topic_id="2000", query="grain", representation="report_text",
            facets=self.facets, backend=backend,
            clusterer=self._clusterer(**{"grain exports fell": [1.0, 0.0]}),
        )
        coverage = {f.facet_index: (f.evidence_count, f.nugget_count) for f in bundle.facets}
        self.assertEqual(coverage, {0: (2, 1), 1: (1, 0)})

    def test_truncates_to_the_nugget_maximum(self) -> None:
        claims = tuple((f"claim {index}", "okay") for index in range(4))
        backend = ScriptedBackend(
            claims=claims, supported={claim: ("grain passage",) for claim, _ in claims}
        )
        vectors = {f"claim {index}": [float(index == n) for n in range(4)] for index in range(4)}
        bundle = run_topic_nuggets(
            topic_id="2000", query="grain", representation="report_text",
            facets=self.facets, backend=backend, clusterer=self._clusterer(**vectors),
            max_nuggets=2,
        )
        self.assertEqual(len(bundle.nuggets), 2)
        self.assertEqual({r.reason: r.count for r in bundle.rejected}, {"over_budget": 2})


class CheckBundleTestCase(unittest.TestCase):
    def _bundle(self) -> NuggetBundle:
        backend = ScriptedBackend(
            claims=(("grain exports fell", "vital"),),
            supported={"grain exports fell": ("grain passage",)},
        )
        return run_topic_nuggets(
            topic_id="2000", query="grain", representation="report_text",
            facets=[facet(0, "grain passage", "port passage")], backend=backend,
            clusterer=NuggetClusterer(
                embed=fixed_embedder({normalize_claim("grain exports fell"): [1.0]})
            ),
        )

    def test_accepts_a_bundle_the_stage_produced(self) -> None:
        bundle = self._bundle()
        check_bundle(bundle)

    def test_rejects_a_duplicate_alias(self) -> None:
        bundle = self._bundle()
        broken = msgspec.structs.replace(
            bundle, evidence=(bundle.evidence[0], bundle.evidence[0])
        )
        with self.assertRaisesRegex(NuggetError, "reuses an alias"):
            check_bundle(broken)

    def test_rejects_support_outside_the_alias_table(self) -> None:
        bundle = self._bundle()
        nugget = bundle.nuggets[0]
        broken = msgspec.structs.replace(
            bundle,
            nuggets=(
                msgspec.structs.replace(
                    nugget, support=(NuggetSupport(alias="e999", assignment="support"),)
                ),
            ),
        )
        with self.assertRaisesRegex(NuggetError, "outside the alias table"):
            check_bundle(broken)

    def test_rejects_a_nugget_with_no_support(self) -> None:
        bundle = self._bundle()
        broken = msgspec.structs.replace(
            bundle,
            nuggets=(msgspec.structs.replace(bundle.nuggets[0], support=()),),
        )
        with self.assertRaisesRegex(NuggetError, "no supporting evidence"):
            check_bundle(broken)

    def test_rejects_a_claim_whose_id_no_longer_matches(self) -> None:
        bundle = self._bundle()
        broken = msgspec.structs.replace(
            bundle,
            nuggets=(msgspec.structs.replace(bundle.nuggets[0], claim="a different claim"),),
        )
        with self.assertRaisesRegex(NuggetError, "does not match its claim"):
            check_bundle(broken)

    def test_rejects_an_invalid_alias_format(self) -> None:
        bundle = self._bundle()
        broken = msgspec.structs.replace(
            bundle,
            evidence=(msgspec.structs.replace(bundle.evidence[0], alias="XX"),)
            + bundle.evidence[1:],
        )
        with self.assertRaisesRegex(NuggetError, "failed schema validation"):
            check_bundle(broken)

    def test_rejects_evidence_over_the_budget(self) -> None:
        bundle = self._bundle()
        broken = msgspec.structs.replace(bundle, evidence_budget=1)
        with self.assertRaisesRegex(NuggetError, "exceeds the configured budget"):
            check_bundle(broken)

    def test_rejects_nuggets_out_of_deterministic_order(self) -> None:
        backend = ScriptedBackend(
            claims=(("grain exports fell", "vital"), ("ports closed", "okay")),
            supported={
                "grain exports fell": ("grain passage",),
                "ports closed": ("port passage",),
            },
        )
        bundle = run_topic_nuggets(
            topic_id="2000", query="grain", representation="report_text",
            facets=[facet(0, "grain passage", "port passage")], backend=backend,
            clusterer=NuggetClusterer(
                embed=fixed_embedder(
                    {
                        normalize_claim("grain exports fell"): [1.0, 0.0],
                        normalize_claim("ports closed"): [0.0, 1.0],
                    }
                )
            ),
        )
        broken = msgspec.structs.replace(bundle, nuggets=tuple(reversed(bundle.nuggets)))
        with self.assertRaisesRegex(NuggetError, "deterministic order"):
            check_bundle(broken)


if __name__ == "__main__":
    unittest.main()
