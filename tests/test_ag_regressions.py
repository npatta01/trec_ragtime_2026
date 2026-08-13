"""Regressions found by adversarially verifying the architecture against source."""

from __future__ import annotations

import unittest
from typing import Any

from trec_ragtime.artifact_store import ArtifactStoreRemoteError
from trec_ragtime.evidence_bundle import build_evidence_bundle_from_handoff
from trec_ragtime.report_generation import SentenceCandidate, rank_citations
from trec_ragtime.report_handoff import HandoffSourceError, HandoffTopicEntry, require_handoff_bundle
from tests.test_evidence_handoff import TOPICS_DIGEST, described_handoff
from tests.test_report_cache import MemoryArtifactStore
from tests.test_report_handoff import CATALOG, PRODUCER, unstored_key


def bundle_with_ranks(*ranks: int, citation_ready: bool = True):
    """One nugget citing every alias, each carrying its own corpus rank."""

    aliases = [f"e{i:03d}" for i in range(1, len(ranks) + 1)]
    # Citation readiness is not free-standing: only a full support link is
    # citable, so asking for non-citable evidence means a partial_support link.
    assignment = "support" if citation_ready else "partial_support"
    bundle = described_handoff(
        ("A claim under test.", "okay", [(alias, assignment) for alias in aliases]),
        ranks=dict(zip(aliases, ranks, strict=True)),
    )
    return build_evidence_bundle_from_handoff(bundle, topics_sha256=TOPICS_DIGEST)


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
        bundle = described_handoff(
            ("A fully attested claim.", "okay", [("e001", "support")]),
            ("A claim only partially attested.", "okay", [("e002", "partial_support")]),
        )
        return build_evidence_bundle_from_handoff(bundle, topics_sha256=TOPICS_DIGEST)

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
        bundle = build_evidence_bundle_from_handoff(
            described_handoff(
                (
                    "A claim.",
                    "okay",
                    [("e001", "support"), ("e002", "partial_support")],
                )
            ),
            topics_sha256=TOPICS_DIGEST,
        )
        link = bundle.passage("e002").support[0]
        self.assertEqual(link.assignment, "partial_support")
        self.assertFalse(link.citation_ready)


class HandoffFailureReportingTests(unittest.TestCase):
    """Absent, unreadable and unreachable must stay three distinct reports.

    Every key here is canonical and under the real fingerprint root, so the
    reader gets far enough to distinguish them instead of rejecting the key.
    """

    def entry(self, key: str) -> HandoffTopicEntry:
        return HandoffTopicEntry("2000", "written", key, PRODUCER)

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
        key = unstored_key(b"\xbb" * 32)
        store.objects[key] = b"\x00 not a valid envelope"
        with self.assertRaises(HandoffSourceError) as caught:
            require_handoff_bundle(store, self.entry(key), topic_catalog_digest=CATALOG)
        self.assertIn("invalid", str(caught.exception).lower())
        self.assertNotIn("absent", str(caught.exception).lower())

    def test_a_genuinely_missing_artifact_is_reported_as_absent(self) -> None:
        with self.assertRaises(HandoffSourceError) as caught:
            require_handoff_bundle(
                MemoryArtifactStore(),
                self.entry(unstored_key()),
                topic_catalog_digest=CATALOG,
            )
        self.assertIn("absent", str(caught.exception).lower())

    def test_a_remote_store_failure_is_surfaced_not_swallowed(self) -> None:
        store = MemoryArtifactStore()
        store.read_error = ArtifactStoreRemoteError("bucket throttled")
        with self.assertRaises((HandoffSourceError, ArtifactStoreRemoteError)):
            require_handoff_bundle(
                store, self.entry(unstored_key()), topic_catalog_digest=CATALOG
            )


if __name__ == "__main__":
    unittest.main()


class RunContainmentTests(unittest.TestCase):
    """One topic's unexpected failure must not kill the other 102."""

    def test_an_unexpected_provider_error_fails_only_its_topic(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        from trec_ragtime.report_cache import GenerationSettings
        from trec_ragtime.report_runner import run_report_generation
        from tests.test_report_generation import completion, draft, line
        from tests.test_report_runner import RunnerTestCase

        class RateLimited(Exception):
            """Stands in for a provider error the runner has never seen."""

        case = RunnerTestCase("run")
        case.setUp()
        self.addCleanup(case._directory.cleanup)

        payload = completion(
            draft(line("The ban took effect in 2019.", ["S1"]))
        )

        class Flaky:
            def create(self, *, refresh: bool = False, **request: Any) -> Any:
                body = "".join(item["content"] for item in request["messages"])
                if "2000" in body:
                    raise RateLimited("429 slow down")
                return payload

        result = run_report_generation(
            input_dir=case.write_run("2000", "2001"),
            topics_path=case.write_topics("2000", "2001"),
            output_root=case.output_root,
            team_id="PLACEHOLDER-TEAM",
            run_id="containment-1",
            run_desc="containment test",
            settings=GenerationSettings(
                base_url="https://api.example.test/v1",
                model="test-model",
                temperature=0.0,
                top_p=None,
                max_output_tokens=4000,
                seed=7,
                max_repairs=0,
                max_citations=3,
                max_passages=3,
                evidence_chars=0,
                budget_ratio=0.9,
            ),
            completions=[Flaky()],
            store=case.store,
            clock=lambda: 1_700_000_000.0,
        )
        self.assertEqual(result.failed, 1)
        self.assertEqual(result.generated, 1)
        failures = json.loads((result.run_dir / "failures.json").read_text())
        self.assertEqual(failures["failed_topics"], ["2000"])
        self.assertIn("RateLimited", failures["details"]["2000"])


class UsageAccountingTests(unittest.TestCase):
    def test_usage_aggregates_across_repair_attempts(self) -> None:
        from trec_ragtime.report_generation import generate_topic_report
        from tests.test_report_generation import (
            ScriptedCompletions,
            bundle as gen_bundle,
            completion,
            draft,
            identity,
            line,
            settings as gen_settings,
        )

        item = gen_bundle()
        request = identity(item, settings=gen_settings(max_repairs=1))
        broken = completion(draft(line("An invented claim.", ["S9"])))
        fixed = completion(
            draft(
                line("A supported claim.", ["S1"]),
                line("Another supported claim.", ["S2"], ["F2"]),
            )
        )
        report = generate_topic_report(
            bundle=item,
            request=request,
            completions=ScriptedCompletions([broken, fixed]),
            cache=None,
            clock=lambda: 1_700_000_000.0,
        )
        self.assertEqual(report.diagnostics.repair_attempts, 1)
        # Each scripted completion reports 5 total tokens; both calls were paid
        # for, so recorded spend must reflect two calls, not just the kept one.
        self.assertEqual(report.diagnostics.usage["total_tokens"], 10)
        self.assertEqual(report.diagnostics.usage["prompt_tokens"], 6)
