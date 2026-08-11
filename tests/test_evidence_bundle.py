from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from trec_ragtime.evidence_bundle import (
    EVIDENCE_BUNDLE_BUILDER_VERSION,
    EVIDENCE_BUNDLE_SCHEMA_VERSION,
    EvidenceBundle,
    EvidenceBundleError,
    EvidenceBundleSource,
    EvidenceFacet,
    EvidencePassage,
    UnknownEvidenceHandleError,
    build_evidence_bundles,
    load_evidence_run,
    topic_digest,
)
from trec_ragtime.topic_candidates import Topic, load_topics


def topic(topic_id: str = "2000", limit: int = 5000) -> Topic:
    return Topic(
        topic_id=topic_id,
        collection_id="ragtime/2/all",
        problem_statement="A statement about vaping law.",
        background="A legislative aide needs this.",
        limit=limit,
        title=f"Topic {topic_id}",
    )


def facet(facet_id: str = "F1", text: str = "vaping law dates") -> EvidenceFacet:
    return EvidenceFacet(
        facet_id=facet_id, text=text, source="query_baseline_facet_v1", importance="okay"
    )


def passage(handle: str = "S1", document_id: str = "doc-a", rank: int = 1) -> EvidencePassage:
    return EvidencePassage.create(
        handle=handle,
        document_id=document_id,
        report_text=f"Evidence text for {document_id}.",
        language="eng",
        url="https://example.test/a",
        date="2022-09-21T06:34:00.000Z",
        rank=rank,
        retrieval_score=0.5,
        facet_ids=("F1",),
    )


def source(**overrides: object) -> EvidenceBundleSource:
    values: dict[str, object] = {
        "run_schema": "query_baseline_v1",
        "run_id": "baseline-c-test",
        "run_dir_name": "baseline-c-test",
        "manifest_sha256": "a" * 64,
        "records_sha256": "b" * 64,
        "topics_sha256": "c" * 64,
        "corpus_dataset": "trec-ragtime/ragtime2",
        "corpus_revision": "c99bf3e",
        "nugget_bundle_sha256": None,
    }
    values.update(overrides)
    return EvidenceBundleSource(**values)  # type: ignore[arg-type]


def bundle(**overrides: object) -> EvidenceBundle:
    values: dict[str, object] = {
        "topic_id": "2000",
        "collection_id": "ragtime/2/all",
        "title": "Topic 2000",
        "problem_statement": "A statement about vaping law.",
        "background": "A legislative aide needs this.",
        "character_limit": 5000,
        "max_passages": 2,
        "evidence_chars": 0,
        "facets": (facet(),),
        "passages": (passage("S1", "doc-a", 1), passage("S2", "doc-b", 2)),
        "source": source(),
    }
    values.update(overrides)
    return EvidenceBundle.create(**values)  # type: ignore[arg-type]


class SealTests(unittest.TestCase):
    def test_created_bundle_carries_a_verifying_seal(self) -> None:
        item = bundle()
        self.assertEqual(len(item.seal), 64)
        self.assertEqual(item.schema_version, EVIDENCE_BUNDLE_SCHEMA_VERSION)
        self.assertEqual(item.builder_version, EVIDENCE_BUNDLE_BUILDER_VERSION)

    def test_seal_is_deterministic_for_identical_content(self) -> None:
        self.assertEqual(bundle().seal, bundle().seal)

    def test_seal_changes_with_every_sealed_field(self) -> None:
        baseline = bundle().seal
        cases = {
            "topic_id": "2001",
            "collection_id": "other",
            "title": "Another title",
            "problem_statement": "Different statement.",
            "background": "Different background.",
            "character_limit": 4000,
            "max_passages": 3,
            "evidence_chars": 1200,
        }
        for field, value in cases.items():
            with self.subTest(field=field):
                self.assertNotEqual(bundle(**{field: value}).seal, baseline)

    def test_seal_changes_when_evidence_text_changes(self) -> None:
        altered = EvidencePassage.create(
            handle="S1",
            document_id="doc-a",
            report_text="Entirely different evidence text.",
            language="eng",
            url="https://example.test/a",
            date="2022-09-21T06:34:00.000Z",
            rank=1,
            retrieval_score=0.5,
            facet_ids=("F1",),
        )
        self.assertNotEqual(
            bundle(passages=(altered, passage("S2", "doc-b", 2))).seal, bundle().seal
        )

    def test_bundle_rejects_a_passage_whose_text_was_swapped(self) -> None:
        # A stale digest means the evidence text was edited after hashing.
        tampered = replace(passage("S1", "doc-a", 1), report_text="Tampered text.")
        with self.assertRaises(EvidenceBundleError):
            bundle(passages=(tampered, passage("S2", "doc-b", 2)))

    def test_seal_changes_when_the_facet_set_changes(self) -> None:
        self.assertNotEqual(
            bundle(facets=(facet(), facet("F2", "youth outcomes"))).seal, bundle().seal
        )

    def test_seal_changes_when_the_source_changes(self) -> None:
        self.assertNotEqual(
            bundle(source=source(records_sha256="d" * 64)).seal, bundle().seal
        )

    def test_tampered_bundle_with_a_stale_seal_is_rejected(self) -> None:
        original = bundle()
        with self.assertRaises(EvidenceBundleError):
            EvidenceBundle(
                schema_version=original.schema_version,
                builder_version=original.builder_version,
                topic_id="9999",
                collection_id=original.collection_id,
                title=original.title,
                problem_statement=original.problem_statement,
                background=original.background,
                character_limit=original.character_limit,
                max_passages=original.max_passages,
                evidence_chars=original.evidence_chars,
                facets=original.facets,
                passages=original.passages,
                source=original.source,
                seal=original.seal,
            )

    def test_passage_text_hash_matches_its_text(self) -> None:
        item = passage()
        self.assertEqual(len(item.text_sha256), 64)
        with self.assertRaises(EvidenceBundleError):
            replace(item, report_text="Different text.").verify()


class BundleAccessTests(unittest.TestCase):
    def test_handle_allowlist_and_document_ids(self) -> None:
        item = bundle()
        self.assertEqual(item.handle_allowlist, frozenset({"S1", "S2"}))
        self.assertEqual(item.document_ids, ("doc-a", "doc-b"))
        self.assertEqual(item.facet_ids, ("F1",))

    def test_resolve_handle_maps_to_the_canonical_document_id(self) -> None:
        self.assertEqual(bundle().resolve_handle("S2"), "doc-b")

    def test_unknown_handle_fails_closed(self) -> None:
        item = bundle()
        for handle in ("S3", "s1", "doc-a", "", "S", None):
            with self.subTest(handle=handle):
                with self.assertRaises(UnknownEvidenceHandleError):
                    item.resolve_handle(handle)  # type: ignore[arg-type]

    def test_duplicate_handles_are_rejected(self) -> None:
        with self.assertRaises(EvidenceBundleError):
            bundle(passages=(passage("S1", "doc-a", 1), passage("S1", "doc-b", 2)))

    def test_duplicate_document_ids_are_rejected(self) -> None:
        with self.assertRaises(EvidenceBundleError):
            bundle(passages=(passage("S1", "doc-a", 1), passage("S2", "doc-a", 2)))

    def test_passage_facet_ids_must_exist_in_the_bundle(self) -> None:
        stray = replace(passage("S1", "doc-a", 1), facet_ids=("F9",))
        with self.assertRaises(EvidenceBundleError):
            bundle(passages=(stray,))

    def test_empty_passage_set_is_rejected(self) -> None:
        with self.assertRaises(EvidenceBundleError):
            bundle(passages=())

    def test_prompt_excerpt_returns_full_text_when_unbounded(self) -> None:
        item = bundle()
        self.assertEqual(item.prompt_excerpt("S1"), item.passage("S1").report_text)

    def test_prompt_excerpt_cuts_at_a_whitespace_boundary(self) -> None:
        long_passage = EvidencePassage.create(
            handle="S1",
            document_id="doc-a",
            report_text="alpha beta gamma delta epsilon zeta eta theta",
            language="eng",
            url=None,
            date=None,
            rank=1,
            retrieval_score=0.5,
            facet_ids=("F1",),
        )
        item = bundle(passages=(long_passage,), evidence_chars=20)
        excerpt = item.prompt_excerpt("S1")
        self.assertTrue(len(excerpt) <= 21, excerpt)
        self.assertTrue(excerpt.startswith("alpha beta gamma"))
        self.assertNotIn("delt ", excerpt)

    def test_prompt_excerpt_is_deterministic(self) -> None:
        item = bundle(evidence_chars=12)
        self.assertEqual(item.prompt_excerpt("S1"), item.prompt_excerpt("S1"))


class TopicDigestTests(unittest.TestCase):
    def test_digest_is_stable_and_32_bytes(self) -> None:
        self.assertEqual(len(topic_digest(topic())), 32)
        self.assertEqual(topic_digest(topic()), topic_digest(topic()))

    def test_digest_changes_with_every_topic_field(self) -> None:
        baseline = topic_digest(topic())
        for field, value in (
            ("topic_id", "2001"),
            ("collection_id", "other"),
            ("problem_statement", "Other statement."),
            ("background", "Other background."),
            ("limit", 4000),
            ("title", "Other title"),
        ):
            with self.subTest(field=field):
                self.assertNotEqual(topic_digest(replace(topic(), **{field: value})), baseline)


class RunLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)

    def write_baseline_run(self, records: list[dict[str, object]], **manifest: object) -> Path:
        run_dir = self.root / "baseline-c-test"
        run_dir.mkdir(exist_ok=True)
        payload: dict[str, object] = {
            "schema_version": "query_baseline_v1",
            "run_id": "baseline-c-test",
            "status": "complete",
            "corpus": {"dataset": "trec-ragtime/ragtime2", "revision": "c99bf3e"},
        }
        payload.update(manifest)
        (run_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
        (run_dir / "topics.jsonl").write_text(
            "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
        )
        return run_dir

    def write_control_a_run(self, records: list[dict[str, object]]) -> Path:
        run_dir = self.root / "control-a-test"
        run_dir.mkdir(exist_ok=True)
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "control_a_v1",
                    "run_id": "control-a-test",
                    "counts": {"topics": 1, "failed_topics": 0},
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "candidates.jsonl").write_text(
            "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
        )
        return run_dir

    def baseline_candidate(self, **overrides: object) -> dict[str, object]:
        record: dict[str, object] = {
            "document_id": "doc-a",
            "fused_score": 0.087,
            "best_rank": 2,
            "source_query_indices": [1, 3],
            "source_query_count": 2,
            "rank": 1,
            "score": None,
            "hydrated": True,
            "text": "Original language text.",
            "report_text": "English report text.",
            "english_text": None,
            "language": "eng",
            "url": "https://example.test/a",
            "date": "2022-09-21T06:34:00.000Z",
        }
        record.update(overrides)
        return record

    def baseline_topic(self, **overrides: object) -> dict[str, object]:
        record: dict[str, object] = {
            "topic_id": "2000",
            "variant": "baseline-c-facet-rrf",
            "status": "success",
            "plan_status": "model_generated",
            "queries": ["facet one", "facet two", "facet three"],
            "candidate_count": 2,
            "candidates": [
                self.baseline_candidate(),
                self.baseline_candidate(
                    document_id="doc-b", rank=2, source_query_indices=[2]
                ),
            ],
        }
        record.update(overrides)
        return record

    def control_candidate(self, **overrides: object) -> dict[str, object]:
        record: dict[str, object] = {
            "topic_id": "2000",
            "document_id": "doc-a",
            "rank": 1,
            "score": 0.403,
            "hydrated": True,
            "report_ready": True,
            "missing_translation": False,
            "text": "Original language text.",
            "report_text": "English report text.",
            "english_text": None,
            "language": "eng",
            "url": "https://example.test/a",
            "date": "2023-06-12T09:58:44.000Z",
            "limit": 5000,
            "collection_id": "ragtime/2/all",
            "query": "A statement about vaping law.",
            "query_hash": "0" * 64,
        }
        record.update(overrides)
        return record

    def test_query_baseline_run_maps_facets_and_scores(self) -> None:
        run = load_evidence_run(self.write_baseline_run([self.baseline_topic()]))
        self.assertEqual(run.run_schema, "query_baseline_v1")
        self.assertEqual(run.corpus_dataset, "trec-ragtime/ragtime2")
        self.assertEqual(run.corpus_revision, "c99bf3e")
        topic_run = run.topics["2000"]
        self.assertEqual([item.text for item in topic_run.facets], ["facet one", "facet two", "facet three"])
        self.assertEqual([item.facet_id for item in topic_run.facets], ["F1", "F2", "F3"])
        first, second = topic_run.candidates
        self.assertEqual(first.document_id, "doc-a")
        self.assertAlmostEqual(first.retrieval_score, 0.087)
        # source_query_indices are 1-based into `queries`.
        self.assertEqual(first.facet_ids, ("F1", "F3"))
        self.assertEqual(second.facet_ids, ("F2",))

    def test_control_a_run_uses_one_synthetic_facet(self) -> None:
        run = load_evidence_run(
            self.write_control_a_run(
                [self.control_candidate(), self.control_candidate(document_id="doc-b", rank=2)]
            )
        )
        self.assertEqual(run.run_schema, "control_a_v1")
        self.assertIsNone(run.corpus_dataset)
        topic_run = run.topics["2000"]
        self.assertEqual(len(topic_run.facets), 1)
        self.assertEqual(topic_run.facets[0].source, "control_a_problem_v1")
        self.assertEqual([item.facet_ids for item in topic_run.candidates], [("F1",), ("F1",)])
        self.assertAlmostEqual(topic_run.candidates[0].retrieval_score, 0.403)

    def test_control_a_groups_candidates_by_topic(self) -> None:
        run = load_evidence_run(
            self.write_control_a_run(
                [
                    self.control_candidate(topic_id="2000", document_id="doc-a"),
                    self.control_candidate(topic_id="2001", document_id="doc-b"),
                    self.control_candidate(topic_id="2000", document_id="doc-c", rank=2),
                ]
            )
        )
        self.assertEqual(sorted(run.topics), ["2000", "2001"])
        self.assertEqual(
            [item.document_id for item in run.topics["2000"].candidates], ["doc-a", "doc-c"]
        )

    def test_unusable_candidates_are_rejected_and_counted(self) -> None:
        cases = (
            {"hydrated": False},
            {"report_text": ""},
            {"report_text": None},
            {"fused_score": float("inf")},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                record = self.baseline_topic(
                    candidates=[
                        self.baseline_candidate(**overrides),
                        self.baseline_candidate(document_id="doc-b", rank=2),
                    ]
                )
                run = load_evidence_run(self.write_baseline_run([record]))
                self.assertEqual(
                    [item.document_id for item in run.topics["2000"].candidates], ["doc-b"]
                )
                self.assertEqual(run.rejected_candidates, 1)

    def test_control_a_missing_translation_is_rejected(self) -> None:
        run = load_evidence_run(
            self.write_control_a_run(
                [
                    self.control_candidate(missing_translation=True),
                    self.control_candidate(document_id="doc-b", rank=2),
                ]
            )
        )
        self.assertEqual(
            [item.document_id for item in run.topics["2000"].candidates], ["doc-b"]
        )

    def test_duplicate_document_within_a_topic_is_dropped(self) -> None:
        record = self.baseline_topic(
            candidates=[self.baseline_candidate(), self.baseline_candidate(rank=2)]
        )
        run = load_evidence_run(self.write_baseline_run([record]))
        self.assertEqual(len(run.topics["2000"].candidates), 1)

    def test_failed_topic_records_are_skipped(self) -> None:
        run = load_evidence_run(
            self.write_baseline_run(
                [self.baseline_topic(), self.baseline_topic(topic_id="2001", status="failed")]
            )
        )
        self.assertEqual(sorted(run.topics), ["2000"])

    def test_unsupported_schema_version_is_rejected(self) -> None:
        run_dir = self.write_baseline_run([self.baseline_topic()], schema_version="mystery_v9")
        with self.assertRaises(EvidenceBundleError):
            load_evidence_run(run_dir)

    def test_missing_manifest_is_rejected(self) -> None:
        run_dir = self.root / "empty"
        run_dir.mkdir()
        with self.assertRaises(EvidenceBundleError):
            load_evidence_run(run_dir)

    def test_record_and_manifest_digests_are_recorded(self) -> None:
        run = load_evidence_run(self.write_baseline_run([self.baseline_topic()]))
        self.assertEqual(len(run.manifest_sha256), 64)
        self.assertEqual(len(run.records_sha256), 64)


class BundleBuildingTests(RunLoadingTests):
    def write_topics_file(self, *topic_ids: str) -> Path:
        path = self.root / "topics.jsonl"
        path.write_text(
            "\n".join(
                json.dumps(
                    {
                        "topic_id": topic_id,
                        "collection_id": "ragtime/2/all",
                        "title": f"Topic {topic_id}",
                        "problem_statement": "A statement about vaping law.",
                        "background": "A legislative aide needs this.",
                        "limit": 5000,
                    }
                )
                for topic_id in topic_ids
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def build(self, *, max_passages: int = 10, topic_ids: tuple[str, ...] = ("2000",)):
        run = load_evidence_run(self.write_baseline_run([self.baseline_topic()]))
        topics = load_topics(self.write_topics_file(*topic_ids))
        return build_evidence_bundles(
            run,
            topics=topics,
            topics_sha256="c" * 64,
            max_passages=max_passages,
            evidence_chars=0,
        )

    def test_handles_are_assigned_in_rank_order(self) -> None:
        (item,) = self.build()
        self.assertEqual([p.handle for p in item.passages], ["S1", "S2"])
        self.assertEqual(item.document_ids, ("doc-a", "doc-b"))
        self.assertEqual([p.rank for p in item.passages], [1, 2])

    def test_bundle_carries_topic_and_source_provenance(self) -> None:
        (item,) = self.build()
        self.assertEqual(item.topic_id, "2000")
        self.assertEqual(item.character_limit, 5000)
        self.assertEqual(item.problem_statement, "A statement about vaping law.")
        self.assertEqual(item.source.run_schema, "query_baseline_v1")
        self.assertEqual(item.source.run_dir_name, "baseline-c-test")
        self.assertIsNone(item.source.nugget_bundle_sha256)

    def test_max_passages_trims_from_the_tail(self) -> None:
        (item,) = self.build(max_passages=1)
        self.assertEqual(item.document_ids, ("doc-a",))
        self.assertEqual(item.max_passages, 1)

    def test_only_facets_referenced_by_kept_passages_are_retained(self) -> None:
        (item,) = self.build(max_passages=1)
        self.assertEqual(item.facet_ids, ("F1", "F3"))

    def test_missing_topic_evidence_is_an_error(self) -> None:
        with self.assertRaises(EvidenceBundleError):
            self.build(topic_ids=("2000", "2001"))

    def test_bundles_follow_topic_file_order(self) -> None:
        run = load_evidence_run(
            self.write_baseline_run(
                [self.baseline_topic(topic_id="2001"), self.baseline_topic(topic_id="2000")]
            )
        )
        topics = load_topics(self.write_topics_file("2000", "2001"))
        bundles = build_evidence_bundles(
            run, topics=topics, topics_sha256="c" * 64, max_passages=10, evidence_chars=0
        )
        self.assertEqual([item.topic_id for item in bundles], ["2000", "2001"])

    def test_identical_inputs_produce_identical_seals(self) -> None:
        (first,) = self.build()
        (second,) = self.build()
        self.assertEqual(first.seal, second.seal)

    def test_max_passages_changes_the_seal(self) -> None:
        (wide,) = self.build(max_passages=10)
        (narrow,) = self.build(max_passages=1)
        self.assertNotEqual(wide.seal, narrow.seal)


if __name__ == "__main__":
    unittest.main()
