from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import msgspec

from trec_ragtime.report_cache import GenerationSettings
from trec_ragtime.report_handoff import (
    HandoffSourceError,
    load_handoff_manifest,
    read_handoff_bundle,
    require_handoff_bundle,
)
from trec_ragtime.report_runner import run_report_generation
from trec_ragtime.report_submission import validate_submission_file
from trec_ragtime.retrieval_handoff import handoff_bundle_digest
from trec_ragtime.retrieval_handoff_cache import HandoffEnvelope, handoff_object_key
from tests.test_evidence_handoff import evidence, handoff, nugget
from trec_ragtime.retrieval_handoff import HandoffSupport
from tests.test_report_cache import MemoryArtifactStore
from tests.test_report_generation import completion, draft, line

TEAM_ID = "PLACEHOLDER-TEAM"
RUN_ID = "handoff-run-1"
FP = b"\x11" * 32
RQ = b"\x22" * 32


def store_bundle(store: MemoryArtifactStore, bundle: Any, *, request_digest: bytes = RQ) -> str:
    envelope = HandoffEnvelope(
        fingerprint_digest=FP,
        request_digest=request_digest,
        bundle_digest=handoff_bundle_digest(bundle),
        bundle=bundle,
    )
    key = handoff_object_key(FP, request_digest)
    store.objects[key] = msgspec.msgpack.encode(envelope)
    return key


class ReaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = MemoryArtifactStore()

    def test_round_trip_by_object_key(self) -> None:
        bundle = handoff()
        key = store_bundle(self.store, bundle)
        loaded = read_handoff_bundle(self.store, key)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.topic.topic_id, "2000")

    def test_absent_artifact_is_none(self) -> None:
        self.assertIsNone(
            read_handoff_bundle(self.store, handoff_object_key(FP, RQ))
        )

    def test_key_that_disagrees_with_the_envelope_is_rejected(self) -> None:
        bundle = handoff()
        envelope = HandoffEnvelope(
            fingerprint_digest=FP,
            request_digest=RQ,
            bundle_digest=handoff_bundle_digest(bundle),
            bundle=bundle,
        )
        wrong = handoff_object_key(FP, b"\x33" * 32)
        self.store.objects[wrong] = msgspec.msgpack.encode(envelope)
        self.assertIsNone(read_handoff_bundle(self.store, wrong))

    def test_tampered_bundle_digest_is_rejected(self) -> None:
        bundle = handoff()
        envelope = HandoffEnvelope(
            fingerprint_digest=FP, request_digest=RQ, bundle_digest=b"\x00" * 32, bundle=bundle
        )
        key = handoff_object_key(FP, RQ)
        self.store.objects[key] = msgspec.msgpack.encode(envelope)
        self.assertIsNone(read_handoff_bundle(self.store, key))

    def test_garbage_payload_is_rejected(self) -> None:
        key = handoff_object_key(FP, RQ)
        self.store.objects[key] = b"\x00\x01\x02"
        self.assertIsNone(read_handoff_bundle(self.store, key))


class ManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)

    def write(self, payload: dict[str, Any]) -> Path:
        path = self.root / "handoff-manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def base(self, topics: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "fingerprint": "a" * 64,
            "topic_catalog_digest": "b" * 64,
            "complete": True,
            "topics": topics,
        }

    def test_manifest_round_trip(self) -> None:
        path = self.write(
            self.base([{"topic_id": "2000", "status": "written", "object_key": "k1"}])
        )
        manifest = load_handoff_manifest(path)
        self.assertEqual(manifest.available_topic_ids, ("2000",))
        self.assertEqual(manifest.entry("2000").object_key, "k1")
        self.assertIsNone(manifest.entry("9999"))

    def test_failed_topic_is_not_available(self) -> None:
        path = self.write(
            self.base([{"topic_id": "2000", "status": "failed", "object_key": None}])
        )
        manifest = load_handoff_manifest(path)
        self.assertEqual(manifest.available_topic_ids, ())
        with self.assertRaises(HandoffSourceError):
            require_handoff_bundle(MemoryArtifactStore(), manifest.entry("2000"))

    def test_unsupported_schema_is_rejected(self) -> None:
        payload = self.base([{"topic_id": "2000", "status": "written", "object_key": "k"}])
        payload["schema_version"] = 9
        with self.assertRaises(HandoffSourceError):
            load_handoff_manifest(self.write(payload))

    def test_duplicate_topic_is_rejected(self) -> None:
        rows = [
            {"topic_id": "2000", "status": "written", "object_key": "k1"},
            {"topic_id": "2000", "status": "written", "object_key": "k2"},
        ]
        with self.assertRaises(HandoffSourceError):
            load_handoff_manifest(self.write(self.base(rows)))

    def test_missing_artifact_fails_closed(self) -> None:
        path = self.write(
            self.base([{"topic_id": "2000", "status": "written", "object_key": "absent"}])
        )
        manifest = load_handoff_manifest(path)
        with self.assertRaises(HandoffSourceError):
            require_handoff_bundle(MemoryArtifactStore(), manifest.entry("2000"))


class TripwireStore(MemoryArtifactStore):
    """Fails loudly if anything but the handoff and report prefixes is read."""

    def read_validated(self, object_key: str, decoder: Any) -> Any:
        if not object_key.startswith(("retrieval-handoff/", "reports/", "api-responses/")):
            raise AssertionError(f"generation opened a forbidden artifact: {object_key}")
        return super().read_validated(object_key, decoder)


class GenerationFromHandoffTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)
        self.store = TripwireStore()

    def write_topics(self, limit: int = 5000) -> Path:
        path = self.root / "topics.jsonl"
        path.write_text(
            json.dumps(
                {
                    "topic_id": "2000",
                    "collection_id": "ragtime/2/all",
                    "title": "Anti-vaping legislation",
                    "problem_statement": "A statement about vaping law.",
                    "background": "A legislative aide needs this.",
                    "limit": limit,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def write_manifest(self, key: str) -> Path:
        path = self.root / "handoff-manifest.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "fingerprint": "a" * 64,
                    "topic_catalog_digest": "b" * 64,
                    "complete": True,
                    "topics": [
                        {"topic_id": "2000", "status": "written", "object_key": key}
                    ],
                }
            ),
            encoding="utf-8",
        )
        return path

    def generate(self, responses: Any, *, bundle: Any = None, **overrides: Any):
        key = store_bundle(self.store, bundle or handoff())

        class Provider:
            def __init__(self) -> None:
                self.requests: list[dict[str, Any]] = []

            def create(self, *, refresh: bool = False, **request: Any) -> Any:
                self.requests.append(request)
                return responses

        provider = Provider()
        settings = GenerationSettings(
            base_url="https://api.example.test/v1",
            model="test-model",
            temperature=0.0,
            top_p=None,
            max_output_tokens=4000,
            seed=7,
            max_repairs=0,
            max_citations=3,
            max_passages=30,
            evidence_chars=0,
            budget_ratio=0.9,
        )
        result = run_report_generation(
            handoff_manifest=self.write_manifest(key),
            topics_path=self.write_topics(),
            output_root=self.root / "runs",
            team_id=TEAM_ID,
            run_id=RUN_ID,
            run_desc="handoff integration test",
            settings=settings,
            completions=[provider],
            store=self.store,
            clock=lambda: 1_700_000_000.0,
            **overrides,
        )
        return result, provider

    def test_generation_consumes_only_the_handoff(self) -> None:
        payload = completion(
            draft(
                line("The ban took effect in 2019.", ["e001"]),
                line("Youth vaping prevalence fell.", ["e002"], ["F2"]),
            )
        )
        result, _ = self.generate(payload)
        self.assertEqual(result.status, "complete")
        report = validate_submission_file(
            result.run_dir / "submission.jsonl",
            topics={"2000": 5000},
            team_id=TEAM_ID,
            run_id=RUN_ID,
        )
        self.assertTrue(report.ok, report.violations)

    def test_citations_resolve_to_canonical_document_ids(self) -> None:
        payload = completion(draft(line("The ban took effect in 2019.", ["e001"])))
        result, _ = self.generate(payload)
        record = json.loads((result.run_dir / "submission.jsonl").read_text().strip())
        self.assertEqual(record["references"], ["doc-1"])
        self.assertEqual(list(record["responses"][0]["citations"]), ["doc-1"])

    def test_prompt_offers_upstream_aliases_and_hides_document_ids(self) -> None:
        payload = completion(draft(line("The ban took effect in 2019.", ["e001"])))
        _, provider = self.generate(payload)
        body = "".join(item["content"] for item in provider.requests[0]["messages"])
        self.assertIn("[e001]", body)
        self.assertNotIn("doc-1", body)

    def test_partial_support_evidence_is_not_offered_for_citation(self) -> None:
        rows = (evidence("e001", 1), evidence("e002", 2, facet_index=1))
        nuggets = (
            nugget(
                "a",
                "The ban took effect in 2019.",
                (
                    HandoffSupport(alias="e001", assignment="support", variant_index=0, citation_ready=True),
                    HandoffSupport(alias="e002", assignment="partial_support", variant_index=0, citation_ready=False),
                ),
            ),
        )
        partial = handoff(evidence_rows=rows, nuggets=nuggets)
        payload = completion(draft(line("The ban took effect in 2019.", ["e001"])))
        _, provider = self.generate(payload, bundle=partial)
        schema = provider.requests[0]["response_format"]["json_schema"]["schema"]
        enum = schema["properties"]["sentences"]["items"]["properties"]["evidence"]["items"]["enum"]
        self.assertEqual(enum, ["e001"])

    def test_citing_partial_support_evidence_is_refused(self) -> None:
        rows = (evidence("e001", 1), evidence("e002", 2, facet_index=1))
        nuggets = (
            nugget(
                "a",
                "The ban took effect in 2019.",
                (
                    HandoffSupport(alias="e001", assignment="support", variant_index=0, citation_ready=True),
                    HandoffSupport(alias="e002", assignment="partial_support", variant_index=0, citation_ready=False),
                ),
            ),
        )
        partial = handoff(evidence_rows=rows, nuggets=nuggets)
        payload = completion(draft(line("A claim from context only.", ["e002"])))
        result, _ = self.generate(payload, bundle=partial)
        self.assertEqual(result.status, "incomplete")
        self.assertEqual(result.failed, 1)
        self.assertFalse((result.run_dir / "submission.jsonl").exists())

    def test_manifest_records_the_handoff_provenance(self) -> None:
        payload = completion(draft(line("The ban took effect in 2019.", ["e001"])))
        result, _ = self.generate(payload)
        manifest = json.loads((result.run_dir / "manifest.json").read_text())
        self.assertEqual(manifest["input"]["run_schema"], "retrieval_handoff_v1")
        self.assertEqual(manifest["input"]["handoff_fingerprint"], "a" * 64)

    def test_warm_rerun_makes_no_provider_call(self) -> None:
        payload = completion(draft(line("The ban took effect in 2019.", ["e001"])))
        self.generate(payload)

        class Tripwire:
            def create(self, *, refresh: bool = False, **request: Any) -> Any:
                raise AssertionError("provider called")

        key = store_bundle(self.store, handoff())
        result = run_report_generation(
            handoff_manifest=self.write_manifest(key),
            topics_path=self.write_topics(),
            output_root=self.root / "runs",
            team_id=TEAM_ID,
            run_id=RUN_ID,
            run_desc="handoff integration test",
            settings=GenerationSettings(
                base_url="https://api.example.test/v1",
                model="test-model",
                temperature=0.0,
                top_p=None,
                max_output_tokens=4000,
                seed=7,
                max_repairs=0,
                max_citations=3,
                max_passages=30,
                evidence_chars=0,
                budget_ratio=0.9,
            ),
            completions=[Tripwire()],
            store=self.store,
            clock=lambda: 1_700_000_000.0,
        )
        self.assertEqual(result.cached, 1)

    def test_topic_absent_from_the_handoff_is_reported(self) -> None:
        payload = completion(draft(line("The ban took effect in 2019.", ["e001"])))
        key = store_bundle(self.store, handoff())
        manifest_path = self.root / "handoff-manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "fingerprint": "a" * 64,
                    "topic_catalog_digest": "b" * 64,
                    "complete": False,
                    "topics": [
                        {"topic_id": "2000", "status": "written", "object_key": key},
                        {"topic_id": "2001", "status": "failed", "object_key": None},
                    ],
                }
            ),
            encoding="utf-8",
        )
        topics = self.root / "topics.jsonl"
        rows = []
        for topic_id in ("2000", "2001"):
            rows.append(
                json.dumps(
                    {
                        "topic_id": topic_id,
                        "collection_id": "ragtime/2/all",
                        "title": "Anti-vaping legislation",
                        "problem_statement": "A statement about vaping law.",
                        "background": "A legislative aide needs this.",
                        "limit": 5000,
                    }
                )
            )
        topics.write_text("\n".join(rows) + "\n", encoding="utf-8")

        class Provider:
            def create(self, *, refresh: bool = False, **request: Any) -> Any:
                return payload

        result = run_report_generation(
            handoff_manifest=manifest_path,
            topics_path=topics,
            output_root=self.root / "runs2",
            team_id=TEAM_ID,
            run_id=RUN_ID,
            run_desc="handoff integration test",
            settings=GenerationSettings(
                base_url="https://api.example.test/v1",
                model="test-model",
                temperature=0.0,
                top_p=None,
                max_output_tokens=4000,
                seed=7,
                max_repairs=0,
                max_citations=3,
                max_passages=30,
                evidence_chars=0,
                budget_ratio=0.9,
            ),
            completions=[Provider()],
            store=self.store,
            clock=lambda: 1_700_000_000.0,
        )
        self.assertIn("2001", result.missing)
        self.assertEqual(result.status, "incomplete")


if __name__ == "__main__":
    unittest.main()


class MergeManifestTests(unittest.TestCase):
    """A retry run emits its own manifest, so consumption must span several."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)

    def write(self, name: str, topics: list[dict[str, Any]], **overrides: Any) -> Path:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "fingerprint": "a" * 64,
            "topic_catalog_digest": "b" * 64,
            "complete": True,
            "topics": topics,
        }
        payload.update(overrides)
        path = self.root / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_later_manifest_supplies_a_topic_the_first_failed(self) -> None:
        from trec_ragtime.report_handoff import merge_handoff_manifests

        first = self.write(
            "plan.json",
            [
                {"topic_id": "2000", "status": "written", "object_key": "k0"},
                {"topic_id": "2040", "status": "failed", "object_key": None},
            ],
            complete=False,
        )
        second = self.write(
            "retry.json", [{"topic_id": "2040", "status": "written", "object_key": "k40"}]
        )
        merged = merge_handoff_manifests([first, second])
        self.assertEqual(sorted(merged.available_topic_ids), ["2000", "2040"])
        self.assertEqual(merged.entry("2040").object_key, "k40")
        self.assertTrue(merged.complete)

    def test_a_written_entry_is_never_replaced_by_a_failure(self) -> None:
        from trec_ragtime.report_handoff import merge_handoff_manifests

        first = self.write(
            "plan.json", [{"topic_id": "2000", "status": "written", "object_key": "k0"}]
        )
        second = self.write(
            "retry.json", [{"topic_id": "2000", "status": "failed", "object_key": None}]
        )
        merged = merge_handoff_manifests([first, second])
        self.assertEqual(merged.entry("2000").object_key, "k0")

    def test_later_written_entry_overrides_an_earlier_one(self) -> None:
        from trec_ragtime.report_handoff import merge_handoff_manifests

        first = self.write(
            "plan.json", [{"topic_id": "2000", "status": "written", "object_key": "old"}]
        )
        second = self.write(
            "retry.json", [{"topic_id": "2000", "status": "written", "object_key": "new"}]
        )
        self.assertEqual(
            merge_handoff_manifests([first, second]).entry("2000").object_key, "new"
        )

    def test_a_fingerprint_mismatch_fails_closed(self) -> None:
        from trec_ragtime.report_handoff import merge_handoff_manifests

        first = self.write(
            "plan.json", [{"topic_id": "2000", "status": "written", "object_key": "k"}]
        )
        second = self.write(
            "other.json",
            [{"topic_id": "2001", "status": "written", "object_key": "k2"}],
            fingerprint="c" * 64,
        )
        with self.assertRaises(HandoffSourceError):
            merge_handoff_manifests([first, second])

    def test_a_topic_catalog_mismatch_fails_closed(self) -> None:
        from trec_ragtime.report_handoff import merge_handoff_manifests

        first = self.write(
            "plan.json", [{"topic_id": "2000", "status": "written", "object_key": "k"}]
        )
        second = self.write(
            "other.json",
            [{"topic_id": "2001", "status": "written", "object_key": "k2"}],
            topic_catalog_digest="d" * 64,
        )
        with self.assertRaises(HandoffSourceError):
            merge_handoff_manifests([first, second])

    def test_a_single_manifest_merges_to_itself(self) -> None:
        from trec_ragtime.report_handoff import merge_handoff_manifests

        path = self.write(
            "plan.json", [{"topic_id": "2000", "status": "written", "object_key": "k"}]
        )
        merged = merge_handoff_manifests([path])
        self.assertEqual(merged.available_topic_ids, ("2000",))

    def test_no_manifest_is_rejected(self) -> None:
        from trec_ragtime.report_handoff import merge_handoff_manifests

        with self.assertRaises(HandoffSourceError):
            merge_handoff_manifests([])
