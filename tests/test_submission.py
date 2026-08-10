from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

from trec_ragtime.submission import (
    MAX_MLIR_HITS,
    MLIRRankedHit,
    MLIRRun,
    ReportMetadata,
    ReportSentence,
    ReportSubmission,
    SubmissionValidationError,
    validate_mlir_runs,
    validate_report_run,
    write_mlir_run,
    write_report_jsonl,
)


def metadata(
    topic_id: str = "2000",
    *,
    team_id: str = "team",
    run_id: str = "multi-rag",
    run_desc: str = "Multilingual RAG baseline",
) -> ReportMetadata:
    return ReportMetadata(
        team_id=team_id,
        topic_id=topic_id,
        run_id=run_id,
        run_desc=run_desc,
    )


def report(topic_id: str = "2000", text: str = "Supported claim.") -> ReportSubmission:
    return ReportSubmission.build(
        metadata=metadata(topic_id),
        responses=(ReportSentence(text, {f"doc-{topic_id}": 1.0}),),
    )


class ReportSubmissionTests(unittest.TestCase):
    def test_metadata_is_frozen_and_serializes_official_fields(self) -> None:
        value = metadata()

        self.assertEqual(
            value.as_dict(),
            {
                "team_id": "team",
                "topic_id": "2000",
                "run_id": "multi-rag",
                "run_desc": "Multilingual RAG baseline",
            },
        )
        with self.assertRaises(FrozenInstanceError):
            value.run_id = "changed"  # type: ignore[misc]

    def test_metadata_rejects_invalid_identifiers_and_description(self) -> None:
        invalid_cases = (
            {"team_id": "", "topic_id": "2000", "run_id": "run", "run_desc": "x"},
            {
                "team_id": "my team",
                "topic_id": "2000",
                "run_id": "run",
                "run_desc": "x",
            },
            {"team_id": "team", "topic_id": "", "run_id": "run", "run_desc": "x"},
            {"team_id": "team", "topic_id": "20 00", "run_id": "run", "run_desc": "x"},
            {"team_id": "team", "topic_id": "2000", "run_id": "", "run_desc": "x"},
            {
                "team_id": "team",
                "topic_id": "2000",
                "run_id": "bad run",
                "run_desc": "x",
            },
            {"team_id": "team", "topic_id": "2000", "run_id": "run", "run_desc": "  "},
            {"team_id": "team", "topic_id": "2000", "run_id": "run", "run_desc": " x"},
        )
        for values in invalid_cases:
            with self.subTest(values=values):
                with self.assertRaises(SubmissionValidationError):
                    ReportMetadata(**values)

        with self.assertRaisesRegex(SubmissionValidationError, "at most 25"):
            metadata(run_id="r" * 26)
        self.assertEqual(metadata(run_id="r" * 25).run_id, "r" * 25)

    def test_sentence_copies_and_deterministically_orders_citations(self) -> None:
        source = {"doc-z": 0.5, "doc-b": 0.9, "doc-a": 0.9}
        sentence = ReportSentence("A claim.", source)
        source["later"] = 100.0

        self.assertEqual(
            list(sentence.citations.items()),
            [("doc-a", 0.9), ("doc-b", 0.9), ("doc-z", 0.5)],
        )
        self.assertNotIn("later", sentence.citations)
        with self.assertRaises(TypeError):
            sentence.citations["new"] = 1.0  # type: ignore[index]

    def test_sentence_rejects_bad_text_document_ids_and_scores(self) -> None:
        with self.assertRaisesRegex(SubmissionValidationError, "text"):
            ReportSentence(" ", {})
        with self.assertRaisesRegex(SubmissionValidationError, "whitespace"):
            ReportSentence("claim", {"bad id": 1.0})

        for score in (True, "1", float("nan"), float("inf"), -float("inf")):
            with self.subTest(score=score):
                with self.assertRaises(SubmissionValidationError):
                    ReportSentence("claim", {"doc": score})  # type: ignore[dict-item]

    def test_report_factory_computes_sorted_exact_reference_union(self) -> None:
        submission = ReportSubmission.build(
            metadata=metadata(),
            responses=(
                ReportSentence("First.", {"doc-b": 1, "doc-a": 2}),
                ReportSentence("Second.", {"doc-c": 1, "doc-a": 0.5}),
            ),
        )

        self.assertEqual(submission.references, ("doc-a", "doc-b", "doc-c"))
        self.assertEqual(submission.responses[0].citations["doc-b"], 1.0)
        self.assertEqual(
            submission.as_dict()["references"], ["doc-a", "doc-b", "doc-c"]
        )

    def test_official_schema_allows_an_empty_no_output_report(self) -> None:
        submission = ReportSubmission.build(metadata=metadata(), responses=())

        self.assertEqual(submission.responses, ())
        self.assertEqual(submission.references, ())

    def test_report_constructor_rejects_bad_references(self) -> None:
        sentence = ReportSentence("Claim.", {"doc-a": 1.0})
        for references in ((), ("doc-a", "doc-b"), ("doc-a", "doc-a")):
            with self.subTest(references=references):
                with self.assertRaisesRegex(SubmissionValidationError, "references"):
                    ReportSubmission(metadata(), (sentence,), references)

    def test_nfkc_limit_counts_space_joined_complete_report(self) -> None:
        # The ffi ligature expands from one code point to three under NFKC.
        submission = ReportSubmission.build(
            metadata=metadata(),
            responses=(
                ReportSentence("\ufb03", {}),
                ReportSentence("x", {}),
            ),
        )

        self.assertEqual(submission.report_text, "\ufb03 x")
        self.assertEqual(submission.nfkc_character_count, 5)
        submission.validate_limit(5)
        with self.assertRaisesRegex(SubmissionValidationError, "5 > 4"):
            submission.validate_limit(4)
        for bad_limit in (True, -1, 5.0):
            with self.subTest(limit=bad_limit):
                with self.assertRaises(SubmissionValidationError):
                    submission.validate_limit(bad_limit)  # type: ignore[arg-type]

    def test_validate_report_run_checks_complete_consistent_identity(self) -> None:
        first = report("2000")
        second = report("2001")
        self.assertEqual(
            tuple(item.metadata.topic_id for item in validate_report_run(
                (second, first), topic_limits={"2000": 100, "2001": 100}
            )),
            ("2000", "2001"),
        )

        with self.assertRaisesRegex(SubmissionValidationError, "duplicate topics"):
            validate_report_run(
                (first, first), topic_limits={"2000": 100}
            )
        with self.assertRaisesRegex(SubmissionValidationError, "missing=.*2001"):
            validate_report_run(
                (first,), topic_limits={"2000": 100, "2001": 100}
            )
        with self.assertRaisesRegex(SubmissionValidationError, "consistent"):
            validate_report_run(
                (
                    first,
                    ReportSubmission.build(
                        metadata=metadata("2001", run_id="other"),
                        responses=(ReportSentence("Claim.", {"doc": 1}),),
                    ),
                ),
                topic_limits={"2000": 100, "2001": 100},
            )

    def test_jsonl_writer_is_deterministic_and_validates_first(self) -> None:
        first = ReportSubmission.build(
            metadata=metadata("2000"),
            responses=(
                ReportSentence("Caf\u00e9.", {"doc-z": 0.1, "doc-a": 0.9}),
            ),
        )
        second = report("2001")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reports.jsonl"
            write_report_jsonl(
                path,
                (second, first),
                topic_limits={"2000": 100, "2001": 100},
            )
            raw = path.read_text(encoding="utf-8")
            lines = raw.splitlines()

            self.assertEqual(len(lines), 2)
            self.assertIn("Caf\u00e9", raw)
            self.assertNotIn(": ", raw)
            decoded = [json.loads(line) for line in lines]
            self.assertEqual(
                [item["metadata"]["topic_id"] for item in decoded],
                ["2000", "2001"],
            )
            self.assertEqual(
                list(decoded[0]["responses"][0]["citations"]),
                ["doc-a", "doc-z"],
            )

            untouched = Path(directory) / "untouched.jsonl"
            untouched.write_text("sentinel", encoding="utf-8")
            with self.assertRaises(SubmissionValidationError):
                write_report_jsonl(
                    untouched,
                    (first,),
                    topic_limits={"2000": 2},
                )
            self.assertEqual(untouched.read_text(encoding="utf-8"), "sentinel")

            with patch.object(
                Path,
                "replace",
                side_effect=OSError("simulated publication failure"),
            ):
                with self.assertRaisesRegex(OSError, "publication failure"):
                    write_report_jsonl(
                        untouched,
                        (first,),
                        topic_limits={"2000": 100},
                    )
            self.assertEqual(untouched.read_text(encoding="utf-8"), "sentinel")
            self.assertEqual(list(Path(directory).glob(".*.tmp")), [])


class MLIRSubmissionTests(unittest.TestCase):
    def test_ranked_hit_validates_rank_document_and_finite_score(self) -> None:
        hit = MLIRRankedHit(rank=1, document_id="doc-a", score=2)
        self.assertEqual(hit.score, 2.0)
        for kwargs in (
            {"rank": 0, "document_id": "doc", "score": 1.0},
            {"rank": True, "document_id": "doc", "score": 1.0},
            {"rank": 1, "document_id": "bad doc", "score": 1.0},
            {"rank": 1, "document_id": "doc", "score": True},
            {"rank": 1, "document_id": "doc", "score": float("nan")},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(SubmissionValidationError):
                    MLIRRankedHit(**kwargs)

    def test_run_rejects_invalid_rankings(self) -> None:
        with self.assertRaisesRegex(SubmissionValidationError, "consecutive"):
            MLIRRun("2000", "run", (MLIRRankedHit(2, "doc", 1),))
        with self.assertRaisesRegex(SubmissionValidationError, "duplicate document"):
            MLIRRun(
                "2000",
                "run",
                (MLIRRankedHit(1, "doc", 2), MLIRRankedHit(2, "doc", 1)),
            )
        with self.assertRaisesRegex(SubmissionValidationError, "descending score"):
            MLIRRun(
                "2000",
                "run",
                (MLIRRankedHit(1, "doc-a", 1), MLIRRankedHit(2, "doc-b", 2)),
            )
        with self.assertRaisesRegex(SubmissionValidationError, "document ID"):
            MLIRRun(
                "2000",
                "run",
                (MLIRRankedHit(1, "doc-b", 1), MLIRRankedHit(2, "doc-a", 1)),
            )
        too_many = tuple(
            MLIRRankedHit(rank, f"doc-{rank:04d}", MAX_MLIR_HITS - rank)
            for rank in range(1, MAX_MLIR_HITS + 2)
        )
        with self.assertRaisesRegex(SubmissionValidationError, "at most 1000"):
            MLIRRun("2000", "run", too_many)

    def test_from_scores_deduplicates_sorts_ranks_and_caps_fused_hits(self) -> None:
        run = MLIRRun.from_scores(
            topic_id="2000",
            run_id="fusion",
            scores={"doc-z": 0.2, "doc-b": 0.9, "doc-a": 0.9},
        )
        self.assertEqual(
            tuple((hit.rank, hit.document_id, hit.score) for hit in run.hits),
            ((1, "doc-a", 0.9), (2, "doc-b", 0.9), (3, "doc-z", 0.2)),
        )

        capped = MLIRRun.from_scores(
            topic_id="2000",
            run_id="fusion",
            scores=((f"doc-{index}", float(index)) for index in range(10)),
            max_hits=3,
        )
        self.assertEqual(
            tuple(hit.document_id for hit in capped.hits),
            ("doc-9", "doc-8", "doc-7"),
        )
        with self.assertRaisesRegex(SubmissionValidationError, "duplicate"):
            MLIRRun.from_scores(
                topic_id="2000",
                run_id="fusion",
                scores=(("doc", 2), ("doc", 1)),
            )
        for max_hits in (True, 0, 1001):
            with self.subTest(max_hits=max_hits):
                with self.assertRaises(SubmissionValidationError):
                    MLIRRun.from_scores(
                        topic_id="2000",
                        run_id="fusion",
                        scores={"doc": 1},
                        max_hits=max_hits,  # type: ignore[arg-type]
                    )

    def test_validate_mlir_runs_checks_topic_set_duplicates_and_run_id(self) -> None:
        first = MLIRRun.from_scores(topic_id="2000", run_id="run", scores={"a": 1})
        second = MLIRRun.from_scores(topic_id="2001", run_id="run", scores={"b": 1})
        ordered = validate_mlir_runs(
            (second, first), expected_topic_ids=("2000", "2001")
        )
        self.assertEqual(tuple(run.topic_id for run in ordered), ("2000", "2001"))

        with self.assertRaisesRegex(SubmissionValidationError, "duplicate topics"):
            validate_mlir_runs((first, first))
        with self.assertRaisesRegex(SubmissionValidationError, "missing=.*2001"):
            validate_mlir_runs((first,), expected_topic_ids=("2000", "2001"))
        with self.assertRaisesRegex(SubmissionValidationError, "consistent"):
            validate_mlir_runs(
                (
                    first,
                    MLIRRun.from_scores(
                        topic_id="2001", run_id="other", scores={"b": 1}
                    ),
                )
            )

    def test_mlir_writer_emits_six_fields_and_contiguous_sorted_topics(self) -> None:
        first = MLIRRun.from_scores(
            topic_id="2000",
            run_id="fusion",
            scores={"doc-b": 0.5, "doc-a": 1},
        )
        second = MLIRRun.from_scores(
            topic_id="2001", run_id="fusion", scores={"doc-c": -2}
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mlir.run"
            write_mlir_run(
                path,
                (second, first),
                expected_topic_ids=("2000", "2001"),
            )
            lines = path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(
            lines,
            [
                "2000 Q0 doc-a 1 1.0 fusion",
                "2000 Q0 doc-b 2 0.5 fusion",
                "2001 Q0 doc-c 1 -2.0 fusion",
            ],
        )
        self.assertTrue(all(len(line.split()) == 6 for line in lines))

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                SubmissionValidationError,
                "expected_topic_ids is required",
            ):
                write_mlir_run(
                    Path(directory) / "partial.run",
                    (first,),
                    expected_topic_ids=None,  # type: ignore[arg-type]
                )


if __name__ == "__main__":
    unittest.main()
