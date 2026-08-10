from __future__ import annotations

import unittest
import json
from pathlib import Path
from unittest.mock import patch

from trec_ragtime import Boundary, Representation
from trec_ragtime.passages import (
    Passage,
    PassageChunker,
    PassageChunkingError,
    sha256_bytes,
    validate_passages,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "passage_chunking.json"


def load_cases() -> list[dict[str, object]]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["cases"]


class FixturePairLength:
    passage_length_function = staticmethod(lambda text: len(text.split()))

    def __init__(self, case: dict[str, object]) -> None:
        self.fingerprint = f"fixture-pair-{case['name']}"
        self.max_length = int(case["max_length"])
        self.max_passage_length = int(case["max_passage_length"])

    def __call__(self, text: str) -> int:
        return self.passage_length_function(text) + 2


class PassageContractTests(unittest.TestCase):
    def test_package_exports_compact_passage_api(self) -> None:
        namespace: dict[str, object] = {}
        exec(
            "from trec_ragtime import Boundary, Passage, PassageChunker, "
            "PassageChunkingError, Representation, sha256_bytes, validate_passages",
            namespace,
        )
        self.assertIs(namespace["Passage"], Passage)
        self.assertIs(namespace["PassageChunker"], PassageChunker)
        self.assertIs(namespace["PassageChunkingError"], PassageChunkingError)
        self.assertIs(namespace["sha256_bytes"], sha256_bytes)
        self.assertIs(namespace["validate_passages"], validate_passages)
        self.assertEqual(namespace["Boundary"], Boundary)
        self.assertEqual(namespace["Representation"], Representation)

    def test_passage_contract_is_compact_and_immutable(self) -> None:
        fields = tuple(Passage.__dataclass_fields__)
        self.assertEqual(
            fields,
            (
                "document_id",
                "representation",
                "language",
                "text",
                "start_char",
                "end_char",
                "content_hash",
                "boundary",
            ),
        )
        passage = Passage(
            document_id="doc-1",
            representation="canonical",
            language="eng",
            text="One.",
            start_char=0,
            end_char=4,
            content_hash=sha256_bytes("One."),
            boundary="sentence",
        )
        with self.assertRaises((AttributeError, TypeError)):
            passage.text = "changed"  # type: ignore[misc]

    def test_passage_rejects_invalid_domain_values(self) -> None:
        valid = {
            "document_id": "doc-1",
            "representation": "canonical",
            "language": "eng",
            "text": "One.",
            "start_char": 0,
            "end_char": 4,
            "content_hash": sha256_bytes("One."),
            "boundary": "sentence",
        }
        invalid_values = (
            {"document_id": ""},
            {"representation": "other"},
            {"language": ""},
            {"representation": "report_text", "language": "spa"},
            {"start_char": -1},
            {"end_char": 3},
            {"content_hash": b"0" * 32},
            {"boundary": "paragraph"},
        )
        for change in invalid_values:
            with self.subTest(change=change):
                with self.assertRaises(PassageChunkingError):
                    Passage(**(valid | change))  # type: ignore[arg-type]


class PassageValidationTests(unittest.TestCase):
    def make_passage(self, selected_text: str, start: int, end: int) -> Passage:
        text = selected_text[start:end]
        return Passage(
            document_id="doc-1",
            representation="canonical",
            language="eng",
            text=text,
            start_char=start,
            end_char=end,
            content_hash=sha256_bytes(text),
            boundary="sentence",
        )

    def test_validator_allows_overlap_but_not_gaps(self) -> None:
        selected_text = "abcdef"
        validate_passages(
            selected_text,
            (
                self.make_passage(selected_text, 0, 4),
                self.make_passage(selected_text, 2, 6),
            ),
        )
        with self.assertRaisesRegex(PassageChunkingError, "gap"):
            validate_passages(
                selected_text,
                (
                    self.make_passage(selected_text, 0, 2),
                    self.make_passage(selected_text, 3, 6),
                ),
            )


class PassageChunkerTests(unittest.TestCase):
    def test_multi_sentence_over_ceiling_rejects_before_rcts(self) -> None:
        class PairLength:
            fingerprint = "multi-sentence-guard-v1"
            max_length = 5
            max_passage_length = 3
            passage_length_function = staticmethod(lambda text: len(text.split()))

            def __call__(self, text: str) -> int:
                if text.count(".") >= 2:
                    return 6
                return len(text.split()) + 1

        chunker = PassageChunker(PairLength(), overlap_tokens=0)
        with patch.object(
            PassageChunker,
            "_recursive_splitter",
            side_effect=AssertionError("RCTS must not receive multiple sentences"),
        ):
            with self.assertRaisesRegex(
                PassageChunkingError, "exactly one underlying sentence"
            ):
                chunker.chunk(
                    document_id="doc-multi-sentence-over-ceiling",
                    canonical_text="One. Two.",
                    selected_text="One. Two.",
                    representation="canonical",
                    language="eng",
                )

    def assert_exact_passages(
        self,
        selected_text: str,
        passages: tuple[Passage, ...],
        expected_spans: list[tuple[int, int]],
    ) -> None:
        self.assertEqual(
            [(passage.start_char, passage.end_char) for passage in passages],
            expected_spans,
        )
        for passage in passages:
            self.assertEqual(
                selected_text[passage.start_char : passage.end_char],
                passage.text,
            )
            self.assertEqual(passage.content_hash, sha256_bytes(passage.text))
        validate_passages(selected_text, passages)
        covered: set[int] = set()
        for passage in passages:
            covered.update(range(passage.start_char, passage.end_char))
        self.assertEqual(covered, set(range(len(selected_text))))

    def chunk_case(self, case: dict[str, object]) -> tuple[Passage, ...]:
        pair_length = FixturePairLength(case)
        chunker = PassageChunker(
            pair_length,
            overlap_tokens=int(case["overlap_tokens"]),
        )
        return chunker.chunk(
            document_id=str(case["document_id"]),
            canonical_text=str(case["canonical_text"]),
            selected_text=str(case["selected_text"]),
            representation=str(case["representation"]),  # type: ignore[arg-type]
            language=str(case["language"]),
        )

    def test_fixture_cases_preserve_literal_spans_hashes_and_boundaries(self) -> None:
        for case in load_cases():
            with self.subTest(case=case["name"]):
                selected_text = str(case["selected_text"])
                passages = self.chunk_case(case)
                expected_spans = [tuple(span) for span in case["expected_spans"]]
                self.assert_exact_passages(
                    selected_text,
                    passages,
                    expected_spans,  # type: ignore[arg-type]
                )
                self.assertEqual(
                    [passage.boundary for passage in passages],
                    case["expected_boundaries"],
                )
                pair_length = FixturePairLength(case)
                self.assertTrue(
                    all(pair_length(passage.text) <= pair_length.max_length for passage in passages)
                )

    def test_overlap_intersections_are_complete_sentence_units(self) -> None:
        case = next(case for case in load_cases() if case["name"] == "overlap")
        selected_text = str(case["selected_text"])
        passages = self.chunk_case(case)
        sentence_spans = [tuple(span) for span in case["sentence_spans"]]
        overlaps: list[tuple[int, int]] = []
        for first, second in zip(passages, passages[1:]):
            start = max(first.start_char, second.start_char)
            end = min(first.end_char, second.end_char)
            self.assertLess(start, end)
            overlap = (start, end)
            overlaps.append(overlap)
            self.assertIn(overlap, sentence_spans)
            self.assertEqual(selected_text[start:end], case["sentences"][sentence_spans.index(overlap)])
        self.assertEqual(overlaps, sentence_spans)

    def test_repeated_text_keeps_distinct_intervals_with_equal_hashes(self) -> None:
        case = next(
            case
            for case in load_cases()
            if case["name"] == "repeated_identical_sentences"
        )
        passages = self.chunk_case(case)
        self.assertEqual(len({passage.content_hash for passage in passages}), 1)
        self.assertEqual(
            len({(passage.start_char, passage.end_char) for passage in passages}),
            len(passages),
        )
