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


PARAGRAPH_PREFERENCE_CASE: dict[str, object] = {
    "name": "paragraph_preference",
    "document_id": "doc-paragraph-preference-001",
    "canonical_text": ("First alpha. First beta.\n\nSecond alpha. Second beta."),
    "selected_text": "First alpha. First beta.\n\nSecond alpha. Second beta.",
    "representation": "canonical",
    "language": "eng",
    "pipeline_name": "en_core_web_sm",
    "max_length": 8,
    "max_passage_length": 6,
    "overlap_tokens": 0,
    "expected_spans": [[0, 26], [26, 52]],
    "expected_boundaries": ["sentence", "sentence"],
}


class FixturePairLength:
    passage_length_function = staticmethod(lambda text: len(text.split()))

    def __init__(self, case: dict[str, object]) -> None:
        self.fingerprint = f"fixture-pair-{case['name']}"
        self.max_length = int(case["max_length"])
        self.max_passage_length = int(case["max_passage_length"])

    def __call__(self, text: str) -> int:
        return self.passage_length_function(text) + 2


class PassageContractTests(unittest.TestCase):
    def test_benchmark_records_only_compact_passage_fields(self) -> None:
        from scripts.benchmark_passage_chunking import record_passages

        passage = Passage(
            document_id="doc-1",
            representation="canonical",
            language="eng",
            text="One.",
            start_char=0,
            end_char=4,
            content_hash=bytes.fromhex(
                "70277bfc409b502132d7a48b2a3b386294264b31f7bf024a5fb0cafabb8ac122"
            ),
            boundary="sentence",
        )

        self.assertEqual(
            record_passages((passage,)),
            [
                {
                    "boundary": "sentence",
                    "content_hash": (
                        "70277bfc409b502132d7a48b2a3b386294264b31f7bf024a5fb0cafabb8ac122"
                    ),
                    "document_id": "doc-1",
                    "end_char": 4,
                    "language": "eng",
                    "representation": "canonical",
                    "start_char": 0,
                    "text": "One.",
                }
            ],
        )

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
    def test_paragraphs_that_cannot_fit_together_split_on_the_blank_line(self) -> None:
        """Renamed: the two paragraphs here total 8 words against a 6-word
        budget, so this has only ever shown that a split lands on the paragraph
        boundary rather than mid-paragraph. Whether paragraphs that DO fit stay
        separate is a different property, covered by the test below.
        """

        case = PARAGRAPH_PREFERENCE_CASE
        selected_text = str(case["selected_text"])
        passages = self.chunk_case(case)

        self.assert_exact_passages(
            selected_text,
            passages,
            [tuple(span) for span in case["expected_spans"]],  # type: ignore[arg-type]
        )
        self.assertEqual(
            [passage.text for passage in passages],
            [
                "First alpha. First beta.\n\n",
                "Second alpha. Second beta.",
            ],
        )
        self.assertEqual(
            [passage.boundary for passage in passages],
            case["expected_boundaries"],
        )

    def test_paragraphs_that_fit_are_packed_into_one_passage(self) -> None:
        """A short paragraph must not become a passage of its own.

        Flushing at every blank line made headlines and bylines standalone
        passages of 75-200 characters, so the sentence naming their subject
        landed in the next passage and the citation missed it by a few dozen
        characters. Splitting still only happens on a blank line; it just waits
        until the budget is actually spent.
        """

        case = dict(PARAGRAPH_PREFERENCE_CASE)
        case["name"] = "paragraph_packing"
        case["max_length"] = 64
        case["max_passage_length"] = 32
        case["expected_spans"] = [[0, 52]]
        case["expected_boundaries"] = ["sentence"]
        passages = self.chunk_case(case)

        self.assertEqual(len(passages), 1)
        self.assertEqual(
            passages[0].text, "First alpha. First beta.\n\nSecond alpha. Second beta."
        )

    def test_oversized_span_uses_full_document_sentence_context(self) -> None:
        import trec_ragtime.passages as passages_module
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        prefix = "Context. "
        oversized = "Alpha beta gamma. Delta epsilon zeta."
        suffix = " Tail."
        selected_text = prefix + oversized + suffix
        tokenizer_inputs: list[str] = []

        class PairLength:
            fingerprint = "context-sensitive-oversized-sentence-v1"
            max_length = 5
            max_passage_length = 3
            passage_length_function = staticmethod(lambda text: len(text.split()))

            def __call__(self, text: str) -> int:
                return len(text.split())

        class FakeDocument:
            def __init__(self, text: str, start_char: int) -> None:
                self.page_content = text
                self.metadata = {"start_index": start_char}

        class FakeSentence:
            def __init__(self, start_char: int, text_with_ws: str) -> None:
                self.start_char = start_char
                self.text_with_ws = text_with_ws

        class FakeTokenizedDocument:
            def __init__(self, sentences: tuple[FakeSentence, ...]) -> None:
                self.sents = sentences

        class FakeSentenceSplitter:
            def __init__(self, **_: object) -> None:
                pass

            def _tokenizer(self, text: str) -> FakeTokenizedDocument:
                tokenizer_inputs.append(text)
                if text == selected_text:
                    return FakeTokenizedDocument(
                        (
                            FakeSentence(0, prefix),
                            FakeSentence(len(prefix), oversized),
                            FakeSentence(len(prefix) + len(oversized) + 1, "Tail."),
                        )
                    )
                if text == oversized:
                    first = "Alpha beta gamma. "
                    return FakeTokenizedDocument(
                        (
                            FakeSentence(0, first),
                            FakeSentence(len(first), text[len(first) :]),
                        )
                    )
                raise AssertionError(f"unexpected tokenizer input: {text!r}")

            def create_documents(self, texts: list[str]) -> list[FakeDocument]:
                return [
                    FakeDocument(prefix, 0),
                    FakeDocument(oversized, len(prefix)),
                    FakeDocument(" Tail.", len(prefix) + len(oversized)),
                ]

        with patch.object(
            passages_module,
            "_load_splitters",
            return_value=(FakeSentenceSplitter, RecursiveCharacterTextSplitter),
        ):
            chunker = PassageChunker(PairLength(), overlap_tokens=0)
            passages = chunker.chunk(
                document_id="context-sensitive-oversized-sentence",
                canonical_text=selected_text,
                selected_text=selected_text,
                representation="canonical",
                language="eng",
            )

        self.assertEqual(tokenizer_inputs, [selected_text])
        self.assertEqual(
            [passage.boundary for passage in passages],
            ["sentence", "oversized_sentence", "oversized_sentence", "sentence"],
        )
        self.assertEqual("".join(passage.text for passage in passages), selected_text)

    def test_oversized_multi_sentence_span_emits_authoritative_units(self) -> None:
        import trec_ragtime.passages as passages_module
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        selected_text = "Alpha beta.  Gamma delta.\nEpsilon zeta."
        first_sentence_end = selected_text.index("Gamma")
        second_sentence_end = selected_text.index("Epsilon")

        class PairLength:
            fingerprint = "multi-sentence-direct-units-v1"
            max_length = 5
            max_passage_length = 4
            passage_length_function = staticmethod(lambda text: len(text.split()))

            def __call__(self, text: str) -> int:
                return len(text.split())

        class FakeDocument:
            def __init__(self, text: str, start_char: int) -> None:
                self.page_content = text
                self.metadata = {"start_index": start_char}

        class FakeSentence:
            def __init__(self, start_char: int, text_with_ws: str) -> None:
                self.start_char = start_char
                self.text_with_ws = text_with_ws

        first_sentence = "Alpha beta.  "
        second_sentence = "Gamma delta.\n"
        third_sentence = "Epsilon zeta."

        class FakeTokenizedDocument:
            sents = (
                FakeSentence(0, first_sentence),
                FakeSentence(len(first_sentence), second_sentence),
                FakeSentence(
                    len(first_sentence) + len(second_sentence), third_sentence
                ),
            )

        class FakeSentenceSplitter:
            def __init__(self, **_: object) -> None:
                pass

            def _tokenizer(self, text: str) -> FakeTokenizedDocument:
                if text != selected_text:
                    raise AssertionError(f"unexpected tokenizer input: {text!r}")
                return FakeTokenizedDocument()

            def create_documents(self, texts: list[str]) -> list[FakeDocument]:
                return [FakeDocument(texts[0], 0)]

        with patch.object(
            passages_module,
            "_load_splitters",
            return_value=(FakeSentenceSplitter, RecursiveCharacterTextSplitter),
        ):
            chunker = PassageChunker(PairLength(), overlap_tokens=0)
            with patch.object(
                PassageChunker,
                "_recursive_splitter",
                side_effect=AssertionError(
                    "RCTS must not receive multiple authoritative sentences"
                ),
            ):
                passages = chunker.chunk(
                    document_id="doc-multi-sentence-direct-units",
                    canonical_text=selected_text,
                    selected_text=selected_text,
                    representation="canonical",
                    language="eng",
                )

        self.assert_exact_passages(
            selected_text,
            passages,
            [
                (0, first_sentence_end),
                (first_sentence_end, second_sentence_end),
                (second_sentence_end, len(selected_text)),
            ],
        )
        self.assertEqual(
            [passage.text for passage in passages],
            ["Alpha beta.  ", "Gamma delta.\n", "Epsilon zeta."],
        )
        self.assertEqual(
            [passage.boundary for passage in passages],
            ["sentence", "sentence", "sentence"],
        )

    def test_oversized_sentence_retries_rcts_until_exact_pair_fits(self) -> None:
        import trec_ragtime.passages as passages_module
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        selected_text = " " + "x " * 422 + "end"

        class PairLength:
            fingerprint = "nonlinear-oversized-sentence-v1"
            max_length = 512
            max_passage_length = 423
            passage_length_function = staticmethod(lambda text: len(text.split()))

            def __call__(self, text: str) -> int:
                return 89 + len(text.split()) + int(text.endswith("end"))

        class FakeDocument:
            def __init__(self, text: str, start_char: int) -> None:
                self.page_content = text
                self.metadata = {"start_index": start_char}

        class FakeSentence:
            start_char = 0
            text_with_ws = selected_text

        class FakeTokenizedDocument:
            sents = (FakeSentence(),)

        class FakeSentenceSplitter:
            def __init__(self, **_: object) -> None:
                pass

            def _tokenizer(self, text: str) -> FakeTokenizedDocument:
                if text != selected_text:
                    raise AssertionError("RCTS received more than one sentence")
                return FakeTokenizedDocument()

            def create_documents(self, texts: list[str]) -> list[FakeDocument]:
                return [FakeDocument(texts[0], 0)]

        rcts_chunk_sizes: list[int] = []
        rcts_inputs: list[str] = []

        class RecordingRecursiveTextSplitter(RecursiveCharacterTextSplitter):
            def __init__(self, *, chunk_size: int, **kwargs: object) -> None:
                rcts_chunk_sizes.append(chunk_size)
                super().__init__(chunk_size=chunk_size, **kwargs)

            def create_documents(self, texts: list[str]) -> list[FakeDocument]:
                rcts_inputs.extend(texts)
                return super().create_documents(texts)  # type: ignore[return-value]

        pair_length = PairLength()
        with patch.object(
            passages_module,
            "_load_splitters",
            return_value=(FakeSentenceSplitter, RecordingRecursiveTextSplitter),
        ):
            chunker = PassageChunker(pair_length, overlap_tokens=0)
            passages = chunker.chunk(
                document_id="nonlinear-oversized-sentence",
                canonical_text=selected_text,
                selected_text=selected_text,
                representation="canonical",
                language="eng",
            )

        self.assertEqual(pair_length.passage_length_function(selected_text), 423)
        self.assertEqual(pair_length(selected_text), 513)
        self.assertEqual(rcts_chunk_sizes, [423, 422])
        self.assertEqual(rcts_inputs, [selected_text, selected_text])
        self.assert_exact_passages(selected_text, passages, [(0, 844), (844, 848)])
        self.assertEqual(
            [passage.boundary for passage in passages],
            [
                "oversized_sentence",
                "oversized_sentence",
            ],
        )
        self.assertTrue(
            all(
                pair_length(passage.text) <= pair_length.max_length
                for passage in passages
            )
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
                    all(
                        pair_length(passage.text) <= pair_length.max_length
                        for passage in passages
                    )
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
            self.assertEqual(
                selected_text[start:end],
                case["sentences"][sentence_spans.index(overlap)],
            )
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

    def test_repeated_text_overlap_recovers_authoritative_intervals_and_coverage(
        self,
    ) -> None:
        case = next(
            case
            for case in load_cases()
            if case["name"] == "repeated_identical_sentences_overlap"
        )
        selected_text = str(case["selected_text"])
        passages = self.chunk_case(case)

        self.assert_exact_passages(
            selected_text,
            passages,
            [tuple(span) for span in case["expected_spans"]],  # type: ignore[arg-type]
        )
        self.assertEqual(
            [passage.text for passage in passages],
            ["Repeat. Repeat. ", "Repeat. Repeat. ", "Repeat. Repeat. "],
        )
        self.assertEqual(len({passage.content_hash for passage in passages}), 1)
        self.assertEqual(
            len({(passage.start_char, passage.end_char) for passage in passages}),
            len(passages),
        )
        overlaps = [
            (
                max(first.start_char, second.start_char),
                min(first.end_char, second.end_char),
            )
            for first, second in zip(passages, passages[1:])
        ]
        self.assertEqual(overlaps, [(8, 16), (16, 24)])
        self.assertTrue(all(start < end for start, end in overlaps))

    def test_missing_pipeline_metadata_fails_without_source_text(self) -> None:
        import trec_ragtime.passages as passages_module

        versions = {
            "langchain-text-splitters": "1.1.2",
            "spacy": "3.8.0",
            "en-core-web-sm": "3.8.0",
            "es-core-news-sm": "3.8.0",
            "ru-core-news-sm": "3.8.0",
            "zh-core-web-sm": "3.8.0",
        }
        secret_text = "secret source text must not appear in metadata errors"

        def version(distribution: str) -> str:
            if distribution == "es-core-news-sm":
                raise passages_module.importlib.metadata.PackageNotFoundError(
                    distribution
                )
            return versions[distribution]

        with patch.object(
            passages_module.importlib.metadata,
            "version",
            side_effect=version,
        ):
            with self.assertRaises(PassageChunkingError) as context:
                PassageChunker(FixturePairLength(load_cases()[0]), overlap_tokens=0)
        self.assertIn("es-core-news-sm", str(context.exception))
        self.assertNotIn(secret_text, str(context.exception))

    def test_pipeline_distribution_version_rotates_chunker_fingerprint(self) -> None:
        import trec_ragtime.passages as passages_module

        base_versions = {
            "langchain-text-splitters": "1.1.2",
            "spacy": "3.8.0",
            "en-core-web-sm": "3.8.0",
            "es-core-news-sm": "3.8.0",
            "ru-core-news-sm": "3.8.0",
            "zh-core-web-sm": "3.8.0",
        }

        def versions_for(values: dict[str, str]):
            return lambda distribution: values[distribution]

        pair_length = FixturePairLength(load_cases()[0])
        with patch.object(
            passages_module.importlib.metadata,
            "version",
            side_effect=versions_for(base_versions),
        ):
            first = PassageChunker(pair_length, overlap_tokens=0)

        changed_versions = base_versions | {"zh-core-web-sm": "3.8.1"}
        with patch.object(
            passages_module.importlib.metadata,
            "version",
            side_effect=versions_for(changed_versions),
        ):
            second = PassageChunker(pair_length, overlap_tokens=0)

        self.assertNotEqual(first.fingerprint, second.fingerprint)
