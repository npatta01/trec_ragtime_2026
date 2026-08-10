from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import msgspec
import trec_ragtime.passages as passages_module
from trec_ragtime.passages import (
    ChunkerConfig,
    LangChainPassageChunker,
    Passage,
    PassageChunkingError,
    PassageValidationError,
    sha256_bytes,
    validate_passages,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "passage_chunking.json"


def load_cases() -> list[dict[str, object]]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["cases"]


def char_length(text: str) -> int:
    return len(text)


class PassageValidationTests(unittest.TestCase):
    def test_importing_passages_does_not_eagerly_import_scorer_stack(self) -> None:
        code = """
import sys
import trec_ragtime.passages
assert 'sentence_transformers' not in sys.modules
assert 'transformers' not in sys.modules
"""
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_sha256_bytes_hashes_exact_utf8_bytes_without_normalization(self) -> None:
        self.assertEqual(
            sha256_bytes("café"),
            bytes.fromhex("850f7dc43910ff890f8879c0ed26fe697c93a067ad93a7d50f466a7028a9bf4e"),
        )

    def test_zero_chunk_size_reaches_strict_config_validation(self) -> None:
        with self.assertRaises(PassageValidationError):
            LangChainPassageChunker(chunk_size=0)

    def test_custom_length_function_requires_a_stable_identity(self) -> None:
        with self.assertRaises(PassageValidationError):
            ChunkerConfig(length_function=char_length)
        with self.assertRaises(PassageValidationError):
            LangChainPassageChunker(length_function=char_length)

    def test_chunker_fingerprint_changes_for_result_affecting_identity_inputs(self) -> None:
        base = ChunkerConfig(
            chunk_size=32,
            length_function=char_length,
            length_function_fingerprint="chars-v1",
        )

        self.assertNotEqual(
            base.resolved_fingerprint,
            replace(base, chunk_size=33).resolved_fingerprint,
        )
        self.assertNotEqual(
            base.resolved_fingerprint,
            replace(base, pipeline_name="en_core_web_sm").resolved_fingerprint,
        )
        self.assertNotEqual(
            base.resolved_fingerprint,
            replace(base, pipeline_name="es_core_news_sm").resolved_fingerprint,
        )
        self.assertNotEqual(
            base.resolved_fingerprint,
            replace(base, length_function_fingerprint="chars-v2").resolved_fingerprint,
        )

    def test_validate_passages_rejects_whitespace_stripping_at_reconstruction(self) -> None:
        selected_text, passages = _packed_passages()
        first, second = passages
        stripped = first.text.strip()
        mutated_first = replace(
            first,
            text=stripped,
            end_char=first.start_char + len(stripped),
            passage_id=_lineage_id(
                first,
                end_char=first.start_char + len(stripped),
            ),
            content_hash=sha256_bytes(stripped),
        )
        mutated_second = replace(
            second,
            previous_passage_id=mutated_first.passage_id,
        )

        with self.assertRaisesRegex(PassageValidationError, "monotonic"):
            validate_passages(selected_text, (mutated_first, mutated_second))

    def test_validate_passages_rejects_reordered_or_overlapping_spans(self) -> None:
        selected_text, passages = _packed_passages()
        first, second = passages

        reordered_first = replace(
            second,
            chunk_index=0,
            passage_id=_lineage_id(second, chunk_index=0),
            previous_passage_id=None,
        )
        reordered_second = replace(
            first,
            chunk_index=1,
            passage_id=_lineage_id(first, chunk_index=1),
            previous_passage_id=reordered_first.passage_id,
            next_passage_id=None,
        )
        reordered_first = replace(
            reordered_first,
            next_passage_id=reordered_second.passage_id,
        )
        with self.assertRaisesRegex(PassageValidationError, "monotonic"):
            validate_passages(selected_text, (reordered_first, reordered_second))

        overlap_start = first.end_char - 1
        overlap_end = second.end_char
        overlap_text = selected_text[overlap_start:overlap_end]
        overlapping_second = replace(
            second,
            text=overlap_text,
            start_char=overlap_start,
            end_char=overlap_end,
            passage_id=_lineage_id(
                second,
                start_char=overlap_start,
                end_char=overlap_end,
            ),
            content_hash=sha256_bytes(overlap_text),
            previous_passage_id=first.passage_id,
        )
        overlapping_first = replace(
            first,
            next_passage_id=overlapping_second.passage_id,
        )
        with self.assertRaisesRegex(PassageValidationError, "monotonic"):
            validate_passages(selected_text, (overlapping_first, overlapping_second))

    def test_passage_rejects_malformed_lineage_and_fingerprint(self) -> None:
        with self.assertRaises(PassageValidationError):
            Passage(
                document_id="",
                source_content_hash=b"0" * 32,
                representation="canonical",
                representation_hash=b"1" * 32,
                language="eng",
                chunk_index=0,
                passage_id=b"2" * 32,
                text="text",
                start_char=0,
                end_char=4,
                boundary="paragraph_packed",
                previous_passage_id=None,
                next_passage_id=None,
                content_hash=b"3" * 31,
                chunker_fingerprint="",
                overlap_chars=1,
                overlap_tokens=0,
            )

    def test_passage_rejects_offset_length_mismatch(self) -> None:
        with self.assertRaises(PassageValidationError):
            Passage(
                document_id="doc",
                source_content_hash=b"0" * 32,
                representation="canonical",
                representation_hash=b"1" * 32,
                language="eng",
                chunk_index=0,
                passage_id=b"2" * 32,
                text="text",
                start_char=0,
                end_char=5,
                boundary="paragraph_packed",
                previous_passage_id=None,
                next_passage_id=None,
                content_hash=sha256_bytes("text"),
                chunker_fingerprint="test-fingerprint",
            )

    def test_validate_passages_rejects_non_passage_objects(self) -> None:
        with self.assertRaises(PassageValidationError):
            validate_passages("text", (object(),))


class LangChainPassageChunkerTests(unittest.TestCase):
    def test_fixture_cases_preserve_literal_spans_and_hash_lineage(self) -> None:
        for case in load_cases():
            with self.subTest(case=case["name"]):
                selected_text = case["selected_text"]
                chunker = LangChainPassageChunker(
                    ChunkerConfig(
                        chunk_size=case["chunk_size"],
                        length_function=char_length,
                        length_function_fingerprint=case["length_function_fingerprint"],
                        pipeline_name=case["pipeline_name"],
                    )
                )
                passages = chunker.chunk(
                    document_id=case["document_id"],
                    canonical_text=case["canonical_text"],
                    selected_text=selected_text,
                    representation=case["representation"],
                    language=case["language"],
                )

                self.assertEqual(
                    [(p.start_char, p.end_char) for p in passages],
                    [tuple(span) for span in case["expected_spans"]],
                )
                self.assertEqual("".join(p.text for p in passages), selected_text)
                self.assertEqual(
                    [p.text for p in passages],
                    [selected_text[start:end] for start, end in case["expected_spans"]],
                )
                self.assertEqual(
                    passages[0].source_content_hash,
                    hashlib.sha256(case["canonical_text"].encode("utf-8")).digest(),
                )
                self.assertEqual(
                    passages[0].representation_hash,
                    hashlib.sha256(selected_text.encode("utf-8")).digest(),
                )
                self.assertEqual(
                    [p.content_hash for p in passages],
                    [hashlib.sha256(p.text.encode("utf-8")).digest() for p in passages],
                )
                validate_passages(selected_text, passages)

    def test_forced_paragraph_cases_preserve_blank_lines_and_never_use_rcts(self) -> None:
        cases = [
            case
            for case in load_cases()
            if case["name"]
            in {"forced_lf_paragraph_packing", "forced_crlf_paragraph_packing"}
        ]
        for case in cases:
            with self.subTest(case=case["name"]):
                chunker = LangChainPassageChunker(
                    ChunkerConfig(
                        chunk_size=case["chunk_size"],
                        length_function=char_length,
                        length_function_fingerprint=case["length_function_fingerprint"],
                        pipeline_name=case["pipeline_name"],
                    )
                )
                with patch.object(
                    _recursive_splitter_class(),
                    "create_documents",
                    side_effect=AssertionError("RCTS is exceptional only"),
                ):
                    passages = chunker.chunk(
                        document_id=case["document_id"],
                        canonical_text=case["canonical_text"],
                        selected_text=case["selected_text"],
                        representation=case["representation"],
                        language=case["language"],
                    )

                self.assertEqual(
                    [(p.start_char, p.end_char) for p in passages],
                    [tuple(span) for span in case["expected_spans"]],
                )
                self.assertEqual(
                    [p.text for p in passages],
                    [
                        case["selected_text"][start:end]
                        for start, end in case["expected_spans"]
                    ],
                )
                self.assertTrue(all(p.boundary == "paragraph_packed" for p in passages))
                self.assertEqual("".join(p.text for p in passages), case["selected_text"])

    def test_report_text_requires_english_language(self) -> None:
        chunker = LangChainPassageChunker(
            chunk_size=100,
            length_function=char_length,
            length_function_fingerprint="chars-v1",
        )

        with self.assertRaises(PassageChunkingError):
            chunker.chunk(
                document_id="doc-report-language",
                canonical_text="Texto fuente.",
                selected_text="English report text.",
                representation="report_text",
                language="spa",
            )

    def test_packed_sentences_keep_adjacent_whitespace_and_report_boundary(self) -> None:
        text = "First sentence. Second sentence. Third sentence."
        chunker = LangChainPassageChunker(
            ChunkerConfig(
                chunk_size=34,
                length_function=char_length,
                length_function_fingerprint="chars-v1",
            )
        )

        passages = chunker.chunk(
            document_id="doc-packed",
            canonical_text=text,
            selected_text=text,
            representation="canonical",
            language="eng",
        )

        self.assertEqual([p.text for p in passages], ["First sentence. Second sentence. ", "Third sentence."])
        self.assertEqual([p.boundary for p in passages], ["sentence_packed", "sentence_packed"])
        self.assertEqual(passages[0].next_passage_id, passages[1].passage_id)
        self.assertEqual(passages[1].previous_passage_id, passages[0].passage_id)
        self.assertEqual(passages[0].overlap_chars, 0)
        self.assertEqual(passages[1].overlap_tokens, 0)

    def test_recursive_fallback_runs_once_only_for_an_individually_oversized_sentence(self) -> None:
        text = "One two three four five six seven eight nine ten eleven."
        chunker = LangChainPassageChunker(
            ChunkerConfig(
                chunk_size=16,
                length_function=char_length,
                length_function_fingerprint="chars-v1",
            )
        )
        calls = 0
        recursive_splitter = _recursive_splitter_class()
        original = recursive_splitter.create_documents

        def counted_create_documents(splitter, texts, metadatas=None):
            nonlocal calls
            calls += 1
            return original(splitter, texts, metadatas)

        with patch.object(recursive_splitter, "create_documents", counted_create_documents):
            passages = chunker.chunk(
                document_id="doc-fallback",
                canonical_text=text,
                selected_text=text,
                representation="canonical",
                language="eng",
            )

        self.assertEqual(calls, 1)
        self.assertTrue(passages)
        self.assertTrue(all(p.boundary == "oversized_sentence_fallback" for p in passages))
        validate_passages(text, passages)

    def test_packed_multi_sentence_chunk_does_not_use_recursive_fallback(self) -> None:
        text = "One. Two. Three."
        chunker = LangChainPassageChunker(
            ChunkerConfig(
                chunk_size=100,
                length_function=char_length,
                length_function_fingerprint="chars-v1",
            )
        )

        with patch.object(
            _recursive_splitter_class(),
            "create_documents",
            side_effect=AssertionError("recursive fallback must not run"),
        ):
            passages = chunker.chunk(
                document_id="doc-no-fallback",
                canonical_text=text,
                selected_text=text,
                representation="canonical",
                language="eng",
            )

        self.assertEqual([p.text for p in passages], [text])

    def test_recursive_fallback_rejects_chunks_that_still_exceed_the_ceiling(self) -> None:
        text = "Impossible."
        chunker = LangChainPassageChunker(
            ChunkerConfig(
                chunk_size=3,
                length_function=lambda value: len(value) * 4,
                length_function_fingerprint="four-times-chars-v1",
            )
        )

        with self.assertRaises(PassageChunkingError):
            chunker.chunk(
                document_id="doc-overflow",
                canonical_text=text,
                selected_text=text,
                representation="canonical",
                language="eng",
            )

    def test_passage_ids_are_stable_across_subprocesses(self) -> None:
        text = "Stable sentence."
        code = """
from trec_ragtime.passages import ChunkerConfig, LangChainPassageChunker
chunker = LangChainPassageChunker(ChunkerConfig(chunk_size=100, length_function=len))
print(chunker.chunk(document_id='stable', canonical_text='Stable sentence.', selected_text='Stable sentence.', representation='canonical', language='eng')[0].passage_id.hex())
"""
        first = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
        second = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
        self.assertEqual(first, second)


def _make_passages(text: str) -> tuple[Passage, ...]:
    chunker = LangChainPassageChunker(ChunkerConfig(chunk_size=100, length_function=len))
    return chunker.chunk(
        document_id="validation",
        canonical_text=text,
        selected_text=text,
        representation="canonical",
        language="eng",
    )


def _packed_passages() -> tuple[str, tuple[Passage, ...]]:
    text = "First sentence. Second sentence. Third sentence."
    chunker = LangChainPassageChunker(
        ChunkerConfig(
            chunk_size=34,
            length_function=char_length,
            length_function_fingerprint="chars-v1",
        )
    )
    return text, chunker.chunk(
        document_id="validation-packed",
        canonical_text=text,
        selected_text=text,
        representation="canonical",
        language="eng",
    )


def _lineage_id(
    passage: Passage,
    *,
    start_char: int | None = None,
    end_char: int | None = None,
    chunk_index: int | None = None,
) -> bytes:
    return sha256_bytes(
        msgspec.msgpack.encode(
            (
                passage.document_id,
                passage.representation,
                passage.representation_hash,
                passage.start_char if start_char is None else start_char,
                passage.end_char if end_char is None else end_char,
                passage.chunk_index if chunk_index is None else chunk_index,
                passage.chunker_fingerprint,
            )
        )
    )


def _recursive_splitter_class():
    loader = getattr(passages_module, "_load_splitters", None)
    if loader is not None:
        loader()
    return passages_module.RecursiveCharacterTextSplitter


if __name__ == "__main__":
    unittest.main()
