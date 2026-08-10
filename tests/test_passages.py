from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

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

    def test_validate_passages_rejects_whitespace_stripped_text_and_offsets(self) -> None:
        passages = _make_passages(" Alpha\nBeta ")
        stripped = passages[0].text.strip()
        mutated = replace(
            passages[0],
            text=stripped,
            start_char=1,
            end_char=1 + len(stripped),
            content_hash=sha256_bytes(stripped),
        )

        with self.assertRaises(PassageValidationError):
            validate_passages("Alpha\nBeta", (mutated,))

    def test_validate_passages_rejects_reordered_or_overlapping_spans(self) -> None:
        passages = _make_passages("Alpha Beta")
        first = passages[0]
        second = Passage(
            document_id=first.document_id,
            source_content_hash=first.source_content_hash,
            representation=first.representation,
            representation_hash=first.representation_hash,
            language=first.language,
            chunk_index=1,
            passage_id=b"2" * 32,
            text="Beta",
            start_char=6,
            end_char=10,
            boundary="sentence_packed",
            previous_passage_id=first.passage_id,
            next_passage_id=None,
            content_hash=sha256_bytes("Beta"),
            chunker_fingerprint=first.chunker_fingerprint,
            overlap_chars=0,
            overlap_tokens=0,
        )
        with self.assertRaises(PassageValidationError):
            validate_passages("Alpha Beta", (second, first))

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

    def test_packed_sentences_keep_adjacent_whitespace_and_report_boundary(self) -> None:
        text = "First sentence. Second sentence. Third sentence."
        chunker = LangChainPassageChunker(
            ChunkerConfig(chunk_size=34, length_function=char_length)
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
            ChunkerConfig(chunk_size=16, length_function=char_length)
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
            ChunkerConfig(chunk_size=100, length_function=char_length)
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


def _recursive_splitter_class():
    loader = getattr(passages_module, "_load_splitters", None)
    if loader is not None:
        loader()
    return passages_module.RecursiveCharacterTextSplitter


if __name__ == "__main__":
    unittest.main()
