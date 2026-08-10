from __future__ import annotations

import hashlib
import importlib.metadata
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


class FingerprintedLength:
    fingerprint = "fake-pair-v1"

    def __call__(self, text: str) -> int:
        return len(text)


class PairAwareCjkLength:
    fingerprint = "fake-cross-encoder-pair-v1"
    passage_length_function = len
    max_passage_length = 8

    def __call__(self, text: str) -> int:
        return len(text) + 8


class PairAwareContiguousCjkLength:
    fingerprint = "fake-cross-encoder-contiguous-cjk-v1"
    passage_length_function = len
    max_passage_length = 501

    def __call__(self, text: str) -> int:
        return len(text) + 11


class PairAwareBudgetLength:
    fingerprint = "fake-cross-encoder-budget-v1"
    passage_length_function = len

    def __init__(self, max_passage_length: int) -> None:
        self.max_passage_length = max_passage_length

    def __call__(self, text: str) -> int:
        return len(text) + 11


RUNTIME_DISTRIBUTION_VERSIONS = {
    "langchain-text-splitters": "1.1.2",
    "spacy": "3.8.15",
    "en-core-web-sm": "3.8.0",
    "es-core-news-sm": "3.8.0",
    "ru-core-news-sm": "3.8.0",
    "zh-core-web-sm": "3.8.0",
}


def fake_distribution_version(overrides: dict[str, str] | None = None):
    versions = {**RUNTIME_DISTRIBUTION_VERSIONS, **(overrides or {})}

    def version(name: str) -> str:
        return versions[name]

    return version


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
            bytes.fromhex(
                "850f7dc43910ff890f8879c0ed26fe697c93a067ad93a7d50f466a7028a9bf4e"
            ),
        )

    def test_zero_chunk_size_reaches_strict_config_validation(self) -> None:
        with self.assertRaises(PassageValidationError):
            LangChainPassageChunker(chunk_size=0)

    def test_chunker_requires_an_explicit_length_function(self) -> None:
        with self.assertRaisesRegex(
            PassageValidationError, "length_function is required"
        ):
            ChunkerConfig()
        with self.assertRaisesRegex(
            PassageValidationError, "length_function is required"
        ):
            LangChainPassageChunker()

    def test_explicit_builtin_len_is_allowed_for_boundary_chunking(self) -> None:
        config = ChunkerConfig(length_function=len)

        self.assertEqual(config.resolved_length_function_fingerprint, "builtin:len")

    def test_callable_fingerprint_is_used_when_explicit_identity_is_omitted(
        self,
    ) -> None:
        config = ChunkerConfig(length_function=FingerprintedLength())

        self.assertEqual(
            config.resolved_length_function_fingerprint,
            "fake-pair-v1",
        )

    def test_chunker_fingerprint_includes_additive_passage_budget(self) -> None:
        first = ChunkerConfig(
            chunk_size=512,
            length_function=PairAwareBudgetLength(501),
        )
        second = ChunkerConfig(
            chunk_size=512,
            length_function=PairAwareBudgetLength(500),
        )

        self.assertNotEqual(first.resolved_fingerprint, second.resolved_fingerprint)

    def test_custom_length_function_requires_a_stable_identity(self) -> None:
        with self.assertRaises(PassageValidationError):
            ChunkerConfig(length_function=char_length)
        with self.assertRaises(PassageValidationError):
            LangChainPassageChunker(length_function=char_length)

    def test_chunker_fingerprint_changes_for_result_affecting_identity_inputs(
        self,
    ) -> None:
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

    def test_chunker_fingerprint_composes_caller_namespace_with_runtime_identity(
        self,
    ) -> None:
        base = ChunkerConfig(length_function=len)
        namespaced = ChunkerConfig(length_function=len, fingerprint="caller-v1")

        self.assertNotEqual(base.resolved_fingerprint, namespaced.resolved_fingerprint)
        self.assertIn(base.resolved_fingerprint, namespaced.resolved_fingerprint)
        self.assertIn("caller-v1", namespaced.resolved_fingerprint)
        self.assertNotEqual(
            namespaced.resolved_fingerprint,
            replace(
                namespaced, chunk_size=namespaced.chunk_size + 1
            ).resolved_fingerprint,
        )

    def test_chunker_fingerprint_rotates_for_every_selected_runtime_distribution(
        self,
    ) -> None:
        distributions = tuple(RUNTIME_DISTRIBUTION_VERSIONS)
        with patch.object(
            importlib.metadata,
            "version",
            side_effect=fake_distribution_version(),
        ):
            baseline = ChunkerConfig(length_function=len)

        for distribution in distributions:
            with self.subTest(distribution=distribution):
                with patch.object(
                    importlib.metadata,
                    "version",
                    side_effect=fake_distribution_version(
                        {
                            distribution: f"{RUNTIME_DISTRIBUTION_VERSIONS[distribution]}+changed"
                        }
                    ),
                ):
                    changed = ChunkerConfig(length_function=len)
                self.assertNotEqual(
                    baseline.resolved_fingerprint, changed.resolved_fingerprint
                )

    def test_custom_pipeline_identity_uses_its_distribution_version(self) -> None:
        calls: list[str] = []

        def version(name: str) -> str:
            calls.append(name)
            return "test-version"

        with patch.object(importlib.metadata, "version", side_effect=version):
            ChunkerConfig(length_function=len, pipeline_name="custom_pipeline")

        self.assertEqual(
            calls,
            ["langchain-text-splitters", "spacy", "custom-pipeline"],
        )

    def test_missing_runtime_distribution_identity_fails_during_config_creation(
        self,
    ) -> None:
        def missing_spacy(name: str) -> str:
            if name == "spacy":
                raise importlib.metadata.PackageNotFoundError(name)
            return RUNTIME_DISTRIBUTION_VERSIONS[name]

        with patch.object(importlib.metadata, "version", side_effect=missing_spacy):
            with self.assertRaisesRegex(PassageValidationError, "spacy"):
                ChunkerConfig(length_function=len, fingerprint="caller-v1")

    def test_validate_passages_rejects_whitespace_stripping_at_reconstruction(
        self,
    ) -> None:
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
                        length_function=len,
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

    def test_forced_paragraph_cases_preserve_blank_lines_and_never_use_rcts(
        self,
    ) -> None:
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
                        length_function=len,
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
                self.assertEqual(
                    "".join(p.text for p in passages), case["selected_text"]
                )

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

    def test_packed_sentences_keep_adjacent_whitespace_and_report_boundary(
        self,
    ) -> None:
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

        self.assertEqual(
            [p.text for p in passages],
            ["First sentence. Second sentence. ", "Third sentence."],
        )
        self.assertEqual(
            [p.boundary for p in passages], ["sentence_packed", "sentence_packed"]
        )
        self.assertEqual(passages[0].next_passage_id, passages[1].passage_id)
        self.assertEqual(passages[1].previous_passage_id, passages[0].passage_id)
        self.assertEqual(passages[0].overlap_chars, 0)
        self.assertEqual(passages[1].overlap_tokens, 0)

    def test_recursive_fallback_runs_once_only_for_an_individually_oversized_sentence(
        self,
    ) -> None:
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

        with patch.object(
            recursive_splitter, "create_documents", counted_create_documents
        ):
            passages = chunker.chunk(
                document_id="doc-fallback",
                canonical_text=text,
                selected_text=text,
                representation="canonical",
                language="eng",
            )

        self.assertEqual(calls, 1)
        self.assertTrue(passages)
        self.assertTrue(
            all(p.boundary == "oversized_sentence_fallback" for p in passages)
        )
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

    def test_recursive_fallback_rejects_chunks_that_still_exceed_the_ceiling(
        self,
    ) -> None:
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

    def test_pair_aware_cjk_length_forces_sentence_fallback_under_its_ceiling(
        self,
    ) -> None:
        text = "第一二三四五六七八九十。"
        length_function = PairAwareCjkLength()
        self.assertLessEqual(len(text), 16)
        self.assertGreater(length_function(text), 16)
        chunker = LangChainPassageChunker(
            ChunkerConfig(chunk_size=16, length_function=length_function)
        )

        passages = chunker.chunk(
            document_id="doc-pair-aware-cjk",
            canonical_text=text,
            selected_text=text,
            representation="canonical",
            language="zho",
        )

        self.assertTrue(passages)
        self.assertTrue(
            all(length_function(passage.text) <= 16 for passage in passages)
        )
        self.assertTrue(
            all(
                passage.boundary == "oversized_sentence_fallback"
                for passage in passages
            )
        )
        self.assertEqual("".join(passage.text for passage in passages), text)
        validate_passages(text, passages)

    def test_contiguous_cjk_uses_additive_passage_budget_without_under_packing(
        self,
    ) -> None:
        text = "中" * 502
        length_function = PairAwareContiguousCjkLength()
        self.assertEqual(length_function(text), 513)
        chunker = LangChainPassageChunker(
            ChunkerConfig(chunk_size=512, length_function=length_function)
        )

        passages = chunker.chunk(
            document_id="doc-contiguous-cjk",
            canonical_text=text,
            selected_text=text,
            representation="canonical",
            language="zho",
        )
        repeated = chunker.chunk(
            document_id="doc-contiguous-cjk",
            canonical_text=text,
            selected_text=text,
            representation="canonical",
            language="zho",
        )

        self.assertEqual(len(passages), 2)
        self.assertEqual(
            [(passage.start_char, passage.end_char) for passage in passages],
            [(0, 501), (501, 502)],
        )
        self.assertEqual("".join(passage.text for passage in passages), text)
        self.assertTrue(
            all(length_function(passage.text) <= 512 for passage in passages)
        )
        self.assertEqual(
            [passage.passage_id for passage in passages],
            [passage.passage_id for passage in repeated],
        )
        validate_passages(text, passages)

    def test_passage_ids_are_stable_across_subprocesses(self) -> None:
        code = """
from trec_ragtime.passages import ChunkerConfig, LangChainPassageChunker
chunker = LangChainPassageChunker(ChunkerConfig(chunk_size=100, length_function=len))
print(chunker.chunk(document_id='stable', canonical_text='Stable sentence.', selected_text='Stable sentence.', representation='canonical', language='eng')[0].passage_id.hex())
"""
        first = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
        second = subprocess.check_output(
            [sys.executable, "-c", code], text=True
        ).strip()
        self.assertEqual(first, second)


def _make_passages(text: str) -> tuple[Passage, ...]:
    chunker = LangChainPassageChunker(
        ChunkerConfig(chunk_size=100, length_function=len)
    )
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
