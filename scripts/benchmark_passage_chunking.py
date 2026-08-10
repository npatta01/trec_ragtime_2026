#!/usr/bin/env python3
"""Benchmark and validate the maintained exact passage chunking pipeline."""

from __future__ import annotations

import importlib.metadata
import json
import time
import warnings
from pathlib import Path
from typing import Any

from langchain_text_splitters.character import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_text_splitters.spacy import SpacyTextSplitter

from trec_ragtime.passages import (
    ChunkerConfig,
    LangChainPassageChunker,
    Passage,
    validate_passages,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "passage_chunking.json"
PACKAGE_NAMES = (
    "langchain-text-splitters",
    "spacy",
    "sentence-transformers",
    "msgspec",
)


class PairAwareCjkLength:
    fingerprint = "benchmark-fake-cross-encoder-pair-v1"
    passage_length_function = len
    max_passage_length = 501

    def __call__(self, text: str) -> int:
        return len(text) + 11


def package_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in PACKAGE_NAMES}


def required_model_versions(cases: list[dict[str, Any]]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for pipeline_name in sorted({case["pipeline_name"] for case in cases}):
        distribution_name = pipeline_name.replace("_", "-")
        try:
            version = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(
                f"required spaCy model distribution {distribution_name!r} is unavailable; "
                "install the locked project dependencies before benchmarking"
            ) from error
        versions[pipeline_name] = str(version)
    return versions


def _splitter_documents(splitter: Any, parent_text: str, stage: str) -> list[Any]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"\[W108\].*")
        documents = splitter.create_documents([parent_text])
    cursor = 0
    for document in documents:
        content = document.page_content
        start = document.metadata.get("start_index")
        if not isinstance(start, int) or isinstance(start, bool) or start != cursor:
            raise AssertionError(f"{stage}: invalid start_index {start!r}")
        end = start + len(content)
        if parent_text[start:end] != content:
            raise AssertionError(f"{stage}: non-source slice")
        cursor = end
    if (
        cursor != len(parent_text)
        or "".join(d.page_content for d in documents) != parent_text
    ):
        raise AssertionError(f"{stage}: incomplete reconstruction")
    return documents


def verify_splitter_levels(case: dict[str, Any]) -> None:
    if case.get("length_function_mode") != "boundary-only":
        raise AssertionError(
            f"{case['name']}: fixtures must explicitly declare boundary-only length"
        )
    selected_text = case["selected_text"]
    chunk_size = case["chunk_size"]
    paragraph_splitter = CharacterTextSplitter(
        separator=r"(?:\r?\n){2,}",
        is_separator_regex=True,
        keep_separator="end",
        strip_whitespace=False,
        chunk_overlap=0,
        add_start_index=True,
        chunk_size=chunk_size,
        length_function=len,
    )
    paragraph_documents = _splitter_documents(
        paragraph_splitter, selected_text, "paragraph"
    )
    sentence_splitter = SpacyTextSplitter(
        separator="",
        pipeline=case["pipeline_name"],
        strip_whitespace=False,
        chunk_overlap=0,
        add_start_index=True,
        chunk_size=chunk_size,
        length_function=len,
    )
    for paragraph in paragraph_documents:
        if len(paragraph.page_content) <= chunk_size:
            continue
        sentence_documents = _splitter_documents(
            sentence_splitter, paragraph.page_content, "sentence"
        )
        recursive_splitter = RecursiveCharacterTextSplitter(
            separators=[" ", ""],
            keep_separator="end",
            strip_whitespace=False,
            chunk_overlap=0,
            add_start_index=True,
            chunk_size=chunk_size,
            length_function=len,
        )
        for sentence in sentence_documents:
            if len(sentence.page_content) > chunk_size:
                _splitter_documents(
                    recursive_splitter, sentence.page_content, "recursive sentence"
                )


def record_passages(passages: tuple[Passage, ...]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_index": passage.chunk_index,
            "content_hash": passage.content_hash.hex(),
            "end_char": passage.end_char,
            "next_passage_id": (
                None
                if passage.next_passage_id is None
                else passage.next_passage_id.hex()
            ),
            "passage_id": passage.passage_id.hex(),
            "previous_passage_id": (
                None
                if passage.previous_passage_id is None
                else passage.previous_passage_id.hex()
            ),
            "representation_hash": passage.representation_hash.hex(),
            "source_content_hash": passage.source_content_hash.hex(),
            "start_char": passage.start_char,
            "text": passage.text,
            "boundary": passage.boundary,
        }
        for passage in passages
    ]


def run_once(cases: list[dict[str, Any]]) -> tuple[list[bytes], int, float]:
    records: list[bytes] = []
    passage_count = 0
    started = time.perf_counter()
    chunkers: dict[tuple[int, str], LangChainPassageChunker] = {}
    for case in cases:
        key = (case["chunk_size"], case["pipeline_name"])
        chunker = chunkers.setdefault(
            key,
            LangChainPassageChunker(
                ChunkerConfig(
                    chunk_size=case["chunk_size"],
                    length_function=len,
                    pipeline_name=case["pipeline_name"],
                )
            ),
        )
        verify_splitter_levels(case)
        passages = chunker.chunk(
            document_id=case["document_id"],
            canonical_text=case["canonical_text"],
            selected_text=case["selected_text"],
            representation=case["representation"],
            language=case["language"],
        )
        validate_passages(case["selected_text"], passages)
        if [(p.start_char, p.end_char) for p in passages] != [
            tuple(span) for span in case["expected_spans"]
        ]:
            raise AssertionError(f"{case['name']}: literal fixture span mismatch")
        if any(len(p.text) > case["chunk_size"] for p in passages):
            raise AssertionError(f"{case['name']}: configured ceiling exceeded")
        records.append(
            json.dumps(
                {
                    "case": case["name"],
                    "passages": record_passages(passages),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        passage_count += len(passages)
    return records, passage_count, time.perf_counter() - started


def verify_pair_aware_case() -> tuple[tuple[int, int, str], ...]:
    text = "中" * 502
    length_function = PairAwareCjkLength()
    chunk_size = 512
    if len(text) > chunk_size or length_function(text) <= chunk_size:
        raise AssertionError(
            "pair-aware fixture must fit by characters but exceed by pairs"
        )
    chunker = LangChainPassageChunker(
        ChunkerConfig(
            chunk_size=chunk_size,
            length_function=length_function,
            pipeline_name="zh_core_web_sm",
        )
    )
    passages = chunker.chunk(
        document_id="benchmark-pair-aware-cjk",
        canonical_text=text,
        selected_text=text,
        representation="canonical",
        language="zho",
    )
    validate_passages(text, passages)
    if not all(length_function(passage.text) <= chunk_size for passage in passages):
        raise AssertionError("pair-aware fixture exceeded its configured ceiling")
    if not all(
        passage.boundary == "oversized_sentence_fallback" for passage in passages
    ):
        raise AssertionError("pair-aware fixture did not use sentence fallback")
    records = tuple(
        (passage.start_char, passage.end_char, passage.text) for passage in passages
    )
    expected = ((0, 501, text[:501]), (501, 502, text[501:]))
    if records != expected:
        raise AssertionError(f"pair-aware fixture under-packed: {records!r}")
    return records


def main() -> int:
    cases = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["cases"]
    startup_started = time.perf_counter()
    models = required_model_versions(cases)
    startup_seconds = time.perf_counter() - startup_started
    first_records, first_passages, first_seconds = run_once(cases)
    second_records, second_passages, second_seconds = run_once(cases)
    if first_records != second_records or first_passages != second_passages:
        raise AssertionError("two benchmark runs produced different passage records")
    first_pair_record = verify_pair_aware_case()
    second_pair_record = verify_pair_aware_case()
    if first_pair_record != second_pair_record:
        raise AssertionError(
            "pair-aware benchmark runs produced different passage records"
        )

    print("package_versions=" + json.dumps(package_versions(), sort_keys=True))
    print("spacy_model_versions=" + json.dumps(models, sort_keys=True))
    print(f"fixture_cases={len(cases)}")
    print(
        "fixture_length_function=explicit builtin len (boundary-only; not model-safe)"
    )
    print(f"passes={len(cases) * 2}")
    print("failures=0")
    print(f"startup_seconds={startup_seconds:.6f}")
    print(f"run1_seconds={first_seconds:.6f}")
    print(f"run2_seconds={second_seconds:.6f}")
    print(f"documents_per_second={len(cases) / first_seconds:.3f}")
    print(f"passages={first_passages}")
    print(f"passages_per_second={first_passages / first_seconds:.3f}")
    print("pair_aware_fake_ceiling=true")
    print(f"pair_aware_passages={len(first_pair_record)}")
    print("deterministic_records=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
