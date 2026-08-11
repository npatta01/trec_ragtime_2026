#!/usr/bin/env python3
"""Benchmark and validate the maintained exact passage chunking pipeline."""

from __future__ import annotations

import importlib.metadata
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

from trec_ragtime.passages import (
    Passage,
    PassageChunker,
    sha256_bytes,
    validate_passages,
)
from trec_ragtime.passage_scoring import (
    CrossEncoderPairLengthFunction,
    LocalCrossEncoderConfig,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "passage_chunking.json"
PACKAGE_NAMES = (
    "langchain-text-splitters",
    "spacy",
    "sentence-transformers",
    "msgspec",
)
PIPELINES_BY_LANGUAGE = {
    "eng": "en_core_web_sm",
    "spa": "es_core_news_sm",
    "rus": "ru_core_news_sm",
    "zho": "zh_core_web_sm",
}


class FixturePairLength:
    """Offline pair-length seam matching the fixture contract."""

    passage_length_function = staticmethod(lambda text: len(text.split()))

    def __init__(self, case: dict[str, Any]) -> None:
        self.fingerprint = f"benchmark-pair-{case['name']}"
        self.max_length = int(case["max_length"])
        self.max_passage_length = int(case["max_passage_length"])

    def __call__(self, text: str) -> int:
        return self.passage_length_function(text) + 2


class _OfflinePinnedTokenizer:
    """Deterministic pinned-tokenizer shape used without model weights."""

    QUERY = "shared evidence and practical outcomes"

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    def __call__(self, *texts: str, **kwargs: Any) -> dict[str, object]:
        self.calls.append((texts, dict(kwargs)))
        if len(texts) == 2:
            query, passage = texts
            if query != self.QUERY:
                raise AssertionError(f"unexpected pair query: {query!r}")
            passage_count = 0 if not passage else len(passage) + 1
            count = 10 + passage_count
        elif len(texts) == 1:
            (text,) = texts
            if kwargs.get("add_special_tokens") is not False:
                raise AssertionError("single-sequence counting must disable specials")
            if text == self.QUERY:
                count = 6
            elif not text:
                count = 0
            elif set(text) == {"中"}:
                count = len(text) + 1
            else:
                raise AssertionError(f"unexpected single sequence: {text!r}")
        else:
            raise AssertionError(f"unexpected tokenizer arity: {len(texts)}")
        return {"input_ids": list(range(count)), "length": [count]}


class MultiSentencePairLength:
    """Structural pair seam where a packed multi-sentence span overflows."""

    fingerprint = "benchmark-multi-sentence-guard-v1"
    passage_length_function = staticmethod(lambda text: len(text.split()))
    max_length = 5
    max_passage_length = 3

    def __call__(self, text: str) -> int:
        if text.count(".") >= 2:
            return 6
        return self.passage_length_function(text) + 1


def package_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in PACKAGE_NAMES}


def required_model_versions(cases: list[dict[str, Any]]) -> dict[str, str]:
    languages = {str(case["language"]) for case in cases}
    expected_languages = set(PIPELINES_BY_LANGUAGE)
    if languages != expected_languages:
        raise AssertionError(
            "fixture must cover exactly the four pinned languages: "
            f"{sorted(expected_languages)!r}"
        )
    for case in cases:
        expected_pipeline = PIPELINES_BY_LANGUAGE[str(case["language"])]
        if case.get("pipeline_name") != expected_pipeline:
            raise AssertionError(
                f"{case['name']}: fixture pipeline does not match its language"
            )

    versions: dict[str, str] = {}
    for pipeline_name in sorted({str(case["pipeline_name"]) for case in cases}):
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


def assert_benchmark_invariants(
    selected_text: str,
    passages: tuple[Passage, ...],
    pair_length: Callable[[str], int],
    max_length: int,
) -> None:
    validate_passages(selected_text, passages)
    covered: set[int] = set()
    for passage in passages:
        assert selected_text[passage.start_char : passage.end_char] == passage.text
        assert passage.content_hash == sha256_bytes(passage.text)
        assert pair_length(passage.text) <= max_length
        covered.update(range(passage.start_char, passage.end_char))
    assert covered == set(range(len(selected_text)))


def record_passages(passages: tuple[Passage, ...]) -> list[dict[str, Any]]:
    return [
        {
            "boundary": passage.boundary,
            "content_hash": passage.content_hash.hex(),
            "document_id": passage.document_id,
            "end_char": passage.end_char,
            "language": passage.language,
            "representation": passage.representation,
            "start_char": passage.start_char,
            "text": passage.text,
        }
        for passage in passages
    ]


def assert_fixture_expectations(
    case: dict[str, Any], passages: tuple[Passage, ...]
) -> None:
    actual_spans = [(passage.start_char, passage.end_char) for passage in passages]
    expected_spans = [tuple(span) for span in case["expected_spans"]]
    if actual_spans != expected_spans:
        raise AssertionError(f"{case['name']}: literal fixture span mismatch")

    actual_boundaries = [passage.boundary for passage in passages]
    if actual_boundaries != case["expected_boundaries"]:
        raise AssertionError(f"{case['name']}: literal fixture boundary mismatch")

    if case["name"] == "overlap":
        expected_overlaps = [tuple(span) for span in case["sentence_spans"]]
        actual_overlaps: list[tuple[int, int]] = []
        selected_text = str(case["selected_text"])
        for first, second in zip(passages, passages[1:]):
            start = max(first.start_char, second.start_char)
            end = min(first.end_char, second.end_char)
            if start >= end:
                raise AssertionError("overlap fixture did not produce an overlap")
            overlap = (start, end)
            actual_overlaps.append(overlap)
            if overlap not in expected_overlaps:
                raise AssertionError(
                    f"overlap fixture intersects at non-sentence span {overlap!r}"
                )
            sentence_index = expected_overlaps.index(overlap)
            if selected_text[start:end] != case["sentences"][sentence_index]:
                raise AssertionError("overlap fixture changed a sentence unit")
        if actual_overlaps != expected_overlaps:
            raise AssertionError(
                f"overlap fixture mismatch: {actual_overlaps!r} != {expected_overlaps!r}"
            )


def run_once(cases: list[dict[str, Any]]) -> tuple[list[bytes], int, float]:
    records: list[bytes] = []
    passage_count = 0
    started = time.perf_counter()
    chunkers: dict[tuple[str, str, str, int, int, int], PassageChunker] = {}
    for case in cases:
        pair_length = FixturePairLength(case)
        key = (
            str(case["language"]),
            str(case["pipeline_name"]),
            pair_length.fingerprint,
            int(case["max_length"]),
            int(case["max_passage_length"]),
            int(case["overlap_tokens"]),
        )
        chunker = chunkers.setdefault(
            key,
            PassageChunker(
                pair_length,
                overlap_tokens=int(case["overlap_tokens"]),
            ),
        )
        selected_text = str(case["selected_text"])
        passages = chunker.chunk(
            document_id=str(case["document_id"]),
            canonical_text=str(case["canonical_text"]),
            selected_text=selected_text,
            representation=str(case["representation"]),  # type: ignore[arg-type]
            language=str(case["language"]),
        )
        assert_benchmark_invariants(
            selected_text,
            passages,
            pair_length,
            int(case["max_length"]),
        )
        assert_fixture_expectations(case, passages)
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


def verify_multi_sentence_units() -> tuple[tuple[int, int, str], ...]:
    text = "One. Two."
    pair_length = MultiSentencePairLength()
    individual_sentences = ("One.", "Two.")
    if any(
        pair_length(sentence) > pair_length.max_length
        for sentence in individual_sentences
    ):
        raise AssertionError("guard fixture sentence does not fit its pair ceiling")
    if pair_length(text) <= pair_length.max_length:
        raise AssertionError("guard fixture does not exceed its pair ceiling")

    chunker = PassageChunker(pair_length, overlap_tokens=0)
    with patch.object(
        PassageChunker,
        "_recursive_splitter",
        side_effect=AssertionError(
            "RCTS must not receive a multi-sentence over-ceiling span"
        ),
    ):
        passages = chunker.chunk(
            document_id="benchmark-multi-sentence-units",
            canonical_text=text,
            selected_text=text,
            representation="canonical",
            language="eng",
        )

    assert_benchmark_invariants(text, passages, pair_length, pair_length.max_length)
    actual = tuple(
        (passage.start_char, passage.end_char, passage.text) for passage in passages
    )
    expected = ((0, 5, "One. "), (5, 9, "Two."))
    if actual != expected:
        raise AssertionError(
            f"multi-sentence repacking changed source spans: {actual!r}"
        )
    if any(passage.boundary != "sentence" for passage in passages):
        raise AssertionError(
            "multi-sentence repacking must preserve sentence boundaries"
        )
    if "".join(passage.text for passage in passages) != text:
        raise AssertionError("multi-sentence repacking left source text uncovered")
    return actual


class _FakeDocument:
    def __init__(self, text: str, start_char: int) -> None:
        self.page_content = text
        self.metadata = {"start_index": start_char}


class _FakeSentence:
    def __init__(self, text: str) -> None:
        self.start_char = 0
        self.text_with_ws = text


class _FakeTokenizerDocument:
    def __init__(self, text: str) -> None:
        self.sents = (_FakeSentence(text),)


class _PinnedTokenizerSplitter:
    pipelines: list[str] = []

    def __init__(self, *, pipeline: str, **_: Any) -> None:
        self.pipelines.append(pipeline)

    def _tokenizer(self, text: str) -> _FakeTokenizerDocument:
        return _FakeTokenizerDocument(text)

    def create_documents(self, texts: list[str]) -> list[_FakeDocument]:
        return [_FakeDocument(texts[0], 0)]


class _FakeRecursiveSplitter:
    def __init__(self, *, chunk_size: int, **_: Any) -> None:
        self.chunk_size = chunk_size

    def create_documents(self, texts: list[str]) -> list[_FakeDocument]:
        text = texts[0]
        return [
            _FakeDocument(text[start : start + self.chunk_size], start)
            for start in range(0, len(text), self.chunk_size)
        ]


def verify_pair_aware_case() -> tuple[tuple[int, int, str], ...]:
    import spacy
    import trec_ragtime.passages as passages_module

    text = "中" * 502
    tokenizer = _OfflinePinnedTokenizer()
    tokenizer_loader_calls = 0

    def tokenizer_loader(config: LocalCrossEncoderConfig) -> _OfflinePinnedTokenizer:
        nonlocal tokenizer_loader_calls
        if config.max_length != 512:
            raise AssertionError("pair-aware fixture changed its pinned max length")
        tokenizer_loader_calls += 1
        return tokenizer

    pair_length = CrossEncoderPairLengthFunction(
        LocalCrossEncoderConfig(max_length=512),
        _OfflinePinnedTokenizer.QUERY,
        tokenizer_loader=tokenizer_loader,
    )
    if (
        len(text) > pair_length.max_length
        or pair_length(text) <= pair_length.max_length
    ):
        raise AssertionError(
            "pair-aware fixture must fit by characters but exceed by pairs"
        )

    _PinnedTokenizerSplitter.pipelines = []
    model_loader_calls: list[str] = []
    splitter_loader_calls = 0

    def fake_splitter_loader() -> tuple[type[Any], type[Any]]:
        nonlocal splitter_loader_calls
        splitter_loader_calls += 1
        return _PinnedTokenizerSplitter, _FakeRecursiveSplitter

    with (
        patch.object(
            passages_module,
            "_load_splitters",
            side_effect=fake_splitter_loader,
        ),
        patch.object(
            spacy,
            "load",
            side_effect=lambda pipeline: model_loader_calls.append(pipeline),
        ),
    ):
        chunker = PassageChunker(pair_length, overlap_tokens=0)
        passages = chunker.chunk(
            document_id="benchmark-pair-aware-cjk",
            canonical_text=text,
            selected_text=text,
            representation="canonical",
            language="zho",
        )

    assert _PinnedTokenizerSplitter.pipelines == ["zh_core_web_sm"]
    assert splitter_loader_calls == 3
    assert model_loader_calls == []
    assert tokenizer_loader_calls == 1
    assert_benchmark_invariants(text, passages, pair_length, pair_length.max_length)
    assert all(passage.boundary == "oversized_sentence" for passage in passages)
    records = tuple(
        (passage.start_char, passage.end_char, passage.text) for passage in passages
    )
    expected = ((0, 501, text[:501]), (501, 502, text[501:]))
    assert records == expected, f"pair-aware fixture under-packed: {records!r}"
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
    first_multi_sentence_record = verify_multi_sentence_units()
    second_multi_sentence_record = verify_multi_sentence_units()
    if first_multi_sentence_record != second_multi_sentence_record:
        raise AssertionError("multi-sentence units produced different passage records")

    oversized_cases = [case for case in cases if case["name"] == "oversized_sentence"]
    if len(oversized_cases) != 1:
        raise AssertionError("fixture must contain one oversized_sentence case")

    packages = package_versions()
    print("benchmark_scope=exact_fixture_chunking_plus_offline_pair_ceiling")
    print(
        "selected_library_versions="
        + json.dumps(
            {name: packages[name] for name in ("langchain-text-splitters", "spacy")},
            sort_keys=True,
        )
    )
    print("pair_length_seam=CrossEncoderPairLengthFunction")
    print("pair_length_tokenizer=offline_pinned_tokenizer_shape")
    print("package_versions=" + json.dumps(packages, sort_keys=True))
    print("spacy_model_versions=" + json.dumps(models, sort_keys=True))
    print(f"fixture_cases={len(cases)}")
    print(f"passes={len(cases) * 2}")
    print("failures=0")
    print("full_source_coverage=true")
    print("sentence_only_overlap=true")
    print("multi_sentence_units_without_rcts=true")
    print("pair_ceiling=true")
    print("deterministic_records=true")
    print("pair_aware_model_loader_calls=0")
    print("pair_aware_tokenizer_loader_calls=1")
    print(f"startup_seconds={startup_seconds:.6f}")
    print(f"run1_seconds={first_seconds:.6f}")
    print(f"run2_seconds={second_seconds:.6f}")
    print(f"documents_per_second={len(cases) / first_seconds:.3f}")
    print(f"passages={first_passages}")
    print(f"passages_per_second={first_passages / first_seconds:.3f}")
    print("pair_aware_pinned_tokenizer_ceiling=true")
    print(f"pair_aware_passages={len(first_pair_record)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
