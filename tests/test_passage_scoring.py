from __future__ import annotations

import math
import importlib.metadata
import subprocess
import sys
import types
import unittest
from dataclasses import FrozenInstanceError, is_dataclass, replace
from typing import Any
from unittest.mock import patch

from trec_ragtime.passages import Passage, sha256_bytes
from trec_ragtime.passage_scoring import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    CrossEncoderPairLengthFunction,
    LocalCrossEncoderConfig,
    LocalCrossEncoderScorer,
    PassageScoringError,
    validate_passage_scores,
)


def _passage(
    text: str,
    *,
    document_id: str = "doc",
    representation: str = "canonical",
    language: str = "eng",
    start_char: int = 0,
    end_char: int | None = None,
    boundary: str = "sentence",
) -> Passage:
    if end_char is None:
        end_char = start_char + len(text)
    return Passage(
        document_id=document_id,
        representation=representation,  # type: ignore[arg-type]
        language=language,
        text=text,
        start_char=start_char,
        end_char=end_char,
        content_hash=sha256_bytes(text),
        boundary=boundary,  # type: ignore[arg-type]
    )


def _config(**kwargs: Any) -> LocalCrossEncoderConfig:
    return LocalCrossEncoderConfig(
        sentence_transformers_version=importlib.metadata.version(
            "sentence-transformers"
        ),
        transformers_version=importlib.metadata.version("transformers"),
        tokenizers_version=importlib.metadata.version("tokenizers"),
        torch_version=importlib.metadata.version("torch"),
        **kwargs,
    )


class RecordingTokenizer:
    def __init__(self, *, overflow_text: str | None = None) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.overflow_text = overflow_text

    def __call__(
        self,
        query: str,
        passage: str,
        **kwargs: Any,
    ) -> dict[str, list[list[int]]]:
        self.calls.append((query, passage, kwargs))
        token_count = len(query) + len(passage) + 2
        if passage == self.overflow_text:
            token_count += 100
        return {"input_ids": [list(range(token_count))]}


class RecordingModel:
    def __init__(self, logits_by_text: dict[str, float]) -> None:
        self.logits_by_text = logits_by_text
        self.predict_calls: list[tuple[list[tuple[str, str]], dict[str, Any]]] = []

    def predict(self, inputs: list[tuple[str, str]], **kwargs: Any) -> list[float]:
        self.predict_calls.append((list(inputs), kwargs))
        return [self.logits_by_text[passage] for _, passage in inputs]


class PairCountingTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None, dict[str, Any]]] = []
        self.special_token_calls: list[bool] = []

    def __call__(
        self, query: str, passage: str | None = None, **kwargs: Any
    ) -> dict[str, object]:
        self.calls.append((query, passage, kwargs))
        query_tokens = sum(2 if character in {"你", "中"} else 1 for character in query)
        if passage is None:
            count = query_tokens
        else:
            passage_tokens = sum(2 if character == "中" else 1 for character in passage)
            # Three explicit special tokens model a complete cross-encoder pair.
            count = 3 + query_tokens + passage_tokens
        return {"input_ids": list(range(count)), "length": [count]}

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        self.special_token_calls.append(pair)
        return 3 if pair else 2


class ChatTemplateTokenizer:
    chat_template = "maintained nonblank template"

    def __init__(self, *, template_overhead: int = 10) -> None:
        self.template_overhead = template_overhead
        self.calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []
        self.template_calls: list[tuple[list[dict[str, str]], dict[str, Any]]] = []

    def __call__(self, *texts: str, **kwargs: Any) -> dict[str, object]:
        self.calls.append((texts, dict(kwargs)))
        if len(texts) == 1:
            count = len(texts[0])
        elif len(texts) == 2:
            count = len(texts[0]) + len(texts[1]) + 2
        else:
            raise AssertionError(f"unexpected tokenizer arity: {len(texts)}")
        return {"input_ids": list(range(count)), "length": [count]}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> dict[str, object]:
        self.template_calls.append(
            ([dict(message) for message in messages], dict(kwargs))
        )
        count = self.template_overhead + len(messages[1]["content"])
        return {
            "input_ids": list(range(count)),
            "overflowing_tokens": [],
            "num_truncated_tokens": 0,
        }

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 2 if pair else 1


class ReviewedPinnedTokenizer:
    """Offline shape matching the reviewed pinned-tokenizer reproduction."""

    QUERY = "shared evidence and practical outcomes"

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []
        self.special_token_calls: list[bool] = []

    def __call__(self, *texts: str, **kwargs: Any) -> dict[str, object]:
        self.calls.append((texts, kwargs))
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

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        self.special_token_calls.append(pair)
        return 4 if pair else 2


def _runtime_versions() -> dict[str, str]:
    distributions = {
        "sentence_transformers_version": "sentence-transformers",
        "transformers_version": "transformers",
        "tokenizers_version": "tokenizers",
        "torch_version": "torch",
    }
    return {
        field_name: importlib.metadata.version(distribution)
        for field_name, distribution in distributions.items()
    }


class PassageScoreValidatorTests(unittest.TestCase):
    def test_validator_returns_one_float_for_each_unique_hash(self) -> None:
        first = _passage("same", document_id="first")
        duplicate = _passage("same", document_id="second")
        other = _passage("other", document_id="third")

        result = validate_passage_scores(
            (first, duplicate, other),
            {first.content_hash: 2, other.content_hash: -1.5},
        )

        self.assertEqual(result, {first.content_hash: 2.0, other.content_hash: -1.5})
        self.assertTrue(all(type(value) is float for value in result.values()))

    def test_validator_rejects_missing_extra_and_malformed_keys(self) -> None:
        first = _passage("first")
        second = _passage("second", document_id="second")
        expected = {first.content_hash, second.content_hash}

        cases = (
            {first.content_hash: 1.0},
            {first.content_hash: 1.0, second.content_hash: 2.0, b"x" * 32: 3.0},
            {b"x" * 31: 1.0, second.content_hash: 2.0},
            {"not-bytes": 1.0, second.content_hash: 2.0},
        )
        for scores in cases:
            with self.subTest(scores=scores):
                with self.assertRaises(PassageScoringError):
                    validate_passage_scores((first, second), scores)

        self.assertEqual(expected, {first.content_hash, second.content_hash})

    def test_validator_rejects_boolean_non_numeric_and_non_finite_scores(self) -> None:
        passage = _passage("only")
        invalid_values = (True, False, "1.0", None, math.nan, math.inf, -math.inf)

        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(PassageScoringError):
                    validate_passage_scores((passage,), {passage.content_hash: value})


class LocalCrossEncoderConfigTests(unittest.TestCase):
    def test_config_is_frozen_slotted_and_uses_exact_pinned_defaults(self) -> None:
        config = LocalCrossEncoderConfig()

        self.assertTrue(is_dataclass(config))
        self.assertEqual(config.__class__.__slots__, config.__slots__)
        self.assertEqual(DEFAULT_MODEL_NAME, "mixedbread-ai/mxbai-rerank-base-v2")
        self.assertEqual(
            DEFAULT_MODEL_REVISION,
            "3ea9d4dffa7d12a4f366be8e275c349de9fc9865",
        )
        self.assertEqual(config.model_name, DEFAULT_MODEL_NAME)
        self.assertEqual(config.revision, DEFAULT_MODEL_REVISION)
        self.assertEqual(config.tokenizer_name, DEFAULT_MODEL_NAME)
        self.assertEqual(config.tokenizer_revision, DEFAULT_MODEL_REVISION)
        self.assertEqual(config.effective_device, "cpu")
        with self.assertRaises(FrozenInstanceError):
            config.batch_size = 4  # type: ignore[misc]

    def test_rocm_is_the_same_effective_device_and_fingerprint_as_cuda(self) -> None:
        rocm = LocalCrossEncoderConfig(device="rocm")
        cuda = LocalCrossEncoderConfig(device="cuda")

        self.assertEqual(rocm.effective_device, "cuda")
        self.assertEqual(rocm.fingerprint, cuda.fingerprint)

    def test_corrected_preprocessing_uses_v3_fingerprint_identity(self) -> None:
        prefix, digest = LocalCrossEncoderConfig().fingerprint.split("/", 1)

        self.assertEqual(prefix, "local-cross-encoder-v3")
        self.assertEqual(len(digest), 64)

    def test_fingerprint_changes_for_every_result_affecting_setting(self) -> None:
        base = LocalCrossEncoderConfig()
        variants = (
            replace(base, schema_version=2),
            replace(
                base,
                model_name="other/model",
                tokenizer_name="other/model",
            ),
            replace(
                base,
                revision="other-revision",
                tokenizer_revision="other-revision",
            ),
            replace(base, device="cuda"),
            replace(base, precision="float16"),
            replace(base, batch_size=16),
            replace(base, max_length=256),
            replace(base, query_format="Q: {query}"),
            replace(base, query_policy="literal-v2"),
        )

        self.assertEqual(
            len({variant.fingerprint for variant in variants}), len(variants)
        )
        self.assertTrue(
            all(variant.fingerprint != base.fingerprint for variant in variants)
        )

    def test_config_rejects_a_tokenizer_that_is_not_the_pinned_model_processor(
        self,
    ) -> None:
        base = _runtime_versions()
        for kwargs in (
            {"tokenizer_name": "other/tokenizer"},
            {"tokenizer_revision": "other-tokenizer-revision"},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "match"):
                    LocalCrossEncoderConfig(**base, **kwargs)

    def test_each_runtime_version_rotates_fingerprint_when_installed_metadata_rotates(
        self,
    ) -> None:
        base_versions = _runtime_versions()
        base = LocalCrossEncoderConfig(**base_versions)
        distributions = {
            "sentence_transformers_version": "sentence-transformers",
            "transformers_version": "transformers",
            "tokenizers_version": "tokenizers",
            "torch_version": "torch",
        }

        for field_name, distribution in distributions.items():
            with self.subTest(field_name=field_name):
                rotated_versions = dict(base_versions)
                rotated_versions[field_name] = "rotated-" + base_versions[field_name]

                with patch.object(
                    importlib.metadata,
                    "version",
                    side_effect=lambda name,
                    versions=rotated_versions,
                    mapping=distributions: versions[
                        next(field for field, dist in mapping.items() if dist == name)
                    ],
                ):
                    rotated = LocalCrossEncoderConfig(**rotated_versions)

                self.assertNotEqual(rotated.fingerprint, base.fingerprint)

    def test_declared_runtime_version_mismatch_fails_closed_at_config_construction(
        self,
    ) -> None:
        installed = _runtime_versions()
        declared = dict(installed)
        declared["transformers_version"] = (
            "not-installed-" + installed["transformers_version"]
        )

        with self.assertRaisesRegex(ValueError, "transformers_version"):
            LocalCrossEncoderConfig(**declared)

    def test_fingerprint_encoding_separates_delimiter_containing_values(self) -> None:
        left = LocalCrossEncoderConfig(
            query_format="{query}\x1fquery_policy=y",
            query_policy="z",
        )
        right = LocalCrossEncoderConfig(
            query_format="{query}",
            query_policy="y\x1fquery_policy=z",
        )

        self.assertNotEqual(left.fingerprint, right.fingerprint)

    def test_malformed_config_is_rejected_without_any_model_loader(self) -> None:
        for kwargs in (
            {"batch_size": 0},
            {"max_length": 0},
            {"device": "tpu"},
            {"precision": "int8"},
            {"query_format": "missing-placeholder"},
            {"query_policy": ""},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    LocalCrossEncoderConfig(**kwargs)


class CrossEncoderPairLengthFunctionTests(unittest.TestCase):
    def test_lazy_full_pair_length_formats_query_and_counts_exact_pair(
        self,
    ) -> None:
        config = LocalCrossEncoderConfig(max_length=16, query_format="Q: {query}")
        tokenizer = PairCountingTokenizer()
        load_calls: list[LocalCrossEncoderConfig] = []

        def loader(loaded_config: LocalCrossEncoderConfig) -> PairCountingTokenizer:
            load_calls.append(loaded_config)
            return tokenizer

        length_function = CrossEncoderPairLengthFunction(
            config,
            "exact 你",
            loader=loader,
        )

        self.assertEqual(load_calls, [])
        pair_length = length_function("中文")
        self.assertEqual(pair_length, 17)
        self.assertGreater(pair_length, config.max_length)
        self.assertEqual(length_function.max_length, config.max_length)
        self.assertEqual(length_function.max_passage_length, 2)
        self.assertEqual(
            length_function.fingerprint, length_function.length_function_fingerprint
        )
        self.assertEqual(
            tokenizer.calls,
            [
                ("Q: exact 你", "中文", {"truncation": False, "return_length": True}),
                (
                    "Q: exact 你",
                    "",
                    {"truncation": False, "return_length": True},
                ),
            ],
        )
        self.assertEqual(tokenizer.special_token_calls, [])

    def test_splitter_budget_uses_official_passage_only_tokenization(
        self,
    ) -> None:
        config = LocalCrossEncoderConfig(max_length=16, query_format="Q: {query}")
        tokenizer = PairCountingTokenizer()
        length_function = CrossEncoderPairLengthFunction(
            config,
            "exact 你",
            tokenizer_loader=lambda _: tokenizer,
        )

        self.assertEqual(length_function(""), 14)
        self.assertEqual(length_function.passage_length_function("中文"), 3)
        self.assertEqual(length_function.max_passage_length, 2)
        self.assertGreater(length_function.max_passage_length, 0)
        self.assertLessEqual(length_function.max_passage_length, config.max_length)
        self.assertEqual(
            tokenizer.calls,
            [
                ("Q: exact 你", "", {"truncation": False, "return_length": True}),
                (
                    "中文",
                    None,
                    {
                        "add_special_tokens": False,
                        "truncation": False,
                        "return_length": True,
                    },
                ),
                (
                    "Q: exact 你",
                    "",
                    {"truncation": False, "return_length": True},
                ),
            ],
        )
        self.assertEqual(tokenizer.special_token_calls, [])

    def test_template_pair_length_and_budget_use_exact_rendered_overhead(self) -> None:
        config = LocalCrossEncoderConfig(max_length=16, query_format="Q: {query}")
        tokenizer = ChatTemplateTokenizer(template_overhead=10)
        length_function = CrossEncoderPairLengthFunction(
            config,
            "exact",
            tokenizer_loader=lambda _: tokenizer,
        )

        self.assertEqual(length_function.max_passage_length, 6)
        self.assertEqual(length_function("abc"), 13)
        self.assertEqual(length_function.passage_length_function("abc"), 3)
        self.assertEqual(
            tokenizer.calls,
            [
                (
                    ("abc",),
                    {
                        "add_special_tokens": False,
                        "truncation": False,
                        "return_length": True,
                    },
                )
            ],
        )
        self.assertEqual(
            tokenizer.template_calls,
            [
                (
                    [
                        {"role": "query", "content": "Q: exact"},
                        {"role": "document", "content": ""},
                    ],
                    {
                        "tokenize": True,
                        "add_generation_prompt": False,
                        "truncation": False,
                        "return_dict": True,
                    },
                ),
                (
                    [
                        {"role": "query", "content": "Q: exact"},
                        {"role": "document", "content": "abc"},
                    ],
                    {
                        "tokenize": True,
                        "add_generation_prompt": False,
                        "truncation": False,
                        "return_dict": True,
                    },
                ),
            ],
        )

    def test_nonblank_chat_template_requires_callable_template_api(self) -> None:
        tokenizer = ChatTemplateTokenizer()
        tokenizer.apply_chat_template = None  # type: ignore[method-assign,assignment]
        length_function = CrossEncoderPairLengthFunction(
            LocalCrossEncoderConfig(max_length=16),
            "query",
            tokenizer_loader=lambda _: tokenizer,
        )

        with self.assertRaisesRegex(PassageScoringError, "apply_chat_template"):
            length_function("passage")

    def test_splitter_budget_matches_reviewed_pinned_tokenizer_shape(self) -> None:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        config = LocalCrossEncoderConfig(max_length=512)
        tokenizer = ReviewedPinnedTokenizer()
        length_function = CrossEncoderPairLengthFunction(
            config,
            ReviewedPinnedTokenizer.QUERY,
            tokenizer_loader=lambda _: tokenizer,
        )
        passage = "中" * 502

        self.assertEqual(length_function(passage), 513)
        self.assertEqual(length_function(""), 10)
        self.assertEqual(length_function.passage_length_function("中"), 2)
        self.assertEqual(
            length_function.passage_length_function("中" * 501),
            length_function.max_passage_length,
        )
        self.assertEqual(length_function.passage_length_function(passage), 503)
        self.assertEqual(length_function.max_passage_length, 502)
        self.assertGreater(length_function.max_passage_length, 0)
        self.assertLessEqual(length_function.max_passage_length, config.max_length)

        splitter = RecursiveCharacterTextSplitter(
            separators=[" ", ""],
            keep_separator="end",
            strip_whitespace=False,
            chunk_overlap=0,
            chunk_size=length_function.max_passage_length,
            length_function=length_function.passage_length_function,
        )
        chunks = splitter.split_text(passage)

        self.assertEqual([len(chunk) for chunk in chunks], [251, 251])
        self.assertTrue(
            all(length_function(chunk) <= config.max_length for chunk in chunks)
        )
        self.assertIn(
            (
                (ReviewedPinnedTokenizer.QUERY, ""),
                {"truncation": False, "return_length": True},
            ),
            tokenizer.calls,
        )
        self.assertIn(
            (
                (passage,),
                {
                    "add_special_tokens": False,
                    "truncation": False,
                    "return_length": True,
                },
            ),
            tokenizer.calls,
        )
        self.assertEqual(tokenizer.special_token_calls, [])

    def test_splitter_budget_rejects_malformed_empty_pair_ids(self) -> None:
        class MalformedPairTokenizer(PairCountingTokenizer):
            def __call__(
                self,
                query: str,
                passage: str | None = None,
                **kwargs: Any,
            ) -> dict[str, object]:
                if passage == "":
                    return {"input_ids": None, "length": [0]}
                return super().__call__(query, passage, **kwargs)

        length_function = CrossEncoderPairLengthFunction(
            LocalCrossEncoderConfig(max_length=16),
            "exact 你",
            tokenizer_loader=lambda _: MalformedPairTokenizer(),
        )

        with self.assertRaisesRegex(PassageScoringError, "input_ids"):
            _ = length_function.max_passage_length

    def test_splitter_budget_rejects_reported_empty_pair_truncation(self) -> None:
        class TruncatedPairTokenizer(PairCountingTokenizer):
            def __call__(
                self,
                query: str,
                passage: str | None = None,
                **kwargs: Any,
            ) -> dict[str, object]:
                encoded = super().__call__(query, passage, **kwargs)
                if passage == "":
                    encoded["num_truncated_tokens"] = 1
                return encoded

        length_function = CrossEncoderPairLengthFunction(
            LocalCrossEncoderConfig(max_length=16),
            "exact 你",
            tokenizer_loader=lambda _: TruncatedPairTokenizer(),
        )

        with self.assertRaisesRegex(PassageScoringError, "truncation"):
            _ = length_function.max_passage_length

    def test_splitter_budget_rejects_a_query_with_no_positive_passage_budget(
        self,
    ) -> None:
        config = LocalCrossEncoderConfig(max_length=11)
        length_function = CrossEncoderPairLengthFunction(
            config,
            "exact 你",
            tokenizer_loader=lambda _: PairCountingTokenizer(),
        )

        with self.assertRaisesRegex(PassageScoringError, "positive"):
            _ = length_function.max_passage_length

    def test_splitter_budget_fingerprint_is_stable_and_distinct_from_pair_length(
        self,
    ) -> None:
        config = LocalCrossEncoderConfig(max_length=16, query_format="Q: {query}")
        first = CrossEncoderPairLengthFunction(
            config,
            "exact 你",
            tokenizer_loader=lambda _: PairCountingTokenizer(),
        )
        second = CrossEncoderPairLengthFunction(
            config,
            "exact 你",
            tokenizer_loader=lambda _: PairCountingTokenizer(),
        )

        self.assertIs(first.passage_length_function, first.passage_length_function)
        self.assertEqual(
            first.passage_length_function.fingerprint,
            second.passage_length_function.fingerprint,
        )
        self.assertNotEqual(
            first.passage_length_function.fingerprint,
            first.fingerprint,
        )
        self.assertTrue(
            first.passage_length_function.fingerprint.startswith(
                "cross-encoder-passage-length-v2/"
            )
        )
        self.assertTrue(first.fingerprint.startswith("cross-encoder-pair-length-v1/"))

    def test_pair_length_loader_is_tokenizer_only_and_never_imports_model_stack(
        self,
    ) -> None:
        tokenizer = PairCountingTokenizer()
        loader_calls = 0

        def loader(config: LocalCrossEncoderConfig) -> PairCountingTokenizer:
            nonlocal loader_calls
            loader_calls += 1
            return tokenizer

        length_function = CrossEncoderPairLengthFunction(
            LocalCrossEncoderConfig(), "query", loader=loader
        )
        self.assertEqual(loader_calls, 0)
        model_stack_before = {
            name
            for name in sys.modules
            if name == "sentence_transformers" or name == "torch"
        }

        self.assertEqual(length_function("passage"), 3 + len("query") + len("passage"))
        self.assertEqual(loader_calls, 1)
        self.assertEqual(length_function("passage"), 3 + len("query") + len("passage"))
        self.assertEqual(loader_calls, 1)
        self.assertEqual(
            model_stack_before,
            {
                name
                for name in sys.modules
                if name == "sentence_transformers" or name == "torch"
            },
        )


class LocalCrossEncoderScorerTests(unittest.TestCase):
    def test_score_deduplicates_equal_content_and_returns_finite_float_map(
        self,
    ) -> None:
        first = _passage("same", start_char=0, end_char=4)
        repeated = _passage("same", start_char=5, end_char=9)
        other = _passage("other", start_char=10, end_char=15)
        model = RecordingModel({"same": 2.5, "other": -1})
        scorer = LocalCrossEncoderScorer(
            _config(batch_size=2),
            loader=lambda _: (model, RecordingTokenizer()),
        )

        scores = scorer.score("exact query", (first, repeated, other))

        self.assertEqual(scores, {first.content_hash: 2.5, other.content_hash: -1.0})
        self.assertEqual(len(model.predict_calls), 1)
        self.assertTrue(all(type(value) is float for value in scores.values()))

    def test_runtime_metadata_drift_fails_before_custom_loader(self) -> None:
        config = LocalCrossEncoderConfig(**_runtime_versions())
        load_calls = 0
        installed = _runtime_versions()
        drifted = dict(installed)
        drifted["torch_version"] = "drifted-" + installed["torch_version"]
        distributions = {
            "sentence_transformers_version": "sentence-transformers",
            "transformers_version": "transformers",
            "tokenizers_version": "tokenizers",
            "torch_version": "torch",
        }

        def loader(config: LocalCrossEncoderConfig) -> object:
            nonlocal load_calls
            load_calls += 1
            return object()

        with patch.object(
            importlib.metadata,
            "version",
            side_effect=lambda name, versions=drifted, mapping=distributions: versions[
                next(field for field, dist in mapping.items() if dist == name)
            ],
        ):
            with self.assertRaisesRegex(ValueError, "torch_version"):
                LocalCrossEncoderScorer(config, loader=loader)

        self.assertEqual(load_calls, 0)

    def test_import_is_lazy_and_does_not_import_model_stack(self) -> None:
        code = """
import sys
import trec_ragtime.passage_scoring
assert 'sentence_transformers' not in sys.modules
assert 'torch' not in sys.modules
"""
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_blank_query_and_invalid_passages_fail_before_loading(self) -> None:
        model = RecordingModel({"passage": 1.0})
        tokenizer = RecordingTokenizer()
        load_calls: list[LocalCrossEncoderConfig] = []

        def loader(
            config: LocalCrossEncoderConfig,
        ) -> tuple[RecordingModel, RecordingTokenizer]:
            load_calls.append(config)
            return model, tokenizer

        scorer = LocalCrossEncoderScorer(loader=loader)
        passage = _passage("passage")

        with self.assertRaises(PassageScoringError):
            scorer.score("   ", (passage,))
        with self.assertRaises(PassageScoringError):
            scorer.score("query", (object(),))  # type: ignore[arg-type]

        self.assertEqual(load_calls, [])
        self.assertEqual(model.predict_calls, [])

    def test_config_is_read_only_before_and_after_model_loading(self) -> None:
        passage = _passage("passage")
        original_config = LocalCrossEncoderConfig(batch_size=1)
        replacement = LocalCrossEncoderConfig(batch_size=7)
        model = RecordingModel({"passage": 1.0})
        tokenizer = RecordingTokenizer()
        scorer = LocalCrossEncoderScorer(
            original_config,
            loader=lambda config: (model, tokenizer),
        )

        with self.assertRaises(AttributeError):
            scorer.config = replacement  # type: ignore[misc]

        scorer.score("query", (passage,))

        with self.assertRaises(AttributeError):
            scorer.config = replacement  # type: ignore[misc]
        self.assertIs(scorer.config, original_config)
        self.assertEqual(scorer.fingerprint, original_config.fingerprint)
        self.assertEqual(model.predict_calls[0][1]["batch_size"], 1)

    def test_default_loader_constructs_the_direct_cross_encoder_path(self) -> None:
        passage = _passage("passage")
        constructor_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        tokenizer_calls: list[tuple[str, str, dict[str, Any]]] = []
        predict_calls: list[tuple[list[tuple[str, str]], dict[str, Any]]] = []
        models: list[Any] = []
        dtype = object()

        class FakeTokenizer:
            def __call__(
                self,
                query: str,
                text: str,
                **kwargs: Any,
            ) -> dict[str, list[list[int]]]:
                tokenizer_calls.append((query, text, kwargs))
                return {"input_ids": [[1, 2, 3]]}

        class FakeIdentity:
            pass

        class FakeCrossEncoder:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                constructor_calls.append((args, kwargs))
                self._tokenizer = FakeTokenizer()
                models.append(self)

            @property
            def tokenizer(self) -> Any:
                return self._tokenizer

            def predict(
                self, inputs: list[tuple[str, str]], **kwargs: Any
            ) -> list[float]:
                predict_calls.append((list(inputs), kwargs))
                return [4.75]

        fake_sentence_transformers = types.ModuleType("sentence_transformers")
        fake_sentence_transformers.CrossEncoder = FakeCrossEncoder  # type: ignore[attr-defined]
        fake_sentence_transformers_modules = types.ModuleType(
            "sentence_transformers.sentence_transformer.modules"
        )

        class FakeTransformer:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

        fake_sentence_transformers_modules.Transformer = FakeTransformer  # type: ignore[attr-defined]
        fake_torch = types.ModuleType("torch")
        fake_torch.float32 = dtype  # type: ignore[attr-defined]
        fake_torch.float16 = object()  # type: ignore[attr-defined]
        fake_torch.bfloat16 = object()  # type: ignore[attr-defined]
        fake_torch.nn = types.SimpleNamespace(Identity=FakeIdentity)  # type: ignore[attr-defined]

        config = LocalCrossEncoderConfig(
            device="rocm",
            precision="float32",
            batch_size=4,
            max_length=17,
            query_format="Q: {query}",
        )
        with patch.dict(
            sys.modules,
            {
                "sentence_transformers": fake_sentence_transformers,
                "sentence_transformers.sentence_transformer.modules": fake_sentence_transformers_modules,
                "torch": fake_torch,
            },
        ):
            scorer = LocalCrossEncoderScorer(config)
            scores = scorer.score("query", (passage,))

        self.assertEqual(scores, {passage.content_hash: 4.75})
        self.assertEqual(len(constructor_calls), 1)
        model_args, constructor_kwargs = constructor_calls[0]
        self.assertEqual(model_args, (DEFAULT_MODEL_NAME,))
        self.assertEqual(
            constructor_kwargs,
            {
                "revision": DEFAULT_MODEL_REVISION,
                "device": "cuda",
                "max_length": 17,
                "model_kwargs": {"torch_dtype": dtype},
            },
        )
        self.assertEqual(len(tokenizer_calls), 1)
        self.assertEqual(tokenizer_calls[0][0:2], ("Q: query", "passage"))
        self.assertIs(tokenizer_calls[0][2]["truncation"], False)
        self.assertEqual(tokenizer_calls[0][2]["max_length"], 17)
        self.assertEqual(len(predict_calls), 1)
        self.assertEqual(predict_calls[0][0], [("Q: query", "passage")])
        predict_kwargs = predict_calls[0][1]
        self.assertEqual(predict_kwargs["batch_size"], 4)
        self.assertEqual(predict_kwargs["device"], "cuda")
        self.assertIsInstance(predict_kwargs["activation_fn"], FakeIdentity)
        self.assertFalse(predict_kwargs["apply_softmax"])
        self.assertTrue(predict_kwargs["convert_to_numpy"])
        self.assertFalse(predict_kwargs["convert_to_tensor"])

    def test_lazy_loader_is_called_once_and_scores_deduplicate_in_deterministic_batches(
        self,
    ) -> None:
        first = _passage("first", document_id="first")
        duplicate = _passage("first", document_id="duplicate")
        second = _passage("second", document_id="second")
        third = _passage("third", document_id="third")
        passages = (first, duplicate, second, third)
        model = RecordingModel({"first": -3.25, "second": 0.0, "third": 7.5})
        tokenizer = RecordingTokenizer()
        load_calls: list[LocalCrossEncoderConfig] = []

        def loader(
            config: LocalCrossEncoderConfig,
        ) -> tuple[RecordingModel, RecordingTokenizer]:
            load_calls.append(config)
            return model, tokenizer

        scorer = LocalCrossEncoderScorer(
            LocalCrossEncoderConfig(
                batch_size=2,
                max_length=100,
                query_format="Q: {query}",
            ),
            loader=loader,
        )

        result = scorer.score("exact query", passages)
        second_result = scorer.score("exact query", passages)

        self.assertEqual(
            result,
            {
                first.content_hash: -3.25,
                second.content_hash: 0.0,
                third.content_hash: 7.5,
            },
        )
        self.assertEqual(second_result, result)
        self.assertEqual(len(load_calls), 1)
        self.assertEqual(
            [(query, passage) for query, passage, _ in tokenizer.calls],
            [
                ("Q: exact query", "first"),
                ("Q: exact query", "second"),
                ("Q: exact query", "third"),
                ("Q: exact query", "first"),
                ("Q: exact query", "second"),
                ("Q: exact query", "third"),
            ],
        )
        self.assertEqual(
            [call[0] for call in model.predict_calls],
            [
                [("Q: exact query", "first"), ("Q: exact query", "second")],
                [("Q: exact query", "third")],
                [("Q: exact query", "first"), ("Q: exact query", "second")],
                [("Q: exact query", "third")],
            ],
        )
        for _, kwargs in model.predict_calls:
            self.assertEqual(kwargs["batch_size"], 2)
            self.assertEqual(kwargs["device"], "cpu")
            self.assertFalse(kwargs["show_progress_bar"])
            self.assertFalse(kwargs["apply_softmax"])
            self.assertTrue(kwargs["convert_to_numpy"])
            self.assertFalse(kwargs["convert_to_tensor"])
            self.assertEqual(
                kwargs["processing_kwargs"],
                {"text": {"truncation": False, "max_length": 100}},
            )
            self.assertEqual(type(kwargs["activation_fn"]).__name__, "Identity")

        for _, _, kwargs in tokenizer.calls:
            self.assertIs(kwargs["truncation"], False)
            self.assertEqual(kwargs["max_length"], 100)

    def test_device_is_normalized_and_forwarded_to_loader_and_predict(self) -> None:
        passage = _passage("passage")
        model = RecordingModel({"passage": 1.25})
        tokenizer = RecordingTokenizer()
        load_devices: list[str] = []

        def loader(
            config: LocalCrossEncoderConfig,
        ) -> tuple[RecordingModel, RecordingTokenizer]:
            load_devices.append(config.effective_device)
            return model, tokenizer

        scorer = LocalCrossEncoderScorer(
            LocalCrossEncoderConfig(device="rocm"),
            loader=loader,
        )
        scorer.score("query", (passage,))

        self.assertEqual(load_devices, ["cuda"])
        self.assertEqual(model.predict_calls[0][1]["device"], "cuda")

    def test_overflow_is_rejected_before_any_model_prediction(self) -> None:
        first = _passage("first")
        overflowing = _passage("overflow", document_id="overflow")
        model = RecordingModel({"first": 1.0, "overflow": 2.0})
        tokenizer = RecordingTokenizer(overflow_text="overflow")

        scorer = LocalCrossEncoderScorer(
            LocalCrossEncoderConfig(max_length=20),
            loader=lambda config: (model, tokenizer),
        )

        with self.assertRaisesRegex(PassageScoringError, "overflow"):
            scorer.score("query", (first, overflowing))

        self.assertEqual(len(tokenizer.calls), 2)
        self.assertEqual(model.predict_calls, [])

    def test_template_rendered_overflow_is_rejected_before_prediction(self) -> None:
        passage = _passage("abc")
        model = RecordingModel({"abc": 1.0})
        tokenizer = ChatTemplateTokenizer(template_overhead=10)
        scorer = LocalCrossEncoderScorer(
            LocalCrossEncoderConfig(max_length=12),
            loader=lambda config: (model, tokenizer),
        )

        with self.assertRaisesRegex(PassageScoringError, "overflow"):
            scorer.score("query", (passage,))

        self.assertEqual(model.predict_calls, [])
        self.assertEqual(tokenizer.calls, [])
        self.assertEqual(
            tokenizer.template_calls,
            [
                (
                    [
                        {"role": "query", "content": "query"},
                        {"role": "document", "content": "abc"},
                    ],
                    {
                        "tokenize": True,
                        "add_generation_prompt": False,
                        "truncation": False,
                        "return_dict": True,
                    },
                )
            ],
        )

    def test_query_only_overflow_is_rejected_before_model_prediction(self) -> None:
        passage = _passage("p")
        model = RecordingModel({"p": 1.0})
        tokenizer = RecordingTokenizer()
        scorer = LocalCrossEncoderScorer(
            LocalCrossEncoderConfig(max_length=3),
            loader=lambda config: (model, tokenizer),
        )

        with self.assertRaisesRegex(PassageScoringError, "overflow"):
            scorer.score("query", (passage,))

        self.assertEqual(model.predict_calls, [])

    def test_contradictory_duplicate_hash_text_is_rejected(self) -> None:
        original = _passage("same", document_id="original")
        contradictory_text = object.__new__(Passage)
        for field in original.__slots__:
            object.__setattr__(contradictory_text, field, getattr(original, field))
        object.__setattr__(contradictory_text, "text", "different")

        model = RecordingModel({"same": 1.0})
        tokenizer = RecordingTokenizer()
        load_calls = 0

        def loader(
            config: LocalCrossEncoderConfig,
        ) -> tuple[RecordingModel, RecordingTokenizer]:
            nonlocal load_calls
            load_calls += 1
            return model, tokenizer

        scorer = LocalCrossEncoderScorer(loader=loader)
        with self.assertRaises(PassageScoringError):
            scorer.score("query", (original, contradictory_text))
        self.assertEqual(load_calls, 0)


if __name__ == "__main__":
    unittest.main()
