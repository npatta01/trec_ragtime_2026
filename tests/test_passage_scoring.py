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

from trec_ragtime.passages import ChunkerConfig, Passage, sha256_bytes
from trec_ragtime.passage_scoring import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    CrossEncoderPairLengthFunction,
    LocalCrossEncoderConfig,
    LocalCrossEncoderScorer,
    PassageScoreValidationError,
    PassageScoringError,
    validate_passage_scores,
)


def _passage(
    text: str,
    *,
    document_id: str = "doc",
    passage_id: bytes | None = None,
) -> Passage:
    return Passage(
        document_id=document_id,
        source_content_hash=sha256_bytes(text),
        representation="canonical",
        representation_hash=sha256_bytes(text),
        language="eng",
        chunk_index=0,
        passage_id=passage_id or sha256_bytes(f"passage:{document_id}:{text}"),
        text=text,
        start_char=0,
        end_char=len(text),
        boundary="paragraph_packed",
        previous_passage_id=None,
        next_passage_id=None,
        content_hash=sha256_bytes(text),
        chunker_fingerprint="test-chunker-v1",
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
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def __call__(
        self, query: str, passage: str, **kwargs: Any
    ) -> dict[str, list[list[int]]]:
        self.calls.append((query, passage, kwargs))
        query_tokens = sum(2 if character == "你" else 1 for character in query)
        passage_tokens = sum(2 if character == "中" else 1 for character in passage)
        # Three explicit special tokens model a complete cross-encoder pair.
        count = 3 + query_tokens + passage_tokens
        return {"input_ids": [list(range(count))]}


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
                with self.assertRaises(PassageScoreValidationError):
                    validate_passage_scores((first, second), scores)

        self.assertEqual(expected, {first.content_hash, second.content_hash})

    def test_validator_rejects_boolean_non_numeric_and_non_finite_scores(self) -> None:
        passage = _passage("only")
        invalid_values = (True, False, "1.0", None, math.nan, math.inf, -math.inf)

        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(PassageScoreValidationError):
                    validate_passage_scores((passage,), {passage.content_hash: value})


class LocalCrossEncoderConfigTests(unittest.TestCase):
    def test_config_is_frozen_slotted_and_uses_exact_pinned_defaults(self) -> None:
        config = LocalCrossEncoderConfig()

        self.assertTrue(is_dataclass(config))
        self.assertEqual(config.__class__.__slots__, config.__slots__)
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

    def test_fingerprint_changes_for_every_result_affecting_setting(self) -> None:
        base = LocalCrossEncoderConfig()
        variants = (
            replace(base, schema_version=2),
            replace(base, model_name="other/model"),
            replace(base, revision="other-revision"),
            replace(base, tokenizer_name="other/tokenizer"),
            replace(base, tokenizer_revision="other-tokenizer-revision"),
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
    def test_lazy_full_pair_length_formats_query_and_counts_special_tokens(
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
        self.assertEqual(
            length_function.fingerprint, length_function.length_function_fingerprint
        )
        chunker_config = ChunkerConfig(
            chunk_size=config.max_length,
            length_function=length_function,
            length_function_fingerprint=length_function.fingerprint,
        )
        self.assertEqual(
            chunker_config.resolved_length_function_fingerprint,
            length_function.fingerprint,
        )
        self.assertEqual(
            tokenizer.calls,
            [
                ("Q: exact 你", "中文", {"truncation": False, "return_length": True}),
                ("Q: exact 你", "", {"truncation": False, "return_length": True}),
            ],
        )

    def test_splitter_budget_is_additive_and_uses_exact_empty_pair_overhead(
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
                    "Q: exact 你",
                    "中文",
                    {"truncation": False, "return_length": True},
                ),
            ],
        )

        chunker_config = ChunkerConfig(
            chunk_size=length_function.max_passage_length,
            length_function=length_function.passage_length_function,
        )
        self.assertEqual(
            chunker_config.resolved_length_function_fingerprint,
            length_function.passage_length_function.fingerprint,
        )

    def test_splitter_budget_matches_reviewed_pinned_query_case(self) -> None:
        config = LocalCrossEncoderConfig(max_length=512)
        length_function = CrossEncoderPairLengthFunction(
            config,
            "exact 你",
            tokenizer_loader=lambda _: PairCountingTokenizer(),
        )

        self.assertEqual(length_function(""), 11)
        self.assertEqual(length_function.max_passage_length, 501)
        self.assertGreater(length_function.max_passage_length, 0)
        self.assertLessEqual(length_function.max_passage_length, config.max_length)

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

    def test_default_loader_and_predict_path_uses_pinned_runtime_seam(self) -> None:
        passage = _passage("passage")
        constructor_calls: list[tuple[str, dict[str, Any]]] = []
        transformer_constructor_calls: list[tuple[str, dict[str, Any]]] = []
        tokenizer_loader_calls: list[tuple[str, dict[str, Any]]] = []
        loaded_tokenizers: list[Any] = []
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

        class GetterOnlyTransformer:
            def __init__(self, tokenizer: Any) -> None:
                self.processor = tokenizer

            @property
            def tokenizer(self) -> Any:
                return self.processor

        class FakeTransformer(GetterOnlyTransformer):
            def __init__(self, model_name_or_path: str, **kwargs: Any) -> None:
                transformer_constructor_calls.append((model_name_or_path, kwargs))
                tokenizer = FakeAutoTokenizer.from_pretrained(
                    kwargs["tokenizer_name_or_path"],
                    **kwargs["processor_kwargs"],
                )
                super().__init__(tokenizer)

        class FakeCrossEncoder:
            def __init__(
                self,
                model_name: str | None = None,
                *,
                modules: list[GetterOnlyTransformer] | None = None,
                **kwargs: Any,
            ) -> None:
                constructor_calls.append((model_name, {"modules": modules, **kwargs}))
                self._first_module = (
                    modules[0] if modules else GetterOnlyTransformer(object())
                )
                models.append(self)

            @property
            def tokenizer(self) -> Any:
                return self._first_module.tokenizer

            @tokenizer.setter
            def tokenizer(self, value: Any) -> None:
                self._first_module.tokenizer = value

            def predict(
                self, inputs: list[tuple[str, str]], **kwargs: Any
            ) -> list[float]:
                predict_calls.append((list(inputs), kwargs))
                return [4.75]

        class FakeAutoTokenizer:
            @classmethod
            def from_pretrained(cls, model_name: str, **kwargs: Any) -> FakeTokenizer:
                tokenizer_loader_calls.append((model_name, kwargs))
                tokenizer = FakeTokenizer()
                loaded_tokenizers.append(tokenizer)
                return tokenizer

        fake_sentence_transformers = types.ModuleType("sentence_transformers")
        fake_sentence_transformers.CrossEncoder = FakeCrossEncoder  # type: ignore[attr-defined]
        fake_sentence_transformers_modules = types.ModuleType(
            "sentence_transformers.sentence_transformer.modules"
        )
        fake_sentence_transformers_modules.Transformer = FakeTransformer  # type: ignore[attr-defined]
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoTokenizer = FakeAutoTokenizer  # type: ignore[attr-defined]
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
            tokenizer_name="explicit/tokenizer",
            tokenizer_revision="tokenizer-revision",
        )
        with patch.dict(
            sys.modules,
            {
                "sentence_transformers": fake_sentence_transformers,
                "sentence_transformers.sentence_transformer.modules": fake_sentence_transformers_modules,
                "transformers": fake_transformers,
                "torch": fake_torch,
            },
        ):
            scorer = LocalCrossEncoderScorer(config)
            scores = scorer.score("query", (passage,))

        self.assertEqual(scores, {passage.content_hash: 4.75})
        self.assertEqual(len(constructor_calls), 1)
        model_name, constructor_kwargs = constructor_calls[0]
        self.assertIsNone(model_name)
        self.assertEqual(constructor_kwargs["device"], "cuda")
        self.assertEqual(len(constructor_kwargs["modules"]), 1)
        self.assertEqual(
            transformer_constructor_calls,
            [
                (
                    DEFAULT_MODEL_NAME,
                    {
                        "transformer_task": "sequence-classification",
                        "model_kwargs": {
                            "revision": DEFAULT_MODEL_REVISION,
                            "torch_dtype": dtype,
                        },
                        "processor_kwargs": {
                            "revision": "tokenizer-revision",
                            "model_max_length": 17,
                        },
                        "config_kwargs": {"revision": DEFAULT_MODEL_REVISION},
                        "max_seq_length": 17,
                        "tokenizer_name_or_path": "explicit/tokenizer",
                    },
                )
            ],
        )
        self.assertEqual(
            tokenizer_loader_calls,
            [
                (
                    "explicit/tokenizer",
                    {
                        "revision": "tokenizer-revision",
                        "model_max_length": 17,
                    },
                )
            ],
        )
        self.assertEqual(len(tokenizer_calls), 1)
        self.assertIs(models[0].tokenizer, loaded_tokenizers[0])
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

    def test_contradictory_duplicate_hash_text_or_lineage_is_rejected(self) -> None:
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

        same_id_different_document = replace(original, document_id="different-document")
        object.__setattr__(
            same_id_different_document, "passage_id", original.passage_id
        )
        with self.assertRaises(PassageScoringError):
            scorer.score("query", (original, same_id_different_document))
        self.assertEqual(load_calls, 0)


if __name__ == "__main__":
    unittest.main()
