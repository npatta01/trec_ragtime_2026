from __future__ import annotations

import math
import subprocess
import sys
import unittest
from dataclasses import FrozenInstanceError, is_dataclass, replace
from typing import Any

from trec_ragtime.passages import Passage, sha256_bytes
from trec_ragtime.passage_scoring import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
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
        self.assertEqual(config.effective_device, "cpu")
        with self.assertRaises(FrozenInstanceError):
            config.batch_size = 4  # type: ignore[misc]

    def test_rocm_is_the_same_effective_device_and_fingerprint_as_cuda(self) -> None:
        rocm = LocalCrossEncoderConfig(device="rocm")
        cuda = LocalCrossEncoderConfig(device="cuda")

        self.assertEqual(rocm.effective_device, "cuda")
        self.assertEqual(rocm.fingerprint, cuda.fingerprint)

    def test_fingerprint_changes_for_every_result_affecting_setting(self) -> None:
        base = LocalCrossEncoderConfig(sentence_transformers_version="5.7.0")
        variants = (
            replace(base, schema_version=2),
            replace(base, model_name="other/model"),
            replace(base, revision="other-revision"),
            replace(base, sentence_transformers_version="5.7.1"),
            replace(base, device="cuda"),
            replace(base, precision="float16"),
            replace(base, batch_size=16),
            replace(base, max_length=256),
            replace(base, query_format="Q: {query}"),
            replace(base, query_policy="literal-v2"),
        )

        self.assertEqual(len({variant.fingerprint for variant in variants}), len(variants))
        self.assertTrue(all(variant.fingerprint != base.fingerprint for variant in variants))

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


class LocalCrossEncoderScorerTests(unittest.TestCase):
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

        def loader(config: LocalCrossEncoderConfig) -> tuple[RecordingModel, RecordingTokenizer]:
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

    def test_lazy_loader_is_called_once_and_scores_deduplicate_in_deterministic_batches(self) -> None:
        first = _passage("first", document_id="first")
        duplicate = _passage("first", document_id="duplicate")
        second = _passage("second", document_id="second")
        third = _passage("third", document_id="third")
        passages = (first, duplicate, second, third)
        model = RecordingModel({"first": -3.25, "second": 0.0, "third": 7.5})
        tokenizer = RecordingTokenizer()
        load_calls: list[LocalCrossEncoderConfig] = []

        def loader(config: LocalCrossEncoderConfig) -> tuple[RecordingModel, RecordingTokenizer]:
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

        self.assertEqual(result, {
            first.content_hash: -3.25,
            second.content_hash: 0.0,
            third.content_hash: 7.5,
        })
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
            self.assertEqual(kwargs["processing_kwargs"], {
                "text": {"truncation": False, "max_length": 100}
            })
            self.assertEqual(type(kwargs["activation_fn"]).__name__, "Identity")

        for _, _, kwargs in tokenizer.calls:
            self.assertIs(kwargs["truncation"], False)
            self.assertEqual(kwargs["max_length"], 100)

    def test_device_is_normalized_and_forwarded_to_loader_and_predict(self) -> None:
        passage = _passage("passage")
        model = RecordingModel({"passage": 1.25})
        tokenizer = RecordingTokenizer()
        load_devices: list[str] = []

        def loader(config: LocalCrossEncoderConfig) -> tuple[RecordingModel, RecordingTokenizer]:
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

    def test_contradictory_duplicate_hash_text_or_lineage_is_rejected(self) -> None:
        original = _passage("same", document_id="original")
        contradictory_text = object.__new__(Passage)
        for field in original.__slots__:
            object.__setattr__(contradictory_text, field, getattr(original, field))
        object.__setattr__(contradictory_text, "text", "different")

        model = RecordingModel({"same": 1.0})
        tokenizer = RecordingTokenizer()
        load_calls = 0

        def loader(config: LocalCrossEncoderConfig) -> tuple[RecordingModel, RecordingTokenizer]:
            nonlocal load_calls
            load_calls += 1
            return model, tokenizer

        scorer = LocalCrossEncoderScorer(loader=loader)
        with self.assertRaises(PassageScoringError):
            scorer.score("query", (original, contradictory_text))
        self.assertEqual(load_calls, 0)

        same_id_different_document = replace(original, document_id="different-document")
        object.__setattr__(same_id_different_document, "passage_id", original.passage_id)
        with self.assertRaises(PassageScoringError):
            scorer.score("query", (original, same_id_different_document))
        self.assertEqual(load_calls, 0)


if __name__ == "__main__":
    unittest.main()
