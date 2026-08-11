from __future__ import annotations

import asyncio
from contextlib import redirect_stderr, redirect_stdout
import json
import inspect
from io import StringIO
import math
import sys
import threading
import types
import unittest
from collections import UserDict
from collections.abc import Callable
from dataclasses import replace
from unittest.mock import patch

from trec_ragtime.flash_passage_scorer import (
    FLASH_PAYLOAD_LIMIT_BYTES,
    WIRE_SCHEMA_VERSION,
    FlashEndpointBatchRunner,
    FlashScoringError,
    RunpodFlashConfig,
    RunpodFlashPassageScorer,
    decode_flash_request,
    decode_flash_response,
    encode_flash_request,
    encode_flash_response,
)
from trec_ragtime.flash_worker import FlashPassageWorker, build_flash_worker
from trec_ragtime.passage_scoring import (
    LocalCrossEncoderConfig,
    LocalCrossEncoderScorer,
)
from trec_ragtime.passages import Passage, sha256_bytes


def _passage(text: str, *, document_id: str = "doc", index: int = 0) -> Passage:
    return Passage(
        document_id=document_id,
        representation="canonical",
        language="eng",
        text=text,
        start_char=0,
        end_char=len(text),
        content_hash=sha256_bytes(text),
        boundary="sentence",
    )


def _wire_model_config(config: LocalCrossEncoderConfig) -> dict[str, object]:
    return {
        "schema_version": config.schema_version,
        "model_name": config.model_name,
        "revision": config.revision,
        "tokenizer_name": config.tokenizer_name,
        "tokenizer_revision": config.tokenizer_revision,
        "sentence_transformers_version": config.sentence_transformers_version,
        "transformers_version": config.transformers_version,
        "tokenizers_version": config.tokenizers_version,
        "torch_version": config.torch_version,
        "device": config.effective_device,
        "precision": config.effective_precision,
        "batch_size": config.batch_size,
        "max_length": config.max_length,
        "query_format": config.query_format,
        "query_policy": config.query_policy,
    }


def _request_for_config(
    config: LocalCrossEncoderConfig,
    *,
    query: str = "query",
    text: str = "text",
) -> dict[str, object]:
    return {
        "schema_version": WIRE_SCHEMA_VERSION,
        "scorer_fingerprint": config.fingerprint,
        "model_config": _wire_model_config(config),
        "query": query,
        "rows": [{"hash": sha256_bytes(text).hex(), "text": text}],
    }


class BatchEncodingTokenizer:
    def __init__(self, *, overflow_text: str = "overflow") -> None:
        self.calls: list[tuple[str, str, dict[str, object]]] = []
        self.overflow_text = overflow_text

    def __call__(
        self, query: str, text: str, **kwargs: object
    ) -> UserDict[str, object]:
        self.calls.append((query, text, kwargs))
        length = 100 if text == self.overflow_text else len(query) + len(text)
        return UserDict(
            {
                "input_ids": [[1] * length],
                "length": [length],
                "overflowing_tokens": [],
                "num_truncated_tokens": 0,
            }
        )


class ChatTemplateBatchEncodingTokenizer(BatchEncodingTokenizer):
    chat_template = "maintained nonblank template"

    def __init__(self, *, template_overhead: int = 10) -> None:
        super().__init__()
        self.template_overhead = template_overhead
        self.template_calls: list[tuple[list[dict[str, str]], dict[str, object]]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        **kwargs: object,
    ) -> UserDict[str, object]:
        self.template_calls.append(
            ([dict(message) for message in messages], dict(kwargs))
        )
        length = self.template_overhead + len(messages[1]["content"])
        return UserDict(
            {
                "input_ids": list(range(length)),
                "overflowing_tokens": [],
                "num_truncated_tokens": 0,
            }
        )


class FakeModel:
    def __init__(self, *, multiplier: float = 1.0) -> None:
        self.multiplier = multiplier
        self.predict_calls: list[tuple[list[tuple[str, str]], dict[str, object]]] = []

    def predict(self, pairs: list[tuple[str, str]], **kwargs: object) -> list[float]:
        self.predict_calls.append((pairs, kwargs))
        return [self.multiplier * float(len(text)) for _, text in pairs]


class CountingLoader:
    def __init__(self, model: FakeModel, tokenizer: BatchEncodingTokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.calls: list[object] = []

    def __call__(self, config: object) -> tuple[FakeModel, BatchEncodingTokenizer]:
        self.calls.append(config)
        return self.model, self.tokenizer


class FakeAsyncRunner:
    def __init__(self, *, delay: float = 0.0, mode: str = "normal") -> None:
        self.delay = delay
        self.mode = mode
        self.payloads: list[bytes] = []
        self.active = 0
        self.maximum_active = 0

    async def run_batch(self, payload: bytes) -> bytes:
        self.payloads.append(payload)
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            request = decode_flash_request(payload)
            rows = request["rows"]
            if self.mode == "missing":
                rows = rows[:-1]
            elif self.mode == "extra":
                rows = rows + [{"hash": "00" * 32, "score": 0.0}]
            elif self.mode == "fingerprint":
                return encode_flash_response(
                    [],
                    schema_version=WIRE_SCHEMA_VERSION,
                    scorer_fingerprint="drifted-fingerprint",
                )
            elif self.mode == "nonfinite":
                rows = [{"hash": row["hash"], "score": math.nan} for row in rows]
            else:
                rows = [
                    {"hash": row["hash"], "score": float(len(row["text"]))}
                    for row in rows
                ]
            return encode_flash_response(
                rows,
                schema_version=request["schema_version"],
                scorer_fingerprint=request["scorer_fingerprint"],
            )
        finally:
            self.active -= 1


class FlashWireCodecTests(unittest.TestCase):
    def test_request_carries_strict_model_config_and_hashes_as_lower_hex(self) -> None:
        config = LocalCrossEncoderConfig(query_format="Q: {query}")
        payload = encode_flash_request(
            "q-é",
            ((sha256_bytes("é漢字"), "é漢字"),),
            model_config=_wire_model_config(config),
            scorer_fingerprint=config.fingerprint,
        )

        decoded = decode_flash_request(payload)

        self.assertEqual(decoded["schema_version"], WIRE_SCHEMA_VERSION)
        self.assertEqual(decoded["scorer_fingerprint"], config.fingerprint)
        self.assertEqual(decoded["model_config"], _wire_model_config(config))
        self.assertEqual(decoded["query"], "q-é")
        self.assertEqual(
            decoded["rows"], [{"hash": sha256_bytes("é漢字").hex(), "text": "é漢字"}]
        )
        self.assertEqual(payload, payload.decode("utf-8").encode("utf-8"))
        self.assertLess(len(payload), FLASH_PAYLOAD_LIMIT_BYTES)

    def test_codecs_reject_unknown_fields_bad_model_config_hashes_and_nonfinite_scores(
        self,
    ) -> None:
        config = LocalCrossEncoderConfig()
        request = encode_flash_request(
            "query",
            ((sha256_bytes("text"), "text"),),
            model_config=_wire_model_config(config),
            scorer_fingerprint=config.fingerprint,
        )
        with self.assertRaises(FlashScoringError):
            decode_flash_request(request.replace(b'"query":"query"', b'"extra":1'))
        with self.assertRaises(FlashScoringError):
            decode_flash_request(
                request.replace(
                    f'"model_name":"{config.model_name}"'.encode(),
                    b'"model_name":1',
                )
            )
        with self.assertRaises(FlashScoringError):
            decode_flash_request(
                request.replace(sha256_bytes("text").hex().encode(), b"bad")
            )
        with self.assertRaises(FlashScoringError):
            decode_flash_response(
                b'{"schema_version":1,"scorer_fingerprint":"p","scores":'
                b'[{"hash":"' + sha256_bytes("text").hex().encode() + b'","score":NaN}]'
            )

    def test_model_config_schema_version_is_distinct_from_wire_schema(self) -> None:
        config = LocalCrossEncoderConfig(schema_version=2)
        payload = encode_flash_request(
            "query",
            ((sha256_bytes("text"), "text"),),
            model_config=_wire_model_config(config),
            scorer_fingerprint=config.fingerprint,
        )

        self.assertEqual(
            decode_flash_request(payload)["model_config"]["schema_version"], 2
        )

    def test_request_with_unpaired_surrogate_raises_flash_scoring_error(self) -> None:
        config = LocalCrossEncoderConfig()

        with self.assertRaises(FlashScoringError):
            encode_flash_request(
                "query",
                ((sha256_bytes("text"), chr(0xD800)),),
                model_config=_wire_model_config(config),
                scorer_fingerprint=config.fingerprint,
            )


class RunpodFlashConfigTests(unittest.TestCase):
    def test_transport_settings_do_not_change_model_fingerprint(self) -> None:
        policy = LocalCrossEncoderConfig(batch_size=2)
        base = RunpodFlashConfig(
            endpoint_id="endpoint-a",
            model_config=policy,
            workers=1,
            idle_timeout=60,
            request_timeout=30.0,
            concurrency=2,
        )
        transport_variant = replace(
            base,
            endpoint_id="endpoint-b",
            worker_name="other-worker",
            gpu="AMPERE_48",
            workers=4,
            idle_timeout=600,
            request_timeout=3.0,
            max_payload_bytes=12345,
            concurrency=8,
        )

        self.assertEqual(base.fingerprint, policy.fingerprint)
        self.assertEqual(base.fingerprint, transport_variant.fingerprint)
        self.assertEqual(base.effective_model_config, policy)
        self.assertNotEqual(
            base.fingerprint, replace(base, precision="float16").fingerprint
        )

    def test_transport_values_are_validated_without_runpod_imports(self) -> None:
        with self.assertRaises(ValueError):
            RunpodFlashConfig(endpoint_id="endpoint", worker_name="bad\nname")
        with self.assertRaises(ValueError):
            RunpodFlashConfig(endpoint_id="endpoint", max_payload_bytes=10_000_001)

    def test_idle_timeout_accepts_runpod_inclusive_integer_bounds_without_importing_runpod(
        self,
    ) -> None:
        with patch.dict(sys.modules, {"runpod_flash": None}):
            lower = RunpodFlashConfig(endpoint_id="endpoint", idle_timeout=1)
            upper = RunpodFlashConfig(endpoint_id="endpoint", idle_timeout=3600)

        self.assertEqual(lower.idle_timeout, 1)
        self.assertEqual(upper.idle_timeout, 3600)

    def test_idle_timeout_rejects_fractional_values(self) -> None:
        for value in (1.0, 1.5, 3599.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                RunpodFlashConfig(endpoint_id="endpoint", idle_timeout=value)

    def test_idle_timeout_rejects_values_outside_runpod_range(self) -> None:
        for value in (0, -1, 3601, True, "60", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                RunpodFlashConfig(endpoint_id="endpoint", idle_timeout=value)  # type: ignore[arg-type]


class RunpodFlashPassageScorerTests(unittest.TestCase):
    def test_adapter_and_worker_failures_do_not_emit_sensitive_literals(self) -> None:
        query = "query-sensitive-7f"
        text = "passage-sensitive-8g"
        content_hash = sha256_bytes(text).hex()
        credential = "Bearer credential-sensitive-9h"
        config = LocalCrossEncoderConfig()

        class FailingRunner:
            async def run_batch(self, payload: bytes) -> bytes:
                raise RuntimeError(f"{query} {text} {content_hash} {credential}")

        def failing_loader(model_config: object) -> object:
            raise RuntimeError(f"{query} {text} {content_hash} {credential}")

        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            scorer = RunpodFlashPassageScorer(
                RunpodFlashConfig(endpoint_id="fake", model_config=config),
                FailingRunner(),
            )
            try:
                with self.assertRaises(FlashScoringError):
                    scorer.score(query, (_passage(text),))
            finally:
                scorer.close()

            worker = build_flash_worker(config)(loader=failing_loader)
            with self.assertRaises(FlashScoringError):
                worker.score(_request_for_config(config, query=query, text=text))

        output = stdout.getvalue() + stderr.getvalue()
        for sensitive in (query, text, content_hash, credential):
            self.assertNotIn(sensitive, output)

    def test_batches_by_count_and_returns_exact_unique_hash_coverage(self) -> None:
        passages = (
            _passage("first", document_id="first"),
            _passage("first", document_id="duplicate"),
            _passage("second", document_id="second"),
            _passage("third", document_id="third"),
        )
        config = RunpodFlashConfig(endpoint_id="fake", batch_size=2)
        runner = FakeAsyncRunner()
        scorer = RunpodFlashPassageScorer(config, runner)
        self.addCleanup(scorer.close)

        scores = scorer.score("query", passages)

        self.assertEqual(
            scores,
            {
                passages[0].content_hash: 5.0,
                passages[2].content_hash: 6.0,
                passages[3].content_hash: 5.0,
            },
        )
        self.assertEqual(len(runner.payloads), 2)
        for payload in runner.payloads:
            self.assertEqual(
                decode_flash_request(payload)["scorer_fingerprint"], config.fingerprint
            )
            self.assertEqual(
                decode_flash_request(payload)["model_config"],
                _wire_model_config(config.effective_model_config),
            )

    def test_payloads_are_below_both_limits_and_oversized_rows_fail(self) -> None:
        passages = tuple(
            _passage("漢字-é" * count, document_id=str(count)) for count in range(1, 8)
        )
        config = RunpodFlashConfig(
            endpoint_id="fake", batch_size=99, max_payload_bytes=1200
        )
        runner = FakeAsyncRunner()
        scorer = RunpodFlashPassageScorer(config, runner)
        self.addCleanup(scorer.close)

        scorer.score("query-é", passages)

        self.assertGreater(len(runner.payloads), 1)
        self.assertTrue(all(len(payload) < 1200 for payload in runner.payloads))
        self.assertTrue(
            all(len(payload) < FLASH_PAYLOAD_LIMIT_BYTES for payload in runner.payloads)
        )
        with self.assertRaises(FlashScoringError):
            scorer.score("q", (_passage("x" * 500, document_id="too-large"),))

    def test_payload_limit_measures_the_final_runpod_input_body(self) -> None:
        text = "payload-envelope-check"
        policy = LocalCrossEncoderConfig()
        request = encode_flash_request(
            "query",
            ((sha256_bytes(text), text),),
            model_config=_wire_model_config(policy),
            scorer_fingerprint=policy.fingerprint,
        )
        runpod_body = json.dumps(
            {"input": {"request": decode_flash_request(request)}},
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        self.assertGreater(len(runpod_body), len(request))

        config = RunpodFlashConfig(
            endpoint_id="fake",
            model_config=policy,
            max_payload_bytes=len(runpod_body),
        )
        scorer = RunpodFlashPassageScorer(config, FakeAsyncRunner())
        self.addCleanup(scorer.close)

        with self.assertRaises(FlashScoringError):
            scorer.score("query", (_passage(text),))

    def test_concurrency_is_bounded_and_all_batches_complete(self) -> None:
        passages = tuple(_passage(f"text-{i}", document_id=str(i)) for i in range(8))
        runner = FakeAsyncRunner(delay=0.02)
        scorer = RunpodFlashPassageScorer(
            RunpodFlashConfig(endpoint_id="fake", batch_size=1, concurrency=2), runner
        )
        self.addCleanup(scorer.close)

        self.assertEqual(len(scorer.score("query", passages)), 8)
        self.assertLessEqual(runner.maximum_active, 2)

    def test_failed_batch_cancels_and_awaits_siblings(self) -> None:
        class QuiescingFailureRunner:
            def __init__(self) -> None:
                self.dispatches = 0
                self.cancelled = 0
                self.active = 0

            async def run_batch(self, payload: bytes) -> bytes:
                self.dispatches += 1
                self.active += 1
                try:
                    if self.dispatches == 1:
                        raise RuntimeError("transport failure")
                    try:
                        await asyncio.sleep(60.0)
                    except asyncio.CancelledError:
                        self.cancelled += 1
                        raise
                    raise AssertionError("sibling was not cancelled")
                finally:
                    self.active -= 1

        runner = QuiescingFailureRunner()
        passages = tuple(_passage(f"text-{i}", document_id=str(i)) for i in range(4))
        scorer = RunpodFlashPassageScorer(
            RunpodFlashConfig(endpoint_id="fake", batch_size=1, concurrency=2), runner
        )
        self.addCleanup(scorer.close)

        with self.assertRaises(FlashScoringError):
            scorer.score("query", passages)
        self.assertLessEqual(runner.dispatches, 2)
        self.assertGreaterEqual(runner.cancelled, 1)
        self.assertEqual(runner.active, 0)

    def test_timeout_and_response_fingerprint_or_coverage_drift_fail(self) -> None:
        passage = _passage("passage")
        timeout_scorer = RunpodFlashPassageScorer(
            RunpodFlashConfig(endpoint_id="fake", request_timeout=0.001),
            FakeAsyncRunner(delay=0.1),
        )
        self.addCleanup(timeout_scorer.close)
        with self.assertRaises(FlashScoringError):
            timeout_scorer.score("query", (passage,))

        for mode in ("fingerprint", "missing", "extra", "nonfinite"):
            with self.subTest(mode=mode):
                scorer = RunpodFlashPassageScorer(
                    RunpodFlashConfig(endpoint_id="fake"),
                    FakeAsyncRunner(mode=mode),
                )
                self.addCleanup(scorer.close)
                with self.assertRaises(FlashScoringError):
                    scorer.score(
                        "query", (passage, _passage("second", document_id="2"))
                    )

    def test_endpoint_object_can_be_injected_and_is_run_then_waited(self) -> None:
        calls: list[tuple[str, object]] = []
        config = RunpodFlashConfig(endpoint_id="endpoint", request_timeout=3.0)

        class Job:
            def __init__(self, request: dict[str, object]) -> None:
                self.output = encode_flash_response(
                    ({"hash": sha256_bytes("text"), "score": 4.0},),
                    schema_version=request["schema_version"],
                    scorer_fingerprint=request["scorer_fingerprint"],
                )

            async def wait(self, timeout: float | None = None) -> "Job":
                calls.append(("wait", timeout))
                return self

        class Endpoint:
            async def run(self, job_input: dict[str, object]) -> Job:
                calls.append(("run", job_input))
                return Job(job_input["request"])

        request = encode_flash_request(
            "query",
            ((sha256_bytes("text"), "text"),),
            model_config=_wire_model_config(config.effective_model_config),
            scorer_fingerprint=config.fingerprint,
        )
        runner = FlashEndpointBatchRunner(config, endpoint=Endpoint())
        self.addCleanup(runner.close)

        response = runner.run_batch_sync(request)

        self.assertEqual(decode_flash_response(response)["scores"][0]["score"], 4.0)
        self.assertEqual(calls[0][0], "run")
        self.assertEqual(calls[0][1], {"request": decode_flash_request(request)})
        self.assertEqual(calls[1], ("wait", 3.0))

    def test_endpoint_adapter_matches_flash_keyword_dispatch(self) -> None:
        policy = LocalCrossEncoderConfig()
        config = RunpodFlashConfig(endpoint_id="endpoint", model_config=policy)
        worker = build_flash_worker(policy)(
            loader=CountingLoader(FakeModel(), BatchEncodingTokenizer())
        )
        request = encode_flash_request(
            "query",
            ((sha256_bytes("text"), "text"),),
            model_config=_wire_model_config(policy),
            scorer_fingerprint=config.fingerprint,
        )
        received: list[dict[str, object]] = []

        class Job:
            def __init__(self, output: bytes) -> None:
                self.output = output

            async def wait(self, timeout: float | None = None) -> "Job":
                return self

        class Endpoint:
            async def run(self, job_input: dict[str, object]) -> Job:
                received.append(job_input)
                response = worker.score(**job_input)
                return Job(
                    encode_flash_response(
                        response["scores"],
                        schema_version=response["schema_version"],
                        scorer_fingerprint=response["scorer_fingerprint"],
                    )
                )

        runner = FlashEndpointBatchRunner(config, endpoint=Endpoint())
        self.addCleanup(runner.close)

        response = decode_flash_response(runner.run_batch_sync(request))

        self.assertEqual(
            response["scores"], [{"hash": sha256_bytes("text").hex(), "score": 4.0}]
        )
        self.assertEqual(received, [{"request": decode_flash_request(request)}])

    def test_endpoint_run_or_wait_failure_cancels_job(self) -> None:
        config = RunpodFlashConfig(endpoint_id="endpoint", request_timeout=0.01)
        request = encode_flash_request(
            "query",
            ((sha256_bytes("text"), "text"),),
            model_config=_wire_model_config(config.effective_model_config),
            scorer_fingerprint=config.fingerprint,
        )
        cancelled: list[bool] = []

        class Job:
            status = "RUNNING"
            output = None

            async def wait(self, timeout: float | None = None) -> "Job":
                raise RuntimeError("wait failed")

            async def cancel(self) -> None:
                cancelled.append(True)

        class Endpoint:
            async def run(self, request: dict[str, object]) -> Job:
                return Job()

        runner = FlashEndpointBatchRunner(config, endpoint=Endpoint())
        self.addCleanup(runner.close)
        with self.assertRaises(FlashScoringError):
            runner.run_batch_sync(request)
        self.assertEqual(cancelled, [True])


class StaticFlashWorkerTests(unittest.TestCase):
    def test_worker_fingerprint_matches_rotated_local_identity(self) -> None:
        policy = LocalCrossEncoderConfig()

        self.assertTrue(policy.fingerprint.startswith("local-cross-encoder-v3/"))
        self.assertEqual(
            FlashPassageWorker._fingerprint_for_values(_wire_model_config(policy)),
            policy.fingerprint,
        )

    def test_static_default_loader_constructs_direct_cross_encoder(self) -> None:
        policy = LocalCrossEncoderConfig()
        source = inspect.getsource(FlashPassageWorker)
        namespace = {"__name__": "flash_worker_isolated"}
        constructor_calls: list[dict[str, object]] = []
        dtype = object()

        class FakeTokenizer:
            def __call__(
                self, query: str, text: str, **kwargs: object
            ) -> dict[str, object]:
                return {"input_ids": [1, 2, 3], "length": [3]}

        class FakeModel:
            def __init__(self, *args: object, **kwargs: object) -> None:
                constructor_calls.append({"args": args, **kwargs})
                self.tokenizer = FakeTokenizer()

            def predict(self, pairs: object, **kwargs: object) -> list[float]:
                return [1.0]

        fake_sentence_transformers = types.ModuleType("sentence_transformers")
        fake_sentence_transformers.CrossEncoder = FakeModel  # type: ignore[attr-defined]
        fake_sentence_transformers_modules = types.ModuleType(
            "sentence_transformers.sentence_transformer.modules"
        )

        class FakeTransformer:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

        fake_sentence_transformers_modules.Transformer = FakeTransformer  # type: ignore[attr-defined]
        fake_torch = types.ModuleType("torch")
        fake_torch.float32 = dtype  # type: ignore[attr-defined]
        fake_torch.float16 = object()  # type: ignore[attr-defined]
        fake_torch.bfloat16 = object()  # type: ignore[attr-defined]

        with patch.dict(
            sys.modules,
            {
                "trec_ragtime": None,
                "sentence_transformers": fake_sentence_transformers,
                "sentence_transformers.sentence_transformer.modules": fake_sentence_transformers_modules,
                "torch": fake_torch,
            },
            clear=False,
        ):
            exec(compile(source, "<flash-worker>", "exec"), namespace)
            isolated_worker = namespace["FlashPassageWorker"]
            model, tokenizer = isolated_worker._default_loader(
                _wire_model_config(policy)
            )

        self.assertIsInstance(model, FakeModel)
        self.assertIsInstance(tokenizer, FakeTokenizer)
        self.assertEqual(
            constructor_calls,
            [
                {
                    "args": (policy.model_name,),
                    "revision": policy.revision,
                    "device": policy.effective_device,
                    "max_length": policy.max_length,
                    "model_kwargs": {"torch_dtype": dtype},
                }
            ],
        )

    def test_static_worker_source_executes_without_project_imports(self) -> None:
        policy = LocalCrossEncoderConfig()
        source = inspect.getsource(FlashPassageWorker)
        namespace = {"__name__": "flash_worker_isolated"}
        model = FakeModel()
        tokenizer = BatchEncodingTokenizer()
        load_calls: list[object] = []

        def loader(config: object) -> tuple[FakeModel, BatchEncodingTokenizer]:
            load_calls.append(config)
            return model, tokenizer

        with patch.dict(sys.modules, {"trec_ragtime": None}, clear=False):
            exec(compile(source, "<flash-worker>", "exec"), namespace)
        isolated_worker = namespace["FlashPassageWorker"]
        worker = isolated_worker(loader=loader)
        response = worker.score(_request_for_config(policy))

        self.assertEqual(response["scorer_fingerprint"], policy.fingerprint)
        self.assertEqual(model.predict_calls[0][0], [("query", "text")])
        self.assertEqual(len(load_calls), 1)

    def test_constructor_is_cold_and_first_valid_request_loads_once(self) -> None:
        policy = LocalCrossEncoderConfig(max_length=100)
        loader = CountingLoader(FakeModel(), BatchEncodingTokenizer())
        worker = build_flash_worker(policy)(loader=loader)

        self.assertEqual(len(loader.calls), 0)
        request = _request_for_config(policy)
        worker.score(request)
        worker.score(request)
        self.assertEqual(len(loader.calls), 1)
        drifted = replace(policy, query_policy="drifted")
        with self.assertRaises(FlashScoringError):
            worker.score(_request_for_config(drifted))
        self.assertEqual(len(loader.calls), 1)

    def test_concurrent_cold_start_binds_one_identity_before_prediction(self) -> None:
        policy_a = LocalCrossEncoderConfig(
            revision="revision-a", tokenizer_revision="revision-a"
        )
        policy_b = replace(
            policy_a, revision="revision-b", tokenizer_revision="revision-b"
        )
        validated = threading.Barrier(2)
        load_started = threading.Event()
        release_load = threading.Event()
        load_lock = threading.Lock()
        loaded_revisions: list[object] = []
        loaded_models: list[FakeModel] = []

        def loader(
            config: dict[str, object],
        ) -> tuple[FakeModel, BatchEncodingTokenizer]:
            with load_lock:
                loaded_revisions.append(config["revision"])
            model = FakeModel()
            loaded_models.append(model)
            load_started.set()
            if not release_load.wait(timeout=2.0):
                raise AssertionError("test loader was not released")
            return model, BatchEncodingTokenizer()

        worker = build_flash_worker(policy_a)(loader=loader)
        original_validate = FlashPassageWorker._validate_request

        def synchronized_validate(
            cls: type[FlashPassageWorker], request: object
        ) -> tuple[dict[str, object], str, str, list[tuple[bytes, str]]]:
            result = original_validate(request)
            try:
                validated.wait(timeout=2.0)
            except threading.BrokenBarrierError as error:
                raise AssertionError("requests did not validate together") from error
            return result

        outcomes: list[tuple[str, object]] = []
        outcomes_lock = threading.Lock()

        def score(request: dict[str, object]) -> None:
            try:
                response = worker.score(request)
            except Exception as error:
                outcome = ("error", error)
            else:
                outcome = ("success", response)
            with outcomes_lock:
                outcomes.append(outcome)

        with patch.object(
            FlashPassageWorker,
            "_validate_request",
            classmethod(synchronized_validate),
        ):
            threads = [
                threading.Thread(
                    target=score,
                    args=(_request_for_config(policy_a),),
                ),
                threading.Thread(
                    target=score,
                    args=(_request_for_config(policy_b),),
                ),
            ]
            for thread in threads:
                thread.start()
            self.assertTrue(load_started.wait(timeout=2.0))
            release_load.set()
            for thread in threads:
                thread.join(timeout=2.0)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(len(loaded_revisions), 1)
        self.assertEqual(
            sum(outcome == "success" for outcome, _ in outcomes),
            1,
        )
        self.assertEqual(
            sum(outcome == "error" for outcome, _ in outcomes),
            1,
        )
        self.assertEqual(sum(len(model.predict_calls) for model in loaded_models), 1)
        successful_response = next(
            response for outcome, response in outcomes if outcome == "success"
        )
        self.assertIn(
            successful_response["scorer_fingerprint"],
            {policy_a.fingerprint, policy_b.fingerprint},
        )

    def test_falsey_callable_loader_is_used(self) -> None:
        policy = LocalCrossEncoderConfig()
        model = FakeModel()
        tokenizer = BatchEncodingTokenizer()

        class FalseyLoader:
            def __init__(self) -> None:
                self.calls = 0

            def __bool__(self) -> bool:
                return False

            def __call__(
                self, config: object
            ) -> tuple[FakeModel, BatchEncodingTokenizer]:
                self.calls += 1
                return model, tokenizer

        loader = FalseyLoader()
        default_model = FakeModel(multiplier=3.0)
        default_tokenizer = BatchEncodingTokenizer()
        with patch.object(
            FlashPassageWorker,
            "_default_loader",
            return_value=(default_model, default_tokenizer),
        ):
            worker = build_flash_worker(policy)(loader=loader)
            worker.score(_request_for_config(policy))

        self.assertEqual(loader.calls, 1)
        self.assertEqual(default_model.predict_calls, [])

    def test_failed_prediction_binds_first_fingerprint_before_later_request(
        self,
    ) -> None:
        policy_a = LocalCrossEncoderConfig(query_policy="policy-a")
        policy_b = replace(policy_a, query_policy="policy-b")
        tokenizer = BatchEncodingTokenizer()

        class FailingModel(FakeModel):
            def predict(
                self, pairs: list[tuple[str, str]], **kwargs: object
            ) -> list[float]:
                self.predict_calls.append((pairs, kwargs))
                raise RuntimeError("prediction failed")

        model = FailingModel()
        loader = CountingLoader(model, tokenizer)
        worker = build_flash_worker(policy_a)(loader=loader)

        with self.assertRaises(FlashScoringError):
            worker.score(_request_for_config(policy_a))
        with self.assertRaises(FlashScoringError):
            worker.score(_request_for_config(policy_b))

        self.assertEqual(len(loader.calls), 1)
        self.assertEqual(len(model.predict_calls), 1)

    def test_worker_validates_exact_request_model_config_and_pair_ceiling(self) -> None:
        policy = LocalCrossEncoderConfig(max_length=10)
        loader = CountingLoader(FakeModel(), BatchEncodingTokenizer())
        worker = build_flash_worker(policy)(loader=loader)
        request = _request_for_config(policy)

        with self.assertRaises(FlashScoringError):
            worker.score({**request, "extra": True})
        with self.assertRaises(FlashScoringError):
            worker.score({**request, "scorer_fingerprint": "wrong"})
        with self.assertRaises(FlashScoringError):
            worker.score(_request_for_config(policy, text="overflow"))
        self.assertEqual(len(loader.calls), 1)

    def test_isolated_worker_rejects_template_overflow_before_prediction(
        self,
    ) -> None:
        policy = LocalCrossEncoderConfig(max_length=12)
        source = inspect.getsource(FlashPassageWorker)
        namespace = {"__name__": "flash_worker_isolated"}
        model = FakeModel()
        tokenizer = ChatTemplateBatchEncodingTokenizer(template_overhead=10)

        with patch.dict(sys.modules, {"trec_ragtime": None}, clear=False):
            exec(compile(source, "<flash-worker>", "exec"), namespace)
        isolated_worker = namespace["FlashPassageWorker"]
        worker = isolated_worker(loader=lambda config: (model, tokenizer))

        with self.assertRaisesRegex(ValueError, "overflow"):
            worker.score(_request_for_config(policy, text="abc"))

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

    def test_worker_accepts_batch_encoding_mapping_and_returns_finite_hash_scores(
        self,
    ) -> None:
        policy = LocalCrossEncoderConfig(batch_size=2, max_length=100)
        model = FakeModel(multiplier=2.0)
        tokenizer = BatchEncodingTokenizer()
        worker = build_flash_worker(policy)(loader=CountingLoader(model, tokenizer))
        response = worker.score(
            {
                **_request_for_config(policy),
                "rows": [
                    {"hash": sha256_bytes("a").hex(), "text": "a"},
                    {"hash": sha256_bytes("bb").hex(), "text": "bb"},
                ],
            }
        )

        self.assertEqual(
            response["scores"],
            sorted(
                [
                    {"hash": sha256_bytes("a").hex(), "score": 2.0},
                    {"hash": sha256_bytes("bb").hex(), "score": 4.0},
                ],
                key=lambda row: row["hash"],
            ),
        )
        self.assertEqual(len(model.predict_calls), 1)
        self.assertIs(tokenizer.calls[0][2]["truncation"], False)
        self.assertEqual(tokenizer.calls[0][2]["max_length"], 100)

    def test_worker_and_local_scorer_have_fake_raw_logit_parity(self) -> None:
        policy = LocalCrossEncoderConfig(
            batch_size=1, max_length=100, query_format="Q: {query}"
        )
        passage = (_passage("a", document_id="a"), _passage("longer", document_id="b"))
        local = LocalCrossEncoderScorer(
            policy,
            loader=lambda config: (FakeModel(multiplier=2.0), BatchEncodingTokenizer()),
        )
        remote_worker = build_flash_worker(policy)(
            loader=lambda config: (FakeModel(multiplier=2.0), BatchEncodingTokenizer())
        )
        runner = RunpodFlashPassageScorer(
            RunpodFlashConfig(endpoint_id="fake", model_config=policy),
            _WorkerWireRunner(remote_worker),
        )
        self.addCleanup(runner.close)

        self.assertEqual(local.score("query", passage), runner.score("query", passage))


class _WorkerWireRunner:
    def __init__(self, worker: object) -> None:
        self.worker = worker

    async def run_batch(self, payload: bytes) -> bytes:
        request = decode_flash_request(payload)
        response = self.worker.score(request)
        return encode_flash_response(
            response["scores"],
            schema_version=response["schema_version"],
            scorer_fingerprint=response["scorer_fingerprint"],
        )


class FlashDecorationTests(unittest.TestCase):
    def test_build_is_lazy_and_endpoint_decoration_has_pinned_kwargs(self) -> None:
        policy = LocalCrossEncoderConfig()
        config = RunpodFlashConfig(
            endpoint_id="endpoint",
            model_config=policy,
            worker_name="worker-name",
            gpu="GPU_GROUP",
            workers=3,
            idle_timeout=123,
        )
        calls: list[object] = []

        def endpoint_factory(
            **kwargs: object,
        ) -> Callable[[type[object]], type[object]]:
            calls.append(kwargs)

            def decorate(worker: type[object]) -> type[object]:
                calls.append(worker)
                return worker

            return decorate

        worker = build_flash_worker(config)
        self.assertIs(worker, FlashPassageWorker)
        self.assertEqual(calls, [])

        decorated = build_flash_worker(
            config,
            endpoint_factory=endpoint_factory,
            gpu_resolver=lambda name: f"resolved:{name}",
        )

        self.assertIs(decorated, FlashPassageWorker)
        self.assertEqual(calls[0]["name"], "worker-name")
        self.assertEqual(calls[0]["gpu"], "resolved:GPU_GROUP")
        self.assertEqual(calls[0]["workers"], 3)
        self.assertEqual(calls[0]["idle_timeout"], 123.0)
        self.assertEqual(
            calls[0]["dependencies"],
            [
                f"sentence-transformers=={policy.sentence_transformers_version}",
                f"transformers=={policy.transformers_version}",
                f"tokenizers=={policy.tokenizers_version}",
                f"torch=={policy.torch_version}",
                "msgspec==0.21.1",
            ],
        )

    def test_package_import_and_local_build_do_not_resolve_runpod(self) -> None:
        with patch.dict(sys.modules, {"runpod_flash": None}):
            worker = build_flash_worker(LocalCrossEncoderConfig())
        self.assertIs(worker, FlashPassageWorker)


if __name__ == "__main__":
    unittest.main()
