from __future__ import annotations

import asyncio
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.metadata
import json
import inspect
from io import StringIO
import math
import msgspec
import os
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from collections import UserDict
from collections.abc import Callable
from dataclasses import replace
from unittest.mock import PropertyMock, patch

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
import trec_ragtime.flash_passage_scorer as flash_passage_scorer
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


def _cache_telemetry(
    row_count: int,
    *,
    status: str = "miss_written",
    batch_count: int = 1,
) -> dict[str, object]:
    return {
        "status": status,
        "object_key": "passage-score-requests/v1/fake/00/" + "0" * 64 + ".mpk",
        "row_count": row_count,
        "batch_count": batch_count,
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


class WorkerCacheStore:
    """Tiny strict store fake for the worker-owned cache seam."""

    def __init__(self, *, write_result: bool = True, reread: object = "stored") -> None:
        self.objects: dict[str, bytes] = {}
        self.read_validated_calls: list[str] = []
        self.write_calls: list[tuple[str, bytes]] = []
        self.evict_local_calls: list[str] = []
        self.write_result = write_result
        self.reread = reread

    def read_validated(
        self, key: str, decoder: Callable[[bytes], object]
    ) -> object | None:
        self.read_validated_calls.append(key)
        if self.reread != "stored":
            return self.reread
        payload = self.objects.get(key)
        return None if payload is None else decoder(payload)

    def write(self, key: str, payload: bytes) -> bool:
        self.write_calls.append((key, payload))
        if self.write_result:
            self.objects[key] = payload
        return self.write_result

    def evict_local(self, key: str) -> None:
        self.evict_local_calls.append(key)


class DirectStoreApiFake:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def bucket_info(self, bucket_id: str) -> object:
        return types.SimpleNamespace(private=True)


class DirectStoreHandleFake:
    def __init__(self, filesystem: "DirectStoreFilesystemFake", path: str, mode: str) -> None:
        self.filesystem = filesystem
        self.path = path
        self.mode = mode
        self.payload = bytearray()

    def __enter__(self) -> "DirectStoreHandleFake":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if exc_type is None and "w" in self.mode:
            key = self.path.removeprefix("hf://")
            payload = bytes(self.payload)
            self.filesystem.remote[key] = payload
            candidate = self.filesystem.cache_storage / hashlib.sha256(
                key.encode("utf-8")
            ).hexdigest()
            candidate.write_bytes(payload)

    def read(self) -> bytes:
        return self.filesystem.remote[self.path.removeprefix("hf://")]

    def write(self, payload: bytes) -> int:
        self.payload.extend(payload)
        return len(payload)


class DirectStoreFilesystemFake:
    remote: dict[str, bytes] = {}

    def __init__(self, cache_storage: str) -> None:
        self.cache_storage = Path(cache_storage)
        self.remote = type(self).remote

    def open(self, path: str, mode: str) -> DirectStoreHandleFake:
        key = path.removeprefix("hf://")
        if "r" in mode and key not in self.remote:
            raise FileNotFoundError(key)
        return DirectStoreHandleFake(self, path, mode)


def _request_for_rows(
    config: LocalCrossEncoderConfig,
    rows: list[tuple[str, str]],
    *,
    query: str = "query",
) -> dict[str, object]:
    return {
        "schema_version": WIRE_SCHEMA_VERSION,
        "scorer_fingerprint": config.fingerprint,
        "model_config": _wire_model_config(config),
        "query": query,
        "rows": [
            {"hash": sha256_bytes(text).hex(), "text": text} for _, text in rows
        ],
    }


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
                    [{"hash": request["rows"][0]["hash"], "score": 0.0}],
                    schema_version=WIRE_SCHEMA_VERSION,
                    scorer_fingerprint="drifted-fingerprint",
                    cache=_cache_telemetry(1),
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
                cache=_cache_telemetry(len(rows)),
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

    def test_response_requires_strict_cache_telemetry(self) -> None:
        row = {"hash": sha256_bytes("text").hex(), "score": 1.0}
        cache = {
            "status": "hit",
            "object_key": "passage-score-requests/v1/fake/00/" + "0" * 64 + ".mpk",
            "row_count": 1,
            "batch_count": 0,
        }
        payload = encode_flash_response(
            [row], scorer_fingerprint="fingerprint", cache=cache
        )
        self.assertEqual(
            decode_flash_response(payload),
            {
                "schema_version": 1,
                "scorer_fingerprint": "fingerprint",
                "scores": [row],
                "cache": cache,
            },
        )
        decoded = json.loads(payload)
        for invalid_cache in (
            {**cache, "extra": 1},
            {**cache, "status": "unknown"},
            {**cache, "object_key": "../unsafe"},
            {**cache, "object_key": "safe?query"},
            {**cache, "object_key": "safe#fragment"},
            {**cache, "row_count": 2},
            {**cache, "batch_count": 1},
            {**cache, "row_count": 1.0},
            {**cache, "batch_count": -1},
            {**cache, "row_count": 1, "status": "hit\nunsafe"},
        ):
            with self.subTest(invalid_cache=invalid_cache):
                candidate = dict(decoded)
                candidate["cache"] = invalid_cache
                with self.assertRaises(FlashScoringError):
                    decode_flash_response(candidate)

        with self.assertRaises(FlashScoringError):
            encode_flash_response([row], scorer_fingerprint="fingerprint")

        for extra_field in ("extra",):
            candidate = dict(decoded)
            candidate[extra_field] = 1
            with self.assertRaises(FlashScoringError):
                decode_flash_response(candidate)

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
    def test_worker_tuple_range_and_max_concurrency_reach_endpoint_factory(
        self,
    ) -> None:
        config = RunpodFlashConfig(
            endpoint_id="endpoint",
            workers=(0, 2),
            max_concurrency=3,
        )
        calls: list[dict[str, object]] = []

        def endpoint_factory(
            **kwargs: object,
        ) -> Callable[[type[object]], type[object]]:
            calls.append(kwargs)
            return lambda worker: worker

        build_flash_worker(
            config,
            endpoint_factory=endpoint_factory,
            gpu_resolver=lambda name: name,
        )

        self.assertEqual(calls[0]["workers"], (0, 2))
        self.assertEqual(calls[0]["max_concurrency"], 3)

    def test_worker_tuple_range_is_validated(self) -> None:
        for workers in ((1,), (2, 1), (-1, 1), (0, 0), (0, 1.0), True):
            with self.subTest(workers=workers), self.assertRaises(ValueError):
                RunpodFlashConfig(  # type: ignore[arg-type]
                    endpoint_id="endpoint", workers=workers
                )

    def test_max_concurrency_must_be_positive_integer(self) -> None:
        for value in (0, -1, 1.0, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                RunpodFlashConfig(  # type: ignore[arg-type]
                    endpoint_id="endpoint", max_concurrency=value
                )

    def test_worker_range_must_scale_to_zero(self) -> None:
        with self.assertRaisesRegex(ValueError, "minimum must be zero"):
            RunpodFlashConfig(endpoint_id="endpoint", workers=(1, 2))

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
    def test_one_logical_request_handles_65_passages_without_model_batch_waves(
        self,
    ) -> None:
        passages = tuple(
            _passage(f"text-{index}", document_id=str(index)) for index in range(65)
        )
        config = RunpodFlashConfig(endpoint_id="fake", batch_size=32)
        runner = FakeAsyncRunner()
        scorer = RunpodFlashPassageScorer(config, runner)
        self.addCleanup(scorer.close)

        scores = scorer.score("query", passages)

        self.assertEqual(len(scores), 65)
        self.assertEqual(len(runner.payloads), 1)
        self.assertEqual(len(decode_flash_request(runner.payloads[0])["rows"]), 65)

    def test_wrapped_request_at_10_mb_fails_without_splitting(self) -> None:
        text = "x" * (FLASH_PAYLOAD_LIMIT_BYTES + 100)
        config = RunpodFlashConfig(endpoint_id="fake", batch_size=32)
        runner = FakeAsyncRunner()
        scorer = RunpodFlashPassageScorer(config, runner)
        self.addCleanup(scorer.close)

        with self.assertRaises(FlashScoringError):
            scorer.score("query", (_passage(text),))
        self.assertEqual(runner.payloads, [])

    def test_build_request_encodes_and_measures_one_wrapped_body(self) -> None:
        config = RunpodFlashConfig(endpoint_id="fake", batch_size=7)
        unique = {sha256_bytes(f"text-{index}"): f"text-{index}" for index in range(7)}
        original_request_body = flash_passage_scorer._runpod_request_body
        request_body_calls = 0

        def count_request_body_builds(payload: bytes) -> bytes:
            nonlocal request_body_calls
            request_body_calls += 1
            return original_request_body(payload)

        with patch.object(
            flash_passage_scorer,
            "_runpod_request_body",
            count_request_body_builds,
        ):
            payload = RunpodFlashPassageScorer._build_request("query", unique, config)

        self.assertEqual(request_body_calls, 1)
        self.assertEqual(len(decode_flash_request(payload)["rows"]), len(unique))

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

            worker = build_flash_worker(config)(
                loader=failing_loader, cache_store=WorkerCacheStore()
            )
            with self.assertRaises(ValueError):
                worker.score(_request_for_config(config, query=query, text=text))

        output = stdout.getvalue() + stderr.getvalue()
        for sensitive in (query, text, content_hash, credential):
            self.assertNotIn(sensitive, output)

    def test_one_request_returns_exact_unique_hash_coverage(self) -> None:
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
        self.assertEqual(len(runner.payloads), 1)
        for payload in runner.payloads:
            self.assertEqual(
                decode_flash_request(payload)["scorer_fingerprint"], config.fingerprint
            )
            self.assertEqual(
                decode_flash_request(payload)["model_config"],
                _wire_model_config(config.effective_model_config),
            )

    def test_one_request_obeys_payload_limit_without_splitting(self) -> None:
        passages = tuple(
            _passage("漢字-é" * count, document_id=str(count)) for count in range(1, 8)
        )
        config = RunpodFlashConfig(
            endpoint_id="fake", batch_size=99, max_payload_bytes=1200
        )
        runner = FakeAsyncRunner()
        scorer = RunpodFlashPassageScorer(config, runner)
        self.addCleanup(scorer.close)

        with self.assertRaises(FlashScoringError):
            scorer.score("query-é", passages)
        self.assertEqual(runner.payloads, [])
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
        self.assertEqual(runner.maximum_active, 1)
        self.assertEqual(len(runner.payloads), 1)

    def test_failed_request_is_cancelled_and_awaited(self) -> None:
        class QuiescingFailureRunner:
            def __init__(self) -> None:
                self.dispatches = 0
                self.cancelled = 0
                self.active = 0

            async def run_batch(self, payload: bytes) -> bytes:
                self.dispatches += 1
                self.active += 1
                try:
                    await asyncio.sleep(60.0)
                except asyncio.CancelledError:
                    self.cancelled += 1
                    raise
                finally:
                    self.active -= 1

        runner = QuiescingFailureRunner()
        passages = tuple(_passage(f"text-{i}", document_id=str(i)) for i in range(4))
        scorer = RunpodFlashPassageScorer(
            RunpodFlashConfig(
                endpoint_id="fake",
                batch_size=1,
                concurrency=2,
                request_timeout=0.01,
            ),
            runner,
        )
        self.addCleanup(scorer.close)

        with self.assertRaises(FlashScoringError):
            scorer.score("query", passages)
        self.assertEqual(runner.dispatches, 1)
        self.assertGreaterEqual(runner.cancelled, 1)
        self.assertEqual(runner.active, 0)

    def test_cancelled_batch_runner_is_wrapped_as_flash_scoring_error(self) -> None:
        class QuiescingCancellationRunner:
            def __init__(self) -> None:
                self.dispatches = 0
                self.cancelled = 0
                self.completed = 0

            async def run_batch(self, payload: bytes) -> bytes:
                self.dispatches += 1
                self.cancelled += 1
                raise asyncio.CancelledError

        runner = QuiescingCancellationRunner()
        passages = tuple(_passage(f"text-{index}") for index in range(2))
        scorer = RunpodFlashPassageScorer(
            RunpodFlashConfig(
                endpoint_id="fake",
                batch_size=1,
                concurrency=2,
                request_timeout=0.05,
            ),
            runner,
        )
        self.addCleanup(scorer.close)

        with self.assertRaisesRegex(FlashScoringError, "Flash batch cancelled"):
            scorer.score("query", passages)
        self.assertEqual(runner.dispatches, 1)
        self.assertEqual(runner.cancelled, 1)
        self.assertEqual(runner.completed, 0)

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
                    cache=_cache_telemetry(1),
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
            loader=CountingLoader(FakeModel(), BatchEncodingTokenizer()),
            cache_store=WorkerCacheStore(),
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
                        cache=_cache_telemetry(len(response["scores"])),
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


class FlashEndpointBatchRunnerTests(unittest.TestCase):
    def test_endpoint_runner_submits_one_job_and_waits_for_output(self) -> None:
        config = RunpodFlashConfig(endpoint_id="endpoint", request_timeout=1.0)
        request = encode_flash_request(
            "query",
            ((sha256_bytes("text"), "text"),),
            model_config=_wire_model_config(config.effective_model_config),
            scorer_fingerprint=config.fingerprint,
        )
        calls: list[tuple[str, object]] = []

        class Job:
            output = encode_flash_response(
                ({"hash": sha256_bytes("text"), "score": 4.0},),
                scorer_fingerprint=config.fingerprint,
                cache=_cache_telemetry(1),
            )

            async def wait(self, timeout: float | None = None) -> "Job":
                calls.append(("wait", timeout))
                return self

        class Endpoint:
            async def run(self, job_input: dict[str, object]) -> Job:
                calls.append(("run", job_input))
                return Job()

        runner = FlashEndpointBatchRunner(config, endpoint=Endpoint())
        self.addCleanup(runner.close)

        response = decode_flash_response(runner.run_batch_sync(request))

        self.assertEqual(response["scores"][0]["score"], 4.0)
        self.assertEqual(calls[0], ("run", {"request": decode_flash_request(request)}))
        self.assertEqual(calls[1], ("wait", 1.0))


class WorkerCacheIdentityTests(unittest.TestCase):
    def test_hash_order_does_not_change_identity(self) -> None:
        left = FlashPassageWorker._cache_identity(
            "scorer-v1", "query", [b"a" * 32, b"b" * 32]
        )
        right = FlashPassageWorker._cache_identity(
            "scorer-v1", "query", [b"b" * 32, b"a" * 32]
        )

        self.assertEqual(left, right)
        self.assertEqual(
            FlashPassageWorker._cache_object_key(left),
            FlashPassageWorker._cache_object_key(right),
        )

    def test_identity_rotates_for_every_request_dimension(self) -> None:
        base = FlashPassageWorker._cache_identity(
            "scorer-v1", "query", [b"a" * 32]
        )
        variants = (
            FlashPassageWorker._cache_identity(
                "scorer-v2", "query", [b"a" * 32]
            ),
            FlashPassageWorker._cache_identity(
                "scorer-v1", "query!", [b"a" * 32]
            ),
            FlashPassageWorker._cache_identity(
                "scorer-v1", "query", [b"b" * 32]
            ),
        )

        self.assertTrue(all(value != base for value in variants))
        self.assertEqual(base[0], FlashPassageWorker.CACHE_SCHEMA_VERSION)
        self.assertEqual(base[1], FlashPassageWorker.CACHE_BEHAVIOR_VERSION)

    def test_schema_and_behavior_versions_rotate_identity(self) -> None:
        original_schema = FlashPassageWorker.CACHE_SCHEMA_VERSION
        original_behavior = FlashPassageWorker.CACHE_BEHAVIOR_VERSION
        try:
            base = FlashPassageWorker._cache_identity(
                "scorer-v1", "query", [b"a" * 32]
            )
            FlashPassageWorker.CACHE_SCHEMA_VERSION = original_schema + 1
            schema_variant = FlashPassageWorker._cache_identity(
                "scorer-v1", "query", [b"a" * 32]
            )
            FlashPassageWorker.CACHE_SCHEMA_VERSION = original_schema
            FlashPassageWorker.CACHE_BEHAVIOR_VERSION = original_behavior + "-next"
            behavior_variant = FlashPassageWorker._cache_identity(
                "scorer-v1", "query", [b"a" * 32]
            )
        finally:
            FlashPassageWorker.CACHE_SCHEMA_VERSION = original_schema
            FlashPassageWorker.CACHE_BEHAVIOR_VERSION = original_behavior

        self.assertNotEqual(base, schema_variant)
        self.assertNotEqual(base, behavior_variant)

    def test_identity_rejects_malformed_fields(self) -> None:
        invalid = (
            ("", "query", [b"a" * 32]),
            ("scorer\n", "query", [b"a" * 32]),
            ("scorer", "", [b"a" * 32]),
            ("scorer", "query\n", [b"a" * 32]),
            ("scorer", "query", []),
            ("scorer", "query", [b"a" * 31]),
            ("scorer", "query", ["a" * 32]),
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                FlashPassageWorker._cache_identity(*values)


class WorkerCacheArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.identity = FlashPassageWorker._cache_identity(
            "scorer-v1", "query-sensitive", [b"b" * 32, b"a" * 32]
        )
        self.scores = {b"b" * 32: 2, b"a" * 32: 1.5}

    def test_artifact_bytes_are_canonical_and_complete(self) -> None:
        payload = FlashPassageWorker._encode_cache_artifact(
            self.identity, self.scores
        )

        self.assertEqual(
            FlashPassageWorker._decode_cache_artifact(payload, self.identity),
            {b"a" * 32: 1.5, b"b" * 32: 2.0},
        )
        self.assertEqual(
            payload,
            FlashPassageWorker._encode_cache_artifact(
                self.identity, {b"a" * 32: 1.5, b"b" * 32: 2.0}
            ),
        )
        self.assertNotIn(b"query-sensitive", payload)
        decoded = msgspec.msgpack.decode(payload)
        self.assertEqual(
            [row[0] for row in decoded[-1]], [b"a" * 32, b"b" * 32]
        )

    def test_encode_requires_exact_finite_score_coverage(self) -> None:
        for scores in (
            {b"a" * 32: 1.0},
            {b"a" * 32: 1.0, b"b" * 32: 2.0, b"c" * 32: 3.0},
            {b"a" * 32: math.nan, b"b" * 32: 2.0},
            {b"a" * 32: math.inf, b"b" * 32: 2.0},
            {b"a" * 32: True, b"b" * 32: 2.0},
        ):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                FlashPassageWorker._encode_cache_artifact(self.identity, scores)

    def test_decode_rejects_foreign_truncated_extra_noncanonical_and_bad_artifacts(
        self,
    ) -> None:
        payload = FlashPassageWorker._encode_cache_artifact(
            self.identity, self.scores
        )
        foreign = FlashPassageWorker._cache_identity(
            "scorer-v1", "other-query", [b"b" * 32, b"a" * 32]
        )
        decoded = msgspec.msgpack.decode(payload)
        cases = (
            (payload[:-1], self.identity),
            (msgspec.msgpack.encode(list(decoded) + ["extra"]), self.identity),
            (msgspec.msgpack.encode(decoded[:-1]), self.identity),
            (msgspec.msgpack.encode(decoded), foreign),
            (msgspec.msgpack.encode(decoded[:-1] + [decoded[-1][::-1]]), self.identity),
            (payload + b"\x00", self.identity),
        )
        for candidate, identity in cases:
            with self.subTest(candidate=candidate[:20], identity=identity), self.assertRaises(
                ValueError
            ):
                FlashPassageWorker._decode_cache_artifact(candidate, identity)

        malformed = msgspec.msgpack.encode(
            [
                decoded[0],
                decoded[1],
                decoded[2],
                decoded[3],
                decoded[4],
                [[b"a" * 32, math.nan], [b"b" * 32, 2.0]],
            ]
        )
        with self.assertRaises(ValueError):
            FlashPassageWorker._decode_cache_artifact(malformed, self.identity)

    def test_codec_errors_do_not_echo_source_text(self) -> None:
        query = b"query-sensitive"
        payload = FlashPassageWorker._encode_cache_artifact(
            FlashPassageWorker._cache_identity("scorer-v1", query.decode(), [b"a" * 32]),
            {b"a" * 32: 1.0},
        )
        with self.assertRaises(ValueError) as raised:
            FlashPassageWorker._decode_cache_artifact(payload[:-1], self.identity)
        self.assertNotIn(query.decode(), str(raised.exception))

    def test_isolated_worker_has_identical_identity_and_artifact_codec(self) -> None:
        source = inspect.getsource(FlashPassageWorker)
        namespace = {"__name__": "flash_worker_isolated"}
        with patch.dict(sys.modules, {"trec_ragtime": None}, clear=False):
            exec(compile(source, "<flash-worker>", "exec"), namespace)
        isolated = namespace["FlashPassageWorker"]

        isolated_identity = isolated._cache_identity(
            "scorer-v1", "query-sensitive", [b"b" * 32, b"a" * 32]
        )
        self.assertEqual(isolated_identity, self.identity)
        self.assertEqual(
            isolated._cache_object_key(isolated_identity),
            FlashPassageWorker._cache_object_key(self.identity),
        )
        self.assertEqual(
            isolated._encode_cache_artifact(isolated_identity, self.scores),
            FlashPassageWorker._encode_cache_artifact(self.identity, self.scores),
        )


class StaticFlashWorkerTests(unittest.TestCase):
    def test_cold_miss_writes_rereads_and_returns_complete_scores(self) -> None:
        policy = LocalCrossEncoderConfig(batch_size=2, max_length=100)
        model = FakeModel(multiplier=2.0)
        store = WorkerCacheStore()
        worker = build_flash_worker(policy)(
            loader=CountingLoader(model, BatchEncodingTokenizer()),
            cache_store=store,
        )

        response = worker.score(
            _request_for_rows(policy, [("a", "a"), ("b", "bb")])
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
        self.assertEqual(response["cache"]["status"], "miss_written")
        self.assertEqual(response["cache"]["row_count"], 2)
        self.assertEqual(response["cache"]["batch_count"], 1)
        self.assertEqual(len(store.write_calls), 1)
        self.assertEqual(store.evict_local_calls, [store.write_calls[0][0]])
        self.assertEqual(store.read_validated_calls, [store.write_calls[0][0]] * 2)

    def test_warm_hit_on_fresh_worker_does_not_load_model(self) -> None:
        policy = LocalCrossEncoderConfig(max_length=100)
        store = WorkerCacheStore()
        first_model = FakeModel()
        first = build_flash_worker(policy)(
            loader=CountingLoader(first_model, BatchEncodingTokenizer()),
            cache_store=store,
        )
        request = _request_for_config(policy)
        first.score(request)

        def failing_loader(config: object) -> object:
            raise AssertionError("warm hit must not load the model")

        second = build_flash_worker(policy)(
            loader=failing_loader,
            cache_store=store,
        )
        response = second.score(request)

        self.assertEqual(response["cache"]["status"], "hit")
        self.assertEqual(response["cache"]["row_count"], 1)
        self.assertEqual(response["cache"]["batch_count"], 0)
        self.assertEqual(len(store.write_calls), 1)

    def test_65_rows_use_three_model_batches_and_report_three(self) -> None:
        policy = LocalCrossEncoderConfig(batch_size=32, max_length=1000)
        model = FakeModel()
        store = WorkerCacheStore()
        worker = build_flash_worker(policy)(
            loader=CountingLoader(model, BatchEncodingTokenizer()),
            cache_store=store,
        )
        rows = [(str(index), f"text-{index}") for index in range(65)]

        response = worker.score(_request_for_rows(policy, rows))

        self.assertEqual(len(response["scores"]), 65)
        self.assertEqual(len(model.predict_calls), 3)
        self.assertEqual(response["cache"]["batch_count"], 3)

    def test_false_write_fails_closed(self) -> None:
        policy = LocalCrossEncoderConfig(max_length=100)
        store = WorkerCacheStore(write_result=False)
        model = FakeModel()
        worker = build_flash_worker(policy)(
            loader=CountingLoader(model, BatchEncodingTokenizer()),
            cache_store=store,
        )

        with self.assertRaisesRegex(ValueError, "worker cache write failed"):
            worker.score(_request_for_config(policy))

    def test_missing_reread_fails_closed(self) -> None:
        policy = LocalCrossEncoderConfig(max_length=100)
        store = WorkerCacheStore()
        store.reread = None
        model = FakeModel()
        worker = build_flash_worker(policy)(
            loader=CountingLoader(model, BatchEncodingTokenizer()),
            cache_store=store,
        )

        with self.assertRaisesRegex(ValueError, "worker cache reread failed"):
            worker.score(_request_for_config(policy))

    def test_malformed_hit_fails_closed_before_model_load(self) -> None:
        policy = LocalCrossEncoderConfig(max_length=100)
        store = WorkerCacheStore()
        store.reread = {b"incomplete": 1.0}

        def failing_loader(config: object) -> object:
            raise AssertionError("malformed hit must not load the model")

        worker = build_flash_worker(policy)(
            loader=failing_loader,
            cache_store=store,
        )

        with self.assertRaisesRegex(ValueError, "worker cache artifact is malformed"):
            worker.score(_request_for_config(policy))

    def test_direct_store_repairs_malformed_l1_from_valid_l2(self) -> None:
        policy = LocalCrossEncoderConfig(max_length=100)
        request = _request_for_config(policy)
        content_hash = sha256_bytes("text")
        identity = FlashPassageWorker._cache_identity(
            policy.fingerprint, "query", [content_hash]
        )
        key = FlashPassageWorker._cache_object_key(identity)
        payload = FlashPassageWorker._encode_cache_artifact(
            identity, {content_hash: 4.0}
        )
        with tempfile.TemporaryDirectory() as directory:
            DirectStoreFilesystemFake.remote = {}
            fake_fsspec = types.ModuleType("fsspec")
            fake_fsspec.filesystem = lambda protocol, **kwargs: DirectStoreFilesystemFake(
                kwargs["cache_storage"]
            )
            fake_huggingface = types.ModuleType("huggingface_hub")
            fake_huggingface.HfApi = DirectStoreApiFake
            with (
                patch.dict(
                    sys.modules,
                    {
                        "trec_ragtime.artifact_store": None,
                        "fsspec": fake_fsspec,
                        "huggingface_hub": fake_huggingface,
                    },
                    clear=False,
                ),
                patch.dict(
                    os.environ,
                    {
                        "RAGTIME_HF_BUCKET_ID": "private-owner/private-bucket",
                        "RAGTIME_ARTIFACT_CACHE_DIR": directory,
                    },
                    clear=False,
                ),
            ):
                worker = build_flash_worker(policy)(
                    loader=lambda config: (_ for _ in ()).throw(
                        AssertionError("valid L2 hit must not load the model")
                    )
                )
                store = worker._cache_store
                store._local_candidate(key).write_bytes(b"malformed-l1")
                DirectStoreFilesystemFake.remote[
                    "buckets/private-owner/private-bucket/" + key
                ] = payload

                response = worker.score(request)

                self.assertEqual(response["cache"]["status"], "hit")
                self.assertFalse(store._local_candidate(key).exists())

    def test_direct_store_rejects_malformed_l2_as_fatal(self) -> None:
        policy = LocalCrossEncoderConfig(max_length=100)
        request = _request_for_config(policy)
        with tempfile.TemporaryDirectory() as directory:
            DirectStoreFilesystemFake.remote = {}
            fake_fsspec = types.ModuleType("fsspec")
            fake_fsspec.filesystem = lambda protocol, **kwargs: DirectStoreFilesystemFake(
                kwargs["cache_storage"]
            )
            fake_huggingface = types.ModuleType("huggingface_hub")
            fake_huggingface.HfApi = DirectStoreApiFake
            with (
                patch.dict(
                    sys.modules,
                    {
                        "trec_ragtime.artifact_store": None,
                        "fsspec": fake_fsspec,
                        "huggingface_hub": fake_huggingface,
                    },
                    clear=False,
                ),
                patch.dict(
                    os.environ,
                    {
                        "RAGTIME_HF_BUCKET_ID": "private-owner/private-bucket",
                        "RAGTIME_ARTIFACT_CACHE_DIR": directory,
                    },
                    clear=False,
                ),
            ):
                worker = build_flash_worker(policy)(
                    loader=lambda config: (_ for _ in ()).throw(
                        AssertionError("malformed L2 must not load the model")
                    )
                )
                identity = FlashPassageWorker._cache_identity(
                    policy.fingerprint, "query", [sha256_bytes("text")]
                )
                key = FlashPassageWorker._cache_object_key(identity)
                DirectStoreFilesystemFake.remote[
                    "buckets/private-owner/private-bucket/" + key
                ] = b"malformed-l2"

                with self.assertRaisesRegex(ValueError, "worker cache artifact is malformed"):
                    worker.score(request)

    def test_isolated_worker_scores_with_injected_store_without_project_imports(
        self,
    ) -> None:
        policy = LocalCrossEncoderConfig(max_length=100)
        source = inspect.getsource(FlashPassageWorker)
        namespace = {"__name__": "flash_worker_isolated"}
        model = FakeModel()
        store = WorkerCacheStore()
        with patch.dict(sys.modules, {"trec_ragtime": None}, clear=False):
            exec(compile(source, "<flash-worker>", "exec"), namespace)
        isolated = namespace["FlashPassageWorker"]
        worker = isolated(
            loader=CountingLoader(model, BatchEncodingTokenizer()),
            cache_store=store,
        )

        response = worker.score(_request_for_config(policy))

        self.assertEqual(response["cache"]["status"], "miss_written")

    def test_default_store_uses_private_bucket_configuration_without_leaking_secrets(
        self,
    ) -> None:
        policy = LocalCrossEncoderConfig(max_length=100)
        secret_bucket = "private-owner/private-bucket"
        secret_token = "hf_secret_token_value"
        with tempfile.TemporaryDirectory() as directory:
            fake_fsspec = types.ModuleType("fsspec")
            fake_fsspec.filesystem = lambda protocol, **kwargs: DirectStoreFilesystemFake(
                kwargs["cache_storage"]
            )
            fake_huggingface = types.ModuleType("huggingface_hub")
            fake_huggingface.HfApi = DirectStoreApiFake
            with (
                patch.dict(
                    sys.modules,
                    {
                        "trec_ragtime.artifact_store": None,
                        "fsspec": fake_fsspec,
                        "huggingface_hub": fake_huggingface,
                    },
                    clear=False,
                ),
                patch.dict(
                    os.environ,
                    {
                        "RAGTIME_HF_BUCKET_ID": secret_bucket,
                        "RAGTIME_ARTIFACT_CACHE_DIR": directory,
                        "HF_TOKEN": secret_token,
                        "HF_ENDPOINT": "https://huggingface.example",
                    },
                    clear=False,
                ),
            ):
                worker = build_flash_worker(policy)(
                    loader=CountingLoader(FakeModel(), BatchEncodingTokenizer())
                )

        self.assertEqual(worker._cache_store.__class__.__name__, "DirectArtifactStore")
        self.assertNotIn(secret_token, repr(worker))

    def test_worker_tokenizer_metadata_matches_local_token_length_contract(
        self,
    ) -> None:
        policy = LocalCrossEncoderConfig(max_length=100)

        class MetadataTokenizer(BatchEncodingTokenizer):
            def __init__(
                self,
                truncated: object,
                *,
                include_length: bool = True,
                reported_length: int | None = None,
            ) -> None:
                super().__init__()
                self.truncated = truncated
                self.include_length = include_length
                self.reported_length = reported_length

            def __call__(
                self, query: str, text: str, **kwargs: object
            ) -> UserDict[str, object]:
                encoded = super().__call__(query, text, **kwargs)
                encoded["num_truncated_tokens"] = self.truncated
                if not self.include_length:
                    del encoded["length"]
                elif self.reported_length is not None:
                    encoded["length"] = [self.reported_length]
                return encoded

        def worker_with(tokenizer: MetadataTokenizer) -> tuple[FakeModel, object]:
            model = FakeModel()
            worker = build_flash_worker(policy)(
                loader=CountingLoader(model, tokenizer),
                cache_store=WorkerCacheStore(),
            )
            return model, worker

        for truncated in (None, 0, [0], (0,)):
            with self.subTest(truncated=truncated):
                model, worker = worker_with(MetadataTokenizer(truncated))
                response = worker.score(_request_for_config(policy))
                self.assertEqual(
                    response["scores"],
                    [{"hash": sha256_bytes("text").hex(), "score": 4.0}],
                )
                self.assertEqual(len(model.predict_calls), 1)
        for truncated in (False, [], ()):
            with self.subTest(truncated=truncated):
                model, worker = worker_with(MetadataTokenizer(truncated))
                with self.assertRaises(ValueError):
                    worker.score(_request_for_config(policy))
                self.assertEqual(model.predict_calls, [])
        model, worker = worker_with(MetadataTokenizer(0, include_length=False))
        response = worker.score(_request_for_config(policy))
        self.assertEqual(
            response["scores"],
            [{"hash": sha256_bytes("text").hex(), "score": 4.0}],
        )
        self.assertEqual(len(model.predict_calls), 1)

        model, worker = worker_with(MetadataTokenizer(0, reported_length=999))
        with self.assertRaises(ValueError):
            worker.score(_request_for_config(policy))
        self.assertEqual(model.predict_calls, [])

    def test_runtime_validation_accepts_installed_torch_local_version_suffix(
        self,
    ) -> None:
        values = {
            "sentence_transformers_version": "1.0.0",
            "transformers_version": "1.0.0",
            "tokenizers_version": "1.0.0",
            "torch_version": "2.9.1",
        }
        installed = {
            "sentence-transformers": "1.0.0",
            "transformers": "1.0.0",
            "tokenizers": "1.0.0",
            "torch": "2.9.1+cu128",
        }

        with patch.object(
            importlib.metadata,
            "version",
            side_effect=installed.__getitem__,
        ):
            FlashPassageWorker._validate_runtime_versions(values)

    def test_runtime_validation_accepts_normalized_public_version_with_local_suffix(
        self,
    ) -> None:
        values = {
            "sentence_transformers_version": "1.0.0",
            "transformers_version": "1.0.0",
            "tokenizers_version": "1.0.0",
            "torch_version": "1.0",
        }
        installed = {
            "sentence-transformers": "1.0.0",
            "transformers": "1.0.0",
            "tokenizers": "1.0.0",
            "torch": "1.0.0+vendor",
        }

        with patch.object(
            importlib.metadata,
            "version",
            side_effect=installed.__getitem__,
        ):
            FlashPassageWorker._validate_runtime_versions(values)

    def test_runtime_validation_rejects_public_version_drift_for_each_pinned_distribution(
        self,
    ) -> None:
        fields_and_distributions = (
            ("sentence_transformers_version", "sentence-transformers"),
            ("transformers_version", "transformers"),
            ("tokenizers_version", "tokenizers"),
            ("torch_version", "torch"),
        )
        values = {field_name: "1.0.0" for field_name, _ in fields_and_distributions}

        for field_name, distribution in fields_and_distributions:
            with self.subTest(field_name=field_name):
                installed = {name: "1.0.0" for _, name in fields_and_distributions}
                installed[distribution] = "1.0.1+local"
                with (
                    patch.object(
                        importlib.metadata,
                        "version",
                        side_effect=installed.__getitem__,
                    ),
                    self.assertRaisesRegex(ValueError, field_name),
                ):
                    FlashPassageWorker._validate_runtime_versions(values)

    def test_repeated_builds_do_not_mutate_static_worker_error_type(self) -> None:
        policy_a = LocalCrossEncoderConfig(query_policy="policy-a")
        policy_b = replace(policy_a, query_policy="policy-b")

        first = build_flash_worker(policy_a)
        second = build_flash_worker(policy_b)

        self.assertIs(first, FlashPassageWorker)
        self.assertIs(second, FlashPassageWorker)
        self.assertIs(FlashPassageWorker._ERROR, ValueError)
        self.assertNotIn("FlashScoringError", inspect.getsource(FlashPassageWorker))
        self.assertNotIn("FlashScoringError", inspect.getsource(build_flash_worker))

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
        worker = isolated_worker(loader=loader, cache_store=WorkerCacheStore())
        response = worker.score(_request_for_config(policy))

        self.assertEqual(response["scorer_fingerprint"], policy.fingerprint)
        self.assertEqual(model.predict_calls[0][0], [("query", "text")])
        self.assertEqual(len(load_calls), 1)

    def test_constructor_is_cold_and_first_valid_request_loads_once(self) -> None:
        policy = LocalCrossEncoderConfig(max_length=100)
        loader = CountingLoader(FakeModel(), BatchEncodingTokenizer())
        worker = build_flash_worker(policy)(
            loader=loader, cache_store=WorkerCacheStore()
        )

        self.assertEqual(len(loader.calls), 0)
        request = _request_for_config(policy)
        worker.score(request)
        worker.score(request)
        self.assertEqual(len(loader.calls), 1)
        drifted = replace(policy, query_policy="drifted")
        with self.assertRaises(ValueError):
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

        worker = build_flash_worker(policy_a)(loader=loader, cache_store=WorkerCacheStore())
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
            worker = build_flash_worker(policy)(loader=loader, cache_store=WorkerCacheStore())
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
        worker = build_flash_worker(policy_a)(loader=loader, cache_store=WorkerCacheStore())

        with self.assertRaises(ValueError):
            worker.score(_request_for_config(policy_a))
        with self.assertRaises(ValueError):
            worker.score(_request_for_config(policy_b))

        self.assertEqual(len(loader.calls), 1)
        self.assertEqual(len(model.predict_calls), 1)

    def test_worker_validates_exact_request_model_config_and_pair_ceiling(self) -> None:
        policy = LocalCrossEncoderConfig(max_length=10)
        loader = CountingLoader(FakeModel(), BatchEncodingTokenizer())
        worker = build_flash_worker(policy)(loader=loader, cache_store=WorkerCacheStore())
        request = _request_for_config(policy)

        with self.assertRaises(ValueError):
            worker.score({**request, "extra": True})
        with self.assertRaises(ValueError):
            worker.score({**request, "scorer_fingerprint": "wrong"})
        with self.assertRaises(ValueError):
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
        worker = isolated_worker(
            loader=lambda config: (model, tokenizer), cache_store=WorkerCacheStore()
        )

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
        worker = build_flash_worker(policy)(
            loader=CountingLoader(model, tokenizer), cache_store=WorkerCacheStore()
        )
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
            loader=lambda config: (FakeModel(multiplier=2.0), BatchEncodingTokenizer()),
            cache_store=WorkerCacheStore(),
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
            cache=_cache_telemetry(len(response["scores"])),
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
                f"packaging=={importlib.metadata.version('packaging')}",
                f"fsspec=={importlib.metadata.version('fsspec')}",
                f"huggingface-hub=={importlib.metadata.version('huggingface-hub')}",
            ],
        )

    def test_package_import_and_local_build_do_not_resolve_runpod(self) -> None:
        with patch.dict(sys.modules, {"runpod_flash": None}):
            worker = build_flash_worker(LocalCrossEncoderConfig())
        self.assertIs(worker, FlashPassageWorker)


if __name__ == "__main__":
    unittest.main()
