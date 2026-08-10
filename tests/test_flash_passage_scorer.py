from __future__ import annotations

import asyncio
import contextlib
import gc
import inspect
import io
import json
import logging
import linecache
import math
import pickle
import sys
import textwrap
import threading
import types
import unittest
import warnings
from collections.abc import Callable
from dataclasses import replace
from typing import Any
from unittest.mock import patch

from trec_ragtime.passages import Passage, sha256_bytes
from trec_ragtime.passage_scoring import (
    LocalCrossEncoderConfig,
    LocalCrossEncoderScorer,
    PassageScoringError,
)
from trec_ragtime.flash_passage_scorer import (
    FLASH_PAYLOAD_LIMIT_BYTES,
    WIRE_SCHEMA_VERSION,
    _EventLoopBridge,
    FlashEndpointBatchRunner,
    FlashScoringError,
    RunpodFlashConfig,
    RunpodFlashPassageScorer,
    decode_flash_request,
    decode_flash_response,
    encode_flash_request,
    encode_flash_response,
)
import trec_ragtime.flash_worker as flash_worker_module
from trec_ragtime.flash_worker import build_flash_worker


def _passage(text: str, *, document_id: str = "doc", index: int = 0) -> Passage:
    return Passage(
        document_id=document_id,
        source_content_hash=sha256_bytes(text),
        representation="canonical",
        representation_hash=sha256_bytes(text),
        language="eng",
        chunk_index=index,
        passage_id=sha256_bytes(f"passage:{document_id}:{index}:{text}"),
        text=text,
        start_char=0,
        end_char=len(text),
        boundary="paragraph_packed",
        previous_passage_id=None,
        next_passage_id=None,
        content_hash=sha256_bytes(text),
        chunker_fingerprint="test-chunker-v1",
    )


class FakeAsyncRunner:
    def __init__(self, *, mode: str = "normal", delay: float = 0.0) -> None:
        self.mode = mode
        self.delay = delay
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
            if self.mode == "missing":
                rows = request["rows"][:-1]
            elif self.mode == "extra":
                rows = request["rows"] + [{"hash": "00" * 32, "score": 0.0}]
            elif self.mode == "nan":
                rows = [{"hash": row["hash"], "score": math.nan} for row in request["rows"]]
            elif self.mode == "malformed":
                return b'{"scores":[{"hash": "not-a-hash"}]}'
            else:
                rows = [
                    {"hash": row["hash"], "score": float(len(row["text"]))}
                    for row in request["rows"]
                ]
            return encode_flash_response(
                rows,
                schema_version=request["schema_version"],
                scorer_fingerprint=request["scorer_fingerprint"],
            )
        finally:
            self.active -= 1


class SwappedBatchRunner:
    def __init__(self) -> None:
        self.payloads: list[bytes] = []

    async def run_batch(self, payload: bytes) -> bytes:
        self.payloads.append(payload)
        request = decode_flash_request(payload)
        if len(self.payloads) == 1:
            scores = [{"hash": sha256_bytes("second"), "score": 2.0}]
        else:
            scores = [{"hash": sha256_bytes("first"), "score": 1.0}]
        return encode_flash_response(
            scores,
            schema_version=request["schema_version"],
            scorer_fingerprint=request["scorer_fingerprint"],
        )


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
                raise RuntimeError("first batch failed")
            try:
                await asyncio.sleep(60.0)
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
            raise AssertionError("sibling was not cancelled")
        finally:
            self.active -= 1


class FlashWireCodecTests(unittest.TestCase):
    def test_request_rows_are_deterministic_and_measure_final_utf8_bytes(self) -> None:
        rows = (
            {"hash": sha256_bytes("é漢字"), "text": "é漢字"},
            {"hash": sha256_bytes("plain"), "text": "plain"},
        )

        first = encode_flash_request(
            "q-é",
            rows,
            schema_version=WIRE_SCHEMA_VERSION,
            scorer_fingerprint="local-policy-v1",
        )
        second = encode_flash_request(
            "q-é",
            rows,
            schema_version=WIRE_SCHEMA_VERSION,
            scorer_fingerprint="local-policy-v1",
        )

        self.assertEqual(first, second)
        self.assertEqual(first, first.decode("utf-8").encode("utf-8"))
        decoded = decode_flash_request(first)
        self.assertEqual(decoded["schema_version"], WIRE_SCHEMA_VERSION)
        self.assertEqual(decoded["scorer_fingerprint"], "local-policy-v1")
        self.assertEqual(decoded["query"], "q-é")
        self.assertEqual(decoded["rows"][0], {"hash": sha256_bytes("é漢字").hex(), "text": "é漢字"})
        self.assertLess(len(first), FLASH_PAYLOAD_LIMIT_BYTES)

    def test_wire_decoders_reject_unknown_fields_malformed_hashes_and_non_finite_scores(self) -> None:
        with self.assertRaises(FlashScoringError):
            decode_flash_request(
                b'{"schema_version":1,"scorer_fingerprint":"p",'
                b'"query":"q","rows":[],"extra":1}'
            )
        with self.assertRaises(FlashScoringError):
            decode_flash_request(
                b'{"schema_version":1,"scorer_fingerprint":"p",'
                b'"query":"q","rows":[{"hash":"bad","text":"x"}]}'
            )
        with self.assertRaises(FlashScoringError):
            decode_flash_response(
                b'{"schema_version":1,"scorer_fingerprint":"p","scores":'
                b'[{"hash":"' + b"00" * 32 + b'","score":NaN}]}'
            )
        with self.assertRaises(FlashScoringError):
            decode_flash_response(
                b'{"schema_version":1,"scorer_fingerprint":"p","scores":'
                b'[{"hash":"' + b"00" * 32 + b'","score":1}],"extra":1}'
            )

    def test_wire_schema_and_policy_fingerprint_are_required_and_strict(self) -> None:
        rows = ({"hash": sha256_bytes("text"), "text": "text"},)
        request = encode_flash_request(
            "query",
            rows,
            schema_version=WIRE_SCHEMA_VERSION,
            scorer_fingerprint="policy-v1",
        )
        with self.assertRaises(FlashScoringError):
            decode_flash_request(
                request.replace(b'"schema_version":1', b'"schema_version":2')
            )
        with self.assertRaises(FlashScoringError):
            decode_flash_request(request.replace(b'"scorer_fingerprint":"policy-v1"', b'"scorer_fingerprint":""'))

        response = encode_flash_response(
            ({"hash": sha256_bytes("text"), "score": 1.0},),
            schema_version=WIRE_SCHEMA_VERSION,
            scorer_fingerprint="policy-v1",
        )
        self.assertEqual(decode_flash_response(response)["scorer_fingerprint"], "policy-v1")


class RunpodFlashConfigTests(unittest.TestCase):
    def test_result_fingerprint_excludes_transport_only_settings(self) -> None:
        policy = LocalCrossEncoderConfig(
            batch_size=2,
            sentence_transformers_version="5.7.0",
        )
        base = RunpodFlashConfig(
            endpoint_id="endpoint-a",
            model_config=policy,
            workers=1,
            idle_timeout=60.0,
            request_timeout=30.0,
            concurrency=2,
        )
        transport_variant = replace(
            base,
            endpoint_id="endpoint-b",
            workers=4,
            idle_timeout=600.0,
            request_timeout=3.0,
            concurrency=8,
            max_payload_bytes=12345,
        )
        model_variant = replace(base, precision="float16")

        self.assertEqual(base.fingerprint, policy.fingerprint)
        self.assertEqual(base.fingerprint, transport_variant.fingerprint)
        self.assertNotEqual(base.fingerprint, model_variant.fingerprint)

    def test_transport_payload_limit_is_not_result_identity(self) -> None:
        base = RunpodFlashConfig(endpoint_id="fake")
        self.assertEqual(
            base.fingerprint,
            replace(base, max_payload_bytes=base.max_payload_bytes // 2).fingerprint,
        )

    def test_auth_token_is_not_a_config_field_or_identity_input(self) -> None:
        self.assertNotIn("auth_token", RunpodFlashConfig.__dataclass_fields__)


class RunpodFlashPassageScorerTests(unittest.TestCase):
    def test_client_and_worker_use_the_effective_local_policy_fingerprint(self) -> None:
        policy = LocalCrossEncoderConfig(
            batch_size=2,
            max_length=77,
            query_format="Q: {query}",
        )
        config = RunpodFlashConfig(endpoint_id="fake", model_config=policy)
        self.assertEqual(config.fingerprint, policy.fingerprint)

    def test_client_rejects_response_policy_fingerprint_mismatch(self) -> None:
        passage = _passage("passage")

        class WrongFingerprintRunner:
            async def run_batch(self, payload: bytes) -> bytes:
                request = decode_flash_request(payload)
                return encode_flash_response(
                    ({"hash": passage.content_hash, "score": 1.0},),
                    schema_version=request["schema_version"],
                    scorer_fingerprint="wrong-policy",
                )

        scorer = RunpodFlashPassageScorer(
            RunpodFlashConfig(endpoint_id="fake"), WrongFingerprintRunner()
        )
        with self.assertRaises(FlashScoringError):
            scorer.score("query", (passage,))

    def test_each_response_must_cover_its_own_batch_before_global_merge(self) -> None:
        first = _passage("first", document_id="first")
        second = _passage("second", document_id="second")
        scorer = RunpodFlashPassageScorer(
            RunpodFlashConfig(endpoint_id="fake", batch_size=1),
            SwappedBatchRunner(),
        )
        with self.assertRaises(FlashScoringError):
            scorer.score("query", (first, second))

    def test_failed_batch_cancels_and_awaits_all_siblings_before_returning(self) -> None:
        passages = tuple(_passage(f"text-{index}", document_id=str(index)) for index in range(4))
        runner = QuiescingFailureRunner()
        scorer = RunpodFlashPassageScorer(
            RunpodFlashConfig(endpoint_id="fake", batch_size=1, concurrency=2),
            runner,
        )

        with self.assertRaises(FlashScoringError):
            scorer.score("query", passages)

        self.assertLessEqual(runner.dispatches, 2)
        self.assertGreaterEqual(runner.cancelled, 1)
        self.assertEqual(runner.active, 0)

    def test_batches_by_count_and_returns_exact_unique_hash_coverage(self) -> None:
        first = _passage("first", document_id="first")
        duplicate = _passage("first", document_id="duplicate")
        second = _passage("second", document_id="second")
        third = _passage("third", document_id="third")
        runner = FakeAsyncRunner()
        scorer = RunpodFlashPassageScorer(
            RunpodFlashConfig(endpoint_id="fake", batch_size=2),
            runner,
        )

        scores = scorer.score("query", (first, duplicate, second, third))

        self.assertEqual(scores, {
            first.content_hash: 5.0,
            second.content_hash: 6.0,
            third.content_hash: 5.0,
        })
        self.assertEqual(len(runner.payloads), 2)
        self.assertEqual(
            [row["text"] for row in decode_flash_request(runner.payloads[0])["rows"]],
            ["first", "second"],
        )
        self.assertEqual(
            [row["text"] for row in decode_flash_request(runner.payloads[1])["rows"]],
            ["third"],
        )

    def test_batches_by_actual_final_utf8_payload_and_rejects_oversized_row(self) -> None:
        passages = tuple(_passage("漢字-é" * count, document_id=str(count)) for count in range(1, 8))
        config = RunpodFlashConfig(
            endpoint_id="fake",
            batch_size=99,
            max_payload_bytes=500,
        )
        runner = FakeAsyncRunner()
        scorer = RunpodFlashPassageScorer(config, runner)

        scorer.score("query-é", passages)

        self.assertGreater(len(runner.payloads), 1)
        self.assertTrue(all(len(payload) < 500 for payload in runner.payloads))
        self.assertTrue(all(len(payload) < FLASH_PAYLOAD_LIMIT_BYTES for payload in runner.payloads))

        too_large = _passage("x" * 500, document_id="too-large")
        with self.assertRaises(FlashScoringError):
            scorer.score("q", (too_large,))

    def test_concurrency_is_bounded_and_all_batches_complete(self) -> None:
        passages = tuple(_passage(f"text-{index}", document_id=str(index)) for index in range(8))
        runner = FakeAsyncRunner(delay=0.02)
        scorer = RunpodFlashPassageScorer(
            RunpodFlashConfig(endpoint_id="fake", batch_size=1, concurrency=2),
            runner,
        )

        scores = scorer.score("query", passages)

        self.assertEqual(len(scores), 8)
        self.assertLessEqual(runner.maximum_active, 2)

    def test_remote_response_requires_complete_global_coverage(self) -> None:
        first = _passage("first")
        second = _passage("second", document_id="second")
        for mode in ("missing", "extra", "nan", "malformed"):
            with self.subTest(mode=mode):
                scorer = RunpodFlashPassageScorer(
                    RunpodFlashConfig(endpoint_id="fake"),
                    FakeAsyncRunner(mode=mode),
                )
                with self.assertRaises(FlashScoringError):
                    scorer.score("query", (first, second))

    def test_timeout_and_runner_errors_are_not_silently_converted(self) -> None:
        passage = _passage("passage")
        timeout_scorer = RunpodFlashPassageScorer(
            RunpodFlashConfig(endpoint_id="fake", request_timeout=0.001),
            FakeAsyncRunner(delay=0.1),
        )
        with self.assertRaises(FlashScoringError):
            timeout_scorer.score("query", (passage,))

        class ErrorRunner:
            async def run_batch(self, payload: bytes) -> bytes:
                raise RuntimeError("transport failed")

        with self.assertRaises(FlashScoringError):
            RunpodFlashPassageScorer(
                RunpodFlashConfig(endpoint_id="fake"), ErrorRunner()
            ).score("query", (passage,))

    def test_close_stops_the_private_event_loop_thread_deterministically(self) -> None:
        scorer = RunpodFlashPassageScorer(
            RunpodFlashConfig(endpoint_id="fake"), FakeAsyncRunner()
        )
        loop_thread = scorer._bridge._thread  # type: ignore[attr-defined]

        self.assertTrue(loop_thread.is_alive())
        scorer.close()
        scorer.close()
        self.assertFalse(loop_thread.is_alive())

    def test_score_after_close_rejects_without_creating_an_unawaited_coroutine(self) -> None:
        passage = _passage("passage")
        scorer = RunpodFlashPassageScorer(
            RunpodFlashConfig(endpoint_id="fake"), FakeAsyncRunner()
        )
        scorer.close()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with self.assertRaises(FlashScoringError):
                scorer.score("query", (passage,))
            gc.collect()

        self.assertFalse(any(item.category is RuntimeWarning for item in caught))

    def test_sync_score_from_a_host_event_loop_uses_the_private_bridge(self) -> None:
        passage = _passage("passage")
        scorer = RunpodFlashPassageScorer(
            RunpodFlashConfig(endpoint_id="fake"), FakeAsyncRunner()
        )

        async def host_call() -> dict[bytes, float]:
            return dict(scorer.score("query", (passage,)))

        self.assertEqual(asyncio.run(host_call()), {passage.content_hash: 7.0})

    def test_bridge_startup_error_is_propagated_without_a_live_thread(self) -> None:
        with patch(
            "trec_ragtime.flash_passage_scorer.asyncio.new_event_loop",
            side_effect=RuntimeError("cannot start loop"),
        ):
            with self.assertRaises(FlashScoringError):
                _EventLoopBridge()

        self.assertFalse(
            any(
                thread.name == "trec-ragtime-flash-loop" and thread.is_alive()
                for thread in threading.enumerate()
            )
        )

    def test_bridge_discards_cancellation_resistant_tasks_and_joins_thread(self) -> None:
        bridge = _EventLoopBridge()

        async def cancellation_resistant() -> None:
            try:
                await asyncio.sleep(60.0)
            except asyncio.CancelledError:
                while True:
                    await asyncio.sleep(0)

        with self.assertRaises(TimeoutError):
            bridge.run(cancellation_resistant(), timeout=0.01)
        loop_thread = bridge._thread  # type: ignore[attr-defined]

        logger = logging.getLogger("asyncio")
        with self.assertNoLogs(logger, level="ERROR"):
            bridge.close()

        self.assertFalse(loop_thread.is_alive())

    def test_outer_timeout_quiesces_siblings_before_score_returns(self) -> None:
        class OuterTimeoutRunner:
            def __init__(self) -> None:
                self.dispatches = 0
                self.active = 0
                self.events: list[str] = []

            async def run_batch(self, payload: bytes) -> bytes:
                self.dispatches += 1
                dispatch = self.dispatches
                self.events.append(f"dispatch-{dispatch}")
                self.active += 1
                try:
                    await asyncio.sleep(60.0)
                except asyncio.CancelledError:
                    self.active -= 1
                    self.events.append(f"cancel-{dispatch}")
                    while True:
                        self.events.append(f"drain-{dispatch}")
                        await asyncio.sleep(0)

        passages = tuple(_passage(f"text-{index}", document_id=str(index)) for index in range(4))
        runner = OuterTimeoutRunner()
        scorer = RunpodFlashPassageScorer(
            RunpodFlashConfig(
                endpoint_id="fake",
                batch_size=1,
                concurrency=2,
                request_timeout=0.001,
            ),
            runner,
        )
        loop_thread = scorer._bridge._thread  # type: ignore[attr-defined]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with self.assertRaises(FlashScoringError):
                scorer.score("query", passages)
            events_at_return = tuple(runner.events)
            self.assertEqual(runner.active, 0)
            self.assertFalse(loop_thread.is_alive())
            self.assertTrue(scorer._bridge.closed)  # type: ignore[attr-defined]
            gc.collect()

        self.assertFalse(any(item.category is RuntimeWarning for item in caught))
        self.assertLessEqual(runner.dispatches, 2)
        import time

        time.sleep(0.05)
        self.assertEqual(tuple(runner.events), events_at_return)
        scorer.close()

    def test_stdout_and_stderr_never_contain_query_text_hash_or_credentials(self) -> None:
        query = "secret query"
        text_value = "secret passage"
        token = "bearer-secret"
        passage = _passage(text_value)
        runner = FakeAsyncRunner()
        output = io.StringIO()
        error_output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error_output):
            RunpodFlashPassageScorer(
                RunpodFlashConfig(endpoint_id="fake"), runner
            ).score(query, (passage,))

        combined = output.getvalue() + error_output.getvalue()
        for secret in (query, text_value, token, passage.content_hash.hex()):
            self.assertNotIn(secret, combined)

    def test_local_and_remote_fake_models_have_exact_score_parity_through_wire(self) -> None:
        policy = LocalCrossEncoderConfig(
            device="cuda",
            precision="float16",
            batch_size=1,
            max_length=20,
            query_format="Q: {query}",
            query_policy="q-policy-v2",
        )
        passages = (_passage("a", document_id="a"), _passage("longer", document_id="b"))
        local_calls: dict[str, list[Any]] = {"tokenizer": [], "predict": []}
        worker_calls: dict[str, list[Any]] = {"tokenizer": [], "predict": []}

        class Tokenizer:
            def __init__(self, calls: dict[str, list[Any]]) -> None:
                self.calls = calls

            def __call__(self, query: str, text: str, **kwargs: Any) -> dict[str, list[list[int]]]:
                self.calls["tokenizer"].append((query, text, kwargs))
                length = 100 if text == "overflow" else len(query) + len(text)
                return {"input_ids": [[1] * length]}

        class Model:
            def __init__(self, calls: dict[str, list[Any]]) -> None:
                self.calls = calls

            def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> list[float]:
                self.calls["predict"].append((pairs, kwargs))
                return [float(len(text) * 2 + 0.5) for _, text in pairs]

        def loader_for(calls: dict[str, list[Any]]) -> Callable[[LocalCrossEncoderConfig], object]:
            return lambda config: (Model(calls), Tokenizer(calls))

        local = LocalCrossEncoderScorer(policy, loader=loader_for(local_calls))
        worker_class = build_flash_worker(policy)
        worker = worker_class(loader=loader_for(worker_calls))
        observed: dict[str, object] = {}

        class InMemoryWireRunner:
            async def run_batch(self, payload: bytes) -> bytes:
                request = decode_flash_request(payload)
                observed["request"] = request
                worker_response = worker.score(**request)
                encoded = encode_flash_response(
                    worker_response["scores"],
                    schema_version=worker_response["schema_version"],
                    scorer_fingerprint=worker_response["scorer_fingerprint"],
                )
                decoded = decode_flash_response(encoded)
                observed["response"] = decoded
                return encode_flash_response(
                    decoded["scores"],
                    schema_version=decoded["schema_version"],
                    scorer_fingerprint=decoded["scorer_fingerprint"],
                )

        remote = RunpodFlashPassageScorer(
            RunpodFlashConfig(endpoint_id="fake", model_config=policy),
            InMemoryWireRunner(),
        )

        local_scores = local.score("query", passages)
        remote_scores = remote.score("query", passages)

        self.assertEqual(local_scores, remote_scores)
        request = observed["request"]
        self.assertEqual(request["schema_version"], WIRE_SCHEMA_VERSION)
        self.assertEqual(request["scorer_fingerprint"], policy.fingerprint)
        self.assertEqual(observed["response"]["scorer_fingerprint"], policy.fingerprint)
        for calls in (local_calls, worker_calls):
            self.assertEqual([entry[:2] for entry in calls["tokenizer"]], [
                ("Q: query", "a"),
                ("Q: query", "longer"),
            ])
            self.assertTrue(all(entry[2]["truncation"] is False for entry in calls["tokenizer"]))
            self.assertTrue(all(entry[2]["max_length"] == 20 for entry in calls["tokenizer"]))
            self.assertEqual([entry[0] for entry in calls["predict"]], [[("Q: query", "a")], [("Q: query", "longer")]])
            self.assertTrue(all(entry[1]["apply_softmax"] is False for entry in calls["predict"]))
            self.assertTrue(all(entry[1]["convert_to_numpy"] is True for entry in calls["predict"]))
            self.assertTrue(all(entry[1]["convert_to_tensor"] is False for entry in calls["predict"]))

        overflowing = _passage("overflow", document_id="overflow")
        with self.assertRaises(PassageScoringError):
            local.score("query", (overflowing,))
        with self.assertRaises(FlashScoringError):
            remote.score("query", (overflowing,))


class EndpointRunnerTests(unittest.TestCase):
    def test_endpoint_runner_uses_async_run_then_job_wait(self) -> None:
        calls: list[tuple[str, Any]] = []
        config = RunpodFlashConfig(endpoint_id="endpoint-id", request_timeout=3.0)

        class Job:
            def __init__(self, input_data: dict[str, object]) -> None:
                self.output = {
                    "schema_version": input_data["schema_version"],
                    "scorer_fingerprint": input_data["scorer_fingerprint"],
                    "scores": [{"hash": sha256_bytes("text").hex(), "score": 4.0}],
                }

            async def wait(self, timeout: float | None = None) -> Job:
                calls.append(("wait", timeout))
                return self

        class Endpoint:
            def __init__(self, **kwargs: Any) -> None:
                calls.append(("init", kwargs))

            async def run(self, input_data: Any) -> Job:
                calls.append(("run", input_data))
                return Job(input_data)

        request = encode_flash_request(
            "query",
            ({"hash": sha256_bytes("text"), "text": "text"},),
            schema_version=WIRE_SCHEMA_VERSION,
            scorer_fingerprint=config.fingerprint,
        )
        runner = FlashEndpointBatchRunner(
            config,
            endpoint_factory=Endpoint,
        )

        response = runner.run_batch_sync(request)

        self.assertEqual(
            decode_flash_response(response),
            {
                "schema_version": WIRE_SCHEMA_VERSION,
                "scorer_fingerprint": config.fingerprint,
                "scores": [{"hash": sha256_bytes("text").hex(), "score": 4.0}],
            },
        )
        self.assertEqual(calls[0], ("init", {"id": "endpoint-id"}))
        self.assertEqual(calls[1][0], "run")
        self.assertEqual(calls[2], ("wait", 3.0))

    def test_endpoint_run_and_wait_failures_cancel_the_submitted_job(self) -> None:
        request_config = RunpodFlashConfig(endpoint_id="endpoint-id", request_timeout=0.01)
        request = encode_flash_request(
            "query",
            ({"hash": sha256_bytes("text"), "text": "text"},),
            schema_version=WIRE_SCHEMA_VERSION,
            scorer_fingerprint=request_config.fingerprint,
        )

        class Job:
            def __init__(self, wait_mode: str) -> None:
                self.wait_mode = wait_mode
                self.cancel_calls = 0
                self.output: object | None = None

            async def wait(self, timeout: float | None = None) -> Job:
                if self.wait_mode == "timeout":
                    raise TimeoutError("wait timed out")
                if self.wait_mode == "error":
                    raise RuntimeError("wait failed")
                if self.wait_mode == "failed":
                    self.error = "worker failed"
                    return self
                return self

            async def cancel(self) -> Job:
                self.cancel_calls += 1
                return self

        for wait_mode in ("timeout", "error", "failed"):
            with self.subTest(wait_mode=wait_mode):
                job = Job(wait_mode)

                class Endpoint:
                    def __init__(self, **kwargs: Any) -> None:
                        pass

                    async def run(self, input_data: Any) -> Job:
                        return job

                runner = FlashEndpointBatchRunner(
                    request_config,
                    endpoint_factory=Endpoint,
                )
                with self.assertRaises(FlashScoringError):
                    runner.run_batch_sync(request)
                self.assertEqual(job.cancel_calls, 1)

    def test_endpoint_run_errors_and_malformed_or_missing_output_are_fatal(self) -> None:
        config = RunpodFlashConfig(endpoint_id="endpoint-id")
        request = encode_flash_request(
            "query",
            ({"hash": sha256_bytes("text"), "text": "text"},),
            schema_version=WIRE_SCHEMA_VERSION,
            scorer_fingerprint=config.fingerprint,
        )

        class EndpointRunError:
            def __init__(self, **kwargs: Any) -> None:
                pass

            async def run(self, input_data: Any) -> object:
                raise RuntimeError("run failed")

        with self.assertRaises(FlashScoringError):
            FlashEndpointBatchRunner(config, endpoint_factory=EndpointRunError).run_batch_sync(request)

        for output in (None, {"scores": [{"hash": "bad"}]}):
            with self.subTest(output=output):
                class Job:
                    async def wait(self, timeout: float | None = None) -> Job:
                        return self

                    async def cancel(self) -> Job:
                        return self

                    @property
                    def output(self) -> object:
                        return output

                class Endpoint:
                    def __init__(self, **kwargs: Any) -> None:
                        pass

                    async def run(self, input_data: Any) -> Job:
                        return Job()

                with self.assertRaises(FlashScoringError):
                    FlashEndpointBatchRunner(config, endpoint_factory=Endpoint).run_batch_sync(request)

    def test_endpoint_runner_cancels_job_when_its_async_call_is_cancelled(self) -> None:
        config = RunpodFlashConfig(endpoint_id="endpoint-id")
        started = threading.Event()
        # The fake endpoint is only used inside the explicitly created async test host.
        class Job:
            cancel_calls = 0

            async def wait(self, timeout: float | None = None) -> Job:
                started.set()
                await asyncio.sleep(60.0)
                return self

            async def cancel(self) -> Job:
                self.cancel_calls += 1
                return self

        job = Job()

        class Endpoint:
            def __init__(self, **kwargs: Any) -> None:
                pass

            async def run(self, input_data: Any) -> Job:
                return job

        request = encode_flash_request(
            "query",
            ({"hash": sha256_bytes("text"), "text": "text"},),
            schema_version=WIRE_SCHEMA_VERSION,
            scorer_fingerprint=config.fingerprint,
        )

        async def cancel_call() -> None:
            runner = FlashEndpointBatchRunner(config, endpoint_factory=Endpoint)
            task = asyncio.create_task(runner.run_batch(request))
            while not started.is_set():
                await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(cancel_call())
        self.assertEqual(job.cancel_calls, 1)

    def test_endpoint_dependency_is_lazy_at_module_import(self) -> None:
        self.assertNotIn("runpod_flash", sys.modules)


class FlashWorkerTests(unittest.TestCase):
    def test_generated_worker_has_one_public_kwargs_endpoint_and_fake_decorator_executes_it(self) -> None:
        policy = LocalCrossEncoderConfig(max_length=100, query_format="Q: {query}")
        captured: dict[str, Any] = {}

        def endpoint_factory(worker_class: type[Any]) -> type[Any]:
            captured["class"] = worker_class
            return worker_class

        class Tokenizer:
            def __call__(self, query: str, text: str, **kwargs: Any) -> dict[str, list[list[int]]]:
                return {"input_ids": [[1] * (len(query) + len(text))]}

        class Model:
            tokenizer = Tokenizer()

            def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> list[float]:
                return [10.5 if text == "text" else -2.25 for _, text in pairs]

        Worker = build_flash_worker(policy, endpoint_factory=endpoint_factory)
        self.assertIs(Worker, captured["class"])
        public_methods = [
            name
            for name, method in inspect.getmembers(Worker, inspect.isfunction)
            if not name.startswith("_")
        ]
        self.assertEqual(public_methods, ["score"])
        self.assertEqual(
            list(inspect.signature(Worker.score).parameters),
            ["self", "query", "rows", "schema_version", "scorer_fingerprint"],
        )

        worker = Worker(loader=lambda config: (Model(), Tokenizer()))
        response = worker.score(
            query="query",
            rows=[{"hash": sha256_bytes("text").hex(), "text": "text"}],
            schema_version=WIRE_SCHEMA_VERSION,
            scorer_fingerprint=policy.fingerprint,
        )
        self.assertEqual(response["schema_version"], WIRE_SCHEMA_VERSION)
        self.assertEqual(response["scorer_fingerprint"], policy.fingerprint)
        self.assertEqual(response["scores"], [{"hash": sha256_bytes("text").hex(), "score": 10.5}])

    def test_extracted_worker_source_preserves_non_default_model_policy(self) -> None:
        policy = LocalCrossEncoderConfig(
            model_name="fake/model",
            revision="revision-v2",
            sentence_transformers_version="5.7.0",
            device="cuda",
            precision="float16",
            batch_size=3,
            max_length=77,
            query_format="Q: {query}",
            query_policy="q-policy-v2",
        )
        loader_configs: list[LocalCrossEncoderConfig] = []

        class Tokenizer:
            def __call__(self, query: str, text: str, **kwargs: Any) -> dict[str, list[list[int]]]:
                return {"input_ids": [[1] * (len(query) + len(text))]}

        class Model:
            def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> list[float]:
                return [3.5 for _ in pairs]

        def loader(config: LocalCrossEncoderConfig) -> object:
            loader_configs.append(config)
            return Model(), Tokenizer()

        Worker = build_flash_worker(policy)
        source = textwrap.dedent(inspect.getsource(Worker))
        self.assertIn("_MODEL_CONFIG", source)
        class_source = source[source.index("class FlashPassageWorker:") :]
        namespace: dict[str, object] = {}
        exec(compile(class_source, "<extracted-flash-worker>", "exec"), namespace)
        ExtractedWorker = namespace["FlashPassageWorker"]
        extracted = ExtractedWorker(loader=loader)

        response = extracted.score(
            query="query",
            rows=[{"hash": sha256_bytes("text").hex(), "text": "text"}],
            schema_version=WIRE_SCHEMA_VERSION,
            scorer_fingerprint=policy.fingerprint,
        )

        self.assertEqual(loader_configs, [policy])
        self.assertEqual(response["scorer_fingerprint"], policy.fingerprint)
        self.assertEqual(response["scores"], [{"hash": sha256_bytes("text").hex(), "score": 3.5}])

    def test_same_effective_policy_reuses_source_stable_picklable_class(self) -> None:
        policy = LocalCrossEncoderConfig(
            model_name="cache/model",
            revision="cache-revision",
            device="cuda",
            precision="float16",
            batch_size=3,
            max_length=77,
            query_format="Q: {query}",
            query_policy="cache-policy",
        )
        first = build_flash_worker(policy)
        first_source = inspect.getsource(first)
        first_module = sys.modules[first.__module__]

        second = build_flash_worker(
            RunpodFlashConfig(endpoint_id="independent-transport", model_config=policy)
        )

        self.assertIs(first, second)
        self.assertIs(sys.modules[first.__module__], first_module)
        self.assertEqual(inspect.getsource(first), first_source)
        self.assertEqual(linecache.getlines(first_module.__file__), first_source.splitlines(True))
        self.assertIs(pickle.loads(pickle.dumps(first)), first)

    def test_cached_worker_instances_keep_independent_loaders_and_warm_once(self) -> None:
        policy = LocalCrossEncoderConfig(max_length=100, query_policy="loader-isolation")
        Worker = build_flash_worker(policy)
        load_calls: list[list[LocalCrossEncoderConfig]] = [[], []]

        class Tokenizer:
            def __call__(self, query: str, text: str, **kwargs: Any) -> dict[str, list[list[int]]]:
                return {"input_ids": [[1] * 2]}

        class Model:
            def __init__(self, score: float) -> None:
                self.score_value = score
                self.predict_calls = 0

            def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> list[float]:
                self.predict_calls += 1
                return [self.score_value for _ in pairs]

        models = [Model(1.0), Model(2.0)]

        def loader_at(index: int) -> Callable[[LocalCrossEncoderConfig], object]:
            def loader(config: LocalCrossEncoderConfig) -> object:
                load_calls[index].append(config)
                return models[index], Tokenizer()

            return loader

        first = Worker(loader=loader_at(0))
        second = Worker(loader=loader_at(1))
        request = {
            "query": "query",
            "rows": [{"hash": sha256_bytes("text").hex(), "text": "text"}],
            "schema_version": WIRE_SCHEMA_VERSION,
            "scorer_fingerprint": policy.fingerprint,
        }

        expected_first = [{"hash": sha256_bytes("text").hex(), "score": 1.0}]
        expected_second = [{"hash": sha256_bytes("text").hex(), "score": 2.0}]
        self.assertEqual(first.score(**request)["scores"], expected_first)
        self.assertEqual(first.score(**request)["scores"], expected_first)
        self.assertEqual(second.score(**request)["scores"], expected_second)
        self.assertEqual(load_calls, [[policy], [policy]])
        self.assertEqual(models[0].predict_calls, 2)
        self.assertEqual(models[1].predict_calls, 1)

    def test_worker_rejects_schema_or_fingerprint_mismatch(self) -> None:
        policy = LocalCrossEncoderConfig(max_length=100)
        Worker = build_flash_worker(policy)
        worker = Worker(
            loader=lambda config: (
                types.SimpleNamespace(
                    tokenizer=lambda *args, **kwargs: {"input_ids": [[1, 2]]},
                    predict=lambda *args, **kwargs: [1.0],
                ),
                lambda *args, **kwargs: {"input_ids": [[1, 2]]},
            )
        )
        kwargs = {
            "query": "query",
            "rows": [{"hash": sha256_bytes("text").hex(), "text": "text"}],
            "schema_version": WIRE_SCHEMA_VERSION,
            "scorer_fingerprint": policy.fingerprint,
        }
        with self.assertRaises(FlashScoringError):
            worker.score(**{**kwargs, "schema_version": WIRE_SCHEMA_VERSION + 1})
        with self.assertRaises(FlashScoringError):
            worker.score(**{**kwargs, "scorer_fingerprint": "wrong-policy"})

    def test_worker_and_local_scorer_share_fake_raw_logits_formatting_kwargs_and_overflow(self) -> None:
        policy = LocalCrossEncoderConfig(
            max_length=10,
            batch_size=1,
            query_format="Q: {query}",
        )

        class RecordingTokenizer:
            def __init__(self) -> None:
                self.calls: list[tuple[str, str, dict[str, Any]]] = []

            def __call__(self, query: str, text: str, **kwargs: Any) -> dict[str, list[list[int]]]:
                self.calls.append((query, text, kwargs))
                count = 100 if text == "overflow" else 3
                return {"input_ids": [[1] * count]}

        class RecordingModel:
            def __init__(self, tokenizer: RecordingTokenizer) -> None:
                self.tokenizer = tokenizer
                self.predict_calls: list[tuple[list[tuple[str, str]], dict[str, Any]]] = []

            def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> list[float]:
                self.predict_calls.append((pairs, kwargs))
                return [8.25 for _ in pairs]

        local_tokenizer = RecordingTokenizer()
        local_model = RecordingModel(local_tokenizer)
        worker_tokenizer = RecordingTokenizer()
        worker_model = RecordingModel(worker_tokenizer)
        local = LocalCrossEncoderScorer(policy, loader=lambda config: (local_model, local_tokenizer))
        Worker = build_flash_worker(policy)
        worker = Worker(loader=lambda config: (worker_model, worker_tokenizer))
        passage = _passage("text")

        local_scores = local.score("query", (passage,))
        worker_response = worker.score(
            query="query",
            rows=[{"hash": passage.content_hash.hex(), "text": passage.text}],
            schema_version=WIRE_SCHEMA_VERSION,
            scorer_fingerprint=policy.fingerprint,
        )

        self.assertEqual(local_scores, {passage.content_hash: 8.25})
        self.assertEqual(worker_response["scores"], [{"hash": passage.content_hash.hex(), "score": 8.25}])
        for model, tokenizer in (
            (local_model, local_tokenizer),
            (worker_model, worker_tokenizer),
        ):
            self.assertEqual(tokenizer.calls[0][0:2], ("Q: query", "text"))
            self.assertIs(tokenizer.calls[0][2]["truncation"], False)
            self.assertEqual(tokenizer.calls[0][2]["max_length"], 10)
            self.assertEqual(model.predict_calls[0][0], [("Q: query", "text")])
            self.assertFalse(model.predict_calls[0][1]["apply_softmax"])
            self.assertTrue(model.predict_calls[0][1]["convert_to_numpy"])
            self.assertFalse(model.predict_calls[0][1]["convert_to_tensor"])

        overflowing = _passage("overflow")
        with self.assertRaises(PassageScoringError):
            local.score("query", (overflowing,))
        with self.assertRaises(FlashScoringError):
            worker.score(
                query="query",
                rows=[{"hash": overflowing.content_hash.hex(), "text": overflowing.text}],
                schema_version=WIRE_SCHEMA_VERSION,
                scorer_fingerprint=policy.fingerprint,
            )

    def test_z_worker_class_cache_refuses_growth_without_invalidating_prior_identity(self) -> None:
        cache_size = len(flash_worker_module._WORKER_CLASS_CACHE)
        stable_policy = LocalCrossEncoderConfig(query_policy=f"stable-cache-{cache_size}")
        stable_class = build_flash_worker(stable_policy)
        stable_source = inspect.getsource(stable_class)
        stable_module = sys.modules[stable_class.__module__]
        cache_limit = cache_size + 2

        with patch.object(flash_worker_module, "_WORKER_CLASS_CACHE_MAX", cache_limit):
            while len(flash_worker_module._WORKER_CLASS_CACHE) < cache_limit:
                policy = LocalCrossEncoderConfig(
                    query_policy=f"cache-cap-{cache_size}-{len(flash_worker_module._WORKER_CLASS_CACHE)}"
                )
                build_flash_worker(policy)

            self.assertEqual(len(flash_worker_module._WORKER_CLASS_CACHE), cache_limit)
            with self.assertRaises(FlashScoringError):
                build_flash_worker(
                    LocalCrossEncoderConfig(query_policy=f"refused-cache-{cache_size}")
                )

            self.assertIs(build_flash_worker(stable_policy), stable_class)
            self.assertIs(sys.modules[stable_class.__module__], stable_module)
            self.assertEqual(inspect.getsource(stable_class), stable_source)
            self.assertEqual(linecache.getlines(stable_module.__file__), stable_source.splitlines(True))
            self.assertIs(pickle.loads(pickle.dumps(stable_class)), stable_class)


if __name__ == "__main__":
    unittest.main()
