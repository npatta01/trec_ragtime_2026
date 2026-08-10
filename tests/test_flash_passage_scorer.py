from __future__ import annotations

import asyncio
import contextlib
import io
import json
import math
import sys
import types
import unittest
from dataclasses import replace
from typing import Any

from trec_ragtime.passages import Passage, sha256_bytes
from trec_ragtime.passage_scoring import (
    LocalCrossEncoderConfig,
    LocalCrossEncoderScorer,
)
from trec_ragtime.flash_passage_scorer import (
    FLASH_PAYLOAD_LIMIT_BYTES,
    FlashEndpointBatchRunner,
    FlashScoringError,
    RunpodFlashConfig,
    RunpodFlashPassageScorer,
    decode_flash_request,
    decode_flash_response,
    encode_flash_request,
    encode_flash_response,
)
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
            return encode_flash_response(rows)
        finally:
            self.active -= 1


class FlashWireCodecTests(unittest.TestCase):
    def test_request_rows_are_deterministic_and_measure_final_utf8_bytes(self) -> None:
        rows = (
            {"hash": sha256_bytes("é漢字"), "text": "é漢字"},
            {"hash": sha256_bytes("plain"), "text": "plain"},
        )

        first = encode_flash_request("q-é", rows)
        second = encode_flash_request("q-é", rows)

        self.assertEqual(first, second)
        self.assertEqual(first, first.decode("utf-8").encode("utf-8"))
        decoded = decode_flash_request(first)
        self.assertEqual(decoded["query"], "q-é")
        self.assertEqual(decoded["rows"][0], {"hash": sha256_bytes("é漢字").hex(), "text": "é漢字"})
        self.assertLess(len(first), FLASH_PAYLOAD_LIMIT_BYTES)

    def test_wire_decoders_reject_unknown_fields_malformed_hashes_and_non_finite_scores(self) -> None:
        with self.assertRaises(FlashScoringError):
            decode_flash_request(b'{"query":"q","rows":[],"extra":1}')
        with self.assertRaises(FlashScoringError):
            decode_flash_request(b'{"query":"q","rows":[{"hash":"bad","text":"x"}]}')
        with self.assertRaises(FlashScoringError):
            decode_flash_response(b'{"scores":[{"hash":"' + b"00" * 32 + b'","score":NaN}]}')
        with self.assertRaises(FlashScoringError):
            decode_flash_response(b'{"scores":[{"hash":"' + b"00" * 32 + b'","score":1}],"extra":1}')


class RunpodFlashConfigTests(unittest.TestCase):
    def test_result_fingerprint_excludes_transport_only_settings(self) -> None:
        base = RunpodFlashConfig(
            endpoint_id="endpoint-a",
            model_config=LocalCrossEncoderConfig(sentence_transformers_version="5.7.0"),
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
        )
        model_variant = replace(base, precision="float16")

        self.assertEqual(base.fingerprint, transport_variant.fingerprint)
        self.assertNotEqual(base.fingerprint, model_variant.fingerprint)


class RunpodFlashPassageScorerTests(unittest.TestCase):
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
            max_payload_bytes=220,
        )
        runner = FakeAsyncRunner()
        scorer = RunpodFlashPassageScorer(config, runner)

        scorer.score("query-é", passages)

        self.assertGreater(len(runner.payloads), 1)
        self.assertTrue(all(len(payload) < 220 for payload in runner.payloads))
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
                RunpodFlashConfig(endpoint_id="fake", auth_token=token), runner
            ).score(query, (passage,))

        combined = output.getvalue() + error_output.getvalue()
        for secret in (query, text_value, token, passage.content_hash.hex()):
            self.assertNotIn(secret, combined)

    def test_local_and_remote_fake_models_have_exact_score_parity(self) -> None:
        passages = (_passage("a", document_id="a"), _passage("longer", document_id="b"))

        class Tokenizer:
            def __call__(self, query: str, text: str, **kwargs: Any) -> dict[str, list[list[int]]]:
                return {"input_ids": [[1] * (len(query) + len(text))]}

        class Model:
            def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> list[float]:
                return [float(len(text)) for _, text in pairs]

        local = LocalCrossEncoderScorer(
            LocalCrossEncoderConfig(max_length=100),
            loader=lambda config: (Model(), Tokenizer()),
        )
        remote = RunpodFlashPassageScorer(
            RunpodFlashConfig(endpoint_id="fake", model_config=LocalCrossEncoderConfig(max_length=100)),
            FakeAsyncRunner(),
        )

        self.assertEqual(local.score("query", passages), remote.score("query", passages))


class EndpointRunnerTests(unittest.TestCase):
    def test_endpoint_runner_uses_async_run_then_job_wait(self) -> None:
        calls: list[tuple[str, Any]] = []

        class Job:
            output = {"scores": [{"hash": sha256_bytes("text").hex(), "score": 4.0}]}

            async def wait(self, timeout: float | None = None) -> Job:
                calls.append(("wait", timeout))
                return self

        class Endpoint:
            def __init__(self, **kwargs: Any) -> None:
                calls.append(("init", kwargs))

            async def run(self, input_data: Any) -> Job:
                calls.append(("run", input_data))
                return Job()

        request = encode_flash_request("query", ({"hash": sha256_bytes("text"), "text": "text"},))
        runner = FlashEndpointBatchRunner(
            RunpodFlashConfig(endpoint_id="endpoint-id", request_timeout=3.0),
            endpoint_factory=Endpoint,
        )

        response = runner.run_batch_sync(request)

        self.assertEqual(decode_flash_response(response), {"scores": [{"hash": sha256_bytes("text").hex(), "score": 4.0}]})
        self.assertEqual(calls[0], ("init", {"id": "endpoint-id"}))
        self.assertEqual(calls[1][0], "run")
        self.assertEqual(calls[2], ("wait", 3.0))

    def test_endpoint_dependency_is_lazy_at_module_import(self) -> None:
        self.assertNotIn("runpod_flash", sys.modules)


class FlashWorkerTests(unittest.TestCase):
    def test_class_worker_loads_once_and_uses_task_two_raw_logit_policy(self) -> None:
        load_calls = 0

        class Tokenizer:
            def __call__(self, query: str, text: str, **kwargs: Any) -> dict[str, list[list[int]]]:
                return {"input_ids": [[1] * (len(query) + len(text))]}

        class Model:
            tokenizer = Tokenizer()

            def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> list[float]:
                return [float(len(text)) for _, text in pairs]

        def loader(config: LocalCrossEncoderConfig) -> tuple[Model, Tokenizer]:
            nonlocal load_calls
            load_calls += 1
            model = Model()
            return model, model.tokenizer

        Worker = build_flash_worker(
            LocalCrossEncoderConfig(max_length=100),
            loader=loader,
        )
        worker = Worker()
        request = encode_flash_request(
            "query",
            ({"hash": sha256_bytes("text"), "text": "text"},),
        )

        response = worker(request)
        worker(request)

        self.assertEqual(load_calls, 1)
        self.assertEqual(decode_flash_response(response)["scores"][0]["score"], 4.0)


if __name__ == "__main__":
    unittest.main()
