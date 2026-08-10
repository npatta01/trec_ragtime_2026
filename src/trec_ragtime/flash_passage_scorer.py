"""Optional Runpod Flash transport for the passage-scoring protocol."""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import numbers
import threading
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, cast

from .passage_scoring import (
    LocalCrossEncoderConfig,
    PassageScoringError,
    _unique_passages,
    validate_passage_scores,
)
from .passages import Passage, sha256_bytes


WIRE_SCHEMA_VERSION = 1
FLASH_PAYLOAD_LIMIT_BYTES = 10_000_000
DEFAULT_FLASH_BATCH_SIZE = 32
DEFAULT_FLASH_CONCURRENCY = 4
DEFAULT_FLASH_REQUEST_TIMEOUT = 60.0
DEFAULT_FLASH_IDLE_TIMEOUT = 300.0


class FlashScoringError(PassageScoringError):
    """Raised for invalid Flash wire data, transport failures, or timeouts."""


class FlashWireValidationError(FlashScoringError):
    """Raised when a Flash request or response is not strictly well-formed."""


def _validate_schema_version(value: object) -> int:
    if value != WIRE_SCHEMA_VERSION or isinstance(value, bool) or not isinstance(value, int):
        raise FlashWireValidationError("unsupported Flash wire schema version")
    return value


def _validate_fingerprint(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(character in value for character in "\r\n")
    ):
        raise FlashWireValidationError("scorer_fingerprint must be a non-blank string")
    return value


def _json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise FlashWireValidationError("wire payload is not valid compact UTF-8 JSON") from error


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def _json_object(payload: bytes | bytearray | str | Mapping[str, object]) -> dict[str, object]:
    if isinstance(payload, Mapping):
        value: object = dict(payload)
    else:
        if isinstance(payload, str):
            encoded = payload.encode("utf-8")
        elif isinstance(payload, (bytes, bytearray)):
            encoded = bytes(payload)
        else:
            raise FlashWireValidationError("wire payload must be JSON bytes or an object")
        try:
            value = json.loads(encoded.decode("utf-8"), parse_constant=_reject_json_constant)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise FlashWireValidationError("wire payload is not valid JSON") from error
    if not isinstance(value, dict):
        raise FlashWireValidationError("wire payload must be a JSON object")
    return cast(dict[str, object], value)


def _validate_hash_hex(value: object, field_name: str = "hash") -> str:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise FlashWireValidationError(f"{field_name} must be lowercase hexadecimal SHA-256")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as error:
        raise FlashWireValidationError(f"{field_name} must be lowercase hexadecimal SHA-256") from error
    if len(decoded) != 32:
        raise FlashWireValidationError(f"{field_name} must be exactly 32 bytes")
    return value


def _hash_text_value(value: object, field_name: str = "hash") -> str:
    if isinstance(value, bytes):
        if len(value) != 32:
            raise FlashWireValidationError(f"{field_name} must be exactly 32 bytes")
        return value.hex()
    return _validate_hash_hex(value, field_name)


def _score_value(value: object, field_name: str = "score") -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise FlashWireValidationError(f"{field_name} must be a finite non-Boolean number")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise FlashWireValidationError(f"{field_name} must be a finite number") from error
    if not math.isfinite(normalized):
        raise FlashWireValidationError(f"{field_name} must be finite")
    return normalized


def _request_rows(rows: object) -> list[dict[str, str]]:
    if isinstance(rows, (str, bytes, bytearray)) or not isinstance(rows, Iterable):
        raise FlashWireValidationError("rows must be an iterable")
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if isinstance(row, Mapping):
            if set(row) != {"hash", "text"}:
                raise FlashWireValidationError("request rows must contain exactly hash and text")
            raw_hash = row["hash"]
            text = row["text"]
        elif isinstance(row, (tuple, list)) and len(row) == 2:
            raw_hash, text = row
        else:
            raise FlashWireValidationError("request rows must be mappings or hash/text pairs")
        hash_hex = _hash_text_value(raw_hash)
        if not isinstance(text, str) or not text:
            raise FlashWireValidationError("request row text must be a non-empty string")
        if bytes.fromhex(hash_hex) != sha256_bytes(text):
            raise FlashWireValidationError("request hash does not match request text")
        if hash_hex in seen:
            raise FlashWireValidationError("request rows must have unique hashes")
        seen.add(hash_hex)
        normalized.append({"hash": hash_hex, "text": text})
    return normalized


def _response_rows(rows: object) -> list[dict[str, object]]:
    if isinstance(rows, (str, bytes, bytearray)) or not isinstance(rows, Iterable):
        raise FlashWireValidationError("scores must be an iterable")
    normalized: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"hash", "score"}:
            raise FlashWireValidationError("response rows must contain exactly hash and score")
        hash_hex = _hash_text_value(row["hash"])
        if hash_hex in seen:
            raise FlashWireValidationError("response rows must have unique hashes")
        seen.add(hash_hex)
        normalized.append({"hash": hash_hex, "score": _score_value(row["score"])})
    return normalized


def encode_flash_request(
    query: str,
    rows: Iterable[Mapping[str, object] | Sequence[object]],
    *,
    schema_version: int = WIRE_SCHEMA_VERSION,
    scorer_fingerprint: str,
) -> bytes:
    """Encode a strict, identity-bearing compact request."""

    _validate_schema_version(schema_version)
    _validate_fingerprint(scorer_fingerprint)
    if not isinstance(query, str) or not query.strip():
        raise FlashWireValidationError("query must be a non-blank string")
    return _json_bytes(
        {
            "query": query,
            "rows": _request_rows(rows),
            "schema_version": schema_version,
            "scorer_fingerprint": scorer_fingerprint,
        }
    )


def decode_flash_request(
    payload: bytes | bytearray | str | Mapping[str, object],
) -> dict[str, object]:
    """Decode and validate a strict identity-bearing request."""

    value = _json_object(payload)
    if set(value) != {"query", "rows", "schema_version", "scorer_fingerprint"}:
        raise FlashWireValidationError(
            "request must contain exactly schema_version, scorer_fingerprint, query, and rows"
        )
    schema_version = _validate_schema_version(value["schema_version"])
    scorer_fingerprint = _validate_fingerprint(value["scorer_fingerprint"])
    query = value["query"]
    if not isinstance(query, str) or not query.strip():
        raise FlashWireValidationError("query must be a non-blank string")
    return {
        "schema_version": schema_version,
        "scorer_fingerprint": scorer_fingerprint,
        "query": query,
        "rows": _request_rows(value["rows"]),
    }


def encode_flash_response(
    rows: Iterable[Mapping[str, object] | Sequence[object]],
    *,
    schema_version: int = WIRE_SCHEMA_VERSION,
    scorer_fingerprint: str,
) -> bytes:
    """Encode a strict identity-bearing compact score response."""

    _validate_schema_version(schema_version)
    _validate_fingerprint(scorer_fingerprint)
    return _json_bytes(
        {
            "scores": _response_rows(rows),
            "schema_version": schema_version,
            "scorer_fingerprint": scorer_fingerprint,
        }
    )


def decode_flash_response(
    payload: bytes | bytearray | str | Mapping[str, object],
) -> dict[str, object]:
    """Decode and validate a strict identity-bearing score response."""

    value = _json_object(payload)
    if set(value) != {"scores", "schema_version", "scorer_fingerprint"}:
        raise FlashWireValidationError(
            "response must contain exactly schema_version, scorer_fingerprint, and scores"
        )
    return {
        "schema_version": _validate_schema_version(value["schema_version"]),
        "scorer_fingerprint": _validate_fingerprint(value["scorer_fingerprint"]),
        "scores": _response_rows(value["scores"]),
    }


encode_flash_batch = encode_flash_request
decode_flash_batch = decode_flash_request
encode_flash_batch_response = encode_flash_response
decode_flash_batch_response = decode_flash_response


def _effective_model_config(config: "RunpodFlashConfig") -> LocalCrossEncoderConfig:
    updates: dict[str, object] = {}
    for field_name in (
        "model_name",
        "revision",
        "sentence_transformers_version",
        "device",
        "precision",
        "max_length",
        "query_format",
        "query_policy",
    ):
        value = getattr(config, field_name)
        if value is not None:
            updates[field_name] = value
    if config.batch_size is not None:
        updates["batch_size"] = config.batch_size
    if not updates:
        return config.model_config
    return replace(config.model_config, **updates)


@dataclass(frozen=True, slots=True)
class RunpodFlashConfig:
    """Immutable Flash transport settings and effective local model policy."""

    endpoint_id: str
    model_config: LocalCrossEncoderConfig = field(default_factory=LocalCrossEncoderConfig)
    batch_size: int | None = None
    max_payload_bytes: int = FLASH_PAYLOAD_LIMIT_BYTES
    concurrency: int = DEFAULT_FLASH_CONCURRENCY
    request_timeout: float = DEFAULT_FLASH_REQUEST_TIMEOUT
    workers: int = 1
    idle_timeout: float = DEFAULT_FLASH_IDLE_TIMEOUT
    model_name: str | None = None
    revision: str | None = None
    sentence_transformers_version: str | None = None
    device: str | None = None
    precision: str | None = None
    max_length: int | None = None
    query_format: str | None = None
    query_policy: str | None = None
    _fingerprint: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint_id, str) or not self.endpoint_id.strip():
            raise ValueError("endpoint_id must be a non-blank string")
        if not isinstance(self.model_config, LocalCrossEncoderConfig):
            raise TypeError("model_config must be a LocalCrossEncoderConfig")
        if self.batch_size is not None and (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer when provided")
        for field_name in ("max_payload_bytes", "concurrency", "workers"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.max_payload_bytes > FLASH_PAYLOAD_LIMIT_BYTES:
            raise ValueError("max_payload_bytes cannot exceed the Flash 10 MB limit")
        for field_name in ("request_timeout", "idle_timeout"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise ValueError(f"{field_name} must be a positive number")
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{field_name} must be a positive finite number")
        for field_name in (
            "model_name",
            "revision",
            "sentence_transformers_version",
            "device",
            "precision",
            "query_format",
            "query_policy",
        ):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field_name} must be a non-blank string when provided")
        if self.max_length is not None and (
            isinstance(self.max_length, bool)
            or not isinstance(self.max_length, int)
            or self.max_length <= 0
        ):
            raise ValueError("max_length must be a positive integer when provided")
        object.__setattr__(self, "_fingerprint", self.effective_model_config.fingerprint)

    @property
    def effective_model_config(self) -> LocalCrossEncoderConfig:
        return _effective_model_config(self)

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def request_timeout_seconds(self) -> float:
        return self.request_timeout

    @property
    def idle_timeout_seconds(self) -> float:
        return self.idle_timeout

    @property
    def worker_count(self) -> int:
        return self.workers

    @property
    def effective_batch_size(self) -> int:
        return self.effective_model_config.batch_size


class _EventLoopBridge:
    """Own one private loop thread and quiesce it deterministically."""

    def __init__(self, *, startup_timeout: float = 5.0) -> None:
        self._ready = threading.Event()
        self._closed = False
        self._startup_error: BaseException | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="trec-ragtime-flash-loop",
            daemon=False,
        )
        self._thread.start()
        if not self._ready.wait(startup_timeout):
            self._closed = True
            self._thread.join()
            raise FlashScoringError("Flash event-loop thread did not start")
        if self._startup_error is not None:
            error = self._startup_error
            self._closed = True
            self._thread.join()
            raise FlashScoringError("Flash event-loop thread failed to start") from error
        if self._loop is None:
            self._closed = True
            self._thread.join()
            raise FlashScoringError("Flash event-loop thread did not publish a loop")

    def _run(self) -> None:
        try:
            loop = asyncio.new_event_loop()
        except BaseException as error:
            self._startup_error = error
            self._ready.set()
            return
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            self._quiesce_loop(loop)

    @staticmethod
    def _quiesce_loop(loop: asyncio.AbstractEventLoop) -> None:
        pending = list(asyncio.all_tasks(loop))
        for task in pending:
            task.cancel()

        if pending:
            async def drain() -> None:
                await asyncio.wait(pending, timeout=0.2)

            try:
                loop.run_until_complete(drain())
            except BaseException:
                pass

        remaining = list(asyncio.all_tasks(loop))
        for task in remaining:
            task.cancel()
            # A task that ignores cancellation cannot be awaited safely during
            # shutdown.  Closing the loop discards it; suppress asyncio's
            # pending-task destructor report for this deliberate finalization.
            log_destroy_pending = getattr(task, "_log_destroy_pending", None)
            if log_destroy_pending is not None:
                task._log_destroy_pending = False  # type: ignore[attr-defined]
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except BaseException:
            pass
        loop.close()

    @property
    def closed(self) -> bool:
        return self._closed

    def run(
        self,
        awaitable_or_factory: Awaitable[Any] | Callable[[], Awaitable[Any]],
        *,
        timeout: float | None = None,
    ) -> Any:
        if self._closed or self._loop is None:
            if inspect.iscoroutine(awaitable_or_factory):
                awaitable_or_factory.close()
            raise FlashScoringError("Flash event-loop bridge is closed")
        if threading.current_thread() is self._thread:
            raise FlashScoringError("nested synchronous use of the Flash event loop is not allowed")
        awaitable = (
            awaitable_or_factory()
            if callable(awaitable_or_factory)
            else awaitable_or_factory
        )
        if self._closed or self._loop is None:
            if inspect.iscoroutine(awaitable):
                awaitable.close()
            raise FlashScoringError("Flash event-loop bridge is closed")
        future = asyncio.run_coroutine_threadsafe(_as_coroutine(awaitable), self._loop)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            future.cancel()
            raise

    def close(self) -> None:
        if self._closed:
            if threading.current_thread() is not self._thread and self._thread.is_alive():
                self._thread.join()
            return
        self._closed = True
        if threading.current_thread() is self._thread:
            raise FlashScoringError("cannot close the Flash event loop from its own thread")
        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        self._thread.join()

    def __enter__(self) -> "_EventLoopBridge":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


async def _as_coroutine(awaitable: Awaitable[Any]) -> Any:
    return await awaitable


def _runner_awaitable(runner: object, payload: bytes) -> Awaitable[object]:
    method = getattr(runner, "run_batch", None)
    if callable(method):
        result = method(payload)
    elif callable(runner):
        result = runner(payload)  # type: ignore[operator]
    else:
        raise FlashScoringError("batch runner must be callable or expose async run_batch")
    if not hasattr(result, "__await__"):
        raise FlashScoringError("batch runner must return an awaitable")
    return cast(Awaitable[object], result)


def _response_bytes(value: object) -> bytes:
    if isinstance(value, (bytes, bytearray, str, Mapping)):
        decoded = decode_flash_response(value)
        return encode_flash_response(
            cast(Iterable[Mapping[str, object]], decoded["scores"]),
            schema_version=cast(int, decoded["schema_version"]),
            scorer_fingerprint=cast(str, decoded["scorer_fingerprint"]),
        )
    raise FlashWireValidationError("batch runner returned an unsupported response")


class RunpodFlashPassageScorer:
    """Synchronous exact scorer backed by bounded asynchronous Flash batches."""

    __slots__ = ("_config", "_runner", "_bridge")

    def __init__(self, config: RunpodFlashConfig, batch_runner: object) -> None:
        if not isinstance(config, RunpodFlashConfig):
            raise TypeError("config must be a RunpodFlashConfig")
        self._config = config
        self._runner = batch_runner
        self._bridge = _EventLoopBridge()

    @property
    def config(self) -> RunpodFlashConfig:
        return self._config

    @property
    def fingerprint(self) -> str:
        return self._config.fingerprint

    @staticmethod
    def _build_batches(
        query: str,
        unique: Sequence[tuple[bytes, str]],
        config: RunpodFlashConfig,
    ) -> tuple[tuple[bytes, frozenset[bytes]], ...]:
        batches: list[tuple[bytes, frozenset[bytes]]] = []
        current: list[tuple[bytes, str]] = []

        def flush() -> None:
            if current:
                payload = encode_flash_request(
                    query,
                    current,
                    scorer_fingerprint=config.fingerprint,
                )
                if len(payload) >= config.max_payload_bytes or len(payload) >= FLASH_PAYLOAD_LIMIT_BYTES:
                    raise FlashScoringError("Flash payload must be strictly below its byte limit")
                batches.append((payload, frozenset(content_hash for content_hash, _ in current)))
                current.clear()

        for content_hash, text in unique:
            if len(current) >= config.effective_batch_size:
                flush()
            candidate = current + [(content_hash, text)]
            candidate_payload = encode_flash_request(
                query,
                candidate,
                scorer_fingerprint=config.fingerprint,
            )
            if (
                len(candidate_payload) < config.max_payload_bytes
                and len(candidate_payload) < FLASH_PAYLOAD_LIMIT_BYTES
            ):
                current.append((content_hash, text))
                continue
            if current:
                flush()
                single_payload = encode_flash_request(
                    query,
                    [(content_hash, text)],
                    scorer_fingerprint=config.fingerprint,
                )
                if (
                    len(single_payload) >= config.max_payload_bytes
                    or len(single_payload) >= FLASH_PAYLOAD_LIMIT_BYTES
                ):
                    raise FlashScoringError("one passage exceeds the Flash JSON payload limit")
                current.append((content_hash, text))
                continue
            raise FlashScoringError("one passage exceeds the Flash JSON payload limit")
        flush()
        return tuple(batches)

    async def _run_one(
        self,
        payload: bytes,
        expected_batch: frozenset[bytes],
    ) -> dict[bytes, float]:
        try:
            raw = await asyncio.wait_for(
                _runner_awaitable(self._runner, payload),
                timeout=float(self._config.request_timeout),
            )
            response = decode_flash_response(_response_bytes(raw))
        except asyncio.TimeoutError as error:
            raise FlashScoringError("Flash batch timed out") from error
        except FlashScoringError:
            raise
        except Exception as error:
            raise FlashScoringError("Flash batch failed") from error

        if response["scorer_fingerprint"] != self._config.fingerprint:
            raise FlashScoringError("Flash response scorer fingerprint mismatch")
        rows = cast(list[dict[str, object]], response["scores"])
        batch_scores = {
            bytes.fromhex(cast(str, row["hash"])): cast(float, row["score"])
            for row in rows
        }
        if set(batch_scores) != set(expected_batch):
            raise FlashScoringError("Flash response coverage does not match its batch")
        return batch_scores

    @staticmethod
    async def _cancel_and_await(tasks: Sequence[asyncio.Task[Any]]) -> None:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _score_async(
        self,
        passages: Sequence[Passage],
        batches: Sequence[tuple[bytes, frozenset[bytes]]],
    ) -> dict[bytes, float]:
        batch_results: list[dict[bytes, float]] = []
        for start in range(0, len(batches), self._config.concurrency):
            wave = batches[start : start + self._config.concurrency]
            tasks = [
                asyncio.create_task(self._run_one(payload, expected_batch))
                for payload, expected_batch in wave
            ]
            try:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
                failed = next(
                    (
                        task
                        for task in done
                        if not task.cancelled() and task.exception() is not None
                    ),
                    None,
                )
                if failed is not None:
                    await self._cancel_and_await(tasks)
                    failed.result()
                batch_results.extend(task.result() for task in tasks)
            except BaseException:
                try:
                    await self._cancel_and_await(tasks)
                except BaseException:
                    pass
                raise

        merged: dict[bytes, float] = {}
        for result in batch_results:
            overlap = set(result) & set(merged)
            if overlap:
                raise FlashScoringError("Flash batches returned duplicate score hashes")
            merged.update(result)
        expected_global = {
            content_hash
            for _, expected_batch in batches
            for content_hash in expected_batch
        }
        if set(merged) != expected_global:
            raise FlashScoringError("Flash response coverage does not match the global request")
        try:
            return validate_passage_scores(passages, merged)
        except PassageScoringError as error:
            raise FlashScoringError("Flash scores failed the passage scorer contract") from error

    def score(self, query: str, passages: Sequence[Passage]) -> Mapping[bytes, float]:
        if self._bridge.closed:
            raise FlashScoringError("Flash scorer is closed")
        if not isinstance(query, str) or not query.strip():
            raise FlashScoringError("query must be a non-blank string")
        try:
            unique = _unique_passages(passages)
        except PassageScoringError as error:
            raise FlashScoringError("passages failed the passage scorer contract") from error
        if not unique:
            return {}
        batches = self._build_batches(query, unique, self._config)
        try:
            return self._bridge.run(
                lambda: self._score_async(passages, batches),
                timeout=float(self._config.request_timeout) * max(1, len(batches)) + 1.0,
            )
        except FlashScoringError:
            raise
        except TimeoutError as error:
            raise FlashScoringError("Flash scoring timed out") from error
        except Exception as error:
            raise FlashScoringError("Flash scoring failed") from error

    def close(self) -> None:
        self._bridge.close()

    def __enter__(self) -> "RunpodFlashPassageScorer":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class FlashEndpointBatchRunner:
    """Lazy optional Runpod Flash endpoint client using run plus wait."""

    __slots__ = ("_config", "_endpoint_factory", "_endpoint", "_bridge")

    def __init__(
        self,
        config: RunpodFlashConfig,
        *,
        endpoint_factory: Callable[..., object] | None = None,
        endpoint: object | None = None,
    ) -> None:
        if not isinstance(config, RunpodFlashConfig):
            raise TypeError("config must be a RunpodFlashConfig")
        if endpoint_factory is not None and endpoint is not None:
            raise TypeError("pass endpoint_factory or endpoint, not both")
        self._config = config
        self._endpoint_factory = endpoint_factory
        self._endpoint = endpoint
        self._bridge: _EventLoopBridge | None = None

    async def _endpoint_instance(self) -> object:
        if self._endpoint is not None:
            return self._endpoint
        factory = self._endpoint_factory
        if factory is None:
            try:
                from runpod_flash import Endpoint
            except ImportError as error:
                raise FlashScoringError(
                    "Runpod Flash is optional; install the runpod extra to use an endpoint"
                ) from error
            factory = Endpoint
        self._endpoint = factory(id=self._config.endpoint_id)
        return self._endpoint

    @staticmethod
    def _job_is_terminal(job: object) -> bool:
        done = getattr(job, "done", None)
        if isinstance(done, bool):
            return done
        status = getattr(job, "status", None)
        return status in {"COMPLETED", "FAILED", "CANCELLED"}

    async def _cancel_job_best_effort(self, job: object) -> None:
        cancel = getattr(job, "cancel", None)
        if not callable(cancel):
            return
        try:
            result = cancel()
            if hasattr(result, "__await__"):
                await cast(Awaitable[Any], result)
        except BaseException:
            pass

    async def run_batch(self, payload: bytes) -> bytes:
        request = decode_flash_request(payload)
        if request["scorer_fingerprint"] != self._config.fingerprint:
            raise FlashScoringError("Flash request scorer fingerprint mismatch")
        job: object | None = None
        wait_completed = False
        try:
            endpoint = await self._endpoint_instance()
            run = getattr(endpoint, "run", None)
            if not callable(run):
                raise FlashScoringError("Flash endpoint does not expose async run")
            job = await run(request)
            wait = getattr(job, "wait", None)
            if not callable(wait):
                raise FlashScoringError("Flash endpoint job does not expose wait")
            completed = await wait(timeout=float(self._config.request_timeout))
            wait_completed = True
            job_for_state = completed if completed is not None else job
            if getattr(job_for_state, "error", None):
                raise FlashScoringError("Flash endpoint job failed")
            status = getattr(job_for_state, "status", None)
            if status == "FAILED":
                raise FlashScoringError("Flash endpoint job failed")
            output = getattr(job_for_state, "output", None)
            if output is None:
                output = getattr(job, "output", None)
            if output is None:
                raise FlashScoringError("Flash endpoint job returned no output")
            response = decode_flash_response(_response_bytes(output))
            if response["scorer_fingerprint"] != self._config.fingerprint:
                raise FlashScoringError("Flash endpoint response scorer fingerprint mismatch")
            return _response_bytes(response)
        except BaseException:
            if job is not None and (not wait_completed or not self._job_is_terminal(job)):
                await self._cancel_job_best_effort(job)
            raise

    def run_batch_sync(self, payload: bytes) -> bytes:
        if self._bridge is None:
            self._bridge = _EventLoopBridge()
        try:
            return cast(
                bytes,
                self._bridge.run(
                    lambda: self.run_batch(payload),
                    timeout=self._config.request_timeout + 1.0,
                ),
            )
        except FlashScoringError:
            raise
        except TimeoutError as error:
            raise FlashScoringError("Flash endpoint batch timed out") from error
        except Exception as error:
            raise FlashScoringError("Flash endpoint batch failed") from error

    def close(self) -> None:
        if self._bridge is not None:
            self._bridge.close()

    def __enter__(self) -> "FlashEndpointBatchRunner":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "DEFAULT_FLASH_BATCH_SIZE",
    "DEFAULT_FLASH_CONCURRENCY",
    "DEFAULT_FLASH_IDLE_TIMEOUT",
    "DEFAULT_FLASH_REQUEST_TIMEOUT",
    "FLASH_PAYLOAD_LIMIT_BYTES",
    "WIRE_SCHEMA_VERSION",
    "FlashEndpointBatchRunner",
    "FlashScoringError",
    "FlashWireValidationError",
    "RunpodFlashConfig",
    "RunpodFlashPassageScorer",
    "decode_flash_batch",
    "decode_flash_batch_response",
    "decode_flash_request",
    "decode_flash_response",
    "encode_flash_batch",
    "encode_flash_batch_response",
    "encode_flash_request",
    "encode_flash_response",
]
