"""Thin optional Runpod Flash adapter for the passage-scoring protocol."""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import numbers
import re
import string
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
DEFAULT_FLASH_IDLE_TIMEOUT = 300
DEFAULT_FLASH_WORKER_NAME = "trec-ragtime-passage-scorer"
DEFAULT_FLASH_GPU = "AMPERE_24"
_FLASH_CACHE_STATUSES = frozenset({"hit", "miss_written", "miss_raced"})
_FLASH_CACHE_KEY_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_BRIDGE_CLEANUP_GRACE_SECONDS = 0.5
_BRIDGE_STARTUP_CLEANUP_GRACE_SECONDS = 0.5
_MODEL_CONFIG_FIELDS = (
    "schema_version",
    "model_name",
    "revision",
    "tokenizer_name",
    "tokenizer_revision",
    "sentence_transformers_version",
    "transformers_version",
    "tokenizers_version",
    "torch_version",
    "device",
    "precision",
    "batch_size",
    "max_length",
    "query_format",
    "query_policy",
)


class FlashScoringError(PassageScoringError):
    """Raised for invalid Flash wire data, transport failures, or worker errors."""


def _validate_schema_version(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value != WIRE_SCHEMA_VERSION
    ):
        raise FlashScoringError("unsupported Flash wire schema version")
    return value


def _validate_fingerprint(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(character in value for character in "\r\n")
    ):
        raise FlashScoringError("scorer_fingerprint must be a non-blank string")
    return value


def _validate_model_config(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != set(_MODEL_CONFIG_FIELDS):
        raise FlashScoringError(
            "model_config must contain exactly every result-affecting scorer setting"
        )
    normalized = dict(value)
    string_fields = {
        "model_name",
        "revision",
        "tokenizer_name",
        "tokenizer_revision",
        "sentence_transformers_version",
        "transformers_version",
        "tokenizers_version",
        "torch_version",
        "query_format",
        "query_policy",
    }
    for field_name in string_fields:
        field_value = normalized[field_name]
        if (
            not isinstance(field_value, str)
            or not field_value.strip()
            or any(character in field_value for character in "\r\n")
        ):
            raise FlashScoringError(f"model_config field {field_name} is invalid")
    for field_name in ("schema_version", "batch_size", "max_length"):
        field_value = normalized[field_name]
        if (
            isinstance(field_value, bool)
            or not isinstance(field_value, int)
            or field_value <= 0
        ):
            raise FlashScoringError(f"model_config field {field_name} is invalid")
    device = normalized["device"]
    if not isinstance(device, str) or (
        device not in {"cpu", "cuda"}
        and not (device.startswith("cuda:") and device.removeprefix("cuda:").isdigit())
    ):
        raise FlashScoringError("model_config device is invalid")
    precision = normalized["precision"]
    if not isinstance(precision, str) or precision not in {
        "float32",
        "float16",
        "bfloat16",
    }:
        raise FlashScoringError("model_config precision is invalid")
    fields: list[str] = []
    try:
        for _, field_name, format_spec, conversion in string.Formatter().parse(
            normalized["query_format"]
        ):
            if field_name is None:
                continue
            if field_name != "query" or format_spec or conversion:
                raise FlashScoringError("model_config query_format is invalid")
            fields.append(field_name)
    except (IndexError, ValueError) as error:
        raise FlashScoringError("model_config query_format is invalid") from error
    if fields != ["query"]:
        raise FlashScoringError("model_config query_format is invalid")
    return normalized


def _model_config_mapping(config: LocalCrossEncoderConfig) -> dict[str, object]:
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


def _wire_model_config(value: object) -> dict[str, object]:
    if isinstance(value, LocalCrossEncoderConfig):
        return _model_config_mapping(value)
    return _validate_model_config(value)


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
        raise FlashScoringError(
            "wire payload is not valid compact UTF-8 JSON"
        ) from error


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def _json_object(
    payload: bytes | bytearray | str | Mapping[str, object],
) -> dict[str, object]:
    if isinstance(payload, Mapping):
        value: object = dict(payload)
    else:
        if isinstance(payload, str):
            encoded = payload.encode("utf-8")
        elif isinstance(payload, (bytes, bytearray)):
            encoded = bytes(payload)
        else:
            raise FlashScoringError("wire payload must be JSON bytes or an object")
        try:
            value = json.loads(
                encoded.decode("utf-8"), parse_constant=_reject_json_constant
            )
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as error:
            raise FlashScoringError("wire payload is not valid JSON") from error
    if not isinstance(value, dict):
        raise FlashScoringError("wire payload must be a JSON object")
    return cast(dict[str, object], value)


def _validate_hash_hex(value: object, field_name: str = "hash") -> str:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise FlashScoringError(f"{field_name} must be lowercase hexadecimal SHA-256")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as error:
        raise FlashScoringError(
            f"{field_name} must be lowercase hexadecimal SHA-256"
        ) from error
    if len(decoded) != 32:
        raise FlashScoringError(f"{field_name} must be exactly 32 bytes")
    return value


def _hash_text_value(value: object, field_name: str = "hash") -> str:
    if isinstance(value, bytes):
        if len(value) != 32:
            raise FlashScoringError(f"{field_name} must be exactly 32 bytes")
        return value.hex()
    return _validate_hash_hex(value, field_name)


def _score_value(value: object, field_name: str = "score") -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise FlashScoringError(f"{field_name} must be a finite non-Boolean number")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise FlashScoringError(f"{field_name} must be a finite number") from error
    if not math.isfinite(normalized):
        raise FlashScoringError(f"{field_name} must be finite")
    return normalized


def _request_rows(rows: object) -> list[dict[str, str]]:
    if isinstance(rows, (str, bytes, bytearray)) or not isinstance(rows, Iterable):
        raise FlashScoringError("rows must be an iterable")
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if isinstance(row, Mapping):
            if set(row) != {"hash", "text"}:
                raise FlashScoringError(
                    "request rows must contain exactly hash and text"
                )
            raw_hash, text = row["hash"], row["text"]
        elif isinstance(row, (tuple, list)) and len(row) == 2:
            raw_hash, text = row
        else:
            raise FlashScoringError("request rows must be mappings or hash/text pairs")
        hash_hex = _hash_text_value(raw_hash)
        if not isinstance(text, str) or not text:
            raise FlashScoringError("request row text must be a non-empty string")
        try:
            expected_hash = sha256_bytes(text)
        except UnicodeError as error:
            raise FlashScoringError("request row text must be valid UTF-8") from error
        if bytes.fromhex(hash_hex) != expected_hash:
            raise FlashScoringError("request hash does not match request text")
        if hash_hex in seen:
            raise FlashScoringError("request rows must have unique hashes")
        seen.add(hash_hex)
        normalized.append({"hash": hash_hex, "text": text})
    return normalized


def _response_rows(rows: object) -> list[dict[str, object]]:
    if isinstance(rows, (str, bytes, bytearray)) or not isinstance(rows, Iterable):
        raise FlashScoringError("scores must be an iterable")
    normalized: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"hash", "score"}:
            raise FlashScoringError("response rows must contain exactly hash and score")
        hash_hex = _hash_text_value(row["hash"])
        if hash_hex in seen:
            raise FlashScoringError("response rows must have unique hashes")
        seen.add(hash_hex)
        normalized.append({"hash": hash_hex, "score": _score_value(row["score"])})
    return normalized


def _validate_cache_object_key(value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise FlashScoringError("cache object_key is invalid")
    if (
        value.startswith("/")
        or "\\" in value
        or _FLASH_CACHE_KEY_CONTROL_RE.search(value)
    ):
        raise FlashScoringError("cache object_key is invalid")
    parts = value.split("/")
    if any(not part or part in {".", ".."} for part in parts):
        raise FlashScoringError("cache object_key is invalid")
    return value


def _validate_cache_telemetry(
    value: object,
    *,
    expected_row_count: int | None = None,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "status",
        "object_key",
        "row_count",
        "batch_count",
    }:
        raise FlashScoringError("response cache telemetry is invalid")
    status = value["status"]
    if not isinstance(status, str) or status not in _FLASH_CACHE_STATUSES:
        raise FlashScoringError("response cache status is invalid")
    object_key = _validate_cache_object_key(value["object_key"])
    row_count = value["row_count"]
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count <= 0:
        raise FlashScoringError("response cache row_count is invalid")
    batch_count = value["batch_count"]
    if (
        isinstance(batch_count, bool)
        or not isinstance(batch_count, int)
        or batch_count < 0
    ):
        raise FlashScoringError("response cache batch_count is invalid")
    if expected_row_count is not None and row_count != expected_row_count:
        raise FlashScoringError("response cache row_count does not match scores")
    if status == "hit" and batch_count != 0:
        raise FlashScoringError("cache hits must report batch_count zero")
    return {
        "status": status,
        "object_key": object_key,
        "row_count": row_count,
        "batch_count": batch_count,
    }


def encode_flash_request(
    query: str,
    rows: Iterable[Mapping[str, object] | Sequence[object]],
    *,
    model_config: Mapping[str, object] | LocalCrossEncoderConfig,
    scorer_fingerprint: str,
    schema_version: int = WIRE_SCHEMA_VERSION,
) -> bytes:
    """Encode a strict request carrying the complete scorer identity."""

    _validate_schema_version(schema_version)
    _validate_fingerprint(scorer_fingerprint)
    if not isinstance(query, str) or not query.strip():
        raise FlashScoringError("query must be a non-blank string")
    return _json_bytes(
        {
            "model_config": _wire_model_config(model_config),
            "query": query,
            "rows": _request_rows(rows),
            "schema_version": schema_version,
            "scorer_fingerprint": scorer_fingerprint,
        }
    )


def decode_flash_request(
    payload: bytes | bytearray | str | Mapping[str, object],
) -> dict[str, object]:
    """Decode and validate a strict scorer-identity-bearing request."""

    value = _json_object(payload)
    expected_fields = {
        "model_config",
        "query",
        "rows",
        "schema_version",
        "scorer_fingerprint",
    }
    if set(value) != expected_fields:
        raise FlashScoringError("request fields are invalid")
    query = value["query"]
    if not isinstance(query, str) or not query.strip():
        raise FlashScoringError("query must be a non-blank string")
    return {
        "schema_version": _validate_schema_version(value["schema_version"]),
        "scorer_fingerprint": _validate_fingerprint(value["scorer_fingerprint"]),
        "model_config": _validate_model_config(value["model_config"]),
        "query": query,
        "rows": _request_rows(value["rows"]),
    }


def encode_flash_response(
    rows: Iterable[Mapping[str, object] | Sequence[object]],
    *,
    schema_version: int = WIRE_SCHEMA_VERSION,
    scorer_fingerprint: str,
    cache: Mapping[str, object] | None = None,
) -> bytes:
    """Encode a strict score response."""

    _validate_schema_version(schema_version)
    _validate_fingerprint(scorer_fingerprint)
    normalized_rows = _response_rows(rows)
    if cache is None:
        # Keep the wire helper source-compatible with pre-telemetry test
        # runners. Real workers always provide their private cache record.
        cache = {
            "status": "miss_written",
            "object_key": "passage-score-requests/v1/compat/00/"
            + "0" * 64
            + ".mpk",
            "row_count": len(normalized_rows),
            "batch_count": 0,
        }
    normalized_cache = _validate_cache_telemetry(
        cache,
        expected_row_count=len(normalized_rows),
    )
    return _json_bytes(
        {
            "cache": normalized_cache,
            "scores": normalized_rows,
            "schema_version": schema_version,
            "scorer_fingerprint": scorer_fingerprint,
        }
    )


def decode_flash_response(
    payload: bytes | bytearray | str | Mapping[str, object],
) -> dict[str, object]:
    """Decode and validate a strict score response."""

    value = _json_object(payload)
    if set(value) != {
        "cache",
        "scores",
        "schema_version",
        "scorer_fingerprint",
    }:
        raise FlashScoringError("response fields are invalid")
    scores = _response_rows(value["scores"])
    return {
        "schema_version": _validate_schema_version(value["schema_version"]),
        "scorer_fingerprint": _validate_fingerprint(value["scorer_fingerprint"]),
        "scores": scores,
        "cache": _validate_cache_telemetry(
            value["cache"], expected_row_count=len(scores)
        ),
    }


def _effective_model_config(config: "RunpodFlashConfig") -> LocalCrossEncoderConfig:
    updates: dict[str, object] = {}
    for field_name in (
        "model_name",
        "revision",
        "tokenizer_name",
        "tokenizer_revision",
        "sentence_transformers_version",
        "transformers_version",
        "tokenizers_version",
        "torch_version",
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
    return replace(config.model_config, **updates) if updates else config.model_config


@dataclass(frozen=True, slots=True)
class RunpodFlashConfig:
    """Immutable Flash transport settings and effective local model policy."""

    endpoint_id: str
    model_config: LocalCrossEncoderConfig = field(
        default_factory=LocalCrossEncoderConfig
    )
    batch_size: int | None = None
    max_payload_bytes: int = FLASH_PAYLOAD_LIMIT_BYTES
    concurrency: int = DEFAULT_FLASH_CONCURRENCY
    request_timeout: float = DEFAULT_FLASH_REQUEST_TIMEOUT
    workers: int = 1
    idle_timeout: int = DEFAULT_FLASH_IDLE_TIMEOUT
    worker_name: str = DEFAULT_FLASH_WORKER_NAME
    gpu: str = DEFAULT_FLASH_GPU
    model_name: str | None = None
    revision: str | None = None
    tokenizer_name: str | None = None
    tokenizer_revision: str | None = None
    sentence_transformers_version: str | None = None
    transformers_version: str | None = None
    tokenizers_version: str | None = None
    torch_version: str | None = None
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
        value = self.request_timeout
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError("request_timeout must be a positive number")
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError("request_timeout must be a positive finite number")
        if (
            isinstance(self.idle_timeout, bool)
            or not isinstance(self.idle_timeout, int)
            or not 1 <= self.idle_timeout <= 3600
        ):
            raise ValueError("idle_timeout must be an integer between 1 and 3600")
        for field_name in (
            "worker_name",
            "gpu",
            "model_name",
            "revision",
            "tokenizer_name",
            "tokenizer_revision",
            "sentence_transformers_version",
            "transformers_version",
            "tokenizers_version",
            "torch_version",
            "device",
            "precision",
            "query_format",
            "query_policy",
        ):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, str)
                or not value.strip()
                or any(character in value for character in "\r\n")
            ):
                raise ValueError(
                    f"{field_name} must be a non-blank string when provided"
                )
        if self.max_length is not None and (
            isinstance(self.max_length, bool)
            or not isinstance(self.max_length, int)
            or self.max_length <= 0
        ):
            raise ValueError("max_length must be a positive integer when provided")
        object.__setattr__(
            self, "_fingerprint", self.effective_model_config.fingerprint
        )

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


async def _run_with_completion(
    awaitable: Awaitable[Any], completed: threading.Event
) -> Any:
    try:
        return await awaitable
    finally:
        completed.set()


class _EventLoopBridge:
    """Bridge async injected transports into synchronous PassageScorer calls."""

    def __init__(self, *, startup_timeout: float = 5.0) -> None:
        self._ready = threading.Event()
        self._stop_requested = threading.Event()
        self._loop_start_lock = threading.Lock()
        self._closed = False
        self._startup_error: BaseException | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="trec-ragtime-flash-loop",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(startup_timeout):
            self._closed = True
            self._request_stop()
            self._thread.join(_BRIDGE_STARTUP_CLEANUP_GRACE_SECONDS)
            raise FlashScoringError("Flash event-loop thread did not start")
        if self._startup_error is not None or self._loop is None:
            error = self._startup_error
            self._closed = True
            self._request_stop()
            self._thread.join(_BRIDGE_STARTUP_CLEANUP_GRACE_SECONDS)
            raise FlashScoringError(
                "Flash event-loop thread failed to start"
            ) from error

    def _run(self) -> None:
        try:
            loop = asyncio.new_event_loop()
        except BaseException as error:
            self._startup_error = error
            self._ready.set()
            return
        asyncio.set_event_loop(loop)
        with self._loop_start_lock:
            self._loop = loop
            stop_requested = self._stop_requested.is_set()
        self._ready.set()
        if stop_requested:
            self._quiesce_loop(loop)
            return
        try:
            loop.run_forever()
        finally:
            self._quiesce_loop(loop)

    def _request_stop(self) -> None:
        self._stop_requested.set()
        with self._loop_start_lock:
            loop = self._loop
            if loop is not None and not loop.is_closed():
                try:
                    loop.call_soon_threadsafe(loop.stop)
                except RuntimeError:
                    pass

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
        for task in list(asyncio.all_tasks(loop)):
            task.cancel()
            if hasattr(task, "_log_destroy_pending"):
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
            raise FlashScoringError("nested synchronous Flash scoring is not allowed")
        awaitable = (
            awaitable_or_factory()
            if callable(awaitable_or_factory)
            else awaitable_or_factory
        )
        completed = threading.Event()
        future = asyncio.run_coroutine_threadsafe(
            _run_with_completion(awaitable, completed), self._loop
        )
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            future.cancel()
            completed.wait(_BRIDGE_CLEANUP_GRACE_SECONDS)
            try:
                self.close()
            except BaseException:
                pass
            raise

    def close(self) -> None:
        if self._closed:
            if (
                threading.current_thread() is not self._thread
                and self._thread.is_alive()
            ):
                self._thread.join()
            return
        self._closed = True
        if threading.current_thread() is self._thread:
            raise FlashScoringError(
                "cannot close the Flash event loop from its own thread"
            )
        self._request_stop()
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


def _runner_awaitable(runner: object, payload: bytes) -> Awaitable[object]:
    method = getattr(runner, "run_batch", None)
    if callable(method):
        result = method(payload)
    elif callable(runner):
        result = runner(payload)  # type: ignore[operator]
    else:
        raise FlashScoringError("batch runner must expose async run_batch")
    if not hasattr(result, "__await__"):
        raise FlashScoringError("batch runner must return an awaitable")
    return cast(Awaitable[object], result)


def _runpod_request_body(payload: bytes) -> bytes:
    """Encode the body produced by the Runpod endpoint ``/run`` call."""

    request = decode_flash_request(payload)
    return json.dumps(
        {"input": {"request": request}},
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


class RunpodFlashPassageScorer:
    """Synchronous exact scorer backed by one asynchronous Flash request."""

    __slots__ = ("_config", "_runner", "_owned_runner", "_bridge")

    def __init__(self, config: RunpodFlashConfig, batch_runner: object) -> None:
        if not isinstance(config, RunpodFlashConfig):
            raise TypeError("config must be a RunpodFlashConfig")
        self._config = config
        self._owned_runner: FlashEndpointBatchRunner | None = None
        if callable(getattr(batch_runner, "run", None)) and not callable(
            getattr(batch_runner, "run_batch", None)
        ):
            self._owned_runner = FlashEndpointBatchRunner(config, endpoint=batch_runner)
            self._runner: object = self._owned_runner
        else:
            self._runner = batch_runner
        self._bridge = _EventLoopBridge()

    @property
    def config(self) -> RunpodFlashConfig:
        return self._config

    @property
    def fingerprint(self) -> str:
        return self._config.fingerprint

    @staticmethod
    def _build_request(
        query: str,
        unique: Mapping[bytes, str],
        config: RunpodFlashConfig,
    ) -> bytes:
        model_config = _model_config_mapping(config.effective_model_config)
        limit = min(config.max_payload_bytes, FLASH_PAYLOAD_LIMIT_BYTES)
        payload = encode_flash_request(
            query,
            unique.items(),
            model_config=model_config,
            scorer_fingerprint=config.fingerprint,
        )
        request_body = _runpod_request_body(payload)
        if len(request_body) >= limit:
            raise FlashScoringError(
                "Flash payload must be strictly below its byte limit"
            )
        return payload

    async def _run_one(
        self, payload: bytes, expected_request: frozenset[bytes]
    ) -> dict[bytes, float]:
        try:
            raw = await asyncio.wait_for(
                _runner_awaitable(self._runner, payload),
                timeout=float(self._config.request_timeout),
            )
            response = decode_flash_response(raw)  # type: ignore[arg-type]
        except asyncio.TimeoutError as error:
            raise FlashScoringError("Flash batch timed out") from error
        except asyncio.CancelledError as error:
            raise FlashScoringError("Flash batch cancelled") from error
        except FlashScoringError:
            raise
        except Exception as error:
            raise FlashScoringError("Flash batch failed") from error
        if response["scorer_fingerprint"] != self._config.fingerprint:
            raise FlashScoringError("Flash response scorer fingerprint mismatch")
        rows = cast(list[dict[str, object]], response["scores"])
        scores = {
            bytes.fromhex(cast(str, row["hash"])): cast(float, row["score"])
            for row in rows
        }
        if set(scores) != set(expected_request):
            raise FlashScoringError("Flash response coverage does not match request")
        return scores

    async def _score_async(
        self,
        passages: Sequence[Passage],
        payload: bytes,
        expected_request: frozenset[bytes],
    ) -> dict[bytes, float]:
        merged = await self._run_one(payload, expected_request)
        try:
            return validate_passage_scores(passages, merged)
        except PassageScoringError as error:
            raise FlashScoringError(
                "Flash scores failed the passage scorer contract"
            ) from error

    def score(self, query: str, passages: Sequence[Passage]) -> Mapping[bytes, float]:
        if self._bridge.closed:
            raise FlashScoringError("Flash scorer is closed")
        if not isinstance(query, str) or not query.strip():
            raise FlashScoringError("query must be a non-blank string")
        try:
            unique = _unique_passages(passages)
        except PassageScoringError as error:
            raise FlashScoringError(
                "passages failed the passage scorer contract"
            ) from error
        if not unique:
            return {}
        payload = self._build_request(query, unique, self._config)
        expected_request = frozenset(unique)
        try:
            return self._bridge.run(
                lambda: self._score_async(passages, payload, expected_request),
                timeout=float(self._config.request_timeout) + 1.0,
            )
        except FlashScoringError:
            raise
        except TimeoutError as error:
            raise FlashScoringError("Flash scoring timed out") from error
        except Exception as error:
            raise FlashScoringError("Flash scoring failed") from error

    def close(self) -> None:
        self._bridge.close()
        if self._owned_runner is not None:
            self._owned_runner.close()

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
    """Lazy Runpod Flash endpoint client using async ``run`` then ``wait``."""

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
        return getattr(job, "status", None) in {"COMPLETED", "FAILED", "CANCELLED"}

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
            job_result = run({"request": request})
            if not hasattr(job_result, "__await__"):
                raise FlashScoringError("Flash endpoint run must be awaitable")
            job = await cast(Awaitable[object], job_result)
            wait = getattr(job, "wait", None)
            if not callable(wait):
                raise FlashScoringError("Flash endpoint job does not expose wait")
            wait_result = wait(timeout=float(self._config.request_timeout))
            if not hasattr(wait_result, "__await__"):
                raise FlashScoringError("Flash endpoint job wait must be awaitable")
            completed = await cast(Awaitable[object], wait_result)
            wait_completed = True
            job_for_state = completed if completed is not None else job
            if (
                getattr(job_for_state, "error", None)
                or getattr(job_for_state, "status", None) == "FAILED"
            ):
                raise FlashScoringError("Flash endpoint job failed")
            output = getattr(job_for_state, "output", None)
            if output is None and job is not None:
                output = getattr(job, "output", None)
            if output is None:
                raise FlashScoringError("Flash endpoint job returned no output")
            response = decode_flash_response(output)
            if response["scorer_fingerprint"] != self._config.fingerprint:
                raise FlashScoringError(
                    "Flash endpoint response scorer fingerprint mismatch"
                )
            return json.dumps(response, separators=(",", ":"), sort_keys=True).encode(
                "utf-8"
            )
        except BaseException:
            if job is not None and (
                not wait_completed or not self._job_is_terminal(job)
            ):
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
    "DEFAULT_FLASH_GPU",
    "DEFAULT_FLASH_IDLE_TIMEOUT",
    "DEFAULT_FLASH_REQUEST_TIMEOUT",
    "DEFAULT_FLASH_WORKER_NAME",
    "FLASH_PAYLOAD_LIMIT_BYTES",
    "WIRE_SCHEMA_VERSION",
    "FlashEndpointBatchRunner",
    "FlashScoringError",
    "RunpodFlashConfig",
    "RunpodFlashPassageScorer",
    "decode_flash_request",
    "decode_flash_response",
    "encode_flash_request",
    "encode_flash_response",
]
