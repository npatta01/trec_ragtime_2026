"""Self-contained class-style warm worker definition for Flash."""

from __future__ import annotations

import threading
from collections.abc import Callable
from types import ModuleType
from typing import Any


# Generated classes are intentionally never evicted: their module and source
# identities remain valid for inspect, pickle, and already-created instances.
# Refusing a new policy at this modest cap is safer than invalidating a class
# that Flash or a caller may still hold.
_WORKER_CLASS_CACHE_MAX = 32
_WORKER_CLASS_CACHE_LOCK = threading.RLock()
_WORKER_CLASS_CACHE: dict[str, type[Any]] = {}
_WORKER_MODULE_CACHE: dict[str, ModuleType] = {}


def _resolve_gpu(gpu_name: str) -> object:
    """Resolve a maintained Flash GPU group or type only when decoration is requested."""

    try:
        from runpod_flash import GpuGroup, GpuType
    except ImportError as error:
        raise RuntimeError(
            "Runpod Flash is required to decorate a worker; inject a GPU resolver for offline use"
        ) from error

    for gpu_enum in (GpuGroup, GpuType):
        try:
            return getattr(gpu_enum, gpu_name)
        except AttributeError:
            continue
    raise ValueError(f"unknown Runpod Flash GPU group or type: {gpu_name!r}")


def _pinned_dependencies(config: object) -> list[str]:
    return [
        f"sentence-transformers=={config.sentence_transformers_version}",
        f"transformers=={config.transformers_version}",
        f"tokenizers=={config.tokenizers_version}",
        f"torch=={config.torch_version}",
    ]


def _uses_endpoint_style(endpoint_factory: Callable[..., object]) -> bool:
    import inspect

    try:
        parameters = inspect.signature(endpoint_factory).parameters
    except (TypeError, ValueError):
        parameters = {}
    return "name" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _decorate_worker(
    endpoint_factory: Callable[..., object],
    worker_class: type[Any],
    endpoint_kwargs: dict[str, object],
) -> type[Any]:
    """Apply an Endpoint-style factory, retaining the pre-existing test seam."""

    endpoint_style = _uses_endpoint_style(endpoint_factory)
    if not endpoint_style:
        # Compatibility for the original class-only injection seam.  The real
        # Flash path always receives the exact Endpoint constructor kwargs.
        decorated = endpoint_factory(worker_class)
    else:
        decorator = endpoint_factory(**endpoint_kwargs)
        if not callable(decorator):
            raise TypeError("Flash endpoint factory must return a class decorator")
        decorated = decorator(worker_class)
    if not isinstance(decorated, type):
        raise TypeError("Flash endpoint decorator must return a worker class")
    return decorated


def build_flash_worker(
    config: object | None = None,
    *,
    endpoint_factory: Callable[..., object] | None = None,
    gpu_resolver: Callable[[str], object] | None = None,
) -> type[Any]:
    """Build or lazily decorate a source-stable class-style Flash worker.

    The effective local model policy is rendered as primitive literals inside
    the returned class body because Flash ships only that source.  A loader is
    an explicit constructor seam on the returned class for local tests; it is
    never stored on the cached class or shared between worker instances.
    """

    import hashlib
    import json
    import linecache
    import sys
    import types

    from .flash_passage_scorer import RunpodFlashConfig
    from .passage_scoring import LocalCrossEncoderConfig

    if config is None:
        effective_config = LocalCrossEncoderConfig()
    elif isinstance(config, RunpodFlashConfig):
        effective_config = config.effective_model_config
    elif isinstance(config, LocalCrossEncoderConfig):
        effective_config = config
    else:
        raise TypeError("config must be RunpodFlashConfig, LocalCrossEncoderConfig, or None")

    model_config = {
        "batch_size": effective_config.batch_size,
        "device": effective_config.effective_device,
        "max_length": effective_config.max_length,
        "model_name": effective_config.model_name,
        "precision": effective_config.effective_precision,
        "query_format": effective_config.query_format,
        "query_policy": effective_config.query_policy,
        "revision": effective_config.revision,
        "schema_version": effective_config.schema_version,
        "sentence_transformers_version": effective_config.sentence_transformers_version,
        "tokenizer_name": effective_config.tokenizer_name,
        "tokenizer_revision": effective_config.tokenizer_revision,
        "transformers_version": effective_config.transformers_version,
        "tokenizers_version": effective_config.tokenizers_version,
        "torch_version": effective_config.torch_version,
    }
    model_literal = json.dumps(
        model_config,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    class_source = """class FlashPassageWorker:
    _MODEL_CONFIG = __MODEL_CONFIG__

    def __init__(self, model_config: object | None = None, loader: object | None = None) -> None:
        from trec_ragtime.passage_scoring import (
            LocalCrossEncoderConfig,
            LocalCrossEncoderScorer,
        )

        if model_config is None:
            effective_config = LocalCrossEncoderConfig(**self._MODEL_CONFIG)
        elif isinstance(model_config, LocalCrossEncoderConfig):
            effective_config = model_config
        elif isinstance(model_config, dict):
            effective_config = LocalCrossEncoderConfig(**model_config)
        else:
            raise TypeError("model_config must be LocalCrossEncoderConfig or a config mapping")
        if loader is None:
            self._scorer = LocalCrossEncoderScorer(effective_config)
        else:
            self._scorer = LocalCrossEncoderScorer(effective_config, loader=loader)
        self._scorer._ensure_loaded()

    @staticmethod
    def _validate_rows(rows: object) -> list[dict[str, str]]:
        from trec_ragtime.flash_passage_scorer import FlashScoringError
        from trec_ragtime.passages import sha256_bytes

        if isinstance(rows, (str, bytes, bytearray)) or not isinstance(rows, list):
            raise FlashScoringError("rows must be a list")
        normalized: list[dict[str, str]] = []
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"hash", "text"}:
                raise FlashScoringError("worker rows must contain exactly hash and text")
            hash_value = row["hash"]
            text = row["text"]
            if not isinstance(hash_value, str) or len(hash_value) != 64:
                raise FlashScoringError("worker row hash must be a 64-character hex string")
            try:
                content_hash = bytes.fromhex(hash_value)
            except ValueError as error:
                raise FlashScoringError("worker row hash is malformed") from error
            if hash_value.lower() != hash_value or len(content_hash) != 32:
                raise FlashScoringError("worker row hash is malformed")
            if not isinstance(text, str) or not text:
                raise FlashScoringError("worker row text must be non-empty")
            if content_hash != sha256_bytes(text):
                raise FlashScoringError("worker row hash does not match text")
            if hash_value in seen:
                raise FlashScoringError("worker rows must have unique hashes")
            seen.add(hash_value)
            normalized.append({"hash": hash_value, "text": text})
        return normalized

    @staticmethod
    def _make_passages(rows: list[dict[str, str]]) -> tuple[object, ...]:
        from trec_ragtime.passages import Passage, sha256_bytes

        passages: list[Passage] = []
        for index, row in enumerate(rows):
            content_hash = bytes.fromhex(row["hash"])
            passages.append(
                Passage(
                    document_id=f"flash-row-{index}",
                    source_content_hash=content_hash,
                    representation="canonical",
                    representation_hash=content_hash,
                    language="eng",
                    chunk_index=index,
                    passage_id=sha256_bytes(f"flash-row:{index}:{row['hash']}"),
                    text=row["text"],
                    start_char=0,
                    end_char=len(row["text"]),
                    boundary="paragraph_packed",
                    previous_passage_id=None,
                    next_passage_id=None,
                    content_hash=content_hash,
                    chunker_fingerprint="flash-worker-v1",
                )
            )
        return tuple(passages)

    def score(
        self,
        query: str,
        rows: list[dict[str, str]],
        schema_version: int,
        scorer_fingerprint: str,
    ) -> dict[str, object]:
        from trec_ragtime.flash_passage_scorer import (
            FlashScoringError,
            WIRE_SCHEMA_VERSION,
        )

        if schema_version != WIRE_SCHEMA_VERSION or isinstance(schema_version, bool):
            raise FlashScoringError("unsupported Flash wire schema version")
        if scorer_fingerprint != self._scorer.fingerprint:
            raise FlashScoringError("worker scorer fingerprint mismatch")
        normalized_rows = self._validate_rows(rows)
        passages = self._make_passages(normalized_rows)
        try:
            scores = self._scorer.score(query, passages)  # type: ignore[arg-type]
        except FlashScoringError:
            raise
        except Exception as error:
            raise FlashScoringError("worker scoring failed") from error
        return {
            "schema_version": WIRE_SCHEMA_VERSION,
            "scorer_fingerprint": self._scorer.fingerprint,
            "scores": [
                {"hash": content_hash.hex(), "score": score}
                for content_hash, score in scores.items()
            ],
        }
""".replace("__MODEL_CONFIG__", model_literal)

    source_digest = hashlib.sha256(class_source.encode("utf-8")).hexdigest()
    with _WORKER_CLASS_CACHE_LOCK:
        worker_class = _WORKER_CLASS_CACHE.get(source_digest)
        if worker_class is None:
            if len(_WORKER_CLASS_CACHE) >= _WORKER_CLASS_CACHE_MAX:
                from .flash_passage_scorer import FlashScoringError

                raise FlashScoringError(
                    "Flash worker class cache is full; refusing a distinct model policy"
                )
            filename = f"<trec-ragtime-flash-worker-{source_digest}.py>"
            module_name = f"_trec_ragtime_flash_worker_{source_digest}"
            linecache.cache[filename] = (
                len(class_source),
                None,
                class_source.splitlines(keepends=True),
                filename,
            )
            module = types.ModuleType(module_name)
            module.__file__ = filename
            namespace = module.__dict__
            exec(compile(class_source, filename, "exec"), namespace)
            worker_class = namespace["FlashPassageWorker"]
            # Install a generated module only once for this source digest.
            sys.modules[module_name] = module
            _WORKER_MODULE_CACHE[source_digest] = module
            _WORKER_CLASS_CACHE[source_digest] = worker_class
    if endpoint_factory is not None:
        from .flash_passage_scorer import (
            DEFAULT_FLASH_GPU,
            DEFAULT_FLASH_IDLE_TIMEOUT,
            DEFAULT_FLASH_WORKER_NAME,
        )

        if isinstance(config, RunpodFlashConfig):
            worker_name = config.worker_name
            gpu_name = config.gpu
            workers = config.workers
            idle_timeout = config.idle_timeout
        else:
            worker_name = DEFAULT_FLASH_WORKER_NAME
            gpu_name = DEFAULT_FLASH_GPU
            workers = 1
            idle_timeout = DEFAULT_FLASH_IDLE_TIMEOUT
        if not _uses_endpoint_style(endpoint_factory):
            return _decorate_worker(endpoint_factory, worker_class, {})
        resolve_gpu = gpu_resolver or _resolve_gpu
        return _decorate_worker(
            endpoint_factory,
            worker_class,
            {
                "name": worker_name,
                "gpu": resolve_gpu(gpu_name),
                "workers": workers,
                "idle_timeout": idle_timeout,
                "dependencies": _pinned_dependencies(effective_config),
            },
        )
    return worker_class


__all__ = ["build_flash_worker"]
