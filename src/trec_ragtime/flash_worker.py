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
    import importlib.metadata

    try:
        msgspec_version = importlib.metadata.version("msgspec")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "msgspec is required by the generated Flash worker"
        ) from error
    return [
        f"sentence-transformers=={config.sentence_transformers_version}",
        f"transformers=={config.transformers_version}",
        f"tokenizers=={config.tokenizers_version}",
        f"torch=={config.torch_version}",
        f"msgspec=={msgspec_version}",
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

    from .flash_passage_scorer import FlashScoringError, RunpodFlashConfig
    from .passage_scoring import LocalCrossEncoderConfig

    if config is None:
        effective_config = LocalCrossEncoderConfig()
    elif isinstance(config, RunpodFlashConfig):
        effective_config = config.effective_model_config
    elif isinstance(config, LocalCrossEncoderConfig):
        effective_config = config
    else:
        raise TypeError(
            "config must be RunpodFlashConfig, LocalCrossEncoderConfig, or None"
        )

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
    _WIRE_SCHEMA_VERSION = 1
    _FINGERPRINT_PREFIX = "local-cross-encoder-v2/"
    _ERROR = ValueError
    _CONFIG_FIELDS = (
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

    class _ModelConfig:
        __slots__ = ("_values", "fingerprint")

        def __init__(self, values: dict[str, object], fingerprint: str) -> None:
            self._values = values
            self.fingerprint = fingerprint

        def __getattr__(self, name: str) -> object:
            try:
                return self._values[name]
            except KeyError as error:
                raise AttributeError(name) from error

        def __getitem__(self, name: str) -> object:
            return self._values[name]

        @property
        def effective_device(self) -> str:
            return str(self._values["device"])

        @property
        def effective_precision(self) -> str:
            return str(self._values["precision"])

        def __eq__(self, other: object) -> bool:
            if other is self:
                return True
            for name, value in self._values.items():
                if name == "device":
                    other_value = getattr(other, "effective_device", None)
                elif name == "precision":
                    other_value = getattr(other, "effective_precision", None)
                else:
                    other_value = getattr(other, name, None)
                if other_value != value:
                    return False
            return True

    @classmethod
    def _error(cls, message: str) -> ValueError:
        return cls._ERROR(message)

    @staticmethod
    def _non_blank(value: object, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-blank string")
        if any(character in value for character in "\\r\\n"):
            raise ValueError(f"{field_name} must not contain newlines")
        return value

    @classmethod
    def _effective_device(cls, value: object) -> str:
        device = cls._non_blank(value, "device").lower()
        if device == "rocm":
            return "cuda"
        if device == "cpu" or device == "cuda" or device.startswith("cuda:"):
            if device.startswith("cuda:") and not device.removeprefix("cuda:").isdigit():
                raise cls._error("device must be cpu, cuda, cuda:N, or rocm")
            return device
        raise cls._error("device must be cpu, cuda, cuda:N, or rocm")

    @classmethod
    def _effective_precision(cls, value: object) -> str:
        precision = cls._non_blank(value, "precision").lower()
        aliases = {
            "float32": "float32",
            "fp32": "float32",
            "float16": "float16",
            "fp16": "float16",
            "bfloat16": "bfloat16",
            "bf16": "bfloat16",
        }
        try:
            return aliases[precision]
        except KeyError as error:
            raise cls._error(
                "precision must be float32, float16, or bfloat16"
            ) from error

    @classmethod
    def _validate_query_format(cls, query_format: str) -> None:
        import string

        fields: list[str] = []
        try:
            parsed = string.Formatter().parse(query_format)
            for _, field_name, format_spec, conversion in parsed:
                if field_name is None:
                    continue
                if field_name != "query" or format_spec or conversion:
                    raise cls._error(
                        "query_format may contain only the {query} field"
                    )
                fields.append(field_name)
        except (IndexError, ValueError) as error:
            raise cls._error(
                "query_format must be a valid format containing {query}"
            ) from error
        if fields != ["query"]:
            raise cls._error("query_format must contain exactly one {query} field")

    @classmethod
    def _validate_runtime_versions(cls, values: dict[str, object]) -> None:
        import importlib.metadata

        distributions = (
            ("sentence_transformers_version", "sentence-transformers"),
            ("transformers_version", "transformers"),
            ("tokenizers_version", "tokenizers"),
            ("torch_version", "torch"),
        )
        for field_name, distribution in distributions:
            declared = values[field_name]
            try:
                installed = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError as error:
                raise cls._error(
                    f"{field_name} cannot be validated because {distribution!r} is not installed"
                ) from error
            if declared != installed:
                raise cls._error(
                    f"{field_name}={declared!r} does not match installed "
                    f"{distribution}={installed!r}"
                )

    @classmethod
    def _validated_values(cls, values: dict[str, object]) -> dict[str, object]:
        unknown = set(values) - set(cls._CONFIG_FIELDS)
        if unknown:
            raise cls._error(f"unknown model config fields: {sorted(unknown)!r}")
        for field_name in (
            "model_name",
            "revision",
            "tokenizer_name",
            "tokenizer_revision",
            "sentence_transformers_version",
            "transformers_version",
            "tokenizers_version",
            "torch_version",
            "query_policy",
        ):
            cls._non_blank(values[field_name], field_name)
        schema_version = values["schema_version"]
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise cls._error("schema_version must be an integer")
        if schema_version <= 0:
            raise cls._error("schema_version must be positive")
        for field_name in ("batch_size", "max_length"):
            value = values[field_name]
            if isinstance(value, bool) or not isinstance(value, int):
                raise cls._error(f"{field_name} must be an integer")
            if value <= 0:
                raise cls._error(f"{field_name} must be positive")
        values["device"] = cls._effective_device(values["device"])
        values["precision"] = cls._effective_precision(values["precision"])
        query_format = cls._non_blank(values["query_format"], "query_format")
        cls._validate_query_format(query_format)
        cls._validate_runtime_versions(values)
        return values

    @classmethod
    def _fingerprint_for_values(cls, values: dict[str, object]) -> str:
        import hashlib

        import msgspec

        material = msgspec.msgpack.encode(
            (
                "local-cross-encoder-fingerprint-v1",
                values["schema_version"],
                values["model_name"],
                values["revision"],
                values["tokenizer_name"],
                values["tokenizer_revision"],
                values["sentence_transformers_version"],
                values["transformers_version"],
                values["tokenizers_version"],
                values["torch_version"],
                values["device"],
                values["precision"],
                values["batch_size"],
                values["max_length"],
                values["query_format"],
                values["query_policy"],
            )
        )
        return cls._FINGERPRINT_PREFIX + hashlib.sha256(material).hexdigest()

    @classmethod
    def _coerce_values(cls, model_config: object | None) -> dict[str, object]:
        from collections.abc import Mapping

        values = dict(cls._MODEL_CONFIG)
        if model_config is None:
            return values
        if isinstance(model_config, cls._ModelConfig):
            values.update(model_config._values)
            return values
        if isinstance(model_config, Mapping):
            values.update(model_config)
            return values
        try:
            values.update(
                {field_name: getattr(model_config, field_name) for field_name in cls._CONFIG_FIELDS}
            )
        except AttributeError as error:
            raise cls._error(
                "model_config must be a model config object or mapping"
            ) from error
        return values

    @classmethod
    def _make_config(cls, model_config: object | None) -> _ModelConfig:
        values = cls._validated_values(cls._coerce_values(model_config))
        return cls._ModelConfig(values, cls._fingerprint_for_values(values))

    def __init__(self, model_config: object | None = None, loader: object | None = None) -> None:
        import threading

        if loader is not None and not callable(loader):
            raise self._error("loader must be callable")
        self._config = self._make_config(model_config)
        self._loader = loader or self._default_loader
        self._load_lock = threading.Lock()
        self._model: object | None = None
        self._tokenizer: object | None = None
        self._ensure_loaded()

    @staticmethod
    def _unpack_loader_result(result: object) -> tuple[object, object]:
        if isinstance(result, tuple) and len(result) == 2:
            model, tokenizer = result
        elif hasattr(result, "model") and hasattr(result, "tokenizer"):
            model = getattr(result, "model")
            tokenizer = getattr(result, "tokenizer")
        else:
            model = result
            tokenizer = getattr(result, "tokenizer", None)
        if (
            model is None
            or tokenizer is None
            or not callable(getattr(model, "predict", None))
        ):
            raise ValueError("loader must return a CrossEncoder and tokenizer")
        if not callable(tokenizer):
            raise ValueError("loader must return a callable tokenizer")
        return model, tokenizer

    @staticmethod
    def _default_loader(config: _ModelConfig) -> tuple[object, object]:
        import torch
        from sentence_transformers import CrossEncoder, models

        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.effective_precision]
        transformer = models.Transformer(
            config.model_name,
            transformer_task="sequence-classification",
            tokenizer_name_or_path=config.tokenizer_name,
            processor_kwargs={
                "revision": config.tokenizer_revision,
                "model_max_length": config.max_length,
            },
            model_kwargs={"revision": config.revision, "torch_dtype": dtype},
            config_kwargs={"revision": config.revision},
        )
        tokenizer = getattr(transformer, "tokenizer", None)
        if not callable(tokenizer):
            raise ValueError("Transformer did not expose a callable pinned tokenizer")
        model = CrossEncoder(modules=[transformer], device=config.effective_device)
        if not callable(getattr(model, "predict", None)):
            raise ValueError("CrossEncoder did not expose a callable predict method")
        return model, tokenizer

    def _ensure_loaded(self) -> tuple[object, object]:
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        with self._load_lock:
            if self._model is None or self._tokenizer is None:
                try:
                    result = self._loader(self._config)
                    model, tokenizer = self._unpack_loader_result(result)
                except self._ERROR:
                    raise
                except Exception as error:
                    raise self._error("unable to load the local CrossEncoder") from error
                self._model = model
                self._tokenizer = tokenizer
        return self._model, self._tokenizer

    @staticmethod
    def _as_sequence(value: object) -> object:
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            try:
                return tolist()
            except (TypeError, ValueError):
                pass
        return value

    @classmethod
    def _token_length(cls, encoded: object) -> int:
        if not isinstance(encoded, dict):
            raise cls._error("tokenizer returned a non-mapping result")
        input_ids = cls._as_sequence(encoded.get("input_ids"))
        if input_ids is None:
            raise cls._error("tokenizer did not return input_ids")
        if isinstance(input_ids, (str, bytes, bytearray)):
            raise cls._error("tokenizer returned malformed input_ids")
        try:
            items = list(input_ids)
        except TypeError as error:
            raise cls._error("tokenizer returned malformed input_ids") from error
        if items and isinstance(cls._as_sequence(items[0]), (list, tuple)):
            lengths = [len(cls._as_sequence(item)) for item in items]
            length = max(lengths, default=0)
        else:
            length = len(items)
        overflow = cls._as_sequence(encoded.get("overflowing_tokens"))
        if overflow not in (None, [], (), False):
            raise cls._error("tokenizer reported overflow")
        truncated = encoded.get("num_truncated_tokens")
        if truncated not in (None, 0, [0], (0,)):
            raise cls._error("tokenizer reported truncation")
        return length

    @classmethod
    def _format_query(cls, config: _ModelConfig, query: object) -> str:
        if not isinstance(query, str) or not query.strip():
            raise cls._error("query must be a non-blank string")
        try:
            return query if config.query_format == "{query}" else config.query_format.format(query=query)
        except (IndexError, KeyError, ValueError) as error:
            raise cls._error("query could not be formatted") from error

    @classmethod
    def _validate_rows(cls, rows: object) -> list[tuple[bytes, str]]:
        import hashlib

        if isinstance(rows, (str, bytes, bytearray)) or not isinstance(rows, list):
            raise cls._error("rows must be a list")
        normalized: list[tuple[bytes, str]] = []
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"hash", "text"}:
                raise cls._error("worker rows must contain exactly hash and text")
            hash_value = row["hash"]
            text = row["text"]
            if not isinstance(hash_value, str) or len(hash_value) != 64:
                raise cls._error("worker row hash must be a 64-character hex string")
            try:
                content_hash = bytes.fromhex(hash_value)
            except ValueError as error:
                raise cls._error("worker row hash is malformed") from error
            if hash_value.lower() != hash_value or len(content_hash) != 32:
                raise cls._error("worker row hash is malformed")
            if not isinstance(text, str) or not text:
                raise cls._error("worker row text must be non-empty")
            if content_hash != hashlib.sha256(text.encode("utf-8")).digest():
                raise cls._error("worker row hash does not match text")
            if hash_value in seen:
                raise cls._error("worker rows must have unique hashes")
            seen.add(hash_value)
            normalized.append((content_hash, text))
        return normalized

    @classmethod
    def _prediction_values(cls, result: object, expected_count: int) -> list[object]:
        from collections.abc import Sequence

        converted = cls._as_sequence(result)
        if isinstance(converted, (str, bytes, bytearray)):
            values: list[object] = [converted]
        elif isinstance(converted, Sequence):
            values = list(converted)
        else:
            values = [converted]
        if len(values) != expected_count:
            raise cls._error("CrossEncoder returned an unexpected number of scores")
        return values

    @classmethod
    def _validate_scores(
        cls,
        requested: list[tuple[bytes, str]],
        scores: object,
    ) -> dict[bytes, float]:
        import math
        import numbers
        from collections.abc import Mapping

        if not isinstance(scores, Mapping):
            raise cls._error("scores must be a mapping")
        normalized: dict[bytes, float] = {}
        for key, value in scores.items():
            if not isinstance(key, bytes) or len(key) != 32:
                raise cls._error("score key must be exactly 32 bytes")
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise cls._error(
                    f"score for key {key!r} must be a non-Boolean numeric value"
                )
            try:
                normalized_value = float(value)
            except (OverflowError, TypeError, ValueError) as error:
                raise cls._error("score could not be normalized to float") from error
            if not math.isfinite(normalized_value):
                raise cls._error("scores must be finite")
            normalized[key] = normalized_value
        expected_keys = {content_hash for content_hash, _ in requested}
        if set(normalized) != expected_keys:
            missing = expected_keys - set(normalized)
            extra = set(normalized) - expected_keys
            raise cls._error(
                f"score coverage mismatch (missing={len(missing)}, extra={len(extra)})"
            )
        return {key: normalized[key] for key in sorted(normalized)}

    @staticmethod
    def _identity_activation() -> object:
        import torch

        return torch.nn.Identity()

    def _score_rows(
        self, query: str, rows: list[tuple[bytes, str]]
    ) -> dict[bytes, float]:
        formatted_query = self._format_query(self._config, query)
        if not rows:
            return {}
        pairs = tuple((formatted_query, text) for _, text in rows)
        model, tokenizer = self._ensure_loaded()
        for pair_query, passage in pairs:
            try:
                encoded = tokenizer(
                    pair_query,
                    passage,
                    truncation=False,
                    max_length=self._config.max_length,
                    return_length=True,
                )
            except self._ERROR:
                raise
            except Exception as error:
                raise self._error(
                    "tokenizer failed while validating a passage pair"
                ) from error
            if self._token_length(encoded) > self._config.max_length:
                raise self._error("tokenizer pair overflowed max_length")

        raw_scores: list[object] = []
        for start in range(0, len(pairs), self._config.batch_size):
            batch = list(pairs[start : start + self._config.batch_size])
            try:
                prediction = model.predict(
                    batch,
                    batch_size=self._config.batch_size,
                    show_progress_bar=False,
                    activation_fn=self._identity_activation(),
                    apply_softmax=False,
                    convert_to_numpy=True,
                    convert_to_tensor=False,
                    device=self._config.effective_device,
                    processing_kwargs={
                        "text": {
                            "truncation": False,
                            "max_length": self._config.max_length,
                        }
                    },
                )
            except self._ERROR:
                raise
            except Exception as error:
                raise self._error("CrossEncoder prediction failed") from error
            raw_scores.extend(self._prediction_values(prediction, len(batch)))
        raw_mapping = {
            content_hash: raw_score
            for (content_hash, _), raw_score in zip(rows, raw_scores, strict=True)
        }
        return self._validate_scores(rows, raw_mapping)

    def score(
        self,
        query: str,
        rows: list[dict[str, str]],
        schema_version: int,
        scorer_fingerprint: str,
    ) -> dict[str, object]:
        if schema_version != self._WIRE_SCHEMA_VERSION or isinstance(schema_version, bool):
            raise self._error("unsupported Flash wire schema version")
        if scorer_fingerprint != self._config.fingerprint:
            raise self._error("worker scorer fingerprint mismatch")
        normalized_rows = self._validate_rows(rows)
        try:
            scores = self._score_rows(query, normalized_rows)
        except self._ERROR:
            raise
        except Exception as error:
            raise self._error("worker scoring failed") from error
        return {
            "schema_version": self._WIRE_SCHEMA_VERSION,
            "scorer_fingerprint": self._config.fingerprint,
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
    worker_class._ERROR = FlashScoringError
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
