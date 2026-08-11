"""A static, class-style warm worker and lazy Flash decorator seam."""

from __future__ import annotations

from collections.abc import Callable


class FlashPassageWorker:
    """Self-contained Flash worker; only this class body is sent remotely."""

    _WIRE_SCHEMA_VERSION = 1
    _FINGERPRINT_PREFIX = "local-cross-encoder-v3/"
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
    _REQUEST_FIELDS = {
        "schema_version",
        "scorer_fingerprint",
        "model_config",
        "query",
        "rows",
    }

    @classmethod
    def _error(cls, message: str) -> Exception:
        return cls._ERROR(message)

    @staticmethod
    def _as_sequence(value: object) -> object:
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            try:
                return tolist()
            except (TypeError, ValueError):
                pass
        return value

    @staticmethod
    def _non_blank(value: object, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-blank string")
        if any(character in value for character in "\r\n"):
            raise ValueError(f"{field_name} must not contain newlines")
        return value

    @classmethod
    def _effective_device(cls, value: object) -> str:
        device = cls._non_blank(value, "device").lower()
        if device == "rocm":
            return "cuda"
        if device in {"cpu", "cuda"}:
            return device
        if device.startswith("cuda:") and device.removeprefix("cuda:").isdigit():
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
            for _, field_name, format_spec, conversion in string.Formatter().parse(
                query_format
            ):
                if field_name is None:
                    continue
                if field_name != "query" or format_spec or conversion:
                    raise cls._error("query_format may contain only the {query} field")
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
                    f"{field_name} cannot be validated because the pinned dependency is unavailable"
                ) from error
            if declared != installed:
                raise cls._error(
                    f"{field_name} does not match the installed pinned dependency"
                )

    @classmethod
    def _validated_model_config(cls, value: object) -> dict[str, object]:
        from collections.abc import Mapping

        if not isinstance(value, Mapping) or set(value) != set(cls._CONFIG_FIELDS):
            raise cls._error("model_config must contain exactly the scorer settings")
        values = dict(value)
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
        if values["tokenizer_name"] != values["model_name"]:
            raise cls._error("tokenizer_name must match model_name")
        if values["tokenizer_revision"] != values["revision"]:
            raise cls._error("tokenizer_revision must match revision")
        for field_name in ("schema_version", "batch_size", "max_length"):
            field_value = values[field_name]
            if (
                isinstance(field_value, bool)
                or not isinstance(field_value, int)
                or field_value <= 0
            ):
                raise cls._error(f"{field_name} must be a positive integer")
        values["device"] = cls._effective_device(values["device"])
        values["precision"] = cls._effective_precision(values["precision"])
        query_format = cls._non_blank(values["query_format"], "query_format")
        cls._validate_query_format(query_format)
        values["query_format"] = query_format
        cls._validate_runtime_versions(values)
        return values

    @classmethod
    def _fingerprint_for_values(cls, values: dict[str, object]) -> str:
        import hashlib

        import msgspec

        material = msgspec.msgpack.encode(
            (
                "local-cross-encoder-fingerprint-v2",
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

    def __init__(self, loader: object | None = None) -> None:
        import threading

        if loader is not None and not callable(loader):
            raise self._error("loader must be callable")
        self._loader = loader if loader is not None else self._default_loader
        self._state_lock = threading.Lock()
        self._model: object | None = None
        self._tokenizer: object | None = None
        self._model_config: dict[str, object] | None = None
        self._fingerprint: str | None = None

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
        if model is None or not callable(getattr(model, "predict", None)):
            raise ValueError("loader did not return a predictive model")
        if tokenizer is None or not callable(tokenizer):
            raise ValueError("loader did not return a callable tokenizer")
        return model, tokenizer

    @staticmethod
    def _default_loader(config: dict[str, object]) -> tuple[object, object]:
        import torch
        from sentence_transformers import CrossEncoder

        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config["precision"]]
        model = CrossEncoder(
            config["model_name"],
            revision=config["revision"],
            device=config["device"],
            max_length=config["max_length"],
            model_kwargs={
                "torch_dtype": dtype,
            },
        )
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None or not callable(tokenizer):
            raise ValueError("loaded model did not expose a callable tokenizer")
        return model, tokenizer

    def _ensure_loaded(
        self,
        config: dict[str, object],
        fingerprint: str,
    ) -> tuple[object, object]:
        with self._state_lock:
            if self._fingerprint is not None:
                if self._fingerprint != fingerprint or self._model_config != config:
                    raise self._error("worker scorer fingerprint drift")
            else:
                try:
                    model, tokenizer = self._unpack_loader_result(self._loader(config))
                except Exception as error:
                    raise self._error("worker model loading failed") from error
                self._model_config = config
                self._fingerprint = fingerprint
                self._model = model
                self._tokenizer = tokenizer

            if self._model is None or self._tokenizer is None:
                raise self._error("worker model state is incomplete")
            return self._model, self._tokenizer

    @classmethod
    def _token_length(cls, encoded: object) -> int:
        from collections.abc import Mapping, Sequence

        if not isinstance(encoded, Mapping):
            raise cls._error("tokenizer returned a non-mapping result")
        input_ids = cls._as_sequence(encoded.get("input_ids"))
        if input_ids is None or isinstance(input_ids, (str, bytes, bytearray)):
            raise cls._error("tokenizer returned malformed input_ids")
        try:
            items = list(input_ids)
        except TypeError as error:
            raise cls._error("tokenizer returned malformed input_ids") from error
        if items and isinstance(cls._as_sequence(items[0]), (list, tuple)):
            length = max(
                (len(cls._as_sequence(item)) for item in items),
                default=0,
            )
        else:
            length = len(items)

        if "length" in encoded:
            reported = cls._as_sequence(encoded.get("length"))
            if isinstance(reported, bool):
                raise cls._error("tokenizer returned malformed length")
            if isinstance(reported, int):
                reported_length = reported
            elif isinstance(reported, Sequence) and not isinstance(
                reported, (str, bytes, bytearray)
            ):
                values = list(reported)
                if (
                    len(values) != 1
                    or isinstance(values[0], bool)
                    or not isinstance(values[0], int)
                ):
                    raise cls._error("tokenizer returned malformed length")
                reported_length = values[0]
            else:
                raise cls._error("tokenizer did not return a valid length")
            if reported_length < 0 or reported_length != length:
                raise cls._error("tokenizer length does not match input_ids")

        overflow = cls._as_sequence(encoded.get("overflowing_tokens"))
        if overflow not in (None, [], (), False):
            raise cls._error("tokenizer reported overflow")
        truncated = cls._as_sequence(encoded.get("num_truncated_tokens"))
        if truncated not in (None, 0, [], (), False, [0], (0,)):
            raise cls._error("tokenizer reported truncation")
        return length

    @classmethod
    def _tokenize_pair(
        cls,
        tokenizer: object,
        query: str,
        passage: str,
        *,
        max_length: int,
    ) -> object:
        chat_template = getattr(tokenizer, "chat_template", None)
        if isinstance(chat_template, str) and chat_template.strip():
            apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
            if not callable(apply_chat_template):
                raise cls._error(
                    "tokenizer with a chat_template must expose apply_chat_template"
                )
            return apply_chat_template(
                [
                    {"role": "query", "content": query},
                    {"role": "document", "content": passage},
                ],
                tokenize=True,
                add_generation_prompt=False,
                truncation=False,
                return_dict=True,
            )
        return tokenizer(
            query,
            passage,
            truncation=False,
            max_length=max_length,
            return_length=True,
        )

    @classmethod
    def _format_query(cls, config: dict[str, object], query: object) -> str:
        if not isinstance(query, str) or not query.strip():
            raise cls._error("query must be a non-blank string")
        try:
            return config["query_format"].format(query=query)
        except (IndexError, KeyError, ValueError, AttributeError) as error:
            raise cls._error("query could not be formatted") from error

    @classmethod
    def _validate_rows(cls, rows: object) -> list[tuple[bytes, str]]:
        import hashlib
        from collections.abc import Mapping

        if isinstance(rows, (str, bytes, bytearray)) or not isinstance(rows, list):
            raise cls._error("rows must be a list")
        normalized: list[tuple[bytes, str]] = []
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != {"hash", "text"}:
                raise cls._error("worker rows must contain exactly hash and text")
            hash_value = row["hash"]
            text = row["text"]
            if (
                not isinstance(hash_value, str)
                or len(hash_value) != 64
                or hash_value.lower() != hash_value
            ):
                raise cls._error("worker row hash is malformed")
            try:
                content_hash = bytes.fromhex(hash_value)
            except ValueError as error:
                raise cls._error("worker row hash is malformed") from error
            if len(content_hash) != 32:
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
    def _validate_request(
        cls, request: object
    ) -> tuple[dict[str, object], str, str, list[tuple[bytes, str]]]:
        from collections.abc import Mapping

        if not isinstance(request, Mapping) or set(request) != cls._REQUEST_FIELDS:
            raise cls._error("worker request fields are invalid")
        schema_version = request["schema_version"]
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != cls._WIRE_SCHEMA_VERSION
        ):
            raise cls._error("unsupported Flash wire schema version")
        config = cls._validated_model_config(request["model_config"])
        fingerprint = request["scorer_fingerprint"]
        if not isinstance(fingerprint, str) or not fingerprint.strip():
            raise cls._error("scorer fingerprint is invalid")
        expected_fingerprint = cls._fingerprint_for_values(config)
        if fingerprint != expected_fingerprint:
            raise cls._error("worker scorer fingerprint mismatch")
        query = request["query"]
        if not isinstance(query, str) or not query.strip():
            raise cls._error("query must be a non-blank string")
        return config, expected_fingerprint, query, cls._validate_rows(request["rows"])

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
            raise cls._error("model returned an unexpected number of scores")
        return values

    @classmethod
    def _score_rows(
        cls,
        config: dict[str, object],
        query: str,
        rows: list[tuple[bytes, str]],
        model: object,
        tokenizer: object,
    ) -> dict[bytes, float]:
        import math
        import numbers

        formatted_query = cls._format_query(config, query)
        pairs = [(formatted_query, text) for _, text in rows]
        for pair_query, passage in pairs:
            try:
                encoded = cls._tokenize_pair(
                    tokenizer,
                    pair_query,
                    passage,
                    max_length=config["max_length"],
                )
                length = cls._token_length(encoded)
            except Exception as error:
                raise cls._error(
                    "tokenizer failed while validating a passage pair"
                ) from error
            if length > config["max_length"]:
                raise cls._error("tokenizer pair overflowed max_length")

        raw_scores: list[object] = []
        for start in range(0, len(pairs), config["batch_size"]):
            batch = pairs[start : start + config["batch_size"]]
            try:
                import torch

                prediction = model.predict(
                    batch,
                    batch_size=config["batch_size"],
                    show_progress_bar=False,
                    activation_fn=torch.nn.Identity(),
                    apply_softmax=False,
                    convert_to_numpy=True,
                    convert_to_tensor=False,
                    device=config["device"],
                    processing_kwargs={
                        "text": {
                            "truncation": False,
                            "max_length": config["max_length"],
                        }
                    },
                )
            except Exception as error:
                raise cls._error("model prediction failed") from error
            raw_scores.extend(cls._prediction_values(prediction, len(batch)))

        normalized: dict[bytes, float] = {}
        for (content_hash, _), value in zip(rows, raw_scores, strict=True):
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise cls._error("model returned a non-numeric score")
            try:
                score = float(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise cls._error("model returned an invalid score") from error
            if not math.isfinite(score):
                raise cls._error("model returned a non-finite score")
            normalized[content_hash] = score
        return {key: normalized[key] for key in sorted(normalized)}

    def score(self, request: object) -> dict[str, object]:
        try:
            config, fingerprint, query, rows = self._validate_request(request)
            model, tokenizer = self._ensure_loaded(config, fingerprint)
            scores = self._score_rows(config, query, rows, model, tokenizer)
            return {
                "schema_version": self._WIRE_SCHEMA_VERSION,
                "scorer_fingerprint": fingerprint,
                "scores": [
                    {"hash": content_hash.hex(), "score": score}
                    for content_hash, score in scores.items()
                ],
            }
        except self._ERROR:
            raise
        except Exception as error:
            raise self._error("worker scoring failed") from error


def _resolve_gpu(gpu_name: str) -> object:
    """Resolve Runpod's GPU enum only when decoration is requested."""

    try:
        from runpod_flash import GpuGroup, GpuType
    except ImportError as error:
        raise RuntimeError(
            "Runpod Flash is required to decorate a worker; inject gpu_resolver for offline use"
        ) from error
    for enum in (GpuGroup, GpuType):
        try:
            return getattr(enum, gpu_name)
        except AttributeError:
            continue
    raise ValueError(f"unknown Runpod Flash GPU group or type: {gpu_name!r}")


def _pinned_dependencies(config: object) -> list[str]:
    import importlib.metadata

    try:
        msgspec_version = importlib.metadata.version("msgspec")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("msgspec is required by the Flash worker") from error
    return [
        f"sentence-transformers=={config.sentence_transformers_version}",
        f"transformers=={config.transformers_version}",
        f"tokenizers=={config.tokenizers_version}",
        f"torch=={config.torch_version}",
        f"msgspec=={msgspec_version}",
    ]


def _effective_config(config: object | None) -> tuple[object, str, str, int, float]:
    from .flash_passage_scorer import (
        DEFAULT_FLASH_GPU,
        DEFAULT_FLASH_IDLE_TIMEOUT,
        DEFAULT_FLASH_WORKER_NAME,
        RunpodFlashConfig,
    )
    from .passage_scoring import LocalCrossEncoderConfig

    if config is None:
        model_config = LocalCrossEncoderConfig()
        return (
            model_config,
            DEFAULT_FLASH_WORKER_NAME,
            DEFAULT_FLASH_GPU,
            1,
            DEFAULT_FLASH_IDLE_TIMEOUT,
        )
    if isinstance(config, RunpodFlashConfig):
        return (
            config.effective_model_config,
            config.worker_name,
            config.gpu,
            config.workers,
            config.idle_timeout,
        )
    if isinstance(config, LocalCrossEncoderConfig):
        return (
            config,
            DEFAULT_FLASH_WORKER_NAME,
            DEFAULT_FLASH_GPU,
            1,
            DEFAULT_FLASH_IDLE_TIMEOUT,
        )
    raise TypeError(
        "config must be RunpodFlashConfig, LocalCrossEncoderConfig, or None"
    )


def build_flash_worker(
    config: object | None = None,
    *,
    endpoint_factory: Callable[..., object] | None = None,
    gpu_resolver: Callable[[str], object] | None = None,
) -> type[FlashPassageWorker]:
    """Return the static worker and optionally apply a lazy Flash decorator."""

    model_config, worker_name, gpu_name, workers, idle_timeout = _effective_config(
        config
    )
    if endpoint_factory is None:
        return FlashPassageWorker

    resolve_gpu = gpu_resolver or _resolve_gpu
    decorator = endpoint_factory(
        name=worker_name,
        gpu=resolve_gpu(gpu_name),
        workers=workers,
        idle_timeout=idle_timeout,
        dependencies=_pinned_dependencies(model_config),
    )
    if not callable(decorator):
        raise TypeError("Flash endpoint factory must return a class decorator")
    decorated = decorator(FlashPassageWorker)
    if not isinstance(decorated, type):
        raise TypeError("Flash endpoint decorator must return a worker class")
    return decorated


__all__ = ["FlashPassageWorker", "build_flash_worker"]
