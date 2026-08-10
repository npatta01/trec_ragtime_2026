"""Strict passage scoring contracts and a lazy local CrossEncoder scorer."""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import numbers
import string
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, cast, runtime_checkable

import msgspec

from .passages import Passage, sha256_bytes


DEFAULT_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
DEFAULT_MODEL_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
DEFAULT_TOKENIZER_NAME = DEFAULT_MODEL_NAME
DEFAULT_TOKENIZER_REVISION = DEFAULT_MODEL_REVISION
DEFAULT_SCHEMA_VERSION = 1
DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 32
DEFAULT_QUERY_FORMAT = "{query}"
DEFAULT_QUERY_POLICY = "literal-v1"
_HASH_LENGTH = 32
_FINGERPRINT_PREFIX = "local-cross-encoder-v2/"
_PAIR_LENGTH_FINGERPRINT_PREFIX = "cross-encoder-pair-length-v1/"
_RUNTIME_DISTRIBUTIONS = (
    ("sentence_transformers_version", "sentence-transformers"),
    ("transformers_version", "transformers"),
    ("tokenizers_version", "tokenizers"),
    ("torch_version", "torch"),
)
_PRECISION_ALIASES = {
    "float32": "float32",
    "fp32": "float32",
    "float16": "float16",
    "fp16": "float16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
}


class PassageScoringError(ValueError):
    """Raised when a query, passage, tokenizer, model, or score is invalid."""


class PassageScoreValidationError(PassageScoringError):
    """Raised when a score mapping does not exactly validate."""


def _installed_runtime_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _installed_sentence_transformers_version() -> str:
    return _installed_runtime_version("sentence-transformers")


def _installed_transformers_version() -> str:
    return _installed_runtime_version("transformers")


def _installed_tokenizers_version() -> str:
    return _installed_runtime_version("tokenizers")


def _installed_torch_version() -> str:
    return _installed_runtime_version("torch")


def _non_blank_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")
    if any(character in value for character in "\r\n"):
        raise ValueError(f"{field_name} must not contain newlines")
    return value


def _effective_device(value: object) -> str:
    device = _non_blank_string(value, "device").lower()
    if device == "rocm":
        return "cuda"
    if device == "cpu" or device == "cuda" or device.startswith("cuda:"):
        if device.startswith("cuda:") and not device.removeprefix("cuda:").isdigit():
            raise ValueError("device must be cpu, cuda, cuda:N, or rocm")
        return device
    raise ValueError("device must be cpu, cuda, cuda:N, or rocm")


def _effective_precision(value: object) -> str:
    precision = _non_blank_string(value, "precision").lower()
    try:
        return _PRECISION_ALIASES[precision]
    except KeyError as error:
        raise ValueError("precision must be float32, float16, or bfloat16") from error


def _query_format_fields(query_format: str) -> tuple[str, ...]:
    fields: list[str] = []
    try:
        parsed = string.Formatter().parse(query_format)
        for _, field_name, format_spec, conversion in parsed:
            if field_name is None:
                continue
            if field_name != "query" or format_spec or conversion:
                raise ValueError("query_format may contain only the {query} field")
            fields.append(field_name)
    except (ValueError, IndexError) as error:
        raise ValueError("query_format must be a valid format containing {query}") from error
    if fields != ["query"]:
        raise ValueError("query_format must contain exactly one {query} field")
    return tuple(fields)


def _validate_runtime_versions(config: "LocalCrossEncoderConfig") -> None:
    for field_name, distribution in _RUNTIME_DISTRIBUTIONS:
        declared = getattr(config, field_name)
        installed = _installed_runtime_version(distribution)
        if installed == "unavailable":
            raise ValueError(
                f"{field_name} cannot be validated because {distribution!r} is not installed"
            )
        if declared != installed:
            raise ValueError(
                f"{field_name}={declared!r} does not match installed "
                f"{distribution}={installed!r}"
            )


def _fingerprint_for_config(config: "LocalCrossEncoderConfig") -> str:
    material = msgspec.msgpack.encode(
        (
            "local-cross-encoder-fingerprint-v1",
            config.schema_version,
            config.model_name,
            config.revision,
            config.tokenizer_name,
            config.tokenizer_revision,
            config.sentence_transformers_version,
            config.transformers_version,
            config.tokenizers_version,
            config.torch_version,
            config.effective_device,
            config.effective_precision,
            config.batch_size,
            config.max_length,
            config.query_format,
            config.query_policy,
        )
    )
    digest = hashlib.sha256(material).hexdigest()
    return _FINGERPRINT_PREFIX + digest


@dataclass(frozen=True, slots=True)
class LocalCrossEncoderConfig:
    """Immutable, fingerprinted identity for local passage scoring."""

    schema_version: int = DEFAULT_SCHEMA_VERSION
    model_name: str = DEFAULT_MODEL_NAME
    revision: str = DEFAULT_MODEL_REVISION
    tokenizer_name: str = DEFAULT_TOKENIZER_NAME
    tokenizer_revision: str = DEFAULT_TOKENIZER_REVISION
    sentence_transformers_version: str = field(
        default_factory=_installed_sentence_transformers_version
    )
    transformers_version: str = field(default_factory=_installed_transformers_version)
    tokenizers_version: str = field(default_factory=_installed_tokenizers_version)
    torch_version: str = field(default_factory=_installed_torch_version)
    device: str = "cpu"
    precision: str = "float32"
    batch_size: int = DEFAULT_BATCH_SIZE
    max_length: int = DEFAULT_MAX_LENGTH
    query_format: str = DEFAULT_QUERY_FORMAT
    query_policy: str = DEFAULT_QUERY_POLICY
    _effective_device_value: str = field(init=False, repr=False)
    _effective_precision_value: str = field(init=False, repr=False)
    _fingerprint: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int):
            raise ValueError("schema_version must be an integer")
        if self.schema_version <= 0:
            raise ValueError("schema_version must be positive")
        _non_blank_string(self.model_name, "model_name")
        _non_blank_string(self.revision, "revision")
        _non_blank_string(self.tokenizer_name, "tokenizer_name")
        _non_blank_string(self.tokenizer_revision, "tokenizer_revision")
        _non_blank_string(
            self.sentence_transformers_version,
            "sentence_transformers_version",
        )
        effective_device = _effective_device(self.device)
        effective_precision = _effective_precision(self.precision)
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise ValueError("batch_size must be an integer")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if isinstance(self.max_length, bool) or not isinstance(self.max_length, int):
            raise ValueError("max_length must be an integer")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        _non_blank_string(self.query_format, "query_format")
        _query_format_fields(self.query_format)
        _non_blank_string(self.query_policy, "query_policy")
        _validate_runtime_versions(self)
        object.__setattr__(self, "_effective_device_value", effective_device)
        object.__setattr__(self, "_effective_precision_value", effective_precision)
        object.__setattr__(self, "_fingerprint", _fingerprint_for_config(self))

    @property
    def effective_device(self) -> str:
        """Return the PyTorch device string used for loading and prediction."""

        return self._effective_device_value

    @property
    def effective_precision(self) -> str:
        """Return the canonical precision name used by the loader."""

        return self._effective_precision_value

    @property
    def model_name_or_path(self) -> str:
        """Compatibility name matching sentence-transformers' constructor."""

        return self.model_name

    @property
    def model_revision(self) -> str:
        """Compatibility name for the pinned model revision."""

        return self.revision

    @property
    def max_pair_length(self) -> int:
        """Compatibility name for the configured maximum pair length."""

        return self.max_length

    @property
    def fingerprint(self) -> str:
        """Stable identity for every result-affecting scoring setting."""

        return self._fingerprint


@runtime_checkable
class PassageScorer(Protocol):
    """Common synchronous scorer protocol."""

    @property
    def fingerprint(self) -> str: ...

    def score(
        self,
        query: str,
        passages: Sequence[Passage],
    ) -> Mapping[bytes, float]: ...


def _validate_hash(value: object, field_name: str) -> bytes:
    if not isinstance(value, bytes) or len(value) != _HASH_LENGTH:
        raise PassageScoreValidationError(f"{field_name} must be exactly 32 bytes")
    return value


def _passage_lineage(passage: Passage) -> tuple[object, ...]:
    return (
        passage.document_id,
        passage.representation,
        passage.representation_hash,
        passage.start_char,
        passage.end_char,
        passage.chunk_index,
        passage.passage_id,
    )


def _unique_passages(passages: Sequence[Passage]) -> tuple[tuple[bytes, str], ...]:
    if isinstance(passages, (str, bytes, bytearray)) or not isinstance(passages, Sequence):
        raise PassageScoringError("passages must be a sequence of Passage objects")

    unique: dict[bytes, tuple[str, tuple[object, ...]]] = {}
    for passage in passages:
        if not isinstance(passage, Passage):
            raise PassageScoringError("passages must contain Passage objects")
        content_hash = _validate_hash(passage.content_hash, "content_hash")
        _validate_hash(passage.passage_id, "passage_id")
        if not isinstance(passage.text, str) or not passage.text:
            raise PassageScoringError("passage text must not be empty")
        if content_hash != sha256_bytes(passage.text):
            raise PassageScoringError("content_hash does not match passage text")
        lineage = _passage_lineage(passage)
        previous = unique.get(content_hash)
        if previous is None:
            unique[content_hash] = (passage.text, lineage)
            continue
        previous_text, previous_lineage = previous
        if passage.text != previous_text:
            raise PassageScoringError("a content hash maps to contradictory passage text")
        if passage.passage_id == cast(bytes, previous_lineage[-1]) and lineage != previous_lineage:
            raise PassageScoringError("a content hash maps to contradictory passage lineage")

    return tuple((content_hash, text) for content_hash, (text, _) in unique.items())


def _normalize_score(value: object, *, key: object) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise PassageScoreValidationError(
            f"score for key {key!r} must be a non-Boolean numeric value"
        )
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise PassageScoreValidationError("score could not be normalized to float") from error
    if not math.isfinite(normalized):
        raise PassageScoreValidationError("scores must be finite")
    return normalized


def validate_passage_scores(
    passages: Sequence[Passage],
    scores: Mapping[bytes, object],
) -> dict[bytes, float]:
    """Require exact unique content-hash coverage and finite numeric scores."""

    requested = _unique_passages(passages)
    if not isinstance(scores, Mapping):
        raise PassageScoreValidationError("scores must be a mapping")

    normalized: dict[bytes, float] = {}
    for key, value in scores.items():
        content_hash = _validate_hash(key, "score key")
        normalized[content_hash] = _normalize_score(value, key=key)

    expected_keys = {content_hash for content_hash, _ in requested}
    actual_keys = set(normalized)
    if actual_keys != expected_keys:
        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys
        raise PassageScoreValidationError(
            f"score coverage mismatch (missing={len(missing)}, extra={len(extra)})"
        )
    return {key: normalized[key] for key in sorted(normalized)}


def _format_query(config: LocalCrossEncoderConfig, query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise PassageScoringError("query must be a non-blank string")
    try:
        return config.query_format.format(query=query)
    except (IndexError, KeyError, ValueError) as error:
        raise PassageScoringError("query could not be formatted") from error


def _default_tokenizer_loader(config: LocalCrossEncoderConfig) -> Any:
    """Load only the explicitly pinned tokenizer, without model weights."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name,
        revision=config.tokenizer_revision,
    )
    if not callable(tokenizer):
        raise PassageScoringError("AutoTokenizer returned a non-callable tokenizer")
    return tokenizer


def _default_loader(config: LocalCrossEncoderConfig) -> tuple[Any, Any]:
    """Load the pinned local model and assign its explicitly pinned tokenizer."""

    tokenizer = _default_tokenizer_loader(config)
    from sentence_transformers import CrossEncoder
    import torch

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[config.effective_precision]
    model = CrossEncoder(
        config.model_name,
        revision=config.revision,
        device=config.effective_device,
        max_length=config.max_length,
        model_kwargs={"torch_dtype": dtype},
    )
    try:
        model.tokenizer = tokenizer
    except Exception as error:
        raise PassageScoringError("loaded CrossEncoder does not accept the pinned tokenizer") from error
    return model, tokenizer


def _unpack_loader_result(result: object) -> tuple[Any, Any]:
    if isinstance(result, tuple) and len(result) == 2:
        model, tokenizer = result
    elif hasattr(result, "model") and hasattr(result, "tokenizer"):
        model = getattr(result, "model")
        tokenizer = getattr(result, "tokenizer")
    else:
        model = result
        tokenizer = getattr(result, "tokenizer", None)
    if model is None or tokenizer is None or not callable(getattr(model, "predict", None)):
        raise PassageScoringError("loader must return a CrossEncoder and tokenizer")
    if not callable(tokenizer):
        raise PassageScoringError("loader must return a callable tokenizer")
    return model, tokenizer


def _as_sequence(value: object) -> object:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return tolist()
        except (TypeError, ValueError):
            pass
    return value


def _token_length(encoded: object) -> int:
    if not isinstance(encoded, Mapping):
        raise PassageScoringError("tokenizer returned a non-mapping result")
    input_ids = _as_sequence(encoded.get("input_ids"))
    if input_ids is None:
        raise PassageScoringError("tokenizer did not return input_ids")
    if isinstance(input_ids, (str, bytes, bytearray)):
        raise PassageScoringError("tokenizer returned malformed input_ids")
    try:
        items = list(cast(Sequence[object], input_ids))
    except TypeError as error:
        raise PassageScoringError("tokenizer returned malformed input_ids") from error
    if items and isinstance(_as_sequence(items[0]), (list, tuple)):
        lengths = [len(cast(Sequence[object], _as_sequence(item))) for item in items]
        length = max(lengths, default=0)
    else:
        length = len(items)

    overflow = _as_sequence(encoded.get("overflowing_tokens"))
    if overflow not in (None, [], (), False):
        raise PassageScoringError("tokenizer reported overflow")
    truncated = encoded.get("num_truncated_tokens")
    if truncated not in (None, 0, [0], (0,)):
        raise PassageScoringError("tokenizer reported truncation")
    return length


class CrossEncoderPairLengthFunction:
    """Lazily count complete pinned-tokenizer tokens for one formatted query."""

    __slots__ = (
        "_config",
        "_query",
        "_formatted_query",
        "_loader",
        "_load_lock",
        "_tokenizer",
        "_fingerprint",
    )

    def __init__(
        self,
        config: LocalCrossEncoderConfig,
        query: str,
        *,
        loader: Callable[[LocalCrossEncoderConfig], object] | None = None,
        tokenizer_loader: Callable[[LocalCrossEncoderConfig], object] | None = None,
    ) -> None:
        if not isinstance(config, LocalCrossEncoderConfig):
            raise TypeError("config must be a LocalCrossEncoderConfig")
        if loader is not None and tokenizer_loader is not None:
            raise TypeError("pass either loader or tokenizer_loader, not both")
        if loader is not None and not callable(loader):
            raise TypeError("loader must be callable")
        if tokenizer_loader is not None and not callable(tokenizer_loader):
            raise TypeError("tokenizer_loader must be callable")
        _validate_runtime_versions(config)
        formatted_query = _format_query(config, query)
        material = msgspec.msgpack.encode(
            (
                "cross-encoder-pair-length-v1",
                config.fingerprint,
                query,
                formatted_query,
            )
        )
        self._config = config
        self._query = query
        self._formatted_query = formatted_query
        self._loader = (
            tokenizer_loader
            or loader
            or _default_tokenizer_loader
        )
        self._load_lock = threading.Lock()
        self._tokenizer: Any | None = None
        self._fingerprint = _PAIR_LENGTH_FINGERPRINT_PREFIX + hashlib.sha256(material).hexdigest()

    @property
    def config(self) -> LocalCrossEncoderConfig:
        """Return the immutable scoring configuration used by this callable."""

        return self._config

    @property
    def query(self) -> str:
        """Return the exact raw query captured by this callable."""

        return self._query

    @property
    def formatted_query(self) -> str:
        """Return the formatted query passed to the tokenizer."""

        return self._formatted_query

    @property
    def fingerprint(self) -> str:
        """Return the stable identity suitable for ChunkerConfig."""

        return self._fingerprint

    @property
    def length_function_fingerprint(self) -> str:
        """Alias usable as ChunkerConfig.length_function_fingerprint."""

        return self._fingerprint

    def _ensure_tokenizer(self) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer
        with self._load_lock:
            if self._tokenizer is None:
                try:
                    tokenizer = self._loader(self._config)
                except PassageScoringError:
                    raise
                except Exception as error:
                    raise PassageScoringError("unable to load the pinned tokenizer") from error
                if not callable(tokenizer):
                    raise PassageScoringError("tokenizer loader must return a callable tokenizer")
                self._tokenizer = tokenizer
        return cast(Any, self._tokenizer)

    def __call__(self, passage: str) -> int:
        if not isinstance(passage, str):
            raise PassageScoringError("passage must be a string")
        tokenizer = self._ensure_tokenizer()
        try:
            encoded = tokenizer(
                self._formatted_query,
                passage,
                truncation=False,
                return_length=True,
            )
        except PassageScoringError:
            raise
        except Exception as error:
            raise PassageScoringError("tokenizer failed while counting a passage pair") from error
        return _token_length(encoded)


def _prediction_values(result: object, expected_count: int) -> list[object]:
    converted = _as_sequence(result)
    if isinstance(converted, (str, bytes, bytearray)):
        values: list[object] = [converted]
    elif isinstance(converted, Sequence):
        values = list(converted)
    else:
        values = [converted]
    if len(values) != expected_count:
        raise PassageScoringError("CrossEncoder returned an unexpected number of scores")
    return values


def _identity_activation() -> Any:
    import torch

    return torch.nn.Identity()


class LocalCrossEncoderScorer:
    """Lazily loaded, deterministic local scorer using raw CrossEncoder logits."""

    __slots__ = ("_config", "_loader", "_load_lock", "_model", "_tokenizer")

    def __init__(
        self,
        config: LocalCrossEncoderConfig | None = None,
        *,
        loader: Callable[[LocalCrossEncoderConfig], object] | None = None,
    ) -> None:
        if config is None:
            config = LocalCrossEncoderConfig()
        if not isinstance(config, LocalCrossEncoderConfig):
            raise TypeError("config must be a LocalCrossEncoderConfig")
        if loader is not None and not callable(loader):
            raise TypeError("loader must be callable")
        _validate_runtime_versions(config)
        self._config = config
        self._loader = loader or _default_loader
        self._load_lock = threading.Lock()
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    @property
    def config(self) -> LocalCrossEncoderConfig:
        """Return the immutable scorer configuration."""

        return self._config

    @property
    def fingerprint(self) -> str:
        return self._config.fingerprint

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        with self._load_lock:
            if self._model is None or self._tokenizer is None:
                try:
                    result = self._loader(self._config)
                    model, tokenizer = _unpack_loader_result(result)
                except PassageScoringError:
                    raise
                except Exception as error:
                    raise PassageScoringError("unable to load the local CrossEncoder") from error
                self._model = model
                self._tokenizer = tokenizer
        return cast(Any, self._model), cast(Any, self._tokenizer)

    def _tokenize_pairs(self, tokenizer: Any, pairs: Sequence[tuple[str, str]]) -> None:
        for query, passage in pairs:
            try:
                encoded = tokenizer(
                    query,
                    passage,
                    truncation=False,
                    max_length=self._config.max_length,
                    return_length=True,
                )
            except PassageScoringError:
                raise
            except Exception as error:
                raise PassageScoringError("tokenizer failed while validating a passage pair") from error
            if _token_length(encoded) > self._config.max_length:
                raise PassageScoringError("tokenizer pair overflowed max_length")

    def score(
        self,
        query: str,
        passages: Sequence[Passage],
    ) -> Mapping[bytes, float]:
        formatted_query = _format_query(self._config, query)
        unique = _unique_passages(passages)
        if not unique:
            return {}

        pairs = tuple((formatted_query, text) for _, text in unique)
        model, tokenizer = self._ensure_loaded()
        self._tokenize_pairs(tokenizer, pairs)

        raw_scores: list[object] = []
        for start in range(0, len(pairs), self._config.batch_size):
            batch = list(pairs[start : start + self._config.batch_size])
            try:
                prediction = model.predict(
                    batch,
                    batch_size=self._config.batch_size,
                    show_progress_bar=False,
                    activation_fn=_identity_activation(),
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
            except PassageScoringError:
                raise
            except Exception as error:
                raise PassageScoringError("CrossEncoder prediction failed") from error
            raw_scores.extend(_prediction_values(prediction, len(batch)))

        raw_mapping = {
            content_hash: raw_score
            for (content_hash, _), raw_score in zip(unique, raw_scores, strict=True)
        }
        return validate_passage_scores(passages, raw_mapping)


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_MAX_LENGTH",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_MODEL_REVISION",
    "DEFAULT_TOKENIZER_NAME",
    "DEFAULT_TOKENIZER_REVISION",
    "DEFAULT_QUERY_FORMAT",
    "DEFAULT_QUERY_POLICY",
    "DEFAULT_SCHEMA_VERSION",
    "CrossEncoderPairLengthFunction",
    "LocalCrossEncoderConfig",
    "LocalCrossEncoderScorer",
    "PassageScoreValidationError",
    "PassageScorer",
    "PassageScoringError",
    "validate_passage_scores",
]
