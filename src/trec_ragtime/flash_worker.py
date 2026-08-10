"""Self-contained class-style warm worker definition for Flash."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def build_flash_worker(
    config: object | None = None,
    *,
    loader: Callable[..., object] | None = None,
    endpoint_factory: Callable[[type[Any]], type[Any]] | None = None,
) -> type[Any]:
    """Build or lazily decorate a source-stable class-style Flash worker.

    The effective local model policy is rendered as primitive literals inside
    the returned class body because Flash ships only that source.  ``loader``
    is an explicit local-test seam and is deliberately not part of the class
    source or the normal remote path.
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
        "device": effective_config.device,
        "max_length": effective_config.max_length,
        "model_name": effective_config.model_name,
        "precision": effective_config.precision,
        "query_format": effective_config.query_format,
        "query_policy": effective_config.query_policy,
        "revision": effective_config.revision,
        "schema_version": effective_config.schema_version,
        "sentence_transformers_version": effective_config.sentence_transformers_version,
    }
    model_literal = json.dumps(
        model_config,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    class_source = """class FlashPassageWorker:
    _MODEL_CONFIG = __MODEL_CONFIG__
    _default_loader: object | None = None

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
        selected_loader = loader
        if selected_loader is None:
            selected_loader = getattr(type(self), "_default_loader", None)
        if selected_loader is None:
            self._scorer = LocalCrossEncoderScorer(effective_config)
        else:
            self._scorer = LocalCrossEncoderScorer(effective_config, loader=selected_loader)
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
    sys.modules[module_name] = module
    namespace = module.__dict__
    exec(compile(class_source, filename, "exec"), namespace)
    worker_class = namespace["FlashPassageWorker"]
    worker_class._default_loader = loader
    if endpoint_factory is not None:
        return endpoint_factory(worker_class)
    return worker_class


__all__ = ["build_flash_worker"]
