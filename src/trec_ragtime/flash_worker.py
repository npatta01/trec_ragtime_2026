"""Class-style warm worker definition for optional Runpod Flash scoring."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeAlias, cast

from .flash_passage_scorer import (
    RunpodFlashConfig,
    decode_flash_request,
    encode_flash_response,
)
from .passage_scoring import LocalCrossEncoderConfig, LocalCrossEncoderScorer
from .passages import Passage, sha256_bytes


Loader: TypeAlias = Callable[[LocalCrossEncoderConfig], object]


def _worker_passages(rows: list[dict[str, str]]) -> tuple[Passage, ...]:
    passages: list[Passage] = []
    for index, row in enumerate(rows):
        content_hash = bytes.fromhex(row["hash"])
        text = row["text"]
        passages.append(
            Passage(
                document_id=f"flash-row-{index}",
                source_content_hash=content_hash,
                representation="canonical",
                representation_hash=content_hash,
                language="eng",
                chunk_index=index,
                passage_id=sha256_bytes(f"flash-row:{index}:{row['hash']}"),
                text=text,
                start_char=0,
                end_char=len(text),
                boundary="paragraph_packed",
                previous_passage_id=None,
                next_passage_id=None,
                content_hash=content_hash,
                chunker_fingerprint="flash-worker-v1",
            )
        )
    return tuple(passages)


def build_flash_worker(
    config: RunpodFlashConfig | LocalCrossEncoderConfig | None = None,
    *,
    loader: Loader | None = None,
) -> type[Any]:
    """Return a class-style warm worker without importing ``runpod_flash``.

    The returned class loads the Task 2 scorer and its tokenizer in ``__init__``
    and accepts either the encoded request bytes or the decoded endpoint input.
    """

    if config is None:
        local_config = LocalCrossEncoderConfig()
    elif isinstance(config, RunpodFlashConfig):
        local_config = config.effective_model_config
    elif isinstance(config, LocalCrossEncoderConfig):
        local_config = config
    else:
        raise TypeError("config must be RunpodFlashConfig, LocalCrossEncoderConfig, or None")

    class FlashWorker:
        def __init__(self) -> None:
            self._scorer = LocalCrossEncoderScorer(local_config, loader=loader)
            # Warm exactly once per worker.  The scorer owns the loading lock,
            # tokenizer/model seam, pair overflow check, and raw-logit policy.
            self._scorer._ensure_loaded()

        def __call__(self, payload: bytes | bytearray | str | Mapping[str, object]) -> bytes:
            request = decode_flash_request(payload)
            query = cast(str, request["query"])
            rows = cast(list[dict[str, str]], request["rows"])
            passages = _worker_passages(rows)
            scores = self._scorer.score(query, passages)
            return encode_flash_response(
                [
                    {"hash": content_hash, "score": score}
                    for content_hash, score in scores.items()
                ]
            )

        def score(self, payload: bytes | bytearray | str | Mapping[str, object]) -> bytes:
            return self(payload)

    FlashWorker.__name__ = "FlashPassageWorker"
    FlashWorker.__qualname__ = "FlashPassageWorker"
    return FlashWorker


__all__ = ["build_flash_worker"]
