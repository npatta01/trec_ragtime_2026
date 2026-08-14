"""Resumable CLI for topic-union maximum-information-need scoring."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import tempfile
import time
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from .paths import default_project_cache_dir
from .topic_union_cache import TopicScoreArtifactCache, TopicScoreFingerprint, TopicScoreRequest
from .topic_union_scoring import score_topic_union, select_conservative_pair_length
from .topic_union_snapshot import TopicSnapshot, build_snapshot, hydrate_snapshot_topic


def _experiment_root() -> Path:
    return default_project_cache_dir() / "experiments/planner-static-final-v1"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def select_topics(
    topics: Sequence[TopicSnapshot],
    topic_ids: Sequence[str] | None,
    topic_limit: int | None,
    *,
    complete_only: bool = False,
) -> tuple[TopicSnapshot, ...]:
    """Select exact IDs or richest topics, never filesystem order."""

    rows = tuple(row for row in topics if not complete_only or row.is_complete)
    if topic_ids:
        wanted = set(topic_ids)
        selected = tuple(row for row in rows if row.topic_id in wanted)
        missing = wanted - {row.topic_id for row in selected}
        if missing:
            raise ValueError(f"unknown topics: {sorted(missing)}")
        return tuple(sorted(selected, key=lambda row: row.topic_id))
    ordered = tuple(sorted(rows, key=lambda row: (-row.successful_query_count, row.topic_id)))
    if topic_limit is None:
        return ordered
    if isinstance(topic_limit, bool) or not isinstance(topic_limit, int) or topic_limit < 1:
        raise ValueError("topic_limit must be positive")
    return ordered[:topic_limit]


def _snapshot_record(topics: Sequence[TopicSnapshot]) -> dict[str, Any]:
    return {
        "schema_version": "topic-union-scoring-snapshot-v1",
        "topic_count": len(topics),
        "successful_query_count": sum(row.successful_query_count for row in topics),
        "topics": [
            {
                "topic_id": row.topic_id,
                "plan_sha256": row.plan_sha256,
                "information_need_count": len(row.needs),
                "successful_query_count": row.successful_query_count,
                "sealed_query_count": row.sealed_query_count,
                "complete": row.is_complete,
                "input_object_key_count": len(row.input_object_keys),
                "input_digest": __import__("hashlib").sha256(
                    "\n".join(row.input_object_keys).encode()
                ).hexdigest(),
            }
            for row in topics
        ],
    }


def write_dry_run(output: Path, topics: Sequence[TopicSnapshot]) -> None:
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output, 0o700)
    _atomic_json(output / "snapshot.json", _snapshot_record(topics))


def _shared_tokenizer_loader():
    tokenizer = None

    def load(config):
        nonlocal tokenizer
        if tokenizer is None:
            from .passage_scoring import _default_tokenizer_loader

            tokenizer = _default_tokenizer_loader(config)
        return tokenizer

    return load


class _CachedNeedScorer:
    def __init__(self, store: object, scorer: object, *, chunker_fingerprint: str, representation: str) -> None:
        from .passage_score_cache import PassageScoreCache

        self._cache = PassageScoreCache(
            store,
            scorer,  # type: ignore[arg-type]
            chunker_fingerprint=chunker_fingerprint,
            representation=representation,
        )
        self.fingerprint = self._cache.scorer_fingerprint

    def score(self, query, passages):
        return self._cache.score(query, passages)


def _open_scorer(arguments: argparse.Namespace, stack: ExitStack):
    from .passage_scoring import LocalCrossEncoderConfig, LocalCrossEncoderScorer

    if arguments.scorer == "local":
        config = LocalCrossEncoderConfig(device=arguments.device)
        return config, LocalCrossEncoderScorer(config)
    if not arguments.flash_endpoint_id:
        raise ValueError("--scorer flash requires --flash-endpoint-id")
    from .flash_passage_scorer import FlashEndpointBatchRunner, RunpodFlashConfig, RunpodFlashPassageScorer

    flash = RunpodFlashConfig(
        endpoint_id=arguments.flash_endpoint_id,
        model_config=LocalCrossEncoderConfig(device="cuda"),
        workers=(0, arguments.flash_workers),
        concurrency=arguments.flash_workers,
    )
    runner = stack.enter_context(FlashEndpointBatchRunner(flash))
    return flash.effective_model_config, stack.enter_context(RunpodFlashPassageScorer(flash, runner))


def build_parser() -> argparse.ArgumentParser:
    root = _experiment_root()
    parser = argparse.ArgumentParser(prog="ragtime-topic-score")
    parser.add_argument("--plans", type=Path, default=root / "dry-runs/all-103-2026-08-12/all-103-final-plans.jsonl")
    parser.add_argument("--coverage-index", type=Path, default=root / "retrieval-gates/deadline-all-topics-coverage-2026-08-12/topics.jsonl")
    parser.add_argument("--checkpoint-dir", type=Path, default=root / "retrieval-gates/full-richer-2026-08-13/queries")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topic-id", action="append", dest="topic_ids")
    parser.add_argument("--topic-limit", type=int)
    parser.add_argument("--complete-only", action="store_true")
    parser.add_argument("--representation", choices=("canonical", "report_text"), default="report_text")
    parser.add_argument("--scorer", choices=("local", "flash"), default="local")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--flash-endpoint-id")
    parser.add_argument("--flash-workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from dotenv import load_dotenv

    arguments = build_parser().parse_args(argv)
    load_dotenv()
    try:
        topics = select_topics(
            build_snapshot(arguments.plans, arguments.coverage_index, arguments.checkpoint_dir),
            arguments.topic_ids,
            arguments.topic_limit,
            complete_only=arguments.complete_only,
        )
        write_dry_run(arguments.output, topics)
        if arguments.dry_run:
            return 0

        from .artifact_store import ArtifactStore
        from .local_corpus import LocalCorpusIndex
        from .passage_scoring import CrossEncoderPairLengthFunction
        from .passages import PassageChunker

        store = ArtifactStore.from_env()
        cache = TopicScoreArtifactCache(store)
        outcomes: list[dict[str, Any]] = []
        with ExitStack() as stack, LocalCorpusIndex() as corpus:
            config, scorer = _open_scorer(arguments, stack)
            tokenizer_loader = _shared_tokenizer_loader()
            for position, topic in enumerate(topics, start=1):
                started = time.monotonic()
                try:
                    pair_length = select_conservative_pair_length(
                        config,
                        topic.needs,
                        factory=lambda value, text: CrossEncoderPairLengthFunction(
                            value, text, tokenizer_loader=tokenizer_loader
                        ),
                    )
                    chunker = PassageChunker(pair_length)
                    cached_scorer = _CachedNeedScorer(
                        store,
                        scorer,
                        chunker_fingerprint=chunker.fingerprint,
                        representation=arguments.representation,
                    )
                    fingerprint = TopicScoreFingerprint(
                        cached_scorer.fingerprint,
                        chunker.fingerprint,
                        arguments.representation,
                    )
                    request = TopicScoreRequest(
                        topic.topic_id,
                        topic.plan_sha256,
                        topic.needs,
                        topic.input_object_keys,
                    )
                    result = cache.get(fingerprint, request)
                    status = "cached"
                    if result is None:
                        inputs = hydrate_snapshot_topic(topic, corpus)
                        result = score_topic_union(
                            topic.topic_id,
                            topic.needs,
                            inputs,
                            chunker=chunker,
                            scorer=cached_scorer,
                            representation=arguments.representation,
                        )
                        result = dataclasses.replace(result, input_object_keys=topic.input_object_keys)
                        if not cache.put(fingerprint, request, result):
                            raise RuntimeError("topic-score artifact write failed")
                        status = "written"
                    outcomes.append(
                        {
                            "topic_id": topic.topic_id,
                            "status": status,
                            "object_key": cache.object_key(fingerprint, request),
                            "successful_query_count": topic.successful_query_count,
                            "unique_document_count": result.unique_document_count,
                            "passage_count": result.passage_count,
                            "duration_seconds": round(time.monotonic() - started, 3),
                        }
                    )
                except Exception as error:  # one topic must not stop the wave
                    outcomes.append(
                        {
                            "topic_id": topic.topic_id,
                            "status": "failed",
                            "reason": type(error).__name__,
                            "duration_seconds": round(time.monotonic() - started, 3),
                        }
                    )
                _atomic_json(
                    arguments.output / "progress.json",
                    {
                        "schema_version": "topic-union-scoring-progress-v1",
                        "topic_count": len(topics),
                        "completed_topic_count": position,
                        "successful_topic_count": sum(row["status"] in {"cached", "written"} for row in outcomes),
                        "failed_topic_count": sum(row["status"] == "failed" for row in outcomes),
                        "outcomes": outcomes,
                    },
                )
        _atomic_json(arguments.output / "manifest.json", {
            "schema_version": "topic-union-scoring-manifest-v1",
            "topic_count": len(topics),
            "complete_topic_count": sum(row["status"] in {"cached", "written"} for row in outcomes),
            "failed_topic_count": sum(row["status"] == "failed" for row in outcomes),
            "outcomes": outcomes,
        })
        return 0 if all(row["status"] in {"cached", "written"} for row in outcomes) else 1
    except Exception as error:
        print(f"topic scoring setup failed: {type(error).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
