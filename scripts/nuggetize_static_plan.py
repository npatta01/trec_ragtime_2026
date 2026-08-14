"""Build static-plan nugget bundles from existing retrieval and score caches only."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import msgspec


def _write_manifest(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _shard(value: str) -> tuple[int, int]:
    try:
        index, count = (int(row) for row in value.split("/"))
    except ValueError:
        raise argparse.ArgumentTypeError("shard must look like i/n") from None
    if not 1 <= index <= count:
        raise argparse.ArgumentTypeError("shard must satisfy 1 <= i <= n")
    return index, count


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv()
    from trec_ragtime.artifact_store import ArtifactStore
    from trec_ragtime.nugget_cache import NuggetArtifactCache
    from trec_ragtime.nuggetizer_adapter import (
        NuggetizerBackend,
        NuggetizerConfig,
        backend_fingerprint,
    )
    from trec_ragtime.nuggets import NuggetClusterer
    from trec_ragtime.nuggets_runner import build_fingerprint, run_topic
    from trec_ragtime.passage_score_cache import (
        CacheOnlyPassageScorer,
        CachedPassageScorer,
        PassageScoreCache,
    )
    from trec_ragtime.passage_scoring import CrossEncoderPairLengthFunction, LocalCrossEncoderConfig
    from trec_ragtime.passages import PassageChunker
    from trec_ragtime.paths import default_project_cache_dir
    from trec_ragtime.providers import provider_settings
    from trec_ragtime.reranking import rerank_retrieval
    from trec_ragtime.retrieval_cache import RetrievalEnvelope, _from_wire
    from trec_ragtime.static_nugget_work import StaticRerankingInput, build_static_topic_work
    from trec_ragtime.topic_candidates import load_topics
    from trec_ragtime.topic_union_snapshot import build_snapshot

    root = default_project_cache_dir() / "experiments/planner-static-final-v1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", type=Path, default=root / "dry-runs/all-103-2026-08-12/all-103-final-plans.jsonl")
    parser.add_argument("--coverage-index", type=Path, default=root / "retrieval-gates/deadline-all-topics-coverage-2026-08-12/topics.jsonl")
    parser.add_argument("--checkpoint-dir", type=Path, default=root / "retrieval-gates/full-richer-2026-08-13/queries")
    parser.add_argument("--topics", type=Path, default=Path("topics/topics.all.2026.v0625-fix.jsonl"))
    parser.add_argument("--scoring-wave-manifest", type=Path, required=True)
    parser.add_argument("--model", default="openai/gpt-4.1-mini")
    parser.add_argument("--representation", choices=("canonical", "report_text"), default="report_text")
    parser.add_argument("--shard", type=_shard, default=(1, 1))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    arguments = parser.parse_args()

    wave = json.loads(arguments.scoring_wave_manifest.read_text())
    if wave.get("schema_version") != "richer-per-query-score-wave-v1":
        raise SystemExit("unsupported scoring wave manifest")
    wanted = tuple(wave.get("topic_ids", ()))
    if not wanted or len(set(wanted)) != len(wanted):
        raise SystemExit("scoring wave topic IDs are invalid")
    selected_ids = set(wanted)
    snapshots = {
        row.topic_id: row
        for row in build_snapshot(arguments.plans, arguments.coverage_index, arguments.checkpoint_dir)
        if row.topic_id in selected_ids
    }
    if set(snapshots) != selected_ids or any(not row.is_complete for row in snapshots.values()):
        raise SystemExit("scoring wave no longer resolves to complete topic snapshots")
    if any(any(len(item.object_keys) != 1 for item in row.retrievals) for row in snapshots.values()):
        raise SystemExit("scoring wave contains a fused retrieval input")
    shard_index, shard_count = arguments.shard
    topic_ids = tuple(
        topic_id for position, topic_id in enumerate(wanted) if position % shard_count == shard_index - 1
    )
    if not topic_ids:
        raise SystemExit("selected shard has no topics")
    narratives = {row.topic_id: row.problem_statement for row in load_topics(arguments.topics)}
    if any(topic_id not in narratives for topic_id in topic_ids):
        raise SystemExit("topic catalog lacks a scoring-wave topic")

    store = ArtifactStore.from_env()
    first = snapshots[topic_ids[0]].retrievals[0]
    first_payload = store.read(first.object_keys[0])
    if first_payload is None:
        raise SystemExit("first retrieval artifact is missing")
    first_envelope = msgspec.msgpack.decode(first_payload, type=RetrievalEnvelope)
    if first_envelope.result.search_result.query != first.query:
        raise SystemExit("first retrieval artifact query does not match")
    print(
        f"preflight shard={shard_index}/{shard_count} topics={len(topic_ids)} "
        f"first_topic={topic_ids[0]} first_documents={len(first_envelope.result.documents)}",
        flush=True,
    )
    if arguments.preflight:
        return 0

    scorer_config = LocalCrossEncoderConfig(device="cuda")
    scorer = CacheOnlyPassageScorer(scorer_config.fingerprint)
    nugget_config = NuggetizerConfig(model=arguments.model)
    clusterer = NuggetClusterer()
    fingerprint = build_fingerprint(
        SimpleNamespace(fingerprint=backend_fingerprint(nugget_config)),
        clusterer,
        provider_base_url=provider_settings()[0],
    )
    cache = NuggetArtifactCache(store)
    outcomes: list[dict[str, object]] = []
    started = time.monotonic()
    for position, topic_id in enumerate(topic_ids, start=1):
        topic_started = time.monotonic()
        outcome: dict[str, object] = {"topic_id": topic_id}
        try:
            rows: list[StaticRerankingInput] = []
            for facet_index, snapshot_row in enumerate(snapshots[topic_id].retrievals):
                object_key = snapshot_row.object_keys[0]
                payload = (
                    first_payload
                    if topic_id == topic_ids[0] and facet_index == 0
                    else store.read(object_key)
                )
                if payload is None:
                    raise RuntimeError("retrieval artifact is missing")
                envelope = msgspec.msgpack.decode(payload, type=RetrievalEnvelope)
                retrieval = _from_wire(envelope.result)
                if retrieval.query != snapshot_row.query:
                    raise RuntimeError("retrieval artifact query does not match snapshot")
                chunker = PassageChunker(CrossEncoderPairLengthFunction(scorer_config, snapshot_row.query))
                score_cache = PassageScoreCache(
                    store,
                    scorer,
                    chunker_fingerprint=chunker.fingerprint,
                    representation=arguments.representation,
                )
                reranking = rerank_retrieval(
                    retrieval,
                    chunker=chunker,
                    scorer=CachedPassageScorer(score_cache),
                    representation=arguments.representation,
                )
                rows.append(
                    StaticRerankingInput(
                        facet_index=facet_index,
                        query=snapshot_row.query,
                        object_key=object_key,
                        request_digest=envelope.request_digest,
                        source_fingerprint=f"{chunker.fingerprint}|{score_cache.fingerprint}",
                        result=reranking,
                    )
                )
            work = build_static_topic_work(
                topic_id=topic_id,
                narrative=narratives[topic_id],
                representation=arguments.representation,
                rows=rows,
            )
            result = run_topic(
                work,
                cache=cache,
                fingerprint=fingerprint,
                backend_factory=lambda: NuggetizerBackend(nugget_config),
                clusterer=clusterer,
            )
            outcome.update(
                {
                    "status": result.status,
                    "object_key": result.object_key,
                    "nugget_count": result.nugget_count,
                    "evidence_count": result.evidence_count,
                }
            )
            if result.reason:
                outcome["reason"] = result.reason
        except Exception as error:  # one topic cannot strand its shard
            outcome.update(
                {
                    "status": "failed",
                    "reason": f"{type(error).__name__}: {' '.join(str(error).split())[:180]}",
                }
            )
        outcome["duration_seconds"] = round(time.monotonic() - topic_started, 3)
        outcomes.append(outcome)
        _write_manifest(
            arguments.manifest,
            {
                "schema_version": "static-plan-nugget-manifest-v1",
                "scoring_wave_plan_sha256": wave.get("plan_sha256"),
                "shard": shard_index,
                "shard_count": shard_count,
                "topic_count": len(topic_ids),
                "completed_topic_count": len(outcomes),
                "successful_topic_count": sum(row["status"] in {"cached", "written"} for row in outcomes),
                "failed_topic_count": sum(row["status"] == "failed" for row in outcomes),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "nugget_fingerprint": fingerprint.digest.hex(),
                "outcomes": outcomes,
            },
        )
        print(
            f"[{position}/{len(topic_ids)}] topic={topic_id} status={outcome['status']} "
            f"nuggets={outcome.get('nugget_count', 0)} elapsed={outcome['duration_seconds']}s",
            flush=True,
        )
    return 0 if all(row["status"] in {"cached", "written"} for row in outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
