"""Score a scoring plan's passages on this machine's GPU and cache the results.

Reads retrieval artifacts straight from the shared bucket by object key, so no
corpus is needed here. Writes complete per-query score artifacts back through
PassageScoreCache, which is the same cache a later local nugget run reads.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import msgspec


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--preflight", action="store_true",
                        help="validate wiring and exit without scoring")
    arguments = parser.parse_args()

    from trec_ragtime.artifact_store import ArtifactStore
    from trec_ragtime.passage_score_cache import CachedPassageScorer, PassageScoreCache
    from trec_ragtime.passage_scoring import (
        CrossEncoderPairLengthFunction,
        LocalCrossEncoderConfig,
        LocalCrossEncoderScorer,
    )
    from trec_ragtime.passages import PassageChunker
    from trec_ragtime.reranking import rerank_retrieval
    from trec_ragtime.retrieval_cache import RetrievalEnvelope

    plan = json.loads(arguments.plan.read_text())
    if plan.get("schema_version") != 1:
        raise SystemExit("unsupported scoring plan schema")
    representation = plan["representation"]
    queries = plan["queries"]

    store = ArtifactStore.from_env()
    config = LocalCrossEncoderConfig(device=arguments.device, batch_size=arguments.batch_size)
    scorer = LocalCrossEncoderScorer(config)
    print(f"device={config.effective_device} model={config.model_name}", flush=True)

    if arguments.preflight:
        first = queries[0]
        payload = store.read(first["object_key"])
        if payload is None:
            raise SystemExit(f"artifact missing: {first['object_key']}")
        envelope = msgspec.msgpack.decode(payload, type=RetrievalEnvelope)
        print(f"preflight ok: {len(envelope.result.documents)} documents readable "
              f"for topic {first['topic_id']}", flush=True)
        return 0

    from trec_ragtime.retrieval_cache import _from_wire

    done = failed = 0
    started = time.perf_counter()
    for index, row in enumerate(queries, start=1):
        try:
            payload = store.read(row["object_key"])
            if payload is None:
                failed += 1
                print(f"[{index}/{len(queries)}] MISSING {row['object_key']}", flush=True)
                continue
            envelope = msgspec.msgpack.decode(payload, type=RetrievalEnvelope)
            retrieval = _from_wire(envelope.result)
            chunker = PassageChunker(CrossEncoderPairLengthFunction(config, row["query"]))
            cache = PassageScoreCache(
                store, scorer,
                chunker_fingerprint=chunker.fingerprint,
                representation=representation,
            )
            rerank_retrieval(
                retrieval,
                chunker=chunker,
                scorer=CachedPassageScorer(cache),
                representation=representation,
            )
            done += 1
            rate = (time.perf_counter() - started) / index
            print(f"[{index}/{len(queries)}] scored topic {row['topic_id']} "
                  f"facet {row['facet_index']} ({rate:.1f}s/query avg)", flush=True)
        except Exception as error:  # noqa: BLE001 - one query must not stop the run
            failed += 1
            print(f"[{index}/{len(queries)}] FAILED {row['topic_id']}/"
                  f"{row['facet_index']}: {type(error).__name__}: "
                  f"{' '.join(str(error).split())[:160]}", flush=True)

    elapsed = time.perf_counter() - started
    print(f"scored={done} failed={failed} elapsed={elapsed/60:.1f}min", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
