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


class _PhaseTimer:
    """Wrap the chunker and scorer to attribute wall clock to each phase.

    rerank_retrieval calls chunk() per document and score() once per query, so
    wrapping both is enough to split the time without touching that stage.
    """

    def __init__(self, chunker: object, scorer: object) -> None:
        self._chunker = chunker
        self._scorer = scorer
        self.chunk_seconds = 0.0
        self.score_seconds = 0.0

    @property
    def fingerprint(self) -> str:
        return self._scorer.fingerprint  # type: ignore[attr-defined]

    def chunk(self, **kwargs: object) -> object:
        start = time.perf_counter()
        try:
            return self._chunker.chunk(**kwargs)  # type: ignore[attr-defined]
        finally:
            self.chunk_seconds += time.perf_counter() - start

    def score(self, query: str, passages: object) -> object:
        start = time.perf_counter()
        try:
            return self._scorer.score(query, passages)  # type: ignore[attr-defined]
        finally:
            self.score_seconds += time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--plan-key", default=None,
                        help="object key holding the plan, for remote runs")
    parser.add_argument("--device", default="cuda")
    # Imported rather than repeated: this value feeds the scorer fingerprint,
    # so a literal here that drifts from the library default would key remote
    # scores under an identity no local run would ever look up.
    from trec_ragtime.passage_scoring import DEFAULT_BATCH_SIZE

    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--shard", default="1/1",
        help="process only this shard of the plan, as i/n. Several single-process "
             "runs split the work; threads cannot, because the artifact store's "
             "fsspec cache is not thread-safe (see the loop below).",
    )
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

    store = ArtifactStore.from_env()
    if arguments.plan_key:
        raw = store.read(arguments.plan_key)
        if raw is None:
            raise SystemExit(f"scoring plan not found at {arguments.plan_key}")
        plan = json.loads(raw)
    elif arguments.plan:
        plan = json.loads(arguments.plan.read_text())
    else:
        raise SystemExit("pass --plan or --plan-key")
    if plan.get("schema_version") != 1:
        raise SystemExit("unsupported scoring plan schema")
    representation = plan["representation"]
    queries = plan["queries"]

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

    try:
        shard_index, shard_count = (int(p) for p in arguments.shard.split("/"))
    except ValueError:
        raise SystemExit("--shard must look like i/n") from None
    if not 1 <= shard_index <= shard_count:
        raise SystemExit("--shard must satisfy 1 <= i <= n")

    done = failed = skipped = 0
    chunk_seconds = score_seconds = 0.0
    retry_seconds = 0.0
    started = time.perf_counter()

    def read_required(object_key: str, attempts: int = 6) -> bytes | None:
        """Read an artifact the plan guarantees exists, riding out rate limits.

        The bucket allows 1000 API requests per 5 minutes account-wide. Over
        that, HF answers 429 and fsspec surfaces it as FileNotFoundError, which
        the store reports as a quiet miss indistinguishable from a real absence.
        Every key here comes from the plan, so None means throttled, not absent;
        backing off also applies the backpressure the limit is asking for.
        """

        nonlocal retry_seconds
        for attempt in range(attempts):
            payload = store.read(object_key)
            if payload is not None:
                return payload
            if attempt + 1 == attempts:
                break
            delay = min(60.0, 5.0 * 2**attempt)
            retry_seconds += delay
            time.sleep(delay)
        return None

    def run_one(index: int, row: dict) -> None:
        nonlocal done, failed, skipped, chunk_seconds, score_seconds
        try:
            chunker = PassageChunker(
                CrossEncoderPairLengthFunction(config, row["query"])
            )
            cache = PassageScoreCache(
                store, scorer,
                chunker_fingerprint=chunker.fingerprint,
                representation=representation,
            )
            # A resumed run would otherwise re-download and re-chunk every finished
            # query just to discover its scores are cached. The score map is written
            # once, complete, after the query succeeds, so its presence means done.
            # A key that fails to match simply declines to skip and redoes the work.
            if store.read(cache.object_key(row["query"])) is not None:
                skipped += 1
                return
            payload = read_required(row["object_key"])
            if payload is None:
                failed += 1
                print(f"[{index}/{len(queries)}] MISSING {row['object_key']}", flush=True)
                return
            envelope = msgspec.msgpack.decode(payload, type=RetrievalEnvelope)
            retrieval = _from_wire(envelope.result)
            timed = _PhaseTimer(chunker, CachedPassageScorer(cache))
            rerank_retrieval(
                retrieval, chunker=timed, scorer=timed, representation=representation
            )
            done += 1
            chunk_seconds += timed.chunk_seconds
            score_seconds += timed.score_seconds
            rate = (time.perf_counter() - started) / max(done, 1)
            print(f"[{index}/{len(queries)}] scored topic {row['topic_id']} "
                  f"facet {row['facet_index']} ({rate:.1f}s/query avg)", flush=True)
        except Exception as error:  # noqa: BLE001 - one query must not stop the run
            failed += 1
            # Print the chained cause too: "invalid passage scores" alone does not
            # say whether keys are missing, extra, non-finite, or the map is stale.
            cause = error.__cause__ or error.__context__
            detail = f" <- {type(cause).__name__}: {' '.join(str(cause).split())[:120]}" if cause else ""
            print(f"[{index}/{len(queries)}] FAILED {row['topic_id']}/"
                  f"{row['facet_index']}: {type(error).__name__}: "
                  f"{' '.join(str(error).split())[:160]}{detail}", flush=True)

    # Deliberately sequential. Running queries in threads shares one ArtifactStore,
    # whose fsspec cache raised spurious FileNotFoundError under load; read() maps
    # that to a quiet miss, which is indistinguishable from a genuinely absent
    # artifact, so ~13% of queries were silently reported MISSING and never scored.
    # Parallelism therefore comes from several sharded processes, not threads.
    for index, row in enumerate(queries, start=1):
        if (index - 1) % shard_count == shard_index - 1:
            run_one(index, row)

    elapsed = time.perf_counter() - started
    print(f"scored={done} skipped={skipped} failed={failed} "
          f"elapsed={elapsed/60:.1f}min throttled={retry_seconds/60:.1f}min", flush=True)
    if done:
        other = elapsed - chunk_seconds - score_seconds
        print(
            f"phases: chunk={chunk_seconds/done:.1f}s/query "
            f"score={score_seconds/done:.1f}s/query "
            f"other={other/done:.1f}s/query (artifact io, decode)",
            flush=True,
        )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
