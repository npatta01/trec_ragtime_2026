"""Build self-contained generation handoffs from existing cached artifacts only."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path

import msgspec

from .artifact_store import ArtifactStore
from .nugget_cache import NuggetArtifactCache
from .nuggets import NuggetClusterer
from .nuggets_cli import build_parser as build_nugget_parser
from .nuggets_cli import prepare_topic, select_plans
from .nuggets_runner import build_fingerprint, prepare_nugget_request
from .nuggetizer_adapter import NuggetizerConfig, backend_fingerprint
from .passage_score_cache import CacheOnlyPassageScorer
from .providers import provider_settings
from .query_baselines import default_query_plan_path, load_query_plans
from .retrieval_handoff import HandoffNuggetSource, project_retrieval_handoff
from .retrieval_handoff_cache import HandoffFingerprint, RetrievalHandoffArtifactCache
from .retrieval_handoff_runner import (
    HandoffOutcome,
    HandoffWork,
    run_topics,
    write_handoff_manifest,
)
from .retriever import RagtimeRetriever
from .topic_candidates import load_topics


class _IdentityBackend:
    def __init__(self, identity: Mapping[str, object]) -> None:
        self.fingerprint = identity


def build_parser():
    parser = build_nugget_parser()
    parser.prog = "ragtime-retrieval-handoff"
    parser.description = "Build cache-only retrieval bundles for report generation."
    return parser


def _cache_only_scorer(arguments):
    from .passage_scoring import LocalCrossEncoderConfig

    if arguments.scorer == "local":
        config = LocalCrossEncoderConfig(device=arguments.device)
        return config, CacheOnlyPassageScorer(config.fingerprint)
    if not arguments.flash_endpoint_id:
        raise ValueError("--scorer flash requires --flash-endpoint-id")
    if arguments.flash_workers < 1:
        raise ValueError("--flash-workers must be at least 1")
    from .flash_passage_scorer import RunpodFlashConfig

    flash = RunpodFlashConfig(
        endpoint_id=arguments.flash_endpoint_id,
        model_config=LocalCrossEncoderConfig(device="cuda"),
        workers=(0, arguments.flash_workers),
        concurrency=arguments.flash_concurrency or arguments.flash_workers,
    )
    return flash.effective_model_config, CacheOnlyPassageScorer(flash.fingerprint)


def main(argv: Sequence[str] | None = None) -> int:
    from dotenv import load_dotenv

    arguments = build_parser().parse_args(argv)
    load_dotenv()
    try:
        plan_path = arguments.query_plan or default_query_plan_path()
        plans = select_plans(load_query_plans(plan_path), arguments.topic_ids)
        if not plans:
            raise ValueError("no topics selected")
        topics = {row.topic_id: row for row in load_topics(arguments.topics)}
        missing = [row.topic_id for row in plans if row.topic_id not in topics]
        if missing:
            raise ValueError(f"topics file has no record for: {sorted(missing)}")
        if arguments.representation != "report_text":
            raise ValueError("retrieval handoff requires --representation report_text")
        scorer_config, scorer = _cache_only_scorer(arguments)
        nugget_config = NuggetizerConfig(
            model=arguments.model, max_nuggets=arguments.max_nuggets
        )
        clusterer = NuggetClusterer()
        nugget_fingerprint = build_fingerprint(
            _IdentityBackend(backend_fingerprint(nugget_config)),  # type: ignore[arg-type]
            clusterer,
            provider_base_url=provider_settings()[0],
            evidence_budget=arguments.evidence_budget,
            max_nuggets=arguments.max_nuggets,
        )
        store = ArtifactStore.from_env()
        nugget_cache = NuggetArtifactCache(store)
        handoff_cache = RetrievalHandoffArtifactCache(store)
        topic_catalog_digest = hashlib.sha256(Path(arguments.topics).read_bytes()).digest()
    except Exception as error:  # noqa: BLE001 - CLI reports bounded setup failures
        print(f"{type(error).__name__}: configuration is invalid", file=sys.stderr)
        return 2

    work: list[HandoffWork] = []
    failures: dict[str, HandoffOutcome] = {}
    with ExitStack() as stack:
        retriever = stack.enter_context(RagtimeRetriever.from_env())
        for plan in plans:
            try:
                prepared = prepare_topic(
                    plan,
                    narrative=topics[plan.topic_id].problem_statement,
                    plan_sha256=arguments.plan_sha256,
                    retriever=retriever,
                    store=store,
                    scorer=scorer,
                    scorer_config=scorer_config,
                    representation="report_text",
                    top_k=arguments.top_k,
                    rerank_documents=arguments.rerank_documents,
                )
                if prepared is None:
                    raise ValueError("cached facet retrieval required")
                _, nugget_request = prepare_nugget_request(
                    prepared, nugget_fingerprint
                )
                nugget_bundle = nugget_cache.get(nugget_fingerprint, nugget_request)
                if nugget_bundle is None:
                    raise ValueError("cached nugget bundle required")
                nugget_key = nugget_cache.object_key(
                    nugget_fingerprint, nugget_request
                )
                source = HandoffNuggetSource(
                    object_key=nugget_key,
                    fingerprint_digest=nugget_fingerprint.digest,
                    request_digest=nugget_request.digest,
                    bundle_digest=hashlib.sha256(
                        msgspec.msgpack.encode(nugget_bundle)
                    ).digest(),
                )
                retrievals = {
                    input_row.facet_index: retrieval
                    for input_row, retrieval in zip(
                        prepared.inputs, prepared.retrievals, strict=True
                    )
                }
                bundle = project_retrieval_handoff(
                    topic=topics[plan.topic_id],
                    nugget_bundle=nugget_bundle,
                    nugget_request=nugget_request,
                    nugget_source=source,
                    retrievals=retrievals,
                )
                work.append(HandoffWork(plan.topic_id, topic_catalog_digest, bundle))
            except Exception as error:  # noqa: BLE001 - isolate one topic
                failures[plan.topic_id] = HandoffOutcome(
                    plan.topic_id, "failed", reason=type(error).__name__
                )

    produced = {
        row.topic_id: row
        for row in run_topics(
            work,
            fingerprint=HandoffFingerprint(),
            cache=handoff_cache,
            concurrency=arguments.concurrency,
        )
    }
    outcomes = tuple(
        failures.get(plan.topic_id) or produced[plan.topic_id] for plan in plans
    )
    if arguments.manifest is not None:
        write_handoff_manifest(
            arguments.manifest,
            outcomes,
            topic_catalog_digest=topic_catalog_digest,
            fingerprint=HandoffFingerprint(),
        )
    print(json.dumps({status: sum(row.status == status for row in outcomes)
                      for status in ("written", "cached", "failed")}))
    return 0 if all(row.status != "failed" for row in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
