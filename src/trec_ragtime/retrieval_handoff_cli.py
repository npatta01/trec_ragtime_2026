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
from .nugget_cache import (
    NuggetArtifactCache,
    NuggetEnvelope,
    NuggetInput,
    NuggetRequest,
    evidence_digest,
)
from .nuggets import NuggetBundle, NuggetClusterer
from .nuggets_cli import build_parser as build_nugget_parser
from .nuggets_cli import _head, plan_queries, select_plans
from .nuggets_runner import build_fingerprint
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
    parser.add_argument(
        "--topic-plan-sha256",
        action="append",
        default=[],
        metavar="TOPIC_ID=SHA256",
        help="override the retrieval plan digest for one refreshed topic (repeatable)",
    )
    return parser


def _topic_plan_digests(values: Sequence[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        if not isinstance(value, str) or value.count("=") != 1:
            raise ValueError("--topic-plan-sha256 must use TOPIC_ID=SHA256")
        topic_id, digest = value.split("=", 1)
        if (
            not topic_id
            or topic_id in overrides
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("--topic-plan-sha256 contains an invalid or duplicate value")
        overrides[topic_id] = digest
    return overrides


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


def _l1_nugget_candidates(
    store: ArtifactStore, fingerprint: bytes
) -> dict[str, tuple[NuggetBundle, ...]]:
    """Index validated candidate envelopes from the disposable local cache once."""

    found: dict[str, list[NuggetBundle]] = {}
    for path in Path(store.local_dir).iterdir():
        if not path.is_file():
            continue
        try:
            payload = path.read_bytes()
            envelope = msgspec.msgpack.decode(payload, type=NuggetEnvelope)
            if (
                envelope.fingerprint != fingerprint
                or msgspec.msgpack.encode(envelope) != payload
            ):
                continue
        except (OSError, msgspec.DecodeError, msgspec.ValidationError):
            continue
        found.setdefault(envelope.bundle.topic_id, []).append(envelope.bundle)
    return {topic_id: tuple(rows) for topic_id, rows in found.items()}


def _prepare_source_identity(
    plan,
    *,
    plan_sha256: str,
    retriever,
    scorer,
    scorer_config,
    representation: str,
    top_k: int,
    rerank_documents: int | None,
) -> tuple[tuple[NuggetInput, ...], str, dict[int, object]]:
    """Rebuild request lineage without chunking, scoring, or passage projection."""

    from .passage_scoring import CrossEncoderPairLengthFunction
    from .passages import PassageChunker
    from .query_baselines import PIPELINE, _query_context

    inputs: list[NuggetInput] = []
    sources: list[str] = []
    retrievals: dict[int, object] = {}
    for facet in plan_queries(plan):
        retrieval = retriever.get_cached_retrieval(
            facet.query,
            pipeline=PIPELINE,
            top_k=top_k,
            max_content_fallbacks=0,
            fallback_on_hf_error=False,
            query_context=_query_context(
                baseline=facet.baseline,
                plan_sha256=plan_sha256,
                query_index=facet.query_index,
            ),
        )
        if retrieval is None:
            continue
        if rerank_documents is not None:
            retrieval = _head(retrieval, rerank_documents)
        chunker = PassageChunker(
            CrossEncoderPairLengthFunction(scorer_config, facet.query)
        )
        reference = retrieval.retrieval_artifact
        inputs.append(
            NuggetInput(
                facet_index=facet.facet_index,
                query=facet.query,
                object_key=getattr(reference, "object_key", ""),
                request_digest=getattr(reference, "request_digest", b"\x00" * 32),
            )
        )
        sources.append(f"{chunker.fingerprint}|{scorer.fingerprint}")
        retrievals[facet.facet_index] = retrieval
    return tuple(inputs), "|".join((representation, *sources)), retrievals


def main(argv: Sequence[str] | None = None) -> int:
    from dotenv import load_dotenv

    arguments = build_parser().parse_args(argv)
    load_dotenv()
    try:
        plan_path = arguments.query_plan or default_query_plan_path()
        plans = select_plans(load_query_plans(plan_path), arguments.topic_ids)
        if not plans:
            raise ValueError("no topics selected")
        plan_digests = _topic_plan_digests(arguments.topic_plan_sha256)
        unknown_overrides = set(plan_digests) - {plan.topic_id for plan in plans}
        if unknown_overrides:
            raise ValueError("--topic-plan-sha256 names an unselected topic")
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
        candidates = _l1_nugget_candidates(store, nugget_fingerprint.digest)
    except Exception as error:  # noqa: BLE001 - CLI reports bounded setup failures
        print(f"{type(error).__name__}: configuration is invalid", file=sys.stderr)
        return 2

    work: list[HandoffWork] = []
    failures: dict[str, HandoffOutcome] = {}
    with ExitStack() as stack:
        retriever = stack.enter_context(RagtimeRetriever.from_env())
        for plan in plans:
            try:
                inputs, passage_source_fingerprint, retrievals = _prepare_source_identity(
                    plan,
                    plan_sha256=plan_digests.get(
                        plan.topic_id, arguments.plan_sha256
                    ),
                    retriever=retriever,
                    scorer=scorer,
                    scorer_config=scorer_config,
                    representation="report_text",
                    top_k=arguments.top_k,
                    rerank_documents=arguments.rerank_documents,
                )
                if not inputs:
                    raise ValueError("cached facet retrieval required")
                nugget_bundle = None
                nugget_request = None
                for candidate in candidates.get(plan.topic_id, ()):
                    request = NuggetRequest(
                        topic_id=plan.topic_id,
                        query=topics[plan.topic_id].problem_statement,
                        representation="report_text",
                        inputs=inputs,
                        passage_source_fingerprint=passage_source_fingerprint,
                        evidence_digest=evidence_digest(candidate.evidence),
                    )
                    loaded = nugget_cache.get(nugget_fingerprint, request)
                    if loaded is not None:
                        nugget_bundle, nugget_request = loaded, request
                        break
                if nugget_bundle is None or nugget_request is None:
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
                bundle = project_retrieval_handoff(
                    topic=topics[plan.topic_id],
                    nugget_bundle=nugget_bundle,
                    nugget_request=nugget_request,
                    nugget_source=source,
                    retrievals=retrievals,  # type: ignore[arg-type]
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
