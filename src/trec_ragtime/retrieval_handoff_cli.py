"""Build self-contained generation handoffs from existing cached artifacts only."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Literal

import msgspec

from .artifact_store import ArtifactStore, ArtifactValidationError
from .nugget_cache import (
    NUGGET_OBJECT_KEY_PREFIX,
    NUGGET_SCHEMA_VERSION,
    NuggetArtifactCache,
    NuggetEnvelope,
    NuggetInput,
    NuggetRequest,
    evidence_digest,
)
from .nuggets import NuggetBundle, NuggetClusterer, check_bundle
from .nuggets_cli import build_parser as build_nugget_parser
from .nuggets_cli import _head, plan_queries, select_plans
from .nuggets_runner import build_fingerprint
from .nuggetizer_adapter import NuggetizerConfig, backend_fingerprint
from .passage_score_cache import CacheOnlyPassageScorer
from .providers import provider_settings
from .query_baselines import default_query_plan_path, load_query_plans
from .retrieval_handoff import (
    HandoffNuggetSource,
    HandoffTopic,
    RetrievalHandoffBundle,
    project_retrieval_handoff,
    topic_record,
)
from .retrieval_handoff_cache import (
    OBJECT_ROOT,
    HandoffFingerprint,
    RetrievalHandoffArtifactCache,
)
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


PRODUCER_BEHAVIOR_VERSION = "retrieval-handoff-producer-v3"
EXPECTED_RETRIEVER_FINGERPRINT_SHA256 = (
    "487a18b1c3ebfdb4c967e6fea15c6a522d91f5bb002841d1ea03cb766c4ee1e7"
)


class _ProducerFacet(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    facet_index: int
    query: str
    baseline: Literal["b", "c"]
    query_index: int
    pair_length_fingerprint: str


class _ProducerIdentity(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    schema_version: Literal[1]
    behavior_version: str
    topic_id: str
    query_plan_digest: bytes
    retrieval_pipeline: str
    top_k: int
    rerank_documents: int | None
    max_content_fallbacks: int
    fallback_on_hf_error: bool
    representation: Literal["report_text"]
    retriever_fingerprint_digest: bytes
    scorer_fingerprint: str
    passage_chunker_runtime_fingerprint: str
    facets: tuple[_ProducerFacet, ...]
    nugget_fingerprint_digest: bytes

    @property
    def digest(self) -> bytes:
        return hashlib.sha256(msgspec.msgpack.encode(self)).digest()


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
    parser.add_argument(
        "--retriever-fingerprint-sha256",
        default=EXPECTED_RETRIEVER_FINGERPRINT_SHA256,
        help="expected canonical retrieval/corpus fingerprint",
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


def _sha256_digest(value: object, name: str) -> bytes:
    if (
        not isinstance(value, str)
        or len(value) != hashlib.sha256().digest_size * 2
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return bytes.fromhex(value)


def _producer_scorer_config(arguments):
    """Resolve result-affecting scorer policy without loading a model."""

    from .passage_scoring import LocalCrossEncoderConfig

    if arguments.scorer == "local":
        return LocalCrossEncoderConfig(device=arguments.device)
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
    return flash.effective_model_config


def _cache_only_scorer(arguments, *, scorer_config=None):
    config = scorer_config or _producer_scorer_config(arguments)
    fingerprint = getattr(config, "fingerprint", None)
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("scorer configuration has no fingerprint")
    return config, CacheOnlyPassageScorer(fingerprint)


def _producer_identity(
    plan,
    *,
    plan_sha256: str,
    arguments,
    scorer_config,
    nugget_fingerprint_digest: bytes,
) -> _ProducerIdentity:
    """Bind one handoff to the exact cold-path inputs, without upstream I/O."""

    from .passage_scoring import CrossEncoderPairLengthFunction
    from .passages import passage_chunker_runtime_fingerprint
    from .query_baselines import PIPELINE

    query_plan_digest = _sha256_digest(plan_sha256, "query plan digest")
    retriever_fingerprint_digest = _sha256_digest(
        arguments.retriever_fingerprint_sha256,
        "retriever fingerprint",
    )
    if (
        isinstance(arguments.top_k, bool)
        or not isinstance(arguments.top_k, int)
        or arguments.top_k < 1
    ):
        raise ValueError("--top-k must be a positive integer")
    if arguments.rerank_documents is not None and (
        isinstance(arguments.rerank_documents, bool)
        or not isinstance(arguments.rerank_documents, int)
        or arguments.rerank_documents < 1
    ):
        raise ValueError("--rerank-documents must be a positive integer")
    if (
        not isinstance(nugget_fingerprint_digest, bytes)
        or len(nugget_fingerprint_digest) != hashlib.sha256().digest_size
    ):
        raise ValueError("nugget fingerprint digest must be 32 bytes")
    scorer_fingerprint = getattr(scorer_config, "fingerprint", None)
    if not isinstance(scorer_fingerprint, str) or not scorer_fingerprint:
        raise ValueError("scorer configuration has no fingerprint")

    facets = tuple(
        _ProducerFacet(
            facet_index=facet.facet_index,
            query=facet.query,
            baseline=facet.baseline,
            query_index=facet.query_index,
            pair_length_fingerprint=CrossEncoderPairLengthFunction(
                scorer_config, facet.query
            ).fingerprint,
        )
        for facet in plan_queries(plan)
    )
    topic_id = getattr(plan, "topic_id", None)
    if not isinstance(topic_id, str) or not topic_id or not facets:
        raise ValueError("query plan must identify a topic and at least one facet")
    return _ProducerIdentity(
        schema_version=1,
        behavior_version=PRODUCER_BEHAVIOR_VERSION,
        topic_id=topic_id,
        query_plan_digest=query_plan_digest,
        retrieval_pipeline=PIPELINE,
        top_k=arguments.top_k,
        rerank_documents=arguments.rerank_documents,
        max_content_fallbacks=0,
        fallback_on_hf_error=False,
        representation="report_text",
        retriever_fingerprint_digest=retriever_fingerprint_digest,
        scorer_fingerprint=scorer_fingerprint,
        passage_chunker_runtime_fingerprint=passage_chunker_runtime_fingerprint(),
        facets=facets,
        nugget_fingerprint_digest=nugget_fingerprint_digest,
    )


def _nugget_candidates(
    store: ArtifactStore, fingerprint: bytes
) -> dict[str, tuple[NuggetBundle, ...]]:
    """Index newest-first sealed candidates from the durable nugget prefix."""

    fingerprint_hex = fingerprint.hex()
    prefix = f"{NUGGET_OBJECT_KEY_PREFIX}/{fingerprint_hex}"
    found: dict[str, list[NuggetEnvelope]] = {}
    for object_key in store.list_keys(prefix):
        parts = object_key.split("/")
        try:
            request_hex = parts[-1].removesuffix(".mpk")
            if (
                len(parts) != 5
                or "/".join(parts[:2]) != NUGGET_OBJECT_KEY_PREFIX
                or parts[2] != fingerprint_hex
                or not parts[-1].endswith(".mpk")
                or parts[3] != request_hex[:2]
                or len(request_hex) != 64
                or any(character not in "0123456789abcdef" for character in request_hex)
            ):
                raise ValueError("invalid nugget key")
            request_digest = bytes.fromhex(request_hex)

            def decode(payload: bytes) -> NuggetEnvelope:
                try:
                    envelope = msgspec.msgpack.decode(payload, type=NuggetEnvelope)
                    if (
                        envelope.schema_version != NUGGET_SCHEMA_VERSION
                        or envelope.fingerprint != fingerprint
                        or envelope.request_digest != request_digest
                        or not math.isfinite(envelope.stored_at)
                        or msgspec.msgpack.encode(envelope) != payload
                    ):
                        raise ValueError("invalid nugget envelope")
                    check_bundle(envelope.bundle)
                    return envelope
                except Exception:
                    raise ArtifactValidationError(
                        "cached nugget candidate is invalid"
                    ) from None

            envelope = store.read_validated_strict(object_key, decode)
            if envelope is None:
                continue
        except ArtifactValidationError:
            raise
        except Exception:
            raise ArtifactValidationError("nugget object key is invalid") from None
        found.setdefault(envelope.bundle.topic_id, []).append(envelope)
    return {
        topic_id: tuple(
            envelope.bundle
            for envelope in sorted(
                rows,
                key=lambda envelope: (envelope.stored_at, envelope.request_digest),
                reverse=True,
            )
        )
        for topic_id, rows in found.items()
    }


def _shared_tokenizer_loader():
    """Return a lazy loader that reuses one immutable tokenizer across facets."""

    tokenizer = None

    def load(config):
        nonlocal tokenizer
        if tokenizer is None:
            from .passage_scoring import _default_tokenizer_loader

            tokenizer = _default_tokenizer_loader(config)
        return tokenizer

    return load


def _discover_cached_handoffs(
    store: ArtifactStore,
    cache: RetrievalHandoffArtifactCache,
    fingerprint: HandoffFingerprint,
    *,
    topic_catalog_digest: bytes,
    expected_topics: Mapping[str, HandoffTopic],
    producer_identity_digests: Mapping[str, bytes],
) -> dict[str, tuple[RetrievalHandoffBundle, str]]:
    """Resolve complete handoffs directly, before opening any upstream stage."""

    wanted = set(expected_topics)
    if wanted != set(producer_identity_digests):
        raise ValueError("producer identities must exactly cover selected topics")
    found: dict[str, tuple[RetrievalHandoffBundle, str]] = {}
    prefix = f"{OBJECT_ROOT}/{fingerprint.digest.hex()}"
    for object_key in store.list_keys(prefix):
        bundle = cache.get_by_object_key(
            fingerprint,
            object_key,
            topic_catalog_digest=topic_catalog_digest,
            producer_identity_digests=producer_identity_digests,
        )
        if bundle is None or bundle.topic.topic_id not in wanted:
            continue
        if bundle.topic != expected_topics[bundle.topic.topic_id]:
            raise ArtifactValidationError(
                "cached handoff topic disagrees with current catalog"
            )
        if bundle.topic.topic_id in found:
            raise ValueError("multiple current handoffs exist for one topic")
        found[bundle.topic.topic_id] = (bundle, object_key)
    return found


def _cached_outcome(
    bundle: RetrievalHandoffBundle,
    object_key: str,
    producer_identity_digest: bytes,
) -> HandoffOutcome:
    return HandoffOutcome(
        topic_id=bundle.topic.topic_id,
        status="cached",
        object_key=object_key,
        nugget_count=bundle.counts.nuggets,
        evidence_count=bundle.counts.selected_evidence,
        full_support_count=bundle.counts.full_support,
        partial_support_count=bundle.counts.partial_support,
        citation_ready_count=bundle.counts.citation_ready,
        producer_identity_digest=producer_identity_digest.hex(),
    )


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
    expected_retriever_fingerprint: bytes,
    tokenizer_loader=None,
) -> tuple[tuple[NuggetInput, ...], str, dict[int, object]]:
    """Rebuild request lineage without chunking, scoring, or passage projection."""

    from .passage_scoring import CrossEncoderPairLengthFunction
    from .passages import PassageChunker
    from .query_baselines import PIPELINE, _query_context

    if (
        not isinstance(expected_retriever_fingerprint, bytes)
        or len(expected_retriever_fingerprint) != hashlib.sha256().digest_size
    ):
        raise ValueError("expected retriever fingerprint must be 32 bytes")
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
        reference = retrieval.retrieval_artifact
        if (
            reference is None
            or getattr(reference, "retriever_fingerprint", None)
            != expected_retriever_fingerprint
        ):
            raise ValueError("cached retrieval has an unexpected retriever fingerprint")
        chunker = PassageChunker(
            CrossEncoderPairLengthFunction(
                scorer_config,
                facet.query,
                tokenizer_loader=tokenizer_loader,
            )
        )
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
        scorer_config = _producer_scorer_config(arguments)
        expected_retriever_fingerprint = _sha256_digest(
            arguments.retriever_fingerprint_sha256,
            "retriever fingerprint",
        )
        producer_identity_digests = {
            plan.topic_id: _producer_identity(
                plan,
                plan_sha256=plan_digests.get(
                    plan.topic_id, arguments.plan_sha256
                ),
                arguments=arguments,
                scorer_config=scorer_config,
                nugget_fingerprint_digest=nugget_fingerprint.digest,
            ).digest
            for plan in plans
        }
        expected_topics = {
            plan.topic_id: topic_record(topics[plan.topic_id]) for plan in plans
        }
        store = ArtifactStore.from_env()
        handoff_cache = RetrievalHandoffArtifactCache(store)
        topic_catalog_digest = hashlib.sha256(Path(arguments.topics).read_bytes()).digest()
        handoff_fingerprint = HandoffFingerprint()
        cached_handoffs = _discover_cached_handoffs(
            store,
            handoff_cache,
            handoff_fingerprint,
            topic_catalog_digest=topic_catalog_digest,
            expected_topics=expected_topics,
            producer_identity_digests=producer_identity_digests,
        )
    except Exception as error:  # noqa: BLE001 - CLI reports bounded setup failures
        print(f"{type(error).__name__}: configuration is invalid", file=sys.stderr)
        return 2

    work: list[HandoffWork] = []
    failures: dict[str, HandoffOutcome] = {}
    produced = {
        topic_id: _cached_outcome(
            bundle, object_key, producer_identity_digests[topic_id]
        )
        for topic_id, (bundle, object_key) in cached_handoffs.items()
    }
    missing_plans = tuple(
        plan for plan in plans if plan.topic_id not in cached_handoffs
    )
    if missing_plans:
        try:
            scorer_config, scorer = _cache_only_scorer(
                arguments, scorer_config=scorer_config
            )
            nugget_cache = NuggetArtifactCache(store)
            candidates = _nugget_candidates(store, nugget_fingerprint.digest)
        except Exception as error:  # noqa: BLE001 - bounded cold-path setup
            for plan in missing_plans:
                failures[plan.topic_id] = HandoffOutcome(
                    plan.topic_id, "failed", reason=type(error).__name__
                )
            missing_plans = ()

    with ExitStack() as stack:
        retriever = None
        if missing_plans:
            try:
                retriever = stack.enter_context(RagtimeRetriever.from_env())
            except Exception as error:  # noqa: BLE001 - bounded cold setup
                for plan in missing_plans:
                    failures[plan.topic_id] = HandoffOutcome(
                        plan.topic_id, "failed", reason=type(error).__name__
                    )
                missing_plans = ()
        tokenizer_loader = _shared_tokenizer_loader() if missing_plans else None
        for plan in missing_plans:
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
                    expected_retriever_fingerprint=expected_retriever_fingerprint,
                    tokenizer_loader=tokenizer_loader,
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
                work.append(
                    HandoffWork(
                        topic_id=plan.topic_id,
                        topic_catalog_digest=topic_catalog_digest,
                        producer_identity_digest=producer_identity_digests[
                            plan.topic_id
                        ],
                        bundle=bundle,
                    )
                )
            except Exception as error:  # noqa: BLE001 - isolate one topic
                failures[plan.topic_id] = HandoffOutcome(
                    plan.topic_id, "failed", reason=type(error).__name__
                )

    produced.update(
        {
            row.topic_id: row
            for row in run_topics(
                work,
                fingerprint=handoff_fingerprint,
                cache=handoff_cache,
                concurrency=arguments.concurrency,
            )
        }
    )
    outcomes = tuple(
        failures.get(plan.topic_id) or produced[plan.topic_id] for plan in plans
    )
    if arguments.manifest is not None:
        write_handoff_manifest(
            arguments.manifest,
            outcomes,
            topic_catalog_digest=topic_catalog_digest,
            fingerprint=handoff_fingerprint,
        )
    print(json.dumps({status: sum(row.status == status for row in outcomes)
                      for status in ("written", "cached", "failed")}))
    return 0 if all(row.status != "failed" for row in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
