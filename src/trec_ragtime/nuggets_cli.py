"""``ragtime-nuggets``: build sealed nugget bundles for planned topics.

A thin shell over :mod:`trec_ragtime.nuggets_runner`, which stays the supported
Python API. Nothing result-affecting lives here; this module resolves inputs,
prepares passages through the reranking stage, and reports sanitized outcomes.

This is also where facet numbering is decided. ``facet_index`` is caller-supplied
data that no downstream check can independently verify, so the query plan's own
enumeration is carried through to both the projected facets and the recorded
inputs from a single loop -- a facet with no cached retrieval is skipped without
renumbering the ones around it.

Retrieval is read from the artifact cache only. A topic whose facets have not
been retrieved yet is reported, never turned into an organizer request.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifact_store import ArtifactStore
from .nugget_cache import NuggetArtifactCache, NuggetInput
from .nuggetizer_adapter import NuggetizerBackend, NuggetizerConfig
from .nuggets import DEFAULT_EVIDENCE_BUDGET, DEFAULT_MAX_NUGGETS, NuggetClusterer
from .providers import provider_settings
from .nuggets_runner import (
    DEFAULT_CONCURRENCY,
    TopicOutcome,
    TopicWork,
    build_fingerprint,
    evidence_from_reranking,
    run_topics,
    write_manifest,
)
from .query_baselines import (
    EXPECTED_QUERY_PLAN_SHA256,
    PIPELINE,
    QueryPlan,
    default_query_plan_path,
    load_query_plans,
)
from .query_baselines import _query_context as query_context
from .retriever import RagtimeRetriever
from .topic_candidates import load_topics


DEFAULT_TOP_K = 100


@dataclass(frozen=True, slots=True)
class PlannedFacet:
    """One planned query, and the exact identity its retrieval was written under.

    The baseline runner keys retrievals by a ``query_context`` carrying the
    baseline letter, the plan digest, and a **1-based** query index. Reproducing
    it exactly is the difference between a cache hit and silently reporting the
    topic as unretrieved.
    """

    facet_index: int
    query: str
    baseline: str
    query_index: int


def plan_queries(plan: QueryPlan) -> tuple[PlannedFacet, ...]:
    """Enumerate one topic's queries: facet 0 is the single baseline-B query."""

    return (
        PlannedFacet(facet_index=0, query=plan.single_query, baseline="b", query_index=1),
        *(
            PlannedFacet(facet_index=index, query=query, baseline="c", query_index=index)
            for index, query in enumerate(plan.facet_queries, start=1)
        ),
    )


def prepare_topic(
    plan: QueryPlan,
    *,
    narrative: str,
    plan_sha256: str,
    retriever: Any,
    store: Any,
    scorer: Any,
    scorer_config: Any,
    representation: str,
    top_k: int,
    rerank_documents: int | None = None,
) -> TopicWork | None:
    """Rerank each planned facet that has a cached retrieval, keeping its index.

    Returns ``None`` when no facet has one, so an unretrieved topic is reported
    rather than silently half-processed.
    """

    from .passage_score_cache import CachedPassageScorer, PassageScoreCache
    from .passage_scoring import CrossEncoderPairLengthFunction
    from .passages import PassageChunker
    from .reranking import rerank_retrieval

    indices: list[int] = []
    queries: list[str] = []
    results: list[Any] = []
    inputs: list[NuggetInput] = []
    sources: list[str] = []

    for facet in plan_queries(plan):
        facet_index, query = facet.facet_index, facet.query
        retrieval = retriever.get_cached_retrieval(
            query,
            pipeline=PIPELINE,
            top_k=top_k,
            # These three are part of the retrieval identity, so they have to
            # match how the baseline runner wrote the artifact.
            max_content_fallbacks=0,
            fallback_on_hf_error=False,
            query_context=query_context(
                baseline=facet.baseline,
                plan_sha256=plan_sha256,
                query_index=facet.query_index,
            ),
        )
        if retrieval is None:
            continue
        # Truncating here discards exactly the documents reranking exists to
        # rescue: the cross-encoder can only reorder what survives, so anything
        # the organizer underrated stays lost. Real runs rerank everything
        # retrieved; this is for bounded smoke tests only.
        if rerank_documents is not None:
            retrieval = _head(retrieval, rerank_documents)
        # The pair-length function is query-specific, so the chunker is too.
        chunker = PassageChunker(CrossEncoderPairLengthFunction(scorer_config, query))
        score_cache = PassageScoreCache(
            store,
            scorer,
            chunker_fingerprint=chunker.fingerprint,
            representation=representation,
        )
        results.append(
            rerank_retrieval(
                retrieval,
                chunker=chunker,
                scorer=CachedPassageScorer(score_cache),
                representation=representation,  # type: ignore[arg-type]
            )
        )
        # Both sides of the facet mapping are appended here, from the same
        # plan index, so they cannot drift apart.
        indices.append(facet_index)
        queries.append(query)
        reference = retrieval.retrieval_artifact
        inputs.append(
            NuggetInput(
                facet_index=facet_index,
                query=query,
                object_key=getattr(reference, "object_key", ""),
                request_digest=getattr(reference, "request_digest", b"\x00" * 32),
            )
        )
        sources.append(f"{chunker.fingerprint}|{score_cache.fingerprint}")

    if not results:
        return None
    return TopicWork(
        topic_id=plan.topic_id,
        # The nuggetizer judges importance and support against this, so it gets
        # the topic's stated information need, not the keyword query that was
        # tuned for the retriever.
        query=narrative,
        representation=representation,
        facets=evidence_from_reranking(
            results, queries=queries, facet_indices=indices
        ),
        inputs=tuple(inputs),
        passage_source_fingerprint="|".join((representation, *sources)),
    )


def _head(retrieval: Any, documents: int) -> Any:
    """Keep the highest-ranked documents, preserving the result's own order."""

    import dataclasses

    kept = tuple(sorted(retrieval.documents, key=lambda d: d.rank)[:documents])
    keep_ids = {document.document_id for document in kept}
    search = retrieval.search_result
    return dataclasses.replace(
        retrieval,
        documents=kept,
        search_result=dataclasses.replace(
            search,
            hits=tuple(hit for hit in search.hits if hit.document_id in keep_ids),
        ),
    )


def _open_scorer(arguments: argparse.Namespace, stack: ExitStack) -> tuple[Any, Any]:
    """Return ``(model_config, scorer)`` for the requested transport.

    Flash scores on a remote GPU; the chunker still needs the *same* model
    policy, so it comes from the Flash config rather than being rebuilt locally.
    """

    from .passage_scoring import LocalCrossEncoderConfig, LocalCrossEncoderScorer

    if arguments.scorer == "local":
        config = LocalCrossEncoderConfig(device=arguments.device)
        return config, LocalCrossEncoderScorer(config)

    if not arguments.flash_endpoint_id:
        raise ValueError("--scorer flash requires --flash-endpoint-id")
    from .flash_passage_scorer import (
        FlashEndpointBatchRunner,
        RunpodFlashConfig,
        RunpodFlashPassageScorer,
    )

    flash = RunpodFlashConfig(
        endpoint_id=arguments.flash_endpoint_id,
        model_config=LocalCrossEncoderConfig(device="cuda"),
    )
    runner = stack.enter_context(FlashEndpointBatchRunner(flash))
    scorer = stack.enter_context(RunpodFlashPassageScorer(flash, runner))
    return flash.effective_model_config, scorer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragtime-nuggets",
        description="Build sealed source-linked nugget bundles per topic.",
    )
    parser.add_argument("--model", required=True, help="provider model identifier")
    parser.add_argument(
        "--topic-id", action="append", dest="topic_ids", default=None,
        help="restrict the run to these topics (repeatable)",
    )
    parser.add_argument("--query-plan", type=Path, default=None)
    parser.add_argument(
        "--topics", type=Path, default=Path("topics/topics.all.2026.v0625-fix.jsonl"),
        help="topic records supplying each topic's stated information need",
    )
    parser.add_argument(
        "--plan-sha256", default=EXPECTED_QUERY_PLAN_SHA256,
        help="digest the retrievals were written under",
    )
    parser.add_argument(
        "--representation", choices=("canonical", "report_text"), default="report_text"
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--scorer", choices=("local", "flash"), default="local",
        help="cross-encoder transport. 'flash' scores on a Runpod GPU endpoint "
             "and needs --flash-endpoint-id.",
    )
    parser.add_argument(
        "--flash-endpoint-id", default=None,
        help="an existing Runpod endpoint running the passage-scoring worker",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="cross-encoder device: cpu, cuda, cuda:N, or rocm. "
             "The scorer fingerprint includes it, so changing it re-scores.",
    )
    parser.add_argument(
        "--rerank-documents", type=int, default=None,
        help="smoke-test escape hatch: rerank only this many top documents per "
             "facet. This truncates by ORGANIZER rank, so the cross-encoder can "
             "no longer promote a document the retriever underrated -- which is "
             "what reranking is for. Leave unset for real runs.",
    )
    parser.add_argument("--evidence-budget", type=int, default=DEFAULT_EVIDENCE_BUDGET)
    parser.add_argument("--max-nuggets", type=int, default=DEFAULT_MAX_NUGGETS)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser


def select_plans(
    plans: Sequence[QueryPlan], topic_ids: Sequence[str] | None
) -> tuple[QueryPlan, ...]:
    """Filter to the requested topics, rejecting any that the plan lacks."""

    if not topic_ids:
        return tuple(plans)
    wanted = set(topic_ids)
    selected = tuple(plan for plan in plans if plan.topic_id in wanted)
    missing = wanted - {plan.topic_id for plan in selected}
    if missing:
        raise ValueError(f"unknown topics: {sorted(missing)}")
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    from dotenv import load_dotenv

    arguments = build_parser().parse_args(argv)
    load_dotenv()

    try:
        if arguments.scorer == "flash" and not arguments.flash_endpoint_id:
            raise ValueError("--scorer flash requires --flash-endpoint-id")
        plans = select_plans(
            load_query_plans(arguments.query_plan or default_query_plan_path()),
            arguments.topic_ids,
        )
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    if not plans:
        print("no topics selected", file=sys.stderr)
        return 2
    # The nuggetizer is asked about the topic's information need, so the topic
    # records are required, not optional colour.
    narratives = {topic.topic_id: topic.problem_statement
                  for topic in load_topics(arguments.topics)}
    missing = [plan.topic_id for plan in plans if plan.topic_id not in narratives]
    if missing:
        print(f"topics file has no record for: {sorted(missing)}", file=sys.stderr)
        return 2

    config = NuggetizerConfig(model=arguments.model, max_nuggets=arguments.max_nuggets)
    clusterer = NuggetClusterer()
    fingerprint = build_fingerprint(
        NuggetizerBackend(config),
        clusterer,
        # The same resolution the backend will use, so the fingerprint cannot
        # name a provider other than the one that runs.
        provider_base_url=provider_settings()[0],
        evidence_budget=arguments.evidence_budget,
        max_nuggets=arguments.max_nuggets,
    )
    store = ArtifactStore.from_env()

    outcomes: list[TopicOutcome] = []
    work: list[TopicWork] = []
    with ExitStack() as stack:
        scorer_config, scorer = _open_scorer(arguments, stack)
        retriever = stack.enter_context(RagtimeRetriever.from_env())
        for plan in plans:
            prepared = prepare_topic(
                plan,
                narrative=narratives[plan.topic_id],
                plan_sha256=arguments.plan_sha256,
                retriever=retriever,
                store=store,
                scorer=scorer,
                scorer_config=scorer_config,
                representation=arguments.representation,
                top_k=arguments.top_k,
                rerank_documents=arguments.rerank_documents,
            )
            if prepared is None:
                outcomes.append(
                    TopicOutcome(
                        plan.topic_id, "failed", reason="no cached facet retrieval"
                    )
                )
            else:
                work.append(prepared)

    outcomes.extend(
        run_topics(
            work,
            cache=NuggetArtifactCache(store),
            fingerprint=fingerprint,
            backend_factory=lambda: NuggetizerBackend(config),
            clusterer=clusterer,
            concurrency=arguments.concurrency,
        )
    )
    outcomes.sort(key=lambda outcome: outcome.topic_id)
    if arguments.manifest is not None:
        write_manifest(arguments.manifest, outcomes, model=arguments.model)
    print(
        json.dumps(
            {
                status: sum(outcome.status == status for outcome in outcomes)
                for status in ("written", "cached", "failed")
            }
        )
    )
    return 0 if all(outcome.status != "failed" for outcome in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_TOP_K",
    "build_parser",
    "main",
    "plan_queries",
    "prepare_topic",
    "select_plans",
]
