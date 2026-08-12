"""Build one sealed nugget bundle per topic from ranked passage evidence.

The reranking stage owns chunking and scoring, so this runner consumes its
results rather than re-deriving offsets. ``evidence_from_reranking`` reads those
records structurally, so this module stays importable and fully testable before
``trec_ragtime.reranking`` exists.

A warm topic is a cache hit at no provider cost, so there is no checkpoint file
here: the artifact cache is the resume mechanism.
"""

from __future__ import annotations

import dataclasses
import json
import os
import stat
import tempfile
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cache import base_url_digest

from .nugget_cache import (
    NuggetArtifactCache,
    NuggetFingerprint,
    NuggetInput,
    NuggetRequest,
    evidence_digest,
)
from .nuggets import (
    BEHAVIOR_VERSION,
    DEFAULT_EVIDENCE_BUDGET,
    DEFAULT_MAX_NUGGETS,
    MAX_CLAIM_CHARACTERS,
    MAX_CLAIM_WORDS,
    SCHEMA_VERSION,
    SUPPORT_LABELS,
    EvidenceCandidate,
    FacetEvidence,
    NuggetBackend,
    NuggetBundle,
    NuggetError,
    NuggetClusterer,
    build_evidence_table,
    run_topic_nuggets,
)


DEFAULT_CONCURRENCY = 4
_SCORED = "scored"


@dataclass(frozen=True, slots=True)
class TopicWork:
    """Everything one worker needs to own exactly one bundle."""

    topic_id: str
    query: str
    representation: str
    facets: tuple[FacetEvidence, ...]
    inputs: tuple[NuggetInput, ...]
    passage_source_fingerprint: str


@dataclass(frozen=True, slots=True)
class TopicOutcome:
    """A sanitized per-topic record safe to write into the manifest."""

    topic_id: str
    status: str
    object_key: str | None = None
    nugget_count: int = 0
    evidence_count: int = 0
    reason: str | None = None


def evidence_from_reranking(
    results: Sequence[Any],
    *,
    queries: Sequence[str],
    facet_indices: Sequence[int],
) -> tuple[FacetEvidence, ...]:
    """Project reranked passages onto facet evidence, preserving lineage.

    Offsets and content hashes were already proven by the reranking stage, so
    they are carried across rather than recomputed. Only scored documents carry
    passage evidence; every other status contributes nothing.

    ``facet_indices`` is explicit rather than positional. A topic whose facet
    retrievals are sparse would otherwise have its surviving facets renumbered,
    which silently changes facet coverage and the artifact's cache key.
    """

    if len(results) != len(queries) or len(results) != len(facet_indices):
        raise ValueError(
            "each reranking result needs its originating facet query and index"
        )
    if len(set(facet_indices)) != len(facet_indices):
        raise ValueError("facet indices must be unique")
    facets: list[FacetEvidence] = []
    for facet_index, result, query in zip(facet_indices, results, queries, strict=True):
        representation = result.representation
        candidates: list[EvidenceCandidate] = []
        for row in result.documents:
            if row.status != _SCORED:
                continue
            document = row.document
            selected = (
                document.report_text
                if representation == "report_text"
                else document.text
            )
            if not isinstance(selected, str):
                continue
            language = "eng" if representation == "report_text" else document.language
            candidates.extend(
                EvidenceCandidate(
                    document_id=passage.document_id,
                    representation=representation,
                    language=language,
                    start_char=passage.start_char,
                    end_char=passage.end_char,
                    content_hash=passage.content_hash,
                    passage_score=float(passage.score),
                    document_rank=row.reranked_rank,
                    # Shared reference; the slice happens once a passage is
                    # actually selected into the evidence table.
                    source_text=selected,
                )
                for passage in row.evidence
            )
        facets.append(
            FacetEvidence(
                facet_index=facet_index, query=query, candidates=tuple(candidates)
            )
        )
    return tuple(facets)


def build_fingerprint(
    backend: NuggetBackend,
    clusterer: NuggetClusterer,
    *,
    provider_base_url: str,
    evidence_budget: int = DEFAULT_EVIDENCE_BUDGET,
    max_nuggets: int = DEFAULT_MAX_NUGGETS,
) -> NuggetFingerprint:
    """Describe every result-affecting setting that is not topic specific.

    The backend and clusterer are asked for their identity rather than having
    their settings copied here, so the fingerprint cannot describe a
    configuration other than the one that will actually run. This is the same
    contract ``PassageScoreCache`` has with ``PassageScorer``.
    """

    return NuggetFingerprint(
        schema_version=SCHEMA_VERSION,
        behavior_version=BEHAVIOR_VERSION,
        backend=dict(backend.fingerprint),
        clusterer=dict(clusterer.fingerprint),
        provider_base_url_digest=base_url_digest(provider_base_url),
        # Read from the module that enforces them, so editing a constant changes
        # the fingerprint without anyone remembering to plumb it.
        claim_limits={
            "max_claim_characters": MAX_CLAIM_CHARACTERS,
            "max_claim_words": MAX_CLAIM_WORDS,
        },
        support_labels=tuple(SUPPORT_LABELS),
        evidence_budget=evidence_budget,
        max_nuggets=max_nuggets,
    )


def run_topic(
    work: TopicWork,
    *,
    cache: NuggetArtifactCache,
    fingerprint: NuggetFingerprint,
    backend_factory: Callable[[], NuggetBackend],
    clusterer: NuggetClusterer,
) -> TopicOutcome:
    """Return a cached bundle, or compute, verify, and store a fresh one."""

    rows = build_evidence_table(work.facets, budget=fingerprint.evidence_budget)
    if not rows:
        return TopicOutcome(work.topic_id, "failed", reason="no ranked passages")
    request = NuggetRequest(
        topic_id=work.topic_id,
        query=work.query,
        representation=work.representation,
        inputs=work.inputs,
        passage_source_fingerprint=work.passage_source_fingerprint,
        evidence_digest=evidence_digest(tuple(row.evidence for row in rows)),
    )
    key = cache.object_key(fingerprint, request)

    cached = cache.get(fingerprint, request)
    if cached is not None:
        return _outcome(work.topic_id, "cached", key, cached)

    # The store is inside the guard as well as the compute: cache.put revalidates
    # the bundle against the request, and a rejection there is this topic's
    # failure, not grounds for aborting every other worker in the pool.
    try:
        backend = backend_factory()
        # The fingerprint was built from one backend and execution gets another
        # from the factory, so nothing structurally stops a caller from keying a
        # bundle under a configuration it did not run. Comparing the identity the
        # backend reports against the one already in the key turns that into a
        # failed topic instead of a silently mislabelled artifact.
        if dict(backend.fingerprint) != fingerprint.backend:
            raise NuggetError(
                "backend identity differs from the fingerprint the bundle is "
                "keyed under; the cached artifact would describe a "
                "configuration that never ran"
            )
        bundle = run_topic_nuggets(
            topic_id=work.topic_id,
            query=work.query,
            representation=work.representation,
            facets=work.facets,
            backend=backend,
            clusterer=clusterer,
            evidence_budget=fingerprint.evidence_budget,
            max_nuggets=fingerprint.max_nuggets,
        )
        stored = cache.put(fingerprint, request, bundle)
    except Exception as error:  # noqa: BLE001 - one failed topic must not stop a run
        return TopicOutcome(work.topic_id, "failed", reason=_sanitize(error))
    if not stored:
        return TopicOutcome(work.topic_id, "failed", reason="artifact write refused")

    # Prove the durable copy independently: drop the local candidate and read
    # the typed object back through the store.
    cache.store.evict_local(key)
    verified = cache.get(fingerprint, request)
    if verified is None:
        return TopicOutcome(work.topic_id, "failed", reason="artifact reread failed")
    return _outcome(work.topic_id, "written", key, verified)


def run_topics(
    work: Sequence[TopicWork],
    *,
    cache: NuggetArtifactCache,
    fingerprint: NuggetFingerprint,
    backend_factory: Callable[[], NuggetBackend],
    clusterer: NuggetClusterer,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> tuple[TopicOutcome, ...]:
    """Process topics in parallel; one worker owns exactly one bundle."""

    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    if not work:
        return ()
    def one(item: TopicWork) -> TopicOutcome:
        return run_topic(
            item,
            cache=cache,
            fingerprint=fingerprint,
            backend_factory=backend_factory,
            clusterer=clusterer,
        )

    with ThreadPoolExecutor(max_workers=min(concurrency, len(work))) as pool:
        return tuple(pool.map(one, work))


def write_manifest(path: Path, outcomes: Sequence[TopicOutcome], **extra: Any) -> None:
    """Write an owner-only manifest of sanitized outcomes, atomically."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "behavior_version": BEHAVIOR_VERSION,
        "topics": [dataclasses.asdict(outcome) for outcome in outcomes],
        **extra,
    }
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".nuggets-")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, stat.S_IRUSR | stat.S_IWUSR)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _outcome(
    topic_id: str, status: str, key: str, bundle: NuggetBundle
) -> TopicOutcome:
    return TopicOutcome(
        topic_id=topic_id,
        status=status,
        object_key=key,
        nugget_count=len(bundle.nuggets),
        evidence_count=len(bundle.evidence),
    )


def _sanitize(error: BaseException) -> str:
    """Keep the failure legible without letting payloads into the manifest."""

    text = " ".join(str(error).split())[:200]
    return f"{type(error).__name__}: {text}" if text else type(error).__name__


__all__ = [
    "DEFAULT_CONCURRENCY",
    "TopicOutcome",
    "TopicWork",
    "build_fingerprint",
    "evidence_from_reranking",
    "run_topic",
    "run_topics",
    "write_manifest",
]
