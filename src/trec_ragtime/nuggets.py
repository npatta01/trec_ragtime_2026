"""Canonical source-linked nuggets built from ranked passage evidence.

Upstream Nuggetizer returns claim text with no document identity, so linkage
comes from its assigner: every claim is judged against every evidence passage,
and the resulting support matrix is the lineage. A claim no passage supports is
quarantined rather than emitted.

``support`` and ``partial_support`` both retain a claim, and the distinction is
preserved per link rather than collapsed. Nothing here asserts entailment: a
consumer that needs fully attested claims filters on the label.

Each record is one ``msgspec.Struct`` used as both the domain and the wire type.
``Meta`` and ``Literal`` constraints are enforced on decode, so
``check_bundle`` gets alias formats, offset ranges, enum membership and unknown
field rejection from a single encode/decode round trip and only has to check
what types cannot express.

Passage text is deliberately absent from the bundle. ``content_hash`` and the
offsets prove which text was used, and the text itself is recoverable from the
retrieval artifact, so the artifact stays small and cannot drift from the
corpus.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Literal, Protocol

import msgspec


SCHEMA_VERSION = 1
BEHAVIOR_VERSION = "nuggets-v1"
DEFAULT_EVIDENCE_BUDGET = 24
DEFAULT_MAX_NUGGETS = 20
MAX_CLAIM_CHARACTERS = 400
MAX_CLAIM_WORDS = 30
DEFAULT_SIMILARITY_THRESHOLD = 0.92
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
GUARD_VERSION = "numeric-negation-guard-v1"
SUPPORT_LABELS = ("support", "partial_support")

Representation = Literal["canonical", "report_text"]
_Alias = Annotated[str, msgspec.Meta(pattern=r"^e[0-9]{3,}$")]
_Digest = Annotated[bytes, msgspec.Meta(min_length=32, max_length=32)]
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*%?")
_NEGATION = re.compile(r"\b(?:no|not|never|without|denied|failed)\b", re.IGNORECASE)
# Two sentences in one claim. Requiring a lowercase or digit before the
# terminator and a capital after the space keeps decimals ("fell 3.5%") and
# abbreviations ("U.S. exports fell") out of the quarantine, since losing a
# numeric claim is worse than keeping a rare two-clause one. Upstream emits
# English fragments of 1-12 words, so this is deliberately English-shaped.
_SENTENCE_BREAK = re.compile(r"[a-z0-9][.!?][\"'’”)\]]*\s+[A-Z]")
_IMPORTANCE_ORDER = {"vital": 0, "okay": 1}


class NuggetError(ValueError):
    """Raised when nugget evidence, claims, or bundle structure are invalid."""


class NuggetEvidence(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """One aliased passage in the bundle's complete evidence table."""

    alias: _Alias
    document_id: Annotated[str, msgspec.Meta(min_length=1)]
    representation: Representation
    language: Annotated[str, msgspec.Meta(min_length=1)]
    start_char: Annotated[int, msgspec.Meta(ge=0)]
    end_char: Annotated[int, msgspec.Meta(ge=1)]
    content_hash: _Digest
    passage_score: float
    document_rank: Annotated[int, msgspec.Meta(ge=1)]
    facet_index: Annotated[int, msgspec.Meta(ge=0)]


class NuggetSupport(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """One judged link from a nugget to an aliased passage.

    ``partial_support`` is retained, so a nugget may be attested only partially.
    That is model-indicated partial support, not entailment: a consumer that
    needs full support must filter on this label rather than assume it.
    """

    alias: _Alias
    assignment: Literal["support", "partial_support"]


class Nugget(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """One atomic claim with every passage that attests to it."""

    nugget_id: Annotated[str, msgspec.Meta(pattern=r"^nug1_[0-9a-f]{24}$")]
    claim: Annotated[str, msgspec.Meta(min_length=1, max_length=MAX_CLAIM_CHARACTERS)]
    importance: Literal["vital", "okay"]
    support: tuple[NuggetSupport, ...]
    cluster_id: Annotated[str, msgspec.Meta(pattern=r"^nugc1_[0-9a-f]{24}$")]
    variants: tuple[str, ...]


class FacetCoverage(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """How much of the sealed bundle each planned facet accounts for."""

    facet_index: Annotated[int, msgspec.Meta(ge=0)]
    query: Annotated[str, msgspec.Meta(min_length=1)]
    evidence_count: Annotated[int, msgspec.Meta(ge=0)]
    nugget_count: Annotated[int, msgspec.Meta(ge=0)]


class Rejection(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """A sanitized count of candidates that did not reach the bundle."""

    reason: Literal["unsupported", "not_atomic", "over_budget"]
    count: Annotated[int, msgspec.Meta(ge=1)]


class NuggetBundle(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """One topic's complete, validated nugget set."""

    schema_version: Literal[SCHEMA_VERSION]
    topic_id: Annotated[str, msgspec.Meta(min_length=1)]
    query: Annotated[str, msgspec.Meta(min_length=1)]
    representation: Representation
    evidence: tuple[NuggetEvidence, ...]
    nuggets: tuple[Nugget, ...]
    facets: tuple[FacetCoverage, ...]
    rejected: tuple[Rejection, ...]
    evidence_budget: Annotated[int, msgspec.Meta(ge=1)]
    max_nuggets: Annotated[int, msgspec.Meta(ge=1)]


@dataclass(frozen=True, slots=True)
class EvidenceCandidate:
    """One ranked passage offered to the evidence table, with its exact text.

    Offsets and ``content_hash`` are proven upstream by the reranking stage, so
    they are carried rather than re-derived.
    """

    document_id: str
    representation: str
    language: str
    start_char: int
    end_char: int
    content_hash: bytes
    passage_score: float
    document_rank: int
    text: str


@dataclass(frozen=True, slots=True)
class FacetEvidence:
    """One planned facet query and the ranked passages it contributed."""

    facet_index: int
    query: str
    candidates: tuple[EvidenceCandidate, ...]


@dataclass(frozen=True, slots=True)
class AliasedPassage:
    """A persisted evidence row paired with the text sent to the model."""

    evidence: NuggetEvidence
    text: str


class NuggetBackend(Protocol):
    """The nuggetization operations the stage may use."""

    def create(
        self, *, topic_id: str, query: str, passages: Sequence[tuple[str, str]]
    ) -> tuple[tuple[str, str], ...]: ...

    def assign(
        self, *, query: str, passage_text: str, nuggets: Sequence[tuple[str, str]]
    ) -> tuple[str, ...]: ...


Embedder = Callable[[Sequence[str]], Sequence[Sequence[float]]]


def normalize_claim(claim: str) -> str:
    """Collapse whitespace and case so paraphrase comparison is stable."""

    return " ".join(claim.split()).casefold()


def build_evidence_table(
    facets: Sequence[FacetEvidence], *, budget: int = DEFAULT_EVIDENCE_BUDGET
) -> tuple[AliasedPassage, ...]:
    """Fill the evidence budget by round-robin across facets, then alias it.

    Each facet keeps its own ranking; taking one passage per facet per pass
    stops a single well-served facet from consuming the whole budget.
    """

    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise NuggetError("evidence budget must be a positive integer")
    queues = [list(facet.candidates) for facet in facets]
    rows: list[AliasedPassage] = []
    seen: set[tuple[str, int, int]] = set()
    while len(rows) < budget and any(queues):
        for facet, queue in zip(facets, queues, strict=True):
            if len(rows) >= budget:
                break
            if not queue:
                continue
            candidate = queue.pop(0)
            location = (
                candidate.document_id,
                candidate.start_char,
                candidate.end_char,
            )
            if location in seen:
                continue
            seen.add(location)
            rows.append(
                AliasedPassage(
                    evidence=NuggetEvidence(
                        alias=f"e{len(rows) + 1:03d}",
                        document_id=candidate.document_id,
                        representation=candidate.representation,
                        language=candidate.language,
                        start_char=candidate.start_char,
                        end_char=candidate.end_char,
                        content_hash=candidate.content_hash,
                        passage_score=candidate.passage_score,
                        document_rank=candidate.document_rank,
                        facet_index=facet.facet_index,
                    ),
                    text=candidate.text,
                )
            )
    return tuple(rows)


def default_embedder() -> Embedder:
    """Return the pinned sentence embedder used for paraphrase clustering."""

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL, revision=EMBEDDING_REVISION)

    def embed(texts: Sequence[str]) -> Sequence[Sequence[float]]:
        vectors = model.encode(list(texts), normalize_embeddings=True)
        return [[float(value) for value in vector] for vector in vectors]

    return embed


class NuggetClusterer:
    """Groups paraphrases so every attesting source survives the merge."""

    def __init__(
        self,
        *,
        embed: Embedder | None = None,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        self._embed = embed
        self._threshold = threshold

    def cluster(self, claims: Sequence[str]) -> tuple[tuple[int, ...], ...]:
        """Return claim index groups, each ordered, leaders in input order."""

        groups: dict[str, list[int]] = {}
        for index, claim in enumerate(claims):
            groups.setdefault(normalize_claim(claim), []).append(index)
        keys = list(groups)
        if len(keys) < 2:
            return tuple(tuple(groups[key]) for key in keys)

        if self._embed is None:
            self._embed = default_embedder()
        vectors = self._embed(keys)
        if len(vectors) != len(keys):
            raise NuggetError("embedder returned the wrong number of vectors")
        guards = [_guard_signature(key) for key in keys]

        leaders: list[int] = []
        members: list[list[int]] = []
        for index in range(len(keys)):
            for position, leader in enumerate(leaders):
                # A numeric or polarity difference blocks the merge outright, so
                # "fell 30%" can never collapse into "fell 13%".
                if guards[index] != guards[leader]:
                    continue
                if _cosine(vectors[index], vectors[leader]) >= self._threshold:
                    members[position].append(index)
                    break
            else:
                leaders.append(index)
                members.append([index])
        return tuple(
            tuple(sorted(claim for key in group for claim in groups[keys[key]]))
            for group in members
        )


def run_topic_nuggets(
    *,
    topic_id: str,
    query: str,
    representation: str,
    facets: Sequence[FacetEvidence],
    backend: NuggetBackend,
    clusterer: NuggetClusterer,
    evidence_budget: int = DEFAULT_EVIDENCE_BUDGET,
    max_nuggets: int = DEFAULT_MAX_NUGGETS,
) -> NuggetBundle:
    """Produce one topic's validated bundle, or fail without writing anything."""

    rows = build_evidence_table(facets, budget=evidence_budget)
    if not rows:
        raise NuggetError("no ranked passages were supplied for this topic")

    rejected: Counter[str] = Counter()
    candidates = []
    for claim, importance in backend.create(
        topic_id=topic_id,
        query=query,
        passages=tuple((row.evidence.alias, row.text) for row in rows),
    ):
        if _is_atomic(claim):
            candidates.append((claim.strip(), importance))
        else:
            rejected["not_atomic"] += 1
    if not candidates:
        raise NuggetError("nuggetizer produced no atomic claims for this topic")

    support: list[dict[str, str]] = [{} for _ in candidates]
    for row in rows:
        labels = backend.assign(
            query=query, passage_text=row.text, nuggets=tuple(candidates)
        )
        if len(labels) != len(candidates):
            raise NuggetError("assignment did not cover every candidate claim")
        for position, label in enumerate(labels):
            if label in SUPPORT_LABELS:
                support[position][row.evidence.alias] = label

    kept = [index for index, links in enumerate(support) if links]
    rejected["unsupported"] += len(candidates) - len(kept)
    if not kept:
        raise NuggetError("no candidate claim was supported by known evidence")

    scores = {row.evidence.alias: row.evidence.passage_score for row in rows}
    nuggets = [
        _merge_cluster(topic_id, [kept[member] for member in group], candidates, support)
        for group in clusterer.cluster([candidates[index][0] for index in kept])
    ]
    nuggets.sort(key=lambda nugget: _order_key(nugget, scores))
    if len(nuggets) > max_nuggets:
        rejected["over_budget"] += len(nuggets) - max_nuggets
        nuggets = nuggets[:max_nuggets]

    bundle = NuggetBundle(
        schema_version=SCHEMA_VERSION,
        topic_id=topic_id,
        query=query,
        representation=representation,  # type: ignore[arg-type]
        evidence=tuple(row.evidence for row in rows),
        nuggets=tuple(nuggets),
        facets=_facet_coverage(facets, rows, nuggets),
        rejected=tuple(
            Rejection(reason=reason, count=count)  # type: ignore[arg-type]
            for reason, count in sorted(rejected.items())
            if count
        ),
        evidence_budget=evidence_budget,
        max_nuggets=max_nuggets,
    )
    check_bundle(bundle, document_ids={row.evidence.document_id for row in rows})
    return bundle


def check_bundle(bundle: NuggetBundle, *, document_ids: set[str]) -> None:
    """Fail closed on any bundle whose lineage or ordering is not provable."""

    try:
        decoded = msgspec.msgpack.decode(
            msgspec.msgpack.encode(bundle), type=NuggetBundle
        )
    except msgspec.ValidationError as error:
        raise NuggetError(f"nugget bundle failed schema validation: {error}") from None
    if decoded != bundle:
        raise NuggetError("nugget bundle is not canonically encodable")

    aliases: set[str] = set()
    for evidence in bundle.evidence:
        if evidence.alias in aliases:
            raise NuggetError("nugget evidence reuses an alias")
        aliases.add(evidence.alias)
        if evidence.end_char <= evidence.start_char:
            raise NuggetError("nugget evidence offsets do not span any text")
        if evidence.document_id not in document_ids:
            raise NuggetError("nugget evidence names an unknown document")
    if len(bundle.evidence) > bundle.evidence_budget:
        raise NuggetError("nugget evidence exceeds the configured budget")
    if len(bundle.nuggets) > bundle.max_nuggets:
        raise NuggetError("nugget count exceeds the configured maximum")

    scores = {evidence.alias: evidence.passage_score for evidence in bundle.evidence}
    identifiers: set[str] = set()
    for nugget in bundle.nuggets:
        if not nugget.support:
            raise NuggetError("nugget has no supporting evidence")
        linked = {link.alias for link in nugget.support}
        if len(linked) != len(nugget.support):
            raise NuggetError("nugget links one alias more than once")
        if not linked <= aliases:
            raise NuggetError("nugget links evidence outside the alias table")
        if nugget.nugget_id in identifiers:
            raise NuggetError("nugget bundle reuses a nugget id")
        identifiers.add(nugget.nugget_id)
        if nugget.nugget_id != _nugget_id(bundle.topic_id, nugget.claim, linked):
            raise NuggetError("nugget id does not match its claim and evidence")
        if nugget.cluster_id != _cluster_id(bundle.topic_id, nugget.variants):
            raise NuggetError("cluster id does not match its merged claims")
        if normalize_claim(nugget.claim) not in nugget.variants:
            raise NuggetError("nugget claim is missing from its own cluster")

    ordered = sorted(bundle.nuggets, key=lambda nugget: _order_key(nugget, scores))
    if list(bundle.nuggets) != ordered:
        raise NuggetError("nuggets are not in deterministic order")


def _merge_cluster(
    topic_id: str,
    members: Sequence[int],
    candidates: Sequence[tuple[str, str]],
    support: Sequence[Mapping[str, str]],
) -> Nugget:
    """Fold paraphrases into one nugget, keeping every attesting passage."""

    links: dict[str, str] = {}
    for member in members:
        for alias, assignment in support[member].items():
            # A full support judgement outranks a partial one for the same alias.
            if links.get(alias) != "support":
                links[alias] = assignment
    claim, _ = candidates[members[0]]
    importance = (
        "vital"
        if any(candidates[member][1] == "vital" for member in members)
        else "okay"
    )
    variants = tuple(sorted({normalize_claim(candidates[m][0]) for m in members}))
    return Nugget(
        nugget_id=_nugget_id(topic_id, claim, links),
        claim=claim,
        importance=importance,  # type: ignore[arg-type]
        support=tuple(
            NuggetSupport(alias=alias, assignment=links[alias])  # type: ignore[arg-type]
            for alias in sorted(links)
        ),
        cluster_id=_cluster_id(topic_id, variants),
        variants=variants,
    )


def _facet_coverage(
    facets: Sequence[FacetEvidence],
    rows: Sequence[AliasedPassage],
    nuggets: Sequence[Nugget],
) -> tuple[FacetCoverage, ...]:
    facet_by_alias = {row.evidence.alias: row.evidence.facet_index for row in rows}
    coverage: list[FacetCoverage] = []
    for facet in facets:
        aliases = {
            alias for alias, index in facet_by_alias.items() if index == facet.facet_index
        }
        coverage.append(
            FacetCoverage(
                facet_index=facet.facet_index,
                query=facet.query,
                evidence_count=len(aliases),
                nugget_count=sum(
                    1
                    for nugget in nuggets
                    if aliases & {link.alias for link in nugget.support}
                ),
            )
        )
    return tuple(coverage)


def _order_key(nugget: Nugget, scores: Mapping[str, float]) -> tuple[int, float, str]:
    best = max(scores.get(link.alias, 0.0) for link in nugget.support)
    return _IMPORTANCE_ORDER[nugget.importance], -best, nugget.nugget_id


def _nugget_id(topic_id: str, claim: str, aliases: Iterable[str]) -> str:
    """Identify a nugget by topic, normalized claim, and supporting aliases.

    The model, prompt, and provider are deliberately excluded so a re-run keeps
    the same identity for the same claim over the same evidence.
    """

    payload = msgspec.msgpack.encode(
        {
            "schema_version": SCHEMA_VERSION,
            "topic_id": topic_id,
            "claim": normalize_claim(claim),
            "aliases": sorted(aliases),
        }
    )
    return "nug1_" + hashlib.sha256(payload).hexdigest()[:24]


def _cluster_id(topic_id: str, variants: Sequence[str]) -> str:
    payload = msgspec.msgpack.encode(
        {
            "schema_version": SCHEMA_VERSION,
            "topic_id": topic_id,
            "variants": sorted(variants),
        }
    )
    return "nugc1_" + hashlib.sha256(payload).hexdigest()[:24]


def _is_atomic(claim: str) -> bool:
    stripped = claim.strip()
    return (
        bool(stripped)
        and len(stripped) <= MAX_CLAIM_CHARACTERS
        and len(stripped.split()) <= MAX_CLAIM_WORDS
        and not _SENTENCE_BREAK.search(stripped)
    )


def _guard_signature(claim: str) -> tuple[object, ...]:
    numbers = tuple(
        sorted(match.group().replace(",", "") for match in _NUMBER.finditer(claim))
    )
    return numbers, bool(_NEGATION.search(claim))


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    # Embeddings are unit-normalized, so the dot product is the cosine.
    return sum(a * b for a, b in zip(left, right, strict=True))


__all__ = [
    "BEHAVIOR_VERSION",
    "DEFAULT_EVIDENCE_BUDGET",
    "DEFAULT_MAX_NUGGETS",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "EMBEDDING_MODEL",
    "EMBEDDING_REVISION",
    "GUARD_VERSION",
    "SCHEMA_VERSION",
    "AliasedPassage",
    "EvidenceCandidate",
    "FacetCoverage",
    "FacetEvidence",
    "Nugget",
    "NuggetBackend",
    "NuggetBundle",
    "NuggetClusterer",
    "NuggetError",
    "NuggetEvidence",
    "NuggetSupport",
    "Rejection",
    "Representation",
    "build_evidence_table",
    "check_bundle",
    "default_embedder",
    "normalize_claim",
    "run_topic_nuggets",
]
