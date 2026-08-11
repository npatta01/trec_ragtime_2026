"""Sealed evidence bundles: the only input a report generator may read.

A bundle pins exactly one topic request together with the allowlisted English
evidence the generator may cite, the facets that evidence covers, and the
provenance of every input that produced it. Its ``seal`` is a SHA-256 over a
canonical MessagePack encoding of all of that, recomputed and verified whenever
a bundle is constructed — so an unsealed or tampered bundle cannot exist as an
instance. Construction *is* the validation.

Evidence text is stored verbatim. ``document.text`` remains the canonical source
content and ``document.report_text`` is the English synthesis boundary
(AGENTS.md); nothing here mutates either. Compact handles ``S1..Sn`` stand in
for document IDs in prompts so the model can never emit a corpus identifier it
was not given, and ``resolve_handle`` maps them back deterministically.

Facets are the coverage unit. Today they come from the retrieval facet queries
of a ``query_baseline_v1`` run, or from the problem statement of a
``control_a_v1`` run. A real nuggetizer (issue #16) supplies
``EvidenceFacet(source="nugget_bundle_v1", importance="vital")`` and fills
``EvidenceBundleSource.nugget_bundle_sha256`` without any schema change here.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import msgspec

from .topic_candidates import Topic

EVIDENCE_BUNDLE_SCHEMA_VERSION = 1
EVIDENCE_BUNDLE_BUILDER_VERSION = "evidence-bundle-v1"

HANDLE_PREFIX = "S"
FACET_PREFIX = "F"

QUERY_BASELINE_SCHEMA = "query_baseline_v1"
CONTROL_A_SCHEMA = "control_a_v1"
SUPPORTED_RUN_SCHEMAS = frozenset({QUERY_BASELINE_SCHEMA, CONTROL_A_SCHEMA})

FACET_SOURCE_QUERY_BASELINE = "query_baseline_facet_v1"
FACET_SOURCE_CONTROL_A = "control_a_problem_v1"
FACET_SOURCE_TOPIC_REQUEST = "topic_request_v1"
FACET_IMPORTANCE = frozenset({"vital", "okay"})

_HANDLE_PATTERN = re.compile(r"S[1-9]\d*\Z")
_FACET_PATTERN = re.compile(r"F[1-9]\d*\Z")


class EvidenceBundleError(ValueError):
    """A bundle, or the run it was built from, violates the evidence contract."""


class UnknownEvidenceHandleError(EvidenceBundleError):
    """A handle is not in the bundle's allowlist."""


def sha256_file(path: Path | str) -> str:
    """Return the hex SHA-256 of a file, read in bounded chunks."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceBundleError(f"{name} must be a non-empty string")
    return value


def _require_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EvidenceBundleError(f"{name} must be a positive integer")
    return value


def _require_finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceBundleError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise EvidenceBundleError(f"{name} must be finite")
    return number


def _require_digest(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise EvidenceBundleError(f"{name} must be a hex SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class EvidenceFacet:
    """One coverage unit the report should address."""

    facet_id: str
    text: str
    source: str
    importance: str = "okay"

    def __post_init__(self) -> None:
        if not _FACET_PATTERN.fullmatch(self.facet_id or ""):
            raise EvidenceBundleError("facet_id must look like F1, F2, ...")
        _require_text(self.text, name="facet text")
        _require_text(self.source, name="facet source")
        if self.importance not in FACET_IMPORTANCE:
            raise EvidenceBundleError(f"facet importance must be one of {sorted(FACET_IMPORTANCE)}")


@dataclass(frozen=True, slots=True)
class EvidencePassage:
    """One allowlisted English passage the generator may cite."""

    handle: str
    document_id: str
    report_text: str
    text_sha256: str
    language: str
    url: str | None
    date: str | None
    rank: int
    retrieval_score: float
    facet_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not _HANDLE_PATTERN.fullmatch(self.handle or ""):
            raise EvidenceBundleError("handle must look like S1, S2, ...")
        _require_text(self.document_id, name="document_id")
        _require_text(self.report_text, name="report_text")
        _require_digest(self.text_sha256, name="text_sha256")
        _require_text(self.language, name="language")
        _require_positive_int(self.rank, name="rank")
        _require_finite(self.retrieval_score, name="retrieval_score")
        for value, name in ((self.url, "url"), (self.date, "date")):
            if value is not None and not isinstance(value, str):
                raise EvidenceBundleError(f"{name} must be a string or None")
        if not isinstance(self.facet_ids, tuple):
            raise EvidenceBundleError("facet_ids must be a tuple")
        for facet_id in self.facet_ids:
            if not _FACET_PATTERN.fullmatch(facet_id or ""):
                raise EvidenceBundleError("facet_ids must look like F1, F2, ...")
        if len(set(self.facet_ids)) != len(self.facet_ids):
            raise EvidenceBundleError("facet_ids must be unique")

    @classmethod
    def create(
        cls,
        *,
        handle: str,
        document_id: str,
        report_text: str,
        language: str,
        url: str | None,
        date: str | None,
        rank: int,
        retrieval_score: float,
        facet_ids: Sequence[str],
    ) -> "EvidencePassage":
        """Build a passage, hashing its text so tampering is detectable."""

        return cls(
            handle=handle,
            document_id=document_id,
            report_text=report_text,
            text_sha256=_text_digest(report_text),
            language=language,
            url=url,
            date=date,
            rank=rank,
            retrieval_score=float(retrieval_score),
            facet_ids=tuple(facet_ids),
        )

    def verify(self) -> None:
        """Raise when the stored digest no longer matches the stored text."""

        if self.text_sha256 != _text_digest(self.report_text):
            raise EvidenceBundleError(
                f"passage {self.handle} text does not match its recorded digest"
            )


@dataclass(frozen=True, slots=True)
class EvidenceBundleSource:
    """Provenance of every input that produced a bundle."""

    run_schema: str
    run_id: str
    run_dir_name: str
    manifest_sha256: str
    records_sha256: str
    topics_sha256: str
    corpus_dataset: str | None = None
    corpus_revision: str | None = None
    nugget_bundle_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.run_schema not in SUPPORTED_RUN_SCHEMAS:
            raise EvidenceBundleError(f"unsupported run schema {self.run_schema!r}")
        _require_text(self.run_id, name="run_id")
        _require_text(self.run_dir_name, name="run_dir_name")
        for value, name in (
            (self.manifest_sha256, "manifest_sha256"),
            (self.records_sha256, "records_sha256"),
            (self.topics_sha256, "topics_sha256"),
        ):
            _require_digest(value, name=name)
        if self.nugget_bundle_sha256 is not None:
            _require_digest(self.nugget_bundle_sha256, name="nugget_bundle_sha256")
        for value, name in (
            (self.corpus_dataset, "corpus_dataset"),
            (self.corpus_revision, "corpus_revision"),
        ):
            if value is not None and not isinstance(value, str):
                raise EvidenceBundleError(f"{name} must be a string or None")


class _FacetWire(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    facet_id: str
    text_utf8: bytes
    source: str
    importance: str


class _PassageWire(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    handle: str
    document_id_utf8: bytes
    report_text_utf8: bytes
    text_sha256: str
    language: str
    url: str | None
    date: str | None
    rank: int
    retrieval_score: float
    facet_ids: tuple[str, ...]


class _SourceWire(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    run_schema: str
    run_id_utf8: bytes
    run_dir_name_utf8: bytes
    manifest_sha256: str
    records_sha256: str
    topics_sha256: str
    corpus_dataset: str | None
    corpus_revision: str | None
    nugget_bundle_sha256: str | None


class _BundleWire(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    schema_version: int
    builder_version: str
    topic_id_utf8: bytes
    collection_id_utf8: bytes
    title_utf8: bytes | None
    problem_statement_utf8: bytes
    background_utf8: bytes
    character_limit: int
    max_passages: int
    evidence_chars: int
    facets: tuple[_FacetWire, ...]
    passages: tuple[_PassageWire, ...]
    source: _SourceWire


def _bundle_wire(
    *,
    schema_version: int,
    builder_version: str,
    topic_id: str,
    collection_id: str,
    title: str | None,
    problem_statement: str,
    background: str,
    character_limit: int,
    max_passages: int,
    evidence_chars: int,
    facets: Sequence[EvidenceFacet],
    passages: Sequence[EvidencePassage],
    source: EvidenceBundleSource,
) -> _BundleWire:
    return _BundleWire(
        schema_version=schema_version,
        builder_version=builder_version,
        topic_id_utf8=topic_id.encode("utf-8"),
        collection_id_utf8=collection_id.encode("utf-8"),
        title_utf8=None if title is None else title.encode("utf-8"),
        problem_statement_utf8=problem_statement.encode("utf-8"),
        background_utf8=background.encode("utf-8"),
        character_limit=character_limit,
        max_passages=max_passages,
        evidence_chars=evidence_chars,
        facets=tuple(
            _FacetWire(
                facet_id=facet.facet_id,
                text_utf8=facet.text.encode("utf-8"),
                source=facet.source,
                importance=facet.importance,
            )
            for facet in facets
        ),
        passages=tuple(
            _PassageWire(
                handle=passage.handle,
                document_id_utf8=passage.document_id.encode("utf-8"),
                report_text_utf8=passage.report_text.encode("utf-8"),
                text_sha256=passage.text_sha256,
                language=passage.language,
                url=passage.url,
                date=passage.date,
                rank=passage.rank,
                retrieval_score=passage.retrieval_score,
                facet_ids=passage.facet_ids,
            )
            for passage in passages
        ),
        source=_SourceWire(
            run_schema=source.run_schema,
            run_id_utf8=source.run_id.encode("utf-8"),
            run_dir_name_utf8=source.run_dir_name.encode("utf-8"),
            manifest_sha256=source.manifest_sha256,
            records_sha256=source.records_sha256,
            topics_sha256=source.topics_sha256,
            corpus_dataset=source.corpus_dataset,
            corpus_revision=source.corpus_revision,
            nugget_bundle_sha256=source.nugget_bundle_sha256,
        ),
    )


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    """One topic request sealed to the evidence a generator may cite."""

    schema_version: int
    builder_version: str
    topic_id: str
    collection_id: str
    title: str | None
    problem_statement: str
    background: str
    character_limit: int
    max_passages: int
    evidence_chars: int
    facets: tuple[EvidenceFacet, ...]
    passages: tuple[EvidencePassage, ...]
    source: EvidenceBundleSource
    seal: str

    def __post_init__(self) -> None:
        if self.schema_version != EVIDENCE_BUNDLE_SCHEMA_VERSION:
            raise EvidenceBundleError("unsupported evidence bundle schema version")
        _require_text(self.builder_version, name="builder_version")
        _require_text(self.topic_id, name="topic_id")
        _require_text(self.collection_id, name="collection_id")
        _require_text(self.problem_statement, name="problem_statement")
        _require_text(self.background, name="background")
        if self.title is not None and not isinstance(self.title, str):
            raise EvidenceBundleError("title must be a string or None")
        _require_positive_int(self.character_limit, name="character_limit")
        _require_positive_int(self.max_passages, name="max_passages")
        if (
            isinstance(self.evidence_chars, bool)
            or not isinstance(self.evidence_chars, int)
            or self.evidence_chars < 0
        ):
            raise EvidenceBundleError("evidence_chars must be a non-negative integer")
        if not isinstance(self.source, EvidenceBundleSource):
            raise EvidenceBundleError("source must be an EvidenceBundleSource")

        if not self.facets:
            raise EvidenceBundleError("a bundle must carry at least one facet")
        if not self.passages:
            raise EvidenceBundleError("a bundle must carry at least one passage")
        facet_ids = [facet.facet_id for facet in self.facets]
        if len(set(facet_ids)) != len(facet_ids):
            raise EvidenceBundleError("facet identifiers must be unique")

        handles = [passage.handle for passage in self.passages]
        if len(set(handles)) != len(handles):
            raise EvidenceBundleError("evidence handles must be unique")
        document_ids = [passage.document_id for passage in self.passages]
        if len(set(document_ids)) != len(document_ids):
            raise EvidenceBundleError("evidence document identifiers must be unique")
        if len(self.passages) > self.max_passages:
            raise EvidenceBundleError("a bundle carries more passages than max_passages")

        known = set(facet_ids)
        for passage in self.passages:
            passage.verify()
            unknown = set(passage.facet_ids) - known
            if unknown:
                raise EvidenceBundleError(
                    f"passage {passage.handle} references unknown facets {sorted(unknown)}"
                )

        expected = self._compute_seal()
        if self.seal != expected:
            raise EvidenceBundleError("evidence bundle seal does not match its contents")

    def _compute_seal(self) -> str:
        wire = _bundle_wire(
            schema_version=self.schema_version,
            builder_version=self.builder_version,
            topic_id=self.topic_id,
            collection_id=self.collection_id,
            title=self.title,
            problem_statement=self.problem_statement,
            background=self.background,
            character_limit=self.character_limit,
            max_passages=self.max_passages,
            evidence_chars=self.evidence_chars,
            facets=self.facets,
            passages=self.passages,
            source=self.source,
        )
        return hashlib.sha256(msgspec.msgpack.encode(wire)).hexdigest()

    @classmethod
    def create(
        cls,
        *,
        topic_id: str,
        collection_id: str,
        title: str | None,
        problem_statement: str,
        background: str,
        character_limit: int,
        max_passages: int,
        evidence_chars: int,
        facets: Sequence[EvidenceFacet],
        passages: Sequence[EvidencePassage],
        source: EvidenceBundleSource,
    ) -> "EvidenceBundle":
        """Build and seal a bundle in one step."""

        facets = tuple(facets)
        passages = tuple(passages)
        seal = hashlib.sha256(
            msgspec.msgpack.encode(
                _bundle_wire(
                    schema_version=EVIDENCE_BUNDLE_SCHEMA_VERSION,
                    builder_version=EVIDENCE_BUNDLE_BUILDER_VERSION,
                    topic_id=topic_id,
                    collection_id=collection_id,
                    title=title,
                    problem_statement=problem_statement,
                    background=background,
                    character_limit=character_limit,
                    max_passages=max_passages,
                    evidence_chars=evidence_chars,
                    facets=facets,
                    passages=passages,
                    source=source,
                )
            )
        ).hexdigest()
        return cls(
            schema_version=EVIDENCE_BUNDLE_SCHEMA_VERSION,
            builder_version=EVIDENCE_BUNDLE_BUILDER_VERSION,
            topic_id=topic_id,
            collection_id=collection_id,
            title=title,
            problem_statement=problem_statement,
            background=background,
            character_limit=character_limit,
            max_passages=max_passages,
            evidence_chars=evidence_chars,
            facets=facets,
            passages=passages,
            source=source,
            seal=seal,
        )

    @property
    def seal_bytes(self) -> bytes:
        return bytes.fromhex(self.seal)

    @property
    def handle_allowlist(self) -> frozenset[str]:
        return frozenset(passage.handle for passage in self.passages)

    @property
    def handles(self) -> tuple[str, ...]:
        return tuple(passage.handle for passage in self.passages)

    @property
    def document_ids(self) -> tuple[str, ...]:
        return tuple(passage.document_id for passage in self.passages)

    @property
    def facet_ids(self) -> tuple[str, ...]:
        return tuple(facet.facet_id for facet in self.facets)

    def passage(self, handle: Any) -> EvidencePassage:
        """Return the passage for ``handle`` or fail closed."""

        if not isinstance(handle, str):
            raise UnknownEvidenceHandleError("evidence handle must be a string")
        for passage in self.passages:
            if passage.handle == handle:
                return passage
        raise UnknownEvidenceHandleError(f"evidence handle {handle!r} is not allowlisted")

    def resolve_handle(self, handle: Any) -> str:
        """Map a compact handle back to its canonical corpus document ID."""

        return self.passage(handle).document_id

    def prompt_excerpt(self, handle: str) -> str:
        """Return the prompt-facing view of a passage, cut at a word boundary."""

        text = self.passage(handle).report_text
        limit = self.evidence_chars
        if limit <= 0 or len(text) <= limit:
            return text
        head = text[:limit]
        boundary = head.rfind(" ")
        if boundary >= limit // 2:
            head = head[:boundary]
        return head.rstrip() + "…"


def topic_digest(topic: Topic) -> bytes:
    """Return the 32-byte digest of one exact topic request."""

    payload = {
        "schema_version": EVIDENCE_BUNDLE_SCHEMA_VERSION,
        "topic_id_utf8": topic.topic_id.encode("utf-8"),
        "collection_id_utf8": topic.collection_id.encode("utf-8"),
        "title_utf8": None if topic.title is None else topic.title.encode("utf-8"),
        "problem_statement_utf8": topic.problem_statement.encode("utf-8"),
        "background_utf8": topic.background.encode("utf-8"),
        "limit": topic.limit,
    }
    return hashlib.sha256(msgspec.msgpack.encode(payload)).digest()


@dataclass(frozen=True, slots=True)
class EvidenceCandidate:
    """One usable candidate read out of a retrieval run."""

    document_id: str
    report_text: str
    language: str
    url: str | None
    date: str | None
    rank: int
    retrieval_score: float
    facet_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvidenceRunTopic:
    """The facets and usable candidates a run recorded for one topic."""

    topic_id: str
    facets: tuple[EvidenceFacet, ...]
    candidates: tuple[EvidenceCandidate, ...]


@dataclass(frozen=True, slots=True)
class EvidenceRun:
    """A validated projection of one retrieval run directory."""

    run_schema: str
    run_id: str
    run_dir_name: str
    manifest_sha256: str
    records_sha256: str
    corpus_dataset: str | None
    corpus_revision: str | None
    topics: Mapping[str, EvidenceRunTopic]
    rejected_candidates: int


class _RecordStream:
    """Streams JSONL records while digesting the exact bytes that were read."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._digest = hashlib.sha256()

    def __iter__(self) -> Iterator[dict[str, Any]]:
        with self._path.open("rb") as handle:
            for number, raw in enumerate(handle, start=1):
                self._digest.update(raw)
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as error:
                    raise EvidenceBundleError(
                        f"{self._path.name} line {number} is not valid JSON: {error}"
                    ) from None
                if not isinstance(record, dict):
                    raise EvidenceBundleError(
                        f"{self._path.name} line {number} must be a JSON object"
                    )
                yield record

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()


def _optional_text(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _candidate_score(record: Mapping[str, Any]) -> float | None:
    for field in ("fused_score", "score"):
        value = record.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if math.isfinite(number):
            return number
    return None


def _usable_candidate(
    record: Mapping[str, Any], *, facet_ids: tuple[str, ...]
) -> EvidenceCandidate | None:
    """Return a candidate, or ``None`` when it fails the evidence contract."""

    document_id = _optional_text(record.get("document_id"))
    report_text = _optional_text(record.get("report_text"))
    language = _optional_text(record.get("language"))
    rank = record.get("rank")
    if document_id is None or report_text is None or language is None:
        return None
    if record.get("hydrated") is not True:
        return None
    if record.get("missing_translation") is True:
        return None
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        return None
    score = _candidate_score(record)
    if score is None:
        return None
    return EvidenceCandidate(
        document_id=document_id,
        report_text=report_text,
        language=language,
        url=_optional_text(record.get("url")),
        date=_optional_text(record.get("date")),
        rank=rank,
        retrieval_score=score,
        facet_ids=facet_ids,
    )


def _resolve_facet_ids(record: Mapping[str, Any], facet_count: int) -> tuple[str, ...]:
    """Map 1-based ``source_query_indices`` onto ``F1..Fn`` facet identifiers."""

    indices = record.get("source_query_indices")
    if not isinstance(indices, list):
        return ()
    resolved: list[str] = []
    for value in indices:
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        if 1 <= value <= facet_count:
            facet_id = f"{FACET_PREFIX}{value}"
            if facet_id not in resolved:
                resolved.append(facet_id)
    return tuple(resolved)


def _collect_candidates(
    records: Sequence[Any], *, facet_count: int, control_a: bool
) -> tuple[tuple[EvidenceCandidate, ...], int]:
    kept: list[EvidenceCandidate] = []
    seen: set[str] = set()
    rejected = 0
    for record in records:
        if not isinstance(record, dict):
            rejected += 1
            continue
        facet_ids = (
            (f"{FACET_PREFIX}1",) if control_a else _resolve_facet_ids(record, facet_count)
        )
        candidate = _usable_candidate(record, facet_ids=facet_ids)
        if candidate is None or candidate.document_id in seen:
            rejected += 1
            continue
        seen.add(candidate.document_id)
        kept.append(candidate)
    kept.sort(key=lambda item: (item.rank, item.document_id))
    return tuple(kept), rejected


def _read_query_baseline(
    stream: _RecordStream,
) -> tuple[dict[str, EvidenceRunTopic], int]:
    topics: dict[str, EvidenceRunTopic] = {}
    rejected = 0
    for record in stream:
        topic_id = _optional_text(record.get("topic_id"))
        if topic_id is None:
            raise EvidenceBundleError("every topic record needs a topic_id")
        if record.get("status") != "success":
            continue
        queries = record.get("queries")
        queries = queries if isinstance(queries, list) else []
        facets = tuple(
            EvidenceFacet(
                facet_id=f"{FACET_PREFIX}{position}",
                text=query,
                source=FACET_SOURCE_QUERY_BASELINE,
            )
            for position, query in enumerate(queries, start=1)
            if isinstance(query, str) and query.strip()
        )
        candidates = record.get("candidates")
        candidates = candidates if isinstance(candidates, list) else []
        kept, dropped = _collect_candidates(
            candidates, facet_count=len(facets), control_a=False
        )
        rejected += dropped
        if not kept:
            continue
        topics[topic_id] = EvidenceRunTopic(topic_id, facets, kept)
    return topics, rejected


def _read_control_a(stream: _RecordStream) -> tuple[dict[str, EvidenceRunTopic], int]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    statements: dict[str, str] = {}
    for record in stream:
        topic_id = _optional_text(record.get("topic_id"))
        if topic_id is None:
            raise EvidenceBundleError("every candidate record needs a topic_id")
        grouped.setdefault(topic_id, []).append(record)
        query = _optional_text(record.get("query"))
        if query is not None:
            statements.setdefault(topic_id, query)

    topics: dict[str, EvidenceRunTopic] = {}
    rejected = 0
    for topic_id, records in grouped.items():
        kept, dropped = _collect_candidates(records, facet_count=1, control_a=True)
        rejected += dropped
        if not kept:
            continue
        facets = (
            EvidenceFacet(
                facet_id=f"{FACET_PREFIX}1",
                text=statements.get(topic_id, topic_id),
                source=FACET_SOURCE_CONTROL_A,
            ),
        )
        topics[topic_id] = EvidenceRunTopic(topic_id, facets, kept)
    return topics, rejected


def load_evidence_run(run_dir: Path | str) -> EvidenceRun:
    """Read a ``query_baseline_v1`` or ``control_a_v1`` run into one shape."""

    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise EvidenceBundleError(f"no manifest.json in {run_dir}")
    raw = manifest_path.read_bytes()
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as error:
        raise EvidenceBundleError(f"manifest.json is not valid JSON: {error}") from None
    if not isinstance(manifest, dict):
        raise EvidenceBundleError("manifest.json must be a JSON object")

    schema = manifest.get("schema_version")
    if schema not in SUPPORTED_RUN_SCHEMAS:
        raise EvidenceBundleError(
            f"unsupported run schema {schema!r};"
            f" expected one of {sorted(SUPPORTED_RUN_SCHEMAS)}"
        )

    records_name = "topics.jsonl" if schema == QUERY_BASELINE_SCHEMA else "candidates.jsonl"
    records_path = run_dir / records_name
    if not records_path.is_file():
        raise EvidenceBundleError(f"no {records_name} in {run_dir}")

    stream = _RecordStream(records_path)
    if schema == QUERY_BASELINE_SCHEMA:
        topics, rejected = _read_query_baseline(stream)
    else:
        topics, rejected = _read_control_a(stream)

    corpus = manifest.get("corpus")
    corpus = corpus if isinstance(corpus, dict) else {}
    return EvidenceRun(
        run_schema=schema,
        run_id=_optional_text(manifest.get("run_id")) or run_dir.name,
        run_dir_name=run_dir.name,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        records_sha256=stream.sha256,
        corpus_dataset=_optional_text(corpus.get("dataset")),
        corpus_revision=_optional_text(corpus.get("revision")),
        topics=topics,
        rejected_candidates=rejected,
    )


def build_evidence_bundles(
    run: EvidenceRun,
    *,
    topics: Sequence[Topic],
    topics_sha256: str,
    max_passages: int,
    evidence_chars: int,
) -> tuple[EvidenceBundle, ...]:
    """Seal one bundle per requested topic, in the topic file's order."""

    _require_positive_int(max_passages, name="max_passages")
    if isinstance(evidence_chars, bool) or not isinstance(evidence_chars, int) or evidence_chars < 0:
        raise EvidenceBundleError("evidence_chars must be a non-negative integer")
    _require_digest(topics_sha256, name="topics_sha256")

    source = EvidenceBundleSource(
        run_schema=run.run_schema,
        run_id=run.run_id,
        run_dir_name=run.run_dir_name,
        manifest_sha256=run.manifest_sha256,
        records_sha256=run.records_sha256,
        topics_sha256=topics_sha256,
        corpus_dataset=run.corpus_dataset,
        corpus_revision=run.corpus_revision,
        nugget_bundle_sha256=None,
    )

    bundles: list[EvidenceBundle] = []
    for topic in topics:
        run_topic = run.topics.get(topic.topic_id)
        if run_topic is None or not run_topic.candidates:
            raise EvidenceBundleError(
                f"evidence run {run.run_id} has no usable candidates for topic {topic.topic_id}"
            )
        kept = run_topic.candidates[:max_passages]
        passages = tuple(
            EvidencePassage.create(
                handle=f"{HANDLE_PREFIX}{position}",
                document_id=candidate.document_id,
                report_text=candidate.report_text,
                language=candidate.language,
                url=candidate.url,
                date=candidate.date,
                rank=candidate.rank,
                retrieval_score=candidate.retrieval_score,
                facet_ids=candidate.facet_ids,
            )
            for position, candidate in enumerate(kept, start=1)
        )

        used = {facet_id for passage in passages for facet_id in passage.facet_ids}
        facets = tuple(facet for facet in run_topic.facets if facet.facet_id in used)
        if not facets:
            # No kept passage carried facet attribution; fall back to the request
            # itself so coverage still has exactly one unit to budget against.
            facets = (
                EvidenceFacet(
                    facet_id=f"{FACET_PREFIX}1",
                    text=topic.problem_statement,
                    source=FACET_SOURCE_TOPIC_REQUEST,
                ),
            )
            passages = tuple(
                replace(passage, facet_ids=(f"{FACET_PREFIX}1",)) for passage in passages
            )

        bundles.append(
            EvidenceBundle.create(
                topic_id=topic.topic_id,
                collection_id=topic.collection_id,
                title=topic.title,
                problem_statement=topic.problem_statement,
                background=topic.background,
                character_limit=topic.limit,
                max_passages=max_passages,
                evidence_chars=evidence_chars,
                facets=facets,
                passages=passages,
                source=source,
            )
        )
    return tuple(bundles)


__all__ = [
    "CONTROL_A_SCHEMA",
    "EVIDENCE_BUNDLE_BUILDER_VERSION",
    "EVIDENCE_BUNDLE_SCHEMA_VERSION",
    "FACET_PREFIX",
    "HANDLE_PREFIX",
    "QUERY_BASELINE_SCHEMA",
    "SUPPORTED_RUN_SCHEMAS",
    "EvidenceBundle",
    "EvidenceBundleError",
    "EvidenceBundleSource",
    "EvidenceCandidate",
    "EvidenceFacet",
    "EvidencePassage",
    "EvidenceRun",
    "EvidenceRunTopic",
    "UnknownEvidenceHandleError",
    "build_evidence_bundles",
    "load_evidence_run",
    "sha256_file",
    "topic_digest",
]
