"""Complete deterministic passage-score maps backed by an ArtifactStoreLike."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar, cast

import msgspec

from .passages import Passage, sha256_bytes
from .passage_scoring import PassageScorer, validate_passage_scores

if TYPE_CHECKING:
    from .artifact_store import ArtifactStoreLike


CACHE_SCHEMA_VERSION = 1
OBJECT_ROOT = "passage-scores/v1"
_HASH_LENGTH = 32
_T = TypeVar("_T")


class PassageScoreArtifactValidationError(ValueError):
    """Raised when a stored passage-score artifact is malformed or inconsistent."""


class PassageScoreEnvelope(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Strict MessagePack representation of one complete score-map union."""

    schema_version: int
    scorer_fingerprint: str
    chunker_fingerprint: str
    representation: str
    score_fingerprint: str
    query_hash: bytes
    coverage: tuple[bytes, ...]
    coverage_digest: bytes
    scores: dict[bytes, float]


ScoreEnvelope = PassageScoreEnvelope


def _non_blank(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")
    if any(character in value for character in "\r\n"):
        raise ValueError(f"{field_name} must not contain newlines")
    return value


def _schema_version(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("schema_version must be a positive integer")
    return value


def _representation(value: object) -> str:
    if value not in {"canonical", "report_text"}:
        raise ValueError("representation must be canonical or report_text")
    return cast(str, value)


def _hash(value: object, field_name: str) -> bytes:
    if not isinstance(value, bytes) or len(value) != _HASH_LENGTH:
        raise PassageScoreArtifactValidationError(
            f"{field_name} must be exactly 32 bytes"
        )
    return value


def query_hash(query: str) -> bytes:
    """Hash exactly the query's UTF-8 bytes, without normalization."""

    if not isinstance(query, str):
        raise TypeError("query must be a string")
    return hashlib.sha256(query.encode("utf-8")).digest()


def _coverage_digest(coverage: Sequence[bytes]) -> bytes:
    return hashlib.sha256(msgspec.msgpack.encode(tuple(coverage))).digest()


def effective_score_fingerprint(
    scorer_fingerprint: str,
    chunker_fingerprint: str,
    representation: str,
    *,
    schema_version: int = CACHE_SCHEMA_VERSION,
) -> str:
    """Return a stable identity for all score-result-affecting cache settings."""

    schema = _schema_version(schema_version)
    scorer = _non_blank(scorer_fingerprint, "scorer_fingerprint")
    chunker = _non_blank(chunker_fingerprint, "chunker_fingerprint")
    selected_representation = _representation(representation)
    material = msgspec.msgpack.encode(
        (
            "passage-score-cache-effective-fingerprint-v1",
            schema,
            scorer,
            chunker,
            selected_representation,
        )
    )
    return hashlib.sha256(material).hexdigest()


def score_object_key(
    query: str,
    scorer_fingerprint: str,
    chunker_fingerprint: str,
    representation: str,
    *,
    schema_version: int = CACHE_SCHEMA_VERSION,
) -> str:
    """Return the stable object key for one effective scorer and exact query."""

    effective = effective_score_fingerprint(
        scorer_fingerprint,
        chunker_fingerprint,
        representation,
        schema_version=schema_version,
    )
    digest = query_hash(query).hex()
    return f"{OBJECT_ROOT}/{effective}/{digest[:2]}/{digest}.mpk"


def _validate_envelope(
    envelope: object,
    *,
    require_sorted: bool,
) -> PassageScoreEnvelope:
    if not isinstance(envelope, PassageScoreEnvelope):
        raise PassageScoreArtifactValidationError(
            "decoded object has the wrong envelope type"
        )
    if (
        isinstance(envelope.schema_version, bool)
        or not isinstance(envelope.schema_version, int)
        or envelope.schema_version <= 0
    ):
        raise PassageScoreArtifactValidationError("schema_version is malformed")
    if (
        not isinstance(envelope.scorer_fingerprint, str)
        or not envelope.scorer_fingerprint.strip()
        or any(character in envelope.scorer_fingerprint for character in "\r\n")
    ):
        raise PassageScoreArtifactValidationError("scorer_fingerprint is malformed")
    if (
        not isinstance(envelope.chunker_fingerprint, str)
        or not envelope.chunker_fingerprint.strip()
        or any(character in envelope.chunker_fingerprint for character in "\r\n")
    ):
        raise PassageScoreArtifactValidationError("chunker_fingerprint is malformed")
    if envelope.representation not in {"canonical", "report_text"}:
        raise PassageScoreArtifactValidationError("representation is malformed")
    expected_fingerprint = effective_score_fingerprint(
        envelope.scorer_fingerprint,
        envelope.chunker_fingerprint,
        envelope.representation,
        schema_version=envelope.schema_version,
    )
    if envelope.score_fingerprint != expected_fingerprint:
        raise PassageScoreArtifactValidationError(
            "score_fingerprint does not match envelope identity"
        )
    _hash(envelope.query_hash, "query_hash")
    _hash(envelope.coverage_digest, "coverage_digest")
    if not isinstance(envelope.coverage, tuple):
        raise PassageScoreArtifactValidationError("coverage must be a tuple")
    coverage = tuple(_hash(value, "coverage hash") for value in envelope.coverage)
    if coverage != tuple(sorted(set(coverage))):
        raise PassageScoreArtifactValidationError("coverage must be unique and sorted")
    if envelope.coverage_digest != _coverage_digest(coverage):
        raise PassageScoreArtifactValidationError(
            "coverage_digest does not match coverage"
        )
    if not isinstance(envelope.scores, dict):
        raise PassageScoreArtifactValidationError("scores must be a dict")
    score_keys: list[bytes] = []
    for key, value in envelope.scores.items():
        score_keys.append(_hash(key, "score key"))
        if (
            isinstance(value, bool)
            or type(value) is not float
            or not math.isfinite(value)
        ):
            raise PassageScoreArtifactValidationError(
                "stored scores must be finite floats"
            )
    if set(score_keys) != set(coverage):
        raise PassageScoreArtifactValidationError(
            "scores and coverage must have exact equality"
        )
    if require_sorted and score_keys != sorted(score_keys):
        raise PassageScoreArtifactValidationError("stored score keys must be sorted")
    return envelope


def encode_score_envelope(envelope: PassageScoreEnvelope) -> bytes:
    """Canonicalize and encode a validated score envelope exactly once."""

    _validate_envelope(envelope, require_sorted=False)
    coverage = tuple(sorted(envelope.coverage))
    scores = {key: envelope.scores[key] for key in coverage}
    canonical = PassageScoreEnvelope(
        schema_version=envelope.schema_version,
        scorer_fingerprint=envelope.scorer_fingerprint,
        chunker_fingerprint=envelope.chunker_fingerprint,
        representation=envelope.representation,
        score_fingerprint=envelope.score_fingerprint,
        query_hash=envelope.query_hash,
        coverage=coverage,
        coverage_digest=envelope.coverage_digest,
        scores=scores,
    )
    _validate_envelope(canonical, require_sorted=True)
    return msgspec.msgpack.encode(canonical)


def decode_score_envelope(payload: bytes) -> PassageScoreEnvelope:
    """Decode and fully validate a strict score envelope."""

    if not isinstance(payload, bytes):
        raise PassageScoreArtifactValidationError("artifact payload must be bytes")
    try:
        envelope = msgspec.msgpack.decode(payload, type=PassageScoreEnvelope)
    except (msgspec.DecodeError, TypeError, ValueError) as error:
        raise PassageScoreArtifactValidationError(
            "artifact payload is not valid MessagePack"
        ) from error
    return _validate_envelope(envelope, require_sorted=True)


def _unique_requested(
    passages: Sequence[Passage],
    *,
    representation: str,
    chunker_fingerprint: str,
) -> tuple[tuple[bytes, Passage], ...]:
    if isinstance(passages, (str, bytes, bytearray)) or not isinstance(
        passages, Sequence
    ):
        raise PassageScoreArtifactValidationError(
            "passages must be a sequence of Passage objects"
        )
    unique: dict[bytes, Passage] = {}
    for passage in passages:
        if not isinstance(passage, Passage):
            raise PassageScoreArtifactValidationError(
                "passages must contain Passage objects"
            )
        if passage.representation != representation:
            raise PassageScoreArtifactValidationError(
                "passage representation does not match cache"
            )
        if passage.chunker_fingerprint != chunker_fingerprint:
            raise PassageScoreArtifactValidationError(
                "passage chunker fingerprint does not match cache"
            )
        content_hash = _hash(passage.content_hash, "passage content_hash")
        if content_hash != sha256_bytes(passage.text):
            raise PassageScoreArtifactValidationError(
                "passage content_hash does not match text"
            )
        unique.setdefault(content_hash, passage)
    return tuple(unique.items())


def _store_validation_exception(
    factory: Callable[..., BaseException],
    error: PassageScoreArtifactValidationError,
) -> BaseException:
    try:
        exception = factory(str(error))
    except TypeError:
        exception = factory()
    if not isinstance(exception, BaseException):
        raise TypeError("artifact_validation_error must create an exception") from error
    return exception


@dataclass(frozen=True, slots=True)
class _CacheIdentity:
    schema_version: int
    scorer_fingerprint: str
    chunker_fingerprint: str
    representation: str
    effective_fingerprint: str


class PassageScoreCache:
    """Cache-aside merge adapter for complete per-query passage score maps."""

    __slots__ = (
        "_store",
        "_scorer",
        "_identity",
        "_artifact_validation_error",
    )

    def __init__(
        self,
        store: "ArtifactStoreLike",
        scorer: PassageScorer | str | None = None,
        chunker_fingerprint: str | None = None,
        representation: str | None = None,
        artifact_validation_error: Callable[..., BaseException] | None = None,
        *,
        scorer_fingerprint: str | None = None,
        schema_version: int = CACHE_SCHEMA_VERSION,
        artifact_validation_error_factory: Callable[..., BaseException] | None = None,
    ) -> None:
        if (
            artifact_validation_error is not None
            and artifact_validation_error_factory is not None
        ):
            raise TypeError("provide only one artifact validation exception factory")
        error_factory = artifact_validation_error or artifact_validation_error_factory
        if error_factory is None or not callable(error_factory):
            raise TypeError(
                "artifact_validation_error must be a callable exception class or factory"
            )
        if chunker_fingerprint is None or representation is None:
            raise TypeError("chunker_fingerprint and representation are required")
        schema = _schema_version(schema_version)
        chunker = _non_blank(chunker_fingerprint, "chunker_fingerprint")
        selected_representation = _representation(representation)

        configured_scorer: PassageScorer | None
        if isinstance(scorer, str):
            if scorer_fingerprint is not None and scorer_fingerprint != scorer:
                raise ValueError("scorer and scorer_fingerprint disagree")
            scorer_fp = scorer
            configured_scorer = None
        elif scorer is None:
            if scorer_fingerprint is None:
                raise TypeError("scorer or scorer_fingerprint is required")
            scorer_fp = scorer_fingerprint
            configured_scorer = None
        else:
            configured_scorer = scorer
            try:
                scorer_fp = getattr(scorer, "fingerprint")
            except AttributeError as error:
                raise TypeError("scorer must expose fingerprint") from error
            if scorer_fingerprint is not None and scorer_fingerprint != scorer_fp:
                raise ValueError("scorer and scorer_fingerprint disagree")
            if not isinstance(scorer_fp, str):
                raise TypeError("scorer fingerprint must be a string")
        scorer_fp = _non_blank(scorer_fp, "scorer_fingerprint")
        effective = effective_score_fingerprint(
            scorer_fp,
            chunker,
            selected_representation,
            schema_version=schema,
        )
        self._store = store
        self._scorer = configured_scorer
        self._identity = _CacheIdentity(
            schema,
            scorer_fp,
            chunker,
            selected_representation,
            effective,
        )
        self._artifact_validation_error = error_factory

    @property
    def schema_version(self) -> int:
        return self._identity.schema_version

    @property
    def scorer_fingerprint(self) -> str:
        return self._identity.scorer_fingerprint

    @property
    def chunker_fingerprint(self) -> str:
        return self._identity.chunker_fingerprint

    @property
    def representation(self) -> str:
        return self._identity.representation

    @property
    def effective_fingerprint(self) -> str:
        return self._identity.effective_fingerprint

    @property
    def fingerprint(self) -> str:
        return self.effective_fingerprint

    def object_key(self, query: str) -> str:
        return score_object_key(
            query,
            self.scorer_fingerprint,
            self.chunker_fingerprint,
            self.representation,
            schema_version=self.schema_version,
        )

    def _decoder(self, query: str) -> Callable[[bytes], PassageScoreEnvelope]:
        expected_query_hash = query_hash(query)

        def decode(payload: bytes) -> PassageScoreEnvelope:
            try:
                envelope = decode_score_envelope(payload)
                self._validate_for_request(envelope, expected_query_hash)
                return envelope
            except PassageScoreArtifactValidationError as error:
                raise _store_validation_exception(
                    self._artifact_validation_error,
                    error,
                ) from error

        return decode

    def _validate_for_request(
        self,
        envelope: PassageScoreEnvelope,
        expected_query_hash: bytes,
    ) -> None:
        _validate_envelope(envelope, require_sorted=True)
        if envelope.schema_version != self.schema_version:
            raise PassageScoreArtifactValidationError(
                "artifact schema does not match cache"
            )
        if envelope.scorer_fingerprint != self.scorer_fingerprint:
            raise PassageScoreArtifactValidationError(
                "artifact scorer fingerprint does not match cache"
            )
        if envelope.chunker_fingerprint != self.chunker_fingerprint:
            raise PassageScoreArtifactValidationError(
                "artifact chunker fingerprint does not match cache"
            )
        if envelope.representation != self.representation:
            raise PassageScoreArtifactValidationError(
                "artifact representation does not match cache"
            )
        if envelope.score_fingerprint != self.effective_fingerprint:
            raise PassageScoreArtifactValidationError(
                "artifact effective fingerprint does not match cache"
            )
        if envelope.query_hash != expected_query_hash:
            raise PassageScoreArtifactValidationError(
                "artifact query hash does not match request"
            )

    def _configured_scorer(self, scorer: PassageScorer | None) -> PassageScorer:
        selected = scorer if scorer is not None else self._scorer
        if selected is None:
            raise TypeError(
                "a scorer is required when the cache was configured by fingerprint"
            )
        try:
            fingerprint = getattr(selected, "fingerprint")
            score_method = getattr(selected, "score")
        except AttributeError as error:
            raise TypeError("scorer must expose fingerprint and score") from error
        if fingerprint != self.scorer_fingerprint:
            raise ValueError("scorer fingerprint drifted from cache identity")
        if not callable(score_method):
            raise TypeError("scorer.score must be callable")
        return selected

    def score(
        self,
        query: str,
        passages: Sequence[Passage],
        scorer: PassageScorer | None = None,
    ) -> dict[bytes, float]:
        if not isinstance(query, str) or not query.strip():
            raise PassageScoreArtifactValidationError(
                "query must be a non-blank string"
            )
        requested = _unique_requested(
            passages,
            representation=self.representation,
            chunker_fingerprint=self.chunker_fingerprint,
        )
        if not requested:
            return {}
        key = self.object_key(query)
        stored = self._store.read_validated(key, self._decoder(query))
        if stored is not None:
            try:
                self._validate_for_request(stored, query_hash(query))
            except PassageScoreArtifactValidationError as error:
                raise _store_validation_exception(
                    self._artifact_validation_error,
                    error,
                ) from error
            prior_scores = dict(stored.scores)
        else:
            prior_scores = {}

        missing = tuple(
            passage
            for content_hash, passage in requested
            if content_hash not in prior_scores
        )
        result_scores = dict(prior_scores)
        if missing:
            selected_scorer = self._configured_scorer(scorer)
            response = selected_scorer.score(query, missing)
            new_scores = validate_passage_scores(missing, response)
            merged = dict(prior_scores)
            merged.update(new_scores)
            sorted_scores = {
                content_hash: merged[content_hash] for content_hash in sorted(merged)
            }
            coverage = tuple(sorted(sorted_scores))
            envelope = PassageScoreEnvelope(
                schema_version=self.schema_version,
                scorer_fingerprint=self.scorer_fingerprint,
                chunker_fingerprint=self.chunker_fingerprint,
                representation=self.representation,
                score_fingerprint=self.effective_fingerprint,
                query_hash=query_hash(query),
                coverage=coverage,
                coverage_digest=_coverage_digest(coverage),
                scores=sorted_scores,
            )
            encoded = encode_score_envelope(envelope)
            decoded = decode_score_envelope(encoded)
            self._validate_for_request(decoded, query_hash(query))
            self._store.write(key, encoded)
            result_scores = sorted_scores

        return {
            content_hash: result_scores[content_hash] for content_hash, _ in requested
        }

    def get_or_score(
        self,
        query: str,
        passages: Sequence[Passage],
        scorer: PassageScorer | None = None,
    ) -> dict[bytes, float]:
        return self.score(query, passages, scorer)


class CachedPassageScorer:
    """PassageScorer wrapper that obtains scores through PassageScoreCache."""

    __slots__ = ("_scorer", "_cache", "_fingerprint")

    def __init__(
        self,
        scorer: PassageScorer,
        cache: PassageScoreCache,
    ) -> None:
        if not isinstance(cache, PassageScoreCache):
            raise TypeError("cache must be a PassageScoreCache")
        try:
            fingerprint = getattr(scorer, "fingerprint")
            score_method = getattr(scorer, "score")
        except AttributeError as error:
            raise TypeError("scorer must expose fingerprint and score") from error
        if not isinstance(fingerprint, str) or not callable(score_method):
            raise TypeError(
                "scorer must expose a string fingerprint and callable score"
            )
        if fingerprint != cache.scorer_fingerprint:
            raise ValueError("scorer fingerprint does not match cache identity")
        self._scorer = scorer
        self._cache = cache
        self._fingerprint = fingerprint

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def cache(self) -> PassageScoreCache:
        return self._cache

    def score(
        self,
        query: str,
        passages: Sequence[Passage],
    ) -> Mapping[bytes, float]:
        return self._cache.score(query, passages, self._scorer)


encode_passage_score_envelope = encode_score_envelope
decode_passage_score_envelope = decode_score_envelope

__all__ = [
    "CACHE_SCHEMA_VERSION",
    "CachedPassageScorer",
    "PassageScoreArtifactValidationError",
    "PassageScoreCache",
    "PassageScoreEnvelope",
    "ScoreEnvelope",
    "decode_passage_score_envelope",
    "decode_score_envelope",
    "effective_score_fingerprint",
    "encode_passage_score_envelope",
    "encode_score_envelope",
    "query_hash",
    "score_object_key",
]
