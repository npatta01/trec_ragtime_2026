"""Complete deterministic passage-score maps backed by an ArtifactStoreLike."""

from __future__ import annotations

import hashlib
import math
import numbers
import threading
from collections.abc import Mapping, Sequence
from typing import cast

import msgspec

from .artifact_store import ArtifactStoreLike, ArtifactValidationError
from .passage_scoring import PassageScorer, validate_passage_scores
from .passages import Passage, sha256_bytes

CACHE_SCHEMA_VERSION = 1
OBJECT_ROOT = "passage-scores/v1"
_HASH_LENGTH = 32
_SCORE_TRANSACTION_LOCK = threading.Lock()


class PassageScoreArtifactValidationError(ValueError):
    """Raised when a stored passage-score artifact is malformed."""


def _non_blank(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")
    if any(character in value for character in "\r\n"):
        raise ValueError(f"{field_name} must not contain newlines")
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


def _score_identity(
    scorer_fingerprint: str,
    chunker_fingerprint: str,
    representation: str,
) -> str:
    material = msgspec.msgpack.encode(
        (
            "passage-score-cache-v1",
            _non_blank(scorer_fingerprint, "scorer_fingerprint"),
            _non_blank(chunker_fingerprint, "chunker_fingerprint"),
            _representation(representation),
        )
    )
    return hashlib.sha256(material).hexdigest()


def score_object_key(
    query: str,
    scorer_fingerprint: str,
    chunker_fingerprint: str,
    representation: str,
) -> str:
    """Return the stable object key for one scorer, chunker, representation, and query."""

    digest = query_hash(query).hex()
    identity = _score_identity(
        scorer_fingerprint,
        chunker_fingerprint,
        representation,
    )
    return f"{OBJECT_ROOT}/{identity}/{digest[:2]}/{digest}.mpk"


def _normalize_score_map(
    scores: object,
    *,
    allow_empty: bool = False,
) -> dict[bytes, float]:
    if not isinstance(scores, Mapping):
        raise PassageScoreArtifactValidationError("score artifact must be a map")
    if not scores and not allow_empty:
        raise PassageScoreArtifactValidationError(
            "score artifact must contain at least one score"
        )

    normalized: dict[bytes, float] = {}
    for key, value in scores.items():
        content_hash = _hash(key, "score key")
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise PassageScoreArtifactValidationError(
                "scores must be finite numeric values"
            )
        try:
            score = float(value)
        except (OverflowError, TypeError, ValueError) as error:
            raise PassageScoreArtifactValidationError(
                "scores must be finite numeric values"
            ) from error
        if not math.isfinite(score):
            raise PassageScoreArtifactValidationError(
                "scores must be finite numeric values"
            )
        normalized[content_hash] = score
    return {key: normalized[key] for key in sorted(normalized)}


def encode_score_map(scores: Mapping[bytes, object]) -> bytes:
    """Encode one validated, nonempty score map deterministically."""

    normalized = _normalize_score_map(scores)
    return msgspec.msgpack.encode(normalized)


def decode_score_map(payload: bytes) -> dict[bytes, float]:
    """Decode and validate one strict MessagePack score map."""

    if not isinstance(payload, bytes):
        raise PassageScoreArtifactValidationError("artifact payload must be bytes")
    try:
        decoded = msgspec.msgpack.decode(payload)
    except (msgspec.DecodeError, TypeError, ValueError) as error:
        raise PassageScoreArtifactValidationError(
            "artifact payload is not valid MessagePack"
        ) from error
    return _normalize_score_map(decoded)


def _unique_requested(
    passages: Sequence[Passage],
    *,
    representation: str,
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
        content_hash = _hash(passage.content_hash, "passage content_hash")
        if content_hash != sha256_bytes(passage.text):
            raise PassageScoreArtifactValidationError(
                "passage content_hash does not match text"
            )
        unique.setdefault(content_hash, passage)
    return tuple(unique.items())


class PassageScoreCache:
    """Cache-aside adapter for complete per-query passage score maps.

    All cache instances share a process-local lock around the complete
    read/score/merge/write transaction. A future cross-process
    ``ArtifactStoreLike`` must provide its own atomic update operation.
    """

    __slots__ = (
        "_store",
        "_scorer",
        "_scorer_fingerprint",
        "_chunker_fingerprint",
        "_representation",
    )

    def __init__(
        self,
        store: ArtifactStoreLike,
        scorer: PassageScorer,
        *,
        chunker_fingerprint: str,
        representation: str,
    ) -> None:
        try:
            scorer_fingerprint = getattr(scorer, "fingerprint")
            score_method = getattr(scorer, "score")
        except AttributeError as error:
            raise TypeError("scorer must expose fingerprint and score") from error
        if not isinstance(scorer_fingerprint, str):
            raise TypeError("scorer fingerprint must be a string")
        if not callable(score_method):
            raise TypeError("scorer.score must be callable")

        self._store = store
        self._scorer = scorer
        self._scorer_fingerprint = _non_blank(scorer_fingerprint, "scorer_fingerprint")
        self._chunker_fingerprint = _non_blank(
            chunker_fingerprint, "chunker_fingerprint"
        )
        self._representation = _representation(representation)

    @property
    def scorer_fingerprint(self) -> str:
        return self._scorer_fingerprint

    @property
    def chunker_fingerprint(self) -> str:
        return self._chunker_fingerprint

    @property
    def representation(self) -> str:
        return self._representation

    @property
    def fingerprint(self) -> str:
        return self.scorer_fingerprint

    def object_key(self, query: str) -> str:
        return score_object_key(
            query,
            self.scorer_fingerprint,
            self.chunker_fingerprint,
            self.representation,
        )

    def _decoder(self, payload: bytes) -> dict[bytes, float]:
        try:
            return decode_score_map(payload)
        except PassageScoreArtifactValidationError as error:
            raise ArtifactValidationError(str(error)) from error

    def score(
        self,
        query: str,
        passages: Sequence[Passage],
    ) -> dict[bytes, float]:
        if not isinstance(query, str) or not query.strip():
            raise PassageScoreArtifactValidationError(
                "query must be a non-blank string"
            )

        requested = _unique_requested(
            passages,
            representation=self.representation,
        )
        if not requested:
            return {}

        key = self.object_key(query)
        with _SCORE_TRANSACTION_LOCK:
            stored = self._store.read_validated(key, self._decoder)
            prior_scores = {} if stored is None else _normalize_score_map(stored)
            missing = tuple(
                passage
                for content_hash, passage in requested
                if content_hash not in prior_scores
            )

            if missing:
                response = self._scorer.score(query, missing)
                new_scores = validate_passage_scores(missing, response)
                merged = dict(prior_scores)
                merged.update(new_scores)
                payload = encode_score_map(merged)
                self._store.write(key, payload)
                prior_scores = decode_score_map(payload)

            return {
                content_hash: prior_scores[content_hash]
                for content_hash, _ in requested
            }


class CachedPassageScorer:
    """PassageScorer wrapper that delegates to one PassageScoreCache."""

    __slots__ = ("_cache",)

    def __init__(self, cache: PassageScoreCache) -> None:
        if not isinstance(cache, PassageScoreCache):
            raise TypeError("cache must be a PassageScoreCache")
        self._cache = cache

    @property
    def fingerprint(self) -> str:
        return self._cache.scorer_fingerprint

    @property
    def cache(self) -> PassageScoreCache:
        return self._cache

    def score(
        self,
        query: str,
        passages: Sequence[Passage],
    ) -> Mapping[bytes, float]:
        return self._cache.score(query, passages)


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "CachedPassageScorer",
    "PassageScoreArtifactValidationError",
    "PassageScoreCache",
    "decode_score_map",
    "encode_score_map",
    "query_hash",
    "score_object_key",
]
