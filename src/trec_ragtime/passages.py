"""Exact, deterministic passage preparation for downstream reranking."""

from __future__ import annotations

import hashlib
import importlib.metadata
import re
import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence, cast

import msgspec


Representation = Literal["canonical", "report_text"]
Boundary = Literal[
    "paragraph_packed", "sentence_packed", "oversized_sentence_fallback"
]

_REPRESENTATIONS = frozenset(("canonical", "report_text"))
_BOUNDARIES = frozenset(
    ("paragraph_packed", "sentence_packed", "oversized_sentence_fallback")
)
_PIPELINES_BY_LANGUAGE: Mapping[str, str] = {
    "eng": "en_core_web_sm",
    "spa": "es_core_news_sm",
    "rus": "ru_core_news_sm",
    "zho": "zh_core_web_sm",
}
_DEFAULT_LENGTH_FUNCTION_FINGERPRINT = "builtin:len"
_LANGCHAIN_SPLITTERS_DISTRIBUTION = "langchain-text-splitters"
_SPACY_DISTRIBUTION = "spacy"
_PINNED_PIPELINE_MAPPING_ID = ";".join(
    f"{language}={pipeline}" for language, pipeline in _PIPELINES_BY_LANGUAGE.items()
)
_FINGERPRINT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")


def _load_splitters() -> tuple[type[Any], type[Any], type[Any]]:
    """Load LangChain splitters only when passage chunking is actually used."""

    if "CharacterTextSplitter" not in globals():
        from langchain_text_splitters.character import (
            CharacterTextSplitter,
            RecursiveCharacterTextSplitter,
        )
        from langchain_text_splitters.spacy import SpacyTextSplitter

        globals().update(
            {
                "CharacterTextSplitter": CharacterTextSplitter,
                "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter,
                "SpacyTextSplitter": SpacyTextSplitter,
            }
        )
    return (
        cast(type[Any], globals()["CharacterTextSplitter"]),
        cast(type[Any], globals()["RecursiveCharacterTextSplitter"]),
        cast(type[Any], globals()["SpacyTextSplitter"]),
    )


class PassageChunkingError(ValueError):
    """Raised when maintained splitters cannot produce exact source spans."""


class PassageValidationError(ValueError):
    """Raised when passage lineage or source reconstruction is invalid."""


def sha256_bytes(value: str | bytes) -> bytes:
    """Return SHA-256 over exact UTF-8 text bytes or already-encoded bytes."""

    if isinstance(value, str):
        value = value.encode("utf-8")
    elif not isinstance(value, bytes):
        raise TypeError("sha256_bytes expects str or bytes")
    return hashlib.sha256(value).digest()


def _validate_hash(value: object, field_name: str) -> None:
    if not isinstance(value, bytes) or len(value) != hashlib.sha256().digest_size:
        raise PassageValidationError(f"{field_name} must be exactly 32 bytes")


def _validate_id(value: object, field_name: str) -> None:
    if value is not None and (
        not isinstance(value, bytes) or len(value) != hashlib.sha256().digest_size
    ):
        raise PassageValidationError(f"{field_name} must be None or exactly 32 bytes")


def _validate_fingerprint(value: object) -> None:
    if not isinstance(value, str) or not _FINGERPRINT_RE.fullmatch(value):
        raise PassageValidationError("chunker_fingerprint is malformed")


def _passage_id(
    *,
    document_id: str,
    representation: Representation,
    representation_hash: bytes,
    start_char: int,
    end_char: int,
    chunk_index: int,
    chunker_fingerprint: str,
) -> bytes:
    lineage = (
        document_id,
        representation,
        representation_hash,
        start_char,
        end_char,
        chunk_index,
        chunker_fingerprint,
    )
    return sha256_bytes(msgspec.msgpack.encode(lineage))


@dataclass(frozen=True, slots=True)
class Passage:
    """One exact, auditable passage and its document-coordinate lineage."""

    document_id: str
    source_content_hash: bytes
    representation: Representation
    representation_hash: bytes
    language: str
    chunk_index: int
    passage_id: bytes
    text: str
    start_char: int
    end_char: int
    boundary: Boundary
    previous_passage_id: bytes | None
    next_passage_id: bytes | None
    content_hash: bytes
    chunker_fingerprint: str
    overlap_chars: int = 0
    overlap_tokens: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str) or not self.document_id.strip():
            raise PassageValidationError("document_id must not be blank")
        _validate_hash(self.source_content_hash, "source_content_hash")
        if self.representation not in _REPRESENTATIONS:
            raise PassageValidationError("representation must be canonical or report_text")
        _validate_hash(self.representation_hash, "representation_hash")
        if not isinstance(self.language, str) or not self.language.strip():
            raise PassageValidationError("language must not be blank")
        if self.representation == "report_text" and self.language != "eng":
            raise PassageValidationError("report_text representation requires language='eng'")
        if isinstance(self.chunk_index, bool) or not isinstance(self.chunk_index, int):
            raise PassageValidationError("chunk_index must be an integer")
        if self.chunk_index < 0:
            raise PassageValidationError("chunk_index must be nonnegative")
        _validate_hash(self.passage_id, "passage_id")
        if not isinstance(self.text, str) or not self.text:
            raise PassageValidationError("text must not be empty")
        if (
            isinstance(self.start_char, bool)
            or not isinstance(self.start_char, int)
            or isinstance(self.end_char, bool)
            or not isinstance(self.end_char, int)
            or self.start_char < 0
            or self.end_char <= self.start_char
            or self.end_char - self.start_char != len(self.text)
        ):
            raise PassageValidationError("passage offsets must exactly cover text")
        if self.boundary not in _BOUNDARIES:
            raise PassageValidationError("boundary is unknown")
        _validate_id(self.previous_passage_id, "previous_passage_id")
        _validate_id(self.next_passage_id, "next_passage_id")
        _validate_hash(self.content_hash, "content_hash")
        if self.content_hash != sha256_bytes(self.text):
            raise PassageValidationError("content_hash does not match text")
        _validate_fingerprint(self.chunker_fingerprint)
        for field_name, value in (
            ("overlap_chars", self.overlap_chars),
            ("overlap_tokens", self.overlap_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PassageValidationError(f"{field_name} must be a nonnegative integer")
            if value != 0:
                raise PassageValidationError("schema version 1 requires zero overlap")

    @property
    def boundary_path(self) -> str:
        """Compatibility name for the boundary decision recorded on the passage."""

        return self.boundary


def validate_passages(selected_text: str, passages: Sequence[Passage]) -> None:
    """Validate exact reconstruction, hashes, indexes, IDs, and neighbors."""

    if not isinstance(selected_text, str) or not selected_text:
        raise PassageValidationError("selected_text must not be empty")
    if not passages:
        raise PassageValidationError("at least one passage is required")

    first = passages[0]
    if not isinstance(first, Passage):
        raise PassageValidationError("passages must contain Passage objects")
    expected_representation_hash = sha256_bytes(selected_text)
    if first.representation_hash != expected_representation_hash:
        raise PassageValidationError("representation_hash does not match selected_text")

    cursor = 0
    passage_ids: set[bytes] = set()
    chunks: list[str] = []
    for index, passage in enumerate(passages):
        if not isinstance(passage, Passage):
            raise PassageValidationError("passages must contain Passage objects")
        if passage.chunk_index != index:
            raise PassageValidationError("chunk indexes must be contiguous and ordered")
        if passage.representation_hash != expected_representation_hash:
            raise PassageValidationError("all representation hashes must match selected_text")
        for field_name in (
            "document_id",
            "source_content_hash",
            "representation",
            "language",
            "chunker_fingerprint",
            "overlap_chars",
            "overlap_tokens",
        ):
            if getattr(passage, field_name) != getattr(first, field_name):
                raise PassageValidationError(f"inconsistent {field_name}")
        if passage.passage_id in passage_ids:
            raise PassageValidationError("passage IDs must be unique")
        passage_ids.add(passage.passage_id)
        if passage.start_char != cursor:
            raise PassageValidationError("passage spans must be monotonic, adjacent, and gap-free")
        if passage.end_char > len(selected_text):
            raise PassageValidationError("passage end offset exceeds selected_text")
        if selected_text[passage.start_char : passage.end_char] != passage.text:
            raise PassageValidationError("passage text does not match its source slice")
        if passage.content_hash != sha256_bytes(passage.text):
            raise PassageValidationError("passage content hash does not match its text")
        expected_id = _passage_id(
            document_id=passage.document_id,
            representation=cast(Representation, passage.representation),
            representation_hash=passage.representation_hash,
            start_char=passage.start_char,
            end_char=passage.end_char,
            chunk_index=passage.chunk_index,
            chunker_fingerprint=passage.chunker_fingerprint,
        )
        if passage.passage_id != expected_id:
            raise PassageValidationError("passage_id does not match canonical lineage")
        expected_previous = None if index == 0 else passages[index - 1].passage_id
        expected_next = None if index == len(passages) - 1 else passages[index + 1].passage_id
        if passage.previous_passage_id != expected_previous:
            raise PassageValidationError("previous_passage_id is inconsistent")
        if passage.next_passage_id != expected_next:
            raise PassageValidationError("next_passage_id is inconsistent")
        cursor = passage.end_char
        chunks.append(passage.text)

    if cursor != len(selected_text) or "".join(chunks) != selected_text:
        raise PassageValidationError("passages do not fully reconstruct selected_text")


def _pipeline_distribution_name(pipeline_name: str) -> str:
    return pipeline_name.replace("_", "-")


def _required_distribution_names(pipeline_name: str | None) -> tuple[str, ...]:
    selected_pipelines = (
        (pipeline_name,)
        if pipeline_name is not None
        else tuple(_PIPELINES_BY_LANGUAGE.values())
    )
    return (
        _LANGCHAIN_SPLITTERS_DISTRIBUTION,
        _SPACY_DISTRIBUTION,
        *tuple(_pipeline_distribution_name(name) for name in selected_pipelines),
    )


def _runtime_distribution_versions(pipeline_name: str | None) -> tuple[str, ...]:
    versions: list[str] = []
    for distribution_name in _required_distribution_names(pipeline_name):
        try:
            version = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError as error:
            raise PassageValidationError(
                f"required runtime distribution {distribution_name!r} is unavailable; "
                "cannot create a safe passage chunker fingerprint"
            ) from error
        if not isinstance(version, str) or not version.strip():
            raise PassageValidationError(
                f"required runtime distribution {distribution_name!r} has no stable version"
            )
        versions.append(f"{distribution_name}={version}")
    return tuple(versions)


def _default_fingerprint(config: "ChunkerConfig") -> str:
    material = "|".join(
        (
            "passages-v2",
            f"chunk_size={config.chunk_size}",
            f"length_function={config.resolved_length_function_fingerprint}",
            f"pipeline={config.pipeline_name or 'language-map'}",
            f"pinned_mapping={_PINNED_PIPELINE_MAPPING_ID}",
            f"runtime_distributions={';'.join(_runtime_distribution_versions(config.pipeline_name))}",
            f"max_length={config.max_length}",
            f"chunk_overlap={config.chunk_overlap}",
            f"overlap_chars={config.overlap_chars}",
            f"overlap_tokens={config.overlap_tokens}",
        )
    )
    return "langchain-spacy-v2/" + hashlib.sha256(material.encode("utf-8")).hexdigest()


def _compose_fingerprint(base: str, caller_fingerprint: str | None) -> str:
    if caller_fingerprint is None:
        return base
    composed = f"{base}/caller/{caller_fingerprint}"
    if len(composed) > 256:
        composed = f"{base}/caller/{hashlib.sha256(caller_fingerprint.encode('utf-8')).hexdigest()}"
    _validate_fingerprint(composed)
    return composed


@dataclass(frozen=True, slots=True)
class ChunkerConfig:
    """Immutable chunking settings shared by all maintained splitter stages."""

    chunk_size: int = 512
    length_function: Callable[[str], int] | None = None
    length_function_fingerprint: str | None = None
    pipeline_name: str | None = None
    fingerprint: str | None = None
    chunk_overlap: int = 0
    overlap_chars: int = 0
    overlap_tokens: int = 0
    max_length: int = 1_000_000
    _resolved_length_function_fingerprint: str = field(init=False, repr=False)
    _resolved_fingerprint: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.chunk_size, bool) or not isinstance(self.chunk_size, int):
            raise PassageValidationError("chunk_size must be an integer")
        if self.chunk_size <= 0:
            raise PassageValidationError("chunk_size must be positive")
        if self.length_function is None:
            raise PassageValidationError(
                "length_function is required; pass len explicitly for boundary-only chunking"
            )
        if not callable(self.length_function):
            raise PassageValidationError("length_function must be callable")
        if self.length_function_fingerprint is not None:
            _validate_fingerprint(self.length_function_fingerprint)
        else:
            callable_fingerprint = getattr(self.length_function, "fingerprint", None)
            if isinstance(callable_fingerprint, str) and callable_fingerprint.strip():
                _validate_fingerprint(callable_fingerprint)
                object.__setattr__(
                    self,
                    "_resolved_length_function_fingerprint",
                    callable_fingerprint,
                )
            elif self.length_function is len:
                object.__setattr__(
                    self,
                    "_resolved_length_function_fingerprint",
                    _DEFAULT_LENGTH_FUNCTION_FINGERPRINT,
                )
            else:
                raise PassageValidationError(
                    "custom length_function requires length_function_fingerprint or a "
                    "stable nonblank .fingerprint"
                )
        if self.length_function_fingerprint is not None:
            object.__setattr__(
                self,
                "_resolved_length_function_fingerprint",
                self.length_function_fingerprint,
            )
        if self.pipeline_name is not None and (
            not isinstance(self.pipeline_name, str) or not self.pipeline_name.strip()
        ):
            raise PassageValidationError("pipeline_name must not be blank")
        if self.fingerprint is not None:
            _validate_fingerprint(self.fingerprint)
        for field_name, value in (
            ("chunk_overlap", self.chunk_overlap),
            ("overlap_chars", self.overlap_chars),
            ("overlap_tokens", self.overlap_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PassageValidationError(f"{field_name} must be a nonnegative integer")
            if value != 0:
                raise PassageValidationError("schema version 1 requires zero overlap")
        if isinstance(self.max_length, bool) or not isinstance(self.max_length, int):
            raise PassageValidationError("max_length must be an integer")
        if self.max_length <= 0:
            raise PassageValidationError("max_length must be positive")
        object.__setattr__(
            self,
            "_resolved_fingerprint",
            _compose_fingerprint(_default_fingerprint(self), self.fingerprint),
        )

    @property
    def resolved_length_function_fingerprint(self) -> str:
        return self._resolved_length_function_fingerprint

    @property
    def resolved_fingerprint(self) -> str:
        return self._resolved_fingerprint


class PassageChunker(Protocol):
    @property
    def fingerprint(self) -> str: ...

    def chunk(
        self,
        *,
        document_id: str,
        canonical_text: str,
        selected_text: str,
        representation: Representation,
        language: str,
    ) -> tuple[Passage, ...]: ...


@dataclass(frozen=True, slots=True)
class _Span:
    text: str
    start_char: int
    end_char: int


class LangChainPassageChunker:
    """Compose exact paragraph, sentence, and exceptional sentence fallback stages."""

    def __init__(
        self,
        config: ChunkerConfig | None = None,
        *,
        chunk_size: int | None = None,
        length_function: Callable[[str], int] | None = None,
        length_function_fingerprint: str | None = None,
        pipeline_name: str | None = None,
        fingerprint: str | None = None,
    ) -> None:
        if config is not None and any(
            value is not None
            for value in (
                chunk_size,
                length_function,
                length_function_fingerprint,
                pipeline_name,
                fingerprint,
            )
        ):
            raise TypeError("pass either config or keyword chunker settings, not both")
        if config is None:
            config = ChunkerConfig(
                chunk_size=512 if chunk_size is None else chunk_size,
                length_function=length_function,
                length_function_fingerprint=length_function_fingerprint,
                pipeline_name=pipeline_name,
                fingerprint=fingerprint,
            )
        self.config = config
        self._sentence_splitters: dict[str, object] = {}

    @property
    def fingerprint(self) -> str:
        return self.config.resolved_fingerprint

    def _pipeline_for_language(self, language: str) -> str:
        if self.config.pipeline_name is not None:
            return self.config.pipeline_name
        try:
            return _PIPELINES_BY_LANGUAGE[language]
        except KeyError as error:
            raise PassageChunkingError(
                f"no pinned spaCy pipeline is configured for language {language!r}"
            ) from error

    def _paragraph_splitter(self) -> object:
        character_splitter, _, _ = _load_splitters()
        return character_splitter(
            separator=r"(?:\r?\n){2,}",
            is_separator_regex=True,
            keep_separator="end",
            strip_whitespace=False,
            chunk_overlap=0,
            add_start_index=True,
            chunk_size=self.config.chunk_size,
            length_function=self.config.length_function,
        )

    def _sentence_splitter(self, pipeline_name: str) -> object:
        splitter = self._sentence_splitters.get(pipeline_name)
        if splitter is not None:
            return splitter
        _, _, spacy_splitter = _load_splitters()
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"\[W108\].*")
                splitter = spacy_splitter(
                    separator="",
                    pipeline=pipeline_name,
                    max_length=self.config.max_length,
                    strip_whitespace=False,
                    chunk_overlap=0,
                    add_start_index=True,
                    chunk_size=self.config.chunk_size,
                    length_function=self.config.length_function,
                )
        except (OSError, ImportError, ValueError) as error:
            raise PassageChunkingError(
                f"required spaCy pipeline {pipeline_name!r} is unavailable"
            ) from error
        self._sentence_splitters[pipeline_name] = splitter
        return splitter

    def _recursive_splitter(self) -> object:
        _, recursive_splitter, _ = _load_splitters()
        return recursive_splitter(
            separators=[" ", ""],
            keep_separator="end",
            strip_whitespace=False,
            chunk_overlap=0,
            add_start_index=True,
            chunk_size=self.config.chunk_size,
            length_function=self.config.length_function,
        )

    @staticmethod
    def _spans_from_documents(
        parent_text: str,
        documents: Sequence[object],
        *,
        parent_name: str,
    ) -> tuple[_Span, ...]:
        spans: list[_Span] = []
        cursor = 0
        for document in documents:
            content = getattr(document, "page_content", None)
            metadata = getattr(document, "metadata", None)
            if not isinstance(content, str) or not content:
                raise PassageChunkingError(f"{parent_name} splitter returned empty text")
            if not isinstance(metadata, dict) or "start_index" not in metadata:
                raise PassageChunkingError(
                    f"{parent_name} splitter did not return start_index metadata"
                )
            start = metadata["start_index"]
            if isinstance(start, bool) or not isinstance(start, int) or start < 0:
                raise PassageChunkingError(f"{parent_name} splitter returned invalid start_index")
            end = start + len(content)
            if start != cursor:
                raise PassageChunkingError(
                    f"{parent_name} splitter returned a gap or overlap at {start}"
                )
            if end > len(parent_text) or parent_text[start:end] != content:
                raise PassageChunkingError(
                    f"{parent_name} splitter returned a non-source slice"
                )
            spans.append(_Span(content, start, end))
            cursor = end
        if cursor != len(parent_text) or "".join(span.text for span in spans) != parent_text:
            raise PassageChunkingError(
                f"{parent_name} splitter did not fully reconstruct its parent text"
            )
        return tuple(spans)

    def _split_exact(self, splitter: object, parent_text: str, *, parent_name: str) -> tuple[_Span, ...]:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"\[W108\].*")
                documents = splitter.create_documents([parent_text])
        except (OSError, ImportError, ValueError) as error:
            raise PassageChunkingError(f"{parent_name} splitter failed") from error
        return self._spans_from_documents(parent_text, documents, parent_name=parent_name)

    def chunk(
        self,
        *,
        document_id: str,
        canonical_text: str,
        selected_text: str,
        representation: Representation,
        language: str,
    ) -> tuple[Passage, ...]:
        if not isinstance(document_id, str) or not document_id.strip():
            raise PassageChunkingError("document_id must not be blank")
        if not isinstance(canonical_text, str) or not canonical_text:
            raise PassageChunkingError("canonical_text must not be empty")
        if not isinstance(selected_text, str) or not selected_text:
            raise PassageChunkingError("selected_text must not be empty")
        if representation not in _REPRESENTATIONS:
            raise PassageChunkingError("representation must be canonical or report_text")
        if not isinstance(language, str) or not language.strip():
            raise PassageChunkingError("language must not be blank")
        if representation == "report_text" and language != "eng":
            raise PassageChunkingError("report_text representation requires language='eng'")

        pipeline_name = self._pipeline_for_language(language)
        paragraph_documents = self._split_exact(
            self._paragraph_splitter(), selected_text, parent_name="paragraph"
        )
        final_spans: list[tuple[_Span, Boundary]] = []
        for paragraph in paragraph_documents:
            if self.config.length_function(paragraph.text) <= self.config.chunk_size:
                final_spans.append((paragraph, "paragraph_packed"))
                continue

            sentence_documents = self._split_exact(
                self._sentence_splitter(pipeline_name),
                paragraph.text,
                parent_name="sentence",
            )
            for sentence in sentence_documents:
                if self.config.length_function(sentence.text) <= self.config.chunk_size:
                    final_spans.append(
                        (
                            _Span(
                                sentence.text,
                                paragraph.start_char + sentence.start_char,
                                paragraph.start_char + sentence.end_char,
                            ),
                            "sentence_packed",
                        )
                    )
                    continue
                recursive_documents = self._split_exact(
                    self._recursive_splitter(), sentence.text, parent_name="recursive sentence"
                )
                for recursive in recursive_documents:
                    if self.config.length_function(recursive.text) > self.config.chunk_size:
                        raise PassageChunkingError(
                            "recursive sentence fallback exceeded the configured ceiling"
                        )
                    final_spans.append(
                        (
                            _Span(
                                recursive.text,
                                paragraph.start_char
                                + sentence.start_char
                                + recursive.start_char,
                                paragraph.start_char
                                + sentence.start_char
                                + recursive.end_char,
                            ),
                            "oversized_sentence_fallback",
                        )
                    )

        if not final_spans:
            raise PassageChunkingError("no passages were produced")
        self._validate_final_spans(selected_text, final_spans)

        source_hash = sha256_bytes(canonical_text)
        representation_hash = sha256_bytes(selected_text)
        passages = tuple(
            Passage(
                document_id=document_id,
                source_content_hash=source_hash,
                representation=representation,
                representation_hash=representation_hash,
                language=language,
                chunk_index=index,
                passage_id=_passage_id(
                    document_id=document_id,
                    representation=representation,
                    representation_hash=representation_hash,
                    start_char=span.start_char,
                    end_char=span.end_char,
                    chunk_index=index,
                    chunker_fingerprint=self.fingerprint,
                ),
                text=span.text,
                start_char=span.start_char,
                end_char=span.end_char,
                boundary=boundary,
                previous_passage_id=None,
                next_passage_id=None,
                content_hash=sha256_bytes(span.text),
                chunker_fingerprint=self.fingerprint,
                overlap_chars=0,
                overlap_tokens=0,
            )
            for index, (span, boundary) in enumerate(final_spans)
        )
        with_neighbors = tuple(
            replace(
                passage,
                previous_passage_id=None if index == 0 else passages[index - 1].passage_id,
                next_passage_id=None
                if index == len(passages) - 1
                else passages[index + 1].passage_id,
            )
            for index, passage in enumerate(passages)
        )
        validate_passages(selected_text, with_neighbors)
        return with_neighbors

    @staticmethod
    def _validate_final_spans(
        selected_text: str, final_spans: Sequence[tuple[_Span, Boundary]]
    ) -> None:
        cursor = 0
        chunks: list[str] = []
        for span, boundary in final_spans:
            if boundary not in _BOUNDARIES:
                raise PassageChunkingError("unknown final boundary")
            if span.start_char != cursor:
                raise PassageChunkingError("final spans are not adjacent")
            if selected_text[span.start_char : span.end_char] != span.text:
                raise PassageChunkingError("final span is not an exact selected-text slice")
            cursor = span.end_char
            chunks.append(span.text)
        if cursor != len(selected_text) or "".join(chunks) != selected_text:
            raise PassageChunkingError("final spans do not reconstruct selected_text")


__all__ = [
    "ChunkerConfig",
    "LangChainPassageChunker",
    "Passage",
    "PassageChunker",
    "PassageChunkingError",
    "PassageValidationError",
    "sha256_bytes",
    "validate_passages",
]
