"""Exact, deterministic passage preparation for downstream reranking."""

from __future__ import annotations

import hashlib
import importlib.metadata
import re
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence, cast


Representation = Literal["canonical", "report_text"]
Boundary = Literal["sentence", "oversized_sentence"]

_REPRESENTATIONS = frozenset(("canonical", "report_text"))
_BOUNDARIES = frozenset(("sentence", "oversized_sentence"))
_PIPELINES_BY_LANGUAGE: Mapping[str, str] = {
    "eng": "en_core_web_sm",
    "spa": "es_core_news_sm",
    "rus": "ru_core_news_sm",
    "zho": "zh_core_web_sm",
}
_PINNED_PIPELINE_MAPPING = ";".join(
    f"{language}={pipeline}" for language, pipeline in _PIPELINES_BY_LANGUAGE.items()
)
_BLANK_LINE_RE = re.compile(r"(?:\r?\n)[ \t]*(?:\r?\n)")


def _load_splitters() -> tuple[type[Any], type[Any]]:
    try:
        from langchain_text_splitters.character import RecursiveCharacterTextSplitter
        from langchain_text_splitters.spacy import SpacyTextSplitter
    except (ImportError, OSError) as error:
        raise PassageChunkingError(
            "langchain-text-splitters and spaCy are required for passage chunking"
        ) from error
    return SpacyTextSplitter, RecursiveCharacterTextSplitter


def _paragraph_aware_spacy_splitter(spacy_splitter: type[Any]) -> type[Any]:
    """Prefer paragraph boundaries as split points, but still fill the budget.

    Sentence merging stays inside a paragraph, so a passage never straddles a
    blank line mid-thought. Whole paragraphs are then packed together until the
    next one would not fit.

    Flushing at every blank line instead, as this did, made each paragraph its
    own passage no matter how short. A headline or byline became a standalone
    75-character passage while the sentence naming its subject sat in the next
    one, which is how a citation ended up a few dozen characters away from the
    entity its claim named.
    """

    class ParagraphAwareSpacyTextSplitter(spacy_splitter):
        def _merge_splits(
            self,
            splits: Iterable[str],
            separator: str,
        ) -> list[str]:
            paragraphs: list[list[str]] = []
            current: list[str] = []
            for split in splits:
                current.append(split)
                if _BLANK_LINE_RE.search(split):
                    paragraphs.append(current)
                    current = []
            if current:
                paragraphs.append(current)

            chunks: list[str] = []
            group: list[str] = []
            for paragraph in paragraphs:
                combined = group + paragraph
                too_long = (
                    self._length_function(separator.join(combined)) > self._chunk_size
                )
                if group and too_long:
                    chunks.extend(super()._merge_splits(group, separator))
                    group = list(paragraph)
                else:
                    group = combined
            if group:
                chunks.extend(super()._merge_splits(group, separator))
            return chunks

    return ParagraphAwareSpacyTextSplitter


class PassageChunkingError(ValueError):
    """Raised when passage structure or exact chunking is invalid."""


def sha256_bytes(value: str | bytes) -> bytes:
    """Return SHA-256 over exact UTF-8 text bytes or already-encoded bytes."""

    if isinstance(value, str):
        value = value.encode("utf-8")
    elif not isinstance(value, bytes):
        raise TypeError("sha256_bytes expects str or bytes")
    return hashlib.sha256(value).digest()


@dataclass(frozen=True, slots=True)
class Passage:
    document_id: str
    representation: Representation
    language: str
    text: str
    start_char: int
    end_char: int
    content_hash: bytes
    boundary: Boundary

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str) or not self.document_id.strip():
            raise PassageChunkingError("document_id must not be blank")
        if (
            not isinstance(self.representation, str)
            or self.representation not in _REPRESENTATIONS
        ):
            raise PassageChunkingError(
                "representation must be canonical or report_text"
            )
        if not isinstance(self.language, str) or not self.language.strip():
            raise PassageChunkingError("language must not be blank")
        if self.representation == "report_text" and self.language != "eng":
            raise PassageChunkingError(
                "report_text representation requires language='eng'"
            )
        if not isinstance(self.text, str) or not self.text:
            raise PassageChunkingError("text must not be empty")
        if (
            isinstance(self.start_char, bool)
            or not isinstance(self.start_char, int)
            or isinstance(self.end_char, bool)
            or not isinstance(self.end_char, int)
            or self.start_char < 0
            or self.end_char <= self.start_char
            or self.end_char - self.start_char != len(self.text)
        ):
            raise PassageChunkingError("passage offsets must exactly cover text")
        if (
            not isinstance(self.content_hash, bytes)
            or len(self.content_hash) != len(sha256_bytes(b""))
            or self.content_hash != sha256_bytes(self.text)
        ):
            raise PassageChunkingError("content_hash does not match text")
        if not isinstance(self.boundary, str) or self.boundary not in _BOUNDARIES:
            raise PassageChunkingError(
                "boundary must be sentence or oversized_sentence"
            )


def validate_passages(selected_text: str, passages: Sequence[Passage]) -> None:
    """Validate exact source slices and gap-free interval-union coverage."""

    if not isinstance(selected_text, str) or not selected_text:
        raise PassageChunkingError("selected_text must not be empty")
    if not isinstance(passages, Sequence) or isinstance(passages, (str, bytes)):
        raise PassageChunkingError("passages must be a sequence of Passage objects")
    if not passages:
        raise PassageChunkingError("at least one passage is required")

    previous_start = -1
    previous_end = -1
    covered_end = 0
    for passage in passages:
        if not isinstance(passage, Passage):
            raise PassageChunkingError("passages must contain Passage objects")
        if passage.start_char <= previous_start or passage.end_char <= previous_end:
            raise PassageChunkingError("passage positions must be strictly increasing")
        if passage.end_char > len(selected_text):
            raise PassageChunkingError("passage end offset exceeds selected_text")
        if selected_text[passage.start_char : passage.end_char] != passage.text:
            raise PassageChunkingError("passage text does not match its source slice")
        if passage.content_hash != sha256_bytes(passage.text):
            raise PassageChunkingError(
                "passage content hash does not match passage text"
            )
        if passage.start_char > covered_end:
            raise PassageChunkingError("passage intervals contain a gap")
        covered_end = max(covered_end, passage.end_char)
        previous_start = passage.start_char
        previous_end = passage.end_char

    if covered_end != len(selected_text):
        raise PassageChunkingError("passage intervals do not cover selected_text")


@dataclass(frozen=True, slots=True)
class _Span:
    text: str
    start_char: int
    end_char: int


def _runtime_version(distribution: str) -> str:
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as error:
        raise PassageChunkingError(
            f"required runtime distribution {distribution!r} is unavailable"
        ) from error
    if not isinstance(version, str) or not version.strip():
        raise PassageChunkingError(
            f"required runtime distribution {distribution!r} has no version"
        )
    return version


def _positive_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PassageChunkingError(f"{field_name} must be a positive integer")
    return value


class PassageChunker:
    """Chunk one selected representation into exact sentence-based passages."""

    __slots__ = (
        "_pair_length",
        "_overlap_tokens",
        "_fingerprint",
        "_sentence_splitters",
    )

    def __init__(
        self,
        pair_length: object,
        *,
        # Only documents past the window are split at all now, which is rare
        # in this corpus; when it happens the seam deserves real context.
        overlap_tokens: int = 256,
    ) -> None:
        if not callable(pair_length):
            raise PassageChunkingError("pair_length must be callable")
        pair_fingerprint = getattr(pair_length, "fingerprint", None)
        if not isinstance(pair_fingerprint, str) or not pair_fingerprint.strip():
            raise PassageChunkingError("pair_length.fingerprint must be nonblank")
        max_length = _positive_integer(
            getattr(pair_length, "max_length", None), "pair_length.max_length"
        )
        max_passage_length = _positive_integer(
            getattr(pair_length, "max_passage_length", None),
            "pair_length.max_passage_length",
        )
        if max_passage_length > max_length:
            raise PassageChunkingError(
                "pair_length.max_passage_length must not exceed max_length"
            )
        passage_length_function = getattr(pair_length, "passage_length_function", None)
        if not callable(passage_length_function):
            raise PassageChunkingError(
                "pair_length.passage_length_function must be callable"
            )
        if (
            isinstance(overlap_tokens, bool)
            or not isinstance(overlap_tokens, int)
            or overlap_tokens < 0
        ):
            raise PassageChunkingError("overlap_tokens must be a nonnegative integer")
        if overlap_tokens >= max_passage_length:
            raise PassageChunkingError(
                "overlap_tokens must be smaller than max_passage_length"
            )

        versions = (
            f"langchain-text-splitters={_runtime_version('langchain-text-splitters')}",
            f"spacy={_runtime_version('spacy')}",
        )
        pipeline_versions = tuple(
            f"{pipeline}={_runtime_version(pipeline.replace('_', '-'))}"
            for pipeline in _PIPELINES_BY_LANGUAGE.values()
        )
        fingerprint_material = "|".join(
            (
                "passage-chunker-v2",
                f"pair-length={pair_fingerprint}",
                f"max-length={max_length}",
                f"max-passage-length={max_passage_length}",
                f"overlap-tokens={overlap_tokens}",
                f"runtime={';'.join(versions)}",
                f"pipelines={_PINNED_PIPELINE_MAPPING}",
                f"pipeline-versions={';'.join(pipeline_versions)}",
            )
        )
        self._pair_length = pair_length
        self._overlap_tokens = overlap_tokens
        self._fingerprint = (
            "langchain-spacy-v2/" + sha256_bytes(fingerprint_material).hex()
        )
        self._sentence_splitters: dict[str, object] = {}

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def _pipeline_for_language(self, language: str) -> str:
        try:
            return _PIPELINES_BY_LANGUAGE[language]
        except KeyError as error:
            raise PassageChunkingError(
                f"no pinned spaCy pipeline is configured for language {language!r}"
            ) from error

    def _sentence_splitter(self, pipeline: str) -> object:
        splitter = self._sentence_splitters.get(pipeline)
        if splitter is not None:
            return splitter
        spacy_splitter, _ = _load_splitters()
        try:
            paragraph_aware_splitter = _paragraph_aware_spacy_splitter(spacy_splitter)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"\[W108\].*")
                splitter = paragraph_aware_splitter(
                    separator="",
                    pipeline=pipeline,
                    chunk_size=self._pair_length.max_passage_length,
                    chunk_overlap=self._overlap_tokens,
                    length_function=self._pair_length.passage_length_function,
                    strip_whitespace=False,
                    add_start_index=True,
                )
        except PassageChunkingError:
            raise
        except Exception as error:
            raise PassageChunkingError(
                f"required spaCy pipeline {pipeline!r} is unavailable"
            ) from error
        self._sentence_splitters[pipeline] = splitter
        return splitter

    def _recursive_splitter(self, *, chunk_size: int | None = None) -> object:
        _, recursive_splitter = _load_splitters()
        try:
            return recursive_splitter(
                separators=[" ", ""],
                chunk_size=(
                    self._pair_length.max_passage_length
                    if chunk_size is None
                    else chunk_size
                ),
                chunk_overlap=0,
                length_function=self._pair_length.passage_length_function,
                strip_whitespace=False,
                add_start_index=True,
            )
        except Exception as error:
            raise PassageChunkingError(
                "recursive sentence splitter is unavailable"
            ) from error

    @staticmethod
    def _spacy_sentence_units(splitter: object, parent_text: str) -> tuple[_Span, ...]:
        tokenizer = getattr(splitter, "_tokenizer", None)
        if not callable(tokenizer):
            return ()
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"\[W108\].*")
                document = tokenizer(parent_text)
            sentence_iter = document.sents
        except Exception:
            return ()
        units: list[_Span] = []
        for sentence in sentence_iter:
            start = getattr(sentence, "start_char", None)
            text_with_ws = getattr(sentence, "text_with_ws", None)
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or start < 0
                or not isinstance(text_with_ws, str)
                or not text_with_ws
            ):
                return ()
            end = start + len(text_with_ws)
            if end > len(parent_text) or parent_text[start:end] != text_with_ws:
                return ()
            units.append(_Span(text_with_ws, start, end))
        return tuple(units)

    @staticmethod
    def _adapt_offset_from_sentence_units(
        parent_text: str,
        content: str,
        sentence_units: Sequence[_Span],
        *,
        previous_start: int,
        previous_end: int,
    ) -> _Span | None:
        for start_index, first in enumerate(sentence_units):
            if first.start_char <= previous_start:
                continue
            end = first.end_char
            if end > previous_end and parent_text[first.start_char : end] == content:
                return _Span(content, first.start_char, end)
            for later in sentence_units[start_index + 1 :]:
                end = later.end_char
                if end <= previous_end:
                    continue
                if parent_text[first.start_char : end] == content:
                    return _Span(content, first.start_char, end)
        return None

    def _documents_to_spans(
        self,
        parent_text: str,
        documents: Sequence[object],
        *,
        parent_name: str,
        splitter: object,
        sentence_units: Sequence[_Span] | None = None,
    ) -> tuple[_Span, ...]:
        if not documents:
            raise PassageChunkingError(f"{parent_name} splitter returned no documents")
        raw_spans: list[_Span] = []
        previous_start = -1
        previous_end = -1
        for document in documents:
            content = getattr(document, "page_content", None)
            metadata = getattr(document, "metadata", None)
            if not isinstance(content, str) or not content:
                raise PassageChunkingError(
                    f"{parent_name} splitter returned empty text"
                )
            if not isinstance(metadata, dict) or "start_index" not in metadata:
                raise PassageChunkingError(
                    f"{parent_name} splitter did not return start_index metadata"
                )
            start = metadata["start_index"]
            if isinstance(start, bool) or not isinstance(start, int):
                raise PassageChunkingError(
                    f"{parent_name} splitter returned invalid start_index"
                )
            if sentence_units is None:
                sentence_units = self._spacy_sentence_units(splitter, parent_text)
            if sentence_units:
                adapted = self._adapt_offset_from_sentence_units(
                    parent_text,
                    content,
                    sentence_units,
                    previous_start=previous_start,
                    previous_end=previous_end,
                )
                if adapted is not None:
                    start = adapted.start_char
            elif start < 0:
                raise PassageChunkingError(
                    f"{parent_name} splitter returned invalid start_index"
                )
            end = start + len(content)
            if end > len(parent_text) or parent_text[start:end] != content:
                raise PassageChunkingError(
                    f"{parent_name} splitter returned a non-source slice"
                )
            if start <= previous_start or end <= previous_end:
                raise PassageChunkingError(
                    f"{parent_name} splitter returned non-increasing offsets"
                )
            raw_spans.append(_Span(content, start, end))
            previous_start = start
            previous_end = end

        spans = list(raw_spans)
        if spans[0].start_char > 0:
            if not parent_text[: spans[0].start_char].isspace():
                raise PassageChunkingError(
                    f"{parent_name} splitter left an uncovered prefix"
                )
            spans[0] = _Span(
                parent_text[: spans[0].end_char],
                0,
                spans[0].end_char,
            )
        for index in range(1, len(spans)):
            previous = spans[index - 1]
            current = spans[index]
            if current.start_char > previous.end_char:
                gap = parent_text[previous.end_char : current.start_char]
                if not gap.isspace():
                    raise PassageChunkingError(
                        f"{parent_name} splitter left an uncovered gap"
                    )
                spans[index - 1] = _Span(
                    parent_text[previous.start_char : current.start_char],
                    previous.start_char,
                    current.start_char,
                )
        last = spans[-1]
        if last.end_char < len(parent_text):
            if not parent_text[last.end_char :].isspace():
                raise PassageChunkingError(
                    f"{parent_name} splitter left an uncovered suffix"
                )
            spans[-1] = _Span(
                parent_text[last.start_char :],
                last.start_char,
                len(parent_text),
            )
        return tuple(spans)

    def _split_exact(
        self,
        splitter: object,
        parent_text: str,
        *,
        parent_name: str,
        sentence_units: Sequence[_Span] | None = None,
    ) -> tuple[_Span, ...]:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"\[W108\].*")
                documents = splitter.create_documents([parent_text])
        except PassageChunkingError:
            raise
        except Exception as error:
            raise PassageChunkingError(f"{parent_name} splitter failed") from error
        return self._documents_to_spans(
            parent_text,
            documents,
            parent_name=parent_name,
            splitter=splitter,
            sentence_units=sentence_units,
        )

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
        if (
            not isinstance(representation, str)
            or representation not in _REPRESENTATIONS
        ):
            raise PassageChunkingError(
                "representation must be canonical or report_text"
            )
        if not isinstance(language, str) or not language.strip():
            raise PassageChunkingError("language must not be blank")
        if representation == "report_text" and language != "eng":
            raise PassageChunkingError(
                "report_text representation requires language='eng'"
            )

        pipeline = self._pipeline_for_language(language)
        sentence_splitter = self._sentence_splitter(pipeline)
        source_sentence_units = self._spacy_sentence_units(
            sentence_splitter,
            selected_text,
        )
        sentence_spans = self._split_exact(
            sentence_splitter,
            selected_text,
            parent_name="sentence",
            sentence_units=source_sentence_units,
        )
        final_spans: list[tuple[_Span, Boundary]] = []
        for sentence_span in sentence_spans:
            underlying_sentences = tuple(
                sentence
                for sentence in source_sentence_units
                if sentence.start_char >= sentence_span.start_char
                and sentence.end_char <= sentence_span.end_char
            )
            if self._pair_length(sentence_span.text) <= self._pair_length.max_length:
                final_spans.append((sentence_span, "sentence"))
                continue
            if len(underlying_sentences) > 1:
                first_sentence = underlying_sentences[0]
                last_sentence = underlying_sentences[-1]
                if (
                    first_sentence.start_char > sentence_span.start_char
                    and not selected_text[
                        sentence_span.start_char : first_sentence.start_char
                    ].isspace()
                ) or (
                    last_sentence.end_char < sentence_span.end_char
                    and not selected_text[
                        last_sentence.end_char : sentence_span.end_char
                    ].isspace()
                ):
                    raise PassageChunkingError(
                        "RCTS sentence span does not cover the complete unit"
                    )
                sentence_spans_to_process_list: list[_Span] = []
                for index, sentence in enumerate(underlying_sentences):
                    start = (
                        sentence_span.start_char if index == 0 else sentence.start_char
                    )
                    end = (
                        sentence_span.end_char
                        if index == len(underlying_sentences) - 1
                        else underlying_sentences[index + 1].start_char
                    )
                    sentence_spans_to_process_list.append(
                        _Span(selected_text[start:end], start, end)
                    )
                sentence_spans_to_process = tuple(sentence_spans_to_process_list)
            else:
                sentence_spans_to_process = (sentence_span,)

            for sentence_span in sentence_spans_to_process:
                if (
                    self._pair_length(sentence_span.text)
                    <= self._pair_length.max_length
                ):
                    final_spans.append((sentence_span, "sentence"))
                    continue

                underlying_sentences = tuple(
                    sentence
                    for sentence in source_sentence_units
                    if sentence.start_char >= sentence_span.start_char
                    and sentence.end_char <= sentence_span.end_char
                )
                if len(underlying_sentences) != 1:
                    raise PassageChunkingError(
                        "RCTS requires exactly one underlying sentence"
                    )
                underlying_sentence = underlying_sentences[0]
                if (
                    underlying_sentence.start_char > sentence_span.start_char
                    and not selected_text[
                        sentence_span.start_char : underlying_sentence.start_char
                    ].isspace()
                ) or (
                    underlying_sentence.end_char < sentence_span.end_char
                    and not selected_text[
                        underlying_sentence.end_char : sentence_span.end_char
                    ].isspace()
                ):
                    raise PassageChunkingError(
                        "RCTS sentence span does not cover the complete unit"
                    )
                mapped_spans: tuple[_Span, ...] | None = None
                chunk_size = self._pair_length.max_passage_length
                while True:
                    recursive_spans = self._split_exact(
                        self._recursive_splitter(chunk_size=chunk_size),
                        sentence_span.text,
                        parent_name="recursive sentence",
                    )
                    candidate_spans = tuple(
                        _Span(
                            selected_text[
                                sentence_span.start_char
                                + recursive_span.start_char : sentence_span.start_char
                                + recursive_span.end_char
                            ],
                            sentence_span.start_char + recursive_span.start_char,
                            sentence_span.start_char + recursive_span.end_char,
                        )
                        for recursive_span in recursive_spans
                    )
                    overflows = [
                        self._pair_length(span.text) - self._pair_length.max_length
                        for span in candidate_spans
                    ]
                    max_overflow = max(overflows, default=0)
                    if max_overflow <= 0:
                        mapped_spans = candidate_spans
                        break
                    if chunk_size == 1:
                        break
                    next_chunk_size = max(1, chunk_size - max(1, max_overflow))
                    if next_chunk_size >= chunk_size:
                        raise PassageChunkingError(
                            "recursive sentence fallback did not reduce chunk size"
                        )
                    chunk_size = next_chunk_size
                if mapped_spans is None:
                    raise PassageChunkingError(
                        "recursive sentence fallback exceeded the full-pair ceiling"
                    )
                for mapped in mapped_spans:
                    final_spans.append((mapped, "oversized_sentence"))

        if not final_spans:
            raise PassageChunkingError("no passages were produced")

        passages = tuple(
            Passage(
                document_id=document_id,
                representation=cast(Representation, representation),
                language=language,
                text=span.text,
                start_char=span.start_char,
                end_char=span.end_char,
                content_hash=sha256_bytes(span.text),
                boundary=boundary,
            )
            for span, boundary in final_spans
        )
        validate_passages(selected_text, passages)
        for passage in passages:
            if self._pair_length(passage.text) > self._pair_length.max_length:
                raise PassageChunkingError(
                    "final passage exceeded the full-pair ceiling"
                )
        return passages


__all__ = [
    "Boundary",
    "Passage",
    "PassageChunker",
    "PassageChunkingError",
    "Representation",
    "sha256_bytes",
    "validate_passages",
]
