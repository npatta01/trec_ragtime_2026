"""Deterministic sentence, normalization, and length primitives for reports.

RAGTIME reports are scored as an ordered list of sentences whose combined
NFKC-normalized text must stay inside a per-topic character limit. This module
owns every text rule that decision depends on, and nothing else: it imports no
other project module so the behavior can be pinned exactly and tested offline.

Segmentation is a conservative rule set rather than a statistical model on
purpose. ``SENTENCE_SEGMENTER_VERSION`` participates in the generator
fingerprint, so the behavior it names must be reproducible from the repository
revision alone. A ``pysbd`` or spaCy backend may be substituted through
``SentenceSegmenter``; doing so requires a new version string, which correctly
invalidates cached reports.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

SENTENCE_SEGMENTER_VERSION = "sentence-segmenter-v1"
LENGTH_PACKING_VERSION = "whole-sentence-pack-v1"

TERMINAL_PUNCTUATION = frozenset(".!?…。！？")
CLOSING_CHARACTERS = frozenset("\"'’”)]}」』»")
OPENING_CHARACTERS = frozenset("\"'‘“([{「『«")

# Tokens that end in a period without ending a sentence. Compared lowercased.
ABBREVIATIONS = frozenset(
    {
        "mr.",
        "mrs.",
        "ms.",
        "dr.",
        "prof.",
        "st.",
        "jr.",
        "sr.",
        "vs.",
        "etc.",
        "e.g.",
        "i.e.",
        "cf.",
        "al.",
        "no.",
        "fig.",
        "sec.",
        "art.",
        "jan.",
        "feb.",
        "mar.",
        "apr.",
        "jun.",
        "jul.",
        "aug.",
        "sep.",
        "sept.",
        "oct.",
        "nov.",
        "dec.",
    }
)

# "J.", "U.S.", "E.U." — a run of single letters each followed by a period.
_DOTTED_INITIALS = re.compile(r"(?:[^\W\d_]\.)+\Z")
# A bare list ordinal such as "1." opening the text.
_LIST_ORDINAL = re.compile(r"\d+\.\Z")


@runtime_checkable
class SentenceSegmenter(Protocol):
    """Splits report prose into sentences and recognizes a lone sentence."""

    version: str

    def segment(self, text: str) -> tuple[str, ...]: ...

    def is_single_sentence(self, text: str) -> bool: ...


def normalize_report_text(text: str) -> str:
    """Return NFKC text with whitespace runs collapsed and edges stripped."""

    if not isinstance(text, str):
        raise TypeError("report text must be a string")
    return " ".join(unicodedata.normalize("NFKC", text).split())


def _validated_texts(texts: Sequence[str]) -> list[str]:
    items = list(texts)
    for item in items:
        if not isinstance(item, str):
            raise TypeError("report sentences must be strings")
    return items


def report_character_count(texts: Sequence[str]) -> int:
    """Count NFKC characters in the concatenated report text.

    The concatenation is normalized *after* joining because a combining mark
    opening one sentence can compose with the tail of the previous one; counting
    the parts separately would overstate the total.
    """

    joined = "".join(_validated_texts(texts))
    return len(unicodedata.normalize("NFKC", joined))


def budgeted_character_count(texts: Sequence[str]) -> int:
    """Count characters allowing one join character between sentences.

    The guidelines do not say whether an evaluator concatenates sentences
    directly or with a separator. Enforcing the larger reading costs one
    character per sentence gap and makes both readings safe.
    """

    items = _validated_texts(texts)
    if not items:
        return 0
    return report_character_count(items) + len(items) - 1


def _ends_with_terminal(text: str) -> bool:
    trimmed = text.rstrip()
    while trimmed and trimmed[-1] in CLOSING_CHARACTERS:
        trimmed = trimmed[:-1]
    return bool(trimmed) and trimmed[-1] in TERMINAL_PUNCTUATION


class RuleBasedSegmenter:
    """Conservative, dependency-free English sentence segmenter."""

    version: str = SENTENCE_SEGMENTER_VERSION

    def segment(self, text: str) -> tuple[str, ...]:
        if not isinstance(text, str):
            raise TypeError("report text must be a string")
        normalized = unicodedata.normalize("NFKC", text)
        if not normalized.strip():
            return ()

        segments: list[str] = []
        length = len(normalized)
        start = 0
        index = 0
        while index < length:
            if normalized[index] not in TERMINAL_PUNCTUATION:
                index += 1
                continue

            run_end = index
            while run_end < length and normalized[run_end] in TERMINAL_PUNCTUATION:
                run_end += 1
            close_end = run_end
            while close_end < length and normalized[close_end] in CLOSING_CHARACTERS:
                close_end += 1

            resume = self._boundary_resume(normalized, index, run_end, close_end, start)
            if resume is None:
                index = max(close_end, index + 1)
                continue

            segments.append(normalized[start:close_end].strip())
            start = resume
            index = resume

        tail = normalized[start:].strip()
        if tail:
            segments.append(tail)
        return tuple(segment for segment in segments if segment)

    def _boundary_resume(
        self,
        text: str,
        index: int,
        run_end: int,
        close_end: int,
        start: int,
    ) -> int | None:
        """Return where the next sentence begins, or ``None`` for no boundary."""

        length = len(text)
        if close_end >= length or not text[close_end].isspace():
            return None
        resume = close_end
        while resume < length and text[resume].isspace():
            resume += 1
        if resume >= length:
            return None
        following = text[resume]
        if not (
            following.isupper() or following.isdigit() or following in OPENING_CHARACTERS
        ):
            return None
        if self._is_suppressed(text, index, run_end, start):
            return None
        return resume

    def _is_suppressed(self, text: str, index: int, run_end: int, start: int) -> bool:
        # Only a lone period is ambiguous; "!", "?", and "..." always terminate.
        if run_end - index != 1 or text[index] != ".":
            return False
        token_start = index
        while token_start > start and not text[token_start - 1].isspace():
            token_start -= 1
        token = text[token_start:run_end]
        if token.lower() in ABBREVIATIONS:
            return True
        if _DOTTED_INITIALS.fullmatch(token):
            return True
        # A bare ordinal only opens a list when it opens the text.
        if token_start == start and _LIST_ORDINAL.fullmatch(token):
            return True
        # A period bracketed by digits is a decimal separator, never a boundary.
        if (
            token_start < index
            and text[index - 1].isdigit()
            and run_end < len(text)
            and text[run_end].isdigit()
        ):
            return True
        return False

    def is_single_sentence(self, text: str) -> bool:
        """Return whether ``text`` is exactly one trimmed, terminated sentence."""

        if not isinstance(text, str) or not text.strip():
            return False
        if "\n" in text or "\r" in text:
            return False
        if text != text.strip():
            return False
        normalized = unicodedata.normalize("NFKC", text).strip()
        if not _ends_with_terminal(normalized):
            return False
        return self.segment(text) == (normalized,)


_DEFAULT_SEGMENTER = RuleBasedSegmenter()


def default_segmenter() -> RuleBasedSegmenter:
    """Return the shared pinned segmenter instance."""

    return _DEFAULT_SEGMENTER


def pack_report_sentences(
    texts: Sequence[str], *, limit: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Keep whole leading sentences that fit ``limit``; drop the rest.

    Returns ``(kept_indices, dropped_indices)``. Packing stops at the first
    sentence that overflows rather than skipping ahead to a shorter later one:
    reports read as prose, so preserving the leading narrative beats maximizing
    character utilization. No sentence is ever split or truncated.
    """

    items = _validated_texts(texts)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("character limit must be a positive integer")
    if not items:
        raise ValueError("cannot pack an empty report")

    kept: list[int] = []
    dropped: tuple[int, ...] = ()
    for index, text in enumerate(items):
        candidate = [items[position] for position in kept]
        candidate.append(text)
        if budgeted_character_count(candidate) > limit:
            dropped = tuple(range(index, len(items)))
            break
        kept.append(index)

    if not kept:
        raise ValueError("the first report sentence exceeds the character limit")
    return tuple(kept), dropped


__all__ = [
    "ABBREVIATIONS",
    "CLOSING_CHARACTERS",
    "LENGTH_PACKING_VERSION",
    "OPENING_CHARACTERS",
    "SENTENCE_SEGMENTER_VERSION",
    "TERMINAL_PUNCTUATION",
    "RuleBasedSegmenter",
    "SentenceSegmenter",
    "budgeted_character_count",
    "default_segmenter",
    "normalize_report_text",
    "pack_report_sentences",
    "report_character_count",
]
