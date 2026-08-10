"""Small, dependency-free helpers for conservative English text splitting."""

from __future__ import annotations

import re


_PARAGRAPH_BREAK = re.compile(r"(?:\r\n?|\n)[ \t]*(?:\r\n?|\n)")
_FIRST_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_ACRONYM_AT_END = re.compile(r"\b(?:[^\W\d_]\.){2,}$", re.UNICODE)
_PREFIX_TITLE_AT_END = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Rev|Hon|Gen|Lt|Col|Sgt)\.$",
    re.IGNORECASE,
)
_OTHER_ABBREVIATION_AT_END = re.compile(
    r"\b(?:Sr|Jr|St|vs|etc|No|Fig|Eq|Dept|Inc|Corp|Co|Mt|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.$",
    re.IGNORECASE,
)

# Uppercase words here commonly complete a proper name that starts with an
# initialism. Lowercase continuations do not need this exception. Keeping the
# set narrow avoids turning a sentence-final "U.S." into a join with any next
# sentence that happens to start with a capitalized noun.
_PROPER_ACRONYM_CONTINUATIONS = frozenset(
    {
        "administration",
        "air",
        "army",
        "bureau",
        "court",
        "department",
        "district",
        "federal",
        "government",
        "military",
        "navy",
        "president",
        "senate",
        "state",
        "states",
        "supreme",
    }
)


def has_paragraph_break(separator: str) -> bool:
    """Return whether whitespace contains at least one blank-line boundary."""

    return bool(_PARAGRAPH_BREAK.search(separator))


def continues_after_period(previous: str, following: str) -> bool:
    """Return whether a period split is clearly inside one sentence.

    This intentionally prefers a false split over joining two independent
    claims. Prefix titles (``Dr. Smith``), lowercase/numeric continuations
    (``Jan. 5``), and common proper-name continuations (``U.S. Department``)
    are repaired. Ambiguous uppercase continuations remain separate.
    """

    if not previous.endswith("."):
        return False
    token_match = _FIRST_TOKEN.search(following)
    if token_match is None:
        return False
    next_token = token_match.group(0)
    first_character = next_token[0]

    if _PREFIX_TITLE_AT_END.search(previous):
        return True

    acronym_match = _ACRONYM_AT_END.search(previous)
    other_match = _OTHER_ABBREVIATION_AT_END.search(previous)
    if acronym_match is None and other_match is None:
        return False

    if first_character.isdigit() or first_character.islower():
        return True
    if acronym_match is None:
        return False

    prefix = previous[: acronym_match.start()].strip()
    if not prefix:
        return True
    return next_token.casefold() in _PROPER_ACRONYM_CONTINUATIONS


__all__ = ["continues_after_period", "has_paragraph_break"]
