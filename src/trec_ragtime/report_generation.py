"""Evidence-only report generation for exactly one topic.

The generator may read the topic request, the sealed bundle, and generation
configuration -- nothing else. It never retrieves, never reaches past the
bundle's allowlist, and never invents a replacement fact for a claim it had to
drop. Canonical document IDs are not shown to the model at all: prompts carry
compact ``S1..Sn`` handles, and only ``EvidenceBundle.resolve_handle`` maps them
back, so an out-of-allowlist citation is unrepresentable rather than merely
detected.

The provider seam is ``ChatCompletionsLike``, structurally satisfied by
``CachedChatCompletions``. That wrapper caches the raw provider response even
when our post-validation rejects it, so a rejected draft would replay forever if
we re-sent the same request. Every repair attempt therefore appends an
attempt-numbered instruction, which changes the completion cache identity and
makes the next attempt a genuinely new request.

A multi-sentence candidate is dropped and repaired rather than re-split with an
inherited citation set: splitting would assert whole-sentence support we never
verified, which is exactly the citation rule the track scores.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from .evidence_bundle import EvidenceBundle, UnknownEvidenceHandleError
from .report_cache import (
    DroppedSentence,
    ReportArtifactCache,
    ReportCitation,
    ReportDiagnostics,
    ReportRequestIdentity,
    ReportSentence,
    ReportValidationError,
    TopicReport,
    validate_report,
)
from .report_sentences import (
    SentenceSegmenter,
    default_segmenter,
    normalize_report_text,
    pack_report_sentences,
)

# Citation scores: position dominates, retrieval rank breaks ties inside a
# position band. The band gap (10.0) is strictly larger than the largest rank
# penalty (< 9.0), so scores strictly descend for any bundle size.
_POSITION_GAP = 10.0
_RANK_SPAN = 9.0
_TOP_SCORE = 100.0


class ReportGenerationError(RuntimeError):
    """Generation could not produce a valid report for this topic."""


class MalformedModelOutputError(ReportGenerationError):
    """The provider returned output that is not usable structured output."""


class ChatCompletionsLike(Protocol):
    """The one provider capability generation needs."""

    def create(self, *, refresh: bool = False, **request: Any) -> Any: ...


class ObservedCacheStore:
    """Wraps a store so callers can tell whether the last read was a hit.

    ``CachedChatCompletions`` returns an identical response on a hit and a miss,
    so completion-cache reuse is otherwise invisible. Wrapping the store is the
    least intrusive way to record it in run diagnostics.
    """

    def __init__(self, store: Any) -> None:
        self._store = store
        self.last_read_hit = False

    def read_validated(self, object_key: str, decoder: Callable[[bytes], Any]) -> Any | None:
        value = self._store.read_validated(object_key, decoder)
        self.last_read_hit = value is not None
        return value

    def write(self, object_key: str, payload: bytes) -> bool:
        self.last_read_hit = False
        return self._store.write(object_key, payload)

    def read(self, object_key: str) -> bytes | None:
        return self._store.read(object_key)

    def evict_local(self, object_key: str) -> None:
        self._store.evict_local(object_key)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)


@dataclass(frozen=True, slots=True)
class FacetBudget:
    """How many characters one facet should receive."""

    facet_id: str
    text: str
    characters: int


@dataclass(frozen=True, slots=True)
class SentenceCandidate:
    """One model-proposed sentence, before local validation."""

    text: str
    handles: tuple[str, ...]
    facet_ids: tuple[str, ...]
    substantive: bool


def allocate_facet_budget(
    bundle: EvidenceBundle, *, limit: int, budget_ratio: float
) -> tuple[FacetBudget, ...]:
    """Split the character budget evenly across facets, remainder first."""

    facets = bundle.facets
    if not facets:
        raise ReportGenerationError("bundle has no facets to budget")
    total = max(len(facets), int(limit * budget_ratio))
    base, remainder = divmod(total, len(facets))
    return tuple(
        FacetBudget(
            facet_id=facet.facet_id,
            text=facet.text,
            characters=base + (1 if position < remainder else 0),
        )
        for position, facet in enumerate(facets)
    )


def build_response_format(bundle: EvidenceBundle, settings: Any) -> dict[str, Any]:
    """Build the strict output contract, inlining the per-topic allowlists."""

    if settings.response_format_mode == "json_object":
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": settings.output_schema_version.replace("-", "_"),
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["sentences"],
                "properties": {
                    "sentences": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["text", "evidence", "facets", "substantive"],
                            "properties": {
                                "text": {"type": "string"},
                                "evidence": {
                                    "type": "array",
                                    "items": {
                                        "type": "string",
                                        "enum": list(bundle.citable_handles),
                                    },
                                },
                                "facets": {
                                    "type": "array",
                                    "items": {"type": "string", "enum": list(bundle.facet_ids)},
                                },
                                "substantive": {"type": "boolean"},
                            },
                        },
                    }
                },
            },
        },
    }


_SYSTEM_PROMPT = """You write factual English intelligence reports from a fixed set of numbered sources.

Rules you must follow exactly:
- Use ONLY the supplied sources. Never add outside knowledge, and never guess.
- Write one atomic factual claim per sentence. Never put two claims in one sentence.
- Every substantive sentence lists the source handles that support the ENTIRE sentence.
- A source supports a sentence only if the whole sentence follows from that source alone.
  If a source supports only part of the sentence, split the sentence instead.
- List the strongest fully supporting source first.
- Never merge conflicting accounts. Report them as separate attributed sentences.
- Use only handles from the SOURCES list. Never invent a handle.
- Write plain prose: no markdown, no headings, no lists, no bracketed citation markers.
- Prefer coverage of every listed topic over depth on any one of them.
- Stay within the character budget."""


def _format_sources(bundle: EvidenceBundle) -> str:
    blocks = []
    for passage in bundle.passages:
        if not passage.citable:
            continue
        meta = " ".join(
            part
            for part in (
                passage.language,
                passage.date[:10] if passage.date else None,
                passage.url,
            )
            if part
        )
        blocks.append(f"[{passage.handle}] {meta}\n{bundle.prompt_excerpt(passage.handle)}")
    return "\n\n".join(blocks)


def build_messages(
    bundle: EvidenceBundle,
    budgets: Sequence[FacetBudget],
    settings: Any,
    *,
    repair: Sequence[str] | None = None,
    attempt: int = 1,
) -> list[dict[str, str]]:
    """Build the chat messages for one generation attempt."""

    total = sum(item.characters for item in budgets)
    facet_lines = "\n".join(
        f"- {item.facet_id} (about {item.characters} characters): {item.text}"
        for item in budgets
    )
    user = (
        "REPORT REQUEST\n"
        f"Title: {bundle.title or bundle.topic_id}\n"
        f"Problem statement: {bundle.problem_statement}\n"
        f"Requester background: {bundle.background}\n\n"
        f"TOTAL CHARACTER BUDGET: {total} characters across all sentences.\n\n"
        f"TOPICS TO COVER\n{facet_lines}\n\n"
        f"SOURCES\n{_format_sources(bundle)}\n\n"
        "Return a JSON object with a \"sentences\" array. Each entry has "
        "\"text\" (one sentence), \"evidence\" (source handles supporting the whole "
        "sentence, strongest first), \"facets\" (which topics it covers), and "
        "\"substantive\" (false only for a connective sentence that states no fact)."
    )
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    if repair:
        problems = "\n".join(f"- {reason}" for reason in sorted(set(repair)))
        messages.append(
            {
                "role": "user",
                "content": (
                    f"REVISION {attempt}\n"
                    "The previous draft was rejected for these problems:\n"
                    f"{problems}\n\n"
                    "Rewrite the complete report and fix every one of them. Do not "
                    "invent facts to replace a dropped claim: if you cannot support a "
                    "claim with a listed source, leave it out."
                ),
            }
        )
    return messages


def _unwrap_fenced_json(content: str) -> str:
    text = content.strip()
    if not text.startswith("```"):
        return text
    body = text[3:]
    if body.lower().startswith("json"):
        body = body[4:]
    return body.rsplit("```", 1)[0].strip()


def parse_sentence_candidates(
    response: Any, bundle: EvidenceBundle
) -> tuple[SentenceCandidate, ...]:
    """Extract sentence candidates from one provider response, or fail closed."""

    choices = getattr(response, "choices", None)
    if not choices:
        raise MalformedModelOutputError("response carried no choices")
    choice = choices[0]

    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason not in (None, "stop"):
        raise MalformedModelOutputError(f"completion did not finish: {finish_reason}")

    message = getattr(choice, "message", None)
    if message is None:
        raise MalformedModelOutputError("response carried no message")
    if getattr(message, "refusal", None):
        raise MalformedModelOutputError("model refused the request")

    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise MalformedModelOutputError("response carried no content")

    try:
        payload = json.loads(_unwrap_fenced_json(content))
    except json.JSONDecodeError as error:
        raise MalformedModelOutputError(f"response is not valid JSON: {error}") from None
    if not isinstance(payload, dict):
        raise MalformedModelOutputError("response must be a JSON object")

    sentences = payload.get("sentences")
    if not isinstance(sentences, list) or not sentences:
        raise MalformedModelOutputError("response must carry a non-empty sentences array")

    known_facets = set(bundle.facet_ids)
    candidates: list[SentenceCandidate] = []
    for item in sentences:
        if not isinstance(item, dict):
            raise MalformedModelOutputError("every sentence must be an object")
        text = item.get("text")
        if not isinstance(text, str):
            raise MalformedModelOutputError("every sentence needs text")
        evidence = item.get("evidence") or []
        if not isinstance(evidence, list):
            raise MalformedModelOutputError("evidence must be an array")
        facets = item.get("facets") or []
        if not isinstance(facets, list):
            raise MalformedModelOutputError("facets must be an array")
        substantive = item.get("substantive", True)
        if not isinstance(substantive, bool):
            raise MalformedModelOutputError("substantive must be a boolean")

        handles: list[str] = []
        for handle in evidence:
            if isinstance(handle, str) and handle not in handles:
                handles.append(handle)
        candidates.append(
            SentenceCandidate(
                text=text,
                handles=tuple(handles),
                facet_ids=tuple(
                    facet
                    for facet in dict.fromkeys(facets)
                    if isinstance(facet, str) and facet in known_facets
                ),
                substantive=substantive,
            )
        )
    return tuple(candidates)


def rank_citations(
    candidate: SentenceCandidate, bundle: EvidenceBundle, *, max_citations: int
) -> tuple[ReportCitation, ...]:
    """Score citations so the strongest fully supporting source ranks first.

    Model order decides the band; retrieval rank orders within it. Extra
    citations beyond the cap are truncated rather than dropping the sentence --
    the cap is our policy, not a model error.
    """

    span = max(bundle.max_passages, 1)
    citations: list[ReportCitation] = []
    for position, handle in enumerate(candidate.handles[:max_citations]):
        passage = bundle.passage(handle)
        penalty = _RANK_SPAN * (passage.rank - 1) / span
        score = round(_TOP_SCORE - _POSITION_GAP * position - penalty, 4)
        if not math.isfinite(score):
            raise ReportGenerationError("citation score is not finite")
        citations.append(
            ReportCitation(document_id=passage.document_id, handle=handle, score=score)
        )
    return tuple(citations)


def _screen_candidates(
    candidates: Sequence[SentenceCandidate],
    bundle: EvidenceBundle,
    *,
    segmenter: SentenceSegmenter,
) -> tuple[list[SentenceCandidate], list[DroppedSentence]]:
    """Keep candidates that satisfy the contract; record why the rest went."""

    kept: list[SentenceCandidate] = []
    dropped: list[DroppedSentence] = []
    seen: set[str] = set()

    for index, candidate in enumerate(candidates, start=1):
        text = normalize_report_text(candidate.text)

        def drop(reason: str) -> None:
            dropped.append(
                DroppedSentence.create(index=index, reason=reason, text=candidate.text)
            )

        if not text:
            drop("empty_text")
            continue
        if not segmenter.is_single_sentence(text):
            drop("not_single_sentence")
            continue
        if text in seen:
            drop("duplicate_text")
            continue

        unknown = [
            handle for handle in candidate.handles if handle not in bundle.handle_allowlist
        ]
        if unknown:
            drop("unknown_handle")
            continue
        citable = set(bundle.citable_handles)
        if any(handle not in citable for handle in candidate.handles):
            # Partial-support evidence is context only. Promoting it to a full
            # citation is exactly what the upstream policy forbids.
            drop("not_citable")
            continue
        if candidate.substantive and not candidate.handles:
            drop("missing_citation")
            continue
        if not candidate.substantive and candidate.handles:
            # A connective sentence that cites is really a factual claim; keep
            # it as one rather than discarding usable content.
            candidate = SentenceCandidate(
                text=text,
                handles=candidate.handles,
                facet_ids=candidate.facet_ids,
                substantive=True,
            )

        seen.add(text)
        kept.append(
            SentenceCandidate(
                text=text,
                handles=candidate.handles,
                facet_ids=candidate.facet_ids,
                substantive=candidate.substantive,
            )
        )
    return kept, dropped


def _usage_mapping(response: Any) -> dict[str, int] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    dump = getattr(usage, "model_dump", None)
    payload = dump(mode="json") if callable(dump) else dict(usage)
    return {
        key: int(value)
        for key, value in payload.items()
        if isinstance(value, int) and not isinstance(value, bool)
    }


def _completion_request(
    bundle: EvidenceBundle,
    budgets: Sequence[FacetBudget],
    settings: Any,
    *,
    repair: Sequence[str] | None,
    attempt: int,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": settings.model,
        "messages": build_messages(
            bundle, budgets, settings, repair=repair, attempt=attempt
        ),
        "temperature": settings.temperature,
        "max_tokens": settings.max_output_tokens,
        "response_format": build_response_format(bundle, settings),
    }
    if settings.top_p is not None:
        request["top_p"] = settings.top_p
    if settings.seed is not None:
        request["seed"] = settings.seed
    return request


def generate_topic_report(
    *,
    bundle: EvidenceBundle,
    request: ReportRequestIdentity,
    completions: ChatCompletionsLike,
    cache: ReportArtifactCache | None = None,
    clock: Callable[[], float] = time.time,
    segmenter: SentenceSegmenter | None = None,
    refresh: bool = False,
    llm_cache_probe: Callable[[], bool] | None = None,
) -> TopicReport:
    """Produce one validated report for one topic, or raise.

    Follows the generation flow in issue #17: validate inputs, budget facets,
    ask for strict structured sentences, validate handles and sentences locally,
    repair without inventing facts, resolve handles to canonical document IDs,
    rank citations, pack whole sentences to the limit, then validate and store.
    """

    segmenter = segmenter or default_segmenter()
    settings = request.settings
    if bundle.seal_bytes != request.bundle_seal:
        raise ReportGenerationError("bundle does not match the request seal")
    if bundle.topic_id != request.topic_id:
        raise ReportGenerationError("bundle does not match the request topic")

    if cache is not None and not refresh:
        cached = cache.get(request, bundle=bundle)
        if cached is not None:
            return cached

    budgets = allocate_facet_budget(
        bundle, limit=request.character_limit, budget_ratio=settings.budget_ratio
    )

    started = clock()
    best: tuple[list[SentenceCandidate], list[DroppedSentence], Any] | None = None
    repair_reasons: list[str] = []
    attempts = 0
    last_error: ReportGenerationError | None = None

    for attempt in range(1, settings.max_repairs + 2):
        attempts = attempt
        payload = _completion_request(
            bundle,
            budgets,
            settings,
            repair=tuple(repair_reasons) if repair_reasons else None,
            attempt=attempt,
        )
        try:
            response = completions.create(refresh=refresh and attempt == 1, **payload)
            candidates = parse_sentence_candidates(response, bundle)
        except MalformedModelOutputError as error:
            last_error = error
            repair_reasons = ["the response was not valid structured output"]
            continue

        model_id = getattr(response, "model", None)
        if not isinstance(model_id, str) or not model_id.strip():
            last_error = ReportGenerationError("provider did not report a model identity")
            repair_reasons = ["the response omitted its model identity"]
            continue

        kept, dropped = _screen_candidates(candidates, bundle, segmenter=segmenter)
        if best is None or len(kept) > len(best[0]):
            best = (kept, dropped, response)
        if kept and not dropped:
            break
        repair_reasons = [_REPAIR_HINTS[item.reason] for item in dropped] or [
            "the draft contained no usable sentences"
        ]

    if best is None or not best[0]:
        raise last_error or ReportGenerationError(
            f"no valid sentences were produced for topic {request.topic_id}"
        )

    kept, dropped, response = best
    texts = [candidate.text for candidate in kept]
    try:
        keep_indices, packed_out = pack_report_sentences(
            texts, limit=request.character_limit
        )
    except ValueError as error:
        raise ReportGenerationError(f"report could not be packed: {error}") from None

    dropped = list(dropped)
    for index in packed_out:
        dropped.append(
            DroppedSentence.create(
                index=index + 1, reason="over_limit_pack", text=texts[index]
            )
        )

    sentences: list[ReportSentence] = []
    for position, index in enumerate(keep_indices, start=1):
        candidate = kept[index]
        citations = (
            rank_citations(candidate, bundle, max_citations=settings.max_citations)
            if candidate.substantive
            else ()
        )
        sentences.append(
            ReportSentence(
                index=position,
                text=candidate.text,
                substantive=candidate.substantive,
                citations=citations,
                facet_ids=candidate.facet_ids,
            )
        )

    finished = clock()
    report = TopicReport.create(
        request=request,
        sentences=sentences,
        created_at=datetime.fromtimestamp(finished, tz=timezone.utc).isoformat(),
        diagnostics=ReportDiagnostics(
            model_id=getattr(response, "model", ""),
            repair_attempts=max(0, attempts - 1),
            latency_seconds=max(0.0, float(finished) - float(started)),
            usage=_usage_mapping(response),
            dropped=tuple(dropped),
            llm_cache_hit=bool(llm_cache_probe()) if llm_cache_probe else False,
        ),
    )

    try:
        validate_report(request, report, bundle=bundle, segmenter=segmenter)
    except ReportValidationError as error:
        raise ReportGenerationError(f"generated report is invalid: {error}") from None

    if cache is None:
        return report
    return cache.put(request, report, bundle=bundle)


_REPAIR_HINTS = {
    "unknown_handle": "a sentence cited a source handle that was not in the SOURCES list",
    "not_single_sentence": "a sentence contained more than one sentence",
    "missing_citation": "a factual sentence listed no supporting source",
    "empty_text": "a sentence was empty",
    "duplicate_text": "a sentence repeated an earlier sentence",
    "not_nfkc": "a sentence used non-normalized text",
    "over_limit_pack": "the report exceeded the character budget",
    "citation_cap": "a sentence listed too many sources",
    "not_citable": "a sentence cited a source that only partially supports its claim",
    "unsupported_claim": "a sentence was not supported by its cited source",
    "handle_mismatch": "a citation did not resolve to its source",
}


__all__ = [
    "ChatCompletionsLike",
    "FacetBudget",
    "MalformedModelOutputError",
    "ObservedCacheStore",
    "ReportGenerationError",
    "SentenceCandidate",
    "allocate_facet_budget",
    "build_messages",
    "build_response_format",
    "generate_topic_report",
    "parse_sentence_candidates",
    "rank_citations",
]
