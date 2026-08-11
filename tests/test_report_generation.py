from __future__ import annotations

import json
import unittest
from typing import Any

from openai.types.chat import ChatCompletion

from trec_ragtime.evidence_bundle import (
    EvidenceBundle,
    EvidenceBundleSource,
    EvidenceFacet,
    EvidencePassage,
)
from trec_ragtime.report_cache import (
    GenerationSettings,
    ReportArtifactCache,
    ReportRequestIdentity,
)
from trec_ragtime.report_generation import (
    MalformedModelOutputError,
    ObservedCacheStore,
    ReportGenerationError,
    allocate_facet_budget,
    build_messages,
    build_response_format,
    generate_topic_report,
    parse_sentence_candidates,
    rank_citations,
)
from trec_ragtime.topic_candidates import Topic
from tests.test_report_cache import MemoryArtifactStore


def topic(limit: int = 5000) -> Topic:
    return Topic(
        topic_id="2000",
        collection_id="ragtime/2/all",
        problem_statement="A statement about vaping law.",
        background="A legislative aide needs this.",
        limit=limit,
        title="Topic 2000",
    )


def bundle(*, limit: int = 5000, passages: int = 3, evidence_chars: int = 0) -> EvidenceBundle:
    return EvidenceBundle.create(
        topic_id="2000",
        collection_id="ragtime/2/all",
        title="Topic 2000",
        problem_statement="A statement about vaping law.",
        background="A legislative aide needs this.",
        character_limit=limit,
        max_passages=max(passages, 1),
        evidence_chars=evidence_chars,
        facets=(
            EvidenceFacet(facet_id="F1", text="dates and countries", source="query_baseline_facet_v1"),
            EvidenceFacet(facet_id="F2", text="youth outcomes", source="query_baseline_facet_v1"),
        ),
        passages=tuple(
            EvidencePassage.create(
                handle=f"S{index}",
                document_id=f"doc-{index}",
                report_text=f"Evidence body {index} about the ban and its outcomes.",
                language="eng",
                url=f"https://example.test/{index}",
                date="2022-09-21T06:34:00.000Z",
                rank=index,
                retrieval_score=1.0 / index,
                facet_ids=("F1",) if index % 2 else ("F2",),
            )
            for index in range(1, passages + 1)
        ),
        source=EvidenceBundleSource(
            run_schema="query_baseline_v1",
            run_id="baseline-c-test",
            run_dir_name="baseline-c-test",
            manifest_sha256="a" * 64,
            records_sha256="b" * 64,
            topics_sha256="c" * 64,
        ),
    )


def settings(**overrides: Any) -> GenerationSettings:
    values: dict[str, Any] = {
        "base_url": "https://api.example.test/v1",
        "model": "test-model",
        "temperature": 0.0,
        "top_p": None,
        "max_output_tokens": 4000,
        "seed": 7,
        "max_repairs": 1,
        "max_citations": 3,
        "max_passages": 3,
        "evidence_chars": 0,
        "budget_ratio": 0.9,
    }
    values.update(overrides)
    return GenerationSettings(**values)


def identity(item: EvidenceBundle | None = None, **overrides: Any) -> ReportRequestIdentity:
    item = item or bundle()
    return ReportRequestIdentity.for_bundle(
        item,
        topic=overrides.pop("topic", topic(limit=item.character_limit)),
        settings=overrides.pop("settings", settings()),
    )


def completion(
    payload: Any,
    *,
    finish_reason: str = "stop",
    refusal: str | None = None,
    model: str = "test-model",
) -> ChatCompletion:
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return ChatCompletion.model_validate(
        {
            "id": "cmpl-1",
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "index": 0,
                    "logprobs": None,
                    "message": {"content": content, "refusal": refusal, "role": "assistant"},
                }
            ],
            "created": 1,
            "model": model,
            "object": "chat.completion",
            "usage": {"completion_tokens": 2, "prompt_tokens": 3, "total_tokens": 5},
        }
    )


def draft(*sentences: dict[str, Any]) -> dict[str, Any]:
    return {"sentences": list(sentences)}


def line(text: str, evidence: list[str], facets: list[str] | None = None, substantive: bool = True):
    return {
        "text": text,
        "evidence": evidence,
        "facets": facets if facets is not None else ["F1"],
        "substantive": substantive,
    }


GOOD_DRAFT = draft(
    line("The ban took effect in 2019.", ["S1"]),
    line("Youth vaping prevalence fell sharply.", ["S2", "S1"], ["F2"]),
)


class ScriptedCompletions:
    """Replays queued completions and records every request it received."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def create(self, *, refresh: bool = False, **request: Any) -> Any:
        self.requests.append({"refresh": refresh, **request})
        if not self.responses:
            raise AssertionError("provider called more times than scripted")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class TripwireCompletions:
    def create(self, *, refresh: bool = False, **request: Any) -> Any:
        raise AssertionError("provider called")


class BudgetTests(unittest.TestCase):
    def test_budget_is_split_across_facets_within_the_ratio(self) -> None:
        budgets = allocate_facet_budget(bundle(), limit=1000, budget_ratio=0.9)
        self.assertEqual([item.facet_id for item in budgets], ["F1", "F2"])
        self.assertEqual(sum(item.characters for item in budgets), 900)

    def test_remainder_goes_to_the_earliest_facets(self) -> None:
        budgets = allocate_facet_budget(bundle(), limit=1001, budget_ratio=1.0)
        self.assertEqual([item.characters for item in budgets], [501, 500])

    def test_every_facet_receives_a_positive_budget(self) -> None:
        budgets = allocate_facet_budget(bundle(), limit=3, budget_ratio=1.0)
        self.assertTrue(all(item.characters >= 1 for item in budgets))


class ResponseFormatTests(unittest.TestCase):
    def test_json_schema_inlines_the_handle_and_facet_allowlists(self) -> None:
        payload = build_response_format(bundle(passages=2), settings())
        self.assertEqual(payload["type"], "json_schema")
        schema = payload["json_schema"]
        self.assertTrue(schema["strict"])
        item = schema["schema"]["properties"]["sentences"]["items"]
        self.assertEqual(item["properties"]["evidence"]["items"]["enum"], ["S1", "S2"])
        self.assertEqual(item["properties"]["facets"]["items"]["enum"], ["F1", "F2"])
        self.assertFalse(schema["schema"]["additionalProperties"])

    def test_json_object_mode_is_available_without_a_silent_downgrade(self) -> None:
        payload = build_response_format(bundle(), settings(response_format_mode="json_object"))
        self.assertEqual(payload, {"type": "json_object"})


class MessageTests(unittest.TestCase):
    def test_messages_carry_the_request_handles_and_budget(self) -> None:
        item = bundle(passages=2)
        budgets = allocate_facet_budget(item, limit=5000, budget_ratio=0.9)
        messages = build_messages(item, budgets, settings())
        self.assertEqual([entry["role"] for entry in messages], ["system", "user"])
        body = messages[1]["content"]
        self.assertIn("A statement about vaping law.", body)
        self.assertIn("A legislative aide needs this.", body)
        self.assertIn("[S1]", body)
        self.assertIn("[S2]", body)
        self.assertIn("Evidence body 1", body)
        self.assertIn("F1", body)

    def test_prompt_never_exposes_canonical_document_ids(self) -> None:
        item = bundle(passages=3)
        budgets = allocate_facet_budget(item, limit=5000, budget_ratio=0.9)
        body = "".join(entry["content"] for entry in build_messages(item, budgets, settings()))
        for document_id in item.document_ids:
            self.assertNotIn(document_id, body)

    def test_evidence_chars_bounds_the_prompt_excerpt(self) -> None:
        item = bundle(passages=1, evidence_chars=12)
        budgets = allocate_facet_budget(item, limit=5000, budget_ratio=0.9)
        body = build_messages(item, budgets, settings(evidence_chars=12))[1]["content"]
        self.assertNotIn("about the ban and its outcomes", body)

    def test_repair_message_is_appended_and_varies_by_attempt(self) -> None:
        item = bundle()
        budgets = allocate_facet_budget(item, limit=5000, budget_ratio=0.9)
        first = build_messages(item, budgets, settings(), repair=("missing_citation",), attempt=2)
        second = build_messages(item, budgets, settings(), repair=("missing_citation",), attempt=3)
        self.assertEqual(len(first), 3)
        self.assertNotEqual(first[2]["content"], second[2]["content"])


class ParseTests(unittest.TestCase):
    def test_valid_payload_parses(self) -> None:
        candidates = parse_sentence_candidates(completion(GOOD_DRAFT), bundle())
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0].handles, ("S1",))
        self.assertEqual(candidates[1].handles, ("S2", "S1"))

    def test_fenced_json_is_unwrapped(self) -> None:
        fenced = "```json\n" + json.dumps(GOOD_DRAFT) + "\n```"
        self.assertEqual(len(parse_sentence_candidates(completion(fenced), bundle())), 2)

    def test_refusal_is_malformed_output(self) -> None:
        with self.assertRaises(MalformedModelOutputError):
            parse_sentence_candidates(completion(GOOD_DRAFT, refusal="no"), bundle())

    def test_truncated_completion_is_malformed_output(self) -> None:
        with self.assertRaises(MalformedModelOutputError):
            parse_sentence_candidates(completion(GOOD_DRAFT, finish_reason="length"), bundle())

    def test_invalid_json_is_malformed_output(self) -> None:
        with self.assertRaises(MalformedModelOutputError):
            parse_sentence_candidates(completion("{not json"), bundle())

    def test_missing_sentences_key_is_malformed_output(self) -> None:
        with self.assertRaises(MalformedModelOutputError):
            parse_sentence_candidates(completion({"lines": []}), bundle())

    def test_empty_sentence_array_is_malformed_output(self) -> None:
        with self.assertRaises(MalformedModelOutputError):
            parse_sentence_candidates(completion(draft()), bundle())


class RankingTests(unittest.TestCase):
    def test_first_listed_source_receives_the_highest_score(self) -> None:
        item = bundle()
        (candidate,) = parse_sentence_candidates(
            completion(draft(line("A claim.", ["S2", "S1"]))), item
        )
        citations = rank_citations(candidate, item, max_citations=3)
        self.assertEqual([c.handle for c in citations], ["S2", "S1"])
        self.assertGreater(citations[0].score, citations[1].score)

    def test_scores_strictly_descend_and_are_finite(self) -> None:
        import math

        item = bundle(passages=3)
        (candidate,) = parse_sentence_candidates(
            completion(draft(line("A claim.", ["S1", "S2", "S3"]))), item
        )
        scores = [c.score for c in rank_citations(candidate, item, max_citations=3)]
        self.assertTrue(all(math.isfinite(score) for score in scores))
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(len(set(scores)), len(scores))

    def test_citations_resolve_to_canonical_document_ids(self) -> None:
        item = bundle()
        (candidate,) = parse_sentence_candidates(completion(draft(line("A claim.", ["S1"]))), item)
        citations = rank_citations(candidate, item, max_citations=3)
        self.assertEqual(citations[0].document_id, "doc-1")

    def test_citation_cap_truncates_to_the_strongest(self) -> None:
        item = bundle(passages=3)
        (candidate,) = parse_sentence_candidates(
            completion(draft(line("A claim.", ["S1", "S2", "S3"]))), item
        )
        self.assertEqual(
            [c.handle for c in rank_citations(candidate, item, max_citations=2)], ["S1", "S2"]
        )


class GenerateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = MemoryArtifactStore()
        self.cache = ReportArtifactCache(self.store, clock=lambda: 1_700_000_000.0)
        self.bundle = bundle()
        self.request = identity(self.bundle)

    def run_generate(self, responses: list[Any], **overrides: Any):
        completions = ScriptedCompletions(responses)
        report = generate_topic_report(
            bundle=overrides.pop("bundle", self.bundle),
            request=overrides.pop("request", self.request),
            completions=completions,
            cache=overrides.pop("cache", self.cache),
            clock=lambda: 1_700_000_000.0,
            **overrides,
        )
        return report, completions

    def test_happy_path_produces_a_validated_cached_report(self) -> None:
        report, completions = self.run_generate([completion(GOOD_DRAFT)])
        self.assertEqual(len(report.sentences), 2)
        self.assertEqual(report.references, ("doc-1", "doc-2"))
        self.assertEqual(report.sentences[0].citations[0].document_id, "doc-1")
        self.assertIsNotNone(report.report_artifact)
        self.assertEqual(report.diagnostics.model_id, "test-model")
        self.assertEqual(report.diagnostics.usage["total_tokens"], 5)
        self.assertEqual(len(completions.requests), 1)
        self.assertEqual(completions.requests[0]["model"], "test-model")
        self.assertEqual(completions.requests[0]["temperature"], 0.0)
        self.assertEqual(completions.requests[0]["seed"], 7)

    def test_second_call_is_served_from_the_report_cache(self) -> None:
        self.run_generate([completion(GOOD_DRAFT)])
        report = generate_topic_report(
            bundle=self.bundle,
            request=self.request,
            completions=TripwireCompletions(),
            cache=self.cache,
            clock=lambda: 0.0,
        )
        self.assertTrue(report.report_artifact.cache_hit)
        self.assertEqual(len(report.sentences), 2)

    def test_refresh_bypasses_the_report_cache(self) -> None:
        self.run_generate([completion(GOOD_DRAFT)])
        report, completions = self.run_generate([completion(GOOD_DRAFT)], refresh=True)
        self.assertEqual(len(completions.requests), 1)
        self.assertTrue(completions.requests[0]["refresh"])
        self.assertEqual(len(report.sentences), 2)

    def test_unknown_handle_sentence_is_dropped(self) -> None:
        payload = draft(
            line("A supported claim.", ["S1"]),
            line("An invented claim.", ["S9"]),
        )
        report, _ = self.run_generate(
            [completion(payload), completion(payload)],
            request=identity(self.bundle, settings=settings(max_repairs=1)),
        )
        self.assertEqual([item.text for item in report.sentences], ["A supported claim."])
        reasons = {item.reason for item in report.diagnostics.dropped}
        self.assertIn("unknown_handle", reasons)

    def test_multi_sentence_candidate_is_dropped_not_resplit(self) -> None:
        payload = draft(
            line("A supported claim.", ["S1"]),
            line("Two claims here. And another one.", ["S2"]),
        )
        report, _ = self.run_generate([completion(payload), completion(payload)])
        self.assertEqual(len(report.sentences), 1)
        self.assertIn(
            "not_single_sentence", {item.reason for item in report.diagnostics.dropped}
        )

    def test_substantive_sentence_without_evidence_is_dropped(self) -> None:
        payload = draft(
            line("A supported claim.", ["S1"]),
            line("An uncited claim.", []),
        )
        report, _ = self.run_generate([completion(payload), completion(payload)])
        self.assertEqual(len(report.sentences), 1)
        self.assertIn("missing_citation", {item.reason for item in report.diagnostics.dropped})

    def test_non_substantive_sentence_is_kept_without_citations(self) -> None:
        payload = draft(
            line("This report covers two themes.", [], substantive=False),
            line("A supported claim.", ["S1"]),
        )
        report, _ = self.run_generate([completion(payload)])
        self.assertEqual(len(report.sentences), 2)
        self.assertEqual(report.sentences[0].citations, ())
        self.assertFalse(report.sentences[0].substantive)

    def test_duplicate_sentences_are_dropped(self) -> None:
        payload = draft(
            line("A supported claim.", ["S1"]),
            line("A supported claim.", ["S2"]),
        )
        report, _ = self.run_generate([completion(payload), completion(payload)])
        self.assertEqual(len(report.sentences), 1)
        self.assertIn("duplicate_text", {item.reason for item in report.diagnostics.dropped})

    def test_conflicting_evidence_stays_as_two_attributed_sentences(self) -> None:
        payload = draft(
            line("The ministry reported a decline.", ["S1"]),
            line("The regulator reported no change.", ["S2"], ["F2"]),
        )
        report, _ = self.run_generate([completion(payload)])
        self.assertEqual(len(report.sentences), 2)
        self.assertEqual(report.sentences[0].citations[0].document_id, "doc-1")
        self.assertEqual(report.sentences[1].citations[0].document_id, "doc-2")

    def test_repair_pass_runs_and_is_counted(self) -> None:
        broken = draft(line("An invented claim.", ["S9"]), line("A claim.", ["S1"]))
        fixed = draft(line("A supported claim.", ["S1"]), line("Another supported claim.", ["S2"], ["F2"]))
        report, completions = self.run_generate([completion(broken), completion(fixed)])
        self.assertEqual(len(completions.requests), 2)
        self.assertEqual(report.diagnostics.repair_attempts, 1)
        self.assertEqual(len(report.sentences), 2)

    def test_repair_is_a_distinct_request_so_it_cannot_replay(self) -> None:
        broken = draft(line("An invented claim.", ["S9"]), line("A claim.", ["S1"]))
        _, completions = self.run_generate([completion(broken), completion(broken)])
        first, second = completions.requests
        self.assertNotEqual(first["messages"], second["messages"])

    def test_repairs_are_bounded(self) -> None:
        broken = draft(line("An invented claim.", ["S9"]), line("A claim.", ["S1"]))
        request = identity(self.bundle, settings=settings(max_repairs=0))
        report, completions = self.run_generate([completion(broken)], request=request)
        self.assertEqual(len(completions.requests), 1)
        self.assertEqual(report.diagnostics.repair_attempts, 0)

    def test_best_attempt_is_kept_when_repair_is_worse(self) -> None:
        good = draft(
            line("A supported claim.", ["S1"]),
            line("Another supported claim.", ["S2"], ["F2"]),
            line("An invented claim.", ["S9"]),
        )
        worse = draft(line("A single claim.", ["S1"]))
        report, _ = self.run_generate([completion(good), completion(worse)])
        self.assertEqual(len(report.sentences), 2)

    def test_over_limit_report_is_packed_to_whole_sentences(self) -> None:
        small = bundle(limit=40)
        request = identity(small, topic=topic(limit=40))
        payload = draft(
            line("The ban took effect in 2019.", ["S1"]),
            line("Youth vaping prevalence fell sharply.", ["S2"], ["F2"]),
        )
        report, _ = self.run_generate(
            [completion(payload)], bundle=small, request=request
        )
        self.assertEqual([item.text for item in report.sentences], ["The ban took effect in 2019."])
        self.assertLessEqual(report.budgeted_character_count, 40)
        self.assertIn("over_limit_pack", {item.reason for item in report.diagnostics.dropped})

    def test_all_candidates_invalid_raises_and_caches_nothing(self) -> None:
        payload = draft(line("An invented claim.", ["S9"]))
        with self.assertRaises(ReportGenerationError):
            self.run_generate([completion(payload), completion(payload)])
        self.assertEqual(self.store.write_calls, [])

    def test_malformed_output_is_never_cached(self) -> None:
        with self.assertRaises(ReportGenerationError):
            self.run_generate([completion("{bad"), completion("{bad")])
        self.assertEqual(self.store.write_calls, [])

    def test_refusal_is_never_cached(self) -> None:
        with self.assertRaises(ReportGenerationError):
            self.run_generate(
                [completion(GOOD_DRAFT, refusal="no"), completion(GOOD_DRAFT, refusal="no")]
            )
        self.assertEqual(self.store.write_calls, [])

    def test_truncated_completion_is_never_cached(self) -> None:
        with self.assertRaises(ReportGenerationError):
            self.run_generate(
                [
                    completion(GOOD_DRAFT, finish_reason="length"),
                    completion(GOOD_DRAFT, finish_reason="length"),
                ]
            )
        self.assertEqual(self.store.write_calls, [])

    def test_blank_model_identity_is_rejected(self) -> None:
        # A missing model identity is retried like any other bad response, then
        # fails closed rather than being stored.
        with self.assertRaises(ReportGenerationError):
            self.run_generate(
                [completion(GOOD_DRAFT, model=""), completion(GOOD_DRAFT, model="")]
            )
        self.assertEqual(self.store.write_calls, [])

    def test_bundle_must_match_the_request(self) -> None:
        with self.assertRaises(ReportGenerationError):
            self.run_generate([completion(GOOD_DRAFT)], bundle=bundle(evidence_chars=99))

    def test_artifact_never_contains_prompt_or_evidence_text(self) -> None:
        self.run_generate([completion(GOOD_DRAFT)])
        payload = self.store.objects[self.request.object_key]
        self.assertNotIn(b"Evidence body 1", payload)
        self.assertNotIn(b"A legislative aide", payload)


class ObservedCacheStoreTests(unittest.TestCase):
    def test_hit_flag_tracks_the_underlying_read(self) -> None:
        inner = MemoryArtifactStore()
        observed = ObservedCacheStore(inner)
        self.assertIsNone(observed.read_validated("k", lambda payload: payload))
        self.assertFalse(observed.last_read_hit)
        observed.write("k", b"\x01")
        self.assertFalse(observed.last_read_hit)
        self.assertIsNotNone(observed.read_validated("k", lambda payload: payload))
        self.assertTrue(observed.last_read_hit)

    def test_unknown_attributes_delegate(self) -> None:
        inner = MemoryArtifactStore()
        inner.namespace = "ns"  # type: ignore[attr-defined]
        self.assertEqual(ObservedCacheStore(inner).namespace, "ns")


if __name__ == "__main__":
    unittest.main()
