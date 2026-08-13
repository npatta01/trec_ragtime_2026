from __future__ import annotations

import json
import unittest
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

from trec_ragtime.planner_static import (
    AsymmetricStaticPlanner,
    CoverageLedger,
    CoverageLedgerArtifactCache,
    CoverageLedgerCacheEntry,
    CoverageExpansionArtifactCache,
    CoverageExpansionCacheEntry,
    PlanArtifactCache,
    StaticTopicEvidence,
    PlanningTopic,
    OpenAIStaticPlannerModel,
    PlannerResponse,
    StaticPlanner,
    StaticResearchPlan,
    allocate_tasks,
    coverage_ledger_from_record,
    parse_coverage_ledger,
    parse_compact_query_expansion,
    parse_static_plan,
    static_plan_from_record,
    static_plan_to_record,
    static_topic_evidence_from_record,
    load_baseline_query_budgets,
    run_planner_dry_run,
    run_static_topics,
    select_topics_by_id,
    TopRankedSentenceReader,
    execute_static_plan,
    ChunkedPassageReader,
)
from trec_ragtime import PipelineResult, RetrievalResult, RetrievedDocument, SearchHit
from trec_ragtime.passages import Passage, sha256_bytes


def _topic(topic_id: str = "2000") -> PlanningTopic:
    return PlanningTopic(
        topic_id=topic_id,
        title="A topic",
        problem_statement="Explain the topic and its evidence.",
        background="The report needs comparisons, causes, and outcomes.",
    )


def _payload(*, duplicate: bool = False) -> dict[str, Any]:
    needs: list[dict[str, Any]] = [
        {
            "need_ref": "context",
            "objective": "Establish the historical context.",
            "why_it_matters": "The report needs a defensible baseline.",
            "completion_guidance": "Cover dates, actors, and scope.",
            "priority": 1,
            "search_tasks": [
                {
                    "task_ref": "context-1",
                    "query": "topic historical origins timeline",
                    "evidence_target": "dated history and origin evidence",
                    "priority": 1,
                    "depends_on": [],
                    "requirement_refs": ["history"],
                    "query_role": "broad_discovery",
                },
                {
                    "task_ref": "context-2",
                    "query": "topic historical actors institutions",
                    "evidence_target": "actors and institutional evidence",
                    "priority": 2,
                    "depends_on": ["context-1"],
                    "requirement_refs": ["history"],
                    "query_role": "precise_facts",
                },
            ],
        },
        {
            "need_ref": "outcomes",
            "objective": "Measure outcomes and limitations.",
            "why_it_matters": "The report should distinguish claims from results.",
            "completion_guidance": "Prefer measured outcomes and caveats.",
            "priority": 2,
            "search_tasks": [
                {
                    "task_ref": "outcomes-1",
                    "query": "topic measurable outcomes evaluation evidence",
                    "evidence_target": "measured outcomes and evaluation evidence",
                    "priority": 1,
                    "depends_on": [],
                    "requirement_refs": ["outcomes"],
                    "query_role": "measurable_outcomes",
                },
                {
                    "task_ref": "outcomes-2",
                    "query": "topic limitations risks conflicting evidence",
                    "evidence_target": "limitations, risks, and disagreement",
                    "priority": 2,
                    "depends_on": ["outcomes-1"],
                    "requirement_refs": ["outcomes"],
                    "query_role": "contrary_evidence",
                },
            ],
        },
    ]
    if duplicate:
        needs[1]["search_tasks"][0]["query"] = needs[0]["search_tasks"][0]["query"]
    return {
        "requirements": [
            {
                "requirement_ref": "history",
                "description": "Explain the dated historical context.",
                "priority": 1,
            },
            {
                "requirement_ref": "outcomes",
                "description": "Report measured outcomes and limitations.",
                "priority": 2,
            },
        ],
        "information_needs": needs,
    }


def _coverage_payload() -> dict[str, Any]:
    return {
        "requirements": [
            {
                "requirement_ref": "history",
                "description": "Explain the dated historical context.",
                "priority": 1,
            },
            {
                "requirement_ref": "outcomes",
                "description": "Report measured outcomes and limitations.",
                "priority": 2,
            },
        ],
        "information_needs": [
            {
                "need_ref": "context",
                "objective": "Establish the historical context.",
                "why_it_matters": "The report needs a defensible baseline.",
                "completion_guidance": "Cover dates, actors, and scope.",
                "priority": 1,
                "search_tasks": [
                    {
                        "task_ref": "context-1",
                        "query": "topic historical origins timeline",
                        "evidence_target": "dated history and origin evidence",
                        "priority": 1,
                        "depends_on": [],
                        "requirement_refs": ["history"],
                        "query_role": "broad_discovery",
                    }
                ],
            },
            {
                "need_ref": "outcomes",
                "objective": "Measure outcomes and limitations.",
                "why_it_matters": "The report should distinguish claims from results.",
                "completion_guidance": "Prefer measured outcomes and caveats.",
                "priority": 2,
                "search_tasks": [
                    {
                        "task_ref": "outcomes-1",
                        "query": "topic measurable outcomes evaluation evidence",
                        "evidence_target": "measured outcomes and evaluation evidence",
                        "priority": 1,
                        "depends_on": [],
                        "requirement_refs": ["outcomes"],
                        "query_role": "measurable_outcomes",
                    }
                ],
            },
        ],
    }


def _ledger_payload() -> dict[str, Any]:
    return {
        "information_needs": [
            {
                "need_ref": "context",
                "objective": "Establish the historical context.",
                "why_it_matters": "The report needs a defensible baseline.",
                "priority": 1,
                "requirements": [
                    {
                        "requirement_ref": "history",
                        "description": "Explain the dated historical context.",
                        "priority": 1,
                        "retrieval_anchors": [
                            "dated timeline",
                            "named actors and institutions",
                        ],
                    }
                ],
            },
            {
                "need_ref": "outcomes",
                "objective": "Measure outcomes and limitations.",
                "why_it_matters": "The report should distinguish claims from results.",
                "priority": 2,
                "requirements": [
                    {
                        "requirement_ref": "outcomes",
                        "description": "Report measured outcomes and limitations.",
                        "priority": 1,
                        "retrieval_anchors": [
                            "measured evaluations",
                            "contrary findings",
                        ],
                    }
                ],
            },
        ]
    }


def _compact_expansion_payload() -> dict[str, Any]:
    return {
        "query_sets": [
            {
                "requirement_ref": "history",
                "queries": [
                    {
                        "query": "topic historical origins timeline",
                        "query_role": "broad_discovery",
                        "anchor_dimension": "dated timeline",
                    },
                    {
                        "query": "topic historical actors institutions",
                        "query_role": "precise_facts",
                        "anchor_dimension": "named actors and institutions",
                    },
                ],
            },
            {
                "requirement_ref": "outcomes",
                "queries": [
                    {
                        "query": "topic measurable outcomes evaluation evidence",
                        "query_role": "measurable_outcomes",
                        "anchor_dimension": "measured evaluations",
                    },
                    {
                        "query": "topic limitations risks conflicting evidence",
                        "query_role": "contrary_evidence",
                        "anchor_dimension": "contrary findings",
                    },
                ],
            },
        ]
    }


def _many_task_payload(count: int = 25) -> dict[str, Any]:
    payload = _coverage_payload()
    payload["requirements"] = [
        {
            "requirement_ref": "scope",
            "description": "Cover every distinct part of the requested scope.",
            "priority": 1,
        }
    ]
    payload["information_needs"] = [
        {
            "need_ref": "scope",
            "objective": "Cover the complete scope.",
            "why_it_matters": "The richer arm must retain every useful task.",
            "completion_guidance": "Use distinct non-overlapping targets.",
            "priority": 1,
            "search_tasks": [
                {
                    "task_ref": f"scope-{index}",
                    "query": f"topic distinct evidence target {index}",
                    "evidence_target": f"distinct evidence target {index}",
                    "priority": index,
                    "depends_on": [],
                    "requirement_refs": ["scope"],
                    "query_role": "precise_facts",
                }
                for index in range(1, count + 1)
            ],
        }
    ]
    return payload


class MemoryPlanCache(PlanArtifactCache):
    def __init__(self) -> None:
        self.values: dict[str, StaticResearchPlan] = {}
        self.put_calls = 0

    def get(self, key: str) -> StaticResearchPlan | None:
        return self.values.get(key)

    def put(self, key: str, plan: StaticResearchPlan) -> bool:
        self.put_calls += 1
        self.values[key] = plan
        return True


class MemoryCoverageLedgerCache(CoverageLedgerArtifactCache):
    def __init__(self) -> None:
        self.values: dict[str, CoverageLedgerCacheEntry] = {}
        self.put_calls = 0

    def get(self, key: str) -> CoverageLedgerCacheEntry | None:
        return self.values.get(key)

    def put(self, key: str, entry: CoverageLedgerCacheEntry) -> bool:
        self.put_calls += 1
        self.values[key] = entry
        return True


class MemoryCoverageExpansionCache(CoverageExpansionArtifactCache):
    def __init__(self) -> None:
        self.values: dict[str, CoverageExpansionCacheEntry] = {}
        self.put_calls = 0

    def get(self, key: str) -> CoverageExpansionCacheEntry | None:
        return self.values.get(key)

    def put(self, key: str, entry: CoverageExpansionCacheEntry) -> bool:
        self.put_calls += 1
        self.values[key] = entry
        return True


@dataclass
class FakePlannerModel:
    payload: dict[str, Any]
    calls: int = 0

    def generate(
        self, topic: PlanningTopic, *, max_total_tasks: int | None
    ) -> PlannerResponse:
        self.calls += 1
        return PlannerResponse(
            payload=self.payload,
            model="fake-planner",
            prompt_version="planner-static-test-v1",
            input_tokens=11,
            output_tokens=22,
            cost=0.001,
            latency_seconds=0.25,
        )


@dataclass
class FakePlannerAuditor:
    payload: dict[str, Any]
    calls: int = 0
    received_draft: StaticResearchPlan | None = None

    def audit(
        self,
        topic: PlanningTopic,
        draft: StaticResearchPlan,
        *,
        max_total_tasks: int | None,
    ) -> PlannerResponse:
        self.calls += 1
        self.received_draft = draft
        return PlannerResponse(
            payload=self.payload,
            model="fake-auditor",
            prompt_version="planner-static-audit-test-v2",
            input_tokens=13,
            output_tokens=17,
            cost=0.002,
            latency_seconds=0.5,
        )


@dataclass
class FakeCoveragePlannerModel:
    ledger_payload: dict[str, Any]
    final_payload: dict[str, Any]
    ledger_calls: int = 0
    expansion_calls: int = 0
    received_ledger: CoverageLedger | None = None
    ledger_output_tokens: int = 11
    expansion_output_tokens: int = 17
    expansion_finish_reason: str = "stop"

    def generate_coverage_ledger(self, topic: PlanningTopic) -> PlannerResponse:
        self.ledger_calls += 1
        return PlannerResponse(
            payload=self.ledger_payload,
            model="fake-planner",
            prompt_version="planner-static-ledger-test-v1",
            input_tokens=7,
            output_tokens=self.ledger_output_tokens,
            cost=0.001,
            latency_seconds=0.2,
            finish_reason="stop",
        )

    def expand_coverage_ledger(
        self,
        topic: PlanningTopic,
        ledger: CoverageLedger,
        *,
        max_total_tasks: int | None,
    ) -> PlannerResponse:
        self.expansion_calls += 1
        self.received_ledger = ledger
        return PlannerResponse(
            payload=self.final_payload,
            model="fake-planner",
            prompt_version="planner-static-expand-test-v1",
            input_tokens=13,
            output_tokens=self.expansion_output_tokens,
            cost=0.002,
            latency_seconds=0.3,
            finish_reason=self.expansion_finish_reason,
        )


class PlannerStaticTests(unittest.TestCase):
    def test_compact_expansion_builds_complete_dependency_free_plan(self) -> None:
        ledger = parse_coverage_ledger(
            _topic(),
            _ledger_payload(),
            model="fake-planner",
            prompt_version="planner-static-ledger-test-v1",
        )

        plan = parse_compact_query_expansion(
            _topic(),
            ledger,
            _compact_expansion_payload(),
            model="fake-planner",
            prompt_version="planner-static-expand-v2",
        )

        self.assertEqual(plan.planning_stage, "audited")
        self.assertEqual(plan.draft_plan_sha256, ledger.ledger_sha256)
        self.assertEqual(len(plan.requirements), 2)
        self.assertEqual(len(plan.tasks), 4)
        self.assertTrue(all(not task.depends_on for task in plan.tasks))
        self.assertEqual(
            {requirement.description for requirement in plan.requirements},
            {requirement.description for requirement in ledger.requirements},
        )
        self.assertEqual(
            {task.evidence_target for task in plan.tasks},
            {
                "dated timeline",
                "named actors and institutions",
                "measured evaluations",
                "contrary findings",
            },
        )

    def test_compact_expansion_projects_only_known_schema_fields(self) -> None:
        ledger = parse_coverage_ledger(
            _topic(),
            _ledger_payload(),
            model="fake-planner",
            prompt_version="planner-static-ledger-test-v1",
        )
        payload = _compact_expansion_payload()
        payload["provider_note"] = "ignored"
        payload["query_sets"][0]["set_note"] = "ignored"
        payload["query_sets"][0]["queries"][0]["explanation"] = "ignored"

        plan = parse_compact_query_expansion(
            _topic(),
            ledger,
            payload,
            model="fake-planner",
            prompt_version="planner-static-expand-v2",
        )

        self.assertEqual(len(plan.tasks), 4)
        self.assertNotIn("ignored", json.dumps(plan.as_record()))

    def test_compact_expansion_merges_only_exact_duplicate_queries(self) -> None:
        ledger = parse_coverage_ledger(
            _topic(),
            _ledger_payload(),
            model="fake-planner",
            prompt_version="planner-static-ledger-test-v1",
        )
        payload = _compact_expansion_payload()
        payload["query_sets"][1]["queries"] = [
            {
                "query": "  TOPIC   historical origins timeline ",
                "query_role": "comparison",
                "anchor_dimension": "outcome comparison",
            },
            {
                "query": "topic historical origins timeline evidence",
                "query_role": "precise_facts",
                "anchor_dimension": "semantically overlapping variant",
            },
        ]

        plan = parse_compact_query_expansion(
            _topic(),
            ledger,
            payload,
            model="fake-planner",
            prompt_version="planner-static-expand-v2",
        )

        self.assertEqual(len(plan.tasks), 3)
        merged = next(
            task
            for task in plan.tasks
            if task.query.casefold() == "topic historical origins timeline"
        )
        self.assertEqual(len(merged.requirement_ids), 2)
        self.assertIn("outcome comparison", merged.evidence_target)
        self.assertIn(
            "topic historical origins timeline evidence",
            {task.query for task in plan.tasks},
        )
        covered = {
            requirement_id
            for task in plan.tasks
            for requirement_id in task.requirement_ids
        }
        self.assertEqual(
            covered,
            {requirement.requirement_id for requirement in plan.requirements},
        )

    def test_compact_expansion_merges_duplicates_even_if_a_need_has_no_unique_task(
        self,
    ) -> None:
        ledger = parse_coverage_ledger(
            _topic(),
            _ledger_payload(),
            model="fake-planner",
            prompt_version="planner-static-ledger-test-v1",
        )
        payload = _compact_expansion_payload()
        payload["query_sets"][1]["queries"] = [
            {
                "query": "topic historical origins timeline",
                "query_role": "comparison",
                "anchor_dimension": "outcome comparison",
            }
        ]

        plan = parse_compact_query_expansion(
            _topic(),
            ledger,
            payload,
            model="fake-planner",
            prompt_version="planner-static-expand-v2",
        )

        self.assertEqual(len(plan.tasks), 2)
        merged = next(
            task
            for task in plan.tasks
            if task.query.casefold() == "topic historical origins timeline"
        )
        self.assertEqual(len(merged.requirement_ids), 2)
        outcomes_need = next(
            need
            for need in plan.information_needs
            if need.objective == "Measure outcomes and limitations."
        )
        self.assertEqual(outcomes_need.task_ids, ())

    def test_final_expansion_falls_back_to_validated_ledger_on_local_failure(
        self,
    ) -> None:
        missing_requirement = _compact_expansion_payload()
        missing_requirement["query_sets"] = missing_requirement["query_sets"][:1]
        model = FakeCoveragePlannerModel(_ledger_payload(), missing_requirement)
        ledger_cache = MemoryCoverageLedgerCache()
        expansion_cache = MemoryCoverageExpansionCache()
        planner = AsymmetricStaticPlanner(
            model,
            ledger_cache=ledger_cache,
            expansion_cache=expansion_cache,
            expansion_prompt_version="planner-static-expand-v3",
        )

        outcome = planner.plan_topic(_topic())
        record = run_planner_dry_run([_topic()], planner)[0]

        self.assertEqual(outcome.status, "success")
        self.assertIsNotNone(outcome.plan)
        assert outcome.plan is not None
        self.assertEqual(len(outcome.plan.tasks), 6)
        self.assertIn(
            "+ledger-fallback-v1-ledger_requirement_coverage-",
            outcome.plan.prompt_version,
        )
        self.assertTrue(outcome.fallback_used)
        self.assertEqual(
            outcome.fallback_reason, "ledger_requirement_coverage"
        )
        self.assertTrue(record["fallback_used"])
        self.assertEqual(
            record["fallback_reason"], "ledger_requirement_coverage"
        )
        queries = {task.query for task in outcome.plan.tasks}
        self.assertIn("Explain the dated historical context.", queries)
        self.assertIn("A topic: dated timeline", queries)
        self.assertEqual(model.ledger_calls, 1)
        self.assertEqual(model.expansion_calls, 1)

    def test_legacy_compact_expansion_keeps_local_failure_fail_closed(self) -> None:
        missing_requirement = _compact_expansion_payload()
        missing_requirement["query_sets"] = missing_requirement["query_sets"][:1]
        model = FakeCoveragePlannerModel(_ledger_payload(), missing_requirement)
        planner = AsymmetricStaticPlanner(
            model,
            expansion_prompt_version="planner-static-expand-v2",
        )

        outcome = planner.plan_topic(_topic())

        self.assertEqual(outcome.status, "failed")
        self.assertEqual(
            outcome.validation_error, "ledger_requirement_coverage"
        )

    def test_compact_expansion_requires_every_ledger_requirement_exactly_once(self) -> None:
        ledger = parse_coverage_ledger(
            _topic(),
            _ledger_payload(),
            model="fake-planner",
            prompt_version="planner-static-ledger-test-v1",
        )
        missing = _compact_expansion_payload()
        missing["query_sets"] = missing["query_sets"][:1]

        with self.assertRaisesRegex(ValueError, "exact ledger requirement coverage"):
            parse_compact_query_expansion(
                _topic(),
                ledger,
                missing,
                model="fake-planner",
                prompt_version="planner-static-expand-v2",
            )

    def test_compact_expansion_retains_more_than_twenty_four_queries(self) -> None:
        ledger = parse_coverage_ledger(
            _topic(),
            _ledger_payload(),
            model="fake-planner",
            prompt_version="planner-static-ledger-test-v1",
        )
        payload = _compact_expansion_payload()
        payload["query_sets"][0]["queries"] = [
            {
                "query": f"topic distinct historical retrieval anchor {index}",
                "query_role": "precise_facts",
                "anchor_dimension": f"historical anchor {index}",
            }
            for index in range(1, 26)
        ]

        plan = parse_compact_query_expansion(
            _topic(),
            ledger,
            payload,
            model="fake-planner",
            prompt_version="planner-static-expand-v2",
        )

        self.assertEqual(len(plan.tasks), 27)
        self.assertEqual(plan.truncated_task_count, 0)

    def test_asymmetric_planner_uses_compact_v2_parser_end_to_end(self) -> None:
        model = FakeCoveragePlannerModel(
            _ledger_payload(), _compact_expansion_payload()
        )
        planner = AsymmetricStaticPlanner(
            model,
            expansion_prompt_version="planner-static-expand-v2",
        )

        outcome = planner.plan_topic(_topic())

        self.assertEqual(outcome.status, "success")
        self.assertIsNotNone(outcome.plan)
        assert outcome.plan is not None
        self.assertEqual(len(outcome.plan.tasks), 4)
        self.assertTrue(all(not task.depends_on for task in outcome.plan.tasks))
        self.assertEqual(
            outcome.plan.prompt_version,
            "planner-static-expand-test-v1+compact-query-assembler-v2",
        )

    def test_fenced_or_prefixed_json_is_extracted_in_memory(self) -> None:
        class FencedCompletions:
            def create(self, **request: Any) -> Any:
                content = "Use {braces} here. JSON follows:\n```json\n" + json.dumps(
                    _ledger_payload()
                ) + "\n```"
                return SimpleNamespace(
                    choices=(
                        SimpleNamespace(
                            finish_reason="stop",
                            message=SimpleNamespace(content=content),
                        ),
                    ),
                    usage=SimpleNamespace(
                        prompt_tokens=10,
                        completion_tokens=20,
                        cost=0.001,
                    ),
                )

        provider = OpenAIStaticPlannerModel(
            SimpleNamespace(chat=SimpleNamespace(completions=FencedCompletions())),
            model="fake-provider",
        )

        response = provider.generate_coverage_ledger(_topic())

        self.assertEqual(response.payload, _ledger_payload())
        self.assertNotIn("Use {braces}", repr(response))

    def test_ledger_v2_repairs_invalid_json_once_and_aggregates_metrics(self) -> None:
        raw_invalid = "PRIVATE_BAD_BODY{"

        class RepairCompletions:
            def __init__(self) -> None:
                self.calls = 0

            def create(self, **request: Any) -> Any:
                self.calls += 1
                content = (
                    raw_invalid
                    if self.calls == 1
                    else json.dumps(_ledger_payload())
                )
                return SimpleNamespace(
                    choices=(
                        SimpleNamespace(
                            finish_reason="stop",
                            message=SimpleNamespace(content=content),
                        ),
                    ),
                    usage=SimpleNamespace(
                        prompt_tokens=10 * self.calls,
                        completion_tokens=20 * self.calls,
                        cost=0.001 * self.calls,
                    ),
                )

        completions = RepairCompletions()
        provider = OpenAIStaticPlannerModel(
            SimpleNamespace(chat=SimpleNamespace(completions=completions)),
            model="fake-provider",
            ledger_prompt_version="planner-static-ledger-v2",
        )

        response = provider.generate_coverage_ledger(_topic())

        self.assertEqual(completions.calls, 2)
        self.assertEqual(response.payload, _ledger_payload())
        self.assertEqual(response.input_tokens, 30)
        self.assertEqual(response.output_tokens, 60)
        self.assertAlmostEqual(response.cost, 0.003)
        self.assertNotIn(raw_invalid, repr(response))

    def test_ledger_v2_double_invalid_failure_is_aggregated_and_sanitized(self) -> None:
        raw_invalid = "PRIVATE_BAD_BODY{"

        class AlwaysInvalidCompletions:
            def __init__(self) -> None:
                self.calls = 0

            def create(self, **request: Any) -> Any:
                self.calls += 1
                return SimpleNamespace(
                    choices=(
                        SimpleNamespace(
                            finish_reason="stop",
                            message=SimpleNamespace(content=raw_invalid),
                        ),
                    ),
                    usage=SimpleNamespace(
                        prompt_tokens=10 * self.calls,
                        completion_tokens=20 * self.calls,
                        cost=0.001 * self.calls,
                    ),
                )

        completions = AlwaysInvalidCompletions()
        provider = OpenAIStaticPlannerModel(
            SimpleNamespace(chat=SimpleNamespace(completions=completions)),
            model="fake-provider",
            ledger_prompt_version="planner-static-ledger-v2",
        )

        outcome = AsymmetricStaticPlanner(provider).plan_topic(_topic())
        record = run_planner_dry_run([_topic()], AsymmetricStaticPlanner(provider))[0]

        self.assertEqual(outcome.status, "failed")
        self.assertEqual(outcome.failure_detail, "invalid_json")
        self.assertEqual(outcome.input_tokens, 30)
        self.assertEqual(outcome.output_tokens, 60)
        self.assertAlmostEqual(outcome.cost, 0.003)
        self.assertEqual(outcome.content_shape, "prose_prefix")
        self.assertNotIn(raw_invalid, repr(outcome))
        self.assertNotIn(raw_invalid, json.dumps(record))
    def test_coverage_ledger_is_nested_validated_and_round_trips(self) -> None:
        ledger = parse_coverage_ledger(
            _topic(),
            _ledger_payload(),
            model="fake-planner",
            prompt_version="planner-static-ledger-test-v1",
        )

        self.assertEqual(len(ledger.information_needs), 2)
        self.assertEqual(len(ledger.requirements), 2)
        self.assertEqual(
            ledger.requirements[0].retrieval_anchors,
            ("dated timeline", "named actors and institutions"),
        )
        self.assertEqual(coverage_ledger_from_record(ledger.as_record()), ledger)
        self.assertEqual(len(ledger.ledger_sha256), 64)

    def test_coverage_ledger_rejects_more_than_five_anchor_seeds(self) -> None:
        payload = _ledger_payload()
        payload["information_needs"][0]["requirements"][0][
            "retrieval_anchors"
        ] = [f"distinct anchor {index}" for index in range(1, 7)]

        with self.assertRaisesRegex(ValueError, "at most five"):
            parse_coverage_ledger(
                _topic(),
                payload,
                model="fake-planner",
                prompt_version="planner-static-ledger-v4",
            )

    def test_asymmetric_planner_caches_both_stages_and_records_stage_metrics(self) -> None:
        model = FakeCoveragePlannerModel(_ledger_payload(), _coverage_payload())
        ledger_cache = MemoryCoverageLedgerCache()
        expansion_cache = MemoryCoverageExpansionCache()
        planner = AsymmetricStaticPlanner(
            model,
            ledger_cache=ledger_cache,
            expansion_cache=expansion_cache,
            provider_base_url="https://provider.example/v1",
        )

        first = planner.plan_topic(_topic())
        second = planner.plan_topic(_topic())
        record = run_planner_dry_run([_topic()], planner)[0]

        self.assertEqual(first.status, "success")
        self.assertIsNotNone(first.plan)
        assert first.plan is not None
        self.assertEqual(first.plan.planning_stage, "audited")
        self.assertEqual(first.plan.draft_plan_sha256, model.received_ledger.ledger_sha256)
        self.assertEqual(first.planner_path, "coverage_ledger_v1")
        self.assertEqual(first.first_stage_input_tokens, 7)
        self.assertEqual(first.first_stage_output_tokens, 11)
        self.assertEqual(first.second_stage_input_tokens, 13)
        self.assertEqual(first.second_stage_output_tokens, 17)
        self.assertEqual(first.second_stage_finish_reason, "stop")
        self.assertTrue(second.draft_cache_hit)
        self.assertTrue(second.audit_cache_hit)
        self.assertEqual(model.ledger_calls, 1)
        self.assertEqual(model.expansion_calls, 1)
        self.assertEqual(ledger_cache.put_calls, 1)
        self.assertEqual(expansion_cache.put_calls, 1)
        self.assertEqual(record["planner_path"], "coverage_ledger_v1")
        self.assertEqual(record["first_stage_output_tokens"], 11)
        self.assertEqual(record["second_stage_output_tokens"], 17)
        self.assertEqual(record["second_stage_finish_reason"], "stop")

    def test_asymmetric_planner_rejects_final_task_dependencies(self) -> None:
        final_payload = _payload()
        model = FakeCoveragePlannerModel(_ledger_payload(), final_payload)
        planner = AsymmetricStaticPlanner(model)

        outcome = planner.plan_topic(_topic())

        self.assertEqual(outcome.status, "failed")
        self.assertEqual(outcome.failure_stage, "audit")
        self.assertEqual(outcome.validation_error, "nonempty_dependencies")

    def test_provider_runs_coverage_ledger_and_expansion_contracts(self) -> None:
        class FakeCompletions:
            def __init__(self) -> None:
                self.requests: list[dict[str, Any]] = []

            def create(self, **request: Any) -> Any:
                self.requests.append(request)
                payload = (
                    _ledger_payload()
                    if len(self.requests) == 1
                    else _coverage_payload()
                )
                return SimpleNamespace(
                    choices=(
                        SimpleNamespace(
                            finish_reason="stop",
                            message=SimpleNamespace(content=json.dumps(payload)),
                        ),
                    ),
                    usage=SimpleNamespace(
                        prompt_tokens=10,
                        completion_tokens=20,
                        cost=0.001,
                    ),
                )

        completions = FakeCompletions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        provider = OpenAIStaticPlannerModel(client, model="fake-provider")
        ledger_response = provider.generate_coverage_ledger(_topic())
        ledger = parse_coverage_ledger(
            _topic(),
            ledger_response.payload,
            model=ledger_response.model,
            prompt_version=ledger_response.prompt_version,
        )
        final_response = provider.expand_coverage_ledger(
            _topic(), ledger, max_total_tasks=None
        )

        ledger_request, final_request = completions.requests
        self.assertEqual(ledger_request["max_tokens"], 16_000)
        self.assertEqual(final_request["max_tokens"], 32_000)
        self.assertIn("coverage ledger", ledger_request["messages"][0]["content"])
        self.assertIn("retrieval-anchor", ledger_request["messages"][0]["content"])
        self.assertIn("empty depends_on", final_request["messages"][0]["content"])
        self.assertIn(
            "preserve overlapping",
            final_request["messages"][0]["content"].casefold(),
        )
        self.assertEqual(
            ledger_response.prompt_version, "planner-static-ledger-v1"
        )
        self.assertEqual(
            final_response.prompt_version, "planner-static-expand-v1"
        )

    def test_provider_compact_expansion_v2_uses_small_schema(self) -> None:
        class CompactCompletions:
            def __init__(self) -> None:
                self.request: dict[str, Any] | None = None

            def create(self, **request: Any) -> Any:
                self.request = request
                return SimpleNamespace(
                    choices=(
                        SimpleNamespace(
                            finish_reason="stop",
                            message=SimpleNamespace(
                                content=json.dumps(_compact_expansion_payload())
                            ),
                        ),
                    ),
                    usage=SimpleNamespace(
                        prompt_tokens=10,
                        completion_tokens=20,
                        cost=0.001,
                    ),
                )

        completions = CompactCompletions()
        provider = OpenAIStaticPlannerModel(
            SimpleNamespace(chat=SimpleNamespace(completions=completions)),
            model="fake-provider",
            expansion_prompt_version="planner-static-expand-v2",
        )
        ledger = parse_coverage_ledger(
            _topic(),
            _ledger_payload(),
            model="fake-provider",
            prompt_version="planner-static-ledger-v1",
        )

        response = provider.expand_coverage_ledger(
            _topic(), ledger, max_total_tasks=None
        )

        assert completions.request is not None
        schema = completions.request["response_format"]["json_schema"]["schema"]
        self.assertEqual(schema["required"], ["query_sets"])
        self.assertNotIn("information_needs", schema["properties"])
        system = completions.request["messages"][0]["content"].casefold()
        self.assertIn("only query sets", system)
        self.assertIn("anchor dimension", system)
        self.assertEqual(response.payload, _compact_expansion_payload())

    def test_provider_minimal_ledger_v3_omits_anchor_emission(self) -> None:
        minimal_payload = {
            "information_needs": [
                {
                    "need_ref": "context",
                    "objective": "Establish the historical context.",
                    "priority": 1,
                    "requirements": [
                        {
                            "requirement_ref": "history",
                            "description": "Explain the dated historical context.",
                            "priority": 1,
                        }
                    ],
                }
            ]
        }

        class MinimalLedgerCompletions:
            def __init__(self) -> None:
                self.request: dict[str, Any] | None = None

            def create(self, **request: Any) -> Any:
                self.request = request
                return SimpleNamespace(
                    choices=(
                        SimpleNamespace(
                            finish_reason="stop",
                            message=SimpleNamespace(
                                content=json.dumps(minimal_payload)
                            ),
                        ),
                    ),
                    usage=SimpleNamespace(
                        prompt_tokens=10,
                        completion_tokens=20,
                        cost=0.001,
                    ),
                )

        completions = MinimalLedgerCompletions()
        provider = OpenAIStaticPlannerModel(
            SimpleNamespace(chat=SimpleNamespace(completions=completions)),
            model="fake-provider",
            ledger_prompt_version="planner-static-ledger-v3",
        )

        response = provider.generate_coverage_ledger(_topic())

        assert completions.request is not None
        schema = completions.request["response_format"]["json_schema"]["schema"]
        requirement_properties = schema["properties"]["information_needs"][
            "items"
        ]["properties"]["requirements"]["items"]["properties"]
        need_properties = schema["properties"]["information_needs"]["items"][
            "properties"
        ]
        self.assertNotIn("retrieval_anchors", requirement_properties)
        self.assertNotIn("why_it_matters", need_properties)
        self.assertNotIn(
            "retrieval-anchor",
            completions.request["messages"][0]["content"].casefold(),
        )
        self.assertEqual(
            response.payload["information_needs"][0]["why_it_matters"],
            "Establish the historical context.",
        )
        self.assertEqual(
            response.payload["information_needs"][0]["requirements"][0][
                "retrieval_anchors"
            ],
            ["Explain the dated historical context."],
        )

    def test_provider_final_contract_uses_request_authority_and_marginal_yield(self) -> None:
        class FinalContractCompletions:
            def __init__(self) -> None:
                self.requests: list[dict[str, Any]] = []

            def create(self, **request: Any) -> Any:
                self.requests.append(request)
                payload = (
                    _ledger_payload()
                    if len(self.requests) == 1
                    else _compact_expansion_payload()
                )
                return SimpleNamespace(
                    choices=(
                        SimpleNamespace(
                            finish_reason="stop",
                            message=SimpleNamespace(content=json.dumps(payload)),
                        ),
                    ),
                    usage=SimpleNamespace(
                        prompt_tokens=10,
                        completion_tokens=20,
                        cost=0.001,
                    ),
                )

        completions = FinalContractCompletions()
        provider = OpenAIStaticPlannerModel(
            SimpleNamespace(chat=SimpleNamespace(completions=completions)),
            model="fake-provider",
            ledger_prompt_version="planner-static-ledger-v4",
            expansion_prompt_version="planner-static-expand-v3",
        )

        ledger_response = provider.generate_coverage_ledger(_topic())
        ledger = parse_coverage_ledger(
            _topic(),
            ledger_response.payload,
            model=ledger_response.model,
            prompt_version=ledger_response.prompt_version,
        )
        expansion_response = provider.expand_coverage_ledger(
            _topic(), ledger, max_total_tasks=None
        )

        ledger_request, expansion_request = completions.requests
        ledger_system = ledger_request["messages"][0]["content"].casefold()
        self.assertIn("problem statement is authoritative", ledger_system)
        self.assertIn("title for disambiguation", ledger_system)
        self.assertIn("background", ledger_system)
        self.assertIn("biographical or workflow context", ledger_system)
        self.assertIn("smallest complete set", ledger_system)
        self.assertIn("merge restatements", ledger_system)
        self.assertIn("do not split", ledger_system)
        ledger_schema = ledger_request["response_format"]["json_schema"][
            "schema"
        ]
        anchors = ledger_schema["properties"]["information_needs"]["items"][
            "properties"
        ]["requirements"]["items"]["properties"]["retrieval_anchors"]
        self.assertEqual(anchors["maxItems"], 5)

        expansion_system = expansion_request["messages"][0][
            "content"
        ].casefold()
        self.assertIn("faithful direct query", expansion_system)
        self.assertIn("candidate-shifting", expansion_system)
        self.assertIn("source-language variants", expansion_system)
        self.assertIn("do not translate every query", expansion_system)
        self.assertIn("no query-count quota", expansion_system)
        self.assertIn("plausible marginal", expansion_system)
        expansion_schema = expansion_request["response_format"]["json_schema"][
            "schema"
        ]
        self.assertEqual(expansion_schema["required"], ["query_sets"])
        self.assertEqual(
            expansion_response.payload, _compact_expansion_payload()
        )

    def test_valid_json_with_length_finish_reason_is_truncation(self) -> None:
        class LengthCompletions:
            def create(self, **request: Any) -> Any:
                return SimpleNamespace(
                    choices=(
                        SimpleNamespace(
                            finish_reason="length",
                            message=SimpleNamespace(
                                content=json.dumps(_ledger_payload())
                            ),
                        ),
                    ),
                    usage=SimpleNamespace(
                        prompt_tokens=10,
                        completion_tokens=100,
                        cost=0.001,
                    ),
                )

        provider = OpenAIStaticPlannerModel(
            SimpleNamespace(
                chat=SimpleNamespace(completions=LengthCompletions())
            ),
            model="fake-provider",
        )
        outcome = AsymmetricStaticPlanner(provider).plan_topic(_topic())

        self.assertEqual(outcome.status, "failed")
        self.assertEqual(outcome.failure_detail, "output_truncated")
        self.assertEqual(outcome.finish_reason, "length")
        self.assertEqual(outcome.first_stage_output_tokens, 100)

    def test_expansion_headroom_is_a_hard_pre_cache_gate(self) -> None:
        model = FakeCoveragePlannerModel(
            _ledger_payload(),
            _coverage_payload(),
            expansion_output_tokens=24_000,
        )
        ledger_cache = MemoryCoverageLedgerCache()
        expansion_cache = MemoryCoverageExpansionCache()
        planner = AsymmetricStaticPlanner(
            model,
            ledger_cache=ledger_cache,
            expansion_cache=expansion_cache,
        )

        outcome = planner.plan_topic(_topic())

        self.assertEqual(outcome.status, "failed")
        self.assertEqual(outcome.failure_stage, "audit")
        self.assertEqual(outcome.failure_detail, "output_headroom_exceeded")
        self.assertEqual(expansion_cache.put_calls, 0)

    def test_ledger_headroom_is_a_hard_pre_cache_gate(self) -> None:
        model = FakeCoveragePlannerModel(
            _ledger_payload(),
            _compact_expansion_payload(),
            ledger_output_tokens=8_000,
        )
        ledger_cache = MemoryCoverageLedgerCache()
        expansion_cache = MemoryCoverageExpansionCache()
        planner = AsymmetricStaticPlanner(
            model,
            ledger_cache=ledger_cache,
            expansion_cache=expansion_cache,
            expansion_prompt_version="planner-static-expand-v3",
        )

        outcome = planner.plan_topic(_topic())

        self.assertEqual(outcome.status, "failed")
        self.assertEqual(outcome.failure_stage, "draft")
        self.assertEqual(outcome.failure_detail, "output_headroom_exceeded")
        self.assertEqual(outcome.first_stage_output_tokens, 8_000)
        self.assertEqual(ledger_cache.put_calls, 0)
        self.assertEqual(model.expansion_calls, 0)

    def test_asymmetric_cache_failures_fail_closed_without_model_calls(self) -> None:
        class RaisingLedgerCache:
            def get(self, key: str) -> CoverageLedgerCacheEntry | None:
                raise PermissionError("private diagnostic")

            def put(self, key: str, entry: CoverageLedgerCacheEntry) -> bool:
                raise AssertionError("put must not run")

        model = FakeCoveragePlannerModel(_ledger_payload(), _coverage_payload())
        planner = AsymmetricStaticPlanner(model, ledger_cache=RaisingLedgerCache())

        outcome = planner.plan_topic(_topic())

        self.assertEqual(outcome.status, "failed")
        self.assertEqual(outcome.failure_kind, "cache_failure")
        self.assertEqual(outcome.failure_detail, "ledger_cache_read_failure")
        self.assertEqual(model.ledger_calls, 0)
        self.assertNotIn("private diagnostic", repr(outcome))

    def test_asymmetric_cache_identity_includes_output_allowances(self) -> None:
        model = FakeCoveragePlannerModel(_ledger_payload(), _coverage_payload())
        standard = AsymmetricStaticPlanner(
            model,
            ledger_max_tokens=16_000,
            expansion_max_tokens=32_000,
        )
        reduced = AsymmetricStaticPlanner(
            model,
            ledger_max_tokens=100,
            ledger_gate_max_output_tokens=75,
            expansion_max_tokens=200,
            expansion_gate_max_output_tokens=150,
        )
        ledger = parse_coverage_ledger(
            _topic(),
            _ledger_payload(),
            model="fake-planner",
            prompt_version="planner-static-ledger-test-v1",
        )

        self.assertNotEqual(
            standard.ledger_cache_key(_topic()),
            reduced.ledger_cache_key(_topic()),
        )
        self.assertNotEqual(
            standard.plan_cache_key(_topic(), ledger),
            reduced.plan_cache_key(_topic(), ledger),
        )

    def test_asymmetric_richer_path_rejects_a_task_cap(self) -> None:
        model = FakeCoveragePlannerModel(_ledger_payload(), _coverage_payload())

        with self.assertRaisesRegex(ValueError, "does not accept a task cap"):
            AsymmetricStaticPlanner(model, max_total_tasks=24)

    def test_explicit_topic_selection_is_exact_and_ordered(self) -> None:
        topics = tuple(_topic(str(2000 + index)) for index in range(6))

        selected = select_topics_by_id(topics, ("2003", "2001", "2004"))

        self.assertEqual(
            [topic.topic_id for topic in selected],
            ["2003", "2001", "2004"],
        )
        with self.assertRaisesRegex(ValueError, "unknown topic_id"):
            select_topics_by_id(topics, ("9999",))

    def test_audit_failure_reuses_validated_draft_on_next_attempt(self) -> None:
        model = FakePlannerModel(_coverage_payload())
        invalid_audit = _coverage_payload()
        invalid_audit["information_needs"][1]["search_tasks"][0][
            "query"
        ] = "topic historical origins timeline"
        auditor = FakePlannerAuditor(invalid_audit)
        cache = MemoryPlanCache()
        planner = StaticPlanner(model, auditor=auditor, cache=cache)

        failed = planner.plan_topic(_topic())
        auditor.payload = _coverage_payload()
        recovered = planner.plan_topic(_topic())

        self.assertEqual(failed.status, "failed")
        self.assertEqual(failed.failure_kind, "audit_invalid_output")
        self.assertEqual(recovered.status, "success")
        self.assertTrue(recovered.draft_cache_hit)
        self.assertFalse(recovered.audit_cache_hit)
        self.assertEqual(model.calls, 1)
        self.assertEqual(auditor.calls, 2)
        self.assertEqual(cache.put_calls, 2)

    def test_uncapped_plan_preserves_more_than_twenty_four_tasks(self) -> None:
        draft = parse_static_plan(
            _topic(),
            _many_task_payload(),
            model="fake-planner",
            prompt_version="planner-static-test-v2",
            max_total_tasks=None,
        )
        audited = parse_static_plan(
            _topic(),
            _many_task_payload(),
            model="fake-auditor",
            prompt_version="planner-static-audit-test-v2",
            max_total_tasks=None,
            planning_stage="audited",
            draft_plan_sha256=draft.plan_sha256,
        )

        self.assertEqual(len(audited.tasks), 25)
        self.assertEqual(audited.truncated_task_count, 0)
        self.assertTrue(allocate_tasks(audited, 25).complete)

    def test_provider_runs_distinct_draft_and_audit_contracts(self) -> None:
        class FakeCompletions:
            def __init__(self) -> None:
                self.requests: list[dict[str, Any]] = []

            def create(self, **request: Any) -> Any:
                self.requests.append(request)
                return SimpleNamespace(
                    choices=(
                        SimpleNamespace(
                            finish_reason="stop",
                            message=SimpleNamespace(
                                content=json.dumps(_coverage_payload())
                            )
                        ),
                    ),
                    usage=SimpleNamespace(
                        prompt_tokens=10,
                        completion_tokens=20,
                        cost=0.001,
                    ),
                )

        completions = FakeCompletions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        provider = OpenAIStaticPlannerModel(client, model="fake-provider")

        draft_response = provider.generate(_topic(), max_total_tasks=None)
        draft = parse_static_plan(
            _topic(),
            draft_response.payload,
            model=draft_response.model,
            prompt_version=draft_response.prompt_version,
            max_total_tasks=None,
            planning_stage="draft",
        )
        audit_response = provider.audit(_topic(), draft, max_total_tasks=None)

        self.assertEqual(len(completions.requests), 2)
        draft_request, audit_request = completions.requests
        draft_body = json.loads(draft_request["messages"][1]["content"])
        audit_body = json.loads(audit_request["messages"][1]["content"])
        self.assertEqual(draft_body["topic"], _topic().as_record())
        self.assertEqual(audit_body["topic"], _topic().as_record())
        self.assertEqual(audit_body["draft_plan"]["plan_sha256"], draft.plan_sha256)
        self.assertEqual(draft_request["max_tokens"], 16_000)
        self.assertEqual(audit_request["max_tokens"], 32_000)
        draft_system = draft_request["messages"][0]["content"]
        audit_system = audit_request["messages"][0]["content"]
        self.assertIn("candidate recall", draft_system)
        self.assertIn("distinct retrieval direction", draft_system)
        self.assertIn(
            "do not create queries merely to fill", draft_system.casefold()
        )
        self.assertIn("preserve overlapping queries", audit_system)
        self.assertIn("harmful unsupported", audit_system)
        schema = draft_request["response_format"]["json_schema"]["schema"]
        self.assertIn("requirements", schema["required"])
        task_schema = schema["properties"]["information_needs"]["items"][
            "properties"
        ]["search_tasks"]["items"]
        self.assertIn("requirement_refs", task_schema["required"])
        self.assertIn("query_role", task_schema["required"])
        self.assertEqual(draft_response.prompt_version, "planner-static-draft-v3")
        self.assertEqual(audit_response.prompt_version, "planner-static-audit-v4")
        self.assertEqual(draft_response.finish_reason, "stop")

    def test_truncated_provider_json_retains_sanitized_failure_metrics(self) -> None:
        raw_invalid_content = "SECRET_RESPONSE_BODY{"

        class TruncatedCompletions:
            def create(self, **request: Any) -> Any:
                return SimpleNamespace(
                    choices=(
                        SimpleNamespace(
                            finish_reason="length",
                            message=SimpleNamespace(content=raw_invalid_content),
                        ),
                    ),
                    usage=SimpleNamespace(
                        prompt_tokens=31,
                        completion_tokens=16_000,
                        cost=0.0042,
                    ),
                )

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=TruncatedCompletions())
        )
        provider = OpenAIStaticPlannerModel(client, model="fake-provider")
        planner = StaticPlanner(provider, auditor=provider, max_total_tasks=None)

        outcome = planner.plan_topic(_topic())
        record = run_planner_dry_run([_topic()], planner)[0]

        self.assertEqual(outcome.status, "failed")
        self.assertEqual(outcome.failure_kind, "draft_invalid_output")
        self.assertEqual(outcome.failure_stage, "draft")
        self.assertEqual(outcome.failure_detail, "output_truncated")
        self.assertEqual(outcome.finish_reason, "length")
        self.assertEqual(outcome.input_tokens, 31)
        self.assertEqual(outcome.output_tokens, 16_000)
        self.assertAlmostEqual(outcome.cost, 0.0042)
        self.assertGreaterEqual(outcome.latency_seconds, 0)
        self.assertEqual(record["failure_stage"], "draft")
        self.assertEqual(record["failure_detail"], "output_truncated")
        self.assertEqual(record["finish_reason"], "length")
        self.assertNotIn(raw_invalid_content, json.dumps(record))

    def test_local_plan_validation_failure_is_classified(self) -> None:
        model = FakePlannerModel(_payload(duplicate=True))
        planner = StaticPlanner(
            model,
            auditor=FakePlannerAuditor(_payload()),
            max_total_tasks=None,
        )

        outcome = planner.plan_topic(_topic())

        self.assertEqual(outcome.status, "failed")
        self.assertEqual(outcome.failure_kind, "draft_invalid_output")
        self.assertEqual(outcome.failure_stage, "draft")
        self.assertEqual(outcome.failure_detail, "local_validation")
        self.assertEqual(outcome.validation_error, "duplicate_query")
        self.assertIsNone(outcome.finish_reason)
        self.assertEqual(outcome.input_tokens, 11)
        self.assertEqual(outcome.output_tokens, 22)
        self.assertAlmostEqual(outcome.cost, 0.001)

    def test_planner_audits_draft_before_sealing(self) -> None:
        model = FakePlannerModel(_coverage_payload())
        auditor = FakePlannerAuditor(_coverage_payload())
        cache = MemoryPlanCache()
        planner = StaticPlanner(
            model,
            auditor=auditor,
            cache=cache,
            max_total_tasks=None,
            provider_base_url="https://provider.example/v1",
        )

        first = planner.plan_topic(_topic())
        second = planner.plan_topic(_topic())

        self.assertEqual(first.status, "success")
        self.assertIsNotNone(first.plan)
        assert first.plan is not None
        self.assertEqual(first.plan.planning_stage, "audited")
        self.assertIsNotNone(auditor.received_draft)
        assert auditor.received_draft is not None
        self.assertEqual(first.plan.draft_plan_sha256, auditor.received_draft.plan_sha256)
        self.assertEqual(first.input_tokens, 24)
        self.assertEqual(first.output_tokens, 39)
        self.assertAlmostEqual(first.cost, 0.003)
        self.assertAlmostEqual(first.latency_seconds, 0.75)
        self.assertFalse(first.draft_cache_hit)
        self.assertFalse(first.audit_cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertTrue(second.draft_cache_hit)
        self.assertTrue(second.audit_cache_hit)
        self.assertEqual(model.calls, 1)
        self.assertEqual(auditor.calls, 1)
        self.assertEqual(cache.put_calls, 2)

    def test_requirement_coverage_and_query_roles_are_sealed(self) -> None:
        plan = parse_static_plan(
            _topic(),
            _coverage_payload(),
            model="fake-planner",
            prompt_version="planner-static-test-v2",
            max_total_tasks=None,
            planning_stage="draft",
        )

        self.assertEqual(plan.planning_stage, "draft")
        self.assertIsNone(plan.draft_plan_sha256)
        self.assertEqual(len(plan.requirements), 2)
        self.assertTrue(plan.requirements[0].requirement_id.startswith("requirement-"))
        self.assertEqual(plan.tasks[0].query_role, "broad_discovery")
        self.assertEqual(
            plan.tasks[0].requirement_ids,
            (plan.requirements[0].requirement_id,),
        )
        self.assertEqual(
            {requirement_id for task in plan.tasks for requirement_id in task.requirement_ids},
            {requirement.requirement_id for requirement in plan.requirements},
        )
        self.assertEqual(static_plan_from_record(plan.as_record()), plan)
        self.assertEqual(plan.schema_version, "planner_static_plan_v2")

    def test_plan_is_typed_ids_are_local_and_dependencies_are_resolved(self) -> None:
        first = parse_static_plan(
            _topic(),
            _payload(),
            model="fake-planner",
            prompt_version="planner-static-test-v1",
            max_total_tasks=24,
        )
        second = parse_static_plan(
            _topic(),
            _payload(),
            model="fake-planner",
            prompt_version="planner-static-test-v1",
            max_total_tasks=24,
        )

        self.assertIsInstance(first, StaticResearchPlan)
        self.assertEqual(first, second)
        self.assertEqual([need.priority for need in first.information_needs], [1, 2])
        self.assertEqual(len(first.tasks), 4)
        self.assertEqual(first.tasks[2].depends_on, (first.tasks[0].task_id,))
        self.assertTrue(first.information_needs[0].need_id.startswith("need-"))
        self.assertTrue(first.tasks[0].task_id.startswith("task-"))
        self.assertEqual(first.raw_task_count, 4)
        self.assertEqual(first.truncated_task_count, 0)

    def test_unknown_fields_and_duplicate_queries_are_rejected(self) -> None:
        payload = _payload()
        payload["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "unknown plan field"):
            parse_static_plan(
                _topic(),
                payload,
                model="fake-planner",
                prompt_version="planner-static-test-v1",
                max_total_tasks=24,
            )

        with self.assertRaisesRegex(ValueError, "duplicate normalized query"):
            parse_static_plan(
                _topic(),
                _payload(duplicate=True),
                model="fake-planner",
                prompt_version="planner-static-test-v1",
                max_total_tasks=24,
            )

    def test_cap_is_deterministic_and_keeps_breadth_first_allocation(self) -> None:
        plan = parse_static_plan(
            _topic(),
            _payload(),
            model="fake-planner",
            prompt_version="planner-static-test-v1",
            max_total_tasks=3,
        )
        self.assertEqual(len(plan.tasks), 3)
        allocation = allocate_tasks(plan, 3)
        self.assertEqual(
            [(item.need_id, item.task.query) for item in allocation.tasks],
            [
                (plan.information_needs[0].need_id, "topic historical origins timeline"),
                (plan.information_needs[1].need_id, "topic measurable outcomes evaluation evidence"),
                (plan.information_needs[0].need_id, "topic historical actors institutions"),
            ],
        )
        self.assertTrue(allocation.complete)
        self.assertEqual(plan.truncated_task_count, 1)

    def test_shortfall_is_explicit_for_matched_budget(self) -> None:
        plan = parse_static_plan(
            _topic(),
            _payload(),
            model="fake-planner",
            prompt_version="planner-static-test-v1",
            max_total_tasks=24,
        )
        allocation = allocate_tasks(plan, 5)
        self.assertEqual(len(allocation.tasks), 4)
        self.assertFalse(allocation.complete)
        self.assertEqual(allocation.shortfall, 1)

    def test_validated_planner_work_is_cached_once_and_reused(self) -> None:
        model = FakePlannerModel(_payload())
        auditor = FakePlannerAuditor(_payload())
        cache = MemoryPlanCache()
        planner = StaticPlanner(
            model,
            auditor=auditor,
            cache=cache,
            max_total_tasks=24,
            provider_base_url="https://provider.example/v1",
        )

        first = planner.plan_topic(_topic())
        second = planner.plan_topic(_topic())

        self.assertEqual(first.status, "success")
        self.assertFalse(first.cache_hit)
        self.assertEqual(second.status, "success")
        self.assertTrue(second.cache_hit)
        self.assertEqual(model.calls, 1)
        self.assertEqual(auditor.calls, 1)
        self.assertEqual(cache.put_calls, 2)
        self.assertIsNotNone(second.plan)

    def test_invalid_model_output_is_failure_and_is_not_cached(self) -> None:
        model = FakePlannerModel(_payload(duplicate=True))
        cache = MemoryPlanCache()
        planner = StaticPlanner(
            model,
            auditor=FakePlannerAuditor(_payload()),
            cache=cache,
            max_total_tasks=24,
        )

        outcome = planner.plan_topic(_topic())

        self.assertEqual(outcome.status, "failed")
        self.assertEqual(outcome.failure_kind, "draft_invalid_output")
        self.assertIsNone(outcome.plan)
        self.assertEqual(cache.put_calls, 0)

    def test_plan_record_round_trip_is_canonical(self) -> None:
        plan = parse_static_plan(
            _topic(),
            _payload(),
            model="fake-planner",
            prompt_version="planner-static-test-v1",
            max_total_tasks=24,
        )
        record = static_plan_to_record(plan)
        self.assertEqual(static_plan_from_record(record), plan)
        self.assertEqual(record["schema_version"], "planner_static_plan_v2")


def _retrieval(query: str, *, document_id: str = "doc-1") -> RetrievalResult:
    hit = SearchHit(rank=1, document_id=document_id, score=0.9)
    return RetrievalResult(
        search_result=PipelineResult(
            query=query,
            pipeline="ragtime2",
            hits=(hit,),
            processed=True,
            local_cache_hit=True,
            remote_cached=True,
            server_timestamp=None,
            cache_stored_at=None,
        ),
        documents=(
            RetrievedDocument(
                rank=1,
                document_id=document_id,
                score=0.9,
                text="Canonical source sentence with dated evidence.",
                url=None,
                date=None,
                language="eng",
                content_source="huggingface",
                content_local_cache_hit=True,
            ),
        ),
    )


class FakeRetriever:
    def __init__(self, failures: set[str] | None = None) -> None:
        self.failures = failures or set()
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.cached_calls: list[str] = []

    def get_cached_retrieval(self, query: str, **kwargs: object) -> RetrievalResult | None:
        self.cached_calls.append(query)
        return None

    def retrieve(self, query: str, **kwargs: object) -> RetrievalResult:
        self.calls.append((query, kwargs))
        if query in self.failures:
            raise RuntimeError("secret provider detail must not escape")
        return _retrieval(query, document_id=f"doc-{len(self.calls)}")


class StaticExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.draft = parse_static_plan(
            _topic(),
            _payload(),
            model="fake-planner",
            prompt_version="planner-static-test-v1",
            max_total_tasks=24,
        )
        self.plan = parse_static_plan(
            _topic(),
            _payload(),
            model="fake-auditor",
            prompt_version="planner-static-audit-test-v2",
            max_total_tasks=24,
            planning_stage="audited",
            draft_plan_sha256=self.draft.plan_sha256,
        )

    def test_executor_rejects_unaudited_draft(self) -> None:
        with self.assertRaisesRegex(ValueError, "audited"):
            execute_static_plan(
                self.draft,
                FakeRetriever(),
                TopRankedSentenceReader(),
                requested_task_count=1,
            )

    def test_executor_uses_exact_matched_budget_and_preserves_lineage(self) -> None:
        retriever = FakeRetriever()
        result = execute_static_plan(
            self.plan,
            retriever,
            TopRankedSentenceReader(),
            requested_task_count=3,
            top_k=100,
            max_passages_per_task=1,
        )

        self.assertIsInstance(result, StaticTopicEvidence)
        self.assertEqual(result.status, "success")
        self.assertEqual(len(retriever.calls), 3)
        self.assertEqual([call[1]["top_k"] for call in retriever.calls], [100] * 3)
        self.assertEqual([p.document_id for p in result.passages], ["doc-1", "doc-2", "doc-3"])
        passage = result.passages[0]
        self.assertEqual(passage.representation, "canonical")
        self.assertEqual(passage.source_language, "eng")
        self.assertEqual(passage.text, "Canonical source sentence with dated evidence.")
        self.assertEqual(passage.end_char - passage.start_char, len(passage.text))
        self.assertEqual(len(passage.content_hash), 64)
        self.assertEqual(len(retriever.cached_calls), 3)

    def test_executor_reports_partial_failures_without_fabricating_empty_success(self) -> None:
        failed_query = self.plan.tasks[0].query
        retriever = FakeRetriever({failed_query})
        result = execute_static_plan(
            self.plan,
            retriever,
            TopRankedSentenceReader(),
            requested_task_count=3,
            max_passages_per_task=1,
        )

        self.assertEqual(result.status, "partial")
        self.assertEqual(result.failure_kind, "task_failures")
        self.assertEqual(result.task_results[0].failure_kind, "retrieval_failure")
        self.assertEqual(len(result.passages), 1)

    def test_execution_artifact_round_trip_is_anchored_to_plan(self) -> None:
        result = execute_static_plan(
            self.plan,
            FakeRetriever(),
            TopRankedSentenceReader(),
            requested_task_count=2,
            max_passages_per_task=1,
        )

        restored = static_topic_evidence_from_record(result.as_record(), plan=self.plan)

        self.assertEqual(restored, result)
        tampered = result.as_record()
        tampered["task_results"][0]["query"] = "different query"
        with self.assertRaisesRegex(ValueError, "anchored"):
            static_topic_evidence_from_record(tampered, plan=self.plan)

    def test_executor_marks_plan_shortfall_before_search(self) -> None:
        retriever = FakeRetriever()
        result = execute_static_plan(
            self.plan,
            retriever,
            TopRankedSentenceReader(),
            requested_task_count=5,
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.failure_kind, "plan_shortfall")
        self.assertEqual(retriever.calls, [])

    def test_chunked_reader_preserves_report_text_representation(self) -> None:
        class FakeChunker:
            def chunk(self, **kwargs: object) -> tuple[Passage, ...]:
                selected = str(kwargs["selected_text"])
                return (
                    Passage(
                        document_id=str(kwargs["document_id"]),
                        representation="report_text",
                        language="eng",
                        text=selected,
                        start_char=0,
                        end_char=len(selected),
                        content_hash=sha256_bytes(selected),
                        boundary="sentence",
                    ),
                )

        document = RetrievedDocument(
            rank=1,
            document_id="doc-translation",
            score=0.8,
            text="Texto fuente.",
            url=None,
            date=None,
            language="spa",
            content_source="huggingface",
            content_local_cache_hit=True,
            english_text="Verified English translation.",
        )
        result = RetrievalResult(
            search_result=PipelineResult(
                query="q",
                pipeline="ragtime2",
                hits=(SearchHit(1, document.document_id, document.score),),
                processed=True,
                local_cache_hit=True,
                remote_cached=True,
                server_timestamp=None,
                cache_stored_at=None,
            ),
            documents=(document,),
        )

        passages = ChunkedPassageReader(
            FakeChunker(), representation="report_text"
        ).read(
            result,
            topic_id="2000",
            need_id="need-1",
            task_id="task-1",
            query="q",
            max_passages=1,
        )

        self.assertEqual(passages[0].text, "Verified English translation.")
        self.assertEqual(passages[0].source_language, "spa")
        self.assertEqual(passages[0].representation, "report_text")
        self.assertEqual(passages[0].source_text_hash, sha256_bytes("Texto fuente.").hex())

    def test_serial_runner_uses_exact_baseline_budget_and_writes_sanitized_records(self) -> None:
        class FakeModel:
            def generate(
                self, topic: PlanningTopic, *, max_total_tasks: int | None
            ) -> PlannerResponse:
                return PlannerResponse(
                    payload=_payload(),
                    model="fake-planner",
                    prompt_version="planner-static-test-v1",
                )

        planner = StaticPlanner(
            FakeModel(),
            auditor=FakePlannerAuditor(_payload()),
            max_total_tasks=24,
        )
        with TemporaryDirectory() as directory:
            output = Path(directory) / "records.jsonl"
            records = run_static_topics(
                [_topic()],
                planner,
                FakeRetriever(),
                TopRankedSentenceReader(),
                budget_mode="matched_baseline_c_v1",
                baseline_budgets={"2000": 3},
                max_passages_per_task=1,
                output_path=output,
            )

            self.assertEqual(records[0]["status"], "success")
            self.assertEqual(records[0]["planner_path"], "full_plan_v3")
            self.assertEqual(records[0]["requested_task_count"], 3)
            self.assertEqual(len(records[0]["evidence"]["task_results"]), 3)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 1)
            serialized = output.read_text(encoding="utf-8").lower()
            self.assertNotIn('"messages"', serialized)
            self.assertNotIn('"authorization"', serialized)

    def test_baseline_budget_loader_rejects_partial_topics(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.jsonl"
            path.write_text(
                '{"topic_id":"2000","status":"success","query_results":[{"status":"success"},{"status":"success"}]}\n',
                encoding="utf-8",
            )
            self.assertEqual(load_baseline_query_budgets(path), {"2000": 2})
            path.write_text(
                '{"topic_id":"2000","status":"partial","query_results":[{"status":"success"}]}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not successful"):
                load_baseline_query_budgets(path)


if __name__ == "__main__":
    unittest.main()
