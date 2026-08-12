"""Contract tests for the upstream Nuggetizer boundary."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from typing import Any

from nuggetizer.core.types import AssignedScoredNugget, ScoredNugget

from trec_ragtime.nuggetizer_adapter import (
    OPENROUTER_BASE_URL,
    BoundedHandler,
    NuggetizerBackend,
    NuggetizerConfig,
    NuggetizerError,
    build,
    package_identity,
    provider_settings,
)


PASSAGES = (
    ("e001", "Grain exports through the southern corridor fell during the quarter."),
    ("e002", "Two of the three deep-water ports remained closed to bulk carriers."),
    ("e003", "Unrelated commentary about regional tourism figures."),
)
CONFIG = NuggetizerConfig(model="test-model", max_nuggets=5)


class ScriptedHandler:
    """Stands in for an upstream ``LLMHandler`` without touching a provider."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[list[dict[str, str]], float]] = []

    def run(
        self, messages: list[dict[str, str]], temperature: float = 0
    ) -> tuple[str, int]:
        self.calls.append((messages, temperature))
        if not self.replies:
            raise AssertionError("scripted handler ran out of replies")
        return self.replies.pop(0), 0


class ScriptedNuggetizer:
    """Returns pre-built upstream records so mapping can be tested directly."""

    def __init__(self, scored: Any = (), assigned: Any = ()) -> None:
        self._scored = list(scored)
        self._assigned = list(assigned)
        self.requests: list[Any] = []
        self.assign_calls: list[tuple[str, str, list[Any]]] = []

    def create(self, request: Any) -> list[Any]:
        self.requests.append(request)
        return self._scored

    def assign(self, query: str, context: str, nuggets: list[Any]) -> list[Any]:
        self.assign_calls.append((query, context, nuggets))
        return self._assigned


def backend(nuggetizer: Any) -> NuggetizerBackend:
    return NuggetizerBackend(CONFIG, nuggetizer=nuggetizer)


def fake_client(*outcomes: Any) -> Any:
    """Build a minimal OpenAI-shaped client returning scripted content."""

    remaining = list(outcomes)

    def _create(**_: Any) -> Any:
        outcome = remaining.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=outcome))]
        )

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create))
    )


class ProviderKeyTestCase(unittest.TestCase):
    """Upstream resolves the key eagerly, so every build needs one present."""

    def setUp(self) -> None:
        self.previous = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "test-key"

    def tearDown(self) -> None:
        if self.previous is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = self.previous


class PackageIdentityTestCase(unittest.TestCase):
    def test_reports_the_installed_distribution(self) -> None:
        identity = package_identity()
        self.assertEqual(identity["name"], "nuggetizer")
        self.assertTrue(identity["version"])
        # A published release carries no VCS metadata; a Git pin would.
        self.assertEqual(set(identity) - {"commit"}, {"name", "version"})


class BuildTestCase(ProviderKeyTestCase):
    def test_shares_one_bounded_handler_across_every_operation(self) -> None:
        nuggetizer = build(NuggetizerConfig(model="test-model", max_nuggets=7))
        handler = nuggetizer.creator_llm
        self.assertIsInstance(handler, BoundedHandler)
        self.assertIs(nuggetizer.scorer_llm, handler)
        self.assertIs(nuggetizer.assigner_llm, handler)

    def test_assigner_window_covers_every_nugget_in_one_call(self) -> None:
        nuggetizer = build(NuggetizerConfig(model="test-model", max_nuggets=7))
        self.assertEqual(nuggetizer.assigner_window_size, 7)

    def test_drives_the_real_package_end_to_end(self) -> None:
        instance = NuggetizerBackend(CONFIG)
        handler = ScriptedHandler(
            '["grain exports fell", "two ports closed"]',
            '["vital", "okay"]',
        )
        instance._nuggetizer.creator_llm = handler
        instance._nuggetizer.scorer_llm = handler
        nuggets = instance.create(
            topic_id="2000", query="grain exports", passages=PASSAGES
        )
        self.assertEqual(
            nuggets, (("grain exports fell", "vital"), ("two ports closed", "okay"))
        )
        # Every passage reaches the creator in a single window.
        creator_context = handler.calls[0][0][1]["content"]
        for _, text in PASSAGES:
            self.assertIn(text, creator_context)

        assigner = ScriptedHandler('["support", "not_support"]')
        instance._nuggetizer.assigner_llm = assigner
        labels = instance.assign(
            query="grain exports", passage_text=PASSAGES[0][1], nuggets=nuggets
        )
        self.assertEqual(labels, ("support", "not_support"))
        self.assertIn(PASSAGES[0][1], assigner.calls[0][0][1]["content"])


class CreateTestCase(unittest.TestCase):
    def test_maps_aliases_onto_upstream_documents(self) -> None:
        nuggetizer = ScriptedNuggetizer(
            scored=[ScoredNugget(text="a", importance="vital")]
        )
        backend(nuggetizer).create(topic_id="2000", query="q", passages=PASSAGES)
        request = nuggetizer.requests[0]
        self.assertEqual(request.query.qid, "2000")
        self.assertEqual(
            [(document.docid, document.segment) for document in request.documents],
            list(PASSAGES),
        )

    def test_rejects_the_upstream_failure_sentinel(self) -> None:
        nuggetizer = ScriptedNuggetizer(
            scored=[ScoredNugget(text="a", importance="failed")]
        )
        with self.assertRaisesRegex(NuggetizerError, "unusable importance"):
            backend(nuggetizer).create(topic_id="2000", query="q", passages=PASSAGES)

    def test_rejects_a_blank_claim(self) -> None:
        nuggetizer = ScriptedNuggetizer(
            scored=[ScoredNugget(text="   ", importance="okay")]
        )
        with self.assertRaisesRegex(NuggetizerError, "empty claim"):
            backend(nuggetizer).create(topic_id="2000", query="q", passages=PASSAGES)


class AssignTestCase(unittest.TestCase):
    nuggets = (("grain exports fell", "vital"), ("two ports closed", "okay"))

    def _assigned(self, *labels: str) -> list[AssignedScoredNugget]:
        return [
            AssignedScoredNugget(text=claim, importance=importance, assignment=label)
            for (claim, importance), label in zip(self.nuggets, labels, strict=True)
        ]

    def test_returns_one_label_per_nugget_in_order(self) -> None:
        nuggetizer = ScriptedNuggetizer(
            assigned=self._assigned("support", "partial_support")
        )
        labels = backend(nuggetizer).assign(
            query="q", passage_text="passage", nuggets=self.nuggets
        )
        self.assertEqual(labels, ("support", "partial_support"))
        query, context, scored = nuggetizer.assign_calls[0]
        self.assertEqual((query, context), ("q", "passage"))
        self.assertEqual(
            [nugget.text for nugget in scored], [claim for claim, _ in self.nuggets]
        )

    def test_rejects_an_incomplete_assignment_set(self) -> None:
        nuggetizer = ScriptedNuggetizer(
            assigned=[
                AssignedScoredNugget(
                    text="grain exports fell", importance="vital", assignment="support"
                )
            ]
        )
        with self.assertRaisesRegex(NuggetizerError, "incomplete assignment"):
            backend(nuggetizer).assign(
                query="q", passage_text="p", nuggets=self.nuggets
            )

    def test_rejects_a_rewritten_or_reordered_claim(self) -> None:
        assigned = self._assigned("support", "support")
        assigned[1] = AssignedScoredNugget(
            text="something else", importance="okay", assignment="support"
        )
        nuggetizer = ScriptedNuggetizer(assigned=assigned)
        with self.assertRaisesRegex(NuggetizerError, "reordered or rewrote"):
            backend(nuggetizer).assign(
                query="q", passage_text="p", nuggets=self.nuggets
            )

    def test_rejects_the_upstream_failure_sentinel(self) -> None:
        nuggetizer = ScriptedNuggetizer(assigned=self._assigned("support", "failed"))
        with self.assertRaisesRegex(NuggetizerError, "unusable assignment"):
            backend(nuggetizer).assign(
                query="q", passage_text="p", nuggets=self.nuggets
            )


class BoundedHandlerTestCase(ProviderKeyTestCase):
    def _handler(self, *outcomes: Any, max_calls: int = 8) -> BoundedHandler:
        handler = BoundedHandler(
            "test-model", max_calls=max_calls,
            base_url="https://provider/v1", api_key="test-key",
        )
        handler.client = fake_client(*outcomes)
        return handler

    def test_returns_provider_content(self) -> None:
        handler = self._handler('["a"]')
        self.assertEqual(handler.run([{"role": "user", "content": "hi"}]), ('["a"]', 0))
        self.assertEqual(handler.remaining, 7)

    def test_stops_once_the_budget_is_exhausted(self) -> None:
        handler = self._handler('["a"]', '["b"]', max_calls=1)
        handler.run([{"role": "user", "content": "hi"}])
        # Upstream would otherwise retry this window forever.
        with self.assertRaisesRegex(NuggetizerError, "call budget is exhausted"):
            handler.run([{"role": "user", "content": "hi"}])

    def test_records_and_reraises_a_provider_failure(self) -> None:
        failure = RuntimeError("provider down")
        handler = self._handler(failure)
        with self.assertRaises(RuntimeError):
            handler.run([{"role": "user", "content": "hi"}])
        self.assertIs(handler.error, failure)

    def test_rejects_an_empty_completion(self) -> None:
        for content in ("", "   ", None):
            with self.subTest(content=content):
                handler = self._handler(content)
                with self.assertRaisesRegex(NuggetizerError, "empty completion"):
                    handler.run([{"role": "user", "content": "hi"}])

    def test_requires_a_positive_budget(self) -> None:
        for max_calls in (0, -1, True):
            with self.subTest(max_calls=max_calls):
                with self.assertRaises(ValueError):
                    BoundedHandler(
                        "test-model", max_calls=max_calls,
                        base_url="https://provider/v1", api_key="k",
                    )


class ProviderSettingsTestCase(unittest.TestCase):
    """This repository documents OpenRouter; pinned 0.0.5 has never heard of it."""

    def test_prefers_the_documented_openrouter_variable(self) -> None:
        base_url, key = provider_settings({"OPENROUTER_API_KEY": "router"})
        self.assertEqual(base_url, OPENROUTER_BASE_URL)
        self.assertEqual(key, "router")

    def test_falls_back_to_openai_with_its_own_base_url(self) -> None:
        base_url, key = provider_settings(
            {"OPENAI_API_KEY": "direct", "OPENAI_BASE_URL": "https://proxy/v1"}
        )
        self.assertEqual((base_url, key), ("https://proxy/v1", "direct"))

    def test_an_explicit_base_url_overrides_the_openrouter_default(self) -> None:
        base_url, _ = provider_settings(
            {"OPENROUTER_API_KEY": "router", "OPENAI_BASE_URL": "https://proxy/v1"}
        )
        self.assertEqual(base_url, "https://proxy/v1")

    def test_names_both_variables_when_neither_is_set(self) -> None:
        with self.assertRaisesRegex(NuggetizerError, "OPENROUTER_API_KEY"):
            provider_settings({})


if __name__ == "__main__":
    unittest.main()
