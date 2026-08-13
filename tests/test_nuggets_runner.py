"""Runner orchestration, manifest hygiene, and the reranking projection."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from trec_ragtime.nugget_cache import NuggetArtifactCache, NuggetInput
from trec_ragtime.nuggets import (
    FacetEvidence,
    NuggetClusterer,
    normalize_claim,
)
from trec_ragtime.nuggets_runner import (
    TopicOutcome,
    TopicWork,
    build_fingerprint,
    evidence_from_reranking,
    prepare_nugget_request,
    run_topic,
    run_topics,
    write_manifest,
)

from tests.test_nugget_cache import MemoryArtifactStore
from tests.test_nuggets import ScriptedBackend, candidate, fixed_embedder


CLAIM = "grain exports fell"
# Fingerprinted from the same kind of backend the factory below returns. Using a
# NuggetizerBackend here while executing a ScriptedBackend keyed every fixture
# bundle under a configuration the test never ran — the exact divergence
# run_topic now rejects.
FINGERPRINT = build_fingerprint(
    ScriptedBackend(claims=((CLAIM, "vital"),), supported={}),
    NuggetClusterer(embed=fixed_embedder({})),
    provider_base_url="https://provider/v1",
)


def facets(topic: str) -> tuple[FacetEvidence, ...]:
    return (
        FacetEvidence(
            facet_index=0,
            query=f"{topic} grain",
            candidates=(
                candidate("grain passage", document_id=f"{topic}-a", rank=1, score=0.9),
                candidate("port passage", document_id=f"{topic}-b", rank=2, score=0.4),
            ),
        ),
    )


def work(topic: str = "2000") -> TopicWork:
    return TopicWork(
        topic_id=topic,
        query="grain exports",
        representation="report_text",
        facets=facets(topic),
        inputs=(
            NuggetInput(
                facet_index=0,
                query=f"{topic} grain",
                object_key=f"retrieval/v1/aa/bb/{topic}.mpk",
                request_digest=hashlib.sha256(topic.encode()).digest(),
            ),
        ),
        passage_source_fingerprint="langchain-spacy-v1/deadbeef",
    )


def backend_factory(supported: Any = ("grain passage",)) -> Any:
    return lambda: ScriptedBackend(
        claims=((CLAIM, "vital"),), supported={CLAIM: supported}
    )


def clusterer() -> NuggetClusterer:
    return NuggetClusterer(embed=fixed_embedder({normalize_claim(CLAIM): [1.0]}))


class RunTopicTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.store = MemoryArtifactStore()
        self.cache = NuggetArtifactCache(self.store, clock=lambda: 1.0)

    def _run(self, **changes: Any) -> TopicOutcome:
        options: dict[str, Any] = {
            "cache": self.cache,
            "fingerprint": FINGERPRINT,
            "backend_factory": backend_factory(),
            "clusterer": clusterer(),
        }
        options.update(changes)
        return run_topic(work(), **options)

    def test_request_can_be_prepared_without_a_backend(self) -> None:
        rows, request = prepare_nugget_request(work(), FINGERPRINT)
        self.assertEqual(tuple(row.evidence.alias for row in rows), ("e001", "e002"))
        self.assertEqual(request.topic_id, "2000")
        self.assertEqual(request.inputs, work().inputs)

    def test_writes_and_then_verifies_the_durable_object(self) -> None:
        outcome = self._run()
        self.assertEqual(outcome.status, "written")
        self.assertEqual(outcome.nugget_count, 1)
        self.assertEqual(outcome.evidence_count, 2)
        # The local candidate is dropped and the typed object reread: one miss
        # before the write, one verification read after the eviction.
        self.assertIn(outcome.object_key, self.store.evictions)
        self.assertEqual(self.store.reads.count(outcome.object_key), 2)

    def test_a_reworded_topic_query_is_not_a_cache_hit(self) -> None:
        """The query drives create() and assign(), so it must be in the identity."""

        first = self._run()
        self.assertEqual(first.status, "written")
        reworded = run_topic(
            TopicWork(
                topic_id=work().topic_id,
                query="what happened to port closures",
                representation=work().representation,
                facets=work().facets,
                inputs=work().inputs,
                passage_source_fingerprint=work().passage_source_fingerprint,
            ),
            cache=self.cache,
            fingerprint=FINGERPRINT,
            backend_factory=backend_factory(),
            clusterer=clusterer(),
        )
        self.assertEqual(reworded.status, "written")
        self.assertNotEqual(reworded.object_key, first.object_key)

    def test_a_warm_topic_costs_no_provider_calls(self) -> None:
        self._run()

        def refuse() -> Any:
            raise AssertionError("a cached topic must not build a backend")

        outcome = self._run(backend_factory=refuse)
        self.assertEqual(outcome.status, "cached")
        self.assertEqual(outcome.nugget_count, 1)

    def test_a_backend_the_fingerprint_does_not_describe_is_refused(self) -> None:
        """The key must never name a configuration that did not run.

        build_fingerprint takes a backend instance while run_topic builds its
        own from the factory, so the two can differ. That would publish a bundle
        under another backend's identity and let a later run read it as a hit.
        """

        def other() -> Any:
            return ScriptedBackend(
                claims=((CLAIM, "vital"),),
                supported={CLAIM: ("grain passage",)},
                identity="a-different-backend",
            )

        outcome = self._run(backend_factory=other)
        self.assertEqual(outcome.status, "failed")
        self.assertIn("backend identity", outcome.reason or "")
        self.assertEqual(self.store.objects, {})

    def test_an_unsupported_topic_fails_and_stores_nothing(self) -> None:
        outcome = self._run(backend_factory=backend_factory(supported=()))
        self.assertEqual(outcome.status, "failed")
        self.assertIn("NuggetError", outcome.reason or "")
        self.assertEqual(self.store.objects, {})

    def test_a_refused_write_is_a_failure_not_a_success(self) -> None:
        self.store.write_result = False
        outcome = self._run()
        self.assertEqual(outcome.status, "failed")
        self.assertEqual(outcome.reason, "artifact write refused")

    def test_a_topic_without_passages_fails_before_any_provider_call(self) -> None:
        def refuse() -> Any:
            raise AssertionError("an empty topic must not build a backend")

        empty = TopicWork(
            topic_id="2000",
            query="q",
            representation="report_text",
            facets=(),
            inputs=(),
            passage_source_fingerprint="fp",
        )
        outcome = run_topic(
            empty,
            cache=self.cache,
            fingerprint=FINGERPRINT,
            backend_factory=refuse,
            clusterer=clusterer(),
        )
        self.assertEqual(outcome.status, "failed")
        self.assertEqual(outcome.reason, "no ranked passages")

    def test_the_failure_reason_carries_no_payload(self) -> None:
        secret = "sk-live-abcdef0123456789"

        def leaky() -> Any:
            raise RuntimeError(f"provider rejected key {secret}\nwith trailing detail" * 40)

        outcome = self._run(backend_factory=leaky)
        self.assertEqual(outcome.status, "failed")
        self.assertLessEqual(len(outcome.reason or ""), 220)
        self.assertNotIn("\n", outcome.reason or "")


class RunTopicsTestCase(unittest.TestCase):
    def test_processes_distinct_topics_in_parallel(self) -> None:
        store = MemoryArtifactStore()
        cache = NuggetArtifactCache(store, clock=lambda: 1.0)
        items = [work(topic) for topic in ("2000", "2001", "2002", "2003")]
        outcomes = run_topics(
            items,
            cache=cache,
            fingerprint=FINGERPRINT,
            backend_factory=backend_factory(),
            clusterer=clusterer(),
            concurrency=4,
        )
        self.assertEqual([outcome.status for outcome in outcomes], ["written"] * 4)
        self.assertEqual(
            [outcome.topic_id for outcome in outcomes], ["2000", "2001", "2002", "2003"]
        )
        # One object per topic; no worker patched another worker's bundle.
        self.assertEqual(len(store.objects), 4)

    def test_returns_nothing_for_no_work(self) -> None:
        cache = NuggetArtifactCache(MemoryArtifactStore(), clock=lambda: 1.0)
        self.assertEqual(
            run_topics(
                [],
                cache=cache,
                fingerprint=FINGERPRINT,
                backend_factory=backend_factory(),
                clusterer=clusterer(),
            ),
            (),
        )

    def test_rejects_an_unusable_concurrency(self) -> None:
        cache = NuggetArtifactCache(MemoryArtifactStore(), clock=lambda: 1.0)
        for concurrency in (0, -1, True):
            with self.subTest(concurrency=concurrency):
                with self.assertRaises(ValueError):
                    run_topics(
                        [work()],
                        cache=cache,
                        fingerprint=FINGERPRINT,
                        backend_factory=backend_factory(),
                        clusterer=clusterer(),
                        concurrency=concurrency,
                    )


class ManifestTestCase(unittest.TestCase):
    def test_is_atomic_owner_only_and_sanitized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "manifest.json"
            write_manifest(
                path,
                [TopicOutcome("2000", "written", "nuggets/v1/a/bb/c.mpk", 3, 12)],
                model="test-model",
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["model"], "test-model")
            self.assertEqual(payload["topics"][0]["nugget_count"], 3)
            mode = stat.S_IMODE(os.stat(path).st_mode)
            self.assertEqual(mode & 0o077, 0)
            # No stray temporary files survive.
            self.assertEqual([p.name for p in path.parent.iterdir()], ["manifest.json"])

    def test_carries_no_query_or_passage_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_manifest(path, [TopicOutcome("2000", "written", "key", 1, 2)])
            body = path.read_text(encoding="utf-8")
            for leak in ("grain passage", "grain exports", "sk-"):
                self.assertNotIn(leak, body)


class RerankingProjectionTestCase(unittest.TestCase):
    """Structural stand-ins matching the reranking stage's published records."""

    def _result(
        self, representation: str = "report_text", document_id: str = "doc-a"
    ) -> Any:
        passage = SimpleNamespace(
            document_id=document_id,
            start_char=6,
            end_char=13,
            score=0.75,
            content_hash=hashlib.sha256(b"passage").digest(),
        )
        document = SimpleNamespace(
            document_id=document_id,
            language="spa",
            text="ORIG: cargamento original",
            english_text="hello passage text",
            report_text="hello passage text",
        )
        return SimpleNamespace(
            query="grain",
            representation=representation,
            documents=(
                SimpleNamespace(
                    document=document,
                    score=0.75,
                    evidence=(passage,),
                    reranked_rank=1,
                    status="scored",
                ),
                SimpleNamespace(
                    document=document,
                    score=None,
                    evidence=(),
                    reranked_rank=2,
                    status="missing_representation",
                ),
            ),
        )

    def test_slices_text_from_the_selected_representation(self) -> None:
        result = self._result()
        facets = evidence_from_reranking([result], queries=["grain"], facet_indices=[0])
        self.assertEqual(len(facets), 1)
        prepared = facets[0].candidates
        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared[0].text, "passage")
        self.assertEqual(prepared[0].document_id, "doc-a")
        self.assertEqual(prepared[0].start_char, 6)
        self.assertEqual(prepared[0].passage_score, 0.75)
        # An English representation is reported as English regardless of source.
        self.assertEqual(prepared[0].language, "eng")

    def test_uses_the_source_language_for_canonical_text(self) -> None:
        result = self._result(representation="canonical")
        prepared = evidence_from_reranking([result], queries=["grain"], facet_indices=[0])[0].candidates
        self.assertEqual(prepared[0].language, "spa")
        self.assertEqual(prepared[0].text, "cargame")

    def test_indexes_facets_by_position(self) -> None:
        facets = evidence_from_reranking(
            [self._result(), self._result()], queries=["a", "b"], facet_indices=[0, 1]
        )
        self.assertEqual([facet.facet_index for facet in facets], [0, 1])
        self.assertEqual([facet.query for facet in facets], ["a", "b"])

    def test_requires_a_query_per_result(self) -> None:
        with self.assertRaises(ValueError):
            evidence_from_reranking([self._result()], queries=[], facet_indices=[])

    def test_keeps_plan_indices_when_a_facet_retrieval_is_missing(self) -> None:
        """Facet 1 has no cached retrieval; facet 2 must not be renumbered to 1."""

        facets = evidence_from_reranking(
            [self._result(document_id="doc-a"), self._result(document_id="doc-b")],
            queries=["facet zero", "facet two"],
            facet_indices=[0, 2],
        )
        self.assertEqual([facet.facet_index for facet in facets], [0, 2])
        # The plan's numbering must reach the persisted evidence rows, since
        # that is what the cache key and facet coverage are built from.
        from trec_ragtime.nuggets import build_evidence_table

        rows = build_evidence_table(facets, budget=24)
        self.assertEqual(
            sorted({row.evidence.facet_index for row in rows}), [0, 2]
        )

    def test_rejects_duplicate_facet_indices(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            evidence_from_reranking(
                [self._result(), self._result()],
                queries=["a", "b"],
                facet_indices=[1, 1],
            )


if __name__ == "__main__":
    unittest.main()


class RealRerankingIntegrationTestCase(unittest.TestCase):
    """Drive the merged reranking stage, not a stand-in, into the nugget stage."""

    query = "grain exports"

    class WordPairLength:
        fingerprint = "nuggets-integration-v1"
        max_length = 40
        max_passage_length = 20
        passage_length_function = staticmethod(lambda text: len(text.split()))

        def __call__(self, text: str) -> int:
            return len(text.split())

    class ConstantScorer:
        fingerprint = "constant-scorer-v1"

        def score(self, query: str, passages: Any) -> dict[bytes, float]:
            return {
                passage.content_hash: 1.0 - index / 100
                for index, passage in enumerate(passages)
            }

    def _reranked(self) -> Any:
        from trec_ragtime.models import (
            PipelineResult,
            RetrievalResult,
            RetrievedDocument,
            SearchHit,
        )
        from trec_ragtime.passages import PassageChunker
        from trec_ragtime.reranking import rerank_retrieval

        english = (
            "Grain exports through the southern corridor fell during the quarter. "
            "Two of the three deep-water ports remained closed to bulk carriers. "
            "Insurers withdrew cover for vessels calling at the northern terminal."
        )
        document = RetrievedDocument(
            rank=1,
            document_id="doc-a",
            score=9.5,
            text="Texto original en espanol sobre el corredor sur.",
            url="https://example.test/doc-a",
            date="2026-01-01",
            language="spa",
            content_source="local_corpus",
            content_local_cache_hit=True,
            english_text=english,
        )
        retrieval = RetrievalResult(
            search_result=PipelineResult(
                query=self.query,
                pipeline="ragtime2",
                hits=(SearchHit(1, "doc-a", 9.5),),
                processed=True,
                local_cache_hit=True,
                remote_cached=True,
                server_timestamp=None,
                cache_stored_at=None,
            ),
            documents=(document,),
        )
        return english, rerank_retrieval(
            retrieval,
            chunker=PassageChunker(self.WordPairLength(), overlap_tokens=0),
            scorer=self.ConstantScorer(),
            representation="report_text",
        )

    def test_projects_real_passages_with_exact_offsets(self) -> None:
        english, result = self._reranked()
        facets = evidence_from_reranking([result], queries=[self.query], facet_indices=[0])
        prepared = facets[0].candidates
        self.assertGreater(len(prepared), 1, "the chunker should split this document")
        for row in prepared:
            # The slice this stage sends the model is the exact reranked span.
            self.assertEqual(english[row.start_char : row.end_char], row.text)
            self.assertEqual(
                hashlib.sha256(row.text.encode("utf-8")).digest(), row.content_hash
            )
            self.assertEqual(row.language, "eng")
            self.assertEqual(row.representation, "report_text")

    def test_real_passages_flow_through_to_a_sealed_bundle(self) -> None:
        _, result = self._reranked()
        facets = evidence_from_reranking([result], queries=[self.query], facet_indices=[0])
        claim = "grain exports fell in the quarter"
        supported = facets[0].candidates[0].text
        backend = lambda: ScriptedBackend(  # noqa: E731
            claims=((claim, "vital"),), supported={claim: (supported,)}
        )
        store = MemoryArtifactStore()
        outcome = run_topic(
            TopicWork(
                topic_id="2000",
                query=self.query,
                representation="report_text",
                facets=facets,
                inputs=(
                    NuggetInput(
                        facet_index=0,
                        query=self.query,
                        object_key="retrieval/v1/aa/bb/2000.mpk",
                        request_digest=hashlib.sha256(b"2000").digest(),
                    ),
                ),
                passage_source_fingerprint="chunker/scorer",
            ),
            cache=NuggetArtifactCache(store, clock=lambda: 1.0),
            fingerprint=FINGERPRINT,
            backend_factory=backend,
            clusterer=NuggetClusterer(
                embed=fixed_embedder({normalize_claim(claim): [1.0]})
            ),
        )
        self.assertEqual(outcome.status, "written")
        self.assertEqual(outcome.nugget_count, 1)
        self.assertEqual(outcome.evidence_count, len(facets[0].candidates))


class FacetNumberingTestCase(unittest.TestCase):
    """A renumbered facet must not reach a cache key."""

    def _work(self, facet_indices: Any, input_indices: Any) -> TopicWork:
        return TopicWork(
            topic_id="2000",
            query="grain exports",
            representation="report_text",
            facets=tuple(
                FacetEvidence(
                    facet_index=index,
                    query=f"facet {plan}",
                    candidates=(candidate("grain passage", document_id=f"d-{plan}"),),
                )
                for index, plan in zip(facet_indices, (0, 2), strict=True)
            ),
            inputs=tuple(
                NuggetInput(
                    facet_index=index,
                    query=f"facet {plan}",
                    object_key=f"retrieval/v1/aa/bb/{plan}.mpk",
                    request_digest=hashlib.sha256(str(plan).encode()).digest(),
                )
                for index, plan in zip(input_indices, (0, 2), strict=True)
            ),
            passage_source_fingerprint="fp",
        )

    def _run(self, work: TopicWork) -> TopicOutcome:
        return run_topic(
            work,
            cache=NuggetArtifactCache(MemoryArtifactStore(), clock=lambda: 1.0),
            fingerprint=FINGERPRINT,
            backend_factory=backend_factory(),
            clusterer=clusterer(),
        )

    def test_plan_numbering_on_both_sides_is_accepted(self) -> None:
        self.assertEqual(self._run(self._work([0, 2], [0, 2])).status, "written")

    def test_facets_renumbered_against_their_inputs_are_rejected(self) -> None:
        outcome = self._run(self._work([0, 1], [0, 2]))
        self.assertEqual(outcome.status, "failed")
        self.assertIn("facet set", outcome.reason or "")

    def test_inputs_renumbered_against_their_facets_are_rejected(self) -> None:
        outcome = self._run(self._work([0, 2], [0, 1]))
        self.assertEqual(outcome.status, "failed")
        self.assertIn("facet set", outcome.reason or "")


class ManifestSchemaTestCase(unittest.TestCase):
    """The manifest is a published boundary, so its key set is a contract."""

    def test_publishes_exactly_the_declared_topic_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_manifest(path, [TopicOutcome("2000", "written", "key", 1, 2)])
            topic = json.loads(path.read_text(encoding="utf-8"))["topics"][0]
        # Adding a TopicOutcome field must be a deliberate decision to publish
        # it, not an automatic consequence of dataclasses.asdict.
        self.assertEqual(
            set(topic),
            {
                "topic_id",
                "status",
                "object_key",
                "nugget_count",
                "evidence_count",
                "reason",
            },
        )
