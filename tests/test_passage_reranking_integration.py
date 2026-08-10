from __future__ import annotations

import asyncio
import unittest
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from trec_ragtime import (
    ChunkerConfig,
    CachedPassageScorer,
    LangChainPassageChunker,
    LocalCrossEncoderConfig,
    LocalCrossEncoderScorer,
    PassageScoreCache,
    PipelineResult,
    RetrievalResult,
    RetrievedDocument,
    RunpodFlashConfig,
    RunpodFlashPassageScorer,
    SearchHit,
    decode_flash_request,
    encode_flash_response,
    rerank_retrieval,
)


def _fake_score(text: str) -> float:
    return float(sum(text.encode("utf-8")) % 997)


class ArtifactStoreFake:
    """Strict test-local structural fake for the future ArtifactStoreLike."""

    class ArtifactValidationError(Exception):
        pass

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.read_calls: list[str] = []
        self.read_validated_calls: list[str] = []
        self.write_calls: list[tuple[str, bytes]] = []
        self.evict_local_calls: list[str] = []

    def read(self, key: str) -> bytes | None:
        self.read_calls.append(key)
        return self.objects.get(key)

    def read_validated(self, key: str, decoder: Callable[[bytes], Any]) -> Any | None:
        self.read_validated_calls.append(key)
        payload = self.objects.get(key)
        return None if payload is None else decoder(payload)

    def write(self, key: str, payload: bytes) -> None:
        self.write_calls.append((key, payload))
        self.objects[key] = payload

    def evict_local(self, key: str) -> None:
        self.evict_local_calls.append(key)
        self.objects.pop(key, None)


class FakeTokenizer:
    def __call__(self, query: str, passage: str, **kwargs: Any) -> dict[str, list[list[int]]]:
        return {"input_ids": [[1] * (len(query) + len(passage) + 2)]}


class FakeCrossEncoder:
    def __init__(self) -> None:
        self.predict_calls = 0

    def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> list[float]:
        self.predict_calls += 1
        return [_fake_score(passage) for _, passage in pairs]


class InMemoryFlashWireRunner:
    async def run_batch(self, payload: bytes) -> bytes:
        request = decode_flash_request(payload)
        rows = [
            {"hash": row["hash"], "score": _fake_score(row["text"])}
            for row in request["rows"]
        ]
        return encode_flash_response(
            rows,
            schema_version=request["schema_version"],
            scorer_fingerprint=request["scorer_fingerprint"],
        )


def _retrieval_fixture() -> RetrievalResult:
    documents = (
        RetrievedDocument(
            rank=1,
            document_id="eng-1",
            score=0.91,
            text="English source discusses shared evidence. It compares practical outcomes for the report.",
            url="https://example.test/eng-1",
            date="2026-01-01",
            language="eng",
            content_source="fixture",
            content_local_cache_hit=True,
        ),
        RetrievedDocument(
            rank=2,
            document_id="spa-1",
            score=0.90,
            text="La fuente española analiza evidencia compartida. Compara resultados prácticos para el informe.",
            url="https://example.test/spa-1",
            date="2026-01-02",
            language="spa",
            content_source="fixture",
            content_local_cache_hit=True,
            english_text="The Spanish source analyzes shared evidence. It compares practical outcomes for the report.",
        ),
        RetrievedDocument(
            rank=3,
            document_id="rus-1",
            score=0.89,
            text="Российский источник анализирует общие данные. Он сравнивает практические результаты для отчёта.",
            url="https://example.test/rus-1",
            date="2026-01-03",
            language="rus",
            content_source="fixture",
            content_local_cache_hit=True,
            english_text="The Russian source analyzes shared evidence. It compares practical outcomes for the report.",
        ),
        RetrievedDocument(
            rank=4,
            document_id="zho-1",
            score=0.88,
            text="中文来源分析共同证据。它比较报告中的实际结果。",
            url="https://example.test/zho-1",
            date="2026-01-04",
            language="zho",
            content_source="fixture",
            content_local_cache_hit=True,
            english_text="The Chinese source analyzes shared evidence. It compares practical outcomes for the report.",
        ),
        RetrievedDocument(
            rank=5,
            document_id="missing-translation",
            score=0.87,
            text="La fuente no tiene traducción inglesa verificada.",
            url="https://example.test/missing-translation",
            date="2026-01-05",
            language="spa",
            content_source="fixture",
            content_local_cache_hit=True,
        ),
        RetrievedDocument(
            rank=6,
            document_id="unresolved",
            score=0.86,
            text=None,
            url=None,
            date=None,
            language="rus",
            content_source=None,
            content_local_cache_hit=False,
        ),
    )
    hits = tuple(SearchHit(document.rank, document.document_id, document.score) for document in documents)
    return RetrievalResult(
        search_result=PipelineResult(
            query="shared evidence and practical outcomes",
            pipeline="ragtime2%100",
            hits=hits,
            processed=True,
            local_cache_hit=True,
            remote_cached=True,
            server_timestamp=None,
            cache_stored_at=None,
        ),
        documents=documents,
        missing_translation_ids=("missing-translation",),
    )


class PassageRerankingIntegrationTests(unittest.TestCase):
    def _chunker(self) -> LangChainPassageChunker:
        return LangChainPassageChunker(
            ChunkerConfig(
                chunk_size=512,
                length_function=len,
                length_function_fingerprint="chars-v1",
            )
        )

    def _local_scorer(self) -> tuple[LocalCrossEncoderScorer, FakeCrossEncoder]:
        policy = LocalCrossEncoderConfig(
            batch_size=2,
            max_length=512,
            query_policy="integration-fake-v1",
        )
        model = FakeCrossEncoder()
        return LocalCrossEncoderScorer(
            policy,
            loader=lambda config: (model, FakeTokenizer()),
        ), model

    def test_public_local_workflow_uses_real_chunking_cache_and_warm_reuse(self) -> None:
        self.assertEqual(
            {
                name
                for name in ("read", "read_validated", "write", "evict_local")
                if callable(getattr(ArtifactStoreFake, name, None))
            },
            {"read", "read_validated", "write", "evict_local"},
        )
        retrieval = _retrieval_fixture()
        chunker = self._chunker()
        local, model = self._local_scorer()
        store = ArtifactStoreFake()
        cache = PassageScoreCache(
            store,
            local,
            chunker.fingerprint,
            "report_text",
            ArtifactStoreFake.ArtifactValidationError,
        )
        cached = CachedPassageScorer(local, cache)

        first = rerank_retrieval(
            retrieval,
            topic_id="topic-integration",
            chunker=chunker,
            scorer=cached,
            representation="report_text",
        )
        predict_calls_after_first = model.predict_calls
        self.assertGreater(predict_calls_after_first, 0)
        writes_after_first = len(store.write_calls)
        second = rerank_retrieval(
            retrieval,
            topic_id="topic-integration",
            chunker=chunker,
            scorer=cached,
            representation="report_text",
        )

        self.assertEqual(first, second)
        self.assertEqual(model.predict_calls, predict_calls_after_first)
        self.assertEqual(first.status_counts["scored"], 4)
        self.assertEqual(first.status_counts["missing_representation"], 1)
        self.assertEqual(first.status_counts["unresolved"], 1)
        expected_statuses = {
            "eng-1": "scored",
            "spa-1": "scored",
            "rus-1": "scored",
            "zho-1": "scored",
            "missing-translation": "missing_representation",
            "unresolved": "unresolved",
        }
        self.assertEqual(
            {document.document_id: document.status for document in first.documents},
            expected_statuses,
        )
        self.assertEqual(
            {document.document_id: document.status for document in second.documents},
            expected_statuses,
        )
        self.assertEqual(len(store.write_calls), writes_after_first)
        self.assertEqual(writes_after_first, 1)
        self.assertTrue(all(document.selected_text for document in first.documents if document.status == "scored"))
        self.assertTrue(all(document.language == "eng" for document in first.documents if document.status == "scored"))
        self.assertEqual(
            [document.document_id for document in first.documents if document.status == "scored"],
            [document.document_id for document in second.documents if document.status == "scored"],
        )

    def test_canonical_multilingual_and_flash_wire_fake_have_identical_rankings(self) -> None:
        retrieval = _retrieval_fixture()
        chunker = self._chunker()
        local, _ = self._local_scorer()
        local_store = ArtifactStoreFake()
        local_cache = PassageScoreCache(
            local_store,
            local,
            chunker.fingerprint,
            "canonical",
            ArtifactStoreFake.ArtifactValidationError,
        )
        local_cached = CachedPassageScorer(local, local_cache)
        local_result = rerank_retrieval(
            retrieval,
            topic_id="topic-canonical",
            chunker=chunker,
            scorer=local_cached,
            representation="canonical",
        )

        config = local.config
        remote = RunpodFlashPassageScorer(
            RunpodFlashConfig(endpoint_id="offline-fake", model_config=config),
            InMemoryFlashWireRunner(),
        )
        self.addCleanup(remote.close)
        remote_result = rerank_retrieval(
            retrieval,
            topic_id="topic-canonical",
            chunker=chunker,
            scorer=remote,
            representation="canonical",
        )

        self.assertEqual(
            [(document.document_id, document.status) for document in local_result.documents],
            [(document.document_id, document.status) for document in remote_result.documents],
        )
        self.assertEqual(
            [document.score for document in local_result.documents],
            [document.score for document in remote_result.documents],
        )
        self.assertEqual(
            [document.winning_content_hash for document in local_result.documents],
            [document.winning_content_hash for document in remote_result.documents],
        )
        self.assertEqual(local_result.status_counts["unresolved"], 1)
        self.assertEqual(local_result.status_counts["missing_representation"], 0)
        self.assertEqual(
            [document.representation for document in local_result.documents],
            ["canonical"] * len(local_result.documents),
        )


if __name__ == "__main__":
    unittest.main()
