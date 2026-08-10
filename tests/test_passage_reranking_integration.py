from __future__ import annotations

import unittest
from collections.abc import Callable
from typing import Any

import trec_ragtime
from trec_ragtime import (
    CachedPassageScorer,
    CrossEncoderPairLengthFunction,
    FlashEndpointBatchRunner,
    FlashPassageWorker,
    LocalCrossEncoderConfig,
    LocalCrossEncoderScorer,
    Passage,
    PassageChunker,
    PassageChunkingError,
    PassageEvidence,
    PassageScoreCache,
    RerankedDocument,
    RerankingResult,
    RunpodFlashConfig,
    RunpodFlashPassageScorer,
    build_flash_worker,
    rerank_retrieval,
    sha256_bytes,
    validate_passages,
)
from trec_ragtime.flash_passage_scorer import decode_flash_request, encode_flash_response
from trec_ragtime.models import PipelineResult, RetrievalResult, RetrievedDocument, SearchHit


_INTEGRATION_QUERY = "shared evidence and practical outcomes"
_SUPPORTED_BASELINE_PUBLIC_EXPORTS = {
    "DEFAULT_CONTENT_COLLECTION",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_FLASH_BATCH_SIZE",
    "DEFAULT_FLASH_CONCURRENCY",
    "DEFAULT_FLASH_GPU",
    "DEFAULT_FLASH_IDLE_TIMEOUT",
    "DEFAULT_FLASH_REQUEST_TIMEOUT",
    "DEFAULT_FLASH_WORKER_NAME",
    "DEFAULT_HF_BASE_URL",
    "DEFAULT_HF_CONFIG",
    "DEFAULT_HF_DATASET",
    "DEFAULT_HF_DOCUMENT_CACHE_TTL_SECONDS",
    "DEFAULT_HF_MT_CONFIG",
    "DEFAULT_HF_MT_SPLITS",
    "DEFAULT_HF_SPLITS",
    "DEFAULT_MAX_CONTENT_FALLBACKS",
    "DEFAULT_MAX_LENGTH",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_MODEL_REVISION",
    "DEFAULT_PIPELINE",
    "DEFAULT_QUERY_FORMAT",
    "DEFAULT_QUERY_POLICY",
    "DEFAULT_SCHEMA_VERSION",
    "DEFAULT_TOKENIZER_NAME",
    "DEFAULT_TOKENIZER_REVISION",
    "CachedPassageScorer",
    "CrossEncoderPairLengthFunction",
    "FLASH_PAYLOAD_LIMIT_BYTES",
    "FlashEndpointBatchRunner",
    "FlashPassageWorker",
    "FlashScoringError",
    "Boundary",
    "LocalCrossEncoderConfig",
    "LocalCrossEncoderScorer",
    "Document",
    "DocumentCorpus",
    "HuggingFaceApiError",
    "HuggingFaceConfigurationError",
    "HuggingFaceCorpus",
    "HuggingFaceDocument",
    "HuggingFaceError",
    "HuggingFaceHydrationError",
    "HuggingFaceProtocolError",
    "HydrationFailure",
    "HydrationResult",
    "LocalCorpusBuildError",
    "LocalCorpusConfigurationError",
    "LocalCorpusError",
    "LocalCorpusIndex",
    "LocalCorpusMetadata",
    "LocalCorpusReadError",
    "LocalCorpusValidationError",
    "Passage",
    "PassageChunker",
    "PassageChunkingError",
    "PassageEvidence",
    "PassageScoreCache",
    "PassageScorer",
    "PassageScoringError",
    "PipelineResult",
    "PROJECT_CACHE_DIR_ENV",
    "RagtimeApiError",
    "RagtimeClient",
    "RagtimeConfigurationError",
    "RagtimeError",
    "RagtimeProtocolError",
    "RagtimeRetriever",
    "RetrievalResult",
    "RetrievedDocument",
    "RerankedDocument",
    "RerankingError",
    "RerankingResult",
    "RunpodFlashConfig",
    "RunpodFlashPassageScorer",
    "Representation",
    "SQLiteResponseCache",
    "SearchHit",
    "WIRE_SCHEMA_VERSION",
    "build_flash_worker",
    "build_from_parquet",
    "decode_flash_request",
    "decode_flash_response",
    "default_cache_path",
    "default_local_corpus_path",
    "default_project_cache_dir",
    "ensure_private_cache_dir",
    "encode_flash_request",
    "encode_flash_response",
    "max_passage_v1",
    "rerank_retrieval",
    "sha256_bytes",
    "validate_passage_scores",
    "validate_passages",
}
_DELETED_ISSUE15_PUBLIC_EXPORTS = {
    "CACHE_SCHEMA_VERSION",
    "ChunkerConfig",
    "FlashWireValidationError",
    "LangChainPassageChunker",
    "PassageScoreArtifactValidationError",
    "PassageScoreEnvelope",
    "PassageScoreValidationError",
    "PassageValidationError",
    "RERANKED_RECORD_FILENAME",
    "RERANKING_SCHEMA",
    "RERANKING_SCHEMA_VERSION",
    "RerankedPassage",
    "ScoreEnvelope",
    "decode_flash_batch",
    "decode_flash_batch_response",
    "decode_passage_score_envelope",
    "decode_score_envelope",
    "decode_score_map",
    "effective_score_fingerprint",
    "encode_flash_batch",
    "encode_flash_batch_response",
    "encode_passage_score_envelope",
    "encode_score_envelope",
    "encode_score_map",
    "query_hash",
    "score_object_key",
    "write_reranking_artifact",
}


def _fake_score(text: str) -> float:
    return float(sum(text.encode("utf-8")) % 997)


def _winning_content_hash(row: RerankedDocument) -> bytes | None:
    matching_evidence = [
        evidence for evidence in row.evidence if evidence.score == row.score
    ]
    if not matching_evidence:
        return None
    return min(
        matching_evidence,
        key=lambda evidence: (
            evidence.start_char,
            evidence.end_char,
            evidence.content_hash,
        ),
    ).content_hash


class ArtifactStoreFake:
    """Strict test-local structural fake for the future ArtifactStoreLike seam."""

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
    QUERY_NO_SPECIAL_TOKENS = 6
    PAIR_SPECIAL_TOKENS = 4
    SINGLE_SEQUENCE_KWARGS = {
        "add_special_tokens": False,
        "truncation": False,
        "return_length": True,
    }
    PAIR_LENGTH_KWARGS = {
        "truncation": False,
        "return_length": True,
    }
    PAIR_VALIDATION_KWARGS = {
        "truncation": False,
        "max_length": 512,
        "return_length": True,
    }

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []
        self.special_token_calls: list[bool] = []

    @staticmethod
    def passage_no_special_token_count(passage: str) -> int:
        if not passage:
            return 0
        if set(passage) == {"中"}:
            return len(passage) + 1
        return len(passage)

    @classmethod
    def pair_token_count(cls, query: str, passage: str) -> int:
        if query != _INTEGRATION_QUERY:
            raise AssertionError(f"unexpected pair query: {query!r}")
        return (
            cls.QUERY_NO_SPECIAL_TOKENS
            + cls.passage_no_special_token_count(passage)
            + cls.PAIR_SPECIAL_TOKENS
        )

    def __call__(self, *texts: str, **kwargs: Any) -> dict[str, object]:
        self.calls.append((texts, dict(kwargs)))
        if len(texts) == 1:
            if kwargs != self.SINGLE_SEQUENCE_KWARGS:
                raise AssertionError(f"unexpected single-sequence kwargs: {kwargs!r}")
            (text,) = texts
            count = (
                self.QUERY_NO_SPECIAL_TOKENS
                if text == _INTEGRATION_QUERY
                else self.passage_no_special_token_count(text)
            )
        elif len(texts) == 2:
            if kwargs not in (self.PAIR_LENGTH_KWARGS, self.PAIR_VALIDATION_KWARGS):
                raise AssertionError(f"unexpected pair kwargs: {kwargs!r}")
            query, passage = texts
            count = self.pair_token_count(query, passage)
        else:
            raise AssertionError(f"unexpected tokenizer arity: {len(texts)}")
        return {"input_ids": list(range(count)), "length": [count]}

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        self.special_token_calls.append(pair)
        if pair is not True:
            raise AssertionError("only pair special-token counting is expected")
        return self.PAIR_SPECIAL_TOKENS


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
    hits = tuple(
        SearchHit(document.rank, document.document_id, document.score)
        for document in documents
    )
    return RetrievalResult(
        search_result=PipelineResult(
            query=_INTEGRATION_QUERY,
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


def _dense_cjk_retrieval_fixture() -> RetrievalResult:
    text = "中" * 502
    document = RetrievedDocument(
        rank=1,
        document_id="dense-cjk",
        score=0.91,
        text=text,
        url="https://example.test/dense-cjk",
        date="2026-01-06",
        language="zho",
        content_source="fixture",
        content_local_cache_hit=True,
    )
    return RetrievalResult(
        search_result=PipelineResult(
            query=_INTEGRATION_QUERY,
            pipeline="ragtime2%100",
            hits=(SearchHit(document.rank, document.document_id, document.score),),
            processed=True,
            local_cache_hit=True,
            remote_cached=True,
            server_timestamp=None,
            cache_stored_at=None,
        ),
        documents=(document,),
    )


class PassageRerankingIntegrationTests(unittest.TestCase):
    def _policy(self) -> LocalCrossEncoderConfig:
        return LocalCrossEncoderConfig(
            batch_size=2,
            max_length=512,
            query_policy="integration-fake-v1",
        )

    def _chunker(
        self,
        policy: LocalCrossEncoderConfig,
        query: str,
        tokenizer: FakeTokenizer,
    ) -> tuple[PassageChunker, CrossEncoderPairLengthFunction]:
        pair_length = CrossEncoderPairLengthFunction(
            policy,
            query,
            tokenizer_loader=lambda _: tokenizer,
        )
        return PassageChunker(pair_length, overlap_tokens=64), pair_length

    def _local_scorer(
        self,
        policy: LocalCrossEncoderConfig,
        tokenizer: FakeTokenizer,
    ) -> tuple[LocalCrossEncoderScorer, FakeCrossEncoder]:
        model = FakeCrossEncoder()
        return LocalCrossEncoderScorer(
            policy,
            loader=lambda _: (model, tokenizer),
        ), model

    def test_public_surface_has_no_deleted_compatibility_symbols(self) -> None:
        public_exports = set(trec_ragtime.__all__)
        self.assertEqual(len(trec_ragtime.__all__), len(public_exports))
        self.assertTrue(_SUPPORTED_BASELINE_PUBLIC_EXPORTS <= public_exports)
        self.assertTrue(
            _DELETED_ISSUE15_PUBLIC_EXPORTS.isdisjoint(public_exports)
        )
        for deleted_name in _DELETED_ISSUE15_PUBLIC_EXPORTS:
            self.assertFalse(hasattr(trec_ragtime, deleted_name), deleted_name)

    def test_offline_multilingual_reranking_cache_and_flash_parity(self) -> None:
        retrieval = _retrieval_fixture()
        policy = self._policy()
        report_tokenizer = FakeTokenizer()
        chunker, pair_length = self._chunker(
            policy,
            retrieval.query,
            report_tokenizer,
        )
        local, model = self._local_scorer(policy, report_tokenizer)
        store = ArtifactStoreFake()
        cache = PassageScoreCache(
            store,
            local,
            chunker_fingerprint=chunker.fingerprint,
            representation="report_text",
            artifact_validation_error=ArtifactStoreFake.ArtifactValidationError,
        )

        first = rerank_retrieval(
            retrieval,
            chunker=chunker,
            scorer=CachedPassageScorer(cache),
            representation="report_text",
        )
        model_calls = model.predict_calls
        write_calls = len(store.write_calls)
        second = rerank_retrieval(
            retrieval,
            chunker=chunker,
            scorer=CachedPassageScorer(cache),
            representation="report_text",
        )

        self.assertEqual(first, second)
        self.assertGreater(model_calls, 0)
        self.assertEqual(model.predict_calls, model_calls)
        self.assertEqual(len(store.write_calls), write_calls)
        self.assertEqual(first.query, retrieval.query)
        self.assertEqual(first.documents[-1].status, "unresolved")
        self.assertIn(
            "missing-translation",
            {row.document.document_id for row in first.documents},
        )
        self.assertEqual(
            {
                row.document.document_id: row.status
                for row in first.documents
            },
            {
                "eng-1": "scored",
                "spa-1": "scored",
                "rus-1": "scored",
                "zho-1": "scored",
                "missing-translation": "missing_representation",
                "unresolved": "unresolved",
            },
        )
        self.assertTrue(
            all(
                row.status == "scored" or not row.evidence
                for row in first.documents
            )
        )

        for row in first.documents:
            if row.status != "scored":
                continue
            selected_text = row.document.report_text
            self.assertIsNotNone(selected_text)
            assert selected_text is not None
            passages = chunker.chunk(
                document_id=row.document.document_id,
                canonical_text=row.document.text or "",
                selected_text=selected_text,
                representation="report_text",
                language="eng",
            )
            validate_passages(selected_text, passages)
            self.assertEqual(
                [
                    (evidence.start_char, evidence.end_char, evidence.content_hash)
                    for evidence in row.evidence
                ],
                [
                    (passage.start_char, passage.end_char, passage.content_hash)
                    for passage in passages
                ],
            )
            for evidence in row.evidence:
                exact_text = selected_text[evidence.start_char : evidence.end_char]
                self.assertEqual(evidence.content_hash, sha256_bytes(exact_text))

        canonical_tokenizer = FakeTokenizer()
        canonical_chunker, _ = self._chunker(
            policy,
            retrieval.query,
            canonical_tokenizer,
        )
        canonical_local, _ = self._local_scorer(policy, canonical_tokenizer)
        canonical_result = rerank_retrieval(
            retrieval,
            chunker=canonical_chunker,
            scorer=canonical_local,
            representation="canonical",
        )

        flash = RunpodFlashPassageScorer(
            RunpodFlashConfig(endpoint_id="offline-fake", model_config=policy),
            InMemoryFlashWireRunner(),
        )
        self.addCleanup(flash.close)
        flash_result = rerank_retrieval(
            retrieval,
            chunker=canonical_chunker,
            scorer=flash,
            representation="canonical",
        )
        self.assertEqual(
            [
                (row.document.document_id, row.status, row.score)
                for row in canonical_result.documents
            ],
            [
                (row.document.document_id, row.status, row.score)
                for row in flash_result.documents
            ],
        )
        self.assertEqual(
            [
                tuple(
                    (
                        evidence.start_char,
                        evidence.end_char,
                        evidence.score,
                        evidence.content_hash,
                    )
                    for evidence in row.evidence
                )
                for row in canonical_result.documents
            ],
            [
                tuple(
                    (
                        evidence.start_char,
                        evidence.end_char,
                        evidence.score,
                        evidence.content_hash,
                    )
                    for evidence in row.evidence
                )
                for row in flash_result.documents
            ],
        )
        self.assertEqual(
            [
                _winning_content_hash(row)
                for row in canonical_result.documents
            ],
            [
                _winning_content_hash(row)
                for row in flash_result.documents
            ],
        )

    def test_contiguous_cjk_stays_within_exact_pair_budget(self) -> None:
        retrieval = _dense_cjk_retrieval_fixture()
        policy = self._policy()
        tokenizer = FakeTokenizer()
        chunker, pair_length = self._chunker(policy, retrieval.query, tokenizer)
        selected_text = retrieval.documents[0].text
        self.assertIsNotNone(selected_text)
        assert selected_text is not None

        self.assertEqual(pair_length(selected_text), 513)
        self.assertEqual(pair_length.passage_length_function(selected_text), 503)
        self.assertEqual(pair_length.max_passage_length, 502)
        passages = chunker.chunk(
            document_id="dense-cjk",
            canonical_text=selected_text,
            selected_text=selected_text,
            representation="canonical",
            language="zho",
        )
        validate_passages(selected_text, passages)
        self.assertEqual(
            [(passage.start_char, passage.end_char) for passage in passages],
            [(0, 251), (251, 502)],
        )
        self.assertEqual("".join(passage.text for passage in passages), selected_text)
        self.assertEqual(
            [passage.boundary for passage in passages],
            ["oversized_sentence", "oversized_sentence"],
        )
        self.assertTrue(
            all(
                pair_length(passage.text) <= policy.max_length
                for passage in passages
            )
        )
        self.assertIn(
            ((_INTEGRATION_QUERY,), FakeTokenizer.SINGLE_SEQUENCE_KWARGS),
            tokenizer.calls,
        )
        self.assertIn(
            ((selected_text,), FakeTokenizer.SINGLE_SEQUENCE_KWARGS),
            tokenizer.calls,
        )
        self.assertEqual(tokenizer.special_token_calls, [True])


if __name__ == "__main__":
    unittest.main()
