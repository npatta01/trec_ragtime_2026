from __future__ import annotations

import unittest
from collections.abc import Callable, Mapping, Sequence
from dataclasses import FrozenInstanceError, fields, is_dataclass

from trec_ragtime.models import (
    PipelineResult,
    RetrievalResult,
    RetrievedDocument,
    SearchHit,
)
from trec_ragtime.passage_score_cache import CachedPassageScorer, PassageScoreCache
from trec_ragtime.passages import Passage, sha256_bytes
from trec_ragtime.reranking import (
    PassageEvidence,
    RerankedDocument,
    RerankingError,
    RerankingResult,
    rerank_retrieval,
)


def _passage(
    document_id: str,
    representation: str,
    language: str,
    text: str,
    start_char: int,
    end_char: int,
) -> Passage:
    return Passage(
        document_id=document_id,
        representation=representation,  # type: ignore[arg-type]
        language=language,
        text=text,
        start_char=start_char,
        end_char=end_char,
        content_hash=sha256_bytes(text),
        boundary="sentence",
    )


def _passages(
    document_id: str,
    representation: str,
    language: str,
    specs: Sequence[tuple[str, int, int]],
) -> tuple[Passage, ...]:
    return tuple(
        _passage(
            document_id,
            representation,
            language,
            text,
            start_char,
            end_char,
        )
        for text, start_char, end_char in specs
    )


class RecordingChunker:
    fingerprint = "chunker-v1"

    def __init__(
        self,
        passages: Mapping[str, Sequence[tuple[str, int, int]]] | None = None,
    ) -> None:
        self.passages = dict(passages or {})
        self.calls: list[dict[str, str]] = []

    def chunk(
        self,
        *,
        document_id: str,
        canonical_text: str,
        selected_text: str,
        representation: str,
        language: str,
    ) -> tuple[Passage, ...]:
        self.calls.append(
            {
                "document_id": document_id,
                "canonical_text": canonical_text,
                "selected_text": selected_text,
                "representation": representation,
                "language": language,
            }
        )
        specs = self.passages.get(
            document_id,
            ((selected_text, 0, len(selected_text)),),
        )
        return _passages(document_id, representation, language, specs)


class EmptyChunker(RecordingChunker):
    def chunk(self, **kwargs: str) -> tuple[Passage, ...]:
        if kwargs["document_id"] not in self.passages:
            self.calls.append(dict(kwargs))
            return ()
        return super().chunk(**kwargs)


class RecordingScorer:
    fingerprint = "scorer-v1"

    def __init__(self, scores: Mapping[str, float]) -> None:
        self.scores = dict(scores)
        self.calls: list[tuple[str, tuple[Passage, ...]]] = []

    def score(self, query: str, passages: Sequence[Passage]) -> Mapping[bytes, float]:
        requested = tuple(passages)
        self.calls.append((query, requested))
        return {
            passage.content_hash: self.scores[passage.text] for passage in requested
        }


class InMemoryArtifactStore:
    class ArtifactValidationError(Exception):
        pass

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def read_validated(
        self,
        key: str,
        decoder: Callable[[bytes], object],
    ) -> object | None:
        payload = self.objects.get(key)
        return None if payload is None else decoder(payload)

    def write(self, key: str, payload: bytes) -> None:
        self.objects[key] = payload

    def evict_local(self, key: str) -> None:
        self.objects.pop(key, None)


def _cached_scorer(
    store: InMemoryArtifactStore,
    scorer: RecordingScorer,
) -> CachedPassageScorer:
    cache = PassageScoreCache(
        store,
        scorer,
        chunker_fingerprint="chunker-v1",
        representation="canonical",
        artifact_validation_error=InMemoryArtifactStore.ArtifactValidationError,
    )
    return CachedPassageScorer(cache)


def _document(
    document_id: str,
    *,
    rank: int,
    score: float,
    text: str | None,
    language: str | None = "eng",
    english_text: str | None = None,
) -> RetrievedDocument:
    return RetrievedDocument(
        rank=rank,
        document_id=document_id,
        score=score,
        text=text,
        url=f"https://example.test/{document_id}",
        date="2026-01-01",
        language=language,
        content_source="fixture",
        content_local_cache_hit=True,
        english_text=english_text,
    )


def _retrieval(
    documents: Sequence[RetrievedDocument],
    *,
    query: str = "exact query",
    hits: Sequence[SearchHit] | None = None,
) -> RetrievalResult:
    selected_hits = (
        tuple(hits)
        if hits is not None
        else tuple(
            SearchHit(document.rank, document.document_id, document.score)
            for document in documents
        )
    )
    return RetrievalResult(
        search_result=PipelineResult(
            query=query,
            pipeline="ragtime2",
            hits=selected_hits,
            processed=True,
            local_cache_hit=True,
            remote_cached=True,
            server_timestamp=None,
            cache_stored_at=None,
        ),
        documents=tuple(documents),
    )


class RerankingTests(unittest.TestCase):
    def test_rerank_returns_compact_deterministic_evidence(self) -> None:
        retrieval = _retrieval(
            (
                _document("doc-b", rank=2, score=0.4, text="same. same."),
                _document("doc-a", rank=1, score=0.5, text="winner."),
            )
        )
        chunker = RecordingChunker(
            {
                "doc-b": (
                    ("same.", 0, 5),
                    ("same.", 6, 11),
                    ("same.", 6, 11),
                ),
                "doc-a": (("winner.", 0, 7),),
            }
        )
        scorer = RecordingScorer({"same.": 1.0, "winner.": 2.0})

        result = rerank_retrieval(
            retrieval,
            chunker=chunker,
            scorer=scorer,
            representation="canonical",
        )

        self.assertEqual(
            [row.document.document_id for row in result.documents],
            ["doc-a", "doc-b"],
        )
        self.assertEqual(result.documents[0].score, 2.0)
        self.assertEqual(
            [(item.start_char, item.end_char) for item in result.documents[1].evidence],
            [(0, 5), (6, 11)],
        )
        self.assertEqual(len(scorer.calls), 1)
        self.assertEqual(len(scorer.calls[0][1]), 2)

    def test_result_records_are_frozen_slotted_and_have_only_compact_fields(
        self,
    ) -> None:
        expected_evidence_fields = (
            "document_id",
            "start_char",
            "end_char",
            "score",
            "content_hash",
        )
        expected_document_fields = (
            "document",
            "score",
            "evidence",
            "reranked_rank",
            "status",
        )
        expected_result_fields = ("query", "representation", "documents")

        for record, expected_fields in (
            (PassageEvidence, expected_evidence_fields),
            (RerankedDocument, expected_document_fields),
            (RerankingResult, expected_result_fields),
        ):
            with self.subTest(record=record):
                self.assertTrue(is_dataclass(record))
                self.assertIsInstance(record.__slots__, tuple)
                self.assertEqual(
                    tuple(field.name for field in fields(record)), expected_fields
                )
                self.assertIsNot(record.__setattr__, object.__setattr__)

        result = rerank_retrieval(
            _retrieval((_document("doc", rank=1, score=0.1, text="text"),)),
            chunker=RecordingChunker(),
            scorer=RecordingScorer({"text": 1.0}),
            representation="canonical",
        )
        with self.assertRaises(FrozenInstanceError):
            result.documents[0].score = 4.0  # type: ignore[misc]

    def test_canonical_selection_uses_exact_source_text_and_language(self) -> None:
        document = _document(
            "doc-spa",
            rank=1,
            score=0.5,
            text="  fuente exacta  ",
            language="spa",
            english_text="translated report",
        )
        chunker = RecordingChunker()

        result = rerank_retrieval(
            _retrieval((document,)),
            chunker=chunker,
            scorer=RecordingScorer({"  fuente exacta  ": 4.0}),
            representation="canonical",
        )

        self.assertEqual(
            chunker.calls,
            [
                {
                    "document_id": "doc-spa",
                    "canonical_text": "  fuente exacta  ",
                    "selected_text": "  fuente exacta  ",
                    "representation": "canonical",
                    "language": "spa",
                }
            ],
        )
        self.assertEqual(result.documents[0].status, "scored")
        self.assertEqual(result.documents[0].evidence[0].start_char, 0)
        self.assertEqual(result.documents[0].evidence[0].end_char, 17)

    def test_report_text_selection_uses_verified_english_representation(self) -> None:
        document = _document(
            "doc-spa",
            rank=1,
            score=0.5,
            text="  fuente exacta  ",
            language="spa",
            english_text="  exact report text  ",
        )
        chunker = RecordingChunker()

        result = rerank_retrieval(
            _retrieval((document,), query="report query"),
            chunker=chunker,
            scorer=RecordingScorer({"  exact report text  ": 4.0}),
            representation="report_text",
        )

        self.assertEqual(result.query, "report query")
        self.assertEqual(result.representation, "report_text")
        self.assertEqual(chunker.calls[0]["selected_text"], "  exact report text  ")
        self.assertEqual(chunker.calls[0]["language"], "eng")
        evidence = result.documents[0].evidence[0]
        self.assertEqual(evidence.end_char - evidence.start_char, 21)
        self.assertEqual(evidence.content_hash, sha256_bytes("  exact report text  "))

    def test_status_rows_are_retained_and_chunking_skips_invalid_representations(
        self,
    ) -> None:
        documents = (
            _document("unresolved", rank=1, score=0.9, text=None),
            _document("missing", rank=2, score=0.8, text="fuente", language="spa"),
            _document("scored", rank=3, score=0.1, text="scored"),
            _document("empty", rank=4, score=0.7, text="empty", language="spa"),
        )
        chunker = EmptyChunker()
        chunker.passages = {"scored": (("scored", 0, 6),)}

        result = rerank_retrieval(
            _retrieval(documents),
            chunker=chunker,
            scorer=RecordingScorer({"scored": 1.0}),
            representation="report_text",
        )

        self.assertEqual(
            [(row.document.document_id, row.status) for row in result.documents],
            [
                ("scored", "scored"),
                ("unresolved", "unresolved"),
                ("missing", "missing_representation"),
                ("empty", "missing_representation"),
            ],
        )
        self.assertEqual(
            [call["document_id"] for call in chunker.calls],
            ["scored"],
        )
        self.assertEqual(
            [row.reranked_rank for row in result.documents],
            [1, 2, 3, 4],
        )

    def test_no_passages_is_retained_as_an_unscored_status(self) -> None:
        result = rerank_retrieval(
            _retrieval((_document("empty", rank=1, score=0.1, text="empty"),)),
            chunker=EmptyChunker(),
            scorer=RecordingScorer({}),
            representation="canonical",
        )

        self.assertEqual(result.documents[0].status, "no_passages")
        self.assertIsNone(result.documents[0].score)
        self.assertEqual(result.documents[0].evidence, ())

    def test_maximum_aggregation_preserves_distinct_offsets_and_hash_evidence(
        self,
    ) -> None:
        document = _document(
            "doc-a",
            rank=1,
            score=0.1,
            text="same. same. other",
        )
        result = rerank_retrieval(
            _retrieval((document,)),
            chunker=RecordingChunker(
                {
                    "doc-a": (
                        ("other", 12, 17),
                        ("same.", 6, 11),
                        ("same.", 0, 5),
                    )
                }
            ),
            scorer=RecordingScorer({"same.": 5.0, "other": 2.0}),
            representation="canonical",
        )

        row = result.documents[0]
        self.assertEqual(row.status, "scored")
        self.assertEqual(row.score, 5.0)
        self.assertEqual(
            [(item.start_char, item.end_char) for item in row.evidence],
            [(0, 5), (6, 11), (12, 17)],
        )
        self.assertEqual(
            [item.content_hash for item in row.evidence],
            [sha256_bytes("same."), sha256_bytes("same."), sha256_bytes("other")],
        )
        for item in row.evidence:
            self.assertEqual(
                item.document_id,
                row.document.document_id,
            )
            self.assertEqual(
                document.text[item.start_char : item.end_char],
                ("same." if item.end_char <= 11 else "other"),
            )
            self.assertEqual(
                item.content_hash,
                sha256_bytes(document.text[item.start_char : item.end_char]),
            )

    def test_exact_location_overlap_is_deduplicated_without_hash_deduplication(
        self,
    ) -> None:
        document = _document(
            "doc-a",
            rank=1,
            score=0.1,
            text="same. same.",
        )
        chunker = RecordingChunker(
            {
                "doc-a": (
                    ("same.", 0, 5),
                    ("same.", 6, 11),
                    ("same.", 6, 11),
                )
            }
        )
        scorer = RecordingScorer({"same.": 3.0})

        result = rerank_retrieval(
            _retrieval((document,)),
            chunker=chunker,
            scorer=scorer,
            representation="canonical",
        )

        self.assertEqual(
            [(item.start_char, item.end_char) for item in result.documents[0].evidence],
            [(0, 5), (6, 11)],
        )
        self.assertEqual([passage.text for passage in scorer.calls[0][1]], ["same."])

    def test_scorer_receives_first_seen_content_hashes_once_across_documents(
        self,
    ) -> None:
        retrieval = _retrieval(
            (
                _document("doc-b", rank=2, score=0.4, text="shared"),
                _document("doc-a", rank=1, score=0.5, text="shared other"),
            )
        )
        scorer = RecordingScorer({"shared": 1.0, "other": 2.0})

        rerank_retrieval(
            retrieval,
            chunker=RecordingChunker({"doc-a": (("shared", 0, 6), ("other", 7, 12))}),
            scorer=scorer,
            representation="canonical",
        )

        self.assertEqual(len(scorer.calls), 1)
        self.assertEqual(
            [passage.text for passage in scorer.calls[0][1]], ["shared", "other"]
        )

    def test_document_ties_use_organizer_rank_then_document_id(self) -> None:
        retrieval = _retrieval(
            (
                _document("doc-z", rank=2, score=0.5, text="same"),
                _document("doc-a", rank=1, score=0.5, text="same"),
            )
        )
        result = rerank_retrieval(
            retrieval,
            chunker=RecordingChunker(),
            scorer=RecordingScorer({"same": 3.0}),
            representation="canonical",
        )
        self.assertEqual(
            [row.document.document_id for row in result.documents],
            ["doc-a", "doc-z"],
        )

    def test_warm_cache_delegation_reuses_complete_passage_score_map(self) -> None:
        store = InMemoryArtifactStore()
        model = RecordingScorer({"cached passage": 3.0})
        scorer = _cached_scorer(store, model)
        retrieval = _retrieval(
            (_document("doc", rank=1, score=0.5, text="cached passage"),),
            query="cache query",
        )

        first = rerank_retrieval(
            retrieval,
            chunker=RecordingChunker(),
            scorer=scorer,
            representation="canonical",
        )
        second = rerank_retrieval(
            retrieval,
            chunker=RecordingChunker(),
            scorer=scorer,
            representation="canonical",
        )

        self.assertEqual(first, second)
        self.assertEqual(len(model.calls), 1)

    def test_retrieval_selection_and_score_normalization_fail_with_one_error_type(
        self,
    ) -> None:
        document = _document("doc", rank=1, score=0.5, text="text")
        invalid_retrieval = _retrieval(
            (document,),
            hits=(SearchHit(1, "other", 0.5),),
        )
        with self.assertRaises(RerankingError):
            rerank_retrieval(
                invalid_retrieval,
                chunker=RecordingChunker(),
                scorer=RecordingScorer({"text": 1.0}),
                representation="canonical",
            )

        with self.assertRaises(RerankingError):
            rerank_retrieval(
                _retrieval((document,)),
                chunker=RecordingChunker(),
                scorer=RecordingScorer({"text": 1.0}),
                representation="unknown",  # type: ignore[arg-type]
            )

        with self.assertRaises(RerankingError):
            rerank_retrieval(
                _retrieval((document,)),
                chunker=RecordingChunker(),
                scorer=RecordingScorer({}),
                representation="canonical",
            )


if __name__ == "__main__":
    unittest.main()
