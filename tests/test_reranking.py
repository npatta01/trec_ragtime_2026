from __future__ import annotations

import hashlib
import errno
import json
import math
import tempfile
import unittest
from dataclasses import is_dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
from unittest.mock import Mock, patch

from trec_ragtime.models import (
    PipelineResult,
    RetrievalResult,
    RetrievedDocument,
    SearchHit,
)
from trec_ragtime.passages import Passage, sha256_bytes
import trec_ragtime.reranking as reranking_module
from trec_ragtime.reranking import (
    PassageEvidence,
    RerankedDocument,
    RerankingResult,
    max_passage_v1,
    rerank_retrieval,
    write_reranking_artifact,
)


def _passage_id(
    *,
    document_id: str,
    representation: str,
    representation_hash: bytes,
    start_char: int,
    end_char: int,
    chunk_index: int,
    chunker_fingerprint: str,
) -> bytes:
    import msgspec

    lineage = (
        document_id,
        representation,
        representation_hash,
        start_char,
        end_char,
        chunk_index,
        chunker_fingerprint,
    )
    return sha256_bytes(msgspec.msgpack.encode(lineage))


def _passages(
    document_id: str,
    canonical_text: str,
    selected_text: str,
    representation: str,
    language: str,
    chunks: Sequence[str],
    *,
    chunker_fingerprint: str = "chunker-v1",
) -> tuple[Passage, ...]:
    source_hash = sha256_bytes(canonical_text)
    representation_hash = sha256_bytes(selected_text)
    values: list[Passage] = []
    cursor = 0
    for chunk_index, text in enumerate(chunks):
        start_char = selected_text.index(text, cursor)
        end_char = start_char + len(text)
        values.append(
            Passage(
                document_id=document_id,
                source_content_hash=source_hash,
                representation=representation,  # type: ignore[arg-type]
                representation_hash=representation_hash,
                language=language,
                chunk_index=chunk_index,
                passage_id=_passage_id(
                    document_id=document_id,
                    representation=representation,
                    representation_hash=representation_hash,
                    start_char=start_char,
                    end_char=end_char,
                    chunk_index=chunk_index,
                    chunker_fingerprint=chunker_fingerprint,
                ),
                text=text,
                start_char=start_char,
                end_char=end_char,
                boundary="paragraph_packed",
                previous_passage_id=None,
                next_passage_id=None,
                content_hash=sha256_bytes(text),
                chunker_fingerprint=chunker_fingerprint,
            )
        )
        cursor = end_char
    return tuple(
        passage.__class__(
            **{
                field: (
                    None
                    if field == "previous_passage_id" and index == 0
                    else values[index - 1].passage_id
                    if field == "previous_passage_id"
                    else None
                    if field == "next_passage_id" and index == len(values) - 1
                    else values[index + 1].passage_id
                    if field == "next_passage_id"
                    else getattr(passage, field)
                )
                for field in passage.__dataclass_fields__
            }
        )
        for index, passage in enumerate(values)
    )


class RecordingChunker:
    fingerprint = "chunker-v1"

    def __init__(self, chunks: Mapping[str, Sequence[str]] | None = None) -> None:
        self.chunks = dict(chunks or {})
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
        chunks = self.chunks.get(document_id, (selected_text,))
        return _passages(
            document_id,
            canonical_text,
            selected_text,
            representation,
            language,
            chunks,
            chunker_fingerprint=self.fingerprint,
        )


class EmptyChunker(RecordingChunker):
    def chunk(self, **kwargs: str) -> tuple[Passage, ...]:
        self.calls.append(dict(kwargs))
        document_id = kwargs["document_id"]
        if document_id in self.chunks:
            return _passages(
                document_id,
                kwargs["canonical_text"],
                kwargs["selected_text"],
                kwargs["representation"],
                kwargs["language"],
                self.chunks[document_id],
                chunker_fingerprint=self.fingerprint,
            )
        return ()


class RecordingScorer:
    fingerprint = "scorer-v1"

    def __init__(self, scores: Mapping[str, float]) -> None:
        self.scores = dict(scores)
        self.calls: list[tuple[str, tuple[bytes, ...]]] = []

    def score(self, query: str, passages: Sequence[Passage]) -> Mapping[bytes, float]:
        self.calls.append((query, tuple(passage.content_hash for passage in passages)))
        return {passage.content_hash: self.scores[passage.text] for passage in passages}


class CachedRecordingScorer(RecordingScorer):
    def __init__(self, scores: Mapping[str, float]) -> None:
        super().__init__(scores)
        self.cache = SimpleNamespace(
            effective_fingerprint="cache-v1",
            fingerprint="cache-v1",
            object_key=lambda query: f"passage-scores/{hashlib.sha256(query.encode()).hexdigest()}",
        )


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
    pipeline: str = "ragtime2",
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
            pipeline=pipeline,
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
    def test_result_types_are_frozen_and_slotted(self) -> None:
        self.assertTrue(is_dataclass(RerankedDocument))
        self.assertTrue(is_dataclass(RerankingResult))
        self.assertIsInstance(RerankedDocument.__slots__, tuple)
        self.assertIsInstance(RerankingResult.__slots__, tuple)
        self.assertIsNot(RerankedDocument.__setattr__, object.__setattr__)
        self.assertIsNot(RerankingResult.__setattr__, object.__setattr__)

    def test_scored_result_retains_complete_frozen_passage_evidence(self) -> None:
        document = _document("doc-1", rank=1, score=0.5, text="first second")
        result = rerank_retrieval(
            _retrieval((document,)),
            topic_id="topic-1",
            chunker=RecordingChunker({"doc-1": ("first", " second")}),
            scorer=RecordingScorer({"first": 1.0, " second": 2.0}),
            representation="canonical",
        )
        ranked = result.documents[0]

        self.assertEqual(ranked.reranked_rank, 1)
        self.assertEqual(len(ranked.passage_evidence), 2)
        self.assertEqual(
            [evidence.text for evidence in ranked.passage_evidence],
            ["first", " second"],
        )
        for evidence in ranked.passage_evidence:
            self.assertIsInstance(evidence, PassageEvidence)
            self.assertTrue(is_dataclass(evidence))
            self.assertIsInstance(evidence.__slots__, tuple)
            self.assertEqual(evidence.document_id, "doc-1")
            self.assertEqual(evidence.source_content_hash, sha256_bytes("first second"))
            self.assertEqual(evidence.representation_hash, sha256_bytes("first second"))
            self.assertEqual(evidence.content_hash, sha256_bytes(evidence.text))
            self.assertEqual(
                evidence.end_char - evidence.start_char, len(evidence.text)
            )
            self.assertEqual(
                evidence.score, ranked.passage_scores[evidence.chunk_index]
            )
            self.assertIsNot(evidence.__setattr__, object.__setattr__)
        self.assertEqual(
            ranked.winning_passage,
            ranked.passage_evidence[1],
        )
        with self.assertRaises(ValueError):
            replace(ranked, passage_scores=(99.0, 2.0))

    def test_result_enforces_reranked_rank_sequence_and_global_identity(self) -> None:
        result = rerank_retrieval(
            _retrieval(
                (
                    _document("doc-a", rank=1, score=0.5, text="a"),
                    _document("doc-b", rank=2, score=0.4, text="b"),
                )
            ),
            topic_id="topic-1",
            chunker=RecordingChunker(),
            scorer=RecordingScorer({"a": 2.0, "b": 1.0}),
            representation="canonical",
        )
        self.assertEqual(
            [document.reranked_rank for document in result.documents], [1, 2]
        )
        with self.assertRaises(ValueError):
            replace(
                result,
                documents=(
                    replace(result.documents[0], reranked_rank=2),
                    result.documents[1],
                ),
            )
        with self.assertRaises(ValueError):
            replace(
                result,
                documents=(
                    replace(result.documents[0], topic_id="other"),
                    result.documents[1],
                ),
            )
        counts = result.status_counts
        counts["scored"] = 0
        self.assertEqual(result.status_counts["scored"], 2)

    def test_document_validation_rejects_invalid_status_score_winner_and_report_language(
        self,
    ) -> None:
        result = rerank_retrieval(
            _retrieval((_document("doc-1", rank=1, score=0.5, text="text"),)),
            topic_id="topic-1",
            chunker=RecordingChunker(),
            scorer=RecordingScorer({"text": 1.0}),
            representation="canonical",
        )
        document = result.documents[0]
        invalid_documents = (
            lambda: replace(document, reranked_rank=0),
            lambda: replace(document, status="unresolved"),
            lambda: replace(document, score=math.nan),
            lambda: replace(document, winning_content_hash=b"x" * 32),
            lambda: replace(document, language="spa"),
        )
        for build_invalid in invalid_documents:
            with self.subTest(build_invalid=build_invalid):
                with self.assertRaises(ValueError):
                    build_invalid()

    def test_result_rejects_swapped_documents_even_when_ranks_are_renumbered(
        self,
    ) -> None:
        result = rerank_retrieval(
            _retrieval(
                (
                    _document("doc-a", rank=1, score=0.5, text="a"),
                    _document("doc-b", rank=2, score=0.4, text="b"),
                )
            ),
            topic_id="topic-1",
            chunker=RecordingChunker(),
            scorer=RecordingScorer({"a": 1.0, "b": 2.0}),
            representation="canonical",
        )
        swapped = (
            replace(result.documents[1], reranked_rank=1),
            replace(result.documents[0], reranked_rank=2),
        )
        with self.assertRaises(ValueError):
            replace(result, documents=swapped)

    def test_result_rejects_duplicate_document_ids_and_organizer_ranks(self) -> None:
        result = rerank_retrieval(
            _retrieval(
                (
                    _document("doc-a", rank=1, score=0.5, text="a"),
                    _document("doc-b", rank=2, score=0.4, text="b"),
                )
            ),
            topic_id="topic-1",
            chunker=RecordingChunker(),
            scorer=RecordingScorer({"a": 2.0, "b": 1.0}),
            representation="canonical",
        )
        duplicate_id_documents = (
            result.documents[0],
            replace(result.documents[0], reranked_rank=2),
        )
        with self.assertRaises(ValueError):
            replace(result, documents=duplicate_id_documents)

        duplicate_rank_documents = (
            result.documents[0],
            replace(
                result.documents[1],
                reranked_rank=2,
                organizer_rank=result.documents[0].organizer_rank,
            ),
        )
        with self.assertRaises(ValueError):
            replace(result, documents=duplicate_rank_documents)

    def test_result_rejects_canonical_selected_text_mismatch(self) -> None:
        result = rerank_retrieval(
            _retrieval((_document("doc-1", rank=1, score=0.5, text="source"),)),
            topic_id="topic-1",
            chunker=EmptyChunker(),
            scorer=RecordingScorer({}),
            representation="canonical",
        )
        self.assertEqual(result.documents[0].status, "no_passages")
        forged_document = object.__new__(RerankedDocument)
        for field in result.documents[0].__dataclass_fields__:
            object.__setattr__(
                forged_document, field, getattr(result.documents[0], field)
            )
        object.__setattr__(forged_document, "selected_text", "forged")
        with self.assertRaises(ValueError):
            replace(result, documents=(forged_document,))

    def test_result_rejects_false_missing_report_text_representation(self) -> None:
        result = rerank_retrieval(
            _retrieval(
                (_document("doc-1", rank=1, score=0.5, text="source", language="spa"),)
            ),
            topic_id="topic-1",
            chunker=EmptyChunker(),
            scorer=RecordingScorer({}),
            representation="report_text",
        )
        self.assertEqual(result.documents[0].status, "missing_representation")
        forged_document = object.__new__(RerankedDocument)
        for field in result.documents[0].__dataclass_fields__:
            object.__setattr__(
                forged_document, field, getattr(result.documents[0], field)
            )
        object.__setattr__(forged_document, "report_text", "forged report")
        with self.assertRaises(ValueError):
            replace(result, documents=(forged_document,))

    def test_result_rejects_contradictory_unresolved_and_missing_canonical_states(
        self,
    ) -> None:
        unresolved = rerank_retrieval(
            _retrieval((_document("unresolved", rank=1, score=0.5, text=None),)),
            topic_id="topic-1",
            chunker=RecordingChunker(),
            scorer=RecordingScorer({}),
            representation="canonical",
        )
        with self.assertRaises(ValueError):
            replace(
                unresolved,
                documents=(replace(unresolved.documents[0], language="eng"),),
            )

        missing_canonical = rerank_retrieval(
            _retrieval(
                (_document("missing", rank=1, score=0.5, text="source", language=None),)
            ),
            topic_id="topic-1",
            chunker=RecordingChunker(),
            scorer=RecordingScorer({}),
            representation="canonical",
        )
        with self.assertRaises(ValueError):
            replace(
                missing_canonical,
                documents=(replace(missing_canonical.documents[0], language="eng"),),
            )

    def test_report_text_selection_preserves_exact_source_and_english_language(
        self,
    ) -> None:
        document = _document(
            "doc-1",
            rank=1,
            score=0.5,
            text="  fuente exacta  ",
            language="spa",
            english_text="  exact report text  ",
        )
        chunker = RecordingChunker()
        scorer = RecordingScorer({"  exact report text  ": 4.0})

        result = rerank_retrieval(
            _retrieval((document,), query="report query"),
            topic_id="topic-1",
            chunker=chunker,
            scorer=scorer,
            representation="report_text",
        )

        self.assertEqual(result.topic_id, "topic-1")
        self.assertEqual(result.query, "report query")
        self.assertEqual(result.pipeline, "ragtime2")
        self.assertEqual(
            chunker.calls,
            [
                {
                    "document_id": "doc-1",
                    "canonical_text": "  fuente exacta  ",
                    "selected_text": "  exact report text  ",
                    "representation": "report_text",
                    "language": "eng",
                }
            ],
        )
        self.assertEqual(result.documents[0].status, "scored")
        self.assertEqual(result.documents[0].representation, "report_text")
        self.assertEqual(result.documents[0].selected_language, "eng")
        self.assertEqual(result.documents[0].canonical_text, "  fuente exacta  ")
        self.assertEqual(result.documents[0].selected_text, "  exact report text  ")

    def test_canonical_selection_uses_exact_text_and_source_language(self) -> None:
        document = _document(
            "doc-1",
            rank=1,
            score=0.5,
            text="  texto fuente  ",
            language="spa",
            english_text="translated report",
        )
        chunker = RecordingChunker()
        scorer = RecordingScorer({"  texto fuente  ": 1.0})

        result = rerank_retrieval(
            _retrieval((document,)),
            topic_id="topic-1",
            chunker=chunker,
            scorer=scorer,
            representation="canonical",
        )

        self.assertEqual(chunker.calls[0]["selected_text"], "  texto fuente  ")
        self.assertEqual(chunker.calls[0]["canonical_text"], "  texto fuente  ")
        self.assertEqual(chunker.calls[0]["language"], "spa")
        self.assertEqual(result.documents[0].selected_language, "spa")

    def test_max_aggregation_tie_breaks_by_hash_and_scorer_deduplicates_content(
        self,
    ) -> None:
        first = _document("doc-a", rank=2, score=0.1, text="shared first second")
        second = _document("doc-b", rank=1, score=0.2, text="shared")
        third = _document("doc-c", rank=3, score=0.3, text="other")
        chunker = RecordingChunker(
            {
                "doc-a": ("shared", " first", " second"),
                "doc-b": ("shared",),
                "doc-c": ("other",),
            }
        )
        scorer = CachedRecordingScorer(
            {"shared": 5.0, " first": 2.0, " second": 5.0, "other": 5.0}
        )

        result = rerank_retrieval(
            _retrieval((third, first, second)),
            topic_id="topic-1",
            chunker=chunker,
            scorer=scorer,
            representation="canonical",
        )

        self.assertEqual(len(scorer.calls), 1)
        self.assertEqual(
            scorer.calls[0][1],
            tuple(
                sorted(
                    {
                        sha256_bytes(text)
                        for text in ("shared", " first", " second", "other")
                    }
                )
            ),
        )
        self.assertEqual(
            [document.document_id for document in result.documents],
            ["doc-b", "doc-a", "doc-c"],
        )
        doc_a = next(
            document for document in result.documents if document.document_id == "doc-a"
        )
        self.assertEqual(doc_a.status, "scored")
        self.assertEqual(doc_a.score, 5.0)
        self.assertEqual(
            doc_a.winning_content_hash,
            min(sha256_bytes("shared"), sha256_bytes(" second")),
        )
        self.assertEqual(doc_a.aggregation, max_passage_v1)
        self.assertEqual(doc_a.cache_fingerprint, "cache-v1")
        self.assertTrue(doc_a.cache_key)

    def test_shuffled_hits_and_documents_produce_identical_result(self) -> None:
        documents = (
            _document("doc-a", rank=2, score=0.9, text="a"),
            _document("doc-b", rank=1, score=0.9, text="b"),
        )
        chunker = RecordingChunker()
        first = rerank_retrieval(
            _retrieval(documents),
            topic_id="topic-1",
            chunker=chunker,
            scorer=RecordingScorer({"a": 2.0, "b": 2.0}),
            representation="canonical",
        )
        shuffled_retrieval = _retrieval(
            tuple(reversed(documents)),
            hits=tuple(
                reversed(
                    (
                        SearchHit(2, "doc-a", 0.9),
                        SearchHit(1, "doc-b", 0.9),
                    )
                )
            ),
        )
        second = rerank_retrieval(
            shuffled_retrieval,
            topic_id="topic-1",
            chunker=RecordingChunker(),
            scorer=RecordingScorer({"a": 2.0, "b": 2.0}),
            representation="canonical",
        )
        self.assertEqual(first, second)

    def test_unresolved_missing_representation_and_no_passages_are_retained_after_scored(
        self,
    ) -> None:
        documents = (
            _document("scored", rank=4, score=0.1, text="scored"),
            _document("unresolved", rank=1, score=0.9, text=None),
            _document("missing", rank=2, score=0.8, text="source", language="spa"),
            _document(
                "empty",
                rank=3,
                score=0.7,
                text="empty",
                language="spa",
                english_text=None,
            ),
        )
        chunker = EmptyChunker()
        chunker.chunks = {"scored": ("scored",)}
        scorer = RecordingScorer({"scored": 1.0})

        result = rerank_retrieval(
            _retrieval(documents),
            topic_id="topic-1",
            chunker=chunker,
            scorer=scorer,
            representation="report_text",
        )

        self.assertEqual(
            [(document.document_id, document.status) for document in result.documents],
            [
                ("scored", "scored"),
                ("unresolved", "unresolved"),
                ("missing", "missing_representation"),
                ("empty", "missing_representation"),
            ],
        )
        self.assertEqual([call["document_id"] for call in chunker.calls], ["scored"])

        no_passages_result = rerank_retrieval(
            _retrieval((_document("empty", rank=1, score=0.1, text="empty"),)),
            topic_id="topic-1",
            chunker=EmptyChunker(),
            scorer=RecordingScorer({}),
            representation="canonical",
        )
        self.assertEqual(no_passages_result.documents[0].status, "no_passages")

    def test_rejects_duplicate_or_misaligned_search_rows(self) -> None:
        document = _document("doc-1", rank=1, score=0.5, text="text")
        cases = (
            _retrieval(
                (document,),
                hits=(SearchHit(1, "doc-1", 0.5), SearchHit(2, "doc-1", 0.4)),
            ),
            _retrieval(
                (_document("doc-1", rank=2, score=0.5, text="text"),),
                hits=(SearchHit(1, "doc-1", 0.5),),
            ),
            _retrieval(
                (document,),
                hits=(SearchHit(1, "other", 0.5),),
            ),
        )
        for retrieval in cases:
            with self.subTest(retrieval=retrieval):
                with self.assertRaises(ValueError):
                    rerank_retrieval(
                        retrieval,
                        topic_id="topic-1",
                        chunker=RecordingChunker(),
                        scorer=RecordingScorer({"text": 1.0}),
                        representation="canonical",
                    )

    def test_rejects_invalid_topic_representation_and_scores(self) -> None:
        retrieval = _retrieval((_document("doc-1", rank=1, score=0.5, text="text"),))
        for kwargs in (
            {"topic_id": "", "representation": "canonical"},
            {"topic_id": "topic-1", "representation": "unknown"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                rerank_retrieval(
                    retrieval,
                    topic_id=kwargs["topic_id"],
                    chunker=RecordingChunker(),
                    scorer=RecordingScorer({"text": 1.0}),
                    representation=kwargs["representation"],  # type: ignore[arg-type]
                )

        class BadScorer(RecordingScorer):
            def score(
                self, query: str, passages: Sequence[Passage]
            ) -> Mapping[bytes, Any]:
                return {passages[0].content_hash: math.nan}

        with self.assertRaises(ValueError):
            rerank_retrieval(
                retrieval,
                topic_id="topic-1",
                chunker=RecordingChunker(),
                scorer=BadScorer({"text": 1.0}),
                representation="canonical",
            )


class RerankingArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.destination = Path(self.temporary_directory.name) / "run"
        result = rerank_retrieval(
            _retrieval(
                (
                    _document("doc-1", rank=1, score=0.5, text="text"),
                    _document("doc-2", rank=2, score=0.4, text=None),
                ),
                query="artifact query",
                pipeline="ragtime2%100",
            ),
            topic_id="topic-7",
            chunker=RecordingChunker(),
            scorer=CachedRecordingScorer({"text": 3.0}),
            representation="canonical",
        )
        self.result = result

    def test_writer_emits_stable_compact_manifest_and_jsonl_bytes(self) -> None:
        write_reranking_artifact(
            self.destination,
            self.result,
            source_retrieval_ref="retrieval-artifact-v1",
        )
        manifest_path = self.destination / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records_path = self.destination / manifest["records"]["filename"]
        records_bytes = records_path.read_bytes()
        self.assertEqual(manifest["schema_version"], "reranking_v1")
        self.assertEqual(manifest["topic_id"], "topic-7")
        self.assertEqual(manifest["query"], "artifact query")
        self.assertEqual(manifest["pipeline"], "ragtime2%100")
        self.assertEqual(manifest["source_retrieval_ref"], "retrieval-artifact-v1")
        self.assertEqual(
            manifest["records"]["sha256"], hashlib.sha256(records_bytes).hexdigest()
        )
        self.assertEqual(
            manifest["counts"],
            {
                "missing_representation": 0,
                "no_passages": 0,
                "scored": 1,
                "unresolved": 1,
            },
        )
        self.assertNotIn("created_at", manifest)
        self.assertNotIn(
            str(self.destination), manifest_path.read_text(encoding="utf-8")
        )
        self.assertEqual(records_bytes.count(b"\n"), 2)
        self.assertEqual(
            records_bytes, (self.destination / "reranked.jsonl").read_bytes()
        )
        rows = [json.loads(line) for line in records_bytes.splitlines()]
        scored_row = rows[0]
        self.assertEqual(scored_row["reranked_rank"], 1)
        self.assertEqual(scored_row["representation"], "canonical")
        self.assertEqual(scored_row["query"], "artifact query")
        self.assertEqual(scored_row["pipeline"], "ragtime2%100")
        self.assertEqual(scored_row["aggregation"], max_passage_v1)
        self.assertEqual(scored_row["chunker_fingerprint"], "chunker-v1")
        self.assertEqual(scored_row["scorer_fingerprint"], "scorer-v1")
        self.assertEqual(scored_row["cache_fingerprint"], "cache-v1")
        self.assertEqual(len(scored_row["passage_evidence"]), 1)
        self.assertEqual(
            scored_row["winning_passage"], scored_row["passage_evidence"][0]
        )
        self.assertEqual(
            set(scored_row["passage_evidence"][0]),
            {
                "boundary",
                "chunk_index",
                "content_hash",
                "document_id",
                "end_char",
                "language",
                "next_passage_id",
                "overlap_chars",
                "overlap_tokens",
                "passage_id",
                "previous_passage_id",
                "representation",
                "representation_hash",
                "score",
                "source_content_hash",
                "start_char",
                "text",
                "chunker_fingerprint",
            },
        )

        first_bytes = (manifest_path.read_bytes(), records_bytes)
        second_destination = Path(self.temporary_directory.name) / "run-2"
        write_reranking_artifact(
            second_destination,
            self.result,
            source_retrieval_ref="retrieval-artifact-v1",
        )
        self.assertEqual(
            first_bytes[0], (second_destination / "manifest.json").read_bytes()
        )
        self.assertEqual(
            first_bytes[1], (second_destination / "reranked.jsonl").read_bytes()
        )

    def test_writer_requires_explicit_nonblank_source_reference(self) -> None:
        with self.assertRaises(TypeError):
            write_reranking_artifact(self.destination, self.result)  # type: ignore[call-arg]
        for reference in ("", "   ", "bad\nref"):
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                write_reranking_artifact(
                    self.destination,
                    self.result,
                    source_retrieval_ref=reference,
                )

    def test_writer_refuses_overwrite_without_changing_existing_destination(
        self,
    ) -> None:
        self.destination.mkdir()
        sentinel = self.destination / "sentinel"
        sentinel.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            write_reranking_artifact(
                self.destination,
                self.result,
                source_retrieval_ref="retrieval-artifact-v1",
            )
        self.assertEqual(sentinel.read_bytes(), b"keep")

    def test_writer_cleans_temporary_sibling_after_atomic_rename_failure(self) -> None:
        parent = self.destination.parent
        with patch(
            "trec_ragtime.reranking._publish_noreplace",
            side_effect=OSError("rename failed"),
        ):
            with self.assertRaises(OSError):
                write_reranking_artifact(
                    self.destination,
                    self.result,
                    source_retrieval_ref="retrieval-artifact-v1",
                )
        self.assertFalse(self.destination.exists())
        self.assertEqual(
            tuple(path for path in parent.iterdir() if path.name.startswith(".run.")),
            (),
        )

    def test_writer_revalidates_a_forged_result_before_emitting(self) -> None:
        forged = object.__new__(RerankingResult)
        for field in self.result.__dataclass_fields__:
            object.__setattr__(forged, field, getattr(self.result, field))
        object.__setattr__(
            forged, "documents", (replace(self.result.documents[0], reranked_rank=2),)
        )
        with self.assertRaises(ValueError):
            write_reranking_artifact(
                self.destination,
                forged,
                source_retrieval_ref="retrieval-artifact-v1",
            )
        self.assertFalse(self.destination.exists())

    def test_writer_revalidates_swapped_and_renumbered_documents(self) -> None:
        first, second = self.result.documents
        forged = object.__new__(RerankingResult)
        for field in self.result.__dataclass_fields__:
            object.__setattr__(forged, field, getattr(self.result, field))
        object.__setattr__(
            forged,
            "documents",
            (
                replace(second, reranked_rank=1),
                replace(first, reranked_rank=2),
            ),
        )
        with self.assertRaises(ValueError):
            write_reranking_artifact(
                self.destination,
                forged,
                source_retrieval_ref="retrieval-artifact-v1",
            )
        self.assertFalse(self.destination.exists())

    def test_writer_revalidates_forged_passage_evidence_before_emitting(self) -> None:
        original_evidence = self.result.documents[0].passage_evidence[0]
        forged_evidence = object.__new__(PassageEvidence)
        for field in original_evidence.__dataclass_fields__:
            object.__setattr__(
                forged_evidence, field, getattr(original_evidence, field)
            )
        object.__setattr__(forged_evidence, "boundary", "forged")
        forged_document = object.__new__(RerankedDocument)
        for field in self.result.documents[0].__dataclass_fields__:
            object.__setattr__(
                forged_document, field, getattr(self.result.documents[0], field)
            )
        object.__setattr__(forged_document, "passage_evidence", (forged_evidence,))
        forged_result = object.__new__(RerankingResult)
        for field in self.result.__dataclass_fields__:
            object.__setattr__(forged_result, field, getattr(self.result, field))
        object.__setattr__(forged_result, "documents", (forged_document,))

        with self.assertRaises(ValueError):
            write_reranking_artifact(
                self.destination,
                forged_result,
                source_retrieval_ref="retrieval-artifact-v1",
            )
        self.assertFalse(self.destination.exists())

    def test_writer_race_preserves_competing_destination_and_sentinel(self) -> None:
        sentinel = self.destination / "sentinel"
        original_publish = reranking_module._publish_noreplace

        def compete_before_publish(source: Path, destination: Path) -> None:
            destination.mkdir()
            sentinel.write_bytes(b"winner")
            original_publish(source, destination)

        with patch(
            "trec_ragtime.reranking._publish_noreplace",
            side_effect=compete_before_publish,
        ):
            with self.assertRaises(FileExistsError):
                write_reranking_artifact(
                    self.destination,
                    self.result,
                    source_retrieval_ref="retrieval-artifact-v1",
                )
        self.assertEqual(sentinel.read_bytes(), b"winner")
        self.assertEqual(
            tuple(
                path
                for path in self.destination.parent.iterdir()
                if path.name.startswith(".run.")
            ),
            (),
        )

    def test_noreplace_dispatches_to_platform_helpers(self) -> None:
        for platform_name, helper_name in (
            ("linux", "_publish_noreplace_linux"),
            ("darwin", "_publish_noreplace_darwin"),
            ("win32", "_publish_noreplace_windows"),
        ):
            with self.subTest(platform_name=platform_name):
                with patch.object(reranking_module.sys, "platform", platform_name):
                    with patch.object(
                        reranking_module,
                        helper_name,
                        autospec=True,
                    ) as helper:
                        reranking_module._publish_noreplace(
                            Path("source"), Path("destination")
                        )
                helper.assert_called_once_with(Path("source"), Path("destination"))

    def test_darwin_noreplace_uses_exclusive_rename_and_maps_errors(self) -> None:
        renamex_np = Mock(return_value=0)
        libc = SimpleNamespace(renamex_np=renamex_np)
        with patch.object(
            reranking_module.ctypes,
            "CDLL",
            return_value=libc,
        ):
            reranking_module._publish_noreplace_darwin(
                Path("source"), Path("destination")
            )
        renamex_np.assert_called_once_with(b"source", b"destination", 4)

        renamex_np = Mock(return_value=-1)
        libc = SimpleNamespace(renamex_np=renamex_np)
        with (
            patch.object(
                reranking_module.ctypes,
                "CDLL",
                return_value=libc,
            ),
            patch.object(
                reranking_module.ctypes, "get_errno", return_value=errno.EEXIST
            ),
        ):
            with self.assertRaises(FileExistsError):
                reranking_module._publish_noreplace_darwin(
                    Path("source"), Path("destination")
                )

    def test_linux_noreplace_uses_renameat2_flag_and_maps_errors(self) -> None:
        renameat2 = Mock(return_value=0)
        libc = SimpleNamespace(renameat2=renameat2)
        with patch.object(
            reranking_module.ctypes,
            "CDLL",
            return_value=libc,
        ):
            reranking_module._publish_noreplace_linux(
                Path("source"), Path("destination")
            )
        renameat2.assert_called_once_with(-100, b"source", -100, b"destination", 1)

        renameat2 = Mock(return_value=-1)
        libc = SimpleNamespace(renameat2=renameat2)
        with (
            patch.object(
                reranking_module.ctypes,
                "CDLL",
                return_value=libc,
            ),
            patch.object(
                reranking_module.ctypes, "get_errno", return_value=errno.EEXIST
            ),
        ):
            with self.assertRaises(FileExistsError):
                reranking_module._publish_noreplace_linux(
                    Path("source"), Path("destination")
                )

    def test_windows_noreplace_uses_no_replace_flag_and_maps_errors(self) -> None:
        move_file_ex = Mock(return_value=1)
        kernel32 = SimpleNamespace(MoveFileExW=move_file_ex)
        with patch.object(
            reranking_module.ctypes,
            "WinDLL",
            return_value=kernel32,
            create=True,
        ) as win_dll:
            reranking_module._publish_noreplace_windows(
                Path("source"), Path("destination")
            )
        win_dll.assert_called_once_with("kernel32", use_last_error=True)
        move_file_ex.assert_called_once_with("source", "destination", 0)

        for windows_error in (80, 183):
            with self.subTest(windows_error=windows_error):
                move_file_ex = Mock(return_value=0)
                kernel32 = SimpleNamespace(MoveFileExW=move_file_ex)
                with (
                    patch.object(
                        reranking_module.ctypes,
                        "WinDLL",
                        return_value=kernel32,
                        create=True,
                    ) as win_dll,
                    patch.object(
                        reranking_module.ctypes,
                        "get_last_error",
                        return_value=windows_error,
                        create=True,
                    ),
                ):
                    with self.assertRaises(FileExistsError):
                        reranking_module._publish_noreplace_windows(
                            Path("source"), Path("destination")
                        )
                win_dll.assert_called_once_with("kernel32", use_last_error=True)
                move_file_ex.assert_called_once_with("source", "destination", 0)

    def test_windows_directory_fsync_is_skipped_but_file_fsync_remains(self) -> None:
        with (
            patch.object(reranking_module.sys, "platform", "win32"),
            patch.object(reranking_module.os, "open") as directory_open,
            patch.object(reranking_module.os, "fsync") as fsync,
        ):
            reranking_module._fsync_directory(Path("directory"))
        directory_open.assert_not_called()
        fsync.assert_not_called()

        file_path = Path(self.temporary_directory.name) / "file"
        with patch.object(reranking_module.os, "fsync") as file_fsync:
            reranking_module._write_fsync(file_path, b"payload")
        file_fsync.assert_called_once()


if __name__ == "__main__":
    unittest.main()
