from __future__ import annotations

import dataclasses
import unittest
from collections.abc import Mapping, Sequence

from trec_ragtime.models import PipelineResult, RetrievalResult, RetrievedDocument, SearchHit
from trec_ragtime.passages import Passage, sha256_bytes
from trec_ragtime.topic_union_scoring import (
    InformationNeed,
    TopicRetrievalInput,
    TopicUnionScoringError,
    deduplicate_topic_documents,
    score_topic_union,
)


def document(document_id: str, rank: int, text: str, *, score: float = 0.5) -> RetrievedDocument:
    return RetrievedDocument(
        rank=rank,
        document_id=document_id,
        score=score,
        text=text,
        url=None,
        date=None,
        language="eng",
        content_source="local",
        content_local_cache_hit=True,
    )


def retrieval(query: str, *documents: RetrievedDocument) -> RetrievalResult:
    return RetrievalResult(
        search_result=PipelineResult(
            query=query,
            pipeline="ragtime2",
            hits=tuple(SearchHit(row.rank, row.document_id, row.score) for row in documents),
            processed=True,
            local_cache_hit=True,
            remote_cached=True,
            server_timestamp=None,
            cache_stored_at=None,
        ),
        documents=tuple(documents),
    )


def source(task_id: str, query: str, *documents: RetrievedDocument) -> TopicRetrievalInput:
    return TopicRetrievalInput(
        topic_id="2000",
        task_id=task_id,
        query=query,
        object_key=f"retrieval/v1/{task_id}",
        result=retrieval(query, *documents),
    )


class RecordingChunker:
    fingerprint = "chunker-v1"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def chunk(self, *, document_id: str, canonical_text: str, selected_text: str,
              representation: str, language: str) -> tuple[Passage, ...]:
        del canonical_text
        self.calls.append(document_id)
        return (
            Passage(
                document_id=document_id,
                representation=representation,  # type: ignore[arg-type]
                language=language,
                text=selected_text,
                start_char=0,
                end_char=len(selected_text),
                content_hash=sha256_bytes(selected_text),
                boundary="sentence",
            ),
        )


class RecordingScorer:
    fingerprint = "scorer-v1"

    def __init__(self, scores: Mapping[tuple[str, str], float]) -> None:
        self.scores = dict(scores)
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def score(self, query: str, passages: Sequence[Passage]) -> Mapping[bytes, float]:
        rows = tuple(passages)
        self.calls.append((query, tuple(row.text for row in rows)))
        return {row.content_hash: self.scores[(query, row.text)] for row in rows}


class TopicUnionScoringTests(unittest.TestCase):
    def test_union_deduplicates_document_ids_and_keeps_all_sources(self) -> None:
        shared = document("doc-shared", 2, "shared")
        rows = deduplicate_topic_documents(
            (
                source("T1", "q1", document("doc-a", 1, "alpha"), shared),
                source("T2", "q2", dataclasses.replace(shared, rank=1), document("doc-b", 2, "beta")),
            )
        )

        self.assertEqual([row.document.document_id for row in rows], ["doc-a", "doc-b", "doc-shared"])
        shared_row = next(row for row in rows if row.document.document_id == "doc-shared")
        self.assertEqual([item.task_id for item in shared_row.sources], ["T1", "T2"])

    def test_union_rejects_conflicting_duplicate_content(self) -> None:
        with self.assertRaisesRegex(TopicUnionScoringError, "conflicting duplicate"):
            deduplicate_topic_documents(
                (
                    source("T1", "q1", document("doc-a", 1, "alpha")),
                    source("T2", "q2", document("doc-a", 1, "different")),
                )
            )

    def test_scores_every_passage_against_every_need_and_keeps_maximum(self) -> None:
        inputs = (
            source("T1", "q1", document("doc-a", 1, "alpha"), document("doc-b", 2, "beta")),
            source("T2", "q2", document("doc-a", 99, "alpha")),
        )
        needs = (InformationNeed("N1", "need one"), InformationNeed("N2", "need two"))
        scorer = RecordingScorer(
            {
                ("need one", "alpha"): 0.3,
                ("need two", "alpha"): 0.9,
                ("need one", "beta"): 0.8,
                ("need two", "beta"): 0.1,
            }
        )
        chunker = RecordingChunker()

        result = score_topic_union(
            "2000", needs, inputs, chunker=chunker, scorer=scorer,
            representation="report_text",
        )

        self.assertEqual(chunker.calls, ["doc-a", "doc-b"])
        self.assertEqual(
            scorer.calls,
            [("need one", ("alpha", "beta")), ("need two", ("alpha", "beta"))],
        )
        self.assertEqual(
            [(row.document_id, row.score, row.winning_need_id) for row in result.documents],
            [("doc-a", 0.9, "N2"), ("doc-b", 0.8, "N1")],
        )
        self.assertEqual(result.unique_document_count, 2)
        self.assertEqual(result.passage_count, 2)

    def test_retrieval_rank_and_duplicate_frequency_do_not_change_ranking(self) -> None:
        needs = (InformationNeed("N1", "need"),)
        scorer = RecordingScorer({("need", "alpha"): 0.5, ("need", "beta"): 0.5})
        result = score_topic_union(
            "2000",
            needs,
            (
                source("T1", "q1", document("doc-b", 1, "beta"), document("doc-a", 2, "alpha")),
                source("T2", "q2", document("doc-b", 1, "beta")),
            ),
            chunker=RecordingChunker(),
            scorer=scorer,
            representation="report_text",
        )

        self.assertEqual([row.document_id for row in result.documents], ["doc-a", "doc-b"])


if __name__ == "__main__":
    unittest.main()
