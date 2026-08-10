from __future__ import annotations

import unicodedata
import unittest

from trec_ragtime.baseline import ExtractiveReportGenerator, _sentences
from trec_ragtime.fusion import (
    QueryRanking,
    RetrievalQuery,
    weighted_reciprocal_rank_fusion,
)
from trec_ragtime.models import HuggingFaceDocument, PipelineResult, SearchHit
from trec_ragtime.multiretriever import (
    FusedDocument,
    MultiQueryRetrievalResult,
)
from trec_ragtime.topics import TopicRequest


def topic(*, limit: int = 5000) -> TopicRequest:
    return TopicRequest(
        topic_id="2000",
        collection_id="ragtime/2/all",
        title="Youth vaping policy",
        background="A legislator is comparing effective public-health laws.",
        problem_statement=(
            "Explain when vaping laws passed, their provisions, legal reasoning, "
            "and measured effects on teenagers."
        ),
        limit=limit,
    )


def retrieval_result(
    texts: list[tuple[str, str, str | None]],
) -> MultiQueryRetrievalResult:
    query = RetrievalQuery("2000:full", "vaping laws teenagers measured effects")
    pipeline = PipelineResult(
        query=query.text,
        pipeline=query.pipeline,
        hits=tuple(
            SearchHit(rank=index, document_id=document_id, score=1.0 / index)
            for index, (document_id, _, _) in enumerate(texts, start=1)
        ),
        processed=True,
        local_cache_hit=False,
        remote_cached=False,
        server_timestamp=None,
        cache_stored_at=None,
    )
    ranking = QueryRanking(query=query, result=pipeline)
    fused = weighted_reciprocal_rank_fusion([ranking])
    source_by_id = {document_id: source for document_id, source, _ in texts}
    english_by_id = {document_id: english for document_id, _, english in texts}
    documents = tuple(
        FusedDocument(
            hit=hit,
            document=HuggingFaceDocument(
                dataset="trec-ragtime/ragtime2",
                config="default",
                split="eng" if english_by_id[hit.document_id] is None else "spa",
                document_id=hit.document_id,
                text=source_by_id[hit.document_id],
                url=f"https://example.test/{hit.document_id}",
                date="2024-01-01",
                local_cache_hit=None,
                cache_stored_at=None,
                source="test",
                english_text=english_by_id[hit.document_id],
            ),
        )
        for hit in fused.hits
    )
    return MultiQueryRetrievalResult(
        rankings=(ranking,),
        fused_ranking=fused,
        documents=documents,
        hydrated_hit_count=len(documents),
    )


class ExtractiveReportGeneratorTests(unittest.TestCase):
    def test_generates_atomic_cited_sentences_with_exact_reference_union(self):
        retrieval = retrieval_result(
            [
                (
                    "doc-a",
                    (
                        "Parliament passed the vaping law in 2022 to restrict sales "
                        "to teenagers. A later survey measured a decline in youth "
                        "vaping after the age restriction took effect."
                    ),
                    None,
                ),
                (
                    "doc-b",
                    (
                        "The court said the vaping restrictions were justified by "
                        "the government's duty to protect public health."
                    ),
                    None,
                ),
            ]
        )

        result = ExtractiveReportGenerator().generate(
            topic(),
            retrieval,
            team_id="team",
            run_id="extractive-baseline",
            run_desc="deterministic extractive report",
        )

        report = result.report
        self.assertTrue(report.responses)
        self.assertTrue(all(sentence.citations for sentence in report.responses))
        cited = {
            document_id
            for sentence in report.responses
            for document_id in sentence.citations
        }
        self.assertEqual(set(report.references), cited)
        self.assertLessEqual(report.nfkc_character_count, 5000)
        self.assertEqual(report.metadata.topic_id, "2000")

    def test_uses_english_translation_without_changing_canonical_document_id(self):
        retrieval = retrieval_result(
            [
                (
                    "doc-spa",
                    "El parlamento aprobó una ley sobre vapeo.",
                    (
                        "Parliament passed the vaping law in 2022 to protect "
                        "teenagers from nicotine sales."
                    ),
                )
            ]
        )

        result = ExtractiveReportGenerator().generate(
            topic(),
            retrieval,
            team_id="team",
            run_id="translated",
            run_desc="translation-aware extractive report",
        )

        self.assertEqual(result.evidence[0].document_id, "doc-spa")
        self.assertEqual(result.evidence[0].language, "spa")
        self.assertTrue(result.evidence[0].translation_used)
        self.assertEqual(tuple(result.report.responses[0].citations), ("doc-spa",))
        self.assertIn("Parliament passed", result.report.responses[0].text)
        self.assertNotIn("El parlamento", result.report.responses[0].text)

    def test_enforces_document_diversity_and_nfkc_limit_during_selection(self):
        retrieval = retrieval_result(
            [
                (
                    "doc-a",
                    (
                        "The vaping law prohibited sales to teenagers in 2022. "
                        "The vaping law also prohibited advertising aimed at "
                        "teenagers. "
                        "The vaping law required warning labels for teenagers."
                    ),
                    None,
                ),
                (
                    "doc-b",
                    "A national survey measured lower youth vaping after the law.",
                    None,
                ),
            ]
        )
        generator = ExtractiveReportGenerator(
            max_sentences=5,
            max_sentences_per_document=1,
            min_sentence_characters=20,
        )

        result = generator.generate(
            topic(limit=130),
            retrieval,
            team_id="team",
            run_id="bounded",
            run_desc="bounded extractive report",
        )

        counts: dict[str, int] = {}
        for evidence in result.evidence:
            counts[evidence.document_id] = counts.get(evidence.document_id, 0) + 1
        self.assertTrue(all(count <= 1 for count in counts.values()))
        self.assertLessEqual(result.report.nfkc_character_count, 130)

    def test_falls_back_to_ranked_sentences_when_lexical_overlap_is_zero(self):
        retrieval = retrieval_result(
            [
                (
                    "doc-a",
                    "Officials announced a comprehensive regulatory measure yesterday.",
                    None,
                )
            ]
        )
        unrelated = TopicRequest(
            topic_id="2000",
            collection_id="ragtime/2/all",
            title="Completely unrelated vocabulary",
            background="A requester studies astronomy.",
            problem_statement="Describe lunar geology and telescope observations.",
            limit=5000,
        )

        result = ExtractiveReportGenerator().generate(
            unrelated,
            retrieval,
            team_id="team",
            run_id="fallback",
            run_desc="fallback extraction",
        )

        self.assertEqual(len(result.report.responses), 1)
        self.assertEqual(tuple(result.report.responses[0].citations), ("doc-a",))

    def test_sentence_extraction_preserves_acronyms(self):
        retrieval = retrieval_result(
            [
                (
                    "doc-a",
                    "U.S. lawmakers passed a vaping law to protect teenagers.",
                    None,
                )
            ]
        )

        result = ExtractiveReportGenerator().generate(
            topic(),
            retrieval,
            team_id="team",
            run_id="acronym",
            run_desc="acronym-safe extraction",
        )

        self.assertTrue(result.evidence[0].text.startswith("U.S. lawmakers"))

    def test_sentence_extraction_splits_after_a_closing_quote(self):
        retrieval = retrieval_result(
            [
                (
                    "doc-a",
                    (
                        "The minister said, “The vaping law protected teenagers.” "
                        "A national survey measured lower youth vaping after the law."
                    ),
                    None,
                )
            ]
        )

        result = ExtractiveReportGenerator(
            min_sentence_characters=20,
        ).generate(
            topic(),
            retrieval,
            team_id="team",
            run_id="quoted-sentences",
            run_desc="quote-safe sentence extraction",
        )

        self.assertEqual(len(result.evidence), 2)
        evidence_texts = tuple(item.text for item in result.evidence)
        self.assertTrue(any(text.endswith("teenagers.”") for text in evidence_texts))
        self.assertTrue(
            any(text.startswith("A national survey") for text in evidence_texts)
        )

    def test_sentence_extraction_repairs_titles_and_uppercase_acronyms(self):
        retrieval = retrieval_result(
            [
                (
                    "doc-a",
                    (
                        "Dr. Smith said the U.S. Department passed a vaping law "
                        "to protect teenagers from nicotine sales."
                    ),
                    None,
                )
            ]
        )

        result = ExtractiveReportGenerator().generate(
            topic(),
            retrieval,
            team_id="team",
            run_id="abbreviations",
            run_desc="abbreviation-safe extraction",
        )

        self.assertEqual(len(result.evidence), 1)
        self.assertTrue(result.evidence[0].text.startswith("Dr. Smith"))
        self.assertIn("U.S. Department", result.evidence[0].text)

    def test_single_hard_wrap_is_normalized_inside_one_sentence(self):
        retrieval = retrieval_result(
            [
                (
                    "doc-a",
                    (
                        "Parliament passed a vaping law to protect\n"
                        "teenagers from nicotine sales nationwide."
                    ),
                    None,
                )
            ]
        )

        result = ExtractiveReportGenerator().generate(
            topic(),
            retrieval,
            team_id="team",
            run_id="hard-wrap",
            run_desc="hard-wrap-safe extraction",
        )

        self.assertEqual(len(result.evidence), 1)
        self.assertIn("protect teenagers", result.evidence[0].text)

    def test_sentence_final_abbreviations_do_not_hide_real_boundaries(self):
        self.assertEqual(
            _sentences(
                "The policy covered taxes, sales, etc. "
                "The court upheld the law."
            ),
            (
                "The policy covered taxes, sales, etc.",
                "The court upheld the law.",
            ),
        )
        self.assertEqual(
            _sentences(
                "The agency operates in the U.S. "
                "Congress opened hearings on the law."
            ),
            (
                "The agency operates in the U.S.",
                "Congress opened hearings on the law.",
            ),
        )

    def test_paragraph_break_prevents_abbreviation_repair(self):
        self.assertEqual(
            _sentences("The list ended with etc.\n\nThe court then ruled."),
            ("The list ended with etc.", "The court then ruled."),
        )

    def test_selection_uses_nfkc_length_for_compatibility_characters(self):
        sentence = "The ﬃ vaping law protected teenagers from nicotine sales."
        self.assertGreater(
            len(unicodedata.normalize("NFKC", sentence)),
            len(sentence),
        )
        retrieval = retrieval_result([("doc-a", sentence, None)])

        with self.assertRaisesRegex(ValueError, "no extractive evidence fits"):
            ExtractiveReportGenerator(min_sentence_characters=1).generate(
                topic(limit=len(sentence)),
                retrieval,
                team_id="team",
                run_id="nfkc-limit",
                run_desc="NFKC-aware selection",
            )

    def test_raises_when_no_complete_sentence_fits(self):
        retrieval = retrieval_result(
            [
                (
                    "doc-a",
                    "The vaping law prohibited all sales to teenagers nationwide.",
                    None,
                )
            ]
        )

        with self.assertRaisesRegex(ValueError, "no extractive evidence fits"):
            ExtractiveReportGenerator(min_sentence_characters=10).generate(
                topic(limit=20),
                retrieval,
                team_id="team",
                run_id="too-short",
                run_desc="cannot fit",
            )


if __name__ == "__main__":
    unittest.main()
