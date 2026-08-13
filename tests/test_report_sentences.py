from __future__ import annotations

import unicodedata
import unittest

from trec_ragtime.report_sentences import (
    LENGTH_PACKING_VERSION,
    SENTENCE_SEGMENTER_VERSION,
    RuleBasedSegmenter,
    budgeted_character_count,
    default_segmenter,
    normalize_report_text,
    pack_report_sentences,
    report_character_count,
)


class VersionConstantTests(unittest.TestCase):
    def test_behavior_versions_are_non_empty_strings(self) -> None:
        for name, value in (
            ("SENTENCE_SEGMENTER_VERSION", SENTENCE_SEGMENTER_VERSION),
            ("LENGTH_PACKING_VERSION", LENGTH_PACKING_VERSION),
        ):
            with self.subTest(constant=name):
                self.assertIsInstance(value, str)
                self.assertTrue(value.strip())

    def test_default_segmenter_reports_the_pinned_version(self) -> None:
        self.assertEqual(default_segmenter().version, SENTENCE_SEGMENTER_VERSION)


class NormalizationTests(unittest.TestCase):
    def test_compatibility_forms_are_folded(self) -> None:
        cases = (
            ("ﬁle", "file"),
            ("①", "1"),
            ("Ａ", "A"),
            ("ｅ-cigarette", "e-cigarette"),
        )
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(normalize_report_text(raw), expected)

    def test_whitespace_runs_collapse_and_edges_strip(self) -> None:
        self.assertEqual(normalize_report_text("  a\t\tb\n\nc  "), "a b c")

    def test_normalization_is_idempotent(self) -> None:
        raw = "  ﬁrst  draft.  "
        once = normalize_report_text(raw)
        self.assertEqual(normalize_report_text(once), once)

    def test_non_string_input_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            normalize_report_text(None)  # type: ignore[arg-type]


class CharacterCountTests(unittest.TestCase):
    def test_count_is_the_normalized_concatenation_length(self) -> None:
        self.assertEqual(report_character_count(["abc", "de"]), 5)

    def test_compatibility_expansion_is_counted_after_normalization(self) -> None:
        # NFKC expands the ligature to two characters, so the count must be 2.
        self.assertEqual(len("ﬁ"), 1)
        self.assertEqual(report_character_count(["ﬁ"]), 2)

    def test_count_normalizes_the_join_not_the_parts(self) -> None:
        # A combining mark leading the second sentence composes with the tail of
        # the first only once they are joined, so the joined count is smaller.
        parts = ["e", "́"]
        self.assertEqual(sum(len(unicodedata.normalize("NFKC", p)) for p in parts), 2)
        self.assertEqual(report_character_count(parts), 1)

    def test_budgeted_count_adds_one_join_character_per_gap(self) -> None:
        self.assertEqual(budgeted_character_count([]), 0)
        self.assertEqual(budgeted_character_count(["abc"]), 3)
        self.assertEqual(budgeted_character_count(["abc", "de"]), 6)
        self.assertEqual(budgeted_character_count(["a", "b", "c"]), 5)


class SegmentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.segmenter = RuleBasedSegmenter()

    def test_single_sentence_returns_one_segment(self) -> None:
        self.assertEqual(
            self.segmenter.segment("The law passed in 2019."),
            ("The law passed in 2019.",),
        )

    def test_two_sentences_split_on_terminal_punctuation(self) -> None:
        self.assertEqual(
            self.segmenter.segment("The law passed. Youth vaping fell."),
            ("The law passed.", "Youth vaping fell."),
        )

    def test_question_and_exclamation_split(self) -> None:
        self.assertEqual(
            self.segmenter.segment("Did it work? It did!"),
            ("Did it work?", "It did!"),
        )

    def test_abbreviations_do_not_split(self) -> None:
        cases = (
            "Dr. Lawrie chaired the panel.",
            "The bill cites Mr. Yang and Ms. Soldaini.",
            "The study covers vaping, e.g. disposable devices.",
            "Regulators, i.e. the ministry, acted.",
            "See Fig. 3 for the trend.",
            "Prof. Mayfield co-chaired the track.",
        )
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(self.segmenter.segment(text), (text,))

    def test_initials_do_not_split(self) -> None:
        text = "J. R. Smith drafted the measure."
        self.assertEqual(self.segmenter.segment(text), (text,))

    def test_dotted_country_abbreviations_do_not_split(self) -> None:
        for text in (
            "The U.S. Congress debated the ban.",
            "The U.K. Parliament followed.",
            "The E.U. adopted the directive.",
        ):
            with self.subTest(text=text):
                self.assertEqual(self.segmenter.segment(text), (text,))

    def test_decimals_do_not_split(self) -> None:
        text = "Prevalence fell to 3.5 percent among teenagers."
        self.assertEqual(self.segmenter.segment(text), (text,))

    def test_leading_list_ordinal_does_not_split(self) -> None:
        text = "1. The ban took effect in March."
        self.assertEqual(self.segmenter.segment(text), (text,))

    def test_lowercase_continuation_does_not_split(self) -> None:
        text = "The ministry acted in 2019. and later revised the rule"
        self.assertEqual(self.segmenter.segment(text), (text,))

    def test_closing_quote_after_terminal_punctuation_splits(self) -> None:
        self.assertEqual(
            self.segmenter.segment('He said "the ban works." Then he resigned.'),
            ('He said "the ban works."', "Then he resigned."),
        )

    def test_cjk_terminal_punctuation_splits(self) -> None:
        self.assertEqual(
            self.segmenter.segment("The rule changed。 Enforcement began."),
            ("The rule changed。", "Enforcement began."),
        )

    def test_ellipsis_run_is_one_boundary(self) -> None:
        self.assertEqual(
            self.segmenter.segment("The outcome was unclear... Later data helped."),
            ("The outcome was unclear...", "Later data helped."),
        )

    def test_blank_and_whitespace_only_text_returns_no_segments(self) -> None:
        for text in ("", "   ", "\n\t"):
            with self.subTest(text=repr(text)):
                self.assertEqual(self.segmenter.segment(text), ())

    def test_segmentation_normalizes_before_splitting(self) -> None:
        self.assertEqual(
            self.segmenter.segment("Ａ ban passed. It held."),
            ("A ban passed.", "It held."),
        )


class SingleSentenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.segmenter = RuleBasedSegmenter()

    def test_one_terminated_sentence_is_single(self) -> None:
        self.assertTrue(self.segmenter.is_single_sentence("The law passed in 2019."))

    def test_sentence_ending_in_closing_quote_is_single(self) -> None:
        self.assertTrue(self.segmenter.is_single_sentence('He called it "effective."'))

    def test_two_sentences_are_not_single(self) -> None:
        self.assertFalse(
            self.segmenter.is_single_sentence("The law passed. Youth vaping fell.")
        )

    def test_unterminated_text_is_not_single(self) -> None:
        self.assertFalse(self.segmenter.is_single_sentence("The law passed in 2019"))

    def test_blank_text_is_not_single(self) -> None:
        for text in ("", "   "):
            with self.subTest(text=repr(text)):
                self.assertFalse(self.segmenter.is_single_sentence(text))

    def test_interior_newline_is_not_single(self) -> None:
        self.assertFalse(self.segmenter.is_single_sentence("The law\npassed."))

    def test_untrimmed_text_is_not_single(self) -> None:
        self.assertFalse(self.segmenter.is_single_sentence(" The law passed. "))

    def test_abbreviation_sentence_is_single(self) -> None:
        self.assertTrue(
            self.segmenter.is_single_sentence("The U.S. Congress debated the ban.")
        )


class PackingTests(unittest.TestCase):
    def test_everything_fits_keeps_all_sentences(self) -> None:
        texts = ["Alpha one.", "Beta two."]
        kept, dropped = pack_report_sentences(texts, limit=1000)
        self.assertEqual(kept, (0, 1))
        self.assertEqual(dropped, ())

    def test_overflow_drops_whole_sentences_from_the_tail(self) -> None:
        texts = ["A" * 10, "B" * 10, "C" * 10]
        # 10 + 1 + 10 = 21 fits; adding the third needs 32.
        kept, dropped = pack_report_sentences(texts, limit=21)
        self.assertEqual(kept, (0, 1))
        self.assertEqual(dropped, (2,))

    def test_a_later_short_sentence_is_not_promoted_past_an_overflow(self) -> None:
        texts = ["A" * 10, "B" * 100, "C"]
        kept, dropped = pack_report_sentences(texts, limit=20)
        self.assertEqual(kept, (0,))
        self.assertEqual(dropped, (1, 2))

    def test_exact_boundary_is_accepted(self) -> None:
        texts = ["A" * 10, "B" * 10]
        kept, dropped = pack_report_sentences(texts, limit=21)
        self.assertEqual(kept, (0, 1))
        self.assertEqual(dropped, ())
        self.assertEqual(budgeted_character_count(texts), 21)

    def test_one_character_over_the_limit_drops_the_sentence(self) -> None:
        texts = ["A" * 10, "B" * 10]
        kept, dropped = pack_report_sentences(texts, limit=20)
        self.assertEqual(kept, (0,))
        self.assertEqual(dropped, (1,))

    def test_packing_is_idempotent(self) -> None:
        texts = ["A" * 10, "B" * 10, "C" * 10]
        kept, _ = pack_report_sentences(texts, limit=21)
        repacked, dropped = pack_report_sentences([texts[i] for i in kept], limit=21)
        self.assertEqual(repacked, (0, 1))
        self.assertEqual(dropped, ())

    def test_kept_sentences_are_never_modified(self) -> None:
        texts = ["Alpha one.", "Beta two.", "Gamma three."]
        kept, _ = pack_report_sentences(texts, limit=21)
        self.assertEqual([texts[i] for i in kept], ["Alpha one.", "Beta two."])

    def test_first_sentence_over_the_limit_raises(self) -> None:
        with self.assertRaises(ValueError):
            pack_report_sentences(["A" * 50], limit=10)

    def test_empty_input_raises(self) -> None:
        with self.assertRaises(ValueError):
            pack_report_sentences([], limit=10)

    def test_non_positive_limit_is_rejected(self) -> None:
        for limit in (0, -1):
            with self.subTest(limit=limit):
                with self.assertRaises(ValueError):
                    pack_report_sentences(["Alpha."], limit=limit)

    def test_packing_counts_normalized_characters(self) -> None:
        # Five ligatures normalize to ten characters, which overflows a limit of 9.
        with self.assertRaises(ValueError):
            pack_report_sentences(["ﬁ" * 5], limit=9)


if __name__ == "__main__":
    unittest.main()
