from __future__ import annotations

import unittest

from trec_ragtime.passage_score_cache import PassageScoreArtifactValidationError
from trec_ragtime.retrieval_handoff_cli import _cache_only_scorer, build_parser


class RetrievalHandoffCliTests(unittest.TestCase):
    def test_local_scorer_identity_is_reconstructed_without_model_loading(self) -> None:
        arguments = build_parser().parse_args(["--model", "provider-model"])
        config, scorer = _cache_only_scorer(arguments)
        self.assertEqual(config.fingerprint, scorer.fingerprint)
        with self.assertRaisesRegex(
            PassageScoreArtifactValidationError, "cached passage scores required"
        ):
            scorer.score("query", ())

    def test_parser_defaults_to_report_text(self) -> None:
        arguments = build_parser().parse_args(["--model", "provider-model"])
        self.assertEqual(arguments.representation, "report_text")
        self.assertEqual(arguments.scorer, "local")


if __name__ == "__main__":
    unittest.main()
