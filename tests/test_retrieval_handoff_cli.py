from __future__ import annotations

import unittest

from trec_ragtime.passage_score_cache import PassageScoreArtifactValidationError
from trec_ragtime.retrieval_handoff_cli import (
    _cache_only_scorer,
    _topic_plan_digests,
    build_parser,
)


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

    def test_per_topic_plan_digest_override_is_strict(self) -> None:
        digest = "a" * 64
        self.assertEqual(_topic_plan_digests([f"2066={digest}"]), {"2066": digest})
        for bad in ("2066", "2066=ABC", "2066=" + "g" * 64):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                _topic_plan_digests([bad])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _topic_plan_digests([f"2066={digest}", f"2066={digest}"])


if __name__ == "__main__":
    unittest.main()
