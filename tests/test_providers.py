from __future__ import annotations

import unittest
from unittest.mock import patch

from trec_ragtime.providers import (
    OPENAI_BASE_URL,
    OPENROUTER_BASE_URL,
    ProviderConfigurationError,
    provider_settings,
)


class ProviderSettingsTests(unittest.TestCase):
    """This repository documents OpenRouter; OpenAI is the fallback."""

    def test_prefers_the_documented_openrouter_variable(self) -> None:
        base_url, key = provider_settings({"OPENROUTER_API_KEY": "router"})
        self.assertEqual((base_url, key), (OPENROUTER_BASE_URL, "router"))

    def test_openrouter_wins_over_a_configured_openai_key(self) -> None:
        base_url, key = provider_settings(
            {"OPENROUTER_API_KEY": "router", "OPENAI_API_KEY": "direct"}
        )
        self.assertEqual((base_url, key), (OPENROUTER_BASE_URL, "router"))

    def test_falls_back_to_openai_with_its_own_default_base_url(self) -> None:
        base_url, key = provider_settings({"OPENAI_API_KEY": "direct"})
        self.assertEqual((base_url, key), (OPENAI_BASE_URL, "direct"))

    def test_falls_back_to_a_litellm_proxy_base_url(self) -> None:
        base_url, key = provider_settings(
            {"OPENAI_API_KEY": "direct", "OPENAI_BASE_URL": "https://proxy/v1"}
        )
        self.assertEqual((base_url, key), ("https://proxy/v1", "direct"))

    def test_accepts_the_pinned_nuggetizer_spelling_of_the_openai_variable(
        self,
    ) -> None:
        base_url, key = provider_settings({"OPEN_AI_API_KEY": "direct"})
        self.assertEqual((base_url, key), (OPENAI_BASE_URL, "direct"))

    def test_an_explicit_base_url_overrides_the_openrouter_default(self) -> None:
        base_url, _ = provider_settings(
            {"OPENROUTER_API_KEY": "router", "OPENAI_BASE_URL": "https://proxy/v1"}
        )
        self.assertEqual(base_url, "https://proxy/v1")

    def test_reads_the_process_environment_when_no_mapping_is_given(self) -> None:
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "router"}, clear=True):
            self.assertEqual(provider_settings(), (OPENROUTER_BASE_URL, "router"))

    def test_an_unfilled_env_placeholder_is_not_a_credential(self) -> None:
        # .env.example ships every name with an empty value, so python-dotenv
        # exports "" for anything the decrypted file did not fill in.
        with self.assertRaises(ProviderConfigurationError):
            provider_settings({"OPENROUTER_API_KEY": "", "OPENAI_API_KEY": "  "})

    def test_an_unfilled_base_url_placeholder_keeps_the_default(self) -> None:
        base_url, _ = provider_settings(
            {"OPENROUTER_API_KEY": "router", "OPENAI_BASE_URL": ""}
        )
        self.assertEqual(base_url, OPENROUTER_BASE_URL)

    def test_surrounding_whitespace_is_stripped_from_a_credential(self) -> None:
        _, key = provider_settings({"OPENROUTER_API_KEY": " router\n"})
        self.assertEqual(key, "router")

    def test_names_both_variables_when_neither_is_set(self) -> None:
        with self.assertRaisesRegex(ProviderConfigurationError, "OPENROUTER_API_KEY"):
            provider_settings({})
        with self.assertRaisesRegex(ProviderConfigurationError, "OPENAI_API_KEY"):
            provider_settings({})

    def test_the_error_does_not_repeat_a_configured_value(self) -> None:
        with self.assertRaises(ProviderConfigurationError) as caught:
            provider_settings({"OPENROUTER_API_KEY": "  ", "OPENAI_BASE_URL": "secret"})
        self.assertNotIn("secret", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
