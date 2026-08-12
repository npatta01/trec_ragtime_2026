"""Resolve the OpenAI-compatible provider credential from the environment."""

from __future__ import annotations

import os
from collections.abc import Mapping

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"


class ProviderConfigurationError(RuntimeError):
    """Raised when no usable provider credential is configured."""


def _setting(values: Mapping[str, str], name: str) -> str | None:
    """Read one environment value, treating blank as unset."""

    # A decrypted .env carries every documented name, so the placeholders that
    # were never filled in arrive as empty strings rather than as absent keys.
    value = values.get(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def provider_settings(environ: Mapping[str, str] | None = None) -> tuple[str, str]:
    """Return ``(base_url, api_key)`` using this repository's documented setup.

    ``OPENROUTER_API_KEY`` wins because that is the key `config/secrets.sops.env`
    carries and the decrypted ``.env`` exports; it implies the OpenRouter base
    URL. ``OPENAI_API_KEY`` with an optional ``OPENAI_BASE_URL`` is the fallback
    for a direct OpenAI account or a LiteLLM proxy. An explicit
    ``OPENAI_BASE_URL`` overrides the default in either case. The key is
    returned for the SDK to use and is never logged, stored, or placed in a
    cache key or fingerprint.
    """

    values = os.environ if environ is None else environ
    explicit_base_url = _setting(values, "OPENAI_BASE_URL")

    router_key = _setting(values, "OPENROUTER_API_KEY")
    if router_key is not None:
        return explicit_base_url or OPENROUTER_BASE_URL, router_key

    # OPEN_AI_API_KEY is pinned nuggetizer 0.0.5's own spelling; accepting it
    # keeps one decrypted .env usable by both this resolution and upstream's.
    openai_key = _setting(values, "OPENAI_API_KEY") or _setting(
        values, "OPEN_AI_API_KEY"
    )
    if openai_key is not None:
        return explicit_base_url or OPENAI_BASE_URL, openai_key

    raise ProviderConfigurationError(
        "no provider credential: set OPENROUTER_API_KEY, or OPENAI_API_KEY with "
        "an optional OPENAI_BASE_URL, in the decrypted .env"
    )


__all__ = [
    "OPENAI_BASE_URL",
    "OPENROUTER_BASE_URL",
    "ProviderConfigurationError",
    "provider_settings",
]
