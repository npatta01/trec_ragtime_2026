"""Cached OpenAI-compatible chat completions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from openai.types.chat import ChatCompletion

from .artifact_store import ArtifactStoreLike
from .cache import DEFAULT_CACHE_NAMESPACE, ResponseCache


LLM_CHAT_COMPLETIONS_OPERATION = "llm-chat-completions-v1"
_TRANSPORT_ONLY_FIELDS = frozenset({"timeout", "extra_headers", "extra_query"})


def _chat_completion_from_mapping(payload: Mapping[str, Any]) -> ChatCompletion:
    """Validate one cached payload as an OpenAI chat completion."""

    return ChatCompletion.model_validate(dict(payload))


class CachedChatCompletions:
    """Thin response-cache adapter around a configured OpenAI client."""

    def __init__(
        self,
        client: Any,
        store: ArtifactStoreLike,
        *,
        namespace: str | None = None,
    ) -> None:
        self._client = client
        if namespace is None:
            namespace = getattr(store, "namespace", DEFAULT_CACHE_NAMESPACE)
        self._cache = ResponseCache(
            store,
            namespace=namespace,
            filter_response_credentials=False,
        )

    def create(self, *, refresh: bool = False, **request: Any) -> Any:
        if not isinstance(refresh, bool):
            raise TypeError("refresh must be a boolean")
        if request.get("stream") is True:
            return self._client.chat.completions.create(**request)
        base_url = str(self._client.base_url)
        identity = {
            key: value
            for key, value in request.items()
            if key not in _TRANSPORT_ONLY_FIELDS
        }
        if refresh:
            self._cache.object_key(
                base_url=base_url,
                operation=LLM_CHAT_COMPLETIONS_OPERATION,
                request=identity,
            )
        else:
            hit = self._cache.get(
                base_url=base_url,
                operation=LLM_CHAT_COMPLETIONS_OPERATION,
                request=identity,
                ttl_seconds=None,
                validator=_chat_completion_from_mapping,
            )
            if hit is not None:
                return hit.value
        response = self._client.chat.completions.create(**request)
        if not isinstance(response, ChatCompletion):
            raise TypeError("chat completions create returned an unexpected response type")
        try:
            self._cache.put(
                base_url=base_url,
                operation=LLM_CHAT_COMPLETIONS_OPERATION,
                request=identity,
                response=response.model_dump(mode="json"),
            )
        except (TypeError, ValueError):
            pass
        return response


__all__ = ["LLM_CHAT_COMPLETIONS_OPERATION", "CachedChatCompletions"]
