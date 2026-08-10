"""Cached OpenAI-compatible chat completions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Set
from datetime import date, datetime
from enum import Enum
from typing import Any

from openai import AsyncOpenAI, Omit, NotGiven
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from .artifact_store import ArtifactStoreLike
from .cache import DEFAULT_CACHE_NAMESPACE, ResponseCache, canonicalize_json


LLM_CHAT_COMPLETIONS_OPERATION = "llm-chat-completions-v1"
_TRANSPORT_ONLY_FIELDS = frozenset({"timeout", "extra_headers", "extra_query"})
_MISSING = object()


def _normalize_request_value(value: Any) -> Any:
    """Match the OpenAI SDK's JSON conversion for common typed arguments."""

    if isinstance(value, (NotGiven, Omit)):
        return _MISSING
    if isinstance(value, BaseModel):
        return _normalize_request_value(
            value.model_dump(mode="json", by_alias=True, exclude_unset=True)
        )
    if isinstance(value, Enum):
        return _normalize_request_value(value.value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError("chat completion request mapping keys must be strings")
            normalized_nested = _normalize_request_value(nested)
            if normalized_nested is not _MISSING:
                normalized[key] = normalized_nested
        return normalized
    if isinstance(value, Set):
        raise TypeError("chat completion request values must not contain unordered collections")
    if isinstance(value, (list, tuple)):
        return [
            normalized
            for item in value
            for normalized in [_normalize_request_value(item)]
            if normalized is not _MISSING
        ]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        return [
            normalized
            for item in value
            for normalized in [_normalize_request_value(item)]
            if normalized is not _MISSING
        ]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(
        "chat completion request values must be JSON-like or OpenAI SDK typed values"
    )


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
        if isinstance(client, AsyncOpenAI):
            raise TypeError("CachedChatCompletions requires a synchronous OpenAI client")
        self._client = client
        if namespace is None:
            namespace = getattr(store, "namespace", DEFAULT_CACHE_NAMESPACE)
        self._cache = ResponseCache(
            store,
            namespace=namespace,
            filter_response_credentials=False,
            reject_nested_request_credentials=False,
        )

    def create(self, *, refresh: bool = False, **request: Any) -> Any:
        if not isinstance(refresh, bool):
            raise TypeError("refresh must be a boolean")
        # Validate only root field names before every provider path, including
        # uncached streams. Nested schemas are validated later in LLM mode.
        canonicalize_json({key: None for key in request})
        if request.get("stream"):
            return self._client.chat.completions.create(**request)
        base_url = str(self._client.base_url)
        provider_request: dict[str, Any] = {}
        for key, value in request.items():
            if key in _TRANSPORT_ONLY_FIELDS:
                provider_request[key] = value
                continue
            normalized = _normalize_request_value(value)
            if normalized is not _MISSING:
                provider_request[key] = normalized
        identity = {
            key: value
            for key, value in provider_request.items()
            if key not in _TRANSPORT_ONLY_FIELDS
        }
        if refresh:
            # Refresh skips the read, so object-key construction performs the
            # same complete identity validation before the provider call.
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
        response = self._client.chat.completions.create(**provider_request)
        if not isinstance(response, ChatCompletion):
            raise TypeError("chat completions create returned an unexpected response type")
        try:
            self._cache.put(
                base_url=base_url,
                operation=LLM_CHAT_COMPLETIONS_OPERATION,
                request=identity,
                response=response.model_dump(mode="json"),
            )
        except (TypeError, ValueError, OverflowError):
            # Provider responses may contain valid typed data outside this
            # cache's canonical JSON/MessagePack range. Caching is best effort.
            pass
        return response


__all__ = ["LLM_CHAT_COMPLETIONS_OPERATION", "CachedChatCompletions"]
