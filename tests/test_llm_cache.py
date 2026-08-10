from __future__ import annotations

import unittest
from collections.abc import Callable
from typing import Any
from unittest.mock import Mock

from openai.types.chat import ChatCompletion

from trec_ragtime.artifact_store import ArtifactStoreRemoteError, ArtifactValidationError
from trec_ragtime.llm_cache import (
    LLM_CHAT_COMPLETIONS_OPERATION,
    CachedChatCompletions,
)


class MemoryArtifactStore:
    """Small in-memory store double with explicit cache operation ledgers."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.read_validated_calls: list[str] = []
        self.write_calls: list[tuple[str, bytes]] = []
        self.evict_local_calls: list[str] = []
        self.read_error: BaseException | None = None
        self.write_error: BaseException | None = None

    def read_validated(
        self, object_key: str, decoder: Callable[[bytes], Any]
    ) -> Any | None:
        self.read_validated_calls.append(object_key)
        if self.read_error is not None:
            raise self.read_error
        payload = self.objects.get(object_key)
        if payload is None:
            return None
        try:
            return decoder(payload)
        except ArtifactValidationError:
            self.objects.pop(object_key, None)
            return None

    def write(self, object_key: str, payload: bytes) -> bool:
        if self.write_error is not None:
            raise self.write_error
        self.write_calls.append((object_key, payload))
        self.objects[object_key] = payload
        return True

    def evict_local(self, object_key: str) -> None:
        self.evict_local_calls.append(object_key)
        self.objects.pop(object_key, None)


def completion(content: str, *, completion_id: str) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": completion_id,
            "choices": [{
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "message": {"content": content, "refusal": None, "role": "assistant"},
            }],
            "created": 1,
            "model": "test-model",
            "object": "chat.completion",
            "usage": {"completion_tokens": 2, "prompt_tokens": 3, "total_tokens": 5},
            "provider_extension": {"token_label": "preserved"},
        }
    )


class FakeCompletions:
    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses = list(responses or [completion("live", completion_id="live-1")])
        self.calls: list[dict[str, Any]] = []

    def create(self, **request: Any) -> Any:
        self.calls.append(request)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class FakeClient:
    def __init__(
        self,
        *,
        base_url: str = "https://api.example.test",
        api_key: str = "client-secret",
        responses: list[Any] | None = None,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.chat = type("Chat", (), {})()
        self.chat.completions = FakeCompletions(responses)


class CachedChatCompletionsTestCase(unittest.TestCase):
    messages = [{"role": "user", "content": "Say hello"}]

    def setUp(self) -> None:
        self.store = MemoryArtifactStore()
        self.client = FakeClient(
            responses=[completion(f"response-{index}", completion_id=f"live-{index}") for index in range(1, 11)]
        )
        self.cached = CachedChatCompletions(self.client, self.store)

    def test_cached_chat_completions_is_exported_from_package_root(self) -> None:
        from trec_ragtime import CachedChatCompletions as Exported

        self.assertIs(Exported, CachedChatCompletions)

    def test_identical_call_uses_cached_typed_completion(self) -> None:
        first = self.cached.create(model="test-model", messages=self.messages)
        self.client.chat.completions.create = Mock(side_effect=AssertionError("provider called"))
        second = self.cached.create(model="test-model", messages=self.messages)
        self.assertEqual(first.model_dump(mode="json"), second.model_dump(mode="json"))
        self.assertIsInstance(second, ChatCompletion)
        self.assertEqual(len(self.store.read_validated_calls), 2)
        self.assertEqual(len(self.store.write_calls), 1)

    def test_mapping_order_is_canonical_but_message_order_is_exact(self) -> None:
        first = self.cached.create(
            model="test-model",
            messages=[{"role": "user", "content": "one"}, {"role": "assistant", "content": "two"}],
            metadata={"first": 1, "second": {"a": 2, "b": 3}},
        )
        second = self.cached.create(
            metadata={"second": {"b": 3, "a": 2}, "first": 1},
            messages=[{"content": "one", "role": "user"}, {"content": "two", "role": "assistant"}],
            model="test-model",
        )
        reversed_messages = self.cached.create(
            model="test-model",
            messages=[{"role": "assistant", "content": "two"}, {"role": "user", "content": "one"}],
            metadata={"first": 1, "second": {"a": 2, "b": 3}},
        )
        self.assertEqual(first.id, second.id)
        self.assertEqual(reversed_messages.id, "live-2")
        self.assertEqual(len(self.client.chat.completions.calls), 2)
        self.assertEqual(len(self.store.write_calls), 2)

    def test_output_affecting_fields_are_identity(self) -> None:
        base = {
            "model": "test-model",
            "messages": self.messages,
            "tools": [{"type": "function", "function": {"name": "one"}}],
            "response_format": {"type": "text"},
            "temperature": 0.0,
            "extra_body": {"provider_mode": "one"},
        }
        self.cached.create(**base)
        self.cached.create(**{**base, "model": "other-model"})
        self.cached.create(**{**base, "tools": [{"type": "function", "function": {"name": "two"}}]})
        self.cached.create(**{**base, "response_format": {"type": "json_object"}})
        self.cached.create(**{**base, "temperature": 0.7})
        self.cached.create(**{**base, "extra_body": {"provider_mode": "two"}})
        self.assertEqual(len(self.client.chat.completions.calls), 6)
        self.assertEqual(len(self.store.write_calls), 6)

    def test_base_url_and_namespace_are_identity(self) -> None:
        first = CachedChatCompletions(
            FakeClient(base_url="https://one.example.test"), self.store, namespace="namespace-one"
        )
        second = CachedChatCompletions(
            FakeClient(base_url="https://two.example.test"), self.store, namespace="namespace-one"
        )
        third = CachedChatCompletions(
            FakeClient(base_url="https://one.example.test"), self.store, namespace="namespace-two"
        )
        first.create(model="m", messages=[])
        second.create(model="m", messages=[])
        third.create(model="m", messages=[])
        keys = [key for key, _ in self.store.write_calls]
        self.assertEqual(len(keys), 3)
        self.assertEqual(len(set(keys)), 3)

    def test_transport_options_and_client_credentials_are_excluded(self) -> None:
        secret_one = "authorization-secret-one"
        secret_two = "authorization-secret-two"
        self.client.api_key = secret_one
        self.cached.create(
            model="m",
            messages=[],
            timeout=1.0,
            extra_query={"request-id": "one"},
            extra_headers={"Authorization": f"Bearer {secret_one}"},
        )
        self.client.api_key = secret_two
        self.cached.create(
            model="m",
            messages=[],
            timeout=30.0,
            extra_query={"request-id": "two"},
            extra_headers={"Authorization": f"Bearer {secret_two}"},
        )
        stored_keys_and_payloads = [key.encode() + payload for key, payload in self.store.write_calls]
        self.assertEqual(len(self.client.chat.completions.calls), 1)
        self.assertNotIn(secret_one.encode(), b"".join(stored_keys_and_payloads))
        self.assertNotIn(secret_two.encode(), b"".join(stored_keys_and_payloads))

    def test_refresh_replaces_exact_entry(self) -> None:
        first = self.cached.create(model="m", messages=[])
        refreshed = self.cached.create(model="m", messages=[], refresh=True)
        replayed = self.cached.create(model="m", messages=[])
        self.assertEqual([first.id, refreshed.id, replayed.id], ["live-1", "live-2", "live-2"])
        self.assertEqual(len(self.client.chat.completions.calls), 2)
        self.assertEqual(len(self.store.write_calls), 2)
        self.assertEqual(self.store.write_calls[0][0], self.store.write_calls[1][0])

    def test_stream_delegates_without_cache_io(self) -> None:
        stream_one = object()
        stream_two = object()
        self.client.chat.completions.responses = [stream_one, stream_two]
        first = self.cached.create(model="m", messages=[], stream=True)
        second = self.cached.create(model="m", messages=[], stream=True)
        self.assertIs(first, stream_one)
        self.assertIs(second, stream_two)
        self.assertEqual(self.store.read_validated_calls, [])
        self.assertEqual(self.store.write_calls, [])
        self.assertEqual(len(self.client.chat.completions.calls), 2)

    def test_invalid_request_fails_before_provider(self) -> None:
        with self.assertRaises((TypeError, ValueError)):
            self.cached.create(model="m", messages=[], api_key="sentinel-secret")
        self.assertEqual(len(self.client.chat.completions.calls), 0)
        self.assertEqual(self.store.read_validated_calls, [])
        self.assertEqual(self.store.write_calls, [])

    def test_invalid_refresh_request_fails_before_provider(self) -> None:
        with self.assertRaises((TypeError, ValueError)):
            self.cached.create(
                model="m",
                messages=[],
                api_key="sentinel-secret",
                refresh=True,
            )
        self.assertEqual(len(self.client.chat.completions.calls), 0)
        self.assertEqual(self.store.read_validated_calls, [])
        self.assertEqual(self.store.write_calls, [])

    def test_provider_error_propagates_without_write(self) -> None:
        self.client.chat.completions.responses = [RuntimeError("provider unavailable")]
        with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
            self.cached.create(model="m", messages=[])
        self.assertEqual(len(self.client.chat.completions.calls), 1)
        self.assertEqual(self.store.write_calls, [])

    def test_malformed_hit_becomes_miss(self) -> None:
        key = self.cached._cache.object_key(
            base_url=self.client.base_url,
            operation=LLM_CHAT_COMPLETIONS_OPERATION,
            request={"model": "m", "messages": []},
        )
        self.store.objects[key] = b"malformed"
        result = self.cached.create(model="m", messages=[])
        self.assertEqual(result.id, "live-1")
        self.assertEqual(len(self.client.chat.completions.calls), 1)
        self.assertEqual(len(self.store.write_calls), 1)

    def test_cache_read_and_write_failures_preserve_live_response(self) -> None:
        self.store.read_error = ArtifactStoreRemoteError("read unavailable")
        read_failure_result = self.cached.create(model="m", messages=[])
        self.assertEqual(read_failure_result.id, "live-1")
        self.assertEqual(len(self.store.write_calls), 1)

        write_store = MemoryArtifactStore()
        write_store.write_error = ArtifactStoreRemoteError("write unavailable")
        write_client = FakeClient(responses=[completion("write", completion_id="live-write-failure")])
        write_cached = CachedChatCompletions(write_client, write_store)
        write_failure_result = write_cached.create(model="m", messages=[])
        self.assertEqual(write_failure_result.id, "live-write-failure")
        self.assertEqual(write_store.write_calls, [])

    def test_complete_provider_payload_round_trips(self) -> None:
        live = completion("preserved", completion_id="complete-1")
        self.client.chat.completions.responses = [live]
        self.cached.create(model="m", messages=[])
        replayed = self.cached.create(model="m", messages=[])
        self.assertEqual(replayed.model_dump(mode="json"), live.model_dump(mode="json"))
        self.assertEqual(replayed.provider_extension, {"token_label": "preserved"})
        self.assertEqual(len(self.client.chat.completions.calls), 1)

    def test_unexpected_non_stream_response_type_is_rejected(self) -> None:
        self.client.chat.completions.responses = [{"id": "not-a-completion"}]
        with self.assertRaisesRegex(TypeError, "unexpected response type"):
            self.cached.create(model="m", messages=[])
        self.assertEqual(self.store.write_calls, [])


if __name__ == "__main__":
    unittest.main()
