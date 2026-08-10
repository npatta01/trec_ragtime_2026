from __future__ import annotations

import hashlib
import math
import unittest
from collections.abc import Callable
from typing import Any

import msgspec

from trec_ragtime.artifact_store import ArtifactValidationError
from trec_ragtime.cache import (
    CACHE_SCHEMA_VERSION,
    DEFAULT_CACHE_NAMESPACE,
    CacheEntry,
    ResponseCache,
    ResponseEnvelope,
    canonical_request_bytes,
    response_object_key,
)


EXPECTED_REQUEST_BYTES_HEX = (
    "85ae736368656d615f76657273696f6e01a96e616d657370616365b072616774696d65322d"
    "323032362d7631a8626173655f75726cb868747470733a2f2f6170692e6578616d706c652e"
    "74657374a96f7065726174696f6ea8706970656c696e65a77265717565737482a66e657374"
    "656482a16101a16292a17802a57175657279a568656c6c6f"
)
EXPECTED_OBJECT_KEY = (
    "api-responses/v1/pipeline/85/"
    "858036d77e315d4de2fe19456a5615ced8af9b441745c708670a22d2cc7a662e.mpk"
)


class MemoryArtifactStore:
    """An L1/L2 store double that exercises the ArtifactStore contract."""

    def __init__(self) -> None:
        self.local: dict[str, bytes] = {}
        self.remote: dict[str, bytes] = {}
        self.read_validated_calls: list[str] = []
        self.write_calls: list[tuple[str, bytes]] = []

    def read_validated(
        self, object_key: str, decoder: Callable[[bytes], Any]
    ) -> Any | None:
        self.read_validated_calls.append(object_key)
        payload = self.local.get(object_key)
        if payload is not None:
            try:
                return decoder(payload)
            except ArtifactValidationError:
                del self.local[object_key]
        payload = self.remote.get(object_key)
        if payload is None:
            return None
        value = decoder(payload)
        self.local[object_key] = payload
        return value

    def write(self, object_key: str, payload: bytes) -> None:
        self.write_calls.append((object_key, payload))
        self.local[object_key] = payload
        self.remote[object_key] = payload


def independent_request_bytes(
    *, base_url: str, operation: str, request: dict[str, Any]
) -> bytes:
    """Build the fixture independently of the production helper."""

    identity = {
        "schema_version": 1,
        "namespace": DEFAULT_CACHE_NAMESPACE,
        "base_url": base_url,
        "operation": operation,
        "request": {
            "nested": {"a": 1, "b": ["x", 2]},
            "query": request["query"],
        },
    }
    return msgspec.msgpack.encode(identity)


class ResponseCacheTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.store = MemoryArtifactStore()
        self.now = 100.0
        self.cache = ResponseCache(self.store, clock=lambda: self.now)
        self.base_url = "https://api.example.test"
        self.operation = "pipeline"
        self.request = {
            "query": "hello",
            "nested": {"b": ["x", 2], "a": 1},
        }
        self.response = {"scores": {"doc-1": 0.9}, "query": "hello"}

    @staticmethod
    def valid_response(payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload.get("scores"), dict):
            raise ValueError("scores must be an object")
        return payload

    def object_key(self) -> str:
        return EXPECTED_OBJECT_KEY

    def test_request_key_is_stable_for_recursively_reordered_mappings(self) -> None:
        reordered = {
            "nested": {"a": 1, "b": ["x", 2]},
            "query": "hello",
        }
        self.assertEqual(
            self.cache.object_key(
                base_url=self.base_url,
                operation=self.operation,
                request=self.request,
            ),
            self.object_key(),
        )
        self.assertEqual(
            self.cache.object_key(
                base_url=self.base_url,
                operation=self.operation,
                request=reordered,
            ),
            self.object_key(),
        )

    def test_list_order_changes_request_key(self) -> None:
        changed = {"query": "hello", "nested": {"a": 1, "b": [2, "x"]}}
        self.assertNotEqual(
            self.cache.object_key(
                base_url=self.base_url,
                operation=self.operation,
                request=changed,
            ),
            self.object_key(),
        )

    def test_base_url_query_and_fragment_credentials_are_rejected(self) -> None:
        for base_url in (
            "https://api.example.test?token=secret-token",
            "https://api.example.test#authorization=secret-token",
        ):
            with self.subTest(base_url=base_url):
                with self.assertRaises(ValueError) as raised:
                    self.cache.object_key(
                        base_url=base_url,
                        operation=self.operation,
                        request=self.request,
                    )
                self.assertNotIn("secret-token", str(raised.exception))

    def test_representative_credential_fields_are_rejected_without_persistence(self) -> None:
        credential_fields = (
            "client_secret",
            "refresh_token",
            "x-api-key",
            "accessToken",
            "clientSecret",
            "bearerToken",
            "clientAPIKey",
            "Authorization",
            "password",
            "secret",
            "token",
            "api-key",
        )
        for field in credential_fields:
            with self.subTest(field=field):
                request = {"query": "hello", field: "secret-value"}
                with self.assertRaises(ValueError) as raised:
                    self.cache.object_key(
                        base_url=self.base_url,
                        operation=self.operation,
                        request=request,
                    )
                self.assertNotIn("secret-value", str(raised.exception))
                with self.assertRaises(ValueError) as raised:
                    self.cache.put(
                        base_url=self.base_url,
                        operation=self.operation,
                        request=self.request,
                        response={"scores": {}, field: "secret-value"},
                        stored_at=100.0,
                    )
                self.assertNotIn("secret-value", str(raised.exception))
        self.assertEqual(self.store.write_calls, [])

    def test_semantic_token_count_field_is_allowed(self) -> None:
        request = {"query": "hello", "token_count": 3}
        response = {"scores": {}, "token_count": 3}

        self.cache.object_key(
            base_url=self.base_url,
            operation=self.operation,
            request=request,
        )
        self.cache.put(
            base_url=self.base_url,
            operation=self.operation,
            request=request,
            response=response,
            stored_at=100.0,
        )
        self.assertEqual(len(self.store.write_calls), 1)

    def test_ordinary_semantically_distinct_mappings_do_not_collide(self) -> None:
        first = self.cache.object_key(
            base_url=self.base_url,
            operation=self.operation,
            request={"query": "hello", "mode": "first"},
        )
        second = self.cache.object_key(
            base_url=self.base_url,
            operation=self.operation,
            request={"query": "hello", "mode": "second"},
        )
        self.assertNotEqual(first, second)

    def test_canonical_request_fixture_and_binary_digests_are_stable(self) -> None:
        expected = independent_request_bytes(
            base_url=self.base_url,
            operation=self.operation,
            request=self.request,
        )
        self.assertEqual(expected.hex(), EXPECTED_REQUEST_BYTES_HEX)
        self.assertEqual(
            canonical_request_bytes(
                base_url=self.base_url,
                namespace=DEFAULT_CACHE_NAMESPACE,
                operation=self.operation,
                request=self.request,
            ).hex(),
            EXPECTED_REQUEST_BYTES_HEX,
        )
        self.cache.put(
            base_url=self.base_url,
            operation=self.operation,
            request=self.request,
            response=self.response,
            stored_at=100.0,
        )
        envelope = msgspec.msgpack.decode(
            self.store.write_calls[-1][1], type=ResponseEnvelope
        )
        self.assertIsInstance(envelope.base_url_digest, bytes)
        self.assertEqual(len(envelope.base_url_digest), 32)
        self.assertIsInstance(envelope.request_digest, bytes)
        self.assertEqual(len(envelope.request_digest), 32)

    def test_envelope_has_strict_unknown_fields(self) -> None:
        self.cache.put(
            base_url=self.base_url,
            operation=self.operation,
            request=self.request,
            response=self.response,
            stored_at=100.0,
        )
        decoded = msgspec.msgpack.decode(self.store.write_calls[-1][1])
        decoded["unknown"] = True
        self.store.remote[self.object_key()] = msgspec.msgpack.encode(decoded)
        self.store.local[self.object_key()] = self.store.remote[self.object_key()]
        with self.assertRaises(ArtifactValidationError):
            self.cache.get(
                base_url=self.base_url,
                operation=self.operation,
                request=self.request,
                ttl_seconds=None,
                validator=self.valid_response,
            )

    def test_wrong_identity_fields_are_fatal(self) -> None:
        self.cache.put(
            base_url=self.base_url,
            operation=self.operation,
            request=self.request,
            response=self.response,
            stored_at=100.0,
        )
        original = self.store.write_calls[-1][1]
        for field, value in (
            ("schema_version", 2),
            ("namespace", "other"),
            ("operation", "content"),
            ("base_url_digest", b"x" * 32),
            ("request_digest", b"y" * 32),
        ):
            with self.subTest(field=field):
                decoded = msgspec.msgpack.decode(original)
                decoded[field] = value
                self.store.remote[self.object_key()] = msgspec.msgpack.encode(decoded)
                self.store.local.clear()
                with self.assertRaises(ArtifactValidationError):
                    self.cache.get(
                        base_url=self.base_url,
                        operation=self.operation,
                        request=self.request,
                        ttl_seconds=None,
                        validator=self.valid_response,
                    )

    def test_finite_stored_at_is_required(self) -> None:
        with self.assertRaises(ValueError):
            self.cache.put(
                base_url=self.base_url,
                operation=self.operation,
                request=self.request,
                response=self.response,
                stored_at=math.inf,
            )
        self.cache.put(
            base_url=self.base_url,
            operation=self.operation,
            request=self.request,
            response=self.response,
            stored_at=100.0,
        )
        decoded = msgspec.msgpack.decode(self.store.write_calls[-1][1])
        decoded["stored_at"] = float("nan")
        self.store.remote[self.object_key()] = msgspec.msgpack.encode(decoded)
        self.store.local.clear()
        with self.assertRaises(ArtifactValidationError):
            self.cache.get(
                base_url=self.base_url,
                operation=self.operation,
                request=self.request,
                ttl_seconds=None,
                validator=self.valid_response,
            )

    def test_malformed_and_non_object_payloads_are_rejected(self) -> None:
        self.cache.put(
            base_url=self.base_url,
            operation=self.operation,
            request=self.request,
            response=self.response,
            stored_at=100.0,
        )
        original = self.store.write_calls[-1][1]
        for payload in ([], "not an object", {"scores": b"not-json"}):
            with self.subTest(payload=payload):
                decoded = msgspec.msgpack.decode(original)
                decoded["payload"] = payload
                self.store.remote[self.object_key()] = msgspec.msgpack.encode(decoded)
                self.store.local.clear()
                with self.assertRaises(ArtifactValidationError):
                    self.cache.get(
                        base_url=self.base_url,
                        operation=self.operation,
                        request=self.request,
                        ttl_seconds=None,
                        validator=self.valid_response,
                    )

    def test_local_domain_corruption_requests_one_store_repair(self) -> None:
        self.cache.put(
            base_url=self.base_url,
            operation=self.operation,
            request=self.request,
            response=self.response,
            stored_at=100.0,
        )
        key = self.object_key()
        local = msgspec.msgpack.decode(self.store.local[key])
        local["payload"] = {"scores": []}
        self.store.local[key] = msgspec.msgpack.encode(local)
        entry = self.cache.get(
            base_url=self.base_url,
            operation=self.operation,
            request=self.request,
            ttl_seconds=None,
            validator=self.valid_response,
        )
        self.assertEqual(entry, CacheEntry(payload=self.response, stored_at=100.0))
        self.assertEqual(self.store.read_validated_calls, [key])
        self.assertEqual(self.store.local[key], self.store.remote[key])

    def test_remote_domain_corruption_is_fatal(self) -> None:
        self.cache.put(
            base_url=self.base_url,
            operation=self.operation,
            request=self.request,
            response=self.response,
            stored_at=100.0,
        )
        key = self.object_key()
        remote = msgspec.msgpack.decode(self.store.remote[key])
        remote["payload"] = {"scores": []}
        self.store.remote[key] = msgspec.msgpack.encode(remote)
        self.store.local.clear()
        with self.assertRaises(ArtifactValidationError):
            self.cache.get(
                base_url=self.base_url,
                operation=self.operation,
                request=self.request,
                ttl_seconds=None,
                validator=self.valid_response,
            )

    def test_validation_precedes_ttl_evaluation(self) -> None:
        self.cache.put(
            base_url=self.base_url,
            operation=self.operation,
            request=self.request,
            response=self.response,
            stored_at=100.0,
        )
        self.now = 200.0
        calls: list[dict[str, Any]] = []

        def validator(payload: dict[str, Any]) -> dict[str, Any]:
            calls.append(payload)
            raise ValueError("invalid domain payload")

        with self.assertRaises(ArtifactValidationError):
            self.cache.get(
                base_url=self.base_url,
                operation=self.operation,
                request=self.request,
                ttl_seconds=1.0,
                validator=validator,
            )
        self.assertEqual(calls, [self.response, self.response])

    def test_valid_expired_envelope_is_a_miss_after_validation(self) -> None:
        self.cache.put(
            base_url=self.base_url,
            operation=self.operation,
            request=self.request,
            response=self.response,
            stored_at=100.0,
        )
        self.now = 101.0
        seen: list[dict[str, Any]] = []
        entry = self.cache.get(
            base_url=self.base_url,
            operation=self.operation,
            request=self.request,
            ttl_seconds=1.0,
            validator=lambda payload: seen.append(payload) or payload,
        )
        self.assertIsNone(entry)
        self.assertEqual(seen, [self.response])
        self.assertIn(self.object_key(), self.store.remote)

    def test_put_encodes_once_and_returns_stored_at_without_read(self) -> None:
        stored_at = self.cache.put(
            base_url=self.base_url,
            operation=self.operation,
            request=self.request,
            response=self.response,
            stored_at=123.5,
        )
        self.assertEqual(stored_at, 123.5)
        self.assertEqual(self.store.read_validated_calls, [])
        self.assertEqual(len(self.store.write_calls), 1)

    def test_injected_clock_makes_bytes_stable_and_writes_are_last_writer_wins(self) -> None:
        self.cache.put(
            base_url=self.base_url,
            operation=self.operation,
            request=self.request,
            response=self.response,
        )
        first = self.store.write_calls[-1][1]
        self.cache.put(
            base_url=self.base_url,
            operation=self.operation,
            request=self.request,
            response=self.response,
        )
        self.assertEqual(first, self.store.write_calls[-1][1])
        newer = {"scores": {"doc-2": 0.8}, "query": "hello"}
        self.now = 101.0
        self.cache.put(
            base_url=self.base_url,
            operation=self.operation,
            request=self.request,
            response=newer,
        )
        self.assertEqual(
            self.cache.get(
                base_url=self.base_url,
                operation=self.operation,
                request=self.request,
                ttl_seconds=None,
                validator=self.valid_response,
            ),
            CacheEntry(payload=newer, stored_at=101.0),
        )

    def test_object_key_helper_uses_exact_layout(self) -> None:
        digest = bytes.fromhex("ab" * 32)
        self.assertEqual(
            response_object_key("content", digest),
            "api-responses/v1/content/ab/" + ("ab" * 32) + ".mpk",
        )


if __name__ == "__main__":
    unittest.main()
