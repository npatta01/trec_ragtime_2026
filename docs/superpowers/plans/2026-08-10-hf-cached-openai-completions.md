# HF-Cached OpenAI Completions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a thin cache-aside wrapper around `OpenAI.chat.completions.create` that reuses exact, validated responses from the existing private Hugging Face Artifact Store.

**Architecture:** `CachedChatCompletions` accepts an already-configured synchronous `OpenAI` client and an existing `ArtifactStoreLike`. It reuses `ResponseCache` for request identity and artifact storage, delegates streams unchanged, normalizes common typed SDK request values, and returns the SDK's normal `ChatCompletion` on both live and cached paths. Narrowly configured `ResponseCache` modes preserve the entire typed LLM response payload and permit legitimate nested schema property names while retaining top-level request credential rejection.

**Tech Stack:** Python 3.11+, `openai>=2.47.0,<3`, `msgspec`, existing `ArtifactStore`/`ResponseCache`, `unittest`, `uv`.

## Global Constraints

- Wrap only synchronous `client.chat.completions.create`; do not add provider, prompt, workflow, retry, batching, async, or streaming-cache frameworks.
- Reuse `ResponseCache` and `ArtifactStore`; do not add another storage implementation or envelope.
- Use operation `llm-chat-completions-v1`, `ttl_seconds=None`, and explicit `refresh=True` for replacement.
- Identity includes normalized base URL, namespace, and every explicit JSON request-body argument, including `extra_body`.
- Identity excludes `refresh`, `timeout`, `extra_headers`, and `extra_query`; client API keys and transport credentials never enter cache identity or payload.
- Preserve list order and canonicalize mapping order. Omitted and explicitly supplied defaults remain distinct exact requests, except `NotGiven`/`Omit` sentinels, which are treated as absent.
- Cache the entire validated `ChatCompletion.model_dump(mode="json")`, including usage and provider-extension fields.
- Cache only complete non-streaming successes. Provider errors, unexpected response types, and streams remain uncached.
- Cache failures must not discard a valid live response.
- Do not make live provider calls or write synthetic artifacts to the real HF Bucket during automated verification.

---

### Task 1: Let a ResponseCache instance preserve complete trusted responses

**Files:**
- Modify: `src/trec_ragtime/cache.py`
- Test: `tests/test_response_cache.py`

**Interfaces:**
- Consumes: existing `ResponseCache(store, *, namespace, clock)`.
- Produces: `ResponseCache(store, *, namespace, clock, filter_response_credentials: bool = True, reject_nested_request_credentials: bool = True)`. Defaults retain recursive request credential rejection; the LLM adapter may permit nested schema property names while preserving top-level rejection.

- [ ] **Step 1: Write failing tests**

Add tests with these complete assertions:

```python
def test_opted_in_cache_preserves_all_response_fields(self) -> None:
    cache = ResponseCache(
        self.store,
        clock=lambda: self.now,
        filter_response_credentials=False,
    )
    response = {
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        "provider_extension": {"token_label": "model-output"},
    }
    stored_at = cache.put(
        base_url=self.base_url,
        operation="llm-chat-completions-v1",
        request={"model": "m", "messages": []},
        response=response,
    )
    entry = cache.get(
        base_url=self.base_url,
        operation="llm-chat-completions-v1",
        request={"model": "m", "messages": []},
        ttl_seconds=None,
        validator=dict,
    )
    self.assertEqual(stored_at, self.now)
    self.assertIsNotNone(entry)
    self.assertEqual(entry.value, response)

def test_complete_response_mode_does_not_relax_request_identity(self) -> None:
    cache = ResponseCache(self.store, filter_response_credentials=False)
    with self.assertRaisesRegex(ValueError, "credential fields"):
        cache.get(
            base_url=self.base_url,
            operation="llm-chat-completions-v1",
            request={"api_key": "never-store-this"},
            ttl_seconds=None,
            validator=dict,
        )
    self.assertEqual(self.store.read_validated_calls, [])

def test_default_cache_still_rejects_sensitive_response_fields(self) -> None:
    with self.assertRaisesRegex(ValueError, "credential fields"):
        self.cache.put(
            base_url=self.base_url,
            operation=self.operation,
            request=self.request,
            response={"access_token": "never-store-this"},
        )
```

- [ ] **Step 2: Run RED**

```bash
uv run python -m unittest tests.test_response_cache -v
```

Expected: the two constructor calls using `filter_response_credentials` fail because the option does not exist; existing tests still pass.

- [ ] **Step 3: Implement the minimal option**

Add `filter_response_credentials: bool = True` to `ResponseCache.__init__`, reject non-booleans, and store it. Extend `canonicalize_json` with an internal keyword-only `filter_credentials: bool = True`; recursive calls must retain that value. Request digest/canonical request functions always use the default. In `ResponseCache._decoder` and `ResponseCache.put`, canonicalize only the response payload with `filter_credentials=self.filter_response_credentials`.

Do not change object-key construction, envelope fields, namespace behavior, TTL behavior, or any existing caller default.

- [ ] **Step 4: Run GREEN and commit**

```bash
uv run python -m unittest tests.test_response_cache -v
git diff --check
git add src/trec_ragtime/cache.py tests/test_response_cache.py
git commit -m "feat: preserve trusted cache response payloads"
```

Expected: all response-cache tests pass without warnings; the commit contains only Task 1 files.

### Task 2: Implement the cached OpenAI chat-completions wrapper

**Files:**
- Create: `src/trec_ragtime/llm_cache.py`
- Create: `tests/test_llm_cache.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: `ArtifactStoreLike`, Task 1 `ResponseCache(..., filter_response_credentials=False)`, an already-configured synchronous `openai.OpenAI`, and `openai.types.chat.ChatCompletion`.
- Produces: `LLM_CHAT_COMPLETIONS_OPERATION = "llm-chat-completions-v1"` and `CachedChatCompletions(client, store, *, namespace=DEFAULT_CACHE_NAMESPACE)` with `create(*, refresh: bool = False, **request: Any)`.

- [ ] **Step 1: Add the SDK dependency**

```bash
uv add 'openai>=2.47.0,<3'
```

Verify `pyproject.toml` retains Python `>=3.11` and `uv.lock` resolves successfully.

- [ ] **Step 2: Write the complete wrapper test suite before production code**

Create a local `MemoryArtifactStore` implementing `read_validated`, `write`, and `evict_local`; a fake client with `base_url`, `api_key`, nested `chat.completions.create`, and a call ledger; and this typed response factory:

```python
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
```

Add tests with explicit provider/cache ledger assertions for:

```python
def test_identical_call_uses_cached_typed_completion(self) -> None:
    first = self.cached.create(model="test-model", messages=self.messages)
    self.client.chat.completions.create = Mock(side_effect=AssertionError("provider called"))
    second = self.cached.create(model="test-model", messages=self.messages)
    self.assertEqual(first.model_dump(mode="json"), second.model_dump(mode="json"))
    self.assertIsInstance(second, ChatCompletion)

def test_mapping_order_is_canonical_but_message_order_is_exact(self) -> None:
    # Reordered mapping keys reuse one call; reversed message list consumes a second response.
    self.assertEqual(provider_call_count, 2)

def test_output_affecting_fields_are_identity(self) -> None:
    # Change model, tools, response_format, temperature, then extra_body one at a time.
    self.assertEqual(provider_call_count, 6)

def test_base_url_and_namespace_are_identity(self) -> None:
    self.assertNotEqual(first_store_key, second_store_key)

def test_transport_options_and_client_credentials_are_excluded(self) -> None:
    # Change timeout, extra_query, Authorization header, and client.api_key.
    self.assertEqual(provider_call_count, 1)
    self.assertNotIn(secret_one, b"".join(stored_keys_and_payloads))
    self.assertNotIn(secret_two, b"".join(stored_keys_and_payloads))

def test_refresh_replaces_exact_entry(self) -> None:
    self.assertEqual([first.id, refreshed.id, replayed.id], ["live-1", "live-2", "live-2"])
    self.assertEqual(provider_call_count, 2)

def test_stream_delegates_without_cache_io(self) -> None:
    self.assertIs(first, stream_one)
    self.assertIs(second, stream_two)
    self.assertEqual(store.read_validated_calls, [])
    self.assertEqual(store.write_calls, [])

def test_invalid_request_fails_before_provider(self) -> None:
    with self.assertRaises((TypeError, ValueError)):
        self.cached.create(model="m", messages=[], api_key="sentinel-secret")
    self.assertEqual(provider_call_count, 0)

def test_provider_error_propagates_without_write(self) -> None:
    with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
        self.cached.create(model="m", messages=[])
    self.assertEqual(store.write_calls, [])

def test_malformed_hit_becomes_miss(self) -> None:
    self.assertEqual(result.id, "live-after-malformed")
    self.assertEqual(provider_call_count, 1)

def test_cache_read_and_write_failures_preserve_live_response(self) -> None:
    self.assertEqual(read_failure_result.id, "live-read-failure")
    self.assertEqual(write_failure_result.id, "live-write-failure")

def test_complete_provider_payload_round_trips(self) -> None:
    self.assertEqual(replayed.model_dump(mode="json"), live.model_dump(mode="json"))

def test_unexpected_non_stream_response_type_is_rejected(self) -> None:
    with self.assertRaisesRegex(TypeError, "unexpected response type"):
        self.cached.create(model="m", messages=[])
```

Replace comment variables above with fixtures created inside each test; no test may depend on execution order or a real network client.

- [ ] **Step 3: Run RED**

```bash
uv run python -m unittest tests.test_llm_cache -v
```

Expected: import failure for `trec_ragtime.llm_cache`. Fix only fixture/syntax errors until the failure is specifically the missing production module.

- [ ] **Step 4: Implement the thin wrapper**

The production module contains the operation constant, a transport-only-name set, one response validator, and one class. Its core remains:

```python
class CachedChatCompletions:
    def __init__(self, client, store, *, namespace=DEFAULT_CACHE_NAMESPACE):
        self._client = client
        self._base_url = str(client.base_url)
        self._cache = ResponseCache(
            store,
            namespace=namespace,
            filter_response_credentials=False,
        )

    def create(self, *, refresh: bool = False, **request):
        if not isinstance(refresh, bool):
            raise TypeError("refresh must be a boolean")
        if request.get("stream") is True:
            return self._client.chat.completions.create(**request)
        identity = {
            key: value
            for key, value in request.items()
            if key not in {"timeout", "extra_headers", "extra_query"}
        }
        if not refresh:
            hit = self._cache.get(
                base_url=self._base_url,
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
                base_url=self._base_url,
                operation=LLM_CHAT_COMPLETIONS_OPERATION,
                request=identity,
                response=response.model_dump(mode="json"),
            )
        except (TypeError, ValueError):
            pass
        return response
```

`_chat_completion_from_mapping` copies the mapping into `ChatCompletion.model_validate`. Request canonicalization must occur before a provider call. Provider exceptions propagate. Do not introduce another class, transport protocol, retry loop, cache envelope, or provider branch.

- [ ] **Step 5: Run GREEN and commit**

```bash
uv run python -m unittest tests.test_llm_cache tests.test_response_cache -v
uv run python -m compileall -q src tests
git diff --check
git add pyproject.toml uv.lock src/trec_ragtime/llm_cache.py tests/test_llm_cache.py
git commit -m "feat: cache OpenAI-compatible chat completions"
```

Expected: all focused tests pass without warnings; compile and diff checks exit zero.

### Task 3: Export, document, and verify the integration

**Files:**
- Modify: `src/trec_ragtime/__init__.py`
- Modify: `README.md`
- Test: `tests/test_llm_cache.py`

**Interfaces:**
- Consumes: Task 2 `CachedChatCompletions`.
- Produces: package-root import `from trec_ragtime import CachedChatCompletions` plus OpenRouter and LiteLLM-proxy examples.

- [ ] **Step 1: Write and run the failing export test**

```python
def test_cached_chat_completions_is_exported_from_package_root(self) -> None:
    from trec_ragtime import CachedChatCompletions as Exported
    self.assertIs(Exported, CachedChatCompletions)
```

```bash
uv run python -m unittest tests.test_llm_cache.CachedChatCompletionsTestCase.test_cached_chat_completions_is_exported_from_package_root -v
```

Expected: import failure because the package root does not export the class.

- [ ] **Step 2: Add the export and README examples**

Export only `CachedChatCompletions` from `trec_ragtime.__init__`. Document this shared example:

```python
client = OpenAI(api_key=provider_key, base_url=provider_url, max_retries=0)
cached = CachedChatCompletions(client, ArtifactStore.from_env())
response = cached.create(model=model, messages=messages)
```

Use `https://openrouter.ai/api/v1` for OpenRouter. Describe LiteLLM using its configured OpenAI-compatible `/v1` proxy URL without inventing a deployment address. Explain exact identity, full typed response preservation, immutable entries, `refresh=True`, uncached streams, cache-failure fallback, and client-credential exclusion.

- [ ] **Step 3: Run focused and full verification**

```bash
sops decrypt --output .env config/secrets.sops.env
git check-ignore .env
uv run python -m unittest tests.test_llm_cache tests.test_response_cache -v
uv run python -m unittest discover -s tests -v
uv run python -m compileall -q src tests
git diff --check
```

Expected: `.env` is ignored; all tests pass with zero failures/errors; compile and diff checks exit zero.

- [ ] **Step 4: Run an offline zero-call probe and commit**

Using the real wrapper and an in-memory store, make one fake provider call, replace the provider with a tripwire, and repeat the exact request. Assert typed equality, one total provider call, and no HF API/filesystem calls on replay. Do not write to the real bucket.

```bash
git add src/trec_ragtime/__init__.py README.md tests/test_llm_cache.py
git commit -m "docs: expose cached LLM completions"
```

### Task 4: Independent review and final verification

**Files:**
- Review all changes since commit `1d8d0cb`.
- Modify only files with a tested, actionable review finding.

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: reviewed implementation with no unresolved correctness, security, regression, or scope finding.

- [ ] **Step 1: Review scope and behavior**

```bash
git diff --stat 1d8d0cb..HEAD
git diff 1d8d0cb..HEAD -- pyproject.toml src tests README.md
```

Reject unrelated refactors, a second cache envelope, provider-specific logic, client credentials in identity/payload, swallowed provider exceptions, filtered provider response fields, or cached streams.

- [ ] **Step 2: Address findings test-first**

For each finding, add one focused regression test, run it to observe the expected failure, apply the smallest production correction, and rerun that test. If there are no findings, do not change files or create an empty commit.

- [ ] **Step 3: Run final verification**

```bash
sops decrypt --output .env config/secrets.sops.env
git check-ignore .env
uv run python -m unittest discover -s tests -v
uv run python -m compileall -q src tests
git diff --check 1d8d0cb..HEAD
git status --short
```

Expected: `.env` is ignored; the full suite, compile check, and diff check pass; status is clean after intentional commits.

- [ ] **Step 4: Commit tested review corrections when present**

```bash
git add pyproject.toml uv.lock README.md \
  src/trec_ragtime/__init__.py src/trec_ragtime/cache.py \
  src/trec_ragtime/llm_cache.py tests/test_response_cache.py \
  tests/test_llm_cache.py
git commit -m "fix: address cached LLM review findings"
```

Skip this step when review produced no changes.
