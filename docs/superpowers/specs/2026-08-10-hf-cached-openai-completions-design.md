# HF-Cached OpenAI Completions Design

## Goal

Add a thin cache-aside wrapper around an existing OpenAI-compatible client so
repeated non-streaming chat-completion calls reuse validated responses from the
project's private Hugging Face Artifact Store instead of calling the provider
again. The wrapper must work with OpenRouter, LiteLLM, and other compatible
endpoints through the normal OpenAI client `base_url` configuration.

## Scope

The first version wraps only `client.chat.completions.create`. It does not add
a provider abstraction, an LLM workflow framework, facet generation, prompt
management, streaming persistence, retries, batching, or a second cache
implementation.

## Public interface

Callers construct and retain the ordinary OpenAI client, then wrap it:

```python
from openai import OpenAI

from trec_ragtime.llm_cache import CachedChatCompletions

client = OpenAI(base_url=provider_url, api_key=provider_key)
completions = CachedChatCompletions(client, artifact_store)

response = completions.create(
    model="deepseek/deepseek-v4-flash",
    messages=messages,
    response_format=response_format,
)
```

`CachedChatCompletions.create` accepts normal synchronous chat-completion
request arguments plus the wrapper-only `refresh=False` flag and returns an
OpenAI `ChatCompletion`. Common typed SDK values are normalized to their JSON
form before hashing and dispatch. Callers that need another endpoint continue
using the underlying client directly.

## Cache identity and storage

Reuse the existing `ResponseCache` over the existing `ArtifactStore`; do not
introduce another storage layer. The operation is versioned separately as
`llm-chat-completions-v1`.

The cache identity contains:

- the normalized OpenAI client base URL;
- the versioned operation;
- every explicitly supplied JSON request-body argument, including model,
  messages, tools, response format, sampling parameters, and `extra_body`;
- the existing artifact-cache namespace.

The identity preserves list order and canonicalizes mapping order through
`ResponseCache`. Omitted defaults and explicitly supplied defaults are distinct
exact requests, except the SDK's `NotGiven`/`Omit` sentinels, which are treated
as absent. `refresh` is control metadata and is not part of the identity.
Transport-only OpenAI options such as timeout, extra headers, and extra query
parameters are passed to the live call but excluded from the response identity.

Client API keys, authorization headers, bearer tokens, and other transport
credentials must never enter the identity, artifact payload, warning, or
exception text.
Top-level request identity retains credential-field rejection. Nested tool
schemas and provider bodies preserve their property names because names such as
`password` or `auth_token` can be legitimate schema fields; client credentials
and transport options are never merged into those bodies by the wrapper.
The complete validated `ChatCompletion.model_dump(mode="json")` response is
cached without response-field-name filtering so usage data and provider
extensions are preserved. The wrapper never merges client configuration,
request headers, or transport credentials into that response payload.

Responses have no TTL. Manual `refresh=True` skips the read, performs one live
call, and replaces the same exact entry after successful validation.

## Data flow

For a non-streaming request with `refresh=False`:

1. Normalize common SDK request values, then separate cache-control and
   transport-only arguments from the JSON request body used for identity.
2. Ask `ResponseCache` for the exact request with `ttl_seconds=None`.
3. On a valid hit, reconstruct and return `openai.types.chat.ChatCompletion`.
4. On a miss, call `client.chat.completions.create` once with the normalized
   provider arguments plus untouched transport-only options.
5. Require a non-streaming `ChatCompletion`, serialize it with
   `model_dump(mode="json")`, and best-effort write it through `ResponseCache`.
6. Return the live response whether or not the cache write succeeds.

When `stream=True`, delegate directly to the OpenAI client without reading or
writing the cache. The wrapper does not consume or transform the stream.

## Validation and failures

Cached response mappings are reconstructed through OpenAI's
`ChatCompletion.model_validate`; invalid or mismatched bytes are rejected by
the existing validated artifact-read path. A malformed L1 candidate receives
the Artifact Store's normal one-time L2 repair behavior.

Provider exceptions propagate unchanged and are never cached. Artifact Store
authentication, transport, permission, server, or write failures retain the
repository's best-effort behavior: a read failure becomes a miss, a live call
may proceed, and a write failure does not discard a valid live response.
Common SDK typed values are normalized deterministically. Values that still
cannot be represented as credential-safe canonical JSON, including unordered
collections, fail before any provider call so the wrapper never creates an
ambiguous cache identity.

The wrapper adds no retry policy. Provider retry behavior remains whatever the
caller configured on the underlying OpenAI client.

## Configuration and documentation

Use the existing `RAGTIME_HF_BUCKET_ID`, `RAGTIME_ARTIFACT_PREFIX`,
`RAGTIME_ARTIFACT_CACHE_DIR`, `RAGTIME_CACHE_NAMESPACE`, and optional
`HF_TOKEN` settings. Provider API keys remain inputs to the ordinary OpenAI
client and are not new Artifact Store configuration.

Add the supported OpenAI SDK dependency and document one OpenRouter example
and one LiteLLM-proxy example. Both examples use the same wrapper and differ
only in `base_url`, API key, and model name.

## Verification

Offline tests use a fake OpenAI-compatible client and the existing in-memory or
temporary artifact-store test seams. They prove:

- two identical calls invoke the provider once and return equivalent typed
  `ChatCompletion` values;
- changing model, messages, tools, response schema, sampling parameters,
  `extra_body`, base URL, or cache namespace causes a miss;
- mapping key order does not change identity, while list order does;
- `refresh=True` invokes the provider and replaces the reusable response;
- provider errors and streams are never cached;
- malformed cached responses become safe misses;
- artifact read/write failures preserve the valid live-call path;
- request credential fields are rejected without leaking values;
- the complete typed provider response, including usage counters and provider
  extensions, survives a cache round trip;
- API keys and authorization headers do not appear in object keys or stored
  payloads;
- an L1 cache hit makes zero provider and zero HF network calls.

Run the focused tests first, then the complete offline unit-test suite after
decrypting the ignored root `.env` as required by the repository.
