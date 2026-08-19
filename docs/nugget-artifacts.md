# Where the nuggets are

The nugget stage's output for all 103 TREC RAGTIME topics, and how to read it.

## Location

| | |
|---|---|
| Bucket | `Npatta01/trec-ragtime-2026-artifacts` (public archive) |
| Object prefix | none (`RAGTIME_ARTIFACT_PREFIX` is empty) |
| Key layout | `nuggets/v1/<fingerprint>/<request[:2]>/<request>.mpk` |
| Behaviour fingerprint | `c13f8d507f3dbf3667a097b3f725020cdc599ada4b648813a2501d1fab8f1802` |
| Local L1 mirror | `<repo>/.git/ragtime-cache/artifacts-v1` |

Contents as validated:

| | |
|---|---|
| Topics | 103 / 103 |
| Nuggets | 1814 |
| Citations | 10791 |
| `check_bundle` | 103 validated, 0 invalid |
| Evidence rows hash-verified | 2472 ok, 0 mismatch, 0 unresolved |
| Schema | 2 (`NuggetSupport` carries `variant_index`) |

## Reading them

Derive the key rather than scanning the cache. `NuggetArtifactCache.get` takes the
fingerprint and the per-topic request and computes exactly one key, so it always resolves
to the current bundle:

```python
from trec_ragtime.artifact_store import ArtifactStore
from trec_ragtime.nugget_cache import NuggetArtifactCache

cache = NuggetArtifactCache(ArtifactStore.from_env())
bundle = cache.get(fingerprint, request)   # None only if genuinely absent
```

`ragtime-nuggets` rebuilds both from the same inputs, so re-running it for a topic that is
already stored is a cache hit at no provider cost — that is the resume mechanism, and it is
why a repeated run reports `cached` rather than `written`.

## Two objects per topic: read the newest

L1 holds **two** schema-2 objects for every topic, under the same behaviour fingerprint but
different request digests. They are not duplicates. The earlier one was built during an
intermediate run when some facets had not finished scoring, so it rests on **less evidence**:
mean 7.24 facets against 8.14 in the current one, differing for 63 of 103 topics.

The deterministic path is safe: a request built from today's complete scores produces the
newer key, so `cache.get` returns the good bundle and never the partial one. Only a naive
scan of the cache directory can pick the wrong object. If you scan, select by the envelope's
`stored_at` and keep the newest per `topic_id`.

## What a bundle contains

`NuggetBundle` (see `src/trec_ragtime/nuggets.py`) holds the topic's claims plus the complete
evidence table they cite:

- `nuggets[].claim` — one atomic English claim, with `importance` of `vital` or `okay`
- `nuggets[].support[]` — `(alias, assignment, variant_index)` per citation
  - `assignment` is `support` or `partial_support`; both are retained and the distinction is
    per link, never collapsed
  - `variant_index` names which of the nugget's `variants` the passage was actually judged
    against, so a citation inherited from a clustered paraphrase is distinguishable from one
    judged against the published wording
- `evidence[]` — `alias`, `document_id`, `start_char`, `end_char`, `content_hash`,
  `passage_score`, `facet_index`

Passage text is deliberately absent. The offsets plus `content_hash` prove which text was
used, and the text is recovered from the retrieval artifact, so a bundle cannot drift from
the corpus. To resolve a citation, slice the document's selected representation at
`[start_char:end_char]` and check `sha256` against `content_hash`.

## Caveats for the consumer

- **Nothing here asserts entailment.** The labels are a model's judgement, not qrels. A
  consumer needing fully attested claims filters on `assignment == "support"` and on
  `variants[variant_index]` matching the normalized claim.
- **Citations are document-scale.** Passages now hold whole documents (median 3365 chars),
  because splitting them was what stranded a claim's subject in a neighbouring passage. That
  makes each citation coarser but faithful to what was scored and judged.
- **Grounding, measured.** Of 1959 citations whose claim names an entity, 76.3% contain that
  entity in the cited span, up from 59.2% before the chunking fix; every window-adjacent miss
  is gone. The remainder are dominated by screen artifacts — split names such as "Hong Kong",
  month names, aggregate claims naming several countries — so the true rate is higher.
