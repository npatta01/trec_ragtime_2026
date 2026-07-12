"""High-level cached search followed by Hugging Face document hydration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .client import DEFAULT_PIPELINE, RagtimeClient, RagtimeError
from .huggingface import (
    HuggingFaceCorpus,
    HuggingFaceHydrationError,
    hf_cache_ttl_from_env,
)
from .models import (
    Document,
    HuggingFaceDocument,
    HydrationFailure,
    RetrievalResult,
    RetrievedDocument,
)


class RagtimeRetriever:
    """Compose cached private retrieval with cached public corpus hydration."""

    def __init__(
        self,
        client: RagtimeClient,
        corpus: HuggingFaceCorpus,
    ) -> None:
        self.client = client
        self.corpus = corpus

    @classmethod
    def from_env(
        cls,
        *,
        corpus_options: Mapping[str, Any] | None = None,
        **client_overrides: Any,
    ) -> "RagtimeRetriever":
        """Build both clients after the root ``.env`` has been decrypted."""

        client = RagtimeClient.from_env(**client_overrides)
        options = dict(corpus_options or {})
        shared_cache_options = {"cache", "cache_path", "cache_namespace", "use_cache"}
        if shared_cache_options.intersection(options):
            client.close()
            raise ValueError("RagtimeRetriever always shares the search client's cache")
        options.setdefault("cache_ttl_seconds", hf_cache_ttl_from_env())
        options["cache"] = client.cache
        options["use_cache"] = client.cache is not None
        try:
            corpus = HuggingFaceCorpus(**options)
        except Exception:
            client.close()
            raise
        return cls(client, corpus)

    @property
    def cache_path(self) -> Path | None:
        return self.client.cache_path

    def close(self) -> None:
        try:
            self.corpus.close()
        finally:
            self.client.close()

    def __enter__(self) -> "RagtimeRetriever":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def retrieve(
        self,
        query: str,
        *,
        pipeline: str = DEFAULT_PIPELINE,
        top_k: int | None = None,
        refresh_search: bool = False,
        refresh_documents: bool = False,
        max_content_fallbacks: int = 0,
        fallback_on_hf_error: bool = False,
    ) -> RetrievalResult:
        """Search, then join ranked IDs to original-language corpus text.

        By default no private ``/content`` calls are made. If every Hugging
        Face split was checked but an ID is absent, ``max_content_fallbacks``
        can opt into a bounded number of cached ``/content`` requests. A broad
        Hugging Face failure raises instead of fanning out; setting
        ``fallback_on_hf_error=True`` explicitly permits the same bounded
        fallback for that case.
        """

        if top_k is not None and (
            isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1
        ):
            raise ValueError("top_k must be a positive integer or None")
        for name, value in (
            ("refresh_search", refresh_search),
            ("refresh_documents", refresh_documents),
            ("fallback_on_hf_error", fallback_on_hf_error),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")
        if (
            isinstance(max_content_fallbacks, bool)
            or not isinstance(max_content_fallbacks, int)
            or max_content_fallbacks < 0
        ):
            raise ValueError("max_content_fallbacks must be a non-negative integer")

        search_result = self.client.search(
            query,
            pipeline=pipeline,
            refresh=refresh_search,
        )
        hits = search_result.hits if top_k is None else search_result.hits[:top_k]
        hydration = self.corpus.get_many(
            (hit.document_id for hit in hits),
            refresh=refresh_documents,
        )
        hydrated_by_id: dict[str, HuggingFaceDocument | Document] = {
            document_id: document for document_id, document in hydration.by_id().items()
        }
        unresolved_ids = [
            hit.document_id for hit in hits if hit.document_id not in hydrated_by_id
        ]
        if hydration.failures and unresolved_ids and not fallback_on_hf_error:
            raise HuggingFaceHydrationError(hydration.failures)

        failures = list(hydration.failures)
        fallback_ids = unresolved_ids[:max_content_fallbacks]
        for document_id in fallback_ids:
            try:
                document = self.client.get_content(
                    document_id,
                    refresh=refresh_documents,
                )
            except RagtimeError as exc:
                failures.append(
                    HydrationFailure(
                        split="ragtime2/content",
                        message=(
                            f"private content fallback failed for {document_id}: {exc}"
                        ),
                    )
                )
                continue
            hydrated_by_id[document_id] = document

        documents: list[RetrievedDocument] = []
        for hit in hits:
            hydrated = hydrated_by_id.get(hit.document_id)
            if hydrated is None:
                documents.append(
                    RetrievedDocument(
                        rank=hit.rank,
                        document_id=hit.document_id,
                        score=hit.score,
                        text=None,
                        url=None,
                        date=None,
                        language=None,
                        content_source=None,
                        content_local_cache_hit=None,
                    )
                )
                continue

            if isinstance(hydrated, HuggingFaceDocument):
                documents.append(
                    RetrievedDocument(
                        rank=hit.rank,
                        document_id=hit.document_id,
                        score=hit.score,
                        text=hydrated.text,
                        url=hydrated.url,
                        date=hydrated.date,
                        language=hydrated.language,
                        content_source="huggingface",
                        content_local_cache_hit=hydrated.local_cache_hit,
                    )
                )
            else:
                documents.append(
                    RetrievedDocument(
                        rank=hit.rank,
                        document_id=hit.document_id,
                        score=hit.score,
                        text=hydrated.text,
                        url=None,
                        date=None,
                        language=None,
                        content_source="ragtime_api",
                        content_local_cache_hit=hydrated.local_cache_hit,
                    )
                )

        return RetrievalResult(
            search_result=search_result,
            documents=tuple(documents),
            hydration_failures=tuple(failures),
        )


__all__ = ["RagtimeRetriever"]
