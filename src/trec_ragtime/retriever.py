"""High-level cached search followed by Hugging Face document hydration."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol

from .client import DEFAULT_PIPELINE, RagtimeClient, RagtimeError
from .huggingface import (
    DEFAULT_HF_DATASET,
    DEFAULT_HF_SPLITS,
    HuggingFaceCorpus,
    HuggingFaceHydrationError,
    hf_cache_ttl_from_env,
)
from .local_corpus import (
    LocalCorpusConfigurationError,
    LocalCorpusIndex,
    default_local_corpus_path,
)
from .models import (
    Document,
    HuggingFaceDocument,
    HydrationFailure,
    HydrationResult,
    RetrievalResult,
    RetrievedDocument,
)


class DocumentCorpus(Protocol):
    """Structural interface shared by local and Dataset Viewer hydration."""

    backend_name: str

    def enrich_documents(
        self,
        document_ids: Iterable[str],
        *,
        refresh: bool = False,
    ) -> HydrationResult: ...

    def get_many(
        self,
        document_ids: Iterable[str],
        *,
        refresh: bool = False,
    ) -> HydrationResult: ...

    def close(self) -> None: ...


class RagtimeRetriever:
    """Compose cached private retrieval with cached public corpus hydration."""

    def __init__(
        self,
        client: RagtimeClient,
        corpus: DocumentCorpus,
    ) -> None:
        self.client = client
        self.corpus = corpus

    @classmethod
    def from_env(
        cls,
        *,
        document_backend: str | None = None,
        local_index_path: str | os.PathLike[str] | None = None,
        local_corpus_options: Mapping[str, Any] | None = None,
        corpus_options: Mapping[str, Any] | None = None,
        **client_overrides: Any,
    ) -> "RagtimeRetriever":
        """Build both clients after the root ``.env`` has been decrypted."""

        client = RagtimeClient.from_env(**client_overrides)
        configured_backend = os.getenv("RAGTIME_DOCUMENT_BACKEND", "auto")
        backend = configured_backend if document_backend is None else document_backend
        if not isinstance(backend, str) or backend.strip().lower() not in {
            "auto",
            "local",
            "viewer",
        }:
            client.close()
            raise ValueError("document_backend must be 'auto', 'local', or 'viewer'")
        backend = backend.strip().lower()
        configured_local_path = os.getenv("RAGTIME_LOCAL_CORPUS_PATH", "").strip()
        local_path_is_explicit = local_index_path is not None or bool(
            configured_local_path
        )
        try:
            index_path = (
                default_local_corpus_path()
                if local_index_path is None
                else Path(local_index_path).expanduser()
            )
        except Exception:
            client.close()
            raise
        if not index_path.is_absolute():
            client.close()
            raise ValueError("local_index_path must be absolute")
        if backend == "auto":
            if corpus_options and local_corpus_options:
                client.close()
                raise ValueError(
                    "corpus_options and local_corpus_options cannot both be supplied"
                )
            if local_corpus_options or local_path_is_explicit:
                backend = "local"
            elif corpus_options:
                # Preserve the pre-local-index API: Dataset Viewer options imply
                # that the caller deliberately selected that implementation.
                backend = "viewer"
            elif index_path.exists() or index_path.is_symlink():
                backend = "local"
            else:
                backend = "viewer"

        try:
            if backend == "local":
                if not index_path.exists() and not index_path.is_symlink():
                    raise LocalCorpusConfigurationError(
                        f"local corpus index does not exist: {index_path}; "
                        "run `uv run ragtime-corpus build` first"
                    )
                if corpus_options:
                    raise ValueError(
                        "corpus_options apply only to the Dataset Viewer backend"
                    )
                local_options = dict(local_corpus_options or {})
                if "path" in local_options or "local_index_path" in local_options:
                    raise ValueError(
                        "pass local_index_path directly, not in local_corpus_options"
                    )
                reserved_identity_options = {
                    "expected_dataset",
                    "expected_splits",
                }.intersection(local_options)
                if reserved_identity_options:
                    raise ValueError(
                        "canonical local corpus identity is managed by RagtimeRetriever"
                    )
                local_options["expected_dataset"] = DEFAULT_HF_DATASET
                local_options["expected_splits"] = DEFAULT_HF_SPLITS
                corpus: DocumentCorpus = LocalCorpusIndex(
                    index_path,
                    **local_options,
                )
            else:
                if local_corpus_options:
                    raise ValueError(
                        "local_corpus_options apply only to the local backend"
                    )
                options = dict(corpus_options or {})
                shared_cache_options = {
                    "cache",
                    "cache_path",
                    "cache_namespace",
                    "use_cache",
                }
                if shared_cache_options.intersection(options):
                    raise ValueError(
                        "RagtimeRetriever always shares the search client's cache"
                    )
                options.setdefault("cache_ttl_seconds", hf_cache_ttl_from_env())
                options["cache"] = client.cache
                options["use_cache"] = client.cache is not None
                corpus = HuggingFaceCorpus(**options)
        except Exception:
            client.close()
            raise
        return cls(client, corpus)

    @property
    def cache_path(self) -> Path | None:
        return self.client.cache_path

    @property
    def document_backend(self) -> str:
        return self.corpus.backend_name

    @property
    def corpus_path(self) -> Path | None:
        return self.corpus.path if isinstance(self.corpus, LocalCorpusIndex) else None

    def close(self) -> None:
        try:
            self.corpus.close()
        finally:
            self.client.close()

    def __enter__(self) -> "RagtimeRetriever":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def enrich_documents(
        self,
        document_ids: Iterable[str],
        *,
        refresh_documents: bool = False,
    ) -> HydrationResult:
        """Enrich any number of exact IDs using the configured corpus backend."""

        if not isinstance(refresh_documents, bool):
            raise ValueError("refresh_documents must be a boolean")
        enrich = getattr(self.corpus, "enrich_documents", None)
        if enrich is not None:
            return enrich(document_ids, refresh=refresh_documents)
        return self.corpus.get_many(document_ids, refresh=refresh_documents)

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

        By default no private ``/content`` calls are made. IDs absent from the
        configured corpus may use a bounded number of cached ``/content``
        requests through ``max_content_fallbacks``. A broad Dataset Viewer
        failure raises instead of fanning out; setting
        ``fallback_on_hf_error=True`` explicitly permits the same bounded
        fallback for that remote-backend case.
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
        hydration = self.enrich_documents(
            (hit.document_id for hit in hits),
            refresh_documents=refresh_documents,
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
                        content_source=hydrated.source,
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


__all__ = ["DocumentCorpus", "RagtimeRetriever"]
