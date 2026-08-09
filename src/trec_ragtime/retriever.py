"""High-level cached search followed by Hugging Face document hydration."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

from .client import DEFAULT_PIPELINE, RagtimeClient, RagtimeError
from .huggingface import (
    DEFAULT_HF_DATASET,
    DEFAULT_HF_MT_SPLITS,
    DEFAULT_HF_SPLITS,
    HuggingFaceHydrationError,
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

DEFAULT_MAX_CONTENT_FALLBACKS = 5


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
        local_index_path: str | os.PathLike[str] | None = None,
        **client_overrides: Any,
    ) -> "RagtimeRetriever":
        """Build a cached search client and the required local corpus index."""

        client = RagtimeClient.from_env(**client_overrides)
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

        try:
            if not index_path.exists() and not index_path.is_symlink():
                raise LocalCorpusConfigurationError(
                    f"local corpus index does not exist: {index_path}; "
                    "run `uv run ragtime-corpus build` first"
                )
            corpus: DocumentCorpus = LocalCorpusIndex(
                index_path,
                expected_dataset=DEFAULT_HF_DATASET,
                expected_splits=DEFAULT_HF_SPLITS,
                expected_translation_splits=DEFAULT_HF_MT_SPLITS,
            )
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

    @property
    def corpus_identity(self) -> dict[str, Any]:
        """Return immutable provenance for artifacts hydrated by the local corpus."""

        if not isinstance(self.corpus, LocalCorpusIndex):
            raise LocalCorpusConfigurationError(
                "query baseline artifacts require a revision-pinned local corpus"
            )
        metadata = self.corpus.metadata
        return {
            "backend": self.corpus.backend_name,
            "path": str(self.corpus.path.resolve(strict=True)),
            "schema_version": metadata.schema_version,
            "dataset": metadata.dataset,
            "config": metadata.config,
            "revision": metadata.revision,
            "document_count": metadata.document_count,
            "shard_count": metadata.shard_count,
            "splits": list(metadata.splits),
            "translation_config": metadata.translation_config,
            "translation_count": metadata.translation_count,
            "translation_shard_count": metadata.translation_shard_count,
            "translation_splits": list(metadata.translation_splits),
        }

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
        max_content_fallbacks: int | None = DEFAULT_MAX_CONTENT_FALLBACKS,
        fallback_on_hf_error: bool = False,
    ) -> RetrievalResult:
        """Search, then join ranked IDs to original-language corpus text.

        IDs genuinely absent from the configured corpus automatically use up
        to five forever-cached private ``/content`` requests. Set
        ``max_content_fallbacks=0`` to disable this, a positive integer to set a
        different bound, or ``None`` for unlimited true-miss fallback. A broad
        Dataset Viewer failure raises instead of fanning out; setting
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
        if max_content_fallbacks is not None and (
            isinstance(max_content_fallbacks, bool)
            or not isinstance(max_content_fallbacks, int)
            or max_content_fallbacks < 0
        ):
            raise ValueError(
                "max_content_fallbacks must be a non-negative integer or None"
            )
        if fallback_on_hf_error and max_content_fallbacks is None:
            raise ValueError(
                "fallback_on_hf_error requires a finite max_content_fallbacks"
            )

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
        api_fallback_ids: list[str] = []
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
                break
            hydrated_by_id[document_id] = document
            api_fallback_ids.append(document_id)

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
                        english_text=hydrated.english_text,
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

        missing_report_text_ids = {
            *hydration.missing_translation_ids,
            *api_fallback_ids,
        }
        return RetrievalResult(
            search_result=search_result,
            documents=tuple(documents),
            hydration_failures=tuple(failures),
            missing_translation_ids=tuple(
                hit.document_id
                for hit in hits
                if hit.document_id in missing_report_text_ids
            ),
        )


__all__ = [
    "DEFAULT_MAX_CONTENT_FALLBACKS",
    "DocumentCorpus",
    "RagtimeRetriever",
]
