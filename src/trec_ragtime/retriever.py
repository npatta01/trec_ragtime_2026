"""High-level cached search followed by Hugging Face document hydration."""

from __future__ import annotations

import copy
import os
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

from .artifact_store import ArtifactStoreLike
from .client import (
    DEFAULT_CONTENT_COLLECTION,
    DEFAULT_PIPELINE,
    RagtimeClient,
    RagtimeError,
)
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
from .retrieval_cache import (
    ENDPOINT_DEPTH_BEHAVIOR_VERSION,
    RetrievalArtifactCache,
    RetrievalRequestIdentity,
)

DEFAULT_MAX_CONTENT_FALLBACKS = 5
_MISSING = object()


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
        retrieval_cache: RetrievalArtifactCache | None = None,
    ) -> None:
        self.client = client
        self.corpus = corpus
        if retrieval_cache is not None and not isinstance(
            retrieval_cache, RetrievalArtifactCache
        ):
            raise TypeError("retrieval_cache must be a RetrievalArtifactCache or None")
        self.retrieval_cache = retrieval_cache
        self._document_backend_snapshot: str | None = None
        self._retrieval_corpus_identity_snapshot: Mapping[str, Any] | None = None
        self._corpus_publication_token: tuple[Any, ...] | None = None
        if retrieval_cache is not None:
            # LocalCorpusIndex validates its published database at
            # construction. Capture the resulting immutable provenance once so
            # warm artifact hits never reopen DuckDB for metadata. The cheap
            # publication token below is only a lifecycle guard; it is not
            # included in the artifact identity or digest.
            self._document_backend_snapshot = corpus.backend_name
            self._corpus_publication_token = (
                self._capture_corpus_publication_token()
            )
            self._retrieval_corpus_identity_snapshot = copy.deepcopy(
                dict(self._capture_retrieval_corpus_identity())
            )

    @classmethod
    def from_env(
        cls,
        *,
        local_index_path: str | os.PathLike[str] | None = None,
        **client_overrides: Any,
    ) -> "RagtimeRetriever":
        """Build a cached search client and the required local corpus index."""

        missing = object()
        retrieval_cache = client_overrides.pop("retrieval_cache", missing)
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
        if retrieval_cache is missing:
            retrieval_cache = (
                None
                if client.artifact_store is None
                else RetrievalArtifactCache(client.artifact_store)
            )
        return cls(client, corpus, retrieval_cache=retrieval_cache)

    @property
    def artifact_store(self) -> ArtifactStoreLike | None:
        return self.client.artifact_store

    @property
    def cache_configuration(self) -> dict[str, object] | None:
        return self.client.cache_configuration

    @property
    def document_backend(self) -> str:
        if self._document_backend_snapshot is not None:
            return self._document_backend_snapshot
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

    @property
    def retrieval_corpus_identity(self) -> dict[str, Any]:
        """Return the construction-captured identity without reopening the corpus.

        Cache-enabled retrievers snapshot the complete corpus provenance while
        the validated corpus is constructed.  Runner resume and cache-only
        assembly use this seam so a warm path never consults ``metadata``
        again.  Direct no-cache retrievers retain the explicit legacy lookup.
        """

        if self._retrieval_corpus_identity_snapshot is not None:
            return copy.deepcopy(dict(self._retrieval_corpus_identity_snapshot))
        return copy.deepcopy(dict(self._capture_retrieval_corpus_identity()))

    def _capture_retrieval_corpus_identity(self) -> Mapping[str, Any]:
        """Return corpus provenance without requiring a hydration lookup.

        Production retrievers use the revision-pinned :class:`LocalCorpusIndex`.
        The structural fallback keeps this identity seam usable by importers and
        deterministic test corpora that expose their own immutable provenance.
        """

        try:
            return self.corpus_identity
        except LocalCorpusConfigurationError:
            identity = getattr(self.corpus, "corpus_identity", None)
            if not isinstance(identity, Mapping):
                raise
            return identity

    @staticmethod
    def _freeze_publication_token(value: Any) -> Any:
        """Return a comparison-safe snapshot of a custom generation token."""

        if value is None or isinstance(value, (bool, int, float, str, bytes)):
            return value
        if isinstance(value, Mapping):
            frozen_items = [
                (
                    RagtimeRetriever._freeze_publication_token(key),
                    RagtimeRetriever._freeze_publication_token(nested),
                )
                for key, nested in value.items()
            ]
            return tuple(sorted(frozen_items, key=repr))
        if isinstance(value, (list, tuple)):
            return tuple(
                RagtimeRetriever._freeze_publication_token(item) for item in value
            )
        raise LocalCorpusConfigurationError(
            "cache-enabled custom corpus_generation must be a scalar or nested "
            "JSON-like value"
        )

    def _capture_corpus_publication_token(self) -> tuple[Any, ...]:
        """Capture a cheap corpus publication token without opening DuckDB."""

        if isinstance(self.corpus, LocalCorpusIndex):
            try:
                file_stat = Path(self.corpus.path).stat()
            except (AttributeError, OSError, TypeError, ValueError) as exc:
                raise LocalCorpusConfigurationError(
                    "local corpus publication token is unavailable; "
                    "create a new RagtimeRetriever after repairing the corpus"
                ) from exc
            return (
                "local-stat",
                file_stat.st_dev,
                file_stat.st_ino,
                file_stat.st_size,
                file_stat.st_mtime_ns,
            )

        generation = getattr(self.corpus, "corpus_generation", _MISSING)
        if generation is _MISSING:
            raise LocalCorpusConfigurationError(
                "cache-enabled custom corpus must expose corpus_generation; "
                "create a new RagtimeRetriever when the corpus changes"
            )
        return ("custom-generation", self._freeze_publication_token(generation))

    def _assert_corpus_publication_unchanged(self) -> None:
        """Reject cache use after the corpus publication boundary moves."""

        if self.retrieval_cache is None:
            return
        current_token = self._capture_corpus_publication_token()
        if current_token != self._corpus_publication_token:
            raise LocalCorpusConfigurationError(
                "corpus publication changed; create a new RagtimeRetriever"
            )

    def _retrieval_corpus_identity(self) -> Mapping[str, Any]:
        if self._retrieval_corpus_identity_snapshot is not None:
            return self._retrieval_corpus_identity_snapshot
        return self._capture_retrieval_corpus_identity()

    @staticmethod
    def _validate_retrieval_arguments(
        query: str,
        *,
        pipeline: str,
        top_k: int | None,
        refresh_search: bool,
        refresh_documents: bool,
        refresh: bool,
        max_content_fallbacks: int | None,
        fallback_on_hf_error: bool,
        query_context: Mapping[str, object] | None,
    ) -> None:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(pipeline, str) or not pipeline.strip():
            raise ValueError("pipeline must be a non-empty string")
        if top_k is not None and (
            isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1
        ):
            raise ValueError("top_k must be a positive integer or None")
        for name, value in (
            ("refresh_search", refresh_search),
            ("refresh_documents", refresh_documents),
            ("refresh", refresh),
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
        if query_context is not None and not isinstance(query_context, Mapping):
            raise TypeError("query_context must be a mapping or None")

    def build_retrieval_identity(
        self,
        query: str,
        *,
        pipeline: str = DEFAULT_PIPELINE,
        top_k: int | None = None,
        max_content_fallbacks: int | None = DEFAULT_MAX_CONTENT_FALLBACKS,
        fallback_on_hf_error: bool = False,
        query_context: Mapping[str, object] | None = None,
    ) -> RetrievalRequestIdentity:
        """Build the exact request identity without search or hydration work."""

        self._validate_retrieval_arguments(
            query,
            pipeline=pipeline,
            top_k=top_k,
            refresh_search=False,
            refresh_documents=False,
            refresh=False,
            max_content_fallbacks=max_content_fallbacks,
            fallback_on_hf_error=fallback_on_hf_error,
            query_context=query_context,
        )
        base_url = getattr(self.client, "base_url", None)
        if not isinstance(base_url, str):
            raise ValueError("client must expose a normalized base_url for retrieval caching")
        backend_policy = {"document_backend": self.document_backend}
        return RetrievalRequestIdentity(
            query=query,
            pipeline=pipeline,
            top_k=top_k,
            endpoint_depth_version=ENDPOINT_DEPTH_BEHAVIOR_VERSION,
            max_content_fallbacks=max_content_fallbacks,
            fallback_on_hf_error=fallback_on_hf_error,
            content_collection=DEFAULT_CONTENT_COLLECTION,
            backend_policy=backend_policy,
            query_context=query_context,
            base_url=base_url,
            corpus_identity=self._retrieval_corpus_identity(),
        )

    # Descriptive aliases keep the pure seam discoverable to runner/importer
    # callers without introducing a second identity implementation.
    def retrieval_identity(self, *args: Any, **kwargs: Any) -> RetrievalRequestIdentity:
        return self.build_retrieval_identity(*args, **kwargs)

    def get_cached_retrieval(
        self,
        query: str,
        *,
        pipeline: str = DEFAULT_PIPELINE,
        top_k: int | None = None,
        max_content_fallbacks: int | None = DEFAULT_MAX_CONTENT_FALLBACKS,
        fallback_on_hf_error: bool = False,
        query_context: Mapping[str, object] | None = None,
    ) -> RetrievalResult | None:
        """Load one exact artifact without organizer or corpus hydration work."""

        identity = self.build_retrieval_identity(
            query,
            pipeline=pipeline,
            top_k=top_k,
            max_content_fallbacks=max_content_fallbacks,
            fallback_on_hf_error=fallback_on_hf_error,
            query_context=query_context,
        )
        if self.retrieval_cache is None:
            return None
        self._assert_corpus_publication_unchanged()
        cached = self.retrieval_cache.get(identity)
        # The corpus may be replaced while artifact I/O is in flight; never
        # return a hit observed across that publication boundary.
        self._assert_corpus_publication_unchanged()
        return cached

    def retrieve_cached(self, *args: Any, **kwargs: Any) -> RetrievalResult | None:
        return self.get_cached_retrieval(*args, **kwargs)

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
        refresh: bool = False,
        query_context: Mapping[str, object] | None = None,
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

        self._validate_retrieval_arguments(
            query,
            pipeline=pipeline,
            top_k=top_k,
            refresh_search=refresh_search,
            refresh_documents=refresh_documents,
            refresh=refresh,
            max_content_fallbacks=max_content_fallbacks,
            fallback_on_hf_error=fallback_on_hf_error,
            query_context=query_context,
        )
        identity: RetrievalRequestIdentity | None = None
        retrieval_refresh = refresh or refresh_search or refresh_documents
        if self.retrieval_cache is not None:
            identity = self.build_retrieval_identity(
                query,
                pipeline=pipeline,
                top_k=top_k,
                max_content_fallbacks=max_content_fallbacks,
                fallback_on_hf_error=fallback_on_hf_error,
                query_context=query_context,
            )
            if not retrieval_refresh:
                self._assert_corpus_publication_unchanged()
                cached = self.retrieval_cache.get(identity)
                # Fail closed if publication moved during the cache read.
                self._assert_corpus_publication_unchanged()
                if cached is not None:
                    return cached

            self._assert_corpus_publication_unchanged()

        search_result = self.client.search(
            query,
            pipeline=pipeline,
            refresh=refresh or refresh_search,
        )
        hits = search_result.hits if top_k is None else search_result.hits[:top_k]
        if top_k is not None:
            # ``RetrievalResult`` is a one-to-one join between the ranked
            # endpoint result and hydrated documents.  Keep the endpoint's
            # metadata and opaque evidence while making the client-side depth
            # truncation visible at that same domain boundary.
            search_result = replace(search_result, hits=hits)
        hydration = self.enrich_documents(
            (hit.document_id for hit in hits),
            refresh_documents=refresh or refresh_documents,
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
                    refresh=refresh or refresh_documents,
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
        result = RetrievalResult(
            search_result=search_result,
            documents=tuple(documents),
            hydration_failures=tuple(failures),
            missing_translation_ids=tuple(
                hit.document_id
                for hit in hits
                if hit.document_id in missing_report_text_ids
            ),
        )
        self._assert_corpus_publication_unchanged()
        if (
            self.retrieval_cache is not None
            and result.fully_hydrated
            and not result.hydration_failures
        ):
            assert identity is not None
            stored = self.retrieval_cache.put(identity, result)
            # ``result`` is fully materialized and ``put`` performs no corpus
            # reads, so a raced write can only persist that old snapshot under
            # its matching old key; reject the stale retriever before return.
            self._assert_corpus_publication_unchanged()
            return stored
        return result


__all__ = [
    "DEFAULT_MAX_CONTENT_FALLBACKS",
    "DocumentCorpus",
    "RagtimeRetriever",
]
