"""Batched, cached access to the public RAGTIME2 Hugging Face corpus."""

from __future__ import annotations

import math
import os
import warnings
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlsplit

import requests
from requests.adapters import HTTPAdapter
from requests.auth import AuthBase
from urllib3.util.retry import Retry

from .artifact_store import ArtifactStore, ArtifactStoreLike, _legacy_cache_setting_name
from .cache import DEFAULT_CACHE_NAMESPACE, CacheEntry, ResponseCache
from .models import (
    HuggingFaceDocument,
    HydrationFailure,
    HydrationResult,
)

DEFAULT_HF_BASE_URL = "https://datasets-server.huggingface.co"
DEFAULT_HF_DATASET = "trec-ragtime/ragtime2"
DEFAULT_HF_CONFIG = "default"
DEFAULT_HF_SPLITS = ("eng", "spa", "rus", "zho")
DEFAULT_HF_MT_CONFIG = "mt_docs"
DEFAULT_HF_MT_SPLITS = ("spa.mt.eng", "rus.mt.eng", "zho.mt.eng")
DEFAULT_HF_DOCUMENT_CACHE_TTL_SECONDS = 30 * 24 * 60 * 60
DEFAULT_HF_TIMEOUT = (5.0, 45.0)
DEFAULT_HF_BATCH_SIZE = 20
RETRIABLE_STATUS_CODES = (429, 500, 502, 503, 504)
HF_CACHE_OPERATION = "huggingface_document"


class _AnonymousAuth(AuthBase):
    """Suppress Requests' implicit netrc authentication for public fetches."""

    def __call__(self, request: requests.PreparedRequest) -> requests.PreparedRequest:
        request.headers.pop("Authorization", None)
        return request


_ANONYMOUS_AUTH = _AnonymousAuth()


class HuggingFaceError(RuntimeError):
    """Base exception for Hugging Face corpus access."""


class HuggingFaceConfigurationError(HuggingFaceError):
    """Raised for invalid Hugging Face client configuration."""


class HuggingFaceApiError(HuggingFaceError):
    """Raised when Dataset Viewer cannot serve a filter request."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class HuggingFaceProtocolError(HuggingFaceError):
    """Raised when Dataset Viewer returns unsafe or malformed content."""


class HuggingFaceHydrationError(HuggingFaceError):
    """Raised when one or more required source splits could not be queried."""

    def __init__(self, failures: tuple[HydrationFailure, ...]) -> None:
        self.failures = failures
        summary = "; ".join(
            f"{failure.split}: {failure.message}" for failure in failures
        )
        super().__init__(f"Hugging Face hydration failed ({summary})")


def _validate_ttl(value: float | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HuggingFaceConfigurationError("cache TTL must be a number or None")
    if not math.isfinite(float(value)) or value < 0:
        raise HuggingFaceConfigurationError("cache TTL must be finite and non-negative")


def hf_cache_ttl_from_env() -> float | None:
    """Read the optional Hugging Face document TTL override."""

    raw_value = os.getenv("RAGTIME_HF_DOCUMENT_CACHE_TTL_SECONDS", "").strip()
    if not raw_value:
        return DEFAULT_HF_DOCUMENT_CACHE_TTL_SECONDS
    if raw_value.lower() in {"none", "off", "disabled"}:
        return None
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise HuggingFaceConfigurationError(
            "RAGTIME_HF_DOCUMENT_CACHE_TTL_SECONDS must be a number or 'none'"
        ) from exc
    _validate_ttl(value)
    return value


def _retrying_session(retries: int) -> requests.Session:
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        allowed_methods=frozenset({"GET"}),
        status_forcelist=RETRIABLE_STATUS_CODES,
        backoff_factor=0.5,
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _quoted_filter_value(value: str) -> str:
    """Return one DuckDB-compatible string literal for Dataset Viewer."""

    return "'" + value.replace("'", "''") + "'"


def _filter_expression(document_ids: Iterable[str]) -> str:
    return " OR ".join(
        f'"id"={_quoted_filter_value(document_id)}' for document_id in document_ids
    )


def _chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


class HuggingFaceCorpus:
    """Deprecated Dataset Viewer hydration backend.

    Documents are cached individually in the same artifact store used by the
    private search client. The private bearer token is neither needed nor sent.
    Use :class:`LocalCorpusIndex` through :class:`RagtimeRetriever` instead.
    """

    backend_name = "viewer"

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_HF_BASE_URL,
        dataset: str = DEFAULT_HF_DATASET,
        config: str = DEFAULT_HF_CONFIG,
        splits: tuple[str, ...] = DEFAULT_HF_SPLITS,
        cache: ResponseCache | None = None,
        artifact_store: ArtifactStoreLike | None = None,
        cache_namespace: str = DEFAULT_CACHE_NAMESPACE,
        use_cache: bool = True,
        cache_ttl_seconds: float | None = DEFAULT_HF_DOCUMENT_CACHE_TTL_SECONDS,
        batch_size: int = DEFAULT_HF_BATCH_SIZE,
        timeout: float | tuple[float, float] = DEFAULT_HF_TIMEOUT,
        retries: int = 2,
        session: requests.Session | None = None,
    ) -> None:
        warnings.warn(
            "HuggingFaceCorpus is deprecated; build and use LocalCorpusIndex "
            "through RagtimeRetriever.from_env() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        if not isinstance(use_cache, bool):
            raise HuggingFaceConfigurationError("use_cache must be a boolean")
        legacy_setting = _legacy_cache_setting_name()
        if legacy_setting is not None:
            raise HuggingFaceConfigurationError(
                f"{legacy_setting} is not supported; use RAGTIME_ARTIFACT_CACHE_DIR"
            )
        normalized_base_url = base_url.strip().rstrip("/")
        parsed_url = urlsplit(normalized_base_url)
        if parsed_url.scheme != "https" or not parsed_url.netloc:
            raise HuggingFaceConfigurationError(
                "Hugging Face base URL must be an absolute HTTPS URL"
            )
        if (
            parsed_url.username is not None
            or parsed_url.password is not None
            or parsed_url.query
            or parsed_url.fragment
        ):
            raise HuggingFaceConfigurationError(
                "Hugging Face base URL must not contain credentials, a query, or a fragment"
            )
        if not dataset.strip() or not config.strip():
            raise HuggingFaceConfigurationError("dataset and config must be non-empty")
        if not splits or any(
            not isinstance(split, str) or not split.strip() for split in splits
        ):
            raise HuggingFaceConfigurationError(
                "at least one non-empty source split is required"
            )
        if len(set(splits)) != len(splits):
            raise HuggingFaceConfigurationError("source splits must be unique")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or not 1 <= batch_size <= 100
        ):
            raise HuggingFaceConfigurationError(
                "batch_size must be an integer between 1 and 100"
            )
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise HuggingFaceConfigurationError("retries must be non-negative")
        _validate_ttl(cache_ttl_seconds)

        self.base_url = normalized_base_url
        self.dataset = dataset.strip()
        self.config = config.strip()
        self.splits = tuple(split.strip() for split in splits)
        self.cache_ttl_seconds = cache_ttl_seconds
        self.batch_size = batch_size
        self.timeout = timeout
        self._session = session if session is not None else _retrying_session(retries)
        self._owns_session = session is None

        if cache is not None and artifact_store is not None:
            raise HuggingFaceConfigurationError(
                "provide cache or artifact_store, not both"
            )
        if not use_cache and (cache is not None or artifact_store is not None):
            raise HuggingFaceConfigurationError(
                "cache or artifact_store cannot be provided when use_cache is false"
            )
        if not use_cache:
            self.cache = None
            self.artifact_store = None
        elif cache is not None:
            self.cache = cache
            self.artifact_store = getattr(cache, "store", None)
        elif artifact_store is not None:
            self.artifact_store = artifact_store
            self.cache = ResponseCache(artifact_store, namespace=cache_namespace)
        else:
            try:
                self.artifact_store = ArtifactStore.from_env(namespace=cache_namespace)
                self.cache = ResponseCache(
                    self.artifact_store, namespace=cache_namespace
                )
            except Exception as exc:
                raise HuggingFaceConfigurationError(
                    f"unable to initialize the shared cache: {exc}"
                ) from None

    @property
    def cache_configuration(self) -> dict[str, object] | None:
        """Return non-secret artifact-cache provenance for callers."""

        if self.cache is None:
            return None
        configuration: dict[str, object] = {"namespace": self.cache.namespace}
        for field in ("bucket_id", "prefix", "local_dir"):
            value = getattr(self.artifact_store, field, None)
            if value is not None:
                configuration[field] = str(value) if field == "local_dir" else value
        return configuration

    def close(self) -> None:
        if self._owns_session:
            self._session.close()

    def __enter__(self) -> "HuggingFaceCorpus":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _cache_request(self, document_id: str) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "config": self.config,
            "splits": list(self.splits),
            "variant": "original",
            "id": document_id,
        }

    def _cache_get(
        self, document_id: str, *, refresh: bool
    ) -> HuggingFaceDocument | None:
        if self.cache is None or refresh:
            return None
        request = self._cache_request(document_id)
        entry = self.cache.get(
            base_url=self.base_url,
            operation=HF_CACHE_OPERATION,
            request=request,
            ttl_seconds=self.cache_ttl_seconds,
            validator=lambda payload: self._parse_cached_payload(
                payload, expected_document_id=document_id
            ),
        )
        if entry is None:
            return None
        return self._parse_cached_document(entry, expected_document_id=document_id)

    def _cache_put(self, document: HuggingFaceDocument) -> HuggingFaceDocument:
        if self.cache is None:
            return document
        payload: dict[str, Any] = {
            "dataset": document.dataset,
            "config": document.config,
            "split": document.split,
            "id": document.document_id,
            "text": document.text,
            "url": document.url,
            "date": document.date,
        }
        stored_at = self.cache.put(
            base_url=self.base_url,
            operation=HF_CACHE_OPERATION,
            request=self._cache_request(document.document_id),
            response=payload,
        )
        return HuggingFaceDocument(
            dataset=document.dataset,
            config=document.config,
            split=document.split,
            document_id=document.document_id,
            text=document.text,
            url=document.url,
            date=document.date,
            local_cache_hit=False,
            cache_stored_at=stored_at,
        )

    def _parse_cached_document(
        self, entry: CacheEntry, *, expected_document_id: str
    ) -> HuggingFaceDocument:
        return self._parse_cached_payload(
            entry.payload,
            expected_document_id=expected_document_id,
            stored_at=entry.stored_at,
        )

    def _parse_cached_payload(
        self,
        payload: Mapping[str, Any],
        *,
        expected_document_id: str,
        stored_at: float = 0.0,
    ) -> HuggingFaceDocument:
        if payload.get("dataset") != self.dataset:
            raise HuggingFaceProtocolError("cached document has a different dataset")
        if payload.get("config") != self.config:
            raise HuggingFaceProtocolError("cached document has a different config")
        split = payload.get("split")
        if split not in self.splits:
            raise HuggingFaceProtocolError("cached document has an invalid split")
        if payload.get("id") != expected_document_id:
            raise HuggingFaceProtocolError("cached document has a different ID")
        text = payload.get("text")
        if not isinstance(text, str):
            raise HuggingFaceProtocolError("cached document text is invalid")
        url = self._optional_string(payload, "url")
        date = self._optional_string(payload, "date")
        return HuggingFaceDocument(
            dataset=self.dataset,
            config=self.config,
            split=split,
            document_id=expected_document_id,
            text=text,
            url=url,
            date=date,
            local_cache_hit=True,
            cache_stored_at=stored_at,
        )

    @staticmethod
    def _optional_string(payload: Mapping[str, Any], key: str) -> str | None:
        value = payload.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise HuggingFaceProtocolError(f"document {key} is not a string")
        return value

    def _get_filter(self, split: str, document_ids: list[str]) -> dict[str, Any]:
        self._assert_anonymous_session()
        try:
            response = self._session.get(
                f"{self.base_url}/filter",
                headers={"Accept": "application/json", "Authorization": None},
                auth=_ANONYMOUS_AUTH,
                params={
                    "dataset": self.dataset,
                    "config": self.config,
                    "split": split,
                    "where": _filter_expression(document_ids),
                    "length": 100,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            suffix = (
                f"HTTP {status_code}"
                if status_code is not None
                else exc.__class__.__name__
            )
            raise HuggingFaceApiError(
                f"Dataset Viewer /filter failed: {suffix}",
                status_code=status_code,
            ) from None
        try:
            payload = response.json()
        except ValueError:
            raise HuggingFaceProtocolError(
                "Dataset Viewer /filter returned invalid JSON"
            ) from None
        if not isinstance(payload, dict):
            raise HuggingFaceProtocolError(
                "Dataset Viewer /filter returned a non-object JSON payload"
            )
        return payload

    def _assert_anonymous_session(self) -> None:
        """Refuse a reused Session that could send credentials to the Hub."""

        headers = getattr(self._session, "headers", None)
        if isinstance(headers, Mapping):
            for name, value in headers.items():
                if str(name).lower() == "authorization" and value:
                    raise HuggingFaceConfigurationError(
                        "Hugging Face session must not contain Authorization"
                    )
        if getattr(self._session, "auth", None):
            raise HuggingFaceConfigurationError(
                "Hugging Face session must not contain authentication"
            )
        cookies = getattr(self._session, "cookies", None)
        if cookies is not None:
            try:
                cookies.clear()
            except (AttributeError, TypeError):
                raise HuggingFaceConfigurationError(
                    "Hugging Face session cookies could not be cleared"
                ) from None

    def _parse_filter_payload(
        self,
        payload: Mapping[str, Any],
        *,
        split: str,
        requested_ids: set[str],
    ) -> tuple[HuggingFaceDocument, ...]:
        if payload.get("partial") is not False:
            raise HuggingFaceProtocolError(
                "Dataset Viewer returned a partial or unconfirmed response"
            )
        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise HuggingFaceProtocolError(
                "Dataset Viewer response is missing a rows array"
            )
        num_rows_total = payload.get("num_rows_total")
        if (
            isinstance(num_rows_total, bool)
            or not isinstance(num_rows_total, int)
            or num_rows_total != len(rows)
            or num_rows_total > len(requested_ids)
        ):
            raise HuggingFaceProtocolError(
                "Dataset Viewer returned inconsistent pagination metadata"
            )
        documents: list[HuggingFaceDocument] = []
        seen_ids: set[str] = set()
        for entry in rows:
            if not isinstance(entry, dict):
                raise HuggingFaceProtocolError("Dataset Viewer returned an invalid row")
            if "truncated_cells" not in entry:
                raise HuggingFaceProtocolError(
                    "Dataset Viewer did not confirm an untruncated document row"
                )
            truncated_cells = entry["truncated_cells"]
            if not isinstance(truncated_cells, list) or truncated_cells:
                raise HuggingFaceProtocolError(
                    "Dataset Viewer returned a truncated document row"
                )
            row = entry.get("row")
            if not isinstance(row, dict):
                raise HuggingFaceProtocolError(
                    "Dataset Viewer row is missing its data object"
                )
            document_id = row.get("id")
            text = row.get("text")
            if not isinstance(document_id, str) or document_id not in requested_ids:
                raise HuggingFaceProtocolError(
                    "Dataset Viewer returned an unexpected document ID"
                )
            if document_id in seen_ids:
                raise HuggingFaceProtocolError(
                    "Dataset Viewer returned a duplicate document ID"
                )
            if not isinstance(text, str):
                raise HuggingFaceProtocolError(
                    "Dataset Viewer document is missing text"
                )
            seen_ids.add(document_id)
            documents.append(
                HuggingFaceDocument(
                    dataset=self.dataset,
                    config=self.config,
                    split=split,
                    document_id=document_id,
                    text=text,
                    url=self._optional_string(row, "url"),
                    date=self._optional_string(row, "date"),
                    local_cache_hit=False,
                    cache_stored_at=None,
                )
            )
        return tuple(documents)

    def get_many(
        self,
        document_ids: Iterable[str],
        *,
        refresh: bool = False,
    ) -> HydrationResult:
        """Hydrate IDs with at most one filter GET per source split and chunk.

        Failures are returned rather than hidden. Callers that require complete
        content can raise :class:`HuggingFaceHydrationError` from the failures.
        """

        if not isinstance(refresh, bool):
            raise ValueError("refresh must be a boolean")

        ordered_ids: list[str] = []
        seen_ids: set[str] = set()
        for document_id in document_ids:
            if not isinstance(document_id, str) or not document_id:
                raise ValueError("document IDs must be non-empty strings")
            if document_id not in seen_ids:
                seen_ids.add(document_id)
                ordered_ids.append(document_id)

        found: dict[str, HuggingFaceDocument] = {}
        missing: list[str] = []
        for document_id in ordered_ids:
            cached = self._cache_get(document_id, refresh=refresh)
            if cached is None:
                missing.append(document_id)
            else:
                found[document_id] = cached

        failures: list[HydrationFailure] = []
        stop_for_rate_limit = False
        for chunk in _chunks(missing, self.batch_size):
            unresolved = list(chunk)
            for split in self.splits:
                if not unresolved or stop_for_rate_limit:
                    break
                try:
                    payload = self._get_filter(split, unresolved)
                    documents = self._parse_filter_payload(
                        payload,
                        split=split,
                        requested_ids=set(unresolved),
                    )
                except (HuggingFaceApiError, HuggingFaceProtocolError) as exc:
                    failures.append(HydrationFailure(split=split, message=str(exc)))
                    if isinstance(exc, HuggingFaceApiError) and exc.status_code == 429:
                        stop_for_rate_limit = True
                    continue

                for document in documents:
                    stored_document = self._cache_put(document)
                    found[document.document_id] = stored_document
                unresolved = [
                    document_id
                    for document_id in unresolved
                    if document_id not in found
                ]

        return HydrationResult(
            documents=tuple(
                found[document_id]
                for document_id in ordered_ids
                if document_id in found
            ),
            unresolved_ids=tuple(
                document_id for document_id in ordered_ids if document_id not in found
            ),
            failures=tuple(failures),
            missing_translation_ids=tuple(
                document_id
                for document_id in ordered_ids
                if document_id in found and found[document_id].split != "eng"
            ),
        )

    def enrich_documents(
        self,
        document_ids: Iterable[str],
        *,
        refresh: bool = False,
    ) -> HydrationResult:
        """Enrich any number of exact document IDs from Viewer plus cache."""

        return self.get_many(document_ids, refresh=refresh)


__all__ = [
    "DEFAULT_HF_BASE_URL",
    "DEFAULT_HF_CONFIG",
    "DEFAULT_HF_DATASET",
    "DEFAULT_HF_DOCUMENT_CACHE_TTL_SECONDS",
    "DEFAULT_HF_MT_CONFIG",
    "DEFAULT_HF_MT_SPLITS",
    "DEFAULT_HF_SPLITS",
    "HuggingFaceApiError",
    "HuggingFaceConfigurationError",
    "HuggingFaceCorpus",
    "HuggingFaceError",
    "HuggingFaceHydrationError",
    "HuggingFaceProtocolError",
    "hf_cache_ttl_from_env",
]
