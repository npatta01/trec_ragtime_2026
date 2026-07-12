"""Reusable client for the TREC RAGTIME retrieval service."""

from .cache import SQLiteResponseCache, default_cache_path
from .client import (
    DEFAULT_CONTENT_COLLECTION,
    DEFAULT_PIPELINE,
    RagtimeApiError,
    RagtimeClient,
    RagtimeConfigurationError,
    RagtimeError,
    RagtimeProtocolError,
)
from .models import Document, PipelineResult, SearchHit

__all__ = [
    "DEFAULT_CONTENT_COLLECTION",
    "DEFAULT_PIPELINE",
    "Document",
    "PipelineResult",
    "RagtimeApiError",
    "RagtimeClient",
    "RagtimeConfigurationError",
    "RagtimeError",
    "RagtimeProtocolError",
    "SQLiteResponseCache",
    "SearchHit",
    "default_cache_path",
]
