"""Private Hugging Face Bucket storage with a disposable local simplecache.

The store deliberately knows nothing about the bytes it stores.  Domain
adapters provide a decoder to :meth:`ArtifactStore.read_validated`; an
``ArtifactValidationError`` from that decoder is the one recoverable artifact
failure.  Remote cache failures are best effort: the instance disables remote
access after the first operational failure while retaining usable L1 values.
"""

from __future__ import annotations

import hashlib
import os
import warnings
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, TypeVar, cast

import fsspec
from huggingface_hub import HfApi

from .paths import default_project_cache_dir, ensure_private_cache_dir


T = TypeVar("T")

DEFAULT_ARTIFACT_CACHE_SUBDIRECTORY = "artifacts-v1"
DEFAULT_CACHE_NAMESPACE = "ragtime2-2026-v1"


class ArtifactStoreError(Exception):
    """Base class for artifact-store failures."""


class ArtifactStoreConfigurationError(ArtifactStoreError, ValueError):
    """The store configuration or an object key is invalid."""


class ArtifactStoreUnavailableError(ArtifactStoreConfigurationError):
    """The optional runtime artifact-cache configuration is absent."""


class ArtifactStoreRemoteError(ArtifactStoreError):
    """A remote Hugging Face or local-cache operation failed."""


class ArtifactValidationError(ArtifactStoreError):
    """A decoder rejected the bytes as an invalid artifact."""


class ArtifactStoreLike(Protocol):
    """Small interface consumed by typed cache adapters."""

    def read(self, object_key: str) -> bytes | None: ...

    def read_validated(
        self, object_key: str, decoder: Callable[[bytes], T]
    ) -> T | None: ...

    def write(self, object_key: str, payload: bytes) -> bool: ...

    def evict_local(self, object_key: str) -> None: ...


ApiFactory = Callable[..., Any]
FilesystemFactory = Callable[..., Any]


def _configuration_error(message: str) -> ArtifactStoreConfigurationError:
    return ArtifactStoreConfigurationError(message)


def _legacy_cache_setting_name() -> str | None:
    """Return the first removed SQLite setting that is still populated."""

    for name in ("RAGTIME_CACHE_PATH", "RAGTIME_CACHE_DIR"):
        if os.getenv(name, "").strip():
            return name
    return None


class ArtifactStore:
    """Store identity-derived artifacts in a private HF Bucket and L1 cache.

    ``api_factory`` and ``filesystem_factory`` are intentionally injectable so
    offline tests can exercise the complete L1/L2 protocol without making a
    network request.  They are both lazy: a valid L1 read never constructs an
    HF client or filesystem.  The first remote operational failure disables
    remote access for this instance; valid L1 reads remain available.
    """

    def __init__(
        self,
        bucket_id: str,
        local_dir: str | os.PathLike[str],
        *,
        prefix: str = "",
        token: str | bool | None = None,
        namespace: str = DEFAULT_CACHE_NAMESPACE,
        endpoint: str | None = None,
        api_factory: ApiFactory | None = None,
        filesystem_factory: FilesystemFactory | None = None,
    ) -> None:
        self.bucket_id = self._validate_bucket_id(bucket_id)
        self.prefix = self._validate_prefix(prefix)
        self.local_dir = self._prepare_local_dir(local_dir)
        if not isinstance(namespace, str) or not namespace.strip():
            raise _configuration_error("RAGTIME_CACHE_NAMESPACE must not be empty")
        self.namespace = namespace.strip()
        self.token = token
        self.endpoint = endpoint
        self._api_factory = api_factory or HfApi
        self._filesystem_factory = filesystem_factory or fsspec.filesystem
        self._api: Any | None = None
        self._filesystem: Any | None = None
        self._private_verified = False
        self._remote_disabled = False
        self._remote_warning_emitted = False

    @classmethod
    def from_env(cls, **overrides: Any) -> "ArtifactStore":
        """Build a store from the locked environment configuration."""

        if _legacy_cache_setting_name() is not None:
            raise _configuration_error(
                "RAGTIME_CACHE_PATH and RAGTIME_CACHE_DIR are not supported by ArtifactStore; "
                "use RAGTIME_ARTIFACT_CACHE_DIR"
            )

        bucket_override = "bucket_id" in overrides
        bucket_id = overrides.pop(
            "bucket_id", os.getenv("RAGTIME_HF_BUCKET_ID", "").strip()
        )
        if not bucket_id:
            if bucket_override:
                raise _configuration_error("RAGTIME_HF_BUCKET_ID is required")
            raise ArtifactStoreUnavailableError(
                "RAGTIME_HF_BUCKET_ID is not configured"
            )

        prefix = overrides.pop("prefix", os.getenv("RAGTIME_ARTIFACT_PREFIX", ""))
        local_value = overrides.pop(
            "local_dir", os.getenv("RAGTIME_ARTIFACT_CACHE_DIR", "").strip()
        )
        if not local_value:
            local_value = default_project_cache_dir() / DEFAULT_ARTIFACT_CACHE_SUBDIRECTORY

        token = overrides.pop("token", os.getenv("HF_TOKEN") or None)
        namespace = overrides.pop(
            "namespace",
            os.getenv("RAGTIME_CACHE_NAMESPACE", DEFAULT_CACHE_NAMESPACE),
        )
        endpoint = overrides.pop("endpoint", None)
        return cls(
            bucket_id,
            local_value,
            prefix=prefix,
            token=token,
            namespace=namespace,
            endpoint=endpoint,
            **overrides,
        )

    @staticmethod
    def _validate_bucket_id(bucket_id: str) -> str:
        if not isinstance(bucket_id, str) or bucket_id != bucket_id.strip():
            raise _configuration_error("HF bucket ID must be an owner/bucket pair")
        parts = bucket_id.split("/")
        if len(parts) != 2 or any(
            not part or part in {".", ".."} or any(ch.isspace() for ch in part)
            for part in parts
        ):
            raise _configuration_error("HF bucket ID must be an owner/bucket pair")
        if any(ch in bucket_id for ch in "?#\\"):
            raise _configuration_error("HF bucket ID contains invalid path data")
        return bucket_id

    @staticmethod
    def _validate_prefix(prefix: str | None) -> str:
        if prefix is None or prefix == "":
            return ""
        if not isinstance(prefix, str) or prefix != prefix.strip():
            raise _configuration_error("artifact prefix must be a relative POSIX path")
        if prefix.startswith("/") or prefix.endswith("/"):
            raise _configuration_error("artifact prefix must be a relative POSIX path")
        if any(ch in prefix for ch in "?#\\"):
            raise _configuration_error("artifact prefix contains invalid path data")
        parts = prefix.split("/")
        if any(not part or part in {".", ".."} for part in parts):
            raise _configuration_error("artifact prefix must be a relative POSIX path")
        if any(any(ord(ch) < 32 for ch in part) for part in parts):
            raise _configuration_error("artifact prefix contains control characters")
        return prefix

    @staticmethod
    def _prepare_local_dir(local_dir: str | os.PathLike[str]) -> Path:
        path = Path(local_dir).expanduser()
        if not path.is_absolute():
            raise _configuration_error("artifact cache directory must be absolute")
        if path.is_symlink():
            raise _configuration_error("artifact cache directory must not be a symlink")
        try:
            ensure_private_cache_dir(path)
        except (OSError, ValueError):
            raise _configuration_error(
                "artifact cache directory must be a private mode-0700 directory"
            ) from None
        if path.is_symlink():
            raise _configuration_error("artifact cache directory must not be a symlink")
        return path

    @staticmethod
    def _validate_object_key(object_key: str) -> str:
        if not isinstance(object_key, str) or not object_key:
            raise _configuration_error("artifact object key must be a relative POSIX path")
        if object_key != object_key.strip():
            raise _configuration_error("artifact object key must be a relative POSIX path")
        if object_key.startswith("/") or object_key.endswith("/"):
            raise _configuration_error("artifact object key must be a relative POSIX path")
        if any(ch in object_key for ch in "?#\\"):
            raise _configuration_error("artifact object key contains invalid path data")
        parts = object_key.split("/")
        if any(not part or part in {".", ".."} for part in parts):
            raise _configuration_error("artifact object key must be a relative POSIX path")
        if any(any(ord(ch) < 32 for ch in part) for part in parts):
            raise _configuration_error("artifact object key contains control characters")
        return object_key

    def _full_object_key(self, object_key: str) -> str:
        return f"{self.prefix}/{object_key}" if self.prefix else object_key

    def _remote_path(self, object_key: str) -> str:
        return f"hf://buckets/{self.bucket_id}/{self._full_object_key(object_key)}"

    def local_candidate(self, object_key: str) -> Path:
        """Return the exact simplecache candidate for ``object_key``."""

        object_key = self._validate_object_key(object_key)
        # SimpleCacheFileSystem's default HashCacheMapper hashes the target
        # filesystem path after both simplecache:// and hf:// are stripped.
        # Keeping this calculation local is what permits an L1 hit to avoid
        # constructing HfFileSystem (which itself constructs HfApi).
        cache_path = self._remote_path(object_key).removeprefix("hf://")
        mapped = hashlib.sha256(cache_path.encode("utf-8")).hexdigest()
        return self.local_dir / mapped

    def _get_api(self) -> Any:
        if self._api is None:
            kwargs: dict[str, Any] = {"token": self.token}
            if self.endpoint is not None:
                kwargs["endpoint"] = self.endpoint
            try:
                self._api = self._api_factory(**kwargs)
            except Exception:
                raise ArtifactStoreRemoteError(
                    "failed to construct the Hugging Face API client"
                ) from None
        return self._api

    def _get_filesystem(self) -> Any:
        if self._filesystem is None:
            target_options: dict[str, Any] = {"token": self.token}
            if self.endpoint is not None:
                target_options["endpoint"] = self.endpoint
            try:
                self._filesystem = self._filesystem_factory(
                    "simplecache",
                    target_protocol="hf",
                    cache_storage=str(self.local_dir),
                    target_options=target_options,
                    skip_instance_cache=True,
                )
            except Exception:
                raise ArtifactStoreRemoteError(
                    "failed to construct the Hugging Face artifact filesystem"
                ) from None
        return self._filesystem

    def _verify_private_bucket(self) -> None:
        if self._private_verified:
            return
        try:
            info = self._get_api().bucket_info(self.bucket_id)
        except Exception:
            raise ArtifactStoreRemoteError(
                "failed to inspect the configured Hugging Face bucket"
            ) from None
        if getattr(info, "private", None) is not True:
            raise ArtifactStoreRemoteError(
                "configured Hugging Face bucket is not private"
            )
        self._private_verified = True

    def _disable_remote(self) -> None:
        """Disable remote operations and emit one credential-safe warning."""

        self._remote_disabled = True
        if not self._remote_warning_emitted:
            warnings.warn(
                "remote artifact cache access disabled after an operational failure",
                RuntimeWarning,
                stacklevel=3,
            )
            self._remote_warning_emitted = True

    def _read_local(self, candidate: Path) -> tuple[bool, bytes | None]:
        try:
            return True, candidate.read_bytes()
        except FileNotFoundError:
            return False, None

    def _best_effort_remove_candidate(self, candidate: Path) -> None:
        """Remove a failed remote candidate without masking the remote miss."""

        try:
            candidate.unlink(missing_ok=True)
        except OSError:
            return

    def _read_remote(self, object_key: str) -> bytes | None:
        if self._remote_disabled:
            return None
        try:
            self._verify_private_bucket()
        except Exception:
            self._disable_remote()
            return None

        candidate = self.local_candidate(object_key)
        try:
            with self._get_filesystem().open(self._remote_path(object_key), "rb") as handle:
                payload = handle.read()
        except FileNotFoundError:
            # Ordinary simplecache absence is a quiet miss.  In particular,
            # do not trip the breaker on the target's normal not-found path.
            self._best_effort_remove_candidate(candidate)
            return None
        except Exception:
            self._best_effort_remove_candidate(candidate)
            self._disable_remote()
            return None
        if not isinstance(payload, (bytes, bytearray)):
            self._best_effort_remove_candidate(candidate)
            self._disable_remote()
            return None
        return bytes(payload)

    def read(self, object_key: str) -> bytes | None:
        """Read raw bytes, using L1 before the private HF Bucket."""

        object_key = self._validate_object_key(object_key)
        candidate = self.local_candidate(object_key)
        try:
            found, payload = self._read_local(candidate)
        except (OSError, ArtifactStoreRemoteError):
            self._disable_remote()
            return None
        if found:
            return cast(bytes, payload)
        return self._read_remote(object_key)

    def evict_local(self, object_key: str) -> None:
        """Evict only one validated L1 candidate; never touch the bucket."""

        object_key = self._validate_object_key(object_key)
        candidate = self.local_candidate(object_key)
        self._best_effort_remove_candidate(candidate)

    def read_validated(
        self, object_key: str, decoder: Callable[[bytes], T]
    ) -> T | None:
        """Decode an artifact, repairing one invalid local candidate."""

        object_key = self._validate_object_key(object_key)
        candidate = self.local_candidate(object_key)
        try:
            found, payload = self._read_local(candidate)
        except (OSError, ArtifactStoreRemoteError):
            self._disable_remote()
            return None
        if found:
            try:
                return decoder(cast(bytes, payload))
            except ArtifactValidationError:
                self._best_effort_remove_candidate(candidate)
        payload = self._read_remote(object_key)
        if payload is None:
            return None
        try:
            return decoder(payload)
        except ArtifactValidationError:
            # The remote response is now also an L1 candidate.  Evict only
            # this exact object so a corrupt artifact becomes a miss without
            # disturbing neighboring cache entries.
            self._best_effort_remove_candidate(candidate)
            return None

    def write(self, object_key: str, payload: bytes) -> bool:
        """Upload bytes when remote access is healthy, returning success."""

        object_key = self._validate_object_key(object_key)
        if not isinstance(payload, bytes):
            raise _configuration_error("artifact payload must be bytes")
        candidate = self.local_candidate(object_key)
        candidate_existed_before = candidate.exists()
        if self._remote_disabled:
            return False
        remote_open_attempted = False
        try:
            self._verify_private_bucket()
            filesystem = self._get_filesystem()
            handle = filesystem.open(self._remote_path(object_key), "wb")
            remote_open_attempted = True
            with handle:
                handle.write(payload)
            if candidate.read_bytes() != payload:
                raise ArtifactStoreRemoteError(
                    "local artifact candidate did not retain the written bytes"
                )
        except Exception:
            if remote_open_attempted or not candidate_existed_before:
                self._best_effort_remove_candidate(candidate)
            self._disable_remote()
            return False
        return True

    def configuration(self) -> Mapping[str, object]:
        """Return storage metadata with credentials omitted."""

        return {
            "bucket_id": self.bucket_id,
            "prefix": self.prefix,
            "local_dir": str(self.local_dir),
            "namespace": self.namespace,
            "token_configured": bool(self.token),
        }


__all__ = [
    "ArtifactStore",
    "ArtifactStoreLike",
    "ArtifactStoreError",
    "ArtifactStoreConfigurationError",
    "ArtifactStoreUnavailableError",
    "ArtifactStoreRemoteError",
    "ArtifactValidationError",
]
