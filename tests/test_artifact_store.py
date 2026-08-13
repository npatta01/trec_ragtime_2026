from __future__ import annotations

import hashlib
import os
import stat
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import fsspec
import httpx

from trec_ragtime import artifact_store
from huggingface_hub.errors import HfHubHTTPError

from trec_ragtime.artifact_store import (
    ArtifactStore,
    ArtifactStoreConfigurationError,
    ArtifactStoreRemoteError,
    ArtifactStoreUnavailableError,
    ArtifactValidationError,
)


class FakeRemoteFile:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return self.payload

    def __enter__(self) -> "FakeRemoteFile":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None


class FakeWriteFile:
    def __init__(self, fs: "FakeSimpleCache", candidate: Path, key: str) -> None:
        self.fs = fs
        self.candidate = candidate
        self.key = key
        self.handle = candidate.open("wb")

    def write(self, payload: bytes) -> int:
        return self.handle.write(payload)

    def __enter__(self) -> "FakeWriteFile":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.handle.close()
        if exc_type is None:
            if self.fs.fail_upload is not None:
                raise self.fs.fail_upload
            self.fs.remote[self.key] = self.candidate.read_bytes()


class FakeSimpleCache:
    def __init__(
        self,
        storage: Path,
        remote: dict[str, bytes],
        *,
        fail_download: BaseException | None = None,
        fail_upload: BaseException | None = None,
    ) -> None:
        self.storage = storage
        self.remote = remote
        self.fail_download = fail_download
        self.fail_upload = fail_upload
        self.open_calls: list[tuple[str, str]] = []
        self.info_calls: list[str] = []

    def info(self, path: str) -> dict[str, object]:
        """Answer presence the way the real filesystem does.

        The store asks here rather than at open() because open() cannot tell a
        throttle from an absence. A transport failure surfaces here first, so
        fail_download raises from info() too.
        """

        self.info_calls.append(path)
        if self.fail_download is not None:
            raise self.fail_download
        key = self._strip_protocol(path).removeprefix("buckets/")
        if key not in self.remote:
            raise FileNotFoundError(key)
        return {"name": key, "size": len(self.remote[key]), "type": "file"}

    @staticmethod
    def _strip_protocol(path: str) -> str:
        return path.removeprefix("hf://")

    @staticmethod
    def _mapper(path: str) -> str:
        return hashlib.sha256(path.encode("utf-8")).hexdigest()

    def _candidate(self, path: str) -> Path:
        candidate = self.storage / self._mapper(self._strip_protocol(path))
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate

    def open(self, path: str, mode: str):
        self.open_calls.append((path, mode))
        key = self._strip_protocol(path).removeprefix("buckets/")
        candidate = self._candidate(path)
        if "r" in mode:
            if self.fail_download is not None:
                raise self.fail_download
            if key not in self.remote:
                raise FileNotFoundError(key)
            candidate.write_bytes(self.remote[key])
            os.chmod(candidate, 0o600)
            return FakeRemoteFile(self.remote[key])
        return FakeWriteFile(self, candidate, key)


class PartialMissingSimpleCache(FakeSimpleCache):
    def open(self, path: str, mode: str):
        if "r" in mode:
            self.open_calls.append((path, mode))
            self._candidate(path).write_bytes(b"partial")
            raise FileNotFoundError("remote disappeared")
        return super().open(path, mode)


class EvictionRequiredSimpleCache(FakeSimpleCache):
    """Return stale L1 bytes unless the caller evicts before L2 hydration."""

    def open(self, path: str, mode: str):
        if "r" not in mode:
            return super().open(path, mode)
        self.open_calls.append((path, mode))
        candidate = self._candidate(path)
        if candidate.exists():
            return FakeRemoteFile(b"stale")
        key = self._strip_protocol(path).removeprefix("buckets/")
        if key not in self.remote:
            raise FileNotFoundError(key)
        candidate.write_bytes(self.remote[key])
        return FakeRemoteFile(self.remote[key])


class TamperingWriteFile(FakeWriteFile):
    """A remote write that leaves a complete but mismatching L1 candidate."""

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        super().__exit__(exc_type, exc_value, traceback)
        if exc_type is None:
            self.candidate.write_bytes(b"tampered")


class TamperingSimpleCache(FakeSimpleCache):
    def open(self, path: str, mode: str):
        if "w" not in mode:
            return super().open(path, mode)
        self.open_calls.append((path, mode))
        key = self._strip_protocol(path).removeprefix("buckets/")
        return TamperingWriteFile(self, self._candidate(path), key)


class PartialOpenFailureSimpleCache(FakeSimpleCache):
    def open(self, path: str, mode: str):
        if "w" in mode:
            self.open_calls.append((path, mode))
            self._candidate(path).write_bytes(b"partial")
            raise OSError("open failed after creating candidate")
        return super().open(path, mode)


class FakeHfApi:
    def __init__(
        self,
        *,
        private: bool = True,
        bucket_error: BaseException | None = None,
        bucket_paths: tuple[object, ...] = (),
    ) -> None:
        self.private = private
        self.bucket_error = bucket_error
        self.bucket_paths = bucket_paths
        self.bucket_calls: list[str] = []
        self.tree_calls: list[tuple[str, str | None, bool | None, object]] = []

    def bucket_info(self, bucket_id: str):
        self.bucket_calls.append(bucket_id)
        if self.bucket_error is not None:
            raise self.bucket_error
        return type("BucketInfo", (), {"private": self.private})()

    def list_bucket_tree(
        self,
        bucket_id: str,
        prefix: str | None = None,
        *,
        recursive: bool | None = None,
        token: object = None,
    ):
        self.tree_calls.append((bucket_id, prefix, recursive, token))
        return iter(self.bucket_paths)


class ArtifactStoreTestCase(unittest.TestCase):
    def setUp(self) -> None:
        # The store retries transient remote failures; sleeping through that in
        # every failure-path test took the suite from 19s to over three minutes.
        patcher = patch.object(artifact_store, "RETRY_BASE_SECONDS", 0.0)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tempdir = tempfile.TemporaryDirectory()
        self.local_dir = Path(self.tempdir.name) / "artifacts"
        self.remote: dict[str, bytes] = {"owner/bucket/foo/object.mpk": b"remote"}
        self.api: FakeHfApi | None = None
        self.fs: FakeSimpleCache | None = None
        self.api_constructed = 0
        self.fs_constructed = 0
        self.bucket_paths: tuple[object, ...] = ()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def make_store(
        self,
        *,
        prefix: str = "foo",
        token: str | bool | None = "secret-token",
        private: bool = True,
        api_error: BaseException | None = None,
        download_error: BaseException | None = None,
        upload_error: BaseException | None = None,
        filesystem_factory_override=None,
    ) -> ArtifactStore:
        def api_factory(**kwargs):
            self.api_constructed += 1
            self.api = FakeHfApi(
                private=private,
                bucket_error=api_error,
                bucket_paths=self.bucket_paths,
            )
            return self.api

        def fs_factory(protocol: str, **kwargs):
            self.fs_constructed += 1
            self.assertEqual(protocol, "simplecache")
            self.assertEqual(kwargs["target_protocol"], "hf")
            self.assertEqual(kwargs["target_options"]["token"], token)
            self.fs = FakeSimpleCache(
                self.local_dir,
                self.remote,
                fail_download=download_error,
                fail_upload=upload_error,
            )
            return self.fs

        return ArtifactStore(
            bucket_id="owner/bucket",
            prefix=prefix,
            local_dir=self.local_dir,
            token=token,
            api_factory=api_factory,
            filesystem_factory=filesystem_factory_override or fs_factory,
        )

    @staticmethod
    def decoder(payload: bytes) -> bytes:
        if payload == b"bad":
            raise ArtifactValidationError("invalid artifact")
        return payload

    def assert_one_sanitized_warning(self, caught: list[warnings.WarningMessage]) -> None:
        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, RuntimeWarning)
        self.assertNotIn("secret-token", str(caught[0].message))

    def test_valid_l1_hit_never_constructs_hf_filesystem(self) -> None:
        store = self.make_store()
        candidate = store.local_candidate("object.mpk")
        candidate.write_bytes(b"local")
        os.chmod(candidate, 0o600)

        self.assertEqual(store.read("object.mpk"), b"local")
        self.assertEqual(self.api_constructed, 0)
        self.assertEqual(self.fs_constructed, 0)

    def test_lists_only_file_keys_beneath_a_validated_durable_prefix(self) -> None:
        self.bucket_paths = (
            SimpleNamespace(type="file", path="foo/retrieval-handoff/v1/fp/b.mpk"),
            SimpleNamespace(type="directory", path="foo/retrieval-handoff/v1/fp/sub"),
            SimpleNamespace(type="file", path="foo/retrieval-handoff/v1/fp/a.mpk"),
            SimpleNamespace(type="file", path="other/unrelated.mpk"),
        )
        store = self.make_store()

        self.assertEqual(
            store.list_keys("retrieval-handoff/v1/fp"),
            (
                "retrieval-handoff/v1/fp/a.mpk",
                "retrieval-handoff/v1/fp/b.mpk",
            ),
        )
        assert self.api is not None
        self.assertEqual(
            self.api.tree_calls,
            [("owner/bucket", "foo/retrieval-handoff/v1/fp", True, "secret-token")],
        )
        self.assertEqual(self.fs_constructed, 0)

    def test_local_candidate_matches_real_simplecache_mapper_without_hf_io(self) -> None:
        with tempfile.TemporaryDirectory() as raw_cache_dir:
            cache_dir = Path(raw_cache_dir) / "artifacts"
            store = ArtifactStore(
                bucket_id="owner/bucket",
                local_dir=cache_dir,
                token=None,
            )
            sc = fsspec.filesystem(
                "simplecache",
                target_protocol="hf",
                cache_storage=str(cache_dir),
                target_options={"token": None},
                skip_instance_cache=True,
            )
            key = "object.mpk"

            self.assertEqual(
                store.local_candidate(key),
                Path(cache_dir)
                / sc._mapper(sc._strip_protocol(store._remote_path(key))),
            )

    def test_a_valid_l1_read_still_works_after_a_remote_failure(self) -> None:
        """An unreachable remote must not cost a usable local value."""

        store = self.make_store(download_error=RuntimeError("token=secret-token"))
        with self.assertRaises(ArtifactStoreRemoteError):
            store.read("object.mpk")
        candidate = store.local_candidate("local.mpk")
        candidate.write_bytes(b"local")
        os.chmod(candidate, 0o600)
        self.assertEqual(store.read("local.mpk"), b"local")

    def test_remote_absence_is_a_quiet_miss_and_does_not_disable_store(self) -> None:
        """Only a confirmed absence may be reported as None.

        Presence is settled by info(); an absent object is never opened, so the
        miss costs one call rather than a download attempt.
        """

        store = self.make_store()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertIsNone(store.read("missing.mpk"))
            self.assertIsNone(store.read("missing.mpk"))
        self.assertEqual(caught, [])
        self.assertEqual(self.fs_constructed, 1)
        assert self.fs is not None
        self.assertEqual(
            self.fs.info_calls,
            [
                "hf://buckets/owner/bucket/foo/missing.mpk",
                "hf://buckets/owner/bucket/foo/missing.mpk",
            ],
        )
        self.assertEqual(self.fs.open_calls, [])

    def test_an_object_that_vanishes_between_probe_and_open_is_not_a_miss(self) -> None:
        """info() said present, open() said gone: a race, not an absence.

        Reporting None here would let a caller record real work as missing, so
        it raises and still clears the partial candidate.
        """

        partial_fs: PartialMissingSimpleCache | None = None

        def factory(protocol: str, **kwargs):
            nonlocal partial_fs
            partial_fs = PartialMissingSimpleCache(self.local_dir, self.remote)
            return partial_fs

        store = self.make_store(filesystem_factory_override=factory)
        with self.assertRaises(ArtifactStoreRemoteError):
            store.read("object.mpk")
        self.assertFalse(store.local_candidate("object.mpk").exists())
        assert partial_fs is not None
        self.assertEqual(
            partial_fs.open_calls,
            [("hf://buckets/owner/bucket/foo/object.mpk", "rb")],
        )

    def test_remote_failures_raise_and_preserve_l1(self) -> None:
        """Every one of these once returned None, i.e. "not there".

        A throttle, an auth failure and a misconfigured bucket are all reasons
        the answer is unknown. Returning None conflated them with a confirmed
        absence, which is how a rate-limited run recorded 1596 unscored queries
        as missing and exited successfully.
        """

        failures = (
            {"api_error": RuntimeError("Authorization: Bearer secret-token")},
            {"private": False},
            {"download_error": OSError("server unavailable")},
        )
        self.tempdir.cleanup()
        for options in failures:
            with self.subTest(options=options):
                self.setUp()
                try:
                    store = self.make_store(**options)
                    for _ in range(2):
                        with self.assertRaises(ArtifactStoreRemoteError):
                            store.read("object.mpk")
                    candidate = store.local_candidate("local.mpk")
                    candidate.write_bytes(b"local")
                    os.chmod(candidate, 0o600)
                    self.assertEqual(store.read("local.mpk"), b"local")
                finally:
                    self.tearDown()

    def test_a_transient_failure_does_not_latch_the_store(self) -> None:
        """One throttle must not poison the process.

        Latching here would make a single 429 turn every later read into a
        failure, so a long run could never recover by retrying.
        """

        store = self.make_store(download_error=OSError("server unavailable"))
        with self.assertRaises(ArtifactStoreRemoteError):
            store.read("object.mpk")
        assert self.fs is not None
        self.fs.fail_download = None
        self.assertEqual(store.read("object.mpk"), b"remote")

    def test_a_public_bucket_is_refused_and_never_written_to(self) -> None:
        store = self.make_store(private=False)
        with self.assertRaises(ArtifactStoreRemoteError):
            store.read("object.mpk")
        with self.assertRaises(ArtifactStoreRemoteError):
            store.write("new.mpk", b"payload")
        self.assertEqual(self.fs_constructed, 0)
        self.assertNotIn("owner/bucket/foo/new.mpk", self.remote)

    def test_corrupt_l1_is_evicted_and_repaired_from_l2(self) -> None:
        store = self.make_store()
        candidate = store.local_candidate("object.mpk")
        candidate.write_bytes(b"bad")
        os.chmod(candidate, 0o600)

        self.assertEqual(store.read_validated("object.mpk", self.decoder), b"remote")
        self.assertEqual(candidate.read_bytes(), b"remote")
        assert self.fs is not None
        self.assertEqual(len(self.fs.open_calls), 1)

    def test_corrupt_l2_is_evicted_and_returned_as_a_miss(self) -> None:
        store = self.make_store()
        self.remote["owner/bucket/foo/object.mpk"] = b"bad"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertIsNone(store.read_validated("object.mpk", self.decoder))
        self.assertEqual(caught, [])
        self.assertFalse(store.local_candidate("object.mpk").exists())

    def test_corrupt_l2_strict_read_raises_validation_error(self) -> None:
        store = self.make_store()
        self.remote["owner/bucket/foo/object.mpk"] = b"bad"

        with self.assertRaises(ArtifactValidationError):
            store.read_validated_strict("object.mpk", self.decoder)
        self.assertFalse(store.local_candidate("object.mpk").exists())

    def test_a_failed_write_raises_and_cleans_its_exact_candidate(self) -> None:
        store = self.make_store(upload_error=RuntimeError("token=secret-token"))
        other = store.local_candidate("other.mpk")
        other.write_bytes(b"keep")
        os.chmod(other, 0o600)

        with self.assertRaises(ArtifactStoreRemoteError):
            store.write("object.mpk", b"payload")
        with self.assertRaises(ArtifactStoreRemoteError):
            store.write("second.mpk", b"payload")
        self.assertFalse(store.local_candidate("object.mpk").exists())
        self.assertFalse(store.local_candidate("second.mpk").exists())
        self.assertEqual(other.read_bytes(), b"keep")
        self.assertEqual(self.fs_constructed, 1)
        assert self.fs is not None
        # Both writes are attempted: a failed upload no longer latches the
        # store, so the second call is not silently skipped.
        self.assertEqual(
            self.fs.open_calls,
            [
                ("hf://buckets/owner/bucket/foo/object.mpk", "wb"),
                ("hf://buckets/owner/bucket/foo/second.mpk", "wb"),
            ],
        )

    def test_failed_write_before_filesystem_open_preserves_existing_l1(self) -> None:
        store = self.make_store(private=False)
        candidate = store.local_candidate("object.mpk")
        candidate.write_bytes(b"keep")

        with self.assertRaises(ArtifactStoreRemoteError):
            store.write("object.mpk", b"replacement")
        self.assertEqual(candidate.read_bytes(), b"keep")
        self.assertEqual(self.fs_constructed, 0)

    def test_failed_filesystem_construction_preserves_existing_l1(self) -> None:
        def factory(protocol: str, **kwargs):
            raise OSError("filesystem unavailable")

        store = self.make_store(filesystem_factory_override=factory)
        candidate = store.local_candidate("object.mpk")
        candidate.write_bytes(b"keep")

        with self.assertRaises(ArtifactStoreRemoteError):
            store.write("object.mpk", b"replacement")
        self.assertEqual(candidate.read_bytes(), b"keep")

    def test_successful_write_returns_true_and_retains_complete_l1_candidate(self) -> None:
        store = self.make_store()

        self.assertTrue(store.write("object.mpk", b"new-payload"))

        candidate = store.local_candidate("object.mpk")
        self.assertEqual(candidate.read_bytes(), b"new-payload")
        self.assertEqual(self.remote["owner/bucket/foo/object.mpk"], b"new-payload")

    def test_write_verifies_exact_l1_bytes_after_remote_close(self) -> None:
        tampering_fs: TamperingSimpleCache | None = None

        def factory(protocol: str, **kwargs):
            nonlocal tampering_fs
            tampering_fs = TamperingSimpleCache(self.local_dir, self.remote)
            return tampering_fs

        store = self.make_store(filesystem_factory_override=factory)
        with self.assertRaises(ArtifactStoreRemoteError):
            store.write("object.mpk", b"payload")
        self.assertFalse(store.local_candidate("object.mpk").exists())
        assert tampering_fs is not None
        self.assertEqual(
            tampering_fs.open_calls,
            [("hf://buckets/owner/bucket/foo/object.mpk", "wb")],
        )

    def test_write_cleans_partial_candidate_when_remote_open_fails(self) -> None:
        partial_fs: PartialOpenFailureSimpleCache | None = None

        def factory(protocol: str, **kwargs):
            nonlocal partial_fs
            partial_fs = PartialOpenFailureSimpleCache(self.local_dir, self.remote)
            return partial_fs

        store = self.make_store(filesystem_factory_override=factory)
        with self.assertRaises(ArtifactStoreRemoteError):
            store.write("object.mpk", b"payload")
        self.assertFalse(store.local_candidate("object.mpk").exists())

    def test_evict_local_removes_only_exact_candidate(self) -> None:
        store = self.make_store()
        store.local_candidate("object.mpk").write_bytes(b"local")
        store.local_candidate("other.mpk").write_bytes(b"other")

        store.evict_local("object.mpk")

        self.assertFalse(store.local_candidate("object.mpk").exists())
        self.assertEqual(store.local_candidate("other.mpk").read_bytes(), b"other")

    def test_evict_local_raises_a_sanitized_error_when_unlink_fails(self) -> None:
        store = self.make_store()
        candidate = store.local_candidate("object.mpk")
        candidate.write_bytes(b"local")

        with patch.object(
            Path, "unlink", side_effect=OSError("token=secret-token")
        ):
            with self.assertRaisesRegex(
                ArtifactStoreRemoteError, "could not evict local candidate"
            ) as raised:
                store.evict_local("object.mpk")

        self.assertNotIn("secret-token", str(raised.exception))
        self.assertTrue(candidate.exists())

    def test_read_validated_only_repairs_artifact_validation_errors(self) -> None:
        store = self.make_store()
        store.local_candidate("object.mpk").write_bytes(b"local")

        def permission_failure(payload: bytes) -> bytes:
            raise PermissionError("permission denied")

        with self.assertRaises(PermissionError):
            store.read_validated("object.mpk", permission_failure)
        self.assertEqual(self.api_constructed, 0)

    def test_local_l1_read_failure_falls_through_to_l2(self) -> None:
        store = self.make_store()
        candidate = store.local_candidate("object.mpk")
        candidate.write_bytes(b"stale")

        with patch.object(Path, "read_bytes", side_effect=OSError("permission denied")):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.assertEqual(store.read("object.mpk"), b"remote")
        self.assertEqual(caught, [])
        self.assertEqual(self.fs_constructed, 1)
        self.assertEqual(store.read("object.mpk"), b"remote")
        assert self.fs is not None
        self.assertEqual(
            self.fs.open_calls,
            [("hf://buckets/owner/bucket/foo/object.mpk", "rb")],
        )

    def test_local_l1_read_repair_requires_exact_eviction(self) -> None:
        l2_fs: EvictionRequiredSimpleCache | None = None

        def factory(protocol: str, **kwargs):
            nonlocal l2_fs
            l2_fs = EvictionRequiredSimpleCache(self.local_dir, self.remote)
            return l2_fs

        store = self.make_store(filesystem_factory_override=factory)
        store.local_candidate("object.mpk").write_bytes(b"stale")

        with patch.object(Path, "read_bytes", side_effect=OSError("permission denied")):
            self.assertEqual(store.read("object.mpk"), b"remote")
        assert l2_fs is not None
        self.assertEqual(
            l2_fs.open_calls,
            [("hf://buckets/owner/bucket/foo/object.mpk", "rb")],
        )

    def test_local_l1_eviction_failure_is_fatal_and_disables_l2(self) -> None:
        store = self.make_store()
        store.local_candidate("object.mpk").write_bytes(b"stale")

        with patch.object(
            Path, "read_bytes", side_effect=OSError("permission denied")
        ), patch.object(Path, "unlink", side_effect=OSError("permission denied")):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with self.assertRaises(ArtifactStoreRemoteError):
                    store.read("object.mpk")
                with self.assertRaises(ArtifactStoreRemoteError):
                    store.write("other.mpk", b"payload")
        self.assert_one_sanitized_warning(caught)
        self.assertEqual(self.api_constructed, 0)
        self.assertEqual(self.fs_constructed, 0)

    def test_strict_validated_read_never_turns_l1_eviction_failure_into_miss(
        self,
    ) -> None:
        store = self.make_store()
        store.local_candidate("object.mpk").write_bytes(b"stale")

        with patch.object(
            Path, "read_bytes", side_effect=OSError("permission denied")
        ), patch.object(Path, "unlink", side_effect=OSError("permission denied")):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with self.assertRaises(ArtifactStoreRemoteError):
                    store.read_validated_strict("object.mpk", self.decoder)
        self.assert_one_sanitized_warning(caught)
        self.assertEqual(self.api_constructed, 0)
        self.assertEqual(self.fs_constructed, 0)

    def test_local_l1_validated_read_failure_falls_through_to_l2(self) -> None:
        store = self.make_store()
        candidate = store.local_candidate("object.mpk")
        candidate.write_bytes(b"stale")

        with patch.object(Path, "read_bytes", side_effect=OSError("permission denied")):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.assertEqual(
                    store.read_validated("object.mpk", self.decoder), b"remote"
                )
        self.assertEqual(caught, [])
        self.assertEqual(self.fs_constructed, 1)
        self.assertEqual(
            store.read_validated("object.mpk", self.decoder), b"remote"
        )
        assert self.fs is not None
        self.assertEqual(
            self.fs.open_calls,
            [("hf://buckets/owner/bucket/foo/object.mpk", "rb")],
        )

    def test_an_unreadable_l1_and_an_unreachable_l2_raise_rather_than_miss(
        self,
    ) -> None:
        """Neither layer could answer, so neither may claim the object is absent."""

        store = self.make_store(download_error=OSError("server unavailable"))
        candidate = store.local_candidate("object.mpk")
        candidate.write_bytes(b"local")

        with patch.object(Path, "read_bytes", side_effect=OSError("permission denied")):
            for _ in range(2):
                with self.assertRaises(ArtifactStoreRemoteError):
                    store.read("object.mpk")
        self.assertEqual(self.api_constructed, 1)

    def test_an_unreadable_l1_and_an_unreachable_l2_raise_on_validated_reads_too(
        self,
    ) -> None:
        store = self.make_store(download_error=OSError("server unavailable"))
        store.local_candidate("object.mpk").write_bytes(b"local")

        with patch.object(Path, "read_bytes", side_effect=OSError("permission denied")):
            for _ in range(2):
                with self.assertRaises(ArtifactStoreRemoteError):
                    store.read_validated("object.mpk", self.decoder)
        self.assertEqual(self.api_constructed, 1)

    def test_bucket_directory_is_private_and_symlink_is_rejected(self) -> None:
        store = self.make_store()
        self.assertEqual(stat.S_IMODE(store.local_dir.stat().st_mode), 0o700)
        linked = Path(self.tempdir.name) / "linked"
        linked.symlink_to(self.local_dir, target_is_directory=True)
        with self.assertRaises(ArtifactStoreConfigurationError):
            ArtifactStore(
                bucket_id="owner/bucket",
                local_dir=linked,
                api_factory=lambda **kwargs: None,
                filesystem_factory=lambda *args, **kwargs: None,
            )

    def test_configuration_and_keys_remain_strictly_validated(self) -> None:
        with patch.dict(os.environ, {"RAGTIME_HF_BUCKET_ID": "wrong"}, clear=False):
            with self.assertRaises(ArtifactStoreConfigurationError):
                ArtifactStore.from_env()
        store = self.make_store()
        for key in ("../bad", "/absolute", "a//b", "a/./b", "a/../b", "a?x=1", ""):
            with self.subTest(key=key):
                with self.assertRaises(ArtifactStoreConfigurationError):
                    store.read(key)

        with self.assertRaises(ArtifactStoreConfigurationError):
            self.make_store(prefix="/invalid")
        with self.assertRaises(ArtifactStoreConfigurationError):
            ArtifactStore(
                bucket_id="owner/bucket",
                local_dir=self.local_dir,
                namespace="",
                api_factory=lambda **kwargs: None,
                filesystem_factory=lambda *args, **kwargs: None,
            )
        with self.assertRaises(ArtifactStoreConfigurationError):
            ArtifactStore(
                bucket_id="owner/bucket",
                local_dir="relative-cache",
                api_factory=lambda **kwargs: None,
                filesystem_factory=lambda *args, **kwargs: None,
            )
        with self.assertRaises(ArtifactStoreConfigurationError):
            store.write("payload.mpk", bytearray(b"not-bytes"))
        for name in ("RAGTIME_CACHE_PATH", "RAGTIME_CACHE_DIR"):
            with patch.dict(os.environ, {name: "/tmp/legacy-cache"}, clear=False):
                with self.assertRaises(ArtifactStoreConfigurationError):
                    ArtifactStore.from_env()

    def test_missing_bucket_still_validates_explicit_cache_settings(self) -> None:
        settings = (
            {"namespace": ""},
            {"prefix": "/invalid"},
            {"local_dir": "relative-cache"},
        )
        with patch.dict(os.environ, {"RAGTIME_HF_BUCKET_ID": ""}, clear=False):
            for overrides in settings:
                with self.subTest(overrides=overrides):
                    with self.assertRaises(ArtifactStoreConfigurationError) as raised:
                        ArtifactStore.from_env(**overrides)
                    self.assertNotIsInstance(
                        raised.exception, ArtifactStoreUnavailableError
                    )

            with self.assertRaises(ArtifactStoreUnavailableError):
                ArtifactStore.from_env()

    def test_configuration_metadata_never_exposes_token(self) -> None:
        store = self.make_store(token="super-secret")
        self.assertNotIn("super-secret", repr(store.configuration()))
        self.assertTrue(store.configuration()["token_configured"])

    def test_write_failure_when_l1_probe_raises_cleans_candidate(self) -> None:
        store = self.make_store()
        real_exists = Path.exists
        calls = 0

        def exists() -> bool:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("candidate status unavailable")
            return real_exists(store.local_candidate("object.mpk"))

        with patch.object(Path, "exists", side_effect=exists):
            with self.assertRaises(ArtifactStoreRemoteError):
                store.write("object.mpk", b"payload")
        self.assertFalse(store.local_candidate("object.mpk").exists())


    # --- storage contract: one case per way the remote can answer, or fail to.
    # The rule is narrow: read() may return None only for a confirmed 404.
    # Every other outcome is "unknown", and reporting unknown as absent is what
    # let a throttled run log 1596 unscored queries as MISSING and exit 0.

    @staticmethod
    def _throttled() -> BaseException:
        response = httpx.Response(
            429, request=httpx.Request("GET", "https://example.invalid/x")
        )
        return HfHubHTTPError("rate limited", response=response)

    def _store_where_info_raises(self, error: BaseException) -> ArtifactStore:
        store = self.make_store()

        class Classifying(FakeSimpleCache):
            def info(self, path: str) -> dict[str, object]:
                raise error

        def factory(protocol: str, **kwargs):
            self.fs_constructed += 1
            self.fs = Classifying(self.local_dir, self.remote)
            return self.fs

        store._filesystem_factory = factory
        store._filesystem = None
        return store

    def test_a_confirmed_404_is_the_only_miss(self) -> None:
        store = self._store_where_info_raises(FileNotFoundError("absent"))
        self.assertIsNone(store.read("object.mpk"))

    def test_a_429_wearing_a_file_not_found_is_not_a_miss(self) -> None:
        """The exact shape that caused the incident.

        fsspec's simplecache decides absence with exists(), HfFileSystem.exists
        swallows OSError, and HfHubHTTPError is an OSError — so a rate limit
        arrives looking like a missing object.
        """

        disguised = FileNotFoundError("absent")
        disguised.__cause__ = self._throttled()
        store = self._store_where_info_raises(disguised)
        with self.assertRaises(ArtifactStoreRemoteError):
            store.read("object.mpk")

    def test_a_bare_429_is_not_a_miss(self) -> None:
        store = self._store_where_info_raises(self._throttled())
        with self.assertRaises(ArtifactStoreRemoteError):
            store.read("object.mpk")

    def test_a_timeout_is_not_a_miss(self) -> None:
        store = self._store_where_info_raises(
            httpx.ConnectTimeout("connect timed out")
        )
        with self.assertRaises(ArtifactStoreRemoteError):
            store.read("object.mpk")

    def test_a_disabled_remote_reports_itself_rather_than_a_miss(self) -> None:
        store = self.make_store()
        store._remote_disabled = True
        with self.assertRaises(ArtifactStoreRemoteError):
            store.read("object.mpk")
        with self.assertRaises(ArtifactStoreRemoteError):
            store.write("object.mpk", b"payload")

    def test_a_write_that_cannot_be_confirmed_never_reports_success(self) -> None:
        store = self.make_store(upload_error=OSError("server unavailable"))
        with self.assertRaises(ArtifactStoreRemoteError):
            store.write("fresh.mpk", b"payload")
        self.assertNotIn("owner/bucket/foo/fresh.mpk", self.remote)

    def test_a_written_object_survives_eviction_and_rereads(self) -> None:
        """The independent verification the runner performs after every write."""

        store = self.make_store()
        self.assertTrue(store.write("fresh.mpk", b"payload"))
        store.evict_local("fresh.mpk")
        self.assertFalse(store.local_candidate("fresh.mpk").exists())
        self.assertEqual(store.read("fresh.mpk"), b"payload")


if __name__ == "__main__":
    unittest.main()
