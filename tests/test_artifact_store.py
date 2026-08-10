from __future__ import annotations

import hashlib
import os
import stat
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from trec_ragtime.artifact_store import (
    ArtifactStore,
    ArtifactStoreConfigurationError,
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
    ) -> None:
        self.private = private
        self.bucket_error = bucket_error
        self.bucket_calls: list[str] = []

    def bucket_info(self, bucket_id: str):
        self.bucket_calls.append(bucket_id)
        if self.bucket_error is not None:
            raise self.bucket_error
        return type("BucketInfo", (), {"private": self.private})()


class ArtifactStoreTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.local_dir = Path(self.tempdir.name) / "artifacts"
        self.remote: dict[str, bytes] = {"owner/bucket/foo/object.mpk": b"remote"}
        self.api: FakeHfApi | None = None
        self.fs: FakeSimpleCache | None = None
        self.api_constructed = 0
        self.fs_constructed = 0

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
            self.api = FakeHfApi(private=private, bucket_error=api_error)
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

    def test_valid_l1_read_survives_a_disabled_remote(self) -> None:
        store = self.make_store(download_error=RuntimeError("token=secret-token"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertIsNone(store.read("object.mpk"))
            self.assertEqual(store.read("object.mpk"), None)
        self.assert_one_sanitized_warning(caught)
        self.assertEqual(self.fs_constructed, 1)
        candidate = store.local_candidate("local.mpk")
        candidate.write_bytes(b"local")
        os.chmod(candidate, 0o600)
        self.assertEqual(store.read("local.mpk"), b"local")
        self.assertEqual(self.fs_constructed, 1)

    def test_remote_absence_is_a_quiet_miss_and_does_not_disable_store(self) -> None:
        store = self.make_store()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertIsNone(store.read("missing.mpk"))
            self.assertIsNone(store.read("missing.mpk"))
        self.assertEqual(caught, [])
        self.assertEqual(self.fs_constructed, 1)
        assert self.fs is not None
        self.assertEqual(
            self.fs.open_calls,
            [
                ("hf://buckets/owner/bucket/foo/missing.mpk", "rb"),
                ("hf://buckets/owner/bucket/foo/missing.mpk", "rb"),
            ],
        )

    def test_remote_file_not_found_evicts_partial_l1_candidate(self) -> None:
        partial_fs: PartialMissingSimpleCache | None = None

        def factory(protocol: str, **kwargs):
            nonlocal partial_fs
            partial_fs = PartialMissingSimpleCache(self.local_dir, self.remote)
            return partial_fs

        store = self.make_store(filesystem_factory_override=factory)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertIsNone(store.read("object.mpk"))
        self.assertEqual(caught, [])
        self.assertFalse(store.local_candidate("object.mpk").exists())
        assert partial_fs is not None
        self.assertEqual(
            partial_fs.open_calls,
            [("hf://buckets/owner/bucket/foo/object.mpk", "rb")],
        )

    def test_remote_failures_warn_once_disable_remote_and_preserve_l1(self) -> None:
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
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        self.assertIsNone(store.read("object.mpk"))
                        self.assertIsNone(store.read("object.mpk"))
                    self.assert_one_sanitized_warning(caught)
                    self.assertEqual(self.fs_constructed, 1 if "download_error" in options else 0)
                    candidate = store.local_candidate("local.mpk")
                    candidate.write_bytes(b"local")
                    os.chmod(candidate, 0o600)
                    self.assertEqual(store.read("local.mpk"), b"local")
                    self.assertEqual(self.fs_constructed, 1 if "download_error" in options else 0)
                finally:
                    self.tearDown()

    def test_public_bucket_miss_warns_once_disables_remote_and_never_writes(self) -> None:
        store = self.make_store(private=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertIsNone(store.read("object.mpk"))
            self.assertFalse(store.write("new.mpk", b"payload"))
            self.assertIsNone(store.read("object.mpk"))
        self.assert_one_sanitized_warning(caught)
        self.assertEqual(self.api_constructed, 1)
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

    def test_failed_write_returns_false_cleans_exact_candidate_and_disables_remote(self) -> None:
        store = self.make_store(upload_error=RuntimeError("token=secret-token"))
        other = store.local_candidate("other.mpk")
        other.write_bytes(b"keep")
        os.chmod(other, 0o600)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertFalse(store.write("object.mpk", b"payload"))
            self.assertFalse(store.write("second.mpk", b"payload"))
        self.assert_one_sanitized_warning(caught)
        self.assertFalse(store.local_candidate("object.mpk").exists())
        self.assertFalse(store.local_candidate("second.mpk").exists())
        self.assertEqual(other.read_bytes(), b"keep")
        self.assertEqual(self.fs_constructed, 1)
        assert self.fs is not None
        self.assertEqual(
            self.fs.open_calls,
            [("hf://buckets/owner/bucket/foo/object.mpk", "wb")],
        )

    def test_failed_write_before_filesystem_open_preserves_existing_l1(self) -> None:
        store = self.make_store(private=False)
        candidate = store.local_candidate("object.mpk")
        candidate.write_bytes(b"keep")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertFalse(store.write("object.mpk", b"replacement"))
        self.assert_one_sanitized_warning(caught)
        self.assertEqual(candidate.read_bytes(), b"keep")
        self.assertEqual(self.fs_constructed, 0)

    def test_failed_filesystem_construction_preserves_existing_l1(self) -> None:
        def factory(protocol: str, **kwargs):
            raise OSError("filesystem unavailable")

        store = self.make_store(filesystem_factory_override=factory)
        candidate = store.local_candidate("object.mpk")
        candidate.write_bytes(b"keep")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertFalse(store.write("object.mpk", b"replacement"))
        self.assert_one_sanitized_warning(caught)
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
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertFalse(store.write("object.mpk", b"payload"))
        self.assert_one_sanitized_warning(caught)
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
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertFalse(store.write("object.mpk", b"payload"))
        self.assert_one_sanitized_warning(caught)
        self.assertFalse(store.local_candidate("object.mpk").exists())

    def test_evict_local_removes_only_exact_candidate(self) -> None:
        store = self.make_store()
        store.local_candidate("object.mpk").write_bytes(b"local")
        store.local_candidate("other.mpk").write_bytes(b"other")

        store.evict_local("object.mpk")

        self.assertFalse(store.local_candidate("object.mpk").exists())
        self.assertEqual(store.local_candidate("other.mpk").read_bytes(), b"other")

    def test_read_validated_only_repairs_artifact_validation_errors(self) -> None:
        store = self.make_store()
        store.local_candidate("object.mpk").write_bytes(b"local")

        def permission_failure(payload: bytes) -> bytes:
            raise PermissionError("permission denied")

        with self.assertRaises(PermissionError):
            store.read_validated("object.mpk", permission_failure)
        self.assertEqual(self.api_constructed, 0)

    def test_local_l1_read_failure_is_a_warned_miss_and_disables_remote(self) -> None:
        store = self.make_store()
        candidate = store.local_candidate("object.mpk")
        candidate.write_bytes(b"local")

        with patch.object(Path, "read_bytes", side_effect=OSError("permission denied")):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.assertIsNone(store.read("object.mpk"))
                self.assertIsNone(store.read("object.mpk"))
        self.assert_one_sanitized_warning(caught)
        self.assertEqual(self.api_constructed, 0)
        self.assertEqual(self.fs_constructed, 0)

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

    def test_configuration_metadata_never_exposes_token(self) -> None:
        store = self.make_store(token="super-secret")
        self.assertNotIn("super-secret", repr(store.configuration()))
        self.assertTrue(store.configuration()["token_configured"])


if __name__ == "__main__":
    unittest.main()
