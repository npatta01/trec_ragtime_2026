from __future__ import annotations

import hashlib
import os
import stat
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import fsspec
from fsspec.spec import AbstractFileSystem

from trec_ragtime.artifact_store import (
    ArtifactStore,
    ArtifactStoreConfigurationError,
    ArtifactStoreRemoteError,
    ArtifactValidationError,
)


class FakeRemoteFile:
    def __init__(self, data: bytes, *, fail_close: BaseException | None = None) -> None:
        self.data = data
        self.fail_close = fail_close

    def read(self) -> bytes:
        return self.data

    def __enter__(self) -> "FakeRemoteFile":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.fail_close is not None and exc_type is None:
            raise self.fail_close


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
            if self.fs.fail_upload:
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
        self.storage = [str(storage)]
        self.remote = remote
        self.fail_download = fail_download
        self.fail_upload = fail_upload
        self.open_calls: list[tuple[str, str]] = []

    @staticmethod
    def _strip_protocol(path: str) -> str:
        return path.removeprefix("hf://")

    def _mapper(self, path: str) -> str:
        return hashlib.sha256(path.encode("utf-8")).hexdigest()

    def open(self, path: str, mode: str):
        self.open_calls.append((path, mode))
        key = self._strip_protocol(path).removeprefix("buckets/")
        candidate = Path(self.storage[-1]) / self._mapper(self._strip_protocol(path))
        candidate.parent.mkdir(parents=True, exist_ok=True)
        if "r" in mode:
            if self.fail_download is not None:
                raise self.fail_download
            if key not in self.remote:
                raise FileNotFoundError(key)
            candidate.write_bytes(self.remote[key])
            os.chmod(candidate, 0o600)
            return FakeRemoteFile(self.remote[key])
        return FakeWriteFile(self, candidate, key)


class PartialFailSimpleCache(FakeSimpleCache):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.partial_then_fail = True

    def open(self, path: str, mode: str):
        if "r" in mode and self.partial_then_fail:
            candidate = Path(self.storage[-1]) / self._mapper(self._strip_protocol(path))
            candidate.write_bytes(b"partial")
            raise RuntimeError("download interrupted")
        return super().open(path, mode)


class BarrierWriteFile(FakeWriteFile):
    def __enter__(self) -> "BarrierWriteFile":
        self.fs.barrier.wait(timeout=2)
        return self


class BarrierSimpleCache(FakeSimpleCache):
    def __init__(self, *args, barrier: threading.Barrier, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.barrier = barrier

    def open(self, path: str, mode: str):
        if "w" not in mode:
            return super().open(path, mode)
        self.open_calls.append((path, mode))
        key = self._strip_protocol(path).removeprefix("buckets/")
        candidate = Path(self.storage[-1]) / self._mapper(self._strip_protocol(path))
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return BarrierWriteFile(self, candidate, key)


class ActualSimpleCacheTarget(AbstractFileSystem):
    """A target FS with direct get_file semantics and observable exists()."""

    protocol = "hf"

    def __init__(self, remote: dict[str, bytes], exists_calls: list[str]) -> None:
        self.remote = remote
        self.exists_calls = exists_calls
        self.get_file_calls: list[tuple[str, str]] = []

    @classmethod
    def _strip_protocol(cls, path: str) -> str:
        return path.removeprefix("hf://")

    def exists(self, path: str, **kwargs) -> bool:
        self.exists_calls.append(path)
        raise AssertionError("authoritative paths-info hit must not call target exists")

    def get_file(
        self,
        rpath: str,
        lpath: str,
        callback=None,
        outfile=None,
        **kwargs,
    ) -> None:
        self.get_file_calls.append((rpath, lpath))
        key = rpath.removeprefix("buckets/")
        payload = self.remote[key]
        if outfile is None:
            Path(lpath).write_bytes(payload)
        else:
            outfile.write(payload)


class ActualSimpleCacheFactory:
    def __init__(self, storage: Path, remote: dict[str, bytes], exists_calls: list[str]) -> None:
        self.storage = storage
        self.remote = remote
        self.exists_calls = exists_calls
        self.target: ActualSimpleCacheTarget | None = None

    def __call__(self, protocol: str, **kwargs):
        self.target = ActualSimpleCacheTarget(self.remote, self.exists_calls)
        return fsspec.filesystem(
            "simplecache",
            fs=self.target,
            cache_storage=str(self.storage),
        )


class FakeHfApi:
    def __init__(
        self,
        *,
        token: str | None,
        private: bool = True,
        paths: list[object] | None = None,
        bucket_error: BaseException | None = None,
        paths_error: BaseException | None = None,
    ) -> None:
        self.token = token
        self.private = private
        self.paths = paths if paths is not None else [SimpleNamespace(path="object")]
        self.bucket_error = bucket_error
        self.paths_error = paths_error
        self.bucket_calls: list[str] = []
        self.paths_calls: list[tuple[str, list[str]]] = []

    def bucket_info(self, bucket_id: str):
        self.bucket_calls.append(bucket_id)
        if self.bucket_error is not None:
            raise self.bucket_error
        return SimpleNamespace(private=self.private)

    def get_bucket_paths_info(self, bucket_id: str, paths: list[str]):
        self.paths_calls.append((bucket_id, list(paths)))
        if self.paths_error is not None:
            raise self.paths_error
        return list(self.paths)


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
        token: str | None = "secret-token",
        private: bool = True,
        paths: list[object] | None = None,
        api_error: BaseException | None = None,
        paths_error: BaseException | None = None,
        download_error: BaseException | None = None,
        upload_error: BaseException | None = None,
        filesystem_factory_override=None,
    ) -> ArtifactStore:
        def api_factory(**kwargs):
            self.api_constructed += 1
            self.api = FakeHfApi(
                token=kwargs.get("token"),
                private=private,
                paths=paths,
                bucket_error=api_error,
                paths_error=paths_error,
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

    def decoder(self, payload: bytes) -> bytes:
        if payload == b"bad":
            raise ArtifactValidationError("invalid artifact")
        return payload

    def test_valid_l1_hit_never_constructs_or_calls_hf(self) -> None:
        store = self.make_store()
        candidate = store.local_candidate("object.mpk")
        candidate.write_bytes(b"local")
        os.chmod(candidate, 0o600)

        self.assertEqual(store.read_validated("object.mpk", self.decoder), b"local")
        self.assertEqual(self.api_constructed, 0)
        self.assertEqual(self.fs_constructed, 0)

    def test_l2_hit_uses_exact_paths_info_and_populates_l1_once(self) -> None:
        store = self.make_store()

        self.assertEqual(store.read("object.mpk"), b"remote")
        self.assertEqual(store.read("object.mpk"), b"remote")
        assert self.api is not None
        self.assertEqual(self.api.paths_calls, [("owner/bucket", ["foo/object.mpk"])])
        assert self.fs is not None
        self.assertEqual(self.fs.open_calls, [("hf://buckets/owner/bucket/foo/object.mpk", "rb")])
        self.assertEqual(store.local_candidate("object.mpk").read_bytes(), b"remote")

    def test_empty_paths_info_is_the_only_remote_miss(self) -> None:
        store = self.make_store(paths=[])
        self.assertIsNone(store.read("missing.mpk"))
        assert self.api is not None
        self.assertEqual(self.api.paths_calls, [("owner/bucket", ["foo/missing.mpk"])])

    def test_public_bucket_and_remote_failures_are_fatal(self) -> None:
        for kwargs in (
            {"private": False},
            {"api_error": RuntimeError("https://huggingface.co/api/buckets?token=secret-token")},
            {"paths_error": RuntimeError("token=secret-token")},
            {"download_error": RuntimeError("https://huggingface.co/?token=secret-token")},
        ):
            with self.subTest(kwargs=kwargs):
                store = self.make_store(**kwargs)
                with self.assertRaises(ArtifactStoreRemoteError):
                    store.read("object.mpk")

    def test_corrupt_l1_is_evicted_and_repaired_once(self) -> None:
        store = self.make_store()
        candidate = store.local_candidate("object.mpk")
        candidate.write_bytes(b"bad")
        os.chmod(candidate, 0o600)

        self.assertEqual(store.read_validated("object.mpk", self.decoder), b"remote")
        self.assertEqual(candidate.read_bytes(), b"remote")
        assert self.api is not None
        self.assertEqual(len(self.api.paths_calls), 1)
        assert self.fs is not None
        self.assertEqual(len(self.fs.open_calls), 1)

    def test_corrupt_l2_is_fatal_without_overwrite(self) -> None:
        store = self.make_store()
        self.remote["owner/bucket/foo/object.mpk"] = b"bad"

        with self.assertRaises(ArtifactValidationError):
            store.read_validated("object.mpk", self.decoder)
        candidate = store.local_candidate("object.mpk")
        self.assertTrue(candidate.exists())
        self.assertEqual(candidate.read_bytes(), b"bad")

    def test_upload_failure_removes_exact_candidate_and_raises(self) -> None:
        store = self.make_store(upload_error=RuntimeError("token=secret-token"))
        candidate = store.local_candidate("object.mpk")

        with self.assertRaises(ArtifactStoreRemoteError) as raised:
            store.write("object.mpk", b"payload")
        self.assertFalse(candidate.exists())
        self.assertNotIn("secret-token", str(raised.exception))

    def test_write_rejects_candidate_symlink_before_simplecache_open(self) -> None:
        store = self.make_store()
        sentinel = Path(self.tempdir.name) / "sentinel"
        sentinel.write_bytes(b"do-not-overwrite")
        candidate = store.local_candidate("object.mpk")
        candidate.symlink_to(sentinel)

        with self.assertRaises(ArtifactStoreRemoteError):
            store.write("object.mpk", b"payload")
        self.assertEqual(sentinel.read_bytes(), b"do-not-overwrite")
        self.assertIsNone(self.fs)
        self.assertFalse(self.remote.get("owner/bucket/foo/object.mpk") == b"payload")

    def test_failed_simplecache_download_evicts_partial_candidate(self) -> None:
        partial_fs: PartialFailSimpleCache | None = None

        def factory(protocol: str, **kwargs):
            nonlocal partial_fs
            partial_fs = PartialFailSimpleCache(self.local_dir, self.remote)
            return partial_fs

        store = self.make_store(filesystem_factory_override=factory)
        candidate = store.local_candidate("object.mpk")
        with self.assertRaises(ArtifactStoreRemoteError):
            store.read("object.mpk")
        self.assertFalse(candidate.exists())

        assert partial_fs is not None
        partial_fs.partial_then_fail = False
        self.assertEqual(store.read("object.mpk"), b"remote")
        assert self.api is not None
        self.assertEqual(len(self.api.paths_calls), 2)

    def test_actual_simplecache_hydration_does_not_call_target_exists(self) -> None:
        exists_calls: list[str] = []
        factory = ActualSimpleCacheFactory(self.local_dir, self.remote, exists_calls)
        store = self.make_store(filesystem_factory_override=factory)

        self.assertEqual(store.read("object.mpk"), b"remote")
        self.assertEqual(exists_calls, [])
        assert factory.target is not None
        self.assertEqual(len(factory.target.get_file_calls), 1)

    def test_successful_write_uploads_and_retains_mode_0600_candidate(self) -> None:
        store = self.make_store()
        store.write("object.mpk", b"new-payload")

        candidate = store.local_candidate("object.mpk")
        self.assertEqual(candidate.read_bytes(), b"new-payload")
        self.assertEqual(stat.S_IMODE(candidate.stat().st_mode), 0o600)
        self.assertEqual(self.remote["owner/bucket/foo/object.mpk"], b"new-payload")

    def test_bucket_directory_is_mode_0700_and_symlinks_are_rejected(self) -> None:
        store = self.make_store()
        self.assertEqual(stat.S_IMODE(self.local_dir.stat().st_mode), 0o700)

        linked = Path(self.tempdir.name) / "linked"
        linked.symlink_to(self.local_dir, target_is_directory=True)
        with self.assertRaises(ArtifactStoreConfigurationError):
            ArtifactStore(
                bucket_id="owner/bucket",
                local_dir=linked,
                api_factory=lambda **kwargs: None,
                filesystem_factory=lambda *args, **kwargs: None,
            )

    def test_duplicate_writes_are_sequential_last_writer_wins(self) -> None:
        store = self.make_store(prefix="")
        store.write("same.mpk", b"first")
        store.write("same.mpk", b"second")
        store.write("other.mpk", b"other")

        self.assertEqual(store.local_candidate("same.mpk").read_bytes(), b"second")
        self.assertEqual(store.local_candidate("other.mpk").read_bytes(), b"other")
        self.assertEqual(self.remote["owner/bucket/same.mpk"], b"second")

    def test_different_key_writes_enter_simplecache_concurrently(self) -> None:
        barrier = threading.Barrier(2)
        fs: BarrierSimpleCache | None = None

        def factory(protocol: str, **kwargs):
            nonlocal fs
            fs = BarrierSimpleCache(
                self.local_dir,
                self.remote,
                barrier=barrier,
            )
            return fs

        store = self.make_store(
            prefix="",
            filesystem_factory_override=factory,
        )
        store._verify_private_bucket()
        store._get_filesystem()
        failures: list[BaseException] = []

        def write(key: str, payload: bytes) -> None:
            try:
                store.write(key, payload)
            except BaseException as exc:  # pragma: no cover - assertion below
                failures.append(exc)

        first = threading.Thread(target=write, args=("first.mpk", b"first"))
        second = threading.Thread(target=write, args=("second.mpk", b"second"))
        first.start()
        second.start()
        first.join(timeout=3)
        second.join(timeout=3)

        self.assertFalse(first.is_alive() or second.is_alive())
        self.assertEqual(failures, [])
        assert fs is not None
        self.assertEqual(len(fs.open_calls), 2)
        self.assertEqual(self.remote["owner/bucket/first.mpk"], b"first")
        self.assertEqual(self.remote["owner/bucket/second.mpk"], b"second")

    def test_configuration_and_errors_never_expose_tokens(self) -> None:
        store = self.make_store(token="super-secret")
        configuration = store.configuration()
        self.assertNotIn("super-secret", repr(configuration))
        self.assertEqual(configuration["token_configured"], True)

        store = self.make_store(
            api_error=RuntimeError("Authorization: Bearer super-secret")
        )
        with self.assertRaises(ArtifactStoreRemoteError) as raised:
            store.read("object.mpk")
        self.assertNotIn("super-secret", str(raised.exception))

    def test_read_validated_only_repairs_artifact_validation_errors(self) -> None:
        store = self.make_store()
        store.local_candidate("object.mpk").write_bytes(b"local")

        def permission_failure(payload: bytes) -> bytes:
            raise PermissionError("permission denied")

        with self.assertRaises(PermissionError):
            store.read_validated("object.mpk", permission_failure)
        self.assertEqual(self.api_constructed, 0)

    def test_from_env_rejects_legacy_cache_settings_and_invalid_values(self) -> None:
        with patch.dict(os.environ, {"RAGTIME_CACHE_PATH": "/tmp/old.sqlite3"}, clear=False):
            with self.assertRaises(ArtifactStoreConfigurationError):
                ArtifactStore.from_env()
        with patch.dict(os.environ, {"RAGTIME_CACHE_PATH": "", "RAGTIME_CACHE_DIR": "/tmp/old"}, clear=False):
            with self.assertRaises(ArtifactStoreConfigurationError):
                ArtifactStore.from_env()
        with patch.dict(os.environ, {"RAGTIME_CACHE_PATH": "", "RAGTIME_CACHE_DIR": "", "RAGTIME_HF_BUCKET_ID": "wrong"}, clear=False):
            with self.assertRaises(ArtifactStoreConfigurationError):
                ArtifactStore.from_env()

    def test_object_keys_reject_traversal_absolute_and_empty_segments(self) -> None:
        store = self.make_store()
        for key in ("../bad", "/absolute", "a//b", "a/./b", "a/../b", "a?x=1", "a#frag", ""):
            with self.subTest(key=key):
                with self.assertRaises(ArtifactStoreConfigurationError):
                    store.local_candidate(key)


if __name__ == "__main__":
    unittest.main()
