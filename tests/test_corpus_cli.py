from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from trec_ragtime.corpus_cli import (
    ParquetShard,
    download_parquet_snapshot,
    fetch_parquet_manifest,
    main,
    resolve_parquet_revision,
    verify_parquet_snapshot,
)
from trec_ragtime.local_corpus import (
    LocalCorpusValidationError,
    build_from_parquet,
    default_local_corpus_path,
)
from trec_ragtime.paths import ensure_private_cache_dir
from tests.test_local_corpus import write_parquet

CONFIG_SPLITS = {
    "default": ("eng", "spa", "rus", "zho"),
    "mt_docs": ("spa.mt.eng", "rus.mt.eng", "zho.mt.eng"),
}


def write_manifest_snapshot(
    snapshot: Path,
    *,
    wrong_hash_path: str | None = None,
) -> list[ParquetShard]:
    manifest = []
    # A real snapshot directory is created by ensure_private_cache_dir at 0700.
    # mkdir(parents=True) would instead leave it at 0777 & ~umask, so the caller
    # rejects it as world-readable under a permissive umask and not under 0077.
    snapshot.mkdir(parents=True, exist_ok=True)
    snapshot.chmod(0o700)
    for config, splits in CONFIG_SPLITS.items():
        for split in splits:
            relative_path = f"{config}/{split}/0000.parquet"
            path = snapshot / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            content = f"content-{config}-{split}".encode()
            path.write_bytes(content)
            expected = b"wrong" if relative_path == wrong_hash_path else content
            manifest.append(
                ParquetShard(
                    path=relative_path,
                    config=config,
                    split=split,
                    size=len(content),
                    sha256=hashlib.sha256(expected).hexdigest(),
                )
            )
    return manifest


class CorpusCliTests(unittest.TestCase):
    def test_revision_is_resolved_once_to_an_immutable_sha(self) -> None:
        revision = "0123456789abcdef0123456789abcdef01234567"
        with patch("trec_ragtime.corpus_cli.HfApi") as api_type:
            api_type.return_value.dataset_info.return_value = SimpleNamespace(
                sha=revision
            )

            resolved = resolve_parquet_revision("refs/convert/parquet")

        self.assertEqual(resolved, revision)
        api_type.return_value.dataset_info.assert_called_once_with(
            "trec-ragtime/ragtime2",
            revision="refs/convert/parquet",
            token=False,
        )

    def test_literal_revision_does_not_require_hub_resolution(self) -> None:
        revision = "0123456789abcdef0123456789abcdef01234567"
        with patch("trec_ragtime.corpus_cli.HfApi") as api_type:
            resolved = resolve_parquet_revision(revision)

        self.assertEqual(resolved, revision)
        api_type.assert_not_called()

    def test_manifest_requires_exact_lfs_metadata_for_all_splits(self) -> None:
        revision = "0123456789abcdef0123456789abcdef01234567"
        siblings = []
        for config, splits in CONFIG_SPLITS.items():
            for split in splits:
                siblings.append(
                    SimpleNamespace(
                        rfilename=f"{config}/{split}/0000.parquet",
                        size=10,
                        lfs=SimpleNamespace(size=10, sha256="a" * 64),
                    )
                )
        siblings.append(
            SimpleNamespace(
                rfilename="unrelated/eng/0000.parquet",
                size=10,
                lfs=SimpleNamespace(size=10, sha256="b" * 64),
            )
        )
        with patch("trec_ragtime.corpus_cli.HfApi") as api_type:
            api_type.return_value.dataset_info.return_value = SimpleNamespace(
                sha=revision,
                siblings=siblings,
            )

            manifest = fetch_parquet_manifest(revision)

        self.assertEqual(len(manifest), 7)
        self.assertEqual(
            {(shard.config, shard.split) for shard in manifest},
            {
                (config, split)
                for config, splits in CONFIG_SPLITS.items()
                for split in splits
            },
        )
        api_type.return_value.dataset_info.assert_called_once_with(
            "trec-ragtime/ragtime2",
            revision=revision,
            files_metadata=True,
            token=False,
        )

    def test_download_selects_original_and_english_translation_splits(self) -> None:
        revision = "0123456789abcdef0123456789abcdef01234567"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / revision
            manifest = write_manifest_snapshot(snapshot)
            with patch(
                "trec_ragtime.corpus_cli.snapshot_download",
                return_value=str(snapshot),
            ) as download:
                result = download_parquet_snapshot(
                    revision,
                    manifest=manifest,
                    source_root=root,
                    max_workers=3,
                )

        self.assertEqual(result, root / revision)
        arguments = download.call_args.kwargs
        self.assertEqual(arguments["revision"], revision)
        self.assertEqual(arguments["token"], False)
        self.assertEqual(
            arguments["allow_patterns"],
            [
                "default/eng/0000.parquet",
                "default/rus/0000.parquet",
                "default/spa/0000.parquet",
                "default/zho/0000.parquet",
                "mt_docs/rus.mt.eng/0000.parquet",
                "mt_docs/spa.mt.eng/0000.parquet",
                "mt_docs/zho.mt.eng/0000.parquet",
            ],
        )

    def test_snapshot_verification_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory)
            manifest = write_manifest_snapshot(
                snapshot,
                wrong_hash_path="default/eng/0000.parquet",
            )

            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                verify_parquet_snapshot(snapshot, manifest)

    def test_corpus_first_initializes_a_private_project_cache(self) -> None:
        revision = "0123456789abcdef0123456789abcdef01234567"
        with tempfile.TemporaryDirectory() as directory:
            project_cache = Path(directory) / "new-project-cache"
            manifest = [
                ParquetShard(
                    path=f"{config}/{split}/0000.parquet",
                    config=config,
                    split=split,
                    size=1,
                    sha256="a" * 64,
                )
                for config, splits in CONFIG_SPLITS.items()
                for split in splits
            ]
            destination = project_cache / "corpus-source" / revision
            with (
                patch.dict(
                    os.environ,
                    {
                        "RAGTIME_PROJECT_CACHE_DIR": str(project_cache),
                        "XDG_CACHE_HOME": "",
                    },
                    clear=False,
                ),
                patch(
                    "trec_ragtime.corpus_cli.snapshot_download",
                    return_value=str(destination),
                ),
            ):
                download_parquet_snapshot(revision, manifest=manifest)
                ensure_private_cache_dir(project_cache / "artifacts-v1")

            self.assertEqual(stat.S_IMODE(project_cache.stat().st_mode), 0o700)
            self.assertEqual(
                stat.S_IMODE((project_cache / "corpus-source").stat().st_mode),
                0o700,
            )
            self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o700)

    def test_status_prints_machine_readable_index_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "eng.parquet"
            write_parquet(shard, [("doc-1", "one", None, None)])
            index_path = build_from_parquet(
                {"eng": [shard]},
                root / "corpus.duckdb",
                revision="revision-1",
            )
            output = io.StringIO()

            with redirect_stdout(output):
                exit_code = main(["status", "--index-path", str(index_path)])

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["document_count"], 1)
        self.assertEqual(payload["revision"], "revision-1")
        self.assertEqual(payload["path"], str(index_path))
        self.assertFalse(payload["canonical"])

    def test_default_status_rejects_a_partial_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment = {
                "RAGTIME_LOCAL_CORPUS_PATH": "",
                "RAGTIME_PROJECT_CACHE_DIR": str(
                    Path(directory) / "cache" / "trec-ragtime-2026"
                ),
                "XDG_CACHE_HOME": "",
            }
            with patch.dict(os.environ, environment, clear=False):
                root = Path(directory)
                shard = root / "eng.parquet"
                write_parquet(shard, [("doc-1", "one", None, None)])
                build_from_parquet(
                    {"eng": [shard]},
                    default_local_corpus_path(),
                    revision="revision-1",
                )

                with self.assertRaisesRegex(
                    LocalCorpusValidationError,
                    "expected_splits",
                ):
                    main(["status"])


if __name__ == "__main__":
    unittest.main()
