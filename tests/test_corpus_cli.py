from __future__ import annotations

import hashlib
import io
import json
import os
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
from tests.test_local_corpus import write_parquet


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
        for split in ("eng", "spa", "rus", "zho"):
            siblings.append(
                SimpleNamespace(
                    rfilename=f"default/{split}/0000.parquet",
                    size=10,
                    lfs=SimpleNamespace(size=10, sha256="a" * 64),
                )
            )
        siblings.append(
            SimpleNamespace(
                rfilename="mt_docs/eng/0000.parquet",
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

        self.assertEqual(len(manifest), 4)
        self.assertEqual(
            {shard.split for shard in manifest}, {"eng", "spa", "rus", "zho"}
        )
        self.assertTrue(all(shard.path.startswith("default/") for shard in manifest))
        api_type.return_value.dataset_info.assert_called_once_with(
            "trec-ragtime/ragtime2",
            revision=revision,
            files_metadata=True,
            token=False,
        )

    def test_download_selects_only_four_original_language_splits(self) -> None:
        revision = "0123456789abcdef0123456789abcdef01234567"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / revision
            manifest = []
            for split in ("eng", "spa", "rus", "zho"):
                relative_path = f"default/{split}/0000.parquet"
                path = snapshot / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                content = f"content-{split}".encode()
                path.write_bytes(content)
                manifest.append(
                    ParquetShard(
                        path=relative_path,
                        split=split,
                        size=len(content),
                        sha256=hashlib.sha256(content).hexdigest(),
                    )
                )
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
            ],
        )

    def test_snapshot_verification_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory)
            manifest = []
            for split in ("eng", "spa", "rus", "zho"):
                relative_path = f"default/{split}/0000.parquet"
                path = snapshot / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"tampered")
                expected = b"original" if split == "eng" else b"tampered"
                manifest.append(
                    ParquetShard(
                        path=relative_path,
                        split=split,
                        size=len(b"tampered"),
                        sha256=hashlib.sha256(expected).hexdigest(),
                    )
                )

            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                verify_parquet_snapshot(snapshot, manifest)

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
                "XDG_CACHE_HOME": str(Path(directory) / "cache"),
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
