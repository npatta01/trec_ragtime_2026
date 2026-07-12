"""CLI for downloading and indexing the canonical RAGTIME2 corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from .huggingface import DEFAULT_HF_CONFIG, DEFAULT_HF_DATASET, DEFAULT_HF_SPLITS
from .local_corpus import (
    LocalCorpusIndex,
    build_from_parquet,
    default_local_corpus_path,
)

DEFAULT_PARQUET_REVISION = "refs/convert/parquet"
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class ParquetShard:
    """One canonical Hub shard and its immutable LFS identity."""

    path: str
    split: str
    size: int
    sha256: str


def _validate_manifest(
    manifest: Sequence[ParquetShard],
) -> tuple[ParquetShard, ...]:
    entries = tuple(manifest)
    if not entries:
        raise ValueError("manifest must not be empty")
    if any(not isinstance(entry, ParquetShard) for entry in entries):
        raise ValueError("manifest entries must be ParquetShard values")
    if any(not isinstance(entry.path, str) for entry in entries):
        raise ValueError("manifest shard paths must be strings")
    if len({entry.path for entry in entries}) != len(entries):
        raise ValueError("manifest must contain unique shard paths")

    represented_splits: set[str] = set()
    for entry in entries:
        expected_prefix = f"{DEFAULT_HF_CONFIG}/{entry.split}/"
        relative_name = entry.path.removeprefix(expected_prefix)
        if (
            entry.split not in DEFAULT_HF_SPLITS
            or not isinstance(entry.split, str)
            or not entry.path.startswith(expected_prefix)
            or not relative_name
            or "/" in relative_name
            or not relative_name.endswith(".parquet")
            or isinstance(entry.size, bool)
            or not isinstance(entry.size, int)
            or entry.size < 1
            or not isinstance(entry.sha256, str)
            or SHA256_PATTERN.fullmatch(entry.sha256) is None
        ):
            raise ValueError(f"invalid canonical Parquet manifest entry: {entry.path}")
        represented_splits.add(entry.split)
    if represented_splits != set(DEFAULT_HF_SPLITS):
        missing = sorted(set(DEFAULT_HF_SPLITS).difference(represented_splits))
        raise ValueError(
            "manifest is missing canonical Parquet splits: " + ", ".join(missing)
        )
    return tuple(sorted(entries, key=lambda entry: entry.path))


def default_corpus_source_root() -> Path:
    """Return the cross-worktree directory for pinned Parquet snapshots."""

    return default_local_corpus_path().parent / "corpus-source"


def resolve_parquet_revision(reference: str = DEFAULT_PARQUET_REVISION) -> str:
    """Resolve a branch, tag, or commit to one immutable Hub commit."""

    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("revision must be a non-empty string")
    normalized_reference = reference.strip()
    if COMMIT_PATTERN.fullmatch(normalized_reference) is not None:
        return normalized_reference
    info = HfApi().dataset_info(
        DEFAULT_HF_DATASET,
        revision=normalized_reference,
        token=False,
    )
    revision = info.sha
    if not isinstance(revision, str) or COMMIT_PATTERN.fullmatch(revision) is None:
        raise RuntimeError("Hugging Face returned an invalid revision SHA")
    return revision


def fetch_parquet_manifest(revision: str) -> tuple[ParquetShard, ...]:
    """Fetch the exact original-language shard manifest for one commit."""

    if not isinstance(revision, str) or COMMIT_PATTERN.fullmatch(revision) is None:
        raise ValueError("revision must be a 40-character lowercase commit SHA")
    info = HfApi().dataset_info(
        DEFAULT_HF_DATASET,
        revision=revision,
        files_metadata=True,
        token=False,
    )
    if info.sha != revision:
        raise RuntimeError("Hugging Face manifest revision does not match request")

    split_prefixes = {
        f"{DEFAULT_HF_CONFIG}/{split}/": split for split in DEFAULT_HF_SPLITS
    }
    manifest: list[ParquetShard] = []
    seen_paths: set[str] = set()
    for sibling in info.siblings or ():
        path = sibling.rfilename
        if not isinstance(path, str):
            raise RuntimeError("Hugging Face returned a non-string repository path")
        split = next(
            (
                candidate
                for prefix, candidate in split_prefixes.items()
                if path.startswith(prefix)
            ),
            None,
        )
        if split is None or not path.endswith(".parquet"):
            continue
        expected_prefix = f"{DEFAULT_HF_CONFIG}/{split}/"
        relative_name = path.removeprefix(expected_prefix)
        if not relative_name or "/" in relative_name:
            raise RuntimeError(f"unexpected canonical Parquet path: {path}")
        if path in seen_paths:
            raise RuntimeError(f"duplicate Parquet path in Hub manifest: {path}")
        seen_paths.add(path)
        lfs = sibling.lfs
        if (
            isinstance(sibling.size, bool)
            or not isinstance(sibling.size, int)
            or sibling.size < 1
            or lfs is None
            or lfs.size != sibling.size
            or not isinstance(lfs.sha256, str)
            or SHA256_PATTERN.fullmatch(lfs.sha256) is None
        ):
            raise RuntimeError(f"missing or invalid LFS metadata for {path}")
        manifest.append(
            ParquetShard(
                path=path,
                split=split,
                size=sibling.size,
                sha256=lfs.sha256,
            )
        )

    try:
        return _validate_manifest(manifest)
    except ValueError as exc:
        raise RuntimeError(f"invalid Hub Parquet manifest: {exc}") from None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_parquet_snapshot(
    snapshot: Path,
    manifest: Sequence[ParquetShard],
) -> dict[str, tuple[Path, ...]]:
    """Verify exact paths, sizes, and LFS hashes before indexing a snapshot."""

    if not snapshot.is_absolute():
        raise ValueError("snapshot path must be absolute")
    entries = _validate_manifest(manifest)
    expected = {shard.path: shard for shard in entries}

    actual_paths: set[str] = set()
    for split in DEFAULT_HF_SPLITS:
        split_root = snapshot / DEFAULT_HF_CONFIG / split
        if split_root.exists():
            actual_paths.update(
                path.relative_to(snapshot).as_posix()
                for path in split_root.rglob("*.parquet")
            )
    expected_paths = set(expected)
    if actual_paths != expected_paths:
        missing = sorted(expected_paths.difference(actual_paths))
        extra = sorted(actual_paths.difference(expected_paths))
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("extra=" + ",".join(extra))
        raise RuntimeError(
            "snapshot does not match Hub manifest (" + "; ".join(details) + ")"
        )

    for relative_path, shard in sorted(expected.items()):
        path = snapshot / relative_path
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"snapshot shard is not a regular file: {relative_path}")
        if path.stat().st_size != shard.size:
            raise RuntimeError(
                f"snapshot shard size mismatch: {relative_path}; "
                "delete that file and download again"
            )
        if _sha256(path) != shard.sha256:
            raise RuntimeError(
                f"snapshot shard SHA-256 mismatch: {relative_path}; "
                "delete that file and download again"
            )
    return {
        split: tuple(snapshot / shard.path for shard in entries if shard.split == split)
        for split in DEFAULT_HF_SPLITS
    }


def download_parquet_snapshot(
    revision: str,
    *,
    manifest: Sequence[ParquetShard] | None = None,
    source_root: Path | None = None,
    max_workers: int = 4,
) -> Path:
    """Download all original-language Parquet shards at a pinned revision."""

    if not isinstance(revision, str) or COMMIT_PATTERN.fullmatch(revision) is None:
        raise ValueError("revision must be a 40-character lowercase commit SHA")
    if (
        isinstance(max_workers, bool)
        or not isinstance(max_workers, int)
        or max_workers < 1
    ):
        raise ValueError("max_workers must be a positive integer")
    canonical_manifest = _validate_manifest(
        fetch_parquet_manifest(revision) if manifest is None else manifest
    )
    root = default_corpus_source_root() if source_root is None else source_root
    if not root.is_absolute():
        raise ValueError("source_root must be an absolute path")
    destination = root / revision
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination.chmod(0o700)
    downloaded = snapshot_download(
        repo_id=DEFAULT_HF_DATASET,
        repo_type="dataset",
        revision=revision,
        allow_patterns=[shard.path for shard in canonical_manifest],
        local_dir=destination,
        token=False,
        max_workers=max_workers,
    )
    return Path(downloaded)


def discover_parquet_shards(
    snapshot: Path,
    manifest: Sequence[ParquetShard],
) -> dict[str, tuple[Path, ...]]:
    """Return the canonical split-to-file mapping from a downloaded snapshot."""

    if not snapshot.is_absolute():
        raise ValueError("snapshot path must be absolute")
    return verify_parquet_snapshot(snapshot, manifest)


def _metadata_record(index: LocalCorpusIndex) -> dict[str, object]:
    metadata = index.metadata
    return {
        "path": str(index.path),
        "schema_version": metadata.schema_version,
        "dataset": metadata.dataset,
        "config": metadata.config,
        "revision": metadata.revision,
        "document_count": metadata.document_count,
        "shard_count": metadata.shard_count,
        "splits": list(metadata.splits),
        "canonical": metadata.dataset == DEFAULT_HF_DATASET
        and set(metadata.splits) == set(DEFAULT_HF_SPLITS),
        "built_at": metadata.built_at,
    }


def _build_command(arguments: argparse.Namespace) -> int:
    revision = resolve_parquet_revision(arguments.revision)
    index_path = (
        default_local_corpus_path()
        if arguments.index_path is None
        else Path(arguments.index_path).expanduser()
    )
    if not index_path.is_absolute():
        raise ValueError("index path must be absolute")

    if index_path.exists() and not arguments.replace and not arguments.download_only:
        with LocalCorpusIndex(
            index_path,
            expected_dataset=DEFAULT_HF_DATASET,
            expected_revision=revision,
            expected_splits=DEFAULT_HF_SPLITS,
        ) as index:
            print(
                json.dumps({"reused": True, **_metadata_record(index)}, sort_keys=True)
            )
        return 0

    source_root = (
        default_corpus_source_root()
        if arguments.source_root is None
        else Path(arguments.source_root).expanduser()
    )
    manifest = fetch_parquet_manifest(revision)
    snapshot = download_parquet_snapshot(
        revision,
        manifest=manifest,
        source_root=source_root,
        max_workers=arguments.max_workers,
    )
    verified_shards = discover_parquet_shards(snapshot, manifest)
    if arguments.download_only:
        print(
            json.dumps(
                {
                    "dataset": DEFAULT_HF_DATASET,
                    "revision": revision,
                    "snapshot": str(snapshot),
                },
                sort_keys=True,
            )
        )
        return 0

    try:
        built_path = build_from_parquet(
            verified_shards,
            index_path,
            revision=revision,
            replace=arguments.replace,
        )
    except FileExistsError:
        if arguments.replace:
            raise
        # Another worktree may have published the same index while this process
        # was downloading and verifying the shared snapshot.
        with LocalCorpusIndex(
            index_path,
            expected_dataset=DEFAULT_HF_DATASET,
            expected_revision=revision,
            expected_splits=DEFAULT_HF_SPLITS,
        ) as index:
            print(
                json.dumps({"reused": True, **_metadata_record(index)}, sort_keys=True)
            )
        return 0
    with LocalCorpusIndex(
        built_path,
        expected_dataset=DEFAULT_HF_DATASET,
        expected_revision=revision,
        expected_splits=DEFAULT_HF_SPLITS,
    ) as index:
        print(json.dumps({"reused": False, **_metadata_record(index)}, sort_keys=True))
    return 0


def _status_command(arguments: argparse.Namespace) -> int:
    path = (
        default_local_corpus_path()
        if arguments.index_path is None
        else Path(arguments.index_path).expanduser()
    )
    index = (
        LocalCorpusIndex(
            path,
            expected_dataset=DEFAULT_HF_DATASET,
            expected_splits=DEFAULT_HF_SPLITS,
        )
        if arguments.index_path is None
        else LocalCorpusIndex(path)
    )
    with index:
        print(json.dumps(_metadata_record(index), sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragtime-corpus",
        description="Download and build a pinned exact-ID RAGTIME2 corpus index.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="download and build the local index")
    build.add_argument("--revision", default=DEFAULT_PARQUET_REVISION)
    build.add_argument("--index-path")
    build.add_argument("--source-root")
    build.add_argument("--max-workers", type=int, default=4)
    build.add_argument("--replace", action="store_true")
    build.add_argument("--download-only", action="store_true")
    build.set_defaults(handler=_build_command)

    status = subparsers.add_parser("status", help="show local index metadata")
    status.add_argument("--index-path")
    status.set_defaults(handler=_status_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the corpus management command."""

    arguments = _parser().parse_args(argv)
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
