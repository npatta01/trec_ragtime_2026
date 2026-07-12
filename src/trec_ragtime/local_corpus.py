"""Persistent exact-ID access to a downloaded RAGTIME2 corpus snapshot."""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import duckdb

from .huggingface import (
    DEFAULT_HF_CONFIG,
    DEFAULT_HF_DATASET,
    DEFAULT_HF_MT_CONFIG,
    DEFAULT_HF_MT_SPLITS,
    DEFAULT_HF_SPLITS,
)
from .models import HuggingFaceDocument, HydrationResult
from .paths import default_project_cache_dir, ensure_private_cache_dir

LOCAL_CORPUS_SCHEMA_VERSION = 2
DEFAULT_LOCAL_CORPUS_FILENAME = "ragtime2-corpus-v2.duckdb"
DEFAULT_LOCAL_CORPUS_BATCH_SIZE = 1_000
LOCAL_CORPUS_PATH_ENV = "RAGTIME_LOCAL_CORPUS_PATH"
DOCUMENT_ID_INDEX_NAME = "documents_document_id_idx"
TRANSLATION_DOCUMENT_ID_INDEX_NAME = "translations_document_id_idx"

MT_SPLIT_TO_SOURCE_SPLIT = {
    translation_split: translation_split.removesuffix(".mt.eng")
    for translation_split in DEFAULT_HF_MT_SPLITS
}

PathLike: TypeAlias = str | os.PathLike[str]


class LocalCorpusError(RuntimeError):
    """Base exception for the persistent local corpus."""


class LocalCorpusConfigurationError(LocalCorpusError):
    """Raised when a local index path or option is invalid."""


class LocalCorpusBuildError(LocalCorpusError):
    """Raised when Parquet shards cannot produce a valid local index."""


class LocalCorpusValidationError(LocalCorpusError):
    """Raised when an existing local index does not satisfy its schema."""


class LocalCorpusReadError(LocalCorpusError):
    """Raised when a validated local index can no longer be read."""


@dataclass(frozen=True, slots=True)
class LocalCorpusMetadata:
    """Identity and completeness metadata stored inside a corpus index."""

    schema_version: int
    dataset: str
    config: str
    revision: str
    document_count: int
    shard_count: int
    splits: tuple[str, ...]
    translation_config: str
    translation_count: int
    translation_shard_count: int
    translation_splits: tuple[str, ...]
    built_at: float


def _absolute_path(value: PathLike, setting: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise LocalCorpusConfigurationError(f"{setting} must be an absolute path")
    return path


def default_local_corpus_path() -> Path:
    """Return the project-level corpus index shared by every worktree."""

    explicit_path = os.getenv(LOCAL_CORPUS_PATH_ENV, "").strip()
    if explicit_path:
        return _absolute_path(explicit_path, LOCAL_CORPUS_PATH_ENV)

    return default_project_cache_dir() / DEFAULT_LOCAL_CORPUS_FILENAME


def _normalize_build_inputs(
    shards: Mapping[str, Iterable[PathLike]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not isinstance(shards, Mapping) or not shards:
        raise ValueError("shards must be a non-empty mapping of split to paths")

    normalized: list[tuple[str, tuple[str, ...]]] = []
    seen_splits: set[str] = set()
    for raw_split, raw_paths in shards.items():
        if not isinstance(raw_split, str) or not raw_split.strip():
            raise ValueError("shard split names must be non-empty strings")
        split = raw_split.strip()
        if split in seen_splits:
            raise ValueError("normalized shard split names must be unique")
        seen_splits.add(split)

        if isinstance(raw_paths, (str, os.PathLike)):
            raise ValueError(f"shards[{raw_split!r}] must be an iterable of paths")
        paths: list[str] = []
        for raw_path in raw_paths:
            try:
                path = os.fspath(raw_path)
            except TypeError:
                raise ValueError(
                    "Parquet shard paths must be strings or path-like"
                ) from None
            if isinstance(path, bytes):
                path = os.fsdecode(path)
            if not path:
                raise ValueError("Parquet shard paths must not be empty")
            paths.append(path)
        if not paths:
            raise ValueError(f"shards[{raw_split!r}] must contain at least one path")
        normalized.append((split, tuple(paths)))

    unsupported_splits = seen_splits.difference(DEFAULT_HF_SPLITS)
    if unsupported_splits:
        raise ValueError(
            "only original-language default splits are supported; unexpected: "
            + ", ".join(sorted(unsupported_splits))
        )
    return tuple(sorted(normalized))


def _normalize_translation_inputs(
    translations: Mapping[str, Iterable[PathLike]] | None,
    original_splits: Iterable[str],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if translations is None:
        return ()
    if not isinstance(translations, Mapping) or not translations:
        raise ValueError("translations must be None or a non-empty mapping")

    represented_splits = set(original_splits)
    expected_splits = {
        translation_split
        for translation_split, source_split in MT_SPLIT_TO_SOURCE_SPLIT.items()
        if source_split in represented_splits
    }
    supplied_splits: set[str] = set()
    normalized: list[tuple[str, tuple[str, ...]]] = []
    for raw_split, raw_paths in translations.items():
        if not isinstance(raw_split, str) or not raw_split.strip():
            raise ValueError("translation split names must be non-empty strings")
        split = raw_split.strip()
        if split in supplied_splits:
            raise ValueError("normalized translation split names must be unique")
        supplied_splits.add(split)
        if split not in DEFAULT_HF_MT_SPLITS:
            raise ValueError(f"unsupported translation split: {split}")
        if isinstance(raw_paths, (str, os.PathLike)):
            raise ValueError(
                f"translations[{raw_split!r}] must be an iterable of paths"
            )

        paths: list[str] = []
        for raw_path in raw_paths:
            try:
                path = os.fspath(raw_path)
            except TypeError:
                raise ValueError(
                    "translation Parquet paths must be strings or path-like"
                ) from None
            if isinstance(path, bytes):
                path = os.fsdecode(path)
            if not path:
                raise ValueError("translation Parquet paths must not be empty")
            paths.append(path)
        if not paths:
            raise ValueError(
                f"translations[{raw_split!r}] must contain at least one path"
            )
        normalized.append((split, tuple(paths)))

    if supplied_splits != expected_splits:
        missing = sorted(expected_splits.difference(supplied_splits))
        unexpected = sorted(supplied_splits.difference(expected_splits))
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unexpected:
            details.append("unexpected=" + ",".join(unexpected))
        raise ValueError(
            "translation splits must exactly match represented non-English "
            "source splits (" + "; ".join(details) + ")"
        )
    return tuple(sorted(normalized))


def _validate_identity(dataset: str, revision: str) -> tuple[str, str]:
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("dataset must be a non-empty string")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("revision must be a non-empty string")
    return dataset.strip(), revision.strip()


def _temporary_database_path(destination: Path) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary_path = Path(raw_path)
    temporary_path.unlink()
    return temporary_path


def _remove_database_files(path: Path) -> None:
    for candidate in (path, Path(f"{path}.wal")):
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass


def _install_database(
    temporary_path: Path,
    destination: Path,
    *,
    replace: bool,
) -> None:
    if replace:
        os.replace(temporary_path, destination)
        return

    # A hard link gives no-replace publication semantics on POSIX. Both files
    # are in the destination directory, so this cannot cross filesystems.
    try:
        os.link(temporary_path, destination)
    except FileExistsError:
        raise FileExistsError(
            f"local corpus index already exists: {destination}"
        ) from None
    temporary_path.unlink()


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _builder_lock(destination: Path) -> Iterator[None]:
    """Serialize builders targeting the same artifact on supported systems."""

    lock_path = destination.with_name(f".{destination.name}.lock")
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        try:
            import fcntl
        except ImportError:  # pragma: no cover - Windows has no fcntl.
            yield
        else:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def build_from_parquet(
    shards: Mapping[str, Iterable[PathLike]],
    destination: PathLike | None = None,
    *,
    revision: str,
    translations: Mapping[str, Iterable[PathLike]] | None = None,
    dataset: str = DEFAULT_HF_DATASET,
    replace: bool = False,
) -> Path:
    """Build and atomically publish an exact-ID multilingual corpus index."""

    destination_path = (
        default_local_corpus_path()
        if destination is None
        else _absolute_path(destination, "destination")
    )
    ensure_private_cache_dir(destination_path.parent)
    with _builder_lock(destination_path):
        return _build_from_parquet_unlocked(
            shards,
            destination_path,
            revision=revision,
            translations=translations,
            dataset=dataset,
            replace=replace,
        )


def _build_from_parquet_unlocked(
    shards: Mapping[str, Iterable[PathLike]],
    destination: PathLike | None = None,
    *,
    revision: str,
    translations: Mapping[str, Iterable[PathLike]] | None = None,
    dataset: str = DEFAULT_HF_DATASET,
    replace: bool = False,
) -> Path:
    """Build and atomically publish an exact-ID DuckDB corpus index.

    ``shards`` maps original-language splits to Parquet files containing
    ``id``, ``text``, ``url``, and ``date``. When supplied, ``translations``
    maps each represented non-English split's ``*.mt.eng`` companion to files
    containing ``id``, ``text``, and ``lang``. No discovery is performed.
    """

    if not isinstance(replace, bool):
        raise ValueError("replace must be a boolean")
    normalized_shards = _normalize_build_inputs(shards)
    normalized_translations = _normalize_translation_inputs(
        translations,
        (split for split, _ in normalized_shards),
    )
    normalized_dataset, normalized_revision = _validate_identity(dataset, revision)
    destination_path = (
        default_local_corpus_path()
        if destination is None
        else _absolute_path(destination, "destination")
    )
    ensure_private_cache_dir(destination_path.parent)
    if destination_path.exists() and not replace:
        raise FileExistsError(f"local corpus index already exists: {destination_path}")

    temporary_path = _temporary_database_path(destination_path)
    connection: duckdb.DuckDBPyConnection | None = None
    try:
        connection = duckdb.connect(str(temporary_path))
        connection.execute(
            """
            CREATE TABLE documents (
                document_id VARCHAR NOT NULL,
                text VARCHAR NOT NULL,
                url VARCHAR,
                date VARCHAR,
                split VARCHAR NOT NULL
            )
            """
        )
        for split, paths in normalized_shards:
            connection.execute(
                """
                INSERT INTO documents (document_id, text, url, date, split)
                SELECT
                    CAST(id AS VARCHAR),
                    CAST(text AS VARCHAR),
                    CAST(url AS VARCHAR),
                    CAST(date AS VARCHAR),
                    ?
                FROM read_parquet(?, union_by_name = true)
                """,
                [split, list(paths)],
            )

        invalid_row = connection.execute(
            """
            SELECT count(*)
            FROM documents
            WHERE document_id = '' OR text IS NULL OR split = ''
            """
        ).fetchone()
        if invalid_row is None:
            raise LocalCorpusBuildError("unable to count invalid document rows")
        invalid_row_count = int(invalid_row[0])
        if invalid_row_count:
            raise LocalCorpusBuildError(
                f"snapshot contains {invalid_row_count} invalid document rows"
            )

        duplicate = connection.execute(
            """
            SELECT document_id, count(*) AS occurrences
            FROM documents
            GROUP BY document_id
            HAVING count(*) > 1
            LIMIT 1
            """
        ).fetchone()
        if duplicate is not None:
            raise LocalCorpusBuildError(
                "snapshot contains duplicate document IDs; "
                f"first duplicate: {duplicate[0]!r}"
            )

        connection.execute(
            """
            CREATE TABLE translations (
                document_id VARCHAR NOT NULL,
                text VARCHAR NOT NULL,
                translation_split VARCHAR NOT NULL,
                source_split VARCHAR NOT NULL,
                target_language VARCHAR NOT NULL
            )
            """
        )
        for translation_split, paths in normalized_translations:
            connection.execute(
                """
                INSERT INTO translations (
                    document_id,
                    text,
                    translation_split,
                    source_split,
                    target_language
                )
                SELECT
                    CAST(id AS VARCHAR),
                    CAST(text AS VARCHAR),
                    ?,
                    ?,
                    CAST(lang AS VARCHAR)
                FROM read_parquet(?, union_by_name = true)
                """,
                [
                    translation_split,
                    MT_SPLIT_TO_SOURCE_SPLIT[translation_split],
                    list(paths),
                ],
            )

        invalid_translation_count_row = connection.execute(
            """
            SELECT count(*)
            FROM translations
            WHERE document_id = ''
               OR text IS NULL
               OR text = ''
               OR translation_split = ''
               OR source_split = ''
               OR target_language != 'eng'
            """
        ).fetchone()
        if invalid_translation_count_row is None:
            raise LocalCorpusBuildError("unable to count invalid translation rows")
        invalid_translation_count = int(invalid_translation_count_row[0])
        if invalid_translation_count:
            raise LocalCorpusBuildError(
                "translation snapshot contains "
                f"{invalid_translation_count} invalid rows; every row must have a "
                "non-empty ID and text with lang='eng'"
            )

        duplicate_translation = connection.execute(
            """
            SELECT document_id, count(*) AS occurrences
            FROM translations
            GROUP BY document_id
            HAVING count(*) > 1
            LIMIT 1
            """
        ).fetchone()
        if duplicate_translation is not None:
            raise LocalCorpusBuildError(
                "translation snapshot contains duplicate document IDs; "
                f"first duplicate: {duplicate_translation[0]!r}"
            )

        orphan_translation = connection.execute(
            """
            SELECT translations.document_id
            FROM translations
            LEFT JOIN documents USING (document_id)
            WHERE documents.document_id IS NULL
            LIMIT 1
            """
        ).fetchone()
        if orphan_translation is not None:
            raise LocalCorpusBuildError(
                "translation snapshot contains an ID without a canonical document; "
                f"first orphan: {orphan_translation[0]!r}"
            )

        mismatched_translation = connection.execute(
            """
            SELECT
                translations.document_id,
                translations.source_split,
                documents.split
            FROM translations
            JOIN documents USING (document_id)
            WHERE translations.source_split != documents.split
            LIMIT 1
            """
        ).fetchone()
        if mismatched_translation is not None:
            raise LocalCorpusBuildError(
                "translation split does not match its canonical document split; "
                f"document {mismatched_translation[0]!r} is in "
                f"{mismatched_translation[2]!r}, not "
                f"{mismatched_translation[1]!r}"
            )

        if normalized_translations:
            translated_source_splits = [
                MT_SPLIT_TO_SOURCE_SPLIT[translation_split]
                for translation_split, _ in normalized_translations
            ]
            placeholders = ",".join("?" for _ in translated_source_splits)
            missing_translation = connection.execute(
                """
                SELECT documents.document_id
                FROM documents
                LEFT JOIN translations USING (document_id)
                WHERE documents.split IN ("""
                + placeholders
                + """
                  )
                  AND translations.document_id IS NULL
                LIMIT 1
                """,
                translated_source_splits,
            ).fetchone()
            if missing_translation is not None:
                raise LocalCorpusBuildError(
                    "translation snapshot does not cover every canonical document; "
                    f"first missing translation: {missing_translation[0]!r}"
                )

        connection.execute(
            f"CREATE UNIQUE INDEX {DOCUMENT_ID_INDEX_NAME} ON documents(document_id)"
        )
        connection.execute(
            f"CREATE UNIQUE INDEX {TRANSLATION_DOCUMENT_ID_INDEX_NAME} "
            "ON translations(document_id)"
        )
        connection.execute("ANALYZE documents")
        connection.execute("ANALYZE translations")
        document_count_row = connection.execute(
            "SELECT count(*) FROM documents"
        ).fetchone()
        if document_count_row is None:
            raise LocalCorpusBuildError("unable to count indexed documents")
        document_count = int(document_count_row[0])
        shard_count = sum(len(paths) for _, paths in normalized_shards)
        translation_count_row = connection.execute(
            "SELECT count(*) FROM translations"
        ).fetchone()
        if translation_count_row is None:
            raise LocalCorpusBuildError("unable to count indexed translations")
        translation_count = int(translation_count_row[0])
        translation_shard_count = sum(
            len(paths) for _, paths in normalized_translations
        )
        built_at = time.time()
        connection.execute(
            """
            CREATE TABLE corpus_metadata (
                schema_version INTEGER NOT NULL,
                dataset VARCHAR NOT NULL,
                config VARCHAR NOT NULL,
                revision VARCHAR NOT NULL,
                document_count UBIGINT NOT NULL,
                shard_count UINTEGER NOT NULL,
                splits_json VARCHAR NOT NULL,
                translation_config VARCHAR NOT NULL,
                translation_count UBIGINT NOT NULL,
                translation_shard_count UINTEGER NOT NULL,
                translation_splits_json VARCHAR NOT NULL,
                built_at DOUBLE NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO corpus_metadata
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                LOCAL_CORPUS_SCHEMA_VERSION,
                normalized_dataset,
                DEFAULT_HF_CONFIG,
                normalized_revision,
                document_count,
                shard_count,
                json.dumps(
                    [split for split, _ in normalized_shards],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                DEFAULT_HF_MT_CONFIG,
                translation_count,
                translation_shard_count,
                json.dumps(
                    [split for split, _ in normalized_translations],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                built_at,
            ],
        )
        connection.execute("CHECKPOINT")
        connection.close()
        connection = None
        os.chmod(temporary_path, 0o600)
        _fsync_file(temporary_path)

        # Re-open read-only and validate the exact artifact before publication.
        LocalCorpusIndex(temporary_path)
        _install_database(temporary_path, destination_path, replace=replace)
        try:
            _fsync_directory(destination_path.parent)
        except OSError as exc:
            raise LocalCorpusBuildError(
                f"local corpus index was published at {destination_path}, but its "
                f"directory durability sync failed ({exc.__class__.__name__}: {exc}); "
                "inspect the published index before retrying"
            ) from None
        return destination_path
    except (LocalCorpusError, FileExistsError):
        raise
    except (duckdb.Error, OSError) as exc:
        raise LocalCorpusBuildError(
            f"unable to build local corpus index ({exc.__class__.__name__}): {exc}"
        ) from None
    finally:
        if connection is not None:
            connection.close()
        _remove_database_files(temporary_path)


def _chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


class LocalCorpusIndex:
    """Concurrent read-only point lookups over a revision-pinned corpus."""

    backend_name = "local"

    def __init__(
        self,
        path: PathLike | None = None,
        *,
        batch_size: int = DEFAULT_LOCAL_CORPUS_BATCH_SIZE,
        expected_dataset: str | None = None,
        expected_revision: str | None = None,
        expected_splits: Iterable[str] | None = None,
        expected_translation_splits: Iterable[str] | None = None,
    ) -> None:
        self.path = (
            default_local_corpus_path()
            if path is None
            else _absolute_path(path, "local corpus path")
        )
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise LocalCorpusConfigurationError("batch_size must be a positive integer")
        if expected_dataset is not None and (
            not isinstance(expected_dataset, str) or not expected_dataset.strip()
        ):
            raise LocalCorpusConfigurationError(
                "expected_dataset must be a non-empty string or None"
            )
        if expected_revision is not None and (
            not isinstance(expected_revision, str) or not expected_revision.strip()
        ):
            raise LocalCorpusConfigurationError(
                "expected_revision must be a non-empty string or None"
            )
        if isinstance(expected_splits, (str, bytes)):
            raise LocalCorpusConfigurationError(
                "expected_splits must be an iterable of unique split names or None"
            )
        normalized_expected_splits: tuple[str, ...] | None = None
        if expected_splits is not None:
            try:
                raw_expected_splits = tuple(expected_splits)
            except TypeError:
                raise LocalCorpusConfigurationError(
                    "expected_splits must be an iterable of unique split names or None"
                ) from None
            if (
                not raw_expected_splits
                or any(
                    not isinstance(split, str) or not split.strip()
                    for split in raw_expected_splits
                )
                or len({split.strip() for split in raw_expected_splits})
                != len(raw_expected_splits)
            ):
                raise LocalCorpusConfigurationError(
                    "expected_splits must be an iterable of unique split names or None"
                )
            normalized_expected_splits = tuple(
                sorted(split.strip() for split in raw_expected_splits)
            )
        if isinstance(expected_translation_splits, (str, bytes)):
            raise LocalCorpusConfigurationError(
                "expected_translation_splits must be an iterable of unique "
                "translation split names or None"
            )
        normalized_expected_translation_splits: tuple[str, ...] | None = None
        if expected_translation_splits is not None:
            try:
                raw_expected_translation_splits = tuple(expected_translation_splits)
            except TypeError:
                raise LocalCorpusConfigurationError(
                    "expected_translation_splits must be an iterable of unique "
                    "translation split names or None"
                ) from None
            normalized_translation_names = tuple(
                split.strip()
                for split in raw_expected_translation_splits
                if isinstance(split, str)
            )
            if (
                len(normalized_translation_names)
                != len(raw_expected_translation_splits)
                or any(not split for split in normalized_translation_names)
                or len(set(normalized_translation_names))
                != len(normalized_translation_names)
                or any(
                    split not in DEFAULT_HF_MT_SPLITS
                    for split in normalized_translation_names
                )
            ):
                raise LocalCorpusConfigurationError(
                    "expected_translation_splits must be an iterable of unique "
                    "supported translation split names or None"
                )
            normalized_expected_translation_splits = tuple(
                sorted(normalized_translation_names)
            )
        if not self.path.exists():
            raise LocalCorpusConfigurationError(
                f"local corpus index does not exist: {self.path}"
            )
        if not self.path.is_file():
            raise LocalCorpusConfigurationError(
                f"local corpus index is not a regular file: {self.path}"
            )

        self.batch_size = batch_size
        self.expected_dataset = (
            None if expected_dataset is None else expected_dataset.strip()
        )
        self.expected_revision = (
            None if expected_revision is None else expected_revision.strip()
        )
        self.expected_splits = normalized_expected_splits
        self.expected_translation_splits = normalized_expected_translation_splits
        connection = self._connect()
        try:
            metadata = self._read_metadata(connection)
            self._validate_schema(connection, metadata)
            self._validate_expected_identity(metadata)
        finally:
            connection.close()

    def _connect(self) -> duckdb.DuckDBPyConnection:
        try:
            return duckdb.connect(str(self.path), read_only=True)
        except duckdb.Error as exc:
            raise LocalCorpusReadError(
                f"unable to open local corpus index read-only: {exc}"
            ) from None

    @staticmethod
    def _validate_schema(
        connection: duckdb.DuckDBPyConnection,
        metadata: LocalCorpusMetadata,
    ) -> None:
        try:
            columns = {
                row[0]: row[1]
                for row in connection.execute(
                    """
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_schema = 'main' AND table_name = 'documents'
                    """
                ).fetchall()
            }
            expected_columns = {
                "document_id": "VARCHAR",
                "text": "VARCHAR",
                "url": "VARCHAR",
                "date": "VARCHAR",
                "split": "VARCHAR",
            }
            if any(
                columns.get(name) != kind for name, kind in expected_columns.items()
            ):
                raise LocalCorpusValidationError(
                    "local corpus documents table has an incompatible schema"
                )

            index_row = connection.execute(
                """
                SELECT is_unique
                FROM duckdb_indexes()
                WHERE schema_name = 'main'
                  AND table_name = 'documents'
                  AND index_name = ?
                """,
                [DOCUMENT_ID_INDEX_NAME],
            ).fetchone()
            if index_row is None or index_row[0] is not True:
                raise LocalCorpusValidationError(
                    "local corpus is missing its unique document-ID index"
                )
            actual_document_count_row = connection.execute(
                "SELECT count(*) FROM documents"
            ).fetchone()
            if actual_document_count_row is None:
                raise LocalCorpusValidationError(
                    "unable to count local corpus documents"
                )
            actual_document_count = int(actual_document_count_row[0])
            if actual_document_count != metadata.document_count:
                raise LocalCorpusValidationError(
                    "local corpus row count does not match its metadata"
                )
            actual_splits = tuple(
                row[0]
                for row in connection.execute(
                    "SELECT DISTINCT split FROM documents ORDER BY split"
                ).fetchall()
            )
            if actual_splits != metadata.splits:
                raise LocalCorpusValidationError(
                    "local corpus splits do not match its metadata"
                )

            translation_columns = {
                row[0]: row[1]
                for row in connection.execute(
                    """
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_schema = 'main' AND table_name = 'translations'
                    """
                ).fetchall()
            }
            expected_translation_columns = {
                "document_id": "VARCHAR",
                "text": "VARCHAR",
                "translation_split": "VARCHAR",
                "source_split": "VARCHAR",
                "target_language": "VARCHAR",
            }
            if any(
                translation_columns.get(name) != kind
                for name, kind in expected_translation_columns.items()
            ):
                raise LocalCorpusValidationError(
                    "local corpus translations table has an incompatible schema"
                )

            translation_index_row = connection.execute(
                """
                SELECT is_unique
                FROM duckdb_indexes()
                WHERE schema_name = 'main'
                  AND table_name = 'translations'
                  AND index_name = ?
                """,
                [TRANSLATION_DOCUMENT_ID_INDEX_NAME],
            ).fetchone()
            if translation_index_row is None or translation_index_row[0] is not True:
                raise LocalCorpusValidationError(
                    "local corpus is missing its unique translation-ID index"
                )
            actual_translation_count_row = connection.execute(
                "SELECT count(*) FROM translations"
            ).fetchone()
            if actual_translation_count_row is None:
                raise LocalCorpusValidationError(
                    "unable to count local corpus translations"
                )
            if int(actual_translation_count_row[0]) != metadata.translation_count:
                raise LocalCorpusValidationError(
                    "local corpus translation count does not match its metadata"
                )
            actual_translation_splits = tuple(
                row[0]
                for row in connection.execute(
                    """
                    SELECT DISTINCT translation_split
                    FROM translations
                    ORDER BY translation_split
                    """
                ).fetchall()
            )
            if actual_translation_splits != metadata.translation_splits:
                raise LocalCorpusValidationError(
                    "local corpus translation splits do not match its metadata"
                )
        except LocalCorpusValidationError:
            raise
        except duckdb.Error as exc:
            raise LocalCorpusValidationError(
                f"unable to validate local corpus schema: {exc}"
            ) from None

    @staticmethod
    def _read_metadata(
        connection: duckdb.DuckDBPyConnection,
    ) -> LocalCorpusMetadata:
        try:
            rows = connection.execute(
                """
                SELECT
                    schema_version,
                    dataset,
                    config,
                    revision,
                    document_count,
                    shard_count,
                    splits_json,
                    translation_config,
                    translation_count,
                    translation_shard_count,
                    translation_splits_json,
                    built_at
                FROM corpus_metadata
                """
            ).fetchall()
        except duckdb.Error as exc:
            raise LocalCorpusValidationError(
                f"local corpus metadata is missing or unreadable: {exc}"
            ) from None
        if len(rows) != 1:
            raise LocalCorpusValidationError(
                "local corpus must contain exactly one metadata row"
            )
        (
            schema_version,
            dataset,
            config,
            revision,
            document_count,
            shard_count,
            splits_json,
            translation_config,
            translation_count,
            translation_shard_count,
            translation_splits_json,
            built_at,
        ) = rows[0]
        try:
            splits_value = json.loads(splits_json)
            translation_splits_value = json.loads(translation_splits_json)
        except (TypeError, json.JSONDecodeError):
            raise LocalCorpusValidationError(
                "local corpus metadata contains invalid split JSON"
            ) from None
        if (
            schema_version != LOCAL_CORPUS_SCHEMA_VERSION
            or not isinstance(dataset, str)
            or not dataset
            or config != DEFAULT_HF_CONFIG
            or not isinstance(revision, str)
            or not revision
            or isinstance(document_count, bool)
            or not isinstance(document_count, int)
            or document_count < 0
            or isinstance(shard_count, bool)
            or not isinstance(shard_count, int)
            or shard_count < 1
            or not isinstance(splits_value, list)
            or not splits_value
            or any(not isinstance(split, str) or not split for split in splits_value)
            or len(set(splits_value)) != len(splits_value)
            or translation_config != DEFAULT_HF_MT_CONFIG
            or isinstance(translation_count, bool)
            or not isinstance(translation_count, int)
            or translation_count < 0
            or isinstance(translation_shard_count, bool)
            or not isinstance(translation_shard_count, int)
            or translation_shard_count < 0
            or not isinstance(translation_splits_value, list)
            or any(
                not isinstance(split, str) or split not in DEFAULT_HF_MT_SPLITS
                for split in translation_splits_value
            )
            or len(set(translation_splits_value)) != len(translation_splits_value)
            or (translation_count == 0) != (translation_shard_count == 0)
            or (translation_count == 0) != (not translation_splits_value)
            or any(
                MT_SPLIT_TO_SOURCE_SPLIT[split] not in splits_value
                for split in translation_splits_value
            )
            or isinstance(built_at, bool)
            or not isinstance(built_at, (int, float))
            or not math.isfinite(float(built_at))
            or built_at <= 0
        ):
            raise LocalCorpusValidationError(
                "local corpus metadata contains invalid values"
            )
        return LocalCorpusMetadata(
            schema_version=schema_version,
            dataset=dataset,
            config=config,
            revision=revision,
            document_count=document_count,
            shard_count=shard_count,
            splits=tuple(splits_value),
            translation_config=translation_config,
            translation_count=translation_count,
            translation_shard_count=translation_shard_count,
            translation_splits=tuple(translation_splits_value),
            built_at=float(built_at),
        )

    def _validate_expected_identity(self, metadata: LocalCorpusMetadata) -> None:
        if (
            self.expected_dataset is not None
            and metadata.dataset != self.expected_dataset
        ):
            raise LocalCorpusValidationError(
                "local corpus dataset does not match expected_dataset"
            )
        if (
            self.expected_revision is not None
            and metadata.revision != self.expected_revision
        ):
            raise LocalCorpusValidationError(
                "local corpus revision does not match expected_revision"
            )
        if (
            self.expected_splits is not None
            and tuple(sorted(metadata.splits)) != self.expected_splits
        ):
            raise LocalCorpusValidationError(
                "local corpus splits do not match expected_splits"
            )
        if (
            self.expected_translation_splits is not None
            and tuple(sorted(metadata.translation_splits))
            != self.expected_translation_splits
        ):
            raise LocalCorpusValidationError(
                "local corpus translation splits do not match "
                "expected_translation_splits"
            )

    @property
    def metadata(self) -> LocalCorpusMetadata:
        """Read metadata from the currently published database artifact."""

        connection = self._connect()
        try:
            metadata = self._read_metadata(connection)
            self._validate_expected_identity(metadata)
            return metadata
        finally:
            connection.close()

    def verify_integrity(self) -> LocalCorpusMetadata:
        """Deep-check translation coverage and return validated metadata."""

        connection = self._connect()
        try:
            metadata = self._read_metadata(connection)
            self._validate_schema(connection, metadata)
            self._validate_expected_identity(metadata)
            expected_groups = {
                (
                    translation_split,
                    MT_SPLIT_TO_SOURCE_SPLIT[translation_split],
                    "eng",
                )
                for translation_split in metadata.translation_splits
            }
            actual_groups = {
                (translation_split, source_split, target_language)
                for translation_split, source_split, target_language in connection.execute(
                    """
                    SELECT DISTINCT
                        translation_split,
                        source_split,
                        target_language
                    FROM translations
                    """
                ).fetchall()
            }
            if actual_groups != expected_groups:
                raise LocalCorpusValidationError(
                    "local corpus translation language or split mapping is invalid"
                )

            invalid_translation = connection.execute(
                """
                SELECT document_id
                FROM translations
                WHERE document_id = '' OR text = ''
                LIMIT 1
                """
            ).fetchone()
            if invalid_translation is not None:
                raise LocalCorpusValidationError(
                    "local corpus contains an empty translation ID or text"
                )
            orphan_translation = connection.execute(
                """
                SELECT translations.document_id
                FROM translations
                LEFT JOIN documents USING (document_id)
                WHERE documents.document_id IS NULL
                LIMIT 1
                """
            ).fetchone()
            if orphan_translation is not None:
                raise LocalCorpusValidationError(
                    "local corpus contains an orphan translation"
                )
            mismatched_translation = connection.execute(
                """
                SELECT translations.document_id
                FROM translations
                JOIN documents USING (document_id)
                WHERE translations.source_split != documents.split
                LIMIT 1
                """
            ).fetchone()
            if mismatched_translation is not None:
                raise LocalCorpusValidationError(
                    "local corpus translation does not match its document split"
                )

            expected_source_splits = [
                MT_SPLIT_TO_SOURCE_SPLIT[split] for split in metadata.translation_splits
            ]
            if expected_source_splits:
                placeholders = ",".join("?" for _ in expected_source_splits)
                missing_translation = connection.execute(
                    """
                    SELECT documents.document_id
                    FROM documents
                    LEFT JOIN translations USING (document_id)
                    WHERE documents.split IN ("""
                    + placeholders
                    + """)
                      AND translations.document_id IS NULL
                    LIMIT 1
                    """,
                    expected_source_splits,
                ).fetchone()
                if missing_translation is not None:
                    raise LocalCorpusValidationError(
                        "local corpus is missing an expected English translation"
                    )
            return metadata
        except LocalCorpusValidationError:
            raise
        except duckdb.Error as exc:
            raise LocalCorpusValidationError(
                f"unable to verify local corpus integrity: {exc}"
            ) from None
        finally:
            connection.close()

    @property
    def dataset(self) -> str:
        return self.metadata.dataset

    @property
    def revision(self) -> str:
        return self.metadata.revision

    def close(self) -> None:
        """Retain interface compatibility; lookups hold no persistent connection."""

    def __enter__(self) -> LocalCorpusIndex:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _ordered_document_ids(document_ids: Iterable[str]) -> list[str]:
        ordered_ids: list[str] = []
        seen_ids: set[str] = set()
        for document_id in document_ids:
            if not isinstance(document_id, str) or not document_id:
                raise ValueError("document IDs must be non-empty strings")
            if document_id not in seen_ids:
                seen_ids.add(document_id)
                ordered_ids.append(document_id)
        return ordered_ids

    def get_many(
        self,
        document_ids: Iterable[str],
        *,
        refresh: bool = False,
    ) -> HydrationResult:
        """Resolve arbitrarily many IDs locally, preserving first-seen order."""

        if not isinstance(refresh, bool):
            raise ValueError("refresh must be a boolean")
        ordered_ids = self._ordered_document_ids(document_ids)
        if not ordered_ids:
            return HydrationResult(documents=(), unresolved_ids=(), failures=())

        found: dict[str, HuggingFaceDocument] = {}
        expected_translation_source_splits: set[str] = set()
        connection = self._connect()
        try:
            metadata = self._read_metadata(connection)
            self._validate_expected_identity(metadata)
            expected_translation_source_splits = {
                MT_SPLIT_TO_SOURCE_SPLIT[split] for split in metadata.translation_splits
            }
            for chunk in _chunks(ordered_ids, self.batch_size):
                placeholders = ",".join("?" for _ in chunk)
                rows = connection.execute(
                    "SELECT documents.document_id, documents.text, documents.url, "
                    "documents.date, documents.split, translations.text "
                    "FROM documents LEFT JOIN translations USING (document_id) "
                    f"WHERE documents.document_id IN ({placeholders})",
                    chunk,
                ).fetchall()
                for document_id, text, url, date, split, english_text in rows:
                    if document_id in found:
                        raise LocalCorpusValidationError(
                            "local corpus returned a duplicate document ID"
                        )
                    if (
                        not isinstance(document_id, str)
                        or not isinstance(text, str)
                        or not isinstance(split, str)
                        or (url is not None and not isinstance(url, str))
                        or (date is not None and not isinstance(date, str))
                        or (
                            english_text is not None
                            and not isinstance(english_text, str)
                        )
                    ):
                        raise LocalCorpusValidationError(
                            "local corpus returned an invalid document row"
                        )
                    found[document_id] = HuggingFaceDocument(
                        dataset=metadata.dataset,
                        config=metadata.config,
                        split=split,
                        document_id=document_id,
                        text=text,
                        url=url,
                        date=date,
                        local_cache_hit=None,
                        cache_stored_at=None,
                        source="huggingface_local_index",
                        english_text=english_text,
                    )
        except LocalCorpusError:
            raise
        except duckdb.Error as exc:
            raise LocalCorpusReadError(
                f"unable to query local corpus index: {exc}"
            ) from None
        finally:
            connection.close()

        return HydrationResult(
            documents=tuple(
                found[document_id]
                for document_id in ordered_ids
                if document_id in found
            ),
            unresolved_ids=tuple(
                document_id for document_id in ordered_ids if document_id not in found
            ),
            failures=(),
            missing_translation_ids=tuple(
                document_id
                for document_id in ordered_ids
                if document_id in found
                and found[document_id].split in expected_translation_source_splits
                and found[document_id].english_text is None
            ),
        )

    def enrich_documents(
        self,
        document_ids: Iterable[str],
        *,
        refresh: bool = False,
    ) -> HydrationResult:
        """Alias for :meth:`get_many` suited to post-retrieval enrichment."""

        return self.get_many(document_ids, refresh=refresh)


LocalHuggingFaceCorpus = LocalCorpusIndex


__all__ = [
    "DEFAULT_LOCAL_CORPUS_BATCH_SIZE",
    "DEFAULT_LOCAL_CORPUS_FILENAME",
    "DOCUMENT_ID_INDEX_NAME",
    "LOCAL_CORPUS_PATH_ENV",
    "LOCAL_CORPUS_SCHEMA_VERSION",
    "TRANSLATION_DOCUMENT_ID_INDEX_NAME",
    "LocalCorpusBuildError",
    "LocalCorpusConfigurationError",
    "LocalCorpusError",
    "LocalCorpusIndex",
    "LocalCorpusMetadata",
    "LocalCorpusReadError",
    "LocalCorpusValidationError",
    "LocalHuggingFaceCorpus",
    "build_from_parquet",
    "default_local_corpus_path",
]
