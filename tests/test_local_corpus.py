from __future__ import annotations

import os
import stat
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import duckdb

from trec_ragtime import HuggingFaceCorpus, RagtimeRetriever
from trec_ragtime.local_corpus import (
    DOCUMENT_ID_INDEX_NAME,
    LOCAL_CORPUS_PATH_ENV,
    LocalCorpusBuildError,
    LocalCorpusConfigurationError,
    LocalCorpusIndex,
    LocalCorpusReadError,
    LocalCorpusValidationError,
    build_from_parquet,
    default_local_corpus_path,
)


def write_parquet(
    path: Path,
    rows: list[tuple[str, str, str | None, str | None]],
) -> None:
    connection = duckdb.connect()
    try:
        connection.execute(
            """
            CREATE TABLE source (
                id VARCHAR,
                text VARCHAR,
                url VARCHAR,
                date VARCHAR
            )
            """
        )
        if rows:
            connection.executemany("INSERT INTO source VALUES (?, ?, ?, ?)", rows)
        connection.execute("COPY source TO ? (FORMAT PARQUET)", [str(path)])
    finally:
        connection.close()


class LocalCorpusIndexTests(unittest.TestCase):
    def test_builds_unique_index_and_enriches_in_requested_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            eng_one = root / "eng-000.parquet"
            eng_two = root / "eng-001.parquet"
            spa = root / "spa.parquet"
            write_parquet(
                eng_one,
                [("doc-1", "one", "https://example.test/1", "2026-01-01")],
            )
            write_parquet(eng_two, [("doc-3", "三", None, None)])
            write_parquet(
                spa,
                [("doc-2", "dos", "https://example.test/2", "2026-02-02")],
            )
            destination = root / "corpus.duckdb"

            result_path = build_from_parquet(
                {"spa": [spa], "eng": [eng_one, eng_two]},
                destination,
                dataset="example/ragtime",
                revision="0123456789abcdef",
            )

            self.assertEqual(result_path, destination)
            self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o600)
            index = LocalCorpusIndex(
                destination,
                batch_size=1,
                expected_dataset="example/ragtime",
                expected_revision="0123456789abcdef",
            )
            hydration = index.enrich_documents(
                ["doc-2", "missing", "doc-1", "doc-2", "doc-3"],
                refresh=True,
            )

            self.assertEqual(
                [document.document_id for document in hydration.documents],
                ["doc-2", "doc-1", "doc-3"],
            )
            self.assertEqual(
                [document.text for document in hydration.documents],
                ["dos", "one", "三"],
            )
            self.assertEqual(
                [document.split for document in hydration.documents],
                ["spa", "eng", "eng"],
            )
            self.assertEqual(hydration.documents[2].url, None)
            self.assertEqual(hydration.documents[2].date, None)
            self.assertTrue(
                all(
                    document.local_cache_hit is None for document in hydration.documents
                )
            )
            self.assertTrue(
                all(
                    document.source == "huggingface_local_index"
                    for document in hydration.documents
                )
            )
            self.assertEqual(hydration.unresolved_ids, ("missing",))
            self.assertEqual(hydration.failures, ())
            self.assertEqual(index.metadata.document_count, 3)
            self.assertEqual(index.metadata.shard_count, 3)
            self.assertEqual(index.metadata.splits, ("eng", "spa"))

            connection = duckdb.connect(str(destination), read_only=True)
            try:
                row = connection.execute(
                    """
                    SELECT is_unique
                    FROM duckdb_indexes()
                    WHERE index_name = ?
                    """,
                    [DOCUMENT_ID_INDEX_NAME],
                ).fetchone()
                self.assertEqual(row, (True,))
                with self.assertRaises(duckdb.Error):
                    connection.execute(
                        "INSERT INTO documents VALUES ('x', 'x', NULL, NULL, 'eng')"
                    )
            finally:
                connection.close()

    def test_large_lookup_is_chunked_and_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "eng.parquet"
            rows = [
                (f"doc-{number:04d}", f"text {number}", None, None)
                for number in range(257)
            ]
            write_parquet(shard, rows)
            destination = build_from_parquet(
                {"eng": [shard]},
                root / "corpus.duckdb",
                revision="revision-1",
            )
            requested = [row[0] for row in reversed(rows)]
            requested.extend(["missing", "doc-0007", "doc-0256"])

            result = LocalCorpusIndex(destination, batch_size=7).get_many(requested)

            self.assertEqual(
                [document.document_id for document in result.documents],
                [row[0] for row in reversed(rows)],
            )
            self.assertEqual(result.unresolved_ids, ("missing",))

    def test_each_lookup_uses_an_independent_read_only_connection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "eng.parquet"
            write_parquet(shard, [("doc-1", "one", None, None)])
            destination = build_from_parquet(
                {"eng": [shard]},
                root / "corpus.duckdb",
                revision="revision-1",
            )
            real_connect = duckdb.connect
            calls: list[dict[str, object]] = []

            def recording_connect(*args, **kwargs):
                calls.append(dict(kwargs))
                return real_connect(*args, **kwargs)

            with patch(
                "trec_ragtime.local_corpus.duckdb.connect",
                side_effect=recording_connect,
            ):
                index = LocalCorpusIndex(destination)
                index.get_many(["doc-1"])
                index.get_many(["doc-1"])

            self.assertEqual(len(calls), 3)
            self.assertTrue(all(call.get("read_only") is True for call in calls))

    def test_one_index_supports_concurrent_readers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "eng.parquet"
            write_parquet(
                shard,
                [(f"doc-{number}", str(number), None, None) for number in range(20)],
            )
            destination = build_from_parquet(
                {"eng": [shard]},
                root / "corpus.duckdb",
                revision="revision-1",
            )
            index = LocalCorpusIndex(destination, batch_size=3)

            def read(number: int) -> tuple[str, ...]:
                result = index.get_many([f"doc-{number}", f"doc-{(number + 1) % 20}"])
                return tuple(document.text for document in result.documents)

            with ThreadPoolExecutor(max_workers=8) as executor:
                values = list(executor.map(read, range(20)))

            self.assertEqual(values[0], ("0", "1"))
            self.assertEqual(values[19], ("19", "0"))

    def test_replacement_is_atomic_and_replace_must_be_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_shard = root / "old.parquet"
            new_shard = root / "new.parquet"
            write_parquet(old_shard, [("old", "old text", None, None)])
            write_parquet(new_shard, [("new", "new text", None, None)])
            destination = build_from_parquet(
                {"eng": [old_shard]},
                root / "corpus.duckdb",
                revision="old-revision",
            )
            old_connection = duckdb.connect(str(destination), read_only=True)
            try:
                with self.assertRaises(FileExistsError):
                    build_from_parquet(
                        {"eng": [new_shard]},
                        destination,
                        revision="new-revision",
                    )
                self.assertEqual(LocalCorpusIndex(destination).revision, "old-revision")

                build_from_parquet(
                    {"eng": [new_shard]},
                    destination,
                    revision="new-revision",
                    replace=True,
                )

                self.assertEqual(
                    old_connection.execute(
                        "SELECT document_id FROM documents"
                    ).fetchall(),
                    [("old",)],
                )
                old_connection.close()
                old_connection = None
                new_index = LocalCorpusIndex(destination)
                self.assertEqual(new_index.revision, "new-revision")
                self.assertEqual(
                    [
                        document.document_id
                        for document in new_index.get_many(["old", "new"]).documents
                    ],
                    ["new"],
                )
            finally:
                if old_connection is not None:
                    old_connection.close()

    def test_duplicate_ids_fail_without_publishing_partial_database(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            eng = root / "eng.parquet"
            spa = root / "spa.parquet"
            write_parquet(eng, [("duplicate", "one", None, None)])
            write_parquet(spa, [("duplicate", "dos", None, None)])
            destination = root / "corpus.duckdb"

            with self.assertRaisesRegex(
                LocalCorpusBuildError, "duplicate document IDs"
            ):
                build_from_parquet(
                    {"eng": [eng], "spa": [spa]},
                    destination,
                    revision="revision-1",
                )

            self.assertFalse(destination.exists())
            self.assertEqual(list(root.glob(f".{destination.name}.*.tmp*")), [])

    def test_post_publication_sync_failure_reports_published_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "eng.parquet"
            write_parquet(shard, [("doc-1", "one", None, None)])
            destination = root / "corpus.duckdb"

            with (
                patch(
                    "trec_ragtime.local_corpus._fsync_directory",
                    side_effect=OSError("unsupported"),
                ),
                self.assertRaisesRegex(
                    LocalCorpusBuildError,
                    "was published.*inspect the published index",
                ),
            ):
                build_from_parquet(
                    {"eng": [shard]},
                    destination,
                    revision="revision-1",
                )

            self.assertTrue(destination.is_file())
            self.assertEqual(LocalCorpusIndex(destination).revision, "revision-1")

    def test_invalid_database_and_expected_identity_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "eng.parquet"
            write_parquet(shard, [("doc-1", "one", None, None)])
            destination = build_from_parquet(
                {"eng": [shard]},
                root / "corpus.duckdb",
                dataset="example/ragtime",
                revision="revision-1",
            )

            with self.assertRaisesRegex(LocalCorpusValidationError, "expected_dataset"):
                LocalCorpusIndex(destination, expected_dataset="other/dataset")
            with self.assertRaisesRegex(
                LocalCorpusValidationError, "expected_revision"
            ):
                LocalCorpusIndex(destination, expected_revision="revision-2")
            with self.assertRaisesRegex(LocalCorpusValidationError, "expected_splits"):
                LocalCorpusIndex(
                    destination,
                    expected_splits=("eng", "spa", "rus", "zho"),
                )

            connection = duckdb.connect(str(destination))
            try:
                connection.execute("UPDATE corpus_metadata SET document_count = 999")
            finally:
                connection.close()
            with self.assertRaisesRegex(
                LocalCorpusValidationError, "row count does not match"
            ):
                LocalCorpusIndex(destination)

    def test_configuration_and_request_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(LocalCorpusConfigurationError):
                LocalCorpusIndex(root / "missing.duckdb")
            with self.assertRaisesRegex(ValueError, "non-empty mapping"):
                build_from_parquet({}, root / "unused.duckdb", revision="revision-1")
            with self.assertRaisesRegex(ValueError, "iterable of paths"):
                build_from_parquet(
                    {"eng": "one.parquet"},
                    root / "unused.duckdb",
                    revision="revision-1",
                )
            with self.assertRaisesRegex(ValueError, "original-language"):
                build_from_parquet(
                    {"mt_docs": [root / "unused.parquet"]},
                    root / "unused.duckdb",
                    revision="revision-1",
                )

            shard = root / "eng.parquet"
            write_parquet(shard, [("doc-1", "one", None, None)])
            destination = build_from_parquet(
                {"eng": [shard]},
                root / "corpus.duckdb",
                revision="revision-1",
            )
            index = LocalCorpusIndex(destination)
            for invalid_ids in ([""], [123], ["doc-1", None]):
                with self.subTest(invalid_ids=invalid_ids):
                    with self.assertRaisesRegex(ValueError, "document IDs"):
                        index.get_many(invalid_ids)
            with self.assertRaisesRegex(ValueError, "refresh must be a boolean"):
                index.get_many(["doc-1"], refresh=1)
            self.assertEqual(index.get_many([]).documents, ())

    def test_default_path_is_shared_across_worktrees_and_honors_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(
                os.environ,
                {
                    LOCAL_CORPUS_PATH_ENV: "",
                    "XDG_CACHE_HOME": str(root / "cache"),
                },
                clear=False,
            ):
                self.assertEqual(
                    default_local_corpus_path(),
                    root / "cache" / "trec-ragtime-2026" / "ragtime2-default-v1.duckdb",
                )

            explicit = root / "shared" / "snapshot.duckdb"
            with patch.dict(
                os.environ,
                {LOCAL_CORPUS_PATH_ENV: str(explicit)},
                clear=False,
            ):
                self.assertEqual(default_local_corpus_path(), explicit)

            with patch.dict(
                os.environ,
                {LOCAL_CORPUS_PATH_ENV: "relative.duckdb"},
                clear=False,
            ):
                with self.assertRaisesRegex(
                    LocalCorpusConfigurationError, "absolute path"
                ):
                    default_local_corpus_path()

    def test_retriever_auto_prefers_existing_local_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shards = {}
            for split in ("eng", "spa", "rus", "zho"):
                shard = root / f"{split}.parquet"
                write_parquet(
                    shard,
                    [(f"doc-{split}", f"text {split}", None, None)],
                )
                shards[split] = [shard]
            destination = build_from_parquet(
                shards,
                root / "corpus.duckdb",
                revision="revision-1",
            )
            environment = {
                "RAGTIME_BASE_URL": "https://api.example",
                "RAGTIME_BEARER_TOKEN": "secret",
                LOCAL_CORPUS_PATH_ENV: str(destination),
                "RAGTIME_DOCUMENT_BACKEND": "auto",
            }
            with patch.dict(os.environ, environment, clear=False):
                with RagtimeRetriever.from_env(use_cache=False) as retriever:
                    self.assertIsInstance(retriever.corpus, LocalCorpusIndex)
                    self.assertEqual(retriever.document_backend, "local")
                    self.assertEqual(retriever.corpus_path, destination)

                with RagtimeRetriever.from_env(
                    document_backend="viewer",
                    use_cache=False,
                ) as retriever:
                    self.assertIsInstance(retriever.corpus, HuggingFaceCorpus)
                    self.assertEqual(retriever.document_backend, "viewer")
                    self.assertIsNone(retriever.corpus_path)

    def test_retriever_auto_uses_viewer_only_when_default_index_is_absent(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment = {
                "RAGTIME_BASE_URL": "https://api.example",
                "RAGTIME_BEARER_TOKEN": "secret",
                "RAGTIME_DOCUMENT_BACKEND": "auto",
                "RAGTIME_LOCAL_CORPUS_PATH": "",
                "XDG_CACHE_HOME": str(Path(directory) / "empty-cache"),
            }
            with patch.dict(os.environ, environment, clear=False):
                with RagtimeRetriever.from_env(use_cache=False) as retriever:
                    self.assertIsInstance(retriever.corpus, HuggingFaceCorpus)
                    self.assertEqual(retriever.document_backend, "viewer")
                with RagtimeRetriever.from_env(
                    corpus_options={"splits": ("eng",)},
                    use_cache=False,
                ) as retriever:
                    self.assertIsInstance(retriever.corpus, HuggingFaceCorpus)

    def test_local_backend_missing_or_corrupt_index_fails_without_viewer_fallback(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "missing.duckdb"
            environment = {
                "RAGTIME_BASE_URL": "https://api.example",
                "RAGTIME_BEARER_TOKEN": "secret",
                LOCAL_CORPUS_PATH_ENV: str(missing),
            }
            with patch.dict(os.environ, environment, clear=False):
                with self.assertRaisesRegex(
                    LocalCorpusConfigurationError,
                    "ragtime-corpus build",
                ):
                    RagtimeRetriever.from_env(
                        document_backend="local",
                        use_cache=False,
                    )
                with self.assertRaisesRegex(
                    LocalCorpusConfigurationError,
                    "ragtime-corpus build",
                ):
                    RagtimeRetriever.from_env(
                        document_backend="auto",
                        corpus_options={"splits": ("eng",)},
                        use_cache=False,
                    )
                with self.assertRaisesRegex(
                    LocalCorpusConfigurationError,
                    "ragtime-corpus build",
                ):
                    RagtimeRetriever.from_env(
                        document_backend="auto",
                        use_cache=False,
                    )

                corrupt = root / "corrupt.duckdb"
                corrupt.write_text("not a DuckDB database")
                with self.assertRaises(LocalCorpusReadError):
                    RagtimeRetriever.from_env(
                        document_backend="auto",
                        local_index_path=corrupt,
                        use_cache=False,
                    )

                shard = root / "eng.parquet"
                write_parquet(shard, [("doc-1", "one", None, None)])
                partial = build_from_parquet(
                    {"eng": [shard]},
                    root / "partial.duckdb",
                    revision="revision-1",
                )
                with self.assertRaisesRegex(
                    LocalCorpusValidationError,
                    "expected_splits",
                ):
                    RagtimeRetriever.from_env(
                        document_backend="auto",
                        local_index_path=partial,
                        use_cache=False,
                    )


if __name__ == "__main__":
    unittest.main()
