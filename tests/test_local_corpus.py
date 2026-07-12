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
    TRANSLATION_DOCUMENT_ID_INDEX_NAME,
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


def write_translation_parquet(
    path: Path,
    rows: list[tuple[str, str, str | None]],
) -> None:
    connection = duckdb.connect()
    try:
        connection.execute(
            """
            CREATE TABLE source (
                id VARCHAR,
                text VARCHAR,
                lang VARCHAR
            )
            """
        )
        if rows:
            connection.executemany("INSERT INTO source VALUES (?, ?, ?)", rows)
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

    def test_unified_index_keeps_original_text_and_exposes_english_report_text(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shards: dict[str, list[Path]] = {}
            original_rows = {
                "eng": ("doc-eng", "English original"),
                "spa": ("doc-spa", "Texto original"),
                "rus": ("doc-rus", "Исходный текст"),
                "zho": ("doc-zho", "原文"),
            }
            for split, (document_id, text) in original_rows.items():
                shard = root / f"{split}.parquet"
                write_parquet(shard, [(document_id, text, None, None)])
                shards[split] = [shard]

            translations: dict[str, list[Path]] = {}
            translated_rows = {
                "spa.mt.eng": ("doc-spa", "Spanish document in English"),
                "rus.mt.eng": ("doc-rus", "Russian document in English"),
                "zho.mt.eng": ("doc-zho", "Chinese document in English"),
            }
            for split, (document_id, text) in translated_rows.items():
                shard = root / f"{split}.parquet"
                write_translation_parquet(shard, [(document_id, text, "eng")])
                translations[split] = [shard]

            destination = build_from_parquet(
                shards,
                root / "corpus.duckdb",
                revision="revision-with-translations",
                translations=translations,
            )
            index = LocalCorpusIndex(destination, batch_size=2)
            result = index.enrich_documents(
                ["doc-spa", "missing", "doc-eng", "doc-zho", "doc-rus"]
            )

            self.assertEqual(
                [document.text for document in result.documents],
                [
                    "Texto original",
                    "English original",
                    "原文",
                    "Исходный текст",
                ],
            )
            self.assertEqual(
                [document.english_text for document in result.documents],
                [
                    "Spanish document in English",
                    None,
                    "Chinese document in English",
                    "Russian document in English",
                ],
            )
            self.assertEqual(
                [document.report_text for document in result.documents],
                [
                    "Spanish document in English",
                    "English original",
                    "Chinese document in English",
                    "Russian document in English",
                ],
            )
            self.assertEqual(result.unresolved_ids, ("missing",))
            self.assertEqual(result.missing_translation_ids, ())
            self.assertEqual(result.translation_unresolved_ids, ())
            self.assertEqual(index.metadata.translation_config, "mt_docs")
            self.assertEqual(index.metadata.translation_count, 3)
            self.assertEqual(index.metadata.translation_shard_count, 3)
            self.assertEqual(
                index.metadata.translation_splits,
                ("rus.mt.eng", "spa.mt.eng", "zho.mt.eng"),
            )
            self.assertEqual(index.verify_integrity(), index.metadata)

            connection = duckdb.connect(str(destination), read_only=True)
            try:
                row = connection.execute(
                    """
                    SELECT is_unique
                    FROM duckdb_indexes()
                    WHERE index_name = ?
                    """,
                    [TRANSLATION_DOCUMENT_ID_INDEX_NAME],
                ).fetchone()
                self.assertEqual(row, (True,))
            finally:
                connection.close()

    def test_missing_expected_translation_is_separate_from_unknown_base_id(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "spa.parquet"
            translated = root / "spa.mt.eng.parquet"
            write_parquet(
                original,
                [
                    ("doc-1", "uno", None, None),
                    ("doc-2", "dos", None, None),
                ],
            )
            write_translation_parquet(
                translated,
                [
                    ("doc-1", "one", "eng"),
                    ("doc-2", "two", "eng"),
                ],
            )
            destination = build_from_parquet(
                {"spa": [original]},
                root / "corpus.duckdb",
                revision="revision-1",
                translations={"spa.mt.eng": [translated]},
            )

            connection = duckdb.connect(str(destination))
            try:
                connection.execute(
                    "DELETE FROM translations WHERE document_id = 'doc-2'"
                )
                connection.execute("UPDATE corpus_metadata SET translation_count = 1")
            finally:
                connection.close()

            result = LocalCorpusIndex(destination).get_many(
                ["doc-2", "unknown", "doc-1"]
            )

            self.assertEqual(result.unresolved_ids, ("unknown",))
            self.assertEqual(result.missing_translation_ids, ("doc-2",))
            self.assertEqual(result.translation_unresolved_ids, ("doc-2",))
            self.assertEqual(result.documents[0].text, "dos")
            self.assertIsNone(result.documents[0].english_text)
            self.assertIsNone(result.documents[0].report_text)
            with self.assertRaisesRegex(
                LocalCorpusValidationError,
                "missing an expected English translation",
            ):
                LocalCorpusIndex(destination).verify_integrity()

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

    def test_translation_ids_must_be_unique_and_language_must_be_english(
        self,
    ) -> None:
        cases = (
            (
                "duplicate",
                [("doc-1", "one", "eng"), ("doc-1", "again", "eng")],
                "duplicate document IDs",
            ),
            ("language", [("doc-1", "uno", "spa")], "lang='eng'"),
        )
        for name, translated_rows, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                original = root / "spa.parquet"
                translated = root / "spa.mt.eng.parquet"
                destination = root / "corpus.duckdb"
                write_parquet(original, [("doc-1", "uno", None, None)])
                write_translation_parquet(translated, translated_rows)

                with self.assertRaisesRegex(LocalCorpusBuildError, message):
                    build_from_parquet(
                        {"spa": [original]},
                        destination,
                        revision="revision-1",
                        translations={"spa.mt.eng": [translated]},
                    )

                self.assertFalse(destination.exists())

    def test_translation_split_must_match_canonical_document_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spa = root / "spa.parquet"
            rus = root / "rus.parquet"
            spa_translation = root / "spa.mt.eng.parquet"
            rus_translation = root / "rus.mt.eng.parquet"
            destination = root / "corpus.duckdb"
            write_parquet(spa, [("doc-spa", "uno", None, None)])
            write_parquet(rus, [("doc-rus", "один", None, None)])
            write_translation_parquet(
                spa_translation,
                [("doc-rus", "Russian in English", "eng")],
            )
            write_translation_parquet(
                rus_translation,
                [("doc-spa", "Spanish in English", "eng")],
            )

            with self.assertRaisesRegex(
                LocalCorpusBuildError,
                "translation split does not match",
            ):
                build_from_parquet(
                    {"spa": [spa], "rus": [rus]},
                    destination,
                    revision="revision-1",
                    translations={
                        "spa.mt.eng": [spa_translation],
                        "rus.mt.eng": [rus_translation],
                    },
                )

            self.assertFalse(destination.exists())

    def test_translation_coverage_is_bidirectional(self) -> None:
        cases = (
            (
                "missing",
                [
                    ("doc-1", "uno", None, None),
                    ("doc-2", "dos", None, None),
                ],
                [("doc-1", "one", "eng")],
                "does not cover every canonical document",
            ),
            (
                "orphan",
                [("doc-1", "uno", None, None)],
                [
                    ("doc-1", "one", "eng"),
                    ("orphan", "orphan", "eng"),
                ],
                "without a canonical document",
            ),
        )
        for name, original_rows, translated_rows, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                original = root / "spa.parquet"
                translated = root / "spa.mt.eng.parquet"
                destination = root / "corpus.duckdb"
                write_parquet(original, original_rows)
                write_translation_parquet(translated, translated_rows)

                with self.assertRaisesRegex(LocalCorpusBuildError, message):
                    build_from_parquet(
                        {"spa": [original]},
                        destination,
                        revision="revision-1",
                        translations={"spa.mt.eng": [translated]},
                    )

                self.assertFalse(destination.exists())

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
            with self.assertRaisesRegex(
                LocalCorpusValidationError,
                "expected_translation_splits",
            ):
                LocalCorpusIndex(
                    destination,
                    expected_translation_splits=("spa.mt.eng",),
                )
            self.assertEqual(
                LocalCorpusIndex(
                    destination,
                    expected_translation_splits=(),
                ).metadata.translation_splits,
                (),
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
            with self.assertRaisesRegex(ValueError, "translation splits"):
                build_from_parquet(
                    {"spa": [root / "unused.parquet"]},
                    root / "unused.duckdb",
                    revision="revision-1",
                    translations={"rus.mt.eng": [root / "unused.parquet"]},
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
                    "RAGTIME_PROJECT_CACHE_DIR": str(
                        root / "cache" / "trec-ragtime-2026"
                    ),
                    "XDG_CACHE_HOME": "",
                },
                clear=False,
            ):
                self.assertEqual(
                    default_local_corpus_path(),
                    root / "cache" / "trec-ragtime-2026" / "ragtime2-corpus-v2.duckdb",
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
            translations = {}
            for split in ("eng", "spa", "rus", "zho"):
                shard = root / f"{split}.parquet"
                write_parquet(
                    shard,
                    [(f"doc-{split}", f"text {split}", None, None)],
                )
                shards[split] = [shard]
                if split != "eng":
                    translation_split = f"{split}.mt.eng"
                    translation_shard = root / f"{translation_split}.parquet"
                    write_translation_parquet(
                        translation_shard,
                        [(f"doc-{split}", f"English text {split}", "eng")],
                    )
                    translations[translation_split] = [translation_shard]
            destination = build_from_parquet(
                shards,
                root / "corpus.duckdb",
                revision="revision-1",
                translations=translations,
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
                "RAGTIME_PROJECT_CACHE_DIR": str(
                    Path(directory) / "empty-cache" / "trec-ragtime-2026"
                ),
                "XDG_CACHE_HOME": "",
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
