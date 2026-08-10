from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import duckdb

from trec_ragtime.artifact_store import ArtifactValidationError
from trec_ragtime.local_corpus import LocalCorpusIndex, build_from_parquet
from trec_ragtime.retrieval_cache import RetrievalArtifactCache
from trec_ragtime.retrieval_import import (
    RetrievalImportError,
    _read_jsonl,
    import_retrieval_run,
    main,
)


class MemoryArtifactStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.local: dict[str, bytes] = {}
        self.read_sources: list[str] = []
        self.bucket_id = "owner/private-cache"
        self.prefix = "retrieval-v1"
        self.namespace = "ragtime2-2026-v1"

    def read_validated(self, object_key: str, decoder: Any) -> Any | None:
        payload = self.local.get(object_key)
        source = "local"
        if payload is None:
            payload = self.objects.get(object_key)
            source = "remote"
        if payload is None:
            return None
        self.read_sources.append(source)
        return decoder(payload)

    def write(self, object_key: str, payload: bytes) -> None:
        self.objects[object_key] = payload
        self.local[object_key] = payload

    def evict_local(self, object_key: str) -> None:
        self.local.pop(object_key, None)


def _write_parquet(path: Path, rows: list[tuple[str, str, str | None, str | None]]) -> None:
    connection = duckdb.connect()
    try:
        connection.execute(
            "CREATE TABLE source (id VARCHAR, text VARCHAR, url VARCHAR, date VARCHAR)"
        )
        connection.executemany("INSERT INTO source VALUES (?, ?, ?, ?)", rows)
        connection.execute("COPY source TO ? (FORMAT PARQUET)", [str(path)])
    finally:
        connection.close()


class RetrievalImportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        source = self.root / "eng.parquet"
        _write_parquet(
            source,
            [
                ("doc-1", "one\x00exact", "https://example.test/1", "2026-01-01"),
                ("doc-2", "two", "https://example.test/2", "2026-01-02"),
                ("doc-3", "three", None, None),
            ],
        )
        self.corpus_path = self.root / "corpus.duckdb"
        build_from_parquet(
            {"eng": [source]},
            self.corpus_path,
            dataset="example/ragtime",
            revision="revision-1",
        )
        self.corpus = LocalCorpusIndex(self.corpus_path)
        self.store = MemoryArtifactStore()
        self.cache = RetrievalArtifactCache(self.store, clock=lambda: 123.0)

    def tearDown(self) -> None:
        self.corpus.close()
        self.tempdir.cleanup()

    def _write_control_run(self) -> Path:
        run = self.root / "control-run"
        run.mkdir()
        topics = self.root / "topics.jsonl"
        topics.write_text(
            json.dumps(
                {
                    "topic_id": "1",
                    "collection_id": "ragtime/2/all",
                    "problem_statement": "exact query",
                    "background": "unused",
                    "limit": 10,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": "control_a_v1",
            "run_id": "legacy-control",
            "topic_source": {
                "path": str(topics),
                "sha256": hashlib.sha256(topics.read_bytes()).hexdigest(),
            },
            "query_construction": "problem_only_v1",
            "requested_top_k": 100,
            "pipeline": "ragtime2",
            "retrieval": {
                "top_k": 100,
                "max_content_fallbacks": 0,
                "fallback_on_hf_error": False,
            },
            "corpus": {"backend": "local", "revision": "revision-1"},
            "topic_status": [
                {
                    "topic_id": "1",
                    "unresolved_ids": [],
                    "missing_translation_ids": [],
                }
            ],
        }
        (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (run / "candidates.jsonl").write_text(
            json.dumps(
                {
                    "topic_id": "1",
                    "query": "exact query",
                    "query_hash": hashlib.sha256(b"exact query").hexdigest(),
                    "rank": 1,
                    "document_id": "doc-1",
                    "score": 0.75,
                    "hydrated": True,
                    "text": "one\x00exact",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return run

    def _write_baseline_run(self) -> Path:
        run = self.root / "baseline-run"
        run.mkdir()
        plans = self.root / "query_plans.jsonl"
        plans.write_text(
            json.dumps(
                {
                    "topic_id": "1",
                    "plan_status": "complete",
                    "single_query": "single",
                    "facet_queries": ["facet one", "facet two"],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        plan_sha = hashlib.sha256(plans.read_bytes()).hexdigest()
        manifest = {
            "schema_version": "query_baseline_v1",
            "run_id": "legacy-c",
            "baseline": "c",
            "variant": "baseline-c-facet-rrf",
            "query_plan": {"path": str(plans), "sha256": plan_sha},
            "corpus": {
                "backend": "local",
                "revision": "revision-1",
                "schema_version": 2,
                "dataset": "example/ragtime",
                "config": "default",
                "document_count": 3,
                "shard_count": 1,
                "splits": ["eng"],
                "translation_config": "mt_docs",
                "translation_count": 0,
                "translation_shard_count": 0,
                "translation_splits": [],
            },
            "pipeline": "ragtime2",
            "top_k": 100,
            "rrf_k": 60,
            "query_construction": "problem_background_facets_v1",
        }
        (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        lines = [
            {
                "topic_id": "1",
                "query_index": 1,
                "query": "facet one",
                "pipeline": "ragtime2",
                "status": "success",
                "hit_count": 1,
                "hits": [{"rank": 1, "document_id": "doc-1", "score": 0.9}],
            },
            {
                "topic_id": "1",
                "query_index": 2,
                "query": "facet two",
                "pipeline": "ragtime2",
                "status": "success",
                "hit_count": 1,
                "hits": [{"rank": 1, "document_id": "doc-2", "score": 0.8}],
            },
        ]
        (run / "queries.jsonl").write_text(
            "".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8"
        )
        # This topic file intentionally omits doc-2: the importer must use the
        # successful facet checkpoints, not only the fused publication.
        (run / "topics.jsonl").write_text(
            json.dumps(
                {
                    "topic_id": "1",
                    "status": "success",
                    "candidates": [{"document_id": "doc-1"}],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return run

    def test_control_import_hydrates_exact_bytes_and_writes_sanitized_manifest(self) -> None:
        run = self._write_control_run()
        before = (run / "candidates.jsonl").read_bytes()
        output = self.root / "import-manifest.json"
        summary = import_retrieval_run(
            run,
            corpus=self.corpus,
            cache=self.cache,
            manifest_path=output,
            base_url="https://organizer.example.test",
        )
        self.assertEqual(summary.imported, 1)
        self.assertEqual(len(self.store.objects), 1)
        self.assertEqual((run / "candidates.jsonl").read_bytes(), before)
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
        manifest_text = output.read_text(encoding="utf-8")
        self.assertNotIn("exact query", manifest_text)
        self.assertNotIn("one\x00exact", manifest_text)
        self.assertNotIn("token", manifest_text.lower())
        self.assertEqual(self.store.read_sources[-1], "remote")

    def test_c_imports_facet_hit_missing_from_fused_topics_and_is_idempotent(self) -> None:
        run = self._write_baseline_run()
        output = self.root / "import-manifest.json"
        first = import_retrieval_run(
            run,
            corpus=self.corpus,
            cache=self.cache,
            manifest_path=output,
            base_url="https://organizer.example.test",
        )
        count = len(self.store.objects)
        second = import_retrieval_run(
            run,
            corpus=self.corpus,
            cache=self.cache,
            manifest_path=output,
            base_url="https://organizer.example.test",
        )
        self.assertEqual(first.imported, 2)
        self.assertEqual(second.imported, 2)
        self.assertEqual(len(self.store.objects), count)
        self.assertEqual(json.loads(output.read_text())['counts']['verified'], 2)
        self.assertGreaterEqual(self.store.read_sources.count("remote"), 2)

    def test_rejects_unsupported_schema_before_writing(self) -> None:
        run = self._write_control_run()
        manifest_path = run / "manifest.json"
        payload = json.loads(manifest_path.read_text())
        payload["schema_version"] = "unknown_v1"
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(RetrievalImportError):
            import_retrieval_run(
                run,
                corpus=self.corpus,
                cache=self.cache,
                manifest_path=self.root / "out.json",
                base_url="https://organizer.example.test",
            )
        self.assertEqual(self.store.objects, {})

    def test_rejects_noncanonical_namespace_before_reading_or_writing(self) -> None:
        run = self._write_control_run()
        self.store.namespace = "other-v2"
        with self.assertRaises(RetrievalImportError):
            import_retrieval_run(
                run,
                corpus=self.corpus,
                cache=self.cache,
                manifest_path=self.root / "out.json",
                base_url="https://organizer.example.test",
            )
        self.assertEqual(self.store.objects, {})

    def test_output_symlink_is_preflighted_before_corpus_or_cache_work(self) -> None:
        run = self._write_control_run()
        target = self.root / "unrelated.json"
        target.write_text("keep me", encoding="utf-8")
        output = self.root / "out.json"
        output.symlink_to(target)
        with patch.object(self.corpus, "verify_integrity", side_effect=AssertionError("corpus must not be opened")):
            with self.assertRaises(RetrievalImportError):
                import_retrieval_run(
                    run,
                    corpus=self.corpus,
                    cache=self.cache,
                    manifest_path=output,
                    base_url="https://organizer.example.test",
                )
        self.assertEqual(self.store.objects, {})
        self.assertEqual(target.read_text(encoding="utf-8"), "keep me")

    def test_rejects_unsupported_control_depth_before_writing(self) -> None:
        run = self._write_control_run()
        manifest_path = run / "manifest.json"
        payload = json.loads(manifest_path.read_text())
        payload["requested_top_k"] = 101
        payload["retrieval"]["top_k"] = 101
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(RetrievalImportError):
            import_retrieval_run(
                run,
                corpus=self.corpus,
                cache=self.cache,
                manifest_path=self.root / "out.json",
                base_url="https://organizer.example.test",
            )
        self.assertEqual(self.store.objects, {})

    def test_cli_rejects_namespace_and_noncanonical_corpus_without_writes(self) -> None:
        run = self._write_control_run()
        output = self.root / "cli-manifest.json"
        self.store.namespace = "other-v2"
        with patch(
            "trec_ragtime.retrieval_import.ArtifactStore.from_env",
            return_value=self.store,
        ), patch(
            "trec_ragtime.retrieval_import.LocalCorpusIndex",
            return_value=self.corpus,
        ):
            with self.assertRaises(RetrievalImportError):
                main(
                    [
                        str(run),
                        "--manifest",
                        str(output),
                        "--corpus",
                        str(self.corpus_path),
                        "--base-url",
                        "https://organizer.example.test",
                    ]
                )
        self.assertEqual(self.store.objects, {})

    def test_cli_constructs_canonical_local_corpus_expectations(self) -> None:
        run = self._write_control_run()
        output = self.root / "cli-manifest.json"
        captured: dict[str, Any] = {}

        def construct(*args: Any, **kwargs: Any) -> Any:
            captured.update(kwargs)
            raise RetrievalImportError("fixture corpus is intentionally rejected")

        with patch(
            "trec_ragtime.retrieval_import.ArtifactStore.from_env",
            return_value=self.store,
        ), patch(
            "trec_ragtime.retrieval_import.LocalCorpusIndex",
            side_effect=construct,
        ):
            with self.assertRaises(RetrievalImportError):
                main(
                    [
                        str(run),
                        "--manifest",
                        str(output),
                        "--corpus",
                        str(self.corpus_path),
                        "--base-url",
                        "https://organizer.example.test",
                    ]
                )
        self.assertEqual(captured["expected_dataset"], "trec-ragtime/ragtime2")
        self.assertEqual(captured["expected_splits"], ("eng", "spa", "rus", "zho"))
        self.assertEqual(
            captured["expected_translation_splits"],
            ("spa.mt.eng", "rus.mt.eng", "zho.mt.eng"),
        )
        self.assertEqual(self.store.objects, {})

    def test_baseline_promised_corpus_identity_mismatch_skips_without_writes(self) -> None:
        for field, value in (
            ("dataset", "wrong/dataset"),
            ("splits", ["spa"]),
            ("translation_splits", ["spa.mt.eng"]),
        ):
            run = self._write_baseline_run()
            renamed = self.root / f"baseline-{field}"
            run.rename(renamed)
            manifest_path = renamed / "manifest.json"
            payload = json.loads(manifest_path.read_text())
            payload["corpus"][field] = value
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.subTest(field=field), self.assertRaises(RetrievalImportError):
                import_retrieval_run(
                    renamed,
                    corpus=self.corpus,
                    cache=self.cache,
                    manifest_path=self.root / f"out-{field}.json",
                    base_url="https://organizer.example.test",
                )
            self.assertEqual(self.store.objects, {})

    def test_control_missing_backend_is_rejected_without_writes(self) -> None:
        run = self._write_control_run()
        manifest_path = run / "manifest.json"
        payload = json.loads(manifest_path.read_text())
        del payload["corpus"]["backend"]
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(RetrievalImportError):
            import_retrieval_run(
                run,
                corpus=self.corpus,
                cache=self.cache,
                manifest_path=self.root / "out.json",
                base_url="https://organizer.example.test",
            )
        self.assertEqual(self.store.objects, {})

    def test_jsonl_reader_is_incremental(self) -> None:
        run = self._write_control_run()
        source = type("Source", (), {"path": run / "candidates.jsonl"})()
        records = _read_jsonl(source)
        self.assertNotIsInstance(records, list)
        self.assertEqual(next(records)["document_id"], "doc-1")


if __name__ == "__main__":
    unittest.main()
