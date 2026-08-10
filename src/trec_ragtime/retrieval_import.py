"""Offline migration of immutable Control-A and baseline retrieval runs.

This module intentionally has no participant client and no Dataset Viewer path.
It reads the already-written run files, hydrates their exact hit IDs from the
revision-pinned local corpus, and stores one typed retrieval object per query.
The mutable JSONL run remains an audit input; the MessagePack object is the
durable shared cache consumed by the runners.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import tempfile
from urllib.parse import urlsplit
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .artifact_store import ArtifactStore
from .cache import DEFAULT_CACHE_NAMESPACE, sanitized_cache_configuration
from .huggingface import (
    DEFAULT_HF_DATASET,
    DEFAULT_HF_MT_SPLITS,
    DEFAULT_HF_SPLITS,
)
from .local_corpus import LocalCorpusIndex
from .models import PipelineResult, RetrievalResult, RetrievedDocument, SearchHit
from .query_baselines import load_query_plans
from .retrieval_cache import (
    ENDPOINT_DEPTH_BEHAVIOR_VERSION,
    RetrievalArtifactCache,
    RetrievalArtifactReference,
    RetrievalRequestIdentity,
)
from .topic_candidates import load_topics

IMPORTER_SCHEMA_VERSION = 1
CONTROL_A_SCHEMA = "control_a_v1"
QUERY_BASELINE_SCHEMA = "query_baseline_v1"
PIPELINE = "ragtime2"
CONTENT_COLLECTION = "ragtime2"
CONTROL_CONTEXT = {
    "kind": "control_a",
    "query_construction": "problem_only_v1",
}
BASELINE_CONSTRUCTIONS = {
    "b": "problem_background_single_query_v1",
    "c": "problem_background_facets_v1",
}


class RetrievalImportError(RuntimeError):
    """Raised when an immutable run cannot be safely interpreted."""


@dataclass(frozen=True, slots=True)
class _SourceFile:
    label: str
    path: Path
    sha256: str
    size: int


@dataclass(frozen=True, slots=True)
class RetrievalImportSummary:
    """Sanitized import counts and source/object provenance."""

    run_schema: str
    source_files: Mapping[str, Mapping[str, object]]
    imported: int
    verified: int
    skipped: int
    skipped_reasons: Mapping[str, int]
    object_keys: tuple[str, ...]

    @property
    def counts(self) -> dict[str, int]:
        return {
            "imported": self.imported,
            "verified": self.verified,
            "skipped": self.skipped,
        }


def _regular_file(path: Path, *, label: str) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        raise RetrievalImportError(f"{label} path must be absolute")
    try:
        status = path.lstat()
    except OSError:
        raise RetrievalImportError(f"{label} path could not be inspected") from None
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise RetrievalImportError(f"{label} path must be a regular non-symlink file")
    return path


def _run_directory(path: Path) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise RetrievalImportError("run directory must be an absolute regular directory")
    return path.resolve()


def _file_digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
    except OSError:
        raise RetrievalImportError("source input could not be read") from None
    return digest.hexdigest(), size


def _source_file(label: str, path: Path) -> _SourceFile:
    path = _regular_file(path, label=label)
    sha256, size = _file_digest(path)
    return _SourceFile(label=label, path=path, sha256=sha256, size=size)


def _read_json(source: _SourceFile) -> dict[str, Any]:
    try:
        payload = json.loads(source.path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise RetrievalImportError("source manifest is not valid UTF-8 JSON") from None
    if not isinstance(payload, dict):
        raise RetrievalImportError("source manifest must be a JSON object")
    return payload


def _read_jsonl(source: _SourceFile) -> Iterator[dict[str, Any]]:
    try:
        with source.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise RetrievalImportError("source JSONL contains a blank line")
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    raise RetrievalImportError("source JSONL contains invalid JSON") from None
                if not isinstance(payload, dict):
                    raise RetrievalImportError("source JSONL records must be objects")
                yield payload
    except UnicodeError:
        raise RetrievalImportError("source JSONL is not valid UTF-8") from None


def _validate_cache_namespace(cache: RetrievalArtifactCache) -> None:
    namespace = getattr(cache.store, "namespace", None)
    if namespace != DEFAULT_CACHE_NAMESPACE:
        raise RetrievalImportError("retrieval artifact cache namespace is not canonical")


def _validate_external_source(
    field: Mapping[str, Any], *, label: str, sources: list[_SourceFile]
) -> Path:
    path_value = field.get("path")
    expected_hash = field.get("sha256")
    if not isinstance(path_value, str) or not isinstance(expected_hash, str):
        raise RetrievalImportError(f"{label} source provenance is incomplete")
    path = _regular_file(Path(path_value), label=label)
    actual_hash, _size = _file_digest(path)
    if actual_hash != expected_hash:
        raise RetrievalImportError(f"{label} source hash does not match its manifest")
    sources.append(_source_file(label, path))
    return path


def _sha256_query(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def _finite_score(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("score is not finite")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("score is not finite")
    return value


def _hits(raw_hits: Any) -> tuple[SearchHit, ...]:
    if not isinstance(raw_hits, list):
        raise ValueError("hits must be a list")
    hits: list[SearchHit] = []
    seen_ids: set[str] = set()
    for expected_rank, raw_hit in enumerate(raw_hits, start=1):
        if not isinstance(raw_hit, Mapping):
            raise ValueError("hit must be an object")
        rank = raw_hit.get("rank")
        document_id = raw_hit.get("document_id")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank != expected_rank:
            raise ValueError("hit ranks must be contiguous and one-based")
        if not isinstance(document_id, str) or not document_id or document_id in seen_ids:
            raise ValueError("hit document IDs must be unique non-empty strings")
        seen_ids.add(document_id)
        hits.append(
            SearchHit(
                rank=rank,
                document_id=document_id,
                score=_finite_score(raw_hit.get("score")),
            )
        )
    return tuple(hits)


def _corpus_identity(corpus: Any) -> dict[str, Any]:
    verify = getattr(corpus, "verify_integrity", None)
    metadata = verify() if callable(verify) else getattr(corpus, "metadata", None)
    if metadata is None:
        raise RetrievalImportError("local corpus does not expose validated metadata")
    required = (
        "schema_version",
        "dataset",
        "config",
        "revision",
        "document_count",
        "shard_count",
        "splits",
        "translation_config",
        "translation_count",
        "translation_shard_count",
        "translation_splits",
    )
    if any(not hasattr(metadata, key) for key in required):
        raise RetrievalImportError("local corpus metadata is incomplete")
    path = getattr(corpus, "path", None)
    if path is None:
        raise RetrievalImportError("local corpus path is unavailable")
    return {
        "backend": "local",
        "path": str(Path(path).resolve()),
        "schema_version": metadata.schema_version,
        "dataset": metadata.dataset,
        "config": metadata.config,
        "revision": metadata.revision,
        "document_count": metadata.document_count,
        "shard_count": metadata.shard_count,
        "splits": list(metadata.splits),
        "translation_config": metadata.translation_config,
        "translation_count": metadata.translation_count,
        "translation_shard_count": metadata.translation_shard_count,
        "translation_splits": list(metadata.translation_splits),
    }


def _check_manifest_corpus(
    manifest: Mapping[str, Any],
    identity: Mapping[str, Any],
    *,
    schema: str,
) -> None:
    expected = manifest.get("corpus")
    if not isinstance(expected, Mapping):
        raise RetrievalImportError("run manifest has no corpus identity")
    if schema == CONTROL_A_SCHEMA:
        # Pre-fc7b3e2 Control-A v1 manifests recorded only path/revision. The
        # importer has already received a validated LocalCorpusIndex, so an
        # absent backend is the one supported legacy spelling of "local".
        if "backend" in expected and expected["backend"] != "local":
            raise RetrievalImportError("run manifest does not describe a local corpus")
    elif expected.get("backend") != "local":
        raise RetrievalImportError("run manifest does not describe a local corpus")
    # Control-A v1 predates the backend field; the validated LocalCorpusIndex
    # supplied by the importer makes that legacy omission unambiguous. Baseline
    # manifests are newer and must continue to carry the complete identity.
    required = {"revision"}
    if schema == QUERY_BASELINE_SCHEMA:
        required.update(
            {
                "backend",
                "schema_version",
                "dataset",
                "config",
                "document_count",
                "shard_count",
                "splits",
                "translation_config",
                "translation_count",
                "translation_shard_count",
                "translation_splits",
            }
        )
    missing = sorted(key for key in required if key not in expected)
    if missing:
        raise RetrievalImportError("run corpus identity is missing promised fields")
    for key in identity:
        if key == "path" or key not in expected:
            continue
        if expected[key] != identity[key]:
            raise RetrievalImportError("run corpus identity does not match the pinned index")


def _build_identity(
    *,
    query: str,
    top_k: int,
    query_context: Mapping[str, object],
    base_url: str,
    corpus_identity: Mapping[str, Any],
) -> RetrievalRequestIdentity:
    return RetrievalRequestIdentity(
        query=query,
        pipeline=PIPELINE,
        top_k=top_k,
        endpoint_depth_version=ENDPOINT_DEPTH_BEHAVIOR_VERSION,
        max_content_fallbacks=0,
        fallback_on_hf_error=False,
        content_collection=CONTENT_COLLECTION,
        backend_policy={"document_backend": "local"},
        query_context=query_context,
        base_url=base_url,
        corpus_identity=corpus_identity,
    )


def _validate_base_url(value: str) -> str:
    if not isinstance(value, str):
        raise RetrievalImportError("base_url is required to reconstruct retrieval identity")
    normalized = value.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if (
        not normalized
        or parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise RetrievalImportError("base_url must be an absolute HTTP(S) URL without credentials")
    return normalized


def _hydrated_result(
    *,
    query: str,
    hits: tuple[SearchHit, ...],
    corpus: Any,
    local_cache_hit: bool = False,
    remote_cached: bool | None = None,
) -> RetrievalResult:
    if not isinstance(local_cache_hit, bool) or (
        remote_cached is not None and not isinstance(remote_cached, bool)
    ):
        raise ValueError("legacy cache provenance is malformed")
    enrich = getattr(corpus, "enrich_documents", None)
    if not callable(enrich):
        raise ValueError("import requires LocalCorpusIndex.enrich_documents")
    hydration = enrich([hit.document_id for hit in hits])
    if hydration.failures or hydration.unresolved_ids:
        raise ValueError("exact hit IDs were not completely hydrated")
    by_id = hydration.by_id()
    if set(by_id) != {hit.document_id for hit in hits}:
        raise ValueError("local corpus returned an unexpected document set")
    documents = tuple(
        RetrievedDocument(
            rank=hit.rank,
            document_id=hit.document_id,
            score=hit.score,
            text=by_id[hit.document_id].text,
            url=by_id[hit.document_id].url,
            date=by_id[hit.document_id].date,
            language=by_id[hit.document_id].language,
            content_source=by_id[hit.document_id].source,
            content_local_cache_hit=by_id[hit.document_id].local_cache_hit,
            english_text=by_id[hit.document_id].english_text,
        )
        for hit in hits
    )
    return RetrievalResult(
        search_result=PipelineResult(
            query=query,
            pipeline=PIPELINE,
            hits=hits,
            processed=None,
            local_cache_hit=local_cache_hit,
            remote_cached=remote_cached,
            server_timestamp=None,
            cache_stored_at=None,
        ),
        documents=documents,
        missing_translation_ids=tuple(hydration.missing_translation_ids),
    )


def _same_result(left: RetrievalResult, right: RetrievalResult) -> bool:
    return (
        left.search_result == right.search_result
        and left.documents == right.documents
        and left.hydration_failures == right.hydration_failures
        and left.missing_translation_ids == right.missing_translation_ids
    )


def _persist_and_verify(
    cache: RetrievalArtifactCache,
    identity: RetrievalRequestIdentity,
    result: RetrievalResult,
) -> RetrievalArtifactReference:
    stored = cache.put(identity, result)
    reference = stored.retrieval_artifact
    if not isinstance(reference, RetrievalArtifactReference):
        raise RetrievalImportError("cache write did not return a retrieval reference")
    store = cache.store
    evict = getattr(store, "evict_local", None)
    if not callable(evict):
        raise RetrievalImportError("artifact store cannot independently verify L2")
    evict(identity.object_key)
    reread = cache.get(identity)
    if reread is None or not _same_result(stored, reread):
        raise RetrievalImportError("retrieval artifact failed independent L2 verification")
    reread_reference = reread.retrieval_artifact
    if not isinstance(reread_reference, RetrievalArtifactReference):
        raise RetrievalImportError("retrieval artifact reread has no reference")
    return reread_reference


def _reference_matches(record: Mapping[str, Any], identity: RetrievalRequestIdentity) -> bool:
    reference = record.get("retrieval_artifact")
    if not isinstance(reference, Mapping):
        return True
    return (
        reference.get("object_key") == identity.object_key
        and reference.get("request_digest") == identity.request_digest_hex
        and reference.get("retriever_fingerprint") == identity.retriever_fingerprint_hex
    )


def _validate_control_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != CONTROL_A_SCHEMA:
        raise RetrievalImportError("unsupported retrieval run schema")
    if manifest.get("pipeline") != PIPELINE:
        raise RetrievalImportError("unsupported Control-A pipeline")
    if manifest.get("query_construction") != "problem_only_v1":
        raise RetrievalImportError("unsupported Control-A query construction")
    top_k = manifest.get("requested_top_k")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
        raise RetrievalImportError("Control-A requested_top_k must be in the range 1..100")
    retrieval = manifest.get("retrieval")
    if not isinstance(retrieval, Mapping) or retrieval.get("max_content_fallbacks") != 0:
        raise RetrievalImportError("Control-A run is not a no-fallback retrieval artifact")
    if retrieval.get("top_k") != manifest.get("requested_top_k"):
        raise RetrievalImportError("Control-A depth is inconsistent in the run manifest")
    if retrieval.get("pipeline", PIPELINE) != PIPELINE:
        raise RetrievalImportError("Control-A retrieval pipeline is inconsistent")
    if retrieval.get("fallback_on_hf_error") is not False:
        raise RetrievalImportError("Control-A fallback policy is not canonical")


def _validate_baseline_manifest(manifest: Mapping[str, Any]) -> str:
    if manifest.get("schema_version") != QUERY_BASELINE_SCHEMA:
        raise RetrievalImportError("unsupported retrieval run schema")
    baseline = manifest.get("baseline")
    if baseline not in BASELINE_CONSTRUCTIONS:
        raise RetrievalImportError("baseline run must be B or C")
    if manifest.get("pipeline") != PIPELINE or manifest.get("top_k") != 100:
        raise RetrievalImportError("baseline retrieval configuration is not canonical")
    if manifest.get("rrf_k") != 60:
        raise RetrievalImportError("baseline fusion configuration is not canonical")
    if manifest.get("query_construction") != BASELINE_CONSTRUCTIONS[baseline]:
        raise RetrievalImportError("baseline query construction is not canonical")
    return baseline


def _prepare_sources(
    run: Path,
    manifest: Mapping[str, Any],
    schema: str,
    *,
    manifest_source: _SourceFile,
) -> tuple[list[_SourceFile], dict[str, Any]]:
    sources = [manifest_source]
    if schema == CONTROL_A_SCHEMA:
        sources.append(_source_file("candidates.jsonl", run / "candidates.jsonl"))
        field = manifest.get("topic_source")
        if not isinstance(field, Mapping):
            raise RetrievalImportError("Control-A topic source provenance is missing")
        _validate_external_source(field, label="topic_source", sources=sources)
    else:
        sources.append(_source_file("queries.jsonl", run / "queries.jsonl"))
        if (run / "topics.jsonl").exists():
            sources.append(_source_file("topics.jsonl", run / "topics.jsonl"))
        field = manifest.get("query_plan")
        if not isinstance(field, Mapping):
            raise RetrievalImportError("baseline query-plan provenance is missing")
        plan_path = _validate_external_source(field, label="query_plan", sources=sources)
        actual, _size = _file_digest(plan_path)
        if actual != field.get("sha256"):
            raise RetrievalImportError("query plan hash does not match its manifest")
    return sources, manifest


def _check_sources_unchanged(sources: Sequence[_SourceFile]) -> None:
    for source in sources:
        current_hash, current_size = _file_digest(source.path)
        if current_hash != source.sha256 or current_size != source.size:
            raise RetrievalImportError("source input changed during retrieval import")


def _source_manifest(sources: Sequence[_SourceFile]) -> dict[str, dict[str, object]]:
    return {
        source.label: {"sha256": source.sha256, "size": source.size}
        for source in sources
    }


def _preflight_manifest_path(path: Path, sources: Sequence[_SourceFile]) -> Path:
    """Reject every output/source alias before corpus or cache work begins."""

    path = Path(path).expanduser()
    if not path.is_absolute():
        raise RetrievalImportError("migration manifest path must be absolute")
    if path.is_symlink():
        raise RetrievalImportError("migration manifest path must not be a symlink")
    try:
        output_resolved = path.resolve(strict=False)
    except (OSError, RuntimeError):
        raise RetrievalImportError("migration manifest path could not be resolved") from None
    for source in sources:
        try:
            if output_resolved == source.path.resolve(strict=True):
                raise RetrievalImportError("migration manifest must not replace a source input")
            if path.exists() and os.path.samestat(path.stat(), source.path.stat()):
                raise RetrievalImportError("migration manifest must not alias a source input")
        except (OSError, RuntimeError):
            raise RetrievalImportError("migration manifest and source could not be compared") from None
    return path


def _safe_cache_metadata(cache: RetrievalArtifactCache) -> dict[str, str]:
    try:
        store_namespace = getattr(cache.store, "namespace", None)
    except Exception:
        raise RetrievalImportError("artifact destination metadata is unsafe") from None
    if store_namespace != DEFAULT_CACHE_NAMESPACE:
        raise RetrievalImportError("artifact destination metadata is unsafe")
    try:
        output = sanitized_cache_configuration(store_namespace, cache.store)
    except (TypeError, ValueError):
        raise RetrievalImportError("artifact destination metadata is unsafe") from None
    if output.get("namespace") != DEFAULT_CACHE_NAMESPACE:
        raise RetrievalImportError("artifact destination metadata is unsafe")
    bucket_id = output.get("bucket_id")
    if not isinstance(bucket_id, str) or not bucket_id:
        raise RetrievalImportError("artifact destination metadata is unsafe")
    return {
        key: output[key]
        for key in ("namespace", "bucket_id", "prefix")
        if key in output and isinstance(output[key], str)
    }


def _write_manifest(path: Path, payload: Mapping[str, Any], *, sources: Sequence[_SourceFile]) -> None:
    path = _preflight_manifest_path(path, sources)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            os.fchmod(handle.fileno(), 0o600)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _control_records(
    run: Path,
    manifest: Mapping[str, Any],
    sources: Sequence[_SourceFile],
    *,
    corpus: Any,
    cache: RetrievalArtifactCache,
    base_url: str,
    corpus_identity: Mapping[str, Any],
    objects: list[str],
    skipped_reasons: Counter[str],
) -> tuple[int, int, int]:
    candidate_source = next(source for source in sources if source.label == "candidates.jsonl")
    topic_source = next(source for source in sources if source.label == "topic_source")
    topics = load_topics(topic_source.path)
    topic_queries = {topic.topic_id: topic.problem_statement for topic in topics}
    statuses = manifest.get("topic_status")
    if not isinstance(statuses, list):
        raise RetrievalImportError("Control-A topic status list is missing")
    status_by_topic: dict[str, Mapping[str, Any]] = {}
    for status in statuses:
        if not isinstance(status, Mapping) or not isinstance(status.get("topic_id"), str):
            raise RetrievalImportError("Control-A topic status is malformed")
        topic_id = status["topic_id"]
        if topic_id in status_by_topic:
            raise RetrievalImportError("Control-A topic statuses contain duplicates")
        status_by_topic[topic_id] = status
    # Validate the complete stream before any artifact write. The subsequent
    # per-topic passes keep only one bounded candidate group in memory.
    invalid_record_count = 0
    for record in _read_jsonl(candidate_source):
        topic_id = record.get("topic_id")
        if not isinstance(topic_id, str) or topic_id not in topic_queries:
            skipped_reasons["invalid_control_topic"] += 1
            invalid_record_count += 1
    imported = verified = 0
    skipped = invalid_record_count
    top_k = manifest.get("requested_top_k")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise RetrievalImportError("Control-A top_k is invalid")
    for topic_id, query in topic_queries.items():
        status = status_by_topic.get(topic_id)
        records: list[Mapping[str, Any]] = []
        for record in _read_jsonl(candidate_source):
            if record.get("topic_id") == topic_id:
                records.append(record)
                if len(records) > top_k:
                    break
        if status is None or status.get("error") or status.get("unresolved_ids"):
            skipped += 1
            skipped_reasons["failed_or_partial_record"] += 1
            continue
        if len(records) > top_k:
            skipped += 1
            skipped_reasons["record_rejected"] += 1
            continue
        try:
            hits: list[dict[str, Any]] = []
            for record in records:
                if record.get("query") != query or record.get("query_hash") != _sha256_query(query):
                    raise ValueError("query identity mismatch")
                if record.get("hydrated") is False:
                    raise ValueError("legacy candidate is not hydrated")
                hits.append(record)
            hits.sort(key=lambda record: record.get("rank", -1))
            hit_records = [
                {
                    "rank": record.get("rank"),
                    "document_id": record.get("document_id"),
                    "score": record.get("score"),
                }
                for record in hits
            ]
            search_hits = _hits(hit_records)
            identity = _build_identity(
                query=query,
                top_k=top_k,
                query_context=CONTROL_CONTEXT,
                base_url=base_url,
                corpus_identity=corpus_identity,
            )
            if not all(_reference_matches(record, identity) for record in records):
                raise ValueError("retrieval reference mismatch")
            local_cache_hit = records[0].get("local_cache_hit", False) if records else False
            remote_cached = records[0].get("remote_cached") if records else None
            if any(
                record.get("local_cache_hit", local_cache_hit) != local_cache_hit
                or record.get("remote_cached") != remote_cached
                for record in records
            ):
                raise ValueError("legacy cache provenance is inconsistent")
            result = _hydrated_result(
                query=query,
                hits=search_hits,
                corpus=corpus,
                local_cache_hit=local_cache_hit,
                remote_cached=remote_cached,
            )
            for record, document in zip(hits, result.documents, strict=True):
                if record.get("text") is not None and record.get("text") != document.text:
                    raise ValueError("legacy candidate text disagrees with corpus")
            expected_missing = status.get("missing_translation_ids", [])
            if not isinstance(expected_missing, list) or set(expected_missing) != set(result.missing_translation_ids):
                raise ValueError("translation status disagrees with local corpus")
        except (TypeError, ValueError, RetrievalImportError):
            skipped += 1
            skipped_reasons["record_rejected"] += 1
            continue
        _persist_and_verify(cache, identity, result)
        objects.append(identity.object_key)
        imported += 1
        verified += 1
    return imported, verified, skipped


def _baseline_records(
    manifest: Mapping[str, Any],
    sources: Sequence[_SourceFile],
    *,
    corpus: Any,
    cache: RetrievalArtifactCache,
    base_url: str,
    corpus_identity: Mapping[str, Any],
    objects: list[str],
    skipped_reasons: Counter[str],
) -> tuple[int, int, int]:
    baseline = _validate_baseline_manifest(manifest)
    plan_source = next(source for source in sources if source.label == "query_plan")
    query_source = next(source for source in sources if source.label == "queries.jsonl")
    plans = load_query_plans(plan_source.path)
    plan_by_topic = {plan.topic_id: plan for plan in plans}
    plan_hash = _file_digest(plan_source.path)[0]
    if plan_hash != manifest["query_plan"].get("sha256"):
        raise RetrievalImportError("query plan hash does not match its manifest")
    records = _read_jsonl(query_source)
    seen: set[tuple[str, int]] = set()
    imported = verified = skipped = 0
    for record in records:
        topic_id = record.get("topic_id")
        query_index = record.get("query_index")
        key = (topic_id, query_index) if isinstance(topic_id, str) and isinstance(query_index, int) else None
        if key is not None and key in seen:
            raise RetrievalImportError("baseline checkpoints contain duplicate query identities")
        if key is not None:
            seen.add(key)
        if record.get("status") != "success":
            skipped += 1
            skipped_reasons["failed_or_partial_record"] += 1
            continue
        try:
            if not isinstance(topic_id, str) or topic_id not in plan_by_topic:
                raise ValueError("unknown topic")
            if isinstance(query_index, bool) or not isinstance(query_index, int) or query_index < 1:
                raise ValueError("invalid query index")
            plan = plan_by_topic[topic_id]
            expected_queries = (plan.single_query,) if baseline == "b" else plan.facet_queries
            if query_index > len(expected_queries) or record.get("query") != expected_queries[query_index - 1]:
                raise ValueError("query does not match pinned plan")
            if record.get("pipeline") != PIPELINE:
                raise ValueError("unsupported pipeline")
            raw_hits = record.get("hits")
            if not isinstance(raw_hits, list) or record.get("hit_count") != len(raw_hits):
                raise ValueError("checkpoint hit count is inconsistent")
            hits = _hits(raw_hits)
            query_context = {
                "kind": f"baseline_{baseline}",
                "query_construction": BASELINE_CONSTRUCTIONS[baseline],
                "query_plan_sha256": plan_hash,
                "query_index": query_index,
            }
            identity = _build_identity(
                query=record["query"],
                top_k=100,
                query_context=query_context,
                base_url=base_url,
                corpus_identity=corpus_identity,
            )
            if not _reference_matches(record, identity):
                raise ValueError("retrieval reference mismatch")
            result = _hydrated_result(
                query=record["query"],
                hits=hits,
                corpus=corpus,
                local_cache_hit=record.get("local_cache_hit", False),
                remote_cached=record.get("remote_cached"),
            )
        except (TypeError, ValueError, RetrievalImportError):
            skipped += 1
            skipped_reasons["record_rejected"] += 1
            continue
        _persist_and_verify(cache, identity, result)
        objects.append(identity.object_key)
        imported += 1
        verified += 1
    return imported, verified, skipped


def import_retrieval_run(
    run_dir: Path,
    *,
    corpus: LocalCorpusIndex,
    cache: RetrievalArtifactCache,
    manifest_path: Path,
    base_url: str,
) -> RetrievalImportSummary:
    """Import one immutable Control-A or B/C run without participant I/O."""

    if not isinstance(cache, RetrievalArtifactCache):
        raise TypeError("cache must be a RetrievalArtifactCache")
    _validate_cache_namespace(cache)
    destination_metadata = _safe_cache_metadata(cache)
    run = _run_directory(run_dir)
    base_url = _validate_base_url(base_url)
    manifest_source = _source_file("manifest.json", run / "manifest.json")
    manifest = _read_json(manifest_source)
    schema = manifest.get("schema_version")
    if schema not in (CONTROL_A_SCHEMA, QUERY_BASELINE_SCHEMA):
        raise RetrievalImportError("unsupported retrieval run schema")
    if schema == CONTROL_A_SCHEMA:
        _validate_control_manifest(manifest)
    else:
        _validate_baseline_manifest(manifest)
    sources, _ = _prepare_sources(
        run, manifest, schema, manifest_source=manifest_source
    )
    output = _preflight_manifest_path(manifest_path, sources)
    corpus_identity = _corpus_identity(corpus)
    _check_manifest_corpus(manifest, corpus_identity, schema=schema)
    objects: list[str] = []
    skipped_reasons: Counter[str] = Counter()
    if schema == CONTROL_A_SCHEMA:
        imported, verified, skipped = _control_records(
            run,
            manifest,
            sources,
            corpus=corpus,
            cache=cache,
            base_url=base_url,
            corpus_identity=corpus_identity,
            objects=objects,
            skipped_reasons=skipped_reasons,
        )
    else:
        imported, verified, skipped = _baseline_records(
            manifest,
            sources,
            corpus=corpus,
            cache=cache,
            base_url=base_url,
            corpus_identity=corpus_identity,
            objects=objects,
            skipped_reasons=skipped_reasons,
        )
    _check_sources_unchanged(sources)
    summary = RetrievalImportSummary(
        run_schema=schema,
        source_files=_source_manifest(sources),
        imported=imported,
        verified=verified,
        skipped=skipped,
        skipped_reasons=dict(sorted(skipped_reasons.items())),
        object_keys=tuple(dict.fromkeys(objects)),
    )
    output_payload = {
        "schema_version": IMPORTER_SCHEMA_VERSION,
        "run_schema": schema,
        "source": {"files": summary.source_files},
        "corpus": {key: value for key, value in corpus_identity.items() if key != "path"},
        "destination": destination_metadata,
        "counts": summary.counts,
        "skipped_reasons": summary.skipped_reasons,
        "objects": {"count": len(summary.object_keys), "keys": list(summary.object_keys)},
    }
    _write_manifest(output, output_payload, sources=sources)
    return summary


def import_retrieval_artifacts(*args: Any, **kwargs: Any) -> RetrievalImportSummary:
    """Descriptive alias for :func:`import_retrieval_run`."""

    return import_retrieval_run(*args, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import an immutable Control-A or baseline retrieval run into the HF artifact cache."
    )
    parser.add_argument("run_dir", type=Path, help="absolute control-a or query-baseline run directory")
    parser.add_argument("--manifest", required=True, type=Path, help="absolute owner-only sanitized import manifest")
    parser.add_argument("--corpus", type=Path, help="absolute pinned local DuckDB corpus (default: configured project corpus)")
    parser.add_argument("--base-url", default=None, help="organizer base URL (default: RAGTIME_BASE_URL; no organizer request is made)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv()
    base_url = args.base_url or os.getenv("RAGTIME_BASE_URL", "")
    store = ArtifactStore.from_env()
    cache = RetrievalArtifactCache(store)
    _validate_cache_namespace(cache)
    corpus = LocalCorpusIndex(
        args.corpus,
        expected_dataset=DEFAULT_HF_DATASET,
        expected_splits=DEFAULT_HF_SPLITS,
        expected_translation_splits=DEFAULT_HF_MT_SPLITS,
    )
    try:
        summary = import_retrieval_run(
            args.run_dir,
            corpus=corpus,
            cache=cache,
            manifest_path=args.manifest,
            base_url=base_url,
        )
    finally:
        corpus.close()
    print(json.dumps({"counts": summary.counts, "skipped_reasons": summary.skipped_reasons}, sort_keys=True))
    return 0


__all__ = [
    "CONTROL_A_SCHEMA",
    "IMPORTER_SCHEMA_VERSION",
    "QUERY_BASELINE_SCHEMA",
    "RetrievalImportError",
    "RetrievalImportSummary",
    "build_parser",
    "import_retrieval_artifacts",
    "import_retrieval_run",
    "main",
]
