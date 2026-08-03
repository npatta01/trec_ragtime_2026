"""Topic parsing and deterministic Control-A retrieval helpers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, Sequence

from .client import RagtimeError
from .huggingface import HuggingFaceError
from .models import RetrievalResult, RetrievedDocument
from .paths import default_project_cache_dir


_TOPIC_FIELDS = frozenset(
    {"topic_id", "collection_id", "problem_statement", "background", "limit"}
)


@dataclass(frozen=True, slots=True)
class Topic:
    """One validated RAGTIME topic record."""

    topic_id: str
    collection_id: str
    problem_statement: str
    background: str
    limit: int


class RetrieverLike(Protocol):
    """Minimal retrieval boundary used by the batch runner and its tests."""

    cache_path: Path | None
    corpus_path: Path | None
    corpus: object

    def retrieve(
        self,
        query: str,
        *,
        pipeline: str,
        top_k: int | None,
        max_content_fallbacks: int | None,
        fallback_on_hf_error: bool,
    ) -> RetrievalResult: ...


def _non_blank_string(value: Any, *, field: str, line_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"line {line_number}: {field} must be a non-blank string")
    return value.strip()


def load_topics(path: Path) -> tuple[Topic, ...]:
    """Load and strictly validate ordered topics from a JSONL file."""

    topic_path = Path(path)
    topics: list[Topic] = []
    topic_ids: set[str] = set()
    try:
        handle = topic_path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"unable to read topics file {topic_path}: {exc}") from None

    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"line {line_number}: blank line is not allowed")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"line {line_number}: invalid JSON ({exc.msg})"
                ) from None
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_number}: topic must be a JSON object")

            fields = set(payload)
            missing = sorted(_TOPIC_FIELDS - fields)
            unexpected = sorted(fields - _TOPIC_FIELDS)
            if missing:
                raise ValueError(
                    f"line {line_number}: missing field {missing[0]!r}"
                )
            if unexpected:
                raise ValueError(
                    f"line {line_number}: unexpected field {unexpected[0]!r}"
                )

            topic_id = _non_blank_string(
                payload["topic_id"], field="topic_id", line_number=line_number
            )
            if topic_id in topic_ids:
                raise ValueError(f"line {line_number}: duplicate topic_id {topic_id!r}")
            topic_ids.add(topic_id)

            collection_id = _non_blank_string(
                payload["collection_id"],
                field="collection_id",
                line_number=line_number,
            )
            problem_statement = _non_blank_string(
                payload["problem_statement"],
                field="problem_statement",
                line_number=line_number,
            )
            background = _non_blank_string(
                payload["background"], field="background", line_number=line_number
            )
            limit = payload["limit"]
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
                raise ValueError(
                    f"line {line_number}: limit must be a positive integer"
                )

            topics.append(
                Topic(
                    topic_id=topic_id,
                    collection_id=collection_id,
                    problem_statement=problem_statement,
                    background=background,
                    limit=limit,
                )
            )

    if not topics:
        raise ValueError(f"topics file is empty: {topic_path}")
    return tuple(topics)


def build_topic_query(topic: Topic) -> str:
    """Build the stable problem-only Control-A retrieval query."""

    if not isinstance(topic, Topic):
        raise TypeError("topic must be a Topic")
    return topic.problem_statement


def pipeline_for_top_k(top_k: int) -> str:
    """Validate depth and return the current organizer pipeline name.

    The current endpoint returns up to 100 candidates for ``ragtime2``. The
    requested depth is therefore applied by ``RagtimeRetriever.top_k`` after
    the response, rather than encoded in a RoutIR suffix.
    """

    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or top_k < 1
        or top_k > 100
    ):
        raise ValueError("top_k must be an integer from 1 through 100")
    return "ragtime2"


def query_hash(query: str) -> str:
    """Return the SHA-256 digest of the exact UTF-8 query bytes."""

    if not isinstance(query, str) or not query:
        raise ValueError("query must be a non-empty string")
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def _write_atomic(path: Path, content: str) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".control-a-",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def write_control_a_artifact(
    run_dir: Path,
    *,
    manifest: Mapping[str, object],
    records: Iterable[Mapping[str, object]],
) -> None:
    """Atomically write one immutable Control-A run directory."""

    destination = Path(run_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.mkdir(mode=0o700)
    except FileExistsError:
        raise FileExistsError(f"Control-A run already exists: {destination}") from None

    try:
        candidate_lines = "".join(
            json.dumps(
                dict(record),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
            for record in records
        )
        manifest_text = (
            json.dumps(
                dict(manifest),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        _write_atomic(destination / "candidates.jsonl", candidate_lines)
        _write_atomic(destination / "manifest.json", manifest_text)
    except Exception:
        for child in destination.iterdir():
            child.unlink()
        destination.rmdir()
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(path: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path.parent,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    revision = completed.stdout.strip()
    return revision or None


def _run_id_path(output_root: Path, run_id: str) -> Path:
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-blank string")
    if run_id in {".", ".."} or Path(run_id).name != run_id:
        raise ValueError("run_id must be a single directory name")
    return output_root / run_id


def _candidate_record(
    topic: Topic,
    query: str,
    document: RetrievedDocument,
    *,
    missing_translation_ids: set[str],
) -> dict[str, object]:
    record = document.as_record(include_text=True)
    record.update(
        {
            "collection_id": topic.collection_id,
            "limit": topic.limit,
            "query": query,
            "query_hash": query_hash(query),
            "topic_id": topic.topic_id,
            "hydrated": document.hydrated,
            "report_ready": document.report_text is not None,
            "missing_translation": document.document_id in missing_translation_ids,
        }
    )
    return record


def _corpus_manifest(retriever: RetrieverLike) -> dict[str, object]:
    metadata = getattr(getattr(retriever, "corpus", None), "metadata", None)
    return {
        "path": (
            str(retriever.corpus_path)
            if retriever.corpus_path is not None
            else None
        ),
        "revision": getattr(metadata, "revision", None),
    }


def run_control_a(
    topics_path: Path,
    *,
    retriever: RetrieverLike,
    output_root: Path,
    run_id: str,
    top_k: int = 100,
    git_revision: str | None = None,
    created_at: str | None = None,
) -> Path:
    """Retrieve all topics once and publish one immutable Control-A artifact."""

    topic_path = Path(topics_path)
    topics = load_topics(topic_path)
    pipeline = pipeline_for_top_k(top_k)
    run_dir = _run_id_path(Path(output_root), run_id)
    if run_dir.exists():
        raise FileExistsError(f"Control-A run already exists: {run_dir}")

    records: list[dict[str, object]] = []
    topic_status: list[dict[str, object]] = []
    unresolved_count = 0
    missing_translation_count = 0
    failed_topic_count = 0
    for topic in topics:
        query = build_topic_query(topic)
        try:
            result = retriever.retrieve(
                query,
                pipeline=pipeline,
                top_k=top_k,
                max_content_fallbacks=0,
                fallback_on_hf_error=False,
            )
        except (RagtimeError, HuggingFaceError) as exc:
            failed_topic_count += 1
            topic_status.append(
                {
                    "error": {"type": type(exc).__name__},
                    "missing_translation_ids": [],
                    "topic_id": topic.topic_id,
                    "unresolved_ids": [],
                }
            )
            continue
        missing_translation_ids = set(result.missing_translation_ids)
        unresolved_ids = list(result.unresolved_ids)
        unresolved_count += len(unresolved_ids)
        missing_translation_count += len(missing_translation_ids)
        topic_status.append(
            {
                "missing_translation_ids": sorted(missing_translation_ids),
                "topic_id": topic.topic_id,
                "unresolved_ids": unresolved_ids,
            }
        )
        records.extend(
            _candidate_record(
                topic,
                query,
                document,
                missing_translation_ids=missing_translation_ids,
            )
            for document in result.documents
        )

    manifest: dict[str, object] = {
        "schema_version": "control_a_v1",
        "run_id": run_id,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision if git_revision is not None else _git_revision(topic_path),
        "topic_source": {
            "path": str(topic_path.resolve()),
            "sha256": _sha256_file(topic_path),
        },
        "query_construction": "problem_only_v1",
        "requested_top_k": top_k,
        "pipeline": pipeline,
        "retrieval": {
            "top_k": top_k,
            "max_content_fallbacks": 0,
            "fallback_on_hf_error": False,
        },
        "api_cache_path": (
            str(retriever.cache_path) if retriever.cache_path is not None else None
        ),
        "corpus": _corpus_manifest(retriever),
        "counts": {
            "topics": len(topics),
            "candidates": len(records),
            "failed_topics": failed_topic_count,
            "unresolved_documents": unresolved_count,
            "missing_translation_documents": missing_translation_count,
        },
        "topic_status": topic_status,
    }
    write_control_a_artifact(run_dir, manifest=manifest, records=records)
    return run_dir


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer") from None
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit Control-A command-line parser."""

    parser = argparse.ArgumentParser(
        description="Run the Control-A organizer retrieval batch.",
    )
    parser.add_argument(
        "--topics",
        type=Path,
        default=Path("topics/topics.all.2026.v0625.jsonl"),
        help="topic JSONL source (default: %(default)s)",
    )
    parser.add_argument(
        "--top-k",
        type=_positive_int,
        default=100,
        help="requested organizer candidate depth (default: %(default)s)",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="unique output directory name for this immutable run",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_project_cache_dir() / "experiments" / "control-a",
        help="shared artifact root (default: %(default)s)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the live Control-A batch after the operator decrypts `.env`."""

    try:
        arguments = build_parser().parse_args(argv)
    except SystemExit as exc:
        if exc.code == 0:
            return 0
        raise

    from .retriever import RagtimeRetriever

    with RagtimeRetriever.from_env() as retriever:
        run_dir = run_control_a(
            arguments.topics,
            retriever=retriever,
            output_root=arguments.output_root,
            run_id=arguments.run_id,
            top_k=arguments.top_k,
        )
    manifest = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    counts = manifest["counts"]
    print(f"Control-A artifact: {run_dir}")
    print(f"Topics: {counts['topics']}; candidates: {counts['candidates']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RetrieverLike",
    "Topic",
    "build_parser",
    "build_topic_query",
    "load_topics",
    "main",
    "pipeline_for_top_k",
    "query_hash",
    "run_control_a",
    "write_control_a_artifact",
]
