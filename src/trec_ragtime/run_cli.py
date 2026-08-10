"""Command-line entry point for reproducible baseline RAGTIME runs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .baseline import ExtractiveReportGenerator
from .query_planning import (
    CompositeQueryPlanner,
    FacetQueryPlanner,
    load_query_plans,
)
from .local_corpus import default_local_corpus_path
from .retriever import RagtimeRetriever
from .runner import BaselineRunner, validate_baseline_parameters
from .topics import load_topic_requests


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragtime-run",
        description=(
            "Generate paired citation-bearing report JSONL and fused MLIR output."
        ),
    )
    parser.add_argument("--topics", required=True, help="official topic JSONL")
    parser.add_argument("--reports", required=True, help="report JSONL output")
    parser.add_argument("--mlir", required=True, help="six-column MLIR output")
    parser.add_argument("--artifacts", help="optional retrieval/evidence audit JSONL")
    parser.add_argument("--team-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--run-desc",
        default="Deterministic extractive multilingual baseline with weighted RRF",
    )
    parser.add_argument(
        "--pipeline",
        default="ragtime2%100",
        help="pipeline for automatic full/facet plans (default: ragtime2%%100)",
    )
    planning = parser.add_mutually_exclusive_group()
    planning.add_argument(
        "--facets",
        action="store_true",
        help="use deterministic problem-statement facets in addition to the full query",
    )
    planning.add_argument(
        "--query-plan",
        help="precomputed multilingual query-plan JSONL",
    )
    parser.add_argument("--max-facets", type=int, default=8)
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument("--max-documents", type=int, default=1000)
    parser.add_argument(
        "--hydrate-top-k",
        type=int,
        default=30,
        help="fused documents to hydrate; must equal --evidence-documents",
    )
    parser.add_argument("--evidence-documents", type=int, default=30)
    parser.add_argument("--max-sentences", type=int, default=24)
    parser.add_argument("--max-sentences-per-document", type=int, default=2)
    parser.add_argument("--min-sentence-characters", type=int, default=40)
    parser.add_argument("--max-sentence-characters", type=int, default=600)
    parser.add_argument("--min-term-overlap", type=int, default=1)
    parser.add_argument("--local-index-path")
    parser.add_argument("--refresh-search", action="store_true")
    parser.add_argument("--refresh-documents", action="store_true")
    return parser


def _progress(index: int, total: int, topic_id: str) -> None:
    print(f"[{index}/{total}] completed topic {topic_id}", flush=True)


def _format_error(error: BaseException) -> str:
    lines = [f"error: {error}"]
    lines.extend(f"note: {note}" for note in getattr(error, "__notes__", ()))
    return "\n".join(lines) + "\n"


def _preflight_paths(
    *,
    topic_path: str,
    query_plan_path: str | None,
    local_index_path: Path | None,
    report_path: str,
    mlir_path: str,
    artifact_path: str | None,
) -> None:
    """Reject unsafe output targets before the first participant-API call."""

    protected_inputs = {
        "topic file": Path(topic_path).expanduser().resolve(),
        "local corpus index": (
            default_local_corpus_path().resolve()
            if local_index_path is None
            else local_index_path.resolve()
        ),
    }
    if query_plan_path is not None:
        protected_inputs["query-plan file"] = (
            Path(query_plan_path).expanduser().resolve()
        )

    raw_outputs = {
        "report output": report_path,
        "MLIR output": mlir_path,
    }
    if artifact_path is not None:
        raw_outputs["artifact output"] = artifact_path
    outputs = {
        label: Path(path).expanduser().resolve()
        for label, path in raw_outputs.items()
    }
    if len(outputs.values()) != len(set(outputs.values())):
        raise ValueError("report, MLIR, and artifact output paths must be distinct")

    for output_label, output_path in outputs.items():
        for input_label, input_path in protected_inputs.items():
            if output_path == input_path:
                raise ValueError(f"{output_label} must not overwrite the {input_label}")
        if output_path.exists() and output_path.is_dir():
            raise ValueError(f"{output_label} points to a directory: {output_path}")
        if not output_path.parent.is_dir():
            raise ValueError(
                f"parent directory for {output_label} does not exist: "
                f"{output_path.parent}"
            )


def main(argv: Sequence[str] | None = None) -> int:
    """Execute a complete automatic extractive baseline run."""

    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        topics = load_topic_requests(arguments.topics)
        if arguments.query_plan:
            planner = load_query_plans(arguments.query_plan)
        elif arguments.facets:
            planner = FacetQueryPlanner(
                pipeline=arguments.pipeline,
                max_facets=arguments.max_facets,
            )
        else:
            planner = CompositeQueryPlanner(arguments.pipeline)

        generator = ExtractiveReportGenerator(
            max_documents=arguments.evidence_documents,
            max_sentences=arguments.max_sentences,
            max_sentences_per_document=arguments.max_sentences_per_document,
            min_sentence_characters=arguments.min_sentence_characters,
            max_sentence_characters=arguments.max_sentence_characters,
            min_term_overlap=arguments.min_term_overlap,
        )
        validate_baseline_parameters(
            generator=generator,
            team_id=arguments.team_id,
            run_id=arguments.run_id,
            run_desc=arguments.run_desc,
            rrf_k=arguments.rrf_k,
            max_documents=arguments.max_documents,
            hydrate_top_k=arguments.hydrate_top_k,
            refresh_search=arguments.refresh_search,
            refresh_documents=arguments.refresh_documents,
        )
        index_path = (
            None
            if arguments.local_index_path is None
            else Path(arguments.local_index_path).expanduser().resolve()
        )
        _preflight_paths(
            topic_path=arguments.topics,
            query_plan_path=arguments.query_plan,
            local_index_path=index_path,
            report_path=arguments.reports,
            mlir_path=arguments.mlir,
            artifact_path=arguments.artifacts,
        )
        with RagtimeRetriever.from_env(local_index_path=index_path) as retriever:
            result = BaselineRunner(
                retriever=retriever,
                planner=planner,
                generator=generator,
                team_id=arguments.team_id,
                run_id=arguments.run_id,
                run_desc=arguments.run_desc,
                rrf_k=arguments.rrf_k,
                max_documents=arguments.max_documents,
                hydrate_top_k=arguments.hydrate_top_k,
                refresh_search=arguments.refresh_search,
                refresh_documents=arguments.refresh_documents,
            ).run(topics, progress=_progress)
        result.write(
            report_path=arguments.reports,
            mlir_path=arguments.mlir,
            artifact_path=arguments.artifacts,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.exit(2, _format_error(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
