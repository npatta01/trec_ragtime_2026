"""Emit a self-contained scoring plan so a GPU job needs no local corpus.

The 24 GB DuckDB corpus is only needed to *construct* a retrieval identity, not
to read one: cached retrieval artifacts already carry document text. This writes
every identity field out as JSON so a remote worker can rebuild the identities,
pull the artifacts from the shared bucket, and score on a GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--publish-key", default=None,
        help="also store the plan at this object key so a remote job can fetch "
             "it; the repo is uploaded without gitignored files",
    )
    parser.add_argument("--topic-id", action="append", dest="topic_ids", default=None)
    parser.add_argument("--representation", default="report_text")
    parser.add_argument("--top-k", type=int, default=100)
    arguments = parser.parse_args()

    load_dotenv(Path.cwd() / ".env")
    from trec_ragtime.nuggets_cli import plan_queries, select_plans
    from trec_ragtime.query_baselines import (
        EXPECTED_QUERY_PLAN_SHA256,
        PIPELINE,
        default_query_plan_path,
        load_query_plans,
    )
    from trec_ragtime.query_baselines import _query_context as query_context
    from trec_ragtime.retriever import RagtimeRetriever

    plans = select_plans(
        load_query_plans(default_query_plan_path()), arguments.topic_ids
    )
    queries: list[dict[str, object]] = []
    missing = 0
    with RagtimeRetriever.from_env() as retriever:
        for plan in plans:
            for facet in plan_queries(plan):
                identity = retriever.build_retrieval_identity(
                    facet.query,
                    pipeline=PIPELINE,
                    top_k=arguments.top_k,
                    max_content_fallbacks=0,
                    fallback_on_hf_error=False,
                    query_context=query_context(
                        baseline=facet.baseline,
                        plan_sha256=EXPECTED_QUERY_PLAN_SHA256,
                        query_index=facet.query_index,
                    ),
                )
                # The object key is all the worker needs: the artifact behind it
                # already carries document text, so no corpus travels with it.
                queries.append(
                    {
                        "topic_id": plan.topic_id,
                        "facet_index": facet.facet_index,
                        "query": facet.query,
                        "object_key": identity.object_key,
                    }
                )

    plan_document = {
        "schema_version": 1,
        "representation": arguments.representation,
        "queries": queries,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(plan_document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"{len(queries)} queries across {len(plans)} topics -> {arguments.output}")
    if arguments.publish_key:
        from trec_ragtime.artifact_store import ArtifactStore

        stored = ArtifactStore.from_env().write(
            arguments.publish_key, arguments.output.read_bytes()
        )
        if not stored:
            raise SystemExit(f"could not publish the plan to {arguments.publish_key}")
        print(f"published -> {arguments.publish_key}")
    del missing
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
