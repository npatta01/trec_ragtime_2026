"""``ragtime-validate-submission``: gate a run through the AutoJudge checker.

The track organizers' validator is Laura Dietz's ``autojudge-base`` package, run
as ``python -m autojudge_base.report_tool check --spec ragtime26``. This module
is a thin, reproducible wrapper around that command. It owns exactly two things
the raw command cannot do for us:

1. **Topic-schema translation.** ``--topics`` is what turns on the per-topic
   coverage cross-check and the per-request character limit, and it is the one
   check that actually depends on this year's topics. The validator parses that
   file into ``autojudge_base.request.Request``, which requires ``request_id``
   and ``title``. The official
   ``topics/topics.all.2026.v0625-fix.jsonl`` carries ``topic_id`` and
   ``collection_id`` instead, so passing it straight through fails to load.
   We project the repository's validated :class:`~trec_ragtime.topic_candidates.Topic`
   onto the Request schema into a throwaway file rather than editing or
   forking the official topics, which stay the pinned source of truth.
2. **Dependency isolation.** ``autojudge-base`` is a submission-time checker,
   not a pipeline dependency, and it is deliberately not in ``pyproject.toml``.
   If it is importable we use the current interpreter; otherwise we shell out
   through ``uv run --isolated --no-project --with`` so the project environment
   and lock file stay untouched.

Nothing here interprets the findings. The checker's exit code is propagated
verbatim -- 255 when it reports a PROBLEM (or a coverage error), 0 otherwise --
so this command works as a pre-submission gate in a shell or in CI.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Sequence
from importlib.util import find_spec
from pathlib import Path

from .topic_candidates import Topic, load_topics


# Pinned to the version Laura announced as carrying the stricter 2026 report
# checks. Kept as a floor rather than an equality pin so a published fix is
# picked up; override with --autojudge-requirement to reproduce an exact run.
AUTOJUDGE_REQUIREMENT = "autojudge-base>=0.4.3"
AUTOJUDGE_MODULE = "autojudge_base.report_tool"
DEFAULT_SPEC = "ragtime26"
DEFAULT_TOPICS_PATH = Path("topics/topics.all.2026.v0625-fix.jsonl")


class SubmissionValidationError(RuntimeError):
    """The checker could not be run at all, as distinct from finding problems."""


def request_record(topic: Topic) -> dict[str, object]:
    """Project one validated topic onto the AutoJudge ``Request`` schema.

    ``limit`` is the field ``ragtime26`` names as its per-request length limit
    (NFKC characters), so it must survive the translation exactly. ``title`` is
    required by the model but unused by the ``ragtime26`` checks, so a topic
    without one falls back to its id rather than borrowing another field's text
    and inventing a title that never appeared in the official file.
    """

    if not isinstance(topic, Topic):
        raise TypeError("topic must be a Topic")
    return {
        "request_id": topic.topic_id,
        "collection_ids": [topic.collection_id],
        "problem_statement": topic.problem_statement,
        "background": topic.background,
        "limit": topic.limit,
        "title": topic.title if topic.title is not None else topic.topic_id,
    }


def write_requests_file(topics: Iterable[Topic], path: Path) -> Path:
    """Write topics as an AutoJudge requests JSONL, preserving source order."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(
            json.dumps(request_record(topic), ensure_ascii=False) + "\n"
            for topic in topics
        ),
        encoding="utf-8",
    )
    return destination


def autojudge_runner(requirement: str = AUTOJUDGE_REQUIREMENT) -> list[str]:
    """Return the argv prefix that runs the checker module.

    Prefers the current interpreter when ``autojudge_base`` is already
    installed, which keeps an explicitly prepared environment authoritative and
    makes the run offline-capable. Otherwise ``uv`` provides it on demand
    without touching the project environment.
    """

    try:
        installed = find_spec("autojudge_base") is not None
    except (ImportError, ValueError):
        installed = False
    if installed:
        return [sys.executable, "-m", AUTOJUDGE_MODULE]

    uv = shutil.which("uv")
    if uv is None:
        raise SubmissionValidationError(
            "autojudge_base is not importable and uv is not on PATH; install it "
            f"with: pip install '{requirement}'"
        )
    return [
        uv,
        "run",
        "--isolated",
        "--no-project",
        "--with",
        requirement,
        "python",
        "-m",
        AUTOJUDGE_MODULE,
    ]


def build_check_command(
    runner: Sequence[str],
    submissions: Sequence[Path],
    *,
    spec: str | None = DEFAULT_SPEC,
    requests_path: Path | None = None,
    strict: bool = False,
    suppress_warnings: bool = False,
) -> list[str]:
    """Build the exact ``report_tool check`` argv, options before the paths."""

    if not submissions:
        raise ValueError("at least one submission path is required")
    command = [*runner, "check"]
    if spec is not None:
        command += ["--spec", spec]
    if requests_path is not None:
        command += ["--topics", str(requests_path)]
    if strict:
        command.append("--strict")
    if suppress_warnings:
        command.append("--suppress-warnings")
    command += [str(path) for path in submissions]
    return command


def run_check(
    submissions: Sequence[Path],
    *,
    topics_path: Path | None,
    spec: str | None = DEFAULT_SPEC,
    strict: bool = False,
    suppress_warnings: bool = False,
    requirement: str = AUTOJUDGE_REQUIREMENT,
    requests_out: Path | None = None,
    echo: bool = True,
) -> int:
    """Translate topics, run the checker, and return its exit code."""

    missing = [path for path in submissions if not Path(path).is_file()]
    if missing:
        raise SubmissionValidationError(f"submission file not found: {missing[0]}")

    runner = autojudge_runner(requirement)
    with tempfile.TemporaryDirectory(prefix="ragtime-requests-") as scratch:
        requests_path: Path | None = None
        if topics_path is not None:
            topics = load_topics(Path(topics_path))
            requests_path = write_requests_file(
                topics,
                Path(requests_out)
                if requests_out is not None
                else Path(scratch) / "requests.jsonl",
            )
        command = build_check_command(
            runner,
            submissions,
            spec=spec,
            requests_path=requests_path,
            strict=strict,
            suppress_warnings=suppress_warnings,
        )
        if echo:
            # shlex.join, not a plain join: the requirement carries `>=`, which a
            # plain join turns into a shell redirect if the line is pasted back.
            print(f"$ {shlex.join(command)}", flush=True)
        try:
            completed = subprocess.run(command, check=False)
        except OSError as exc:
            raise SubmissionValidationError(
                f"unable to run the AutoJudge checker: {exc}"
            ) from None
    return completed.returncode


def build_parser() -> argparse.ArgumentParser:
    """Build the submission-validation command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Check RAGTIME submission JSONL files with the organizers' "
            "AutoJudge report checker."
        ),
    )
    parser.add_argument(
        "submissions",
        nargs="+",
        type=Path,
        help="submission JSONL file(s) to check",
    )
    parser.add_argument(
        "--topics",
        type=Path,
        default=DEFAULT_TOPICS_PATH,
        help=(
            "official topic JSONL used for the coverage and per-topic length "
            "cross-check (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--no-topics",
        dest="topics",
        action="store_const",
        const=None,
        help="skip the topic cross-check and run the spec checks only",
    )
    parser.add_argument(
        "--spec",
        default=DEFAULT_SPEC,
        help="AutoJudge track spec id (default: %(default)s)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="treat smells (warnings) as hard errors",
    )
    parser.add_argument(
        "--suppress-warnings",
        action="store_true",
        help="hide the smell/warning output",
    )
    parser.add_argument(
        "--requests-out",
        type=Path,
        default=None,
        help="keep the translated requests JSONL at this path instead of a temp file",
    )
    parser.add_argument(
        "--autojudge-requirement",
        default=AUTOJUDGE_REQUIREMENT,
        help="pip requirement used when autojudge_base is not installed (default: %(default)s)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the checker and propagate its exit code (255 on any PROBLEM)."""

    arguments = build_parser().parse_args(argv)
    try:
        return run_check(
            arguments.submissions,
            topics_path=arguments.topics,
            spec=arguments.spec,
            strict=arguments.strict,
            suppress_warnings=arguments.suppress_warnings,
            requirement=arguments.autojudge_requirement,
            requests_out=arguments.requests_out,
        )
    except (SubmissionValidationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTOJUDGE_MODULE",
    "AUTOJUDGE_REQUIREMENT",
    "DEFAULT_SPEC",
    "DEFAULT_TOPICS_PATH",
    "SubmissionValidationError",
    "autojudge_runner",
    "build_check_command",
    "build_parser",
    "main",
    "request_record",
    "run_check",
    "write_requests_file",
]
