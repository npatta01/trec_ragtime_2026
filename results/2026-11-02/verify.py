"""Verify the supplied organizer result snapshot without network access."""

import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import tarfile


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
RUN = "np-nugget-sol-v1"
PREFIX = "ragtime2026-results.1102.NP_Labs"
EVALUATIONS = {"repgen.almost-human.17-topics": 17, "repgen.autoargue.34-topics": 34}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def tsv(data):
    return list(csv.DictReader(io.StringIO(data.decode("utf-8")), delimiter="\t"))


def verify():
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        require(digest((ROOT / relative).read_bytes()) == expected, f"Checksum: {relative}")

    manifest = json.loads((ROOT / "manifest.json").read_text())
    require(manifest["run_id"] == RUN and manifest["team_id"] == "NP", "Manifest identity")
    archives = {}
    for source in manifest["sources"]:
        path = ROOT / source["path"]
        require(digest(path.read_bytes()) == source["sha256"], "Source archive checksum")
        members = {}
        with tarfile.open(path) as archive:
            for member in archive:
                parts = PurePosixPath(member.name)
                require(not parts.is_absolute() and ".." not in parts.parts, "Unsafe archive path")
                if member.isdir():
                    continue
                require(member.isfile(), "Non-regular archive member")
                require(member.name not in members, "Duplicate archive member")
                members[member.name] = archive.extractfile(member).read()
        expected = {m["path"]: m for m in source["members"]}
        require(set(members) == set(expected), "Archive member inventory")
        for name, data in members.items():
            require(len(data) == expected[name]["bytes"], "Archive member size")
            require(digest(data) == expected[name]["sha256"], "Archive member checksum")
        archives[path.name] = members
    require(len(archives) == 2, "Expected two source archives")
    results = archives["ragtime2026-results.1102.NP_Labs.tgz"]
    nuggets = archives["nuggets_1101.tgz"]
    require(len(results) == 6 and len(nuggets) == 34, "Archive file counts")

    folder = REPO / "submissions" / RUN
    submission_path = folder / f"NP.{RUN}.ragtime26.jsonl"
    expected = (folder / f"{submission_path.name}.sha256").read_text().split()[0]
    require(digest(submission_path.read_bytes()) == expected, "Original submission checksum")
    records = [json.loads(line) for line in submission_path.read_text().splitlines()]
    submitted = {r["metadata"]["topic_id"]: r for r in records}
    require(len(submitted) == len(records) == 103, "Submission topic count")
    require(all(r["metadata"]["run_id"] == RUN and r["metadata"]["team_id"] == "NP"
                for r in records), "Submission identity")

    topic_sets = {}
    aggregates = {}
    readme = (ROOT / "README.md").read_text()
    for evaluation, count in EVALUATIONS.items():
        for filename in (f"{RUN}.scores.tsv", "score-stats.tsv"):
            original = results[f"{PREFIX}/{evaluation}/{filename}"]
            require((ROOT / "scores" / evaluation / filename).read_bytes() == original,
                    f"Extracted TSV differs: {evaluation}/{filename}")
        rows = tsv(results[f"{PREFIX}/{evaluation}/{RUN}.scores.tsv"])
        topic_field = "topic_id" if "topic_id" in rows[0] else "request_id"
        require(set(rows[0]) == {"run_id", topic_field, "metric", "value"}, "Score schema")
        require(all(r["run_id"] == RUN for r in rows), "Score run identity")
        require(all(math.isfinite(float(r["value"])) for r in rows), "Non-finite score")
        require(len({(r[topic_field], r["metric"]) for r in rows}) == len(rows), "Duplicate score")
        topics = {r[topic_field] for r in rows} - {"all"}
        require(len(topics) == count and topics <= submitted.keys(), "Score topic coverage")
        topic_sets[evaluation] = topics
        aggregate = {r["metric"]: float(r["value"]) for r in rows if r[topic_field] == "all"}
        aggregates[evaluation] = aggregate
        stats = tsv(results[f"{PREFIX}/{evaluation}/score-stats.tsv"])
        require(set(stats[0]) == {"topic_id", "metric", "runs", "min", "median", "max", "mean"},
                "Cross-run statistics schema")
        overall = {r["metric"]: r for r in stats if r["topic_id"] == "all"}
        require(all(int(r["runs"]) == 29 for r in overall.values()), "Comparison run count")
        metrics = (["citation_relevance", "citation_support_llmfilled", "sentence_support_llmfilled",
                    "nugget_coverage_llmfilled", "f1_llmfilled"] if count == 17 else
                   [f"{metric}_{kind}" for metric in ("citation_relevance", "citation_support",
                    "sentence_support", "nugget_coverage", "f1") for kind in ("micro", "macro")])
        for metric in metrics:
            s = overall[metric]
            row = f"| `{metric}` | {aggregate[metric]:.4f} | {float(s['median']):.4f} | {float(s['max']):.4f} |"
            require(row in readme, f"Summary table mismatch: {metric}")
            if metric.startswith(("citation_support", "sentence_support")):
                require(math.isclose(aggregate[metric], float(s["max"]), rel_tol=0, abs_tol=1e-12),
                        "Support maximum claim")
            if metric.startswith(("nugget_coverage", "f1")):
                require(float(s["median"]) < aggregate[metric] < float(s["max"]), "Comparison claim")

    human, auto = EVALUATIONS
    require(topic_sets[human] <= topic_sets[auto], "Evaluation topic overlap")
    human_rows = tsv(results[f"{PREFIX}/{human}/{RUN}.judgments.tsv"])
    require({r["topic_id"] for r in human_rows} == topic_sets[human], "Almost-human judgment topics")
    require(all(r["run_id"] == RUN for r in human_rows), "Almost-human judgment identity")
    require({r["judge"] for r in human_rows} == {"human", "autoargue"}, "Judgment sources")
    reports = [json.loads(line)["report"] for line in
               results[f"{PREFIX}/{auto}/{RUN}.judgments.jsonl"].decode().splitlines()]
    require(len(reports) == 34 and {r["metadata"]["topic_id"] for r in reports} == topic_sets[auto],
            "Autoargue judgment topics")
    for report in reports:
        metadata = report["metadata"]
        require(metadata["team_id"] == "NP" and metadata["run_id"] == RUN, "Organizer identity")
        require(report["responses"] == submitted[metadata["topic_id"]]["responses"],
                "Organizer responses differ from submitted responses")
    banks = [json.loads(data) for data in nuggets.values()]
    require({b["metadata"]["topic_id"] for b in banks} == topic_sets[auto], "Nugget topic coverage")
    require(sum(len(b["nugget_bank"]) for b in banks) == aggregates[auto]["nuggets"] == 498,
            "Nugget bank count")
    print("PASS: checksums, archive inventory, four exact TSV copies, summary tables, 17/34-topic coverage,")
    print("29-run comparison statistics, 34 submitted-response matches, and 34 nugget banks / 498 entries.")


if __name__ == "__main__":
    verify()
