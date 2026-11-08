# Submissions

Every run submitted to NIST for TREC RAGTIME 2026, one folder per run.

A folder is the complete record of a submission: the exact bytes uploaded, a
checksum, and a ledger holding the values entered in the EvalBase form, the run
metadata, the validation evidence, and any limitation worth disclosing. A run
should be reproducible and auditable from its folder alone.

## Index

| Runtag | Task | Topics | Status | Validation | Ledger |
| --- | --- | --- | --- | --- | --- |
| [`np-nugget-sol-v1`](np-nugget-sol-v1/) | Report generation | 103 / 103 | **submitted 2026-08-13** | AutoJudge `ragtime26` `--strict`: 103/103 ✅ | [LEDGER.md](np-nugget-sol-v1/LEDGER.md) |

## Organizer results

The results for `np-nugget-sol-v1` are recorded in
[results/2026-11-02](../results/2026-11-02/README.md), with the original organizer
archives, score tables, and offline verification. The two evaluations cover
17 and 34 topics respectively; the submission itself contains 103 topics.

## Layout

```
submissions/
  README.md                            this index
  <runtag>/
    LEDGER.md                          form values, metadata, validation, limitations
    <TEAM>.<runtag>.<spec>.jsonl       the exact file uploaded
    <TEAM>.<runtag>.<spec>.jsonl.sha256
```

The JSONL is named `<TEAM>.<runtag>.<spec>.jsonl` so a file separated from its
folder still identifies itself.

## Verify a submission

```bash
cd submissions/<runtag> && sha256sum -c *.sha256
uv run ragtime-validate-submission submissions/<runtag>/*.jsonl --strict
```

`--strict` promotes every warning to a hard failure; use it before uploading.
Exit 0 means the organizers' checker accepts the run.

## Adding a run

1. Generate it — see the run's ledger §8 for the exact command shape.
2. `mkdir submissions/<runtag>` and copy in the submission JSONL.
3. Write `<file>.sha256` with `sha256sum`.
4. Copy an existing `LEDGER.md` as the template and fill in every section.
   Leave the **Submitted** section blank until the upload is confirmed — a
   ledger claiming a submission that has not happened is worse than one with a
   gap. Fill it in from the EvalBase listing once the run appears there.
5. Add a row to the index above.

Keep the folder name, the runtag in the filename, and the `run_id` inside every
record identical. The `ragtime26` spec caps `run_id` at 20 characters and treats
over-length as a hard failure, and a mismatch between the form and the file is
the easiest way to have a run rejected or mis-attributed.
