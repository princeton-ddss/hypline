# Encoding — scope and assumptions

What a single encoding training run operates on, what it requires, and what assumptions could break it.

The encoding pipeline fits a model from stimulus-derived features (X) to
BOLD responses (Y) — typically banded ridge regression.

## Scope of a single training run

A single `train(sub_id)` call is scoped to:

- **One subject.** Subjects are modeled independently — different brains,
  different voxel alignment.
- **One task (or no task).** Task is optional in BIDS — all files must
  either specify the same task or none at all. Cross-task encoding models
  are not meaningful. If inconsistent task values are discovered across
  BOLD or feature files, the pipeline raises.
- **Multiple sessions and runs: allowed and expected.** Same task, more
  data — concatenated into a single X/Y.
- **Multiple partitions per run: allowed.** Each partition is a separate row
  block. See [../decisions/partition-entity.md](../decisions/partition-entity.md).

## Alignment contract

Features and BOLD runs are matched by stimulus-side identity entities
(`sub`, `ses`, `task`, `run`). Every BOLD run in scope must have feature
coverage for all requested features, and vice versa. Partial coverage is a
hard error — silently dropping runs would mask upstream bugs in feature
generation.

Acquisition entities (`acq`, `ce`, `rec`, `dir`) must be invariant across
all BOLD files in a training call. Mixing fMRIPrep output variants would
either cause a shape mismatch (`acq`) or silently corrupt alignment
(`ce`, `rec`, `dir`). Features are not required to carry these entities —
they are stimulus-derived and independent of acquisition parameters.

Coverage failures are reported fail-fast: the first missing (cell, feature)
pair raises, without aggregating the full list. Mismatches are typically
systematic (e.g. an entire feature missing across all cells from a failed
extraction run), so the first gap is enough signal for the user to fix the
root cause and rerun. Aggregating would flood the message without changing
what the user does next.

## Module layout

Feature I/O utilities (`read_feature`, `resample_feature`, `save_feature`) live in
`hypline/features/`. The `Encoding` class imports from `features.utils` directly;
external callers should too.

`find_files` is called with `recursive=True` throughout encoding discovery. This is
intentional — fMRIPrep derivatives are often organized in per-subject subdirectories
(`sub-01/func/`), so flat-directory search would miss them.

## Assumptions that could break

- **Consistent TR across all runs** for a subject. Mixed-TR datasets would
  need per-run TR handling.
- **Feature files mirror BOLD identity entities.** See
  [../decisions/feature-files.md](../decisions/feature-files.md).
- **Partitions declared in events** use the `"partition-{value}"` trial_type
  convention. Changing this affects validation and lookup sites.
- **Partitions must tile the full run.** Contiguity and zero-start are checked
  in `_discover_bold`; full BOLD length coverage is checked in `_build_xy`
  once arrays are loaded. See [../decisions/partition-entity.md](../decisions/partition-entity.md).
- **One feature file per (feature, run, partition) cell.** Multiple files is
  ambiguous provenance, not something to merge.
