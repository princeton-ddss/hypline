# Feature files

Design record for hypline's feature file format — schema, naming rules, and alignment contract with BOLD data.

Hypline's own derivative type — stimulus-derived features paired to BOLD
runs for encoding models.

Feature I/O is implemented in `hypline/features/utils.py` (formerly `hypline/featuregen/utils.py`).
`save_feature` writes, `read_feature` reads, `resample_feature` handles TR alignment.

## Format

- Extension: `.parquet`
- Columns:
  - `start_time` (float, seconds from run onset)
  - `feature` (list/array column — per-timepoint feature vector)

## Naming

Feature files inherit **stimulus-side identity entities** from the source
BOLD (`sub`, `ses`, `task`, `run`), plus hypline's own entities (`partition`,
`feature`):

```
sub-01_ses-01_task-movie_run-1_partition-part1_feature-clip.parquet
```

Feature files do **not** carry acquisition entities (`acq`, `ce`, `rec`,
`dir`). Features are stimulus-derived — the same stimulus embedding applies
regardless of scanner acquisition parameters. Encoding validation enforces
this: if BOLD files carry acquisition variants, feature files are not
required to mirror them.

## Mirroring requirement

Feature filenames must carry the same stimulus-side identity entities as
their source BOLD file. If BOLD has `ses-01_run-1`, the feature file must
too.

This is how pipelines match features to BOLD runs. Aggregating features
across sessions or runs (e.g. a subject-level embedding) is out of scope
for encoding models — such features don't map 1:1 to BOLD TRs.

## Temporal alignment

Feature files are **not required** to be TR-aligned. Pipelines that need
TR-level X (encoding) downsample:

- If `start_time` intervals exactly match TR, pass through unchanged.
- Otherwise bin by `floor(start_time / TR)` and aggregate (default: mean).

**Bin boundary convention**: `floor(start_time / TR)` maps each event to the
TR whose window starts at or before the event — i.e. TR `t` covers
`[t*TR, (t+1)*TR)`. An event at `start_time=0.0` lands in TR 0; an event at
`start_time=k*TR` lands in TR `k`, not TR `k-1`. The legacy
`hyperscanning/fconv` codebase used `(t*TR, (t+1)*TR]` (right-inclusive),
silently dropping events at `start_time=0.0`. Hypline uses the standard
left-inclusive convention.

**Pass-through detection**: data evenly spaced at TR cadence but phase-shifted
(e.g. sampled at mid-TR) also triggers pass-through, since it is already at
TR resolution. The output `start_time` is always regenerated as `i * TR`.

**Assumption**: each event's duration ≤ TR. If features gain events spanning
multiple TRs (requires adding `end_time` to the schema), current
binning-by-start-time is incorrect and must be revisited.
