# Feature files

Design record for hypline's feature file format — schema, naming rules, and
alignment contract with BOLD data.

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

Feature files carry **stimulus-side identity entities** from the source BOLD (`sub`, `ses`,
`task`, `run`), the segment entity value (e.g. `trial-1`), and hypline's own `feature` entity:

```
sub-01_ses-01_task-movie_run-1_trial-1_feature-clip.parquet
```

Feature files carry only structural identity — descriptive attributes (condition, stimulus
item, etc.) live in `events.json` under `SegmentMetadata` and are joined at enrichment time. Do not
put descriptive entities on feature filenames; they belong in the sidecar.

Feature files do **not** carry acquisition entities (`acq`, `ce`, `rec`, `dir`). Features are
stimulus-derived — the same stimulus embedding applies regardless of scanner acquisition
parameters. Encoding validation enforces this: if BOLD files carry acquisition variants,
feature files are not required to mirror them.

## CellKey

`CellKey` is the open-schema row key for a feature time window. After enrichment it carries:
- Filename entities: `ses`, `run`, segment entity value (e.g. `trial=1`)
- Metadata entities from `events.json` `SegmentMetadata` (e.g. `cond=R`, `item=101`)

Excluded from `CellKey` (`CellKey.EXCLUDE`): `sub`, `task`, `acq`, `ce`, `rec`, `dir`
(invariant within a training call), `desc`, `res`, `den`, `echo` (BOLD image-variant
derivatives), `space`, `feature` (orthogonal axes). Entities are present or absent — no
`None` sentinel. Equality and hashing are order-independent.

CV splits are expressed by querying `CellKey` entities:
```python
train = [s for k, s in data.row_slices.items() if k["trial"] == "1"]
```

## Filename entity vs. events.json metadata

`events.json` is the authoritative source for descriptive metadata. Four cases:

- Sidecar-only (key in `SegmentMetadata`, absent from filename): merged onto the resolved `CellKey`.
- Both, same value: allowed; redundant but harmless.
- Both, different value: raise — the two sources of truth disagree.
- Filename-only descriptive (key absent from `SegmentMetadata`): raise, pointing user to events.json.

For unsegmented runs (no events.tsv key-value rows), only `ses` and `run` are valid on
feature filenames — any other entity raises.

## One file per segment

When a run is segmented, features must be split at segment boundaries upstream — one file
per `(run, segment)` pair, carrying the segment entity in its filename. Encoding downsamples
each file independently and never bins across segments. Run-level feature files spanning
multiple segments would silently merge timepoints across segment boundaries during TR binning.

## Mirroring requirement

Feature filenames must carry the same stimulus-side identity entities as their source BOLD
file. If BOLD has `ses-01_run-1`, the feature file must too.

This is how pipelines match features to BOLD runs. Aggregating features across sessions or
runs (e.g. a subject-level embedding) is out of scope for encoding models — such features
don't map 1:1 to BOLD TRs.

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
