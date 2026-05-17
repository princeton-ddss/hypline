# Feature files

Design record for hypline's feature file format — schema, naming rules, and
alignment contract with BOLD data.

Hypline's own derivative type — stimulus-derived features paired to BOLD
runs for encoding models.

Feature I/O lives in `hypline.features`: `save_feature` writes, `read_feature`
reads, `resample_feature` handles TR alignment.

## Format

- Extension: `.parquet`
- Required columns:
  - `start_time` (float, seconds from run onset)
  - `feature` (list/array column — per-row feature vector)
- Generators may add unit-identifying columns (e.g. `word`, `phoneme`, `token`).

## Column order

Identifiers → descriptors → payload. `start_time` and any unit-identifying
column come first, descriptive context next, `feature` last. Groups keys for
sorting/joining and keeps the dense payload column out of the way on inspection.

## Row granularity

Each row is at the generator's natural unit — phonemes for phonemic features,
tokens for semantic embeddings, words for word-level features, etc. Multiple
rows may share the same `start_time` when the upstream source is coarser than
the feature unit (e.g. phonemes derived from word-level transcripts inherit the
word's `start_time` — see [../modules/phonemic.md](../modules/phonemic.md)).

TR alignment is the consumer's responsibility: pipelines that need TR-level X
call `resample_feature`, which downsamples by binning on `start_time`.

## Missing-unit rows

When a generator's upstream source yields a timestamped item but no feature
unit can be derived from it (OOV token, empty tokenization, punctuation-only
input, etc.), emit **one row** at that `start_time` with the unit-identifying
column set to `None` and a zero `feature` vector. Do **not** skip the row.

Why: `resample_feature` bins by `start_time` against TRs. Dropping rows
shifts later events into earlier TRs and silently misaligns the feature
matrix with BOLD. A null row preserves the grid; downstream code that wants
to ignore them can filter on the unit column.

## Naming

Feature files carry **stimulus-side identity entities** from the source BOLD (`sub`, `ses`,
`task`, `run`), the segment entity value (e.g. `trial-1`), and hypline's own `feat` entity:

```
sub-01_ses-01_task-movie_run-1_trial-1_feat-clip.parquet   # with sessions
sub-01_task-movie_run-1_trial-1_feat-clip.parquet           # sessionless
```

Feature files live under `features/` — see [layout.md](layout.md) for the root tree.

Feature files carry only structural identity — descriptive attributes (condition, stimulus
item, etc.) live in `events.json` under `trial_type.Levels` (per-segment `metadata`) and are
joined at enrichment time. Do not put descriptive entities on feature filenames; they belong
in the sidecar.

Feature files do **not** carry a BIDS suffix (e.g., `_bold`). The `feat-<label>` entity
already identifies the data type; a suffix would be redundant, and no standard BIDS suffix
exists for derived feature files. Feature path validation rejects any path with a suffix.
`BIDSPath` itself makes suffix optional to allow this — see
[bidspath-validation.md](bidspath-validation.md).

Feature files do **not** carry acquisition entities (`acq`, `ce`, `rec`, `dir`). Features are
stimulus-derived — the same stimulus embedding applies regardless of scanner acquisition
parameters. Encoding validation enforces this: if BOLD files carry acquisition variants,
feature files are not required to mirror them.

## CellKey

`CellKey` is the open-schema row key for a feature time window. After enrichment it carries:

- Filename entities: `ses`, `run`, segment entity value (e.g. `trial=1`)
- Metadata entities from `events.json` `trial_type.Levels` (e.g. `cond=R`, `item=101`)

Excluded from `CellKey` (`CellKey.EXCLUDE`): `sub`, `task`, `acq`, `ce`, `rec`, `dir`
(invariant within a training call), `desc`, `res`, `den`, `echo` (BOLD image-variant
derivatives), `space`, `feat` (orthogonal axes). Entities are present or absent — no
`None` sentinel. Equality and hashing are order-independent.

CV splits are expressed by querying `CellKey` entities:

```python
train = [s for k, s in data.row_slices.items() if k["trial"] == "1"]
```

## Filename entity vs. events.json metadata

`events.json` is the authoritative source for descriptive metadata. Four cases:

- Sidecar-only (key in `trial_type.Levels` metadata, absent from filename): merged onto the resolved `CellKey`.
- Both, same value: allowed; redundant but harmless.
- Both, different value: raise — the two sources of truth disagree.
- Filename-only descriptive (key absent from `trial_type.Levels` metadata): raise, pointing user to events.json.

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

## Parquet metadata

Feature files carry a `hypline` JSON blob in the Parquet footer. Hypline reserves
keys for feature-type identity (validated against the `feat` filename entity on
read) and package-version provenance; callers must not supply them.

Caller-supplied keys should let a consumer reproduce the array — any generator
parameter that changes what was written (e.g. model name/version) belongs in
metadata. `dim_labels` (ordered per-dimension labels) is optional; include when
dimensions are nameable.

Keys prefixed with `_` are exempt from cross-file equality checks, reserved
for genuinely per-file metadata.

At encoding time, all feature files for the same (subject, feature) must have
identical `hypline` metadata; mismatches are rejected.

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
