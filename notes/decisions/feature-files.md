# Feature files

Design record for hypline's feature file format — schema, naming rules, and
alignment contract with BOLD data.

Hypline's own derivative type — stimulus-derived features paired to BOLD
runs for encoding models.

Feature I/O lives in `hypline.io` and is re-exported at `hypline.*`. Writes
are entity-based (`save_feature`) so files land in the canonical layout
location without the caller typing a path. Reads are path-based
(`read_feature`, `read_feature_metadata`) because the realistic user case
is "I already have a file." A path-based writer `write_feature` exists for
in-package generators that already hold a layout-derived path; it is not
re-exported at `hypline.*`. The same shape applies to confound I/O — see
[confound-files.md](confound-files.md). TR alignment is handled by
`hypline.downsample` (also shared with confound files).

## Format

- Extension: `.parquet`
- Required columns:
  - `start_time` (float, seconds from the beginning of the source file — see [Temporal alignment](#temporal-alignment))
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
call `downsample`, which bins on `start_time`.

## Missing-unit rows

When a generator's upstream source yields a timestamped item but no feature
unit can be derived from it (OOV token, empty tokenization, punctuation-only
input, etc.), emit **one row** at that `start_time` with the unit-identifying
column set to `None` and a zero `feature` vector. Do **not** skip the row.

Why: `downsample` bins by `start_time` against TRs. Dropping rows
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

Feature files do **not** carry acquisition entities (`acq`, `ce`, `rec`, `dir`, `echo`,
`part`, `chunk`) — hypline rejects these project-wide at `BIDSPath` construction. See
[unsupported-entities.md](unsupported-entities.md).

### Optional `desc` variant tag

Feature generators may accept an optional `desc` argument that lands as a
`desc-<label>` entity on the output filename, distinguishing features generated
from the same source under different generator settings.

`desc` stays in `CellKey.EXCLUDE` — it labels a settings variant, not identity.
Reproducibility is enforced at encoding time by Parquet-metadata equality across
files sharing the same `(feat, desc)` pair, not by the tag itself; mismatched
settings under the same `desc` fail that check loudly.

`desc-*` variants share the same row set and `start_time` column — they carry
alternative feature *values* over a common event grid. A variant that changes
the timing (different unit selection, different alignment source) is a
different event set and must be a distinct `feat-<kind>`, not a `desc-*` of the
existing one. This invariant lets consumers that depend only on `start_time`
(e.g. timing-based confound generation) treat all `desc-*` variants of a
`feat-<kind>` as equivalent.

This grid invariant is **caller-guaranteed and unenforced** — no layer checks
it. `save_feature` writes one file from one DataFrame and has no view of sibling
variants, so a divergent grid is never caught at write time; checking at read
time would mean reading every variant's `start_time` only to discard all but
one, defeating the point of sourcing a single variant. A variant mislabeled with
a divergent grid is therefore silently dropped, not detected. Unlike the
metadata-equality check above (which guards against silent config drift),
violating this invariant requires actively mislabeling a different event set as
a `desc-*` — a gross authoring error, not subtle drift — so it is left to the
author rather than validated.

## CellKey

`CellKey` is the open-schema row key for a feature time window. After enrichment it carries:

- Filename entities: `ses`, `task`, `run`, segment entity value (e.g. `trial=1`)
- Metadata entities from `events.json` `trial_type.Levels` (e.g. `cond=R`, `item=101`)

Excluded from `CellKey` (`CellKey.EXCLUDE`): `sub` (invariant within a training call),
`desc`, `res`, `den` (BOLD image-variant derivatives), `space`, `feat` (orthogonal axes).
`task` flows through as a cell axis — single-task calls leave it constant, multi-task calls
(`Encoding(tasks=["A","B"])`) yield distinct cells per task. Entities are present or absent
— no `None` sentinel. Equality and hashing are order-independent.

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

For unsegmented runs (no events.tsv key-value rows), only entities in `STRUCTURAL_ENTITIES`
(BOLD identity, hypline category tags `stim`/`feat`/`conf`, and image-variant descriptors
`desc`/`space`/`res`/`den`) are valid on filenames — any other entity raises, since
descriptive attributes belong in `events.json` `Levels` metadata. For segmented runs the
segment entity is additionally permitted.

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
read), package-version provenance, and feature-vector dimension (`feature_dim`);
callers must not supply them.

Caller-supplied keys should let a consumer reproduce the array — any generator
parameter that changes what was written (e.g. model name/version) belongs in
metadata. `feature_dim_labels` (ordered per-dimension labels) is optional; include when
dimensions are nameable.

Keys prefixed with `_` are exempt from cross-file equality checks, reserved
for genuinely per-file metadata.

At encoding time, all feature files for the same (subject, feature) must have
identical `hypline` metadata; mismatches are rejected.

## Temporal alignment

`start_time` is **source-relative**: seconds from the beginning of the source
file (audio clip, transcript, etc.) that the feature file was derived from —
not from the BOLD run onset.

Rationale: stimuli are typically already segmented (one audio file per trial),
so the natural reference is the trial itself. Requiring run-absolute
timestamps would force generators to know the BIDS segment structure, coupling
stimulus processing to the BIDS layout. The `onset` column in events.tsv is
the authoritative bridge between segment-local and run time; encoding reads it
to align Y (BOLD) and does not apply it to X (features).

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

**Assumption**: events.tsv `onset`/`duration` are authored against the raw
BOLD timeline. `Segment.onset`/`duration` (parsed in `events._parse_segments`)
are in seconds and carry these values through unchanged. TR-index conversion
happens at the call site via `events.segment_tr_slice`. If fmriprep trims
dummy scans (see [../modules/bold.md](../modules/bold.md) and
[../external/fmriprep.md](../external/fmriprep.md)), the resulting TR indices
are offset relative to the trimmed image — callers must shift `Segment.onset`
before calling `segment_tr_slice` if they need trim-aware indices.
