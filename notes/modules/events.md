# Events / segments

Domain semantics behind `hypline.events` — events.tsv parsing, segment
loading, and TR-index conversion. Separated from `bold` so events-aware code
paths can run without a BOLD file on disk.

## Why a separate module from `bold`

Events parsing and Levels-metadata validation live here so they don't
depend on a BOLD file being present. `bold.load_bold_meta` delegates to
`load_segments` for its events portion. The BIDSPath-source signature
(`load_segments(layout, source)` accepts any path carrying a `task`
entity, resolving events sidecars from the raw tree via
`layout.path.raw`) also leaves room for future stimulus- or feature-only
callers to reuse it without going through `bold`.

## Segment shape

`Segment` carries seconds-based `onset` / `duration` (mirroring events.tsv
exactly), not TR indices. The events domain is independent of TR — TR is an
acquisition property owned by `bold`, and TR-index conversion is an
encoding-pipeline concern. See
[../decisions/feature-files.md](../decisions/feature-files.md) for the
boundary convention and the dummy-scan-trim interaction.

## TR-index conversion

`segment_tr_slice(segment, repetition_time)` is the one allowed conversion
helper. It preserves the left-inclusive `[t*TR, (t+1)*TR)` convention:
`onset=0.0` lands in TR 0; `onset=k*TR` lands in TR `k`, not `k-1`. Pinned
by a dedicated test in `tests/unit/test_events.py`.

The helper takes no trim offset. If a caller needs trim-aware indices, it
must shift `Segment.onset` before calling. This keeps the helper's contract
identical for the common (no-trim) case and makes any trim handling
explicit at the call site.

## Entry point

Callers always go through `load_segments`, which returns `[]` for
unsegmented runs.

## Speaking turns

`hypline.events` also owns speaking-turn parsing (`load_turns`,
`stamp_turns`), consumed only by `transcribe`. Turns use the flat
`turn_speaker` `trial_type` label, which stays outside `BIDS_ENTITY_RE` and so
never enters the Segment/encoding path. Unlike segment loading — which picks
the dyad's first partner — `load_turns` reads *every* partner and unions their
windows, because turn info is complementary across partners rather than
identical. The cross-partner-overlap-as-cross-talk rule and the timeline
invariant that licenses the union live in
[../decisions/dyad-keyed.md](../decisions/dyad-keyed.md).

`turn_speaker` onsets are **run-relative** (whole-run events.tsv clock), but a
transcript's `start_time` is **frame-local** (0.0 = the segment's start, since
each per-trial audio file is decoded independently). `stamp_turns` therefore
takes a required `frame_onset` and lifts each word's time by it before matching
— never compare the two frames directly. `frame_onset(layout, source)` supplies
it: the matching segment's run-relative onset, or `0.0` for an unsegmented
whole-run source (times already run-relative). A segmented source whose segment
value is missing/unknown is malformed and raises (mirrors `resolve_entities`).

## Filename ↔ sidecar merge

The four-case merge contract between filename entities and `events.json`
`Levels.metadata` (sidecar-only adopted, both-same kept, both-differ raises,
filename-only descriptive raises) lives here as `merge_filename_and_sidecar`.
`resolve_entities(layout, source)` wraps it for arbitrary BIDSPaths, locating
the segment by the filename's segment-entity value. Encoding's
`_resolve_cell_keys` and `BIDSLayout.find.*` descriptive-filter matching both
delegate here — do not re-implement the cases at the call site.

See [../decisions/feature-files.md](../decisions/feature-files.md#filename-entity-vs-eventsjson-metadata)
for the contract statement.
