# Segments and metadata

A single BOLD run often contains several distinct stretches you care about —
trials, blocks, conditions. Hypline calls these **segments**, and it reads them
from your `events.tsv` files. Segments are what let you generate per-trial
features and filter your data down to specific conditions with `--data-filters`.

This page explains where segments come from and how `--data-filters` uses them.
You only need it if your runs have internal structure; whole-run datasets can
skip it.

## Where segments come from: `events.tsv`

Hypline reads run structure from the standard BIDS `events.tsv` file that sits
beside each run in the raw tree:

```
sub-003/ses-1/func/sub-003_ses-1_task-conv_run-1_events.tsv
```

Events files are sub-keyed. A dyad-keyed command (`featuregen`, `confoundgen`)
resolves them through [`participants.tsv`](layout.md#subject-vs-dyad) and reads
either partner's events — segments are shared across a dyad by construction.

Hypline infers segments from the `trial_type` column. A row whose `trial_type`
is an `entity-value` pair (like `trial-1`) **declares a segment** — a named time
window within the run:

```tsv
onset   duration   trial_type
0.0     30.0       trial-1
35.0    30.0       trial-2
70.0    30.0       trial-3
```

This run has three segments. Plain labels (e.g. `rest`, `fixation`) are *not*
segments — they are ignored for segmentation, so you can keep standard
annotations in the same file.

!!! info "One segment entity per run"

    All segment rows in a run must use the **same** entity name — all `trial-*`,
    or all `block-*`, never a mix. That single name becomes the run's *segment
    entity*. Use the finest level you need (e.g. `trial`); coarser groupings go
    in metadata, described below.

### Three kinds of run

| Run type             | `events.tsv` contains                       | Result                          |
| -------------------- | ------------------------------------------- | ------------------------------- |
| **Unsegmented**      | no file, or no `entity-value` rows          | one window = the whole run      |
| **Segmented**        | one or more `entity-value` rows             | one window per segment          |
| **Single-window**    | exactly one `task-<name>` row               | trims padding / holds metadata  |

The last case is an escape hatch: when a run has no internal structure but you
still want to trim leading instructions or attach run-level metadata, add a
single row whose `trial_type` repeats the run's task name (e.g. `task-conv`).

## Attaching metadata: `events.json`

Segment names like `trial-1` carry no meaning on their own. Descriptive
attributes — condition, stimulus item, counterbalance group — live in the
companion `events.json` sidecar, under the BIDS `trial_type.Levels` field:

```json title="sub-003_ses-1_task-conv_run-1_events.json"
{
  "trial_type": {
    "Levels": {
      "trial-1": {"metadata": {"cond": "R", "item": "101"}},
      "trial-2": {"metadata": {"cond": "L", "item": "102"}},
      "trial-3": {"metadata": {"cond": "R", "item": "103"}}
    }
  }
}
```

Now each segment has a condition (`cond`) and an item (`item`). Hypline merges
this metadata onto each segment when it processes the run, so you can filter on
`cond` even though it never appears in any filename.

!!! note "Keep filenames structural, metadata descriptive"

    Filenames carry only *identity* (`sub`, `task`, `run`) and the *segment*
    (`trial-1`). Descriptive attributes (`cond`, `item`) belong in `events.json`,
    not in filenames. Hypline enforces this split — it keeps the same attribute
    from being recorded in two places that could disagree.

## How segments reach `--data-filters`

The segments and metadata defined here are what `--data-filters` selects on. A
token like `cond-R` matches against **both** filename entities and the
`events.json` metadata — so the segment entity (`trial`) and its descriptive
attributes (`cond`, `item`) all become things you can filter by, even though only
the structural ones appear in filenames.

For how tokens combine (OR within an entity, AND across entities) and the full
set of recipes, see [Filter to specific runs or
conditions](../how-to/filter.md).

## Why segments are explicit

Hypline requires you to declare segment onsets and durations directly in
`events.tsv` rather than inferring them from trial-level columns. Encoding
models bin BOLD timepoints into segments, and inferring those boundaries from
indirect cues would quietly change which timepoint lands in which segment — a
scientific decision. Making the windows explicit keeps that decision yours and
reviewable.
