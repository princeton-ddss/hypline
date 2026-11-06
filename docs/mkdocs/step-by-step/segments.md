# Segments and metadata

A single BOLD run often contains several parts you want to analyze separately, such
as trials or blocks. Hypline calls these **segments**. You define each segment's
start time and duration in an `events.tsv` file and can attach descriptive metadata,
such as its condition, in a companion `events.json` file.

This page explains how to define segments, annotate speaker turns, and attach
metadata. If you want to analyze an entire run as one window, you do not need
to declare segments. Speaker-turn annotations and descriptive metadata are
optional additions.

## Define segments in `events.tsv`

Hypline reads event timing from the BIDS `*_events.tsv` file under
each subject's `func/` directory:

```text
sub-041/ses-1/func/sub-041_ses-1_task-conv_run-1_events.tsv
```

You provide these files; hypline does not generate them. If your dataset
already contains events files, check their contents: an empty file or a table
with only standard event labels does not define segments for hypline. You can
create or update the tables using timing information from your experiment,
such as task logs.

Each row describes an event. Its `onset` is the start time in seconds relative
to the beginning of the run, and its `duration` is the event's length in seconds.
Hypline recognizes a row as a segment when its `trial_type` is an
`entity-value` pair, such as `trial-1`:

```tsv
onset   duration   trial_type
0.0     30.0       trial-1
35.0    30.0       trial-2
70.0    30.0       trial-3
```

This example defines three 30-second segments. Plain labels such as `rest`
and `fixation` do not declare segments, so you can retain those annotations
in the same table.

Events files are organized by **subject**, while commands such as `featuregen`
and `confoundgen` process the shared conversation by **dyad**. Hypline uses
`participants.tsv` to find the partners' events
files and can read the segment definitions from either partner. Keep segment
labels, onsets, durations, and `events.json` metadata consistent across both
partners' files. Their speaker-turn rows, described below, differ by subject.

!!! info "One segment entity per run"

    Use the same entity name for every segment in a run: all `trial-*`, for
    example, or all `block-*`. Do not mix them. This name is the run's
    **segment entity**. Choose the finest level you need to analyze; put
    broader groupings, such as condition, in metadata.

### Three kinds of analysis window

| Run type             | `events.tsv` contains                                        | Result                                           |
| -------------------- | ------------------------------------------------------------ | ------------------------------------------------ |
| **Unsegmented**      | no file, or no `entity-value` rows                           | one window covering the entire run               |
| **Segmented**        | one or more `entity-value` rows (e.g., `trial-1`, `trial-2`) | one window per segment                           |
| **Single-window**    | exactly one `task-<name>` row                                | one window with the specified onset and duration |

Use one explicit window when you want to exclude leading instructions or
trailing padding, or attach metadata to an otherwise unsegmented run. Its
`trial_type` must repeat the task entity in the run's filename. For a
`task-conv` run, for example:

```tsv
onset   duration   trial_type
10.0    120.0      task-conv
```

This selects the 120 seconds beginning 10 seconds into the run.

## Annotate speaker turns

You can also use `events.tsv` to indicate when each subject is the assigned
speaker. In each subject's own events file, add one row with `trial_type` set to
`turn_speaker` for each of their turns.

For example, in `sub-041_ses-1_task-conv_run-1_events.tsv`:

```tsv
onset   duration   trial_type
0.0     30.0       trial-1
0.0     10.5       turn_speaker
16.1    13.1       turn_speaker
35.0    30.0       trial-2
35.1    9.7        turn_speaker
45.1    10.9       turn_speaker
70.0    30.0       trial-3
```

A `turn_speaker` row means "the subject named in this events filename is the
assigned speaker," so these rows mark **subject 041's turns**. Subject 042's
turns belong in subject 042's events file. You do not need a separate label for
listening: `transcribe` reads both partners' files and combines their turn
annotations.

- Use the run's clock. If a trial starts 35 seconds into the run and a
  turn starts 5 seconds into that trial, write `onset = 40`, not `5`.
  Use run-relative times even when your audio is split into per-trial files;
  hypline accounts for the segment offset when matching transcript words to turns.
- Each turn covers `[onset, onset + duration)`: it includes the onset but not
  the end time.
- Turns must not overlap, within a subject or across partners. Overlap across
  partners is treated as cross-talk and raises an error. We're planning to
  develop hypline functionality to allow for speaker overlap in the future.
- Gaps with no assigned speaker are allowed.

For each word, hypline checks which annotated turn contains the word's start
time and writes that subject's label, without the `sub-` prefix, to the
transcript's `turn_sub` column. For example, a word beginning during subject 041's turn gets
`turn_sub = "041"`; one beginning during subject 042's turn gets `"042"`.

The value is missing when the word has no start time or begins in a gap between
annotated turns. Hypline logs a warning with the number of gap words in each
file so you can check for timing or annotation mismatches. If neither partner's events file contains
`turn_speaker` rows, the transcript still has a `turn_sub` column, but all its
values are missing.

!!! note "Turn labels reflect the annotations"

    Turn assignment follows your annotated windows, rather than identifying
    each voice independently. If subject 042 says a brief "yeah" during
    subject 041's assigned turn, that word is also labeled `041`. Overlapping
    speech in the recording is different from overlapping turn annotations,
    which hypline rejects.


## Attach metadata in `events.json`

A segment label such as `trial-1` identifies a time window but does not say
which condition or stimulus it represents. Add these descriptive attributes
to the companion `events.json` file, under `trial_type.Levels`, using a
`metadata` entry for each segment:

```json title="sub-041_ses-1_task-conv_run-1_events.json"
{
  "trial_type": {
    "Levels": {
      "trial-1": {"metadata": {"condition": "G", "item": "101"}},
      "trial-2": {"metadata": {"condition": "R", "item": "102"}},
      "trial-3": {"metadata": {"condition": "R", "item": "103"}}
    }
  }
}
```

The keys (`trial-1`, `trial-2`, and `trial-3`) match the segment labels in
`events.tsv`. Hypline attaches the corresponding metadata to each segment.
Here, trial 1 has condition `G`, while trials 2 and 3 have condition `R`.

!!! note "Keep filenames structural, metadata descriptive"

    Filenames carry only *identity* (`sub`, `task`, `run`) and the *segment*
    (`trial-1`). Descriptive attributes (`condition`, `item`) belong in `events.json`,
    not in filenames. Hypline enforces this split — it keeps the same attribute
    from being recorded in two places that could disagree.

### Extra entities in filenames

Besides identity entities (`sub` or `dyad`, `ses`, `task`, `run`), the segment
entity, category entities (`feat`, `conf`, `nuis`, `result`), and variant
entities (`desc`, `space`, `res`, `den`), every entity in a filename must also appear as a key in the segment's
`metadata` in `events.json`. For example, if a run's segments are `trial-*`, an
audio file named `..._run-1_block-1_trial-3_audio.wav` carries an extra entity,
`block`. If no Levels entry has a `block` key, hypline stops with:

```text
Filename carries entities ['block'] absent from events.json Levels metadata — descriptive attributes must live in events.json, not filenames
```

This check runs during `encoding train` and whenever you pass `--data-filters`.
Entities in audio filenames carry into transcript and feature filenames, so fix
them before running `transcribe`. Either:

- Remove the entity from the filenames if you do not need it. Make sure the
  shorter names do not collide.
- Or record it in `events.json`: add `"block": "1"` (with the right value) to
  the `metadata` of every Levels entry. The value must match the filename, or
  hypline stops with `Filename and events.json disagree on 'block'`.

An unsegmented run has no Levels entries to hold the key. Either remove the
entity, or add a single `task-<name>` window row (see
[Three kinds of analysis window](#three-kinds-of-analysis-window))
and record the attribute in its metadata.

## Use segments in later analyses

These definitions let later commands process individual segments or select
them by metadata. For example, you can select trials in condition `G`
without adding the condition to their filenames.

See [Filter to specific runs or conditions](../FAQ/filter.md) for
command syntax and filtering examples.

## Why segment boundaries must be explicit

Hypline requires you to declare segment onsets and durations directly in
`events.tsv` rather than inferring them from trial-level columns. Encoding
models bin BOLD timepoints into segments, and inferring those boundaries from
indirect cues would quietly change which timepoint lands in which segment. That
is a scientific decision, so making the windows explicit keeps it yours and
reviewable.
