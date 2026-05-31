# Regenerate outputs after a fix

You ran part of the pipeline, then something upstream changed — you re-recorded
audio, fixed an `events.tsv`, or chose different confound columns. This guide
shows how to redo just the affected work without recomputing the whole dataset,
and — importantly — which **downstream** steps you must also rerun.

## Why a plain rerun does nothing

By default, hypline **skips any output that already exists**. This makes reruns
cheap, but it means a second run after a fix appears to do nothing: the outputs
are still there, so every step is skipped.

To overwrite, pass **`--force`**. Skipping is decided per output file, so
`--force` combined with [`--sub-ids` / `--data-filters`](filter.md) regenerates
**only** the subset you select — everything else is left untouched.

```bash
# regenerate features for run 1 only; all other runs stay as they were
hypline featuregen phonemic data/ --data-filters run-1 --force
```

!!! warning "Skipping is existence-only — there is no staleness detection"

    Hypline decides to skip by asking *“does this output file exist?”* — **not**
    *“is this output older than its inputs?”* It never compares timestamps. So
    regenerating an input does **not** mark anything downstream as stale.

    If a later step's output already exists, that step will **skip and leave its
    old result in place**, now computed from inputs you have since changed. You
    must rerun every downstream step with `--force` yourself. The propagation
    table below tells you which ones.

## What to rerun after each kind of fix

Find the row for what you changed; rerun the listed steps **in order**, each with
`--force` (scoped with filters as needed). Each step reads what the one before it
wrote, so a gap leaves stale output behind.

| You changed… | Rerun, in order (each `--force`) |
| ------------ | -------------------------------- |
| **Stimulus audio** | `transcribe` → `featuregen phonemic` → `denoise` |
| **An `events.tsv`** (segment onsets/durations) | `featuregen phonemic` → `denoise` |
| **An `events.json`** (metadata only, e.g. `cond`) | nothing to regenerate — metadata is read at filter time, not baked into outputs[^meta] |
| **fMRIPrep confound columns** (`--columns` / `--desc`) | `confoundgen fmriprep` → `denoise` |
| **fMRIPrep preprocessed BOLD** | `confoundgen fmriprep` → `denoise` |
| **Which confounds to regress** (`--confounds` on denoise) | `denoise` only |

[^meta]: `events.json` metadata (like `cond`) is matched by `--data-filters` when
    a command runs; it is never written into a filename or output. Changing it
    changes *which* files a future filter selects, not the contents of files
    already generated. (Segment onsets/durations in `events.tsv`, by contrast,
    *do* change generated confounds — so that row regenerates.)

!!! note "Why `featuregen phonemic`, not `confoundgen phonemic`, in those rows"

    `featuregen phonemic` emits both the features **and** the matching
    `conf-phonemic` confounds in one step, so `--force` refreshes both — no
    separate `confoundgen phonemic` call needed after an audio or `events.tsv`
    fix. The catch worth knowing: segment onsets and durations affect only the
    **confound** half. `feat-phonemic` carries per-word timing and is
    segment-agnostic, so re-emitting it is harmless rework, not a correctness
    fix; the segmenting that `events.tsv` drives happens when the confounds are
    built.

## Worked example: you fixed run 1's `events.tsv`

You corrected a segment onset in `sub-01_task-conv_run-1_events.tsv`. That run's
**phonemic confounds** are now wrong — they are sliced to the segment windows
your `events.tsv` defines — and the cleaned BOLD built on them is stale.

```bash
# 1. refresh phonemic confounds for just that run
#    (featuregen also re-emits feat-phonemic; harmless, segment-agnostic)
hypline featuregen phonemic data/ --sub-ids 01 --data-filters run-1 --force

# 2. rebuild the cleaned BOLD that consumed them
hypline denoise data/ \
  --space fsaverage6 \
  --confounds fmriprep-minimal,phonemic-onset \
  --sub-ids 01 --data-filters run-1 --force
```

Without step 2's `--force`, `denoise` would see the existing `desc-clean` file
and skip — leaving cleaned BOLD built from the old, wrongly-segmented confounds.

!!! tip "When in doubt, force the whole tail"

    If you are unsure how far a change propagates, rerun every step from the
    fix onward with `--force`. Hypline recomputes from scratch each time, so a
    broader `--force` is never *wrong* — only more expensive. Scope it with
    `--sub-ids` / `--data-filters` to keep the cost down.
