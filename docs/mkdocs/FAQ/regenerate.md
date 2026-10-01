# Regenerate outputs after a fix

You ran part of the pipeline, then something upstream changed: you re-recorded
audio, fixed an `events.tsv`, or chose different confound columns. This guide
shows how to redo just the affected work without recomputing the whole dataset,
and, importantly, which downstream steps you must also rerun.

## Why a plain rerun does nothing

By default, hypline skips any output that already exists. This makes reruns
cheap, but it means a second run after a fix appears to do nothing: the outputs
are still there, so every step is skipped.

To overwrite, pass `--force`. Skipping is decided per output file, so
`--force` combined with [the identity option (`--dyad-ids` / `--sub-ids`) /
`--data-filters`](filter.md) regenerates only the subset you select, leaving
everything else untouched.

```bash
# regenerate features for run 1 only; all other runs stay as they were
hypline featuregen phonemic data/ --data-filters run-1 --force
```

!!! warning "Skipping is existence-only — there is no staleness detection"

    Hypline decides to skip by asking *“does this output file exist?”*, never
    *“is this output older than its inputs?”* It never compares timestamps. So
    regenerating an input does not mark anything downstream as stale.

    If a later step's output already exists, that step will skip and leave its
    old result in place, now computed from inputs you have since changed. You
    must rerun every downstream step with `--force` yourself. The propagation
    table below tells you which ones.

## What to rerun after each kind of fix

Find the row for what you changed; rerun the listed steps in order, each with
`--force` (scoped with filters as needed). Each step reads what the one before it
wrote, so a gap leaves stale output behind. If you use `encoding`, end every
row except `events.json` with `encoding train` → `encoding analyze`, since
existing models were fit to the old inputs.

| You changed… | Rerun, in order (each `--force`) |
| ------------ | -------------------------------- |
| **Stimulus audio** | `transcribe` → every `featuregen` family you use |
| **An `events.tsv`** (segment onsets) | `transcribe` → every `featuregen` family you use |
| **An `events.tsv`** (segment durations only) | `featuregen` `phonemic`, `semantic`, `spectral` |
| **An `events.tsv`** (`turn_speaker` rows) | `transcribe` → `featuregen` `phonemic`, `semantic`, `syntactic` |
| **An `events.json`** (metadata only, e.g. `cond`) | nothing to regenerate — metadata is read at filter time, not baked into outputs[^meta] |
| **fMRIPrep preprocessed BOLD or its confounds table** | `denoise` |
| **Custom `nuisance/` files** | `denoise` |
| **Which nuisance regressors to regress** (`--columns` / `--compcor` / `--custom-sources` on denoise) | `denoise` |

[^meta]: `events.json` metadata (like `cond`) is matched by `--data-filters` when
    a command runs; it is never written into a filename or output. Changing it
    changes which files a future filter selects, not the contents of files
    already generated. (Segment and turn rows in `events.tsv`, by contrast, do
    change generated outputs — so those rows regenerate.)

!!! note "Why `featuregen`, not `confoundgen`, in those rows"

    `featuregen phonemic` and `featuregen semantic` also emit the matching
    `conf-phonemic` / `conf-semantic` confounds, so `--force` refreshes both —
    no separate `confoundgen` call needed.

    Which feature files actually change depends on the fix. Segment durations
    reach the confounds and `spectral`, which bins audio per segment. The
    word-level feature files do not change, so re-emitting them is harmless
    rework.

    Turn edits change each word's `turn_sub`, which `transcribe` writes and
    the word-level feature files carry. Moving a segment onset does too when
    turns are marked, since `transcribe` places turns relative to the segment
    start. `syntactic` parses one turn at a time, so its features change as
    well.

!!! note "Stimulus fixes do not propagate to `denoise`"

    `denoise` reads its nuisance regressors from fMRIPrep's confounds table and
    the `nuisance/` area — never from the stimulus-derived `confounds/`. So
    re-recording audio or fixing an `events.tsv` changes features and stimulus
    confounds but leaves `desc-denoised` BOLD untouched: there is no stimulus →
    `denoise` dependency to repair.

## Worked example: you decided to add CompCor regressors

You already cleaned the dataset, then decided the motion model needs CompCor
components. The `desc-denoised` BOLD on disk was built without them and is now
stale, but a plain rerun skips, because the output already exists.

```bash
hypline denoise data/ \
  --columns trans_x,trans_y,trans_z,rot_x,rot_y,rot_z,cosine \
  --compcor a:CSF:5,a:WM:5 \
  --force
```

Without `--force`, `denoise` would see the existing `desc-denoised` file and skip —
leaving denoised BOLD built from the old, CompCor-less regressor set. The same
applies after editing any `nuisance/` file you regress out. Then rerun
`encoding train` and `encoding analyze` with `--force`, since existing models
were fit to the old denoised BOLD.

!!! tip "When in doubt, force the whole tail"

    If you are unsure how far a change propagates, rerun every step from the
    fix onward with `--force`. Hypline recomputes from scratch each time, so a
    broader `--force` is never wrong — only more expensive. Scope it with
    `--dyad-ids` / `--sub-ids` / `--data-filters` to keep the cost down.
