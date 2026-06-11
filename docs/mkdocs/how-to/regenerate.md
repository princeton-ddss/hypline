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
`--force` combined with [the identity option (`--dyad-ids` / `--sub-ids`) /
`--data-filters`](filter.md) regenerates **only** the subset you select —
everything else is left untouched.

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
| **Stimulus audio** | `transcribe` → `featuregen phonemic` |
| **An `events.tsv`** (segment onsets/durations) | `featuregen phonemic` |
| **An `events.json`** (metadata only, e.g. `cond`) | nothing to regenerate — metadata is read at filter time, not baked into outputs[^meta] |
| **fMRIPrep preprocessed BOLD or its confounds table** | `denoise` only |
| **Custom `nuisance/` files** | `denoise` only |
| **Which nuisance regressors to regress** (`--columns` / `--compcor` / `--custom-sources` on denoise) | `denoise` only |

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

!!! note "Stimulus fixes do not propagate to `denoise`"

    `denoise` reads its nuisance regressors from fMRIPrep's confounds table and
    the `nuisance/` area — never from the stimulus-derived `confounds/`. So
    re-recording audio or fixing an `events.tsv` changes features and phonemic
    confounds but leaves `desc-denoised` BOLD untouched: there is no stimulus →
    `denoise` dependency to repair.

## Worked example: you decided to add CompCor regressors

You already cleaned the dataset, then decided the motion model needs CompCor
components. The `desc-denoised` BOLD on disk was built without them and is now stale
— but a plain rerun skips, because the output already exists.

```bash
hypline denoise data/ \
  --columns trans_x,trans_y,trans_z,rot_x,rot_y,rot_z,cosine \
  --compcor a:CSF:5,a:WM:5 \
  --force
```

Without `--force`, `denoise` would see the existing `desc-denoised` file and skip —
leaving denoised BOLD built from the old, CompCor-less regressor set. The same
applies after editing any `nuisance/` file you regress out.

!!! tip "When in doubt, force the whole tail"

    If you are unsure how far a change propagates, rerun every step from the
    fix onward with `--force`. Hypline recomputes from scratch each time, so a
    broader `--force` is never *wrong* — only more expensive. Scope it with
    `--dyad-ids` / `--sub-ids` / `--data-filters` to keep the cost down.
