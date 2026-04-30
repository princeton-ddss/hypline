# BIDS — external spec reference

BIDS entity and filename conventions that hypline code depends on.

BIDS filenames are composed of `entity-value` segments joined by `_`, ending
in a suffix and extension, in a required order:

```
sub-01_ses-01_task-movie_run-1_space-T1w_desc-preproc_bold.nii.gz
```

## Entities we work with

| Entity   | Meaning                   | Presence                                |
|----------|---------------------------|-----------------------------------------|
| `sub`    | subject                   | always                                  |
| `ses`    | session                   | only when dataset has sessions          |
| `task`   | task name                 | always for functional data              |
| `acq`    | acquisition variant       | e.g. `acq-highres`                      |
| `run`    | run index within task/ses | only when task has multiple runs        |
| `echo`   | echo index                | multi-echo acquisitions                 |
| `space`  | reference space           | e.g. `T1w`, `MNI152NLin2009cAsym`       |
| `res`    | resolution                | resampled derivatives                   |
| `den`    | surface density           | surface derivatives                     |
| `desc`   | description / variant     | e.g. `desc-preproc`, `desc-brain`       |

`ses` and `run` being optional is a recurring source of subtle bugs. Code
keying only on `run` collides when a subject has multiple sessions with the
same run number. Always use `(ses, run)` as a composite key.

## Reserved entities to avoid as custom names

Some entity names look tempting for custom use but are already reserved:

- `seg` — anatomical segmentation labels
- `part` — magnitude/phase image disambiguation
- `chunk` — segmented acquisitions

Check the BIDS spec before introducing a new custom entity.

## Structural entities in hypline

Hypline does not register a fixed segment entity name. User-chosen BIDS key-value names in
events.tsv (e.g. `block`, `trial`) are inferred as the segment entity at discovery time —
exactly one entity name is allowed per run. See
[../decisions/semantic-entity.md](../decisions/semantic-entity.md).

## Sidecar naming — two categories

BIDS sidecars fall into two naming categories that hypline handles via separate code paths:

- **Per-file sidecars** (e.g. `*_bold.json`, `*_T1w.json`). Mirror the full stem of the
  data file they describe — including non-identity entities like `space`, `desc`. Resolved
  by stripping the imaging extension and swapping in `.json`. A run with multiple BOLD
  variants (different `space`, `desc`) has multiple `*_bold.json` files, one per variant.
  Implemented in `bold.get_repetition_time`.

- **Run-level sidecars** (e.g. `*_events.tsv`, `*_events.json`, `*_physio.tsv.gz`). Named
  using identity entities only (`sub`, `ses`, `task`, `acq`, `ce`, `rec`, `dir`, `run`) —
  describe the source data, invariant across `space`/`desc`/etc. One sidecar per run,
  shared by all derived variants. Implemented in `bold._resolve_run_sidecar`.

The discrepancy is mandated by BIDS (events.tsv may not carry `space`, `desc`, etc.; per-file
sidecars must mirror their target file). Conflating the two would either over-restrict
per-file sidecars (rejecting valid `space-T1w_bold.json`) or under-restrict run-level
sidecars (allowing divergent `space-T1w_events.tsv` and `space-MNI_events.tsv`).
