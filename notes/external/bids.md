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
