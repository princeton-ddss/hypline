# Python API

Most of hypline is the command-line pipeline, but a small Python API is
re-exported at the top level for reading and writing hypline's Parquet files
directly:

```python
from hypline import (
    read_feature,
    read_feature_metadata,
    read_confound,
    read_confound_metadata,
    save_feature,
    save_confound,
)
```

The two halves are deliberately asymmetric. **Saves are entity-based** —
you pass `bids_root` plus BIDS entities (`dyad`, `feat`/`conf`, `run`, …) and
hypline derives the canonical output path for you, so writes always land where
the pipeline expects. Features and confounds describe the shared conversation, so
they are keyed by `dyad`, not `sub` — see [Subject vs.
dyad](../concepts/layout.md#subject-vs-dyad). **Reads are path-based** — you
usually already have a file in hand. Both enforce the [dataset
layout](../concepts/layout.md) and the file formats; a malformed DataFrame or
path raises rather than writing something the CLI can't later consume.

## Plugging in a custom feature

The CLI only generates `feat-phonemic` features. To drive an encoding model on
a feature hypline doesn't compute — word embeddings, prosody, anything you can
align to the stimulus — build the DataFrame yourself and `save_feature` it into
the dataset. From then on it is a first-class feature: it sits under
`features/`, carries the right entities, and any downstream step that reads
features by name will find it.

A feature DataFrame needs two columns: `start_time` (seconds from the start of
the stimulus) and `feature` (one equal-width vector per row). Match the
`start_time` convention hypline already uses — see the
[feature file format](featuregen.md#outputs).

```python
import polars as pl
from hypline import save_feature

df = pl.DataFrame(
    {
        "start_time": [0.0, 0.48, 0.91],
        "feature": [[0.1, 0.2, 0.3], [0.0, 0.5, 0.1], [0.4, 0.4, 0.2]],
    }
)

path = save_feature(
    df,
    bids_root="data/",
    dyad="101",
    feat="embed",
    task="conv",
    run="1",
)
```

This writes `data/features/dyad-101/embed/dyad-101_task-conv_run-1_feat-embed.parquet`.
Pass `desc="..."` to tag a variant into its own
[`embed-<desc>/` subdirectory](../concepts/layout.md#variants-with-desc), and
`metadata={...}` to stash extra keys in the Parquet footer.

!!! note "Custom confounds need TR alignment"

    `save_confound` is the confound-side parallel, but a confound is regressed
    out of the BOLD, so its rows must align to the BOLD's TR grid: `start_time`
    must begin at `0.0` and step by `repetition_time`, which you pass
    **explicitly** (a single-row table carries no spacing to infer it from).
    See [Segments and metadata](../concepts/segments.md) for how TR-aligned
    confounds relate to the run.

## Round-tripping

Read a feature back into a DataFrame, or peek at its footer metadata without
loading the data:

```python
from hypline import read_feature, read_feature_metadata

df = read_feature(path)
meta = read_feature_metadata(path)   # feature_name, feature_dim, hypline_version, …
```

The reads validate the path and cross-check the footer against the path
entities, so a file that round-trips through `read_feature` is one the pipeline
will accept.

## Reference

### Writing

::: hypline.save_feature

::: hypline.save_confound

### Reading

::: hypline.read_feature

::: hypline.read_confound

### Metadata

::: hypline.read_feature_metadata

::: hypline.read_confound_metadata
