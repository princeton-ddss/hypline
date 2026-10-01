# Features and confounds

Most hypline workflows use the command-line interface. Hypline also exposes a small 
set of functions directly from the hypline package for reading and writing its Parquet files.

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

Reading and writing work slightly differently:

- **Saves are entity-based**. You provide `bids_root` and the relevant BIDS entities (`dyad`, `feat`/`conf`, `run`, …).
  Hypline constructs the canonical output path for you, so that the file lands where
  downstream commands expect it.
- **Reads are path-based**. You provide the path of an existing Parquet file.

Both operations validate the [Hypline dataset layout](layout.md) and the expected file format. Invalid paths,
metadata, or DataFrames raise an error instead of producing a file that hypline
cannot read.

Fitted encoding models and evaluation results use separate loaders under
`hypline.encoding`. See [Encoding results](encoding-results.md).

## Save a custom feature

The CLI can generate `phonemic`, `semantic`, `spectral`, and
`syntactic` features. To use another set of features (e.g., prosody), 
you can construct the feature DataFrame yourself and save it with `save_feature`. 

A feature DataFrame requires two columns: 

- `start_time`: the time in seconds from the beginning of the stimulus
- `feature`: an equal-width feature vector for that time point

Follow the same `start_time` convention as hypline's generated features. See the
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
    dyad="040",
    ses="1",
    feat="embed",
    task="conv",
    run="1",
)
```

This writes `data/features/dyad-040/ses-1/embed/dyad-040_ses-1_task-conv_run-1_feat-embed.parquet`.
The saved file is now part of the hypline dataset, and downstream commands can
locate it using the feature name `embed`.

Use `desc="..."` to save a variant in its own
[`embed-<desc>/` subdirectory](layout.md#desc-variants). Use
`metadata={...}` to add custom keys to the Parquet footer.


## Save a custom confound

`save_confound` is the corresponding function for custom confounds. Unlike 
a feature, a confound is regressed from the BOLD signal and must therefore
contain one row per fMRI volume.

A confound DataFrame requires two columns: 

- `start_time`: the time in seconds from the beginning of the stimulus
- `confound`: an equal-width confound vector for that time point

The `start_time` values must begin at `0.0` and advance by the run's
repetition time. Mark your TR explicitly with `repetition_time`. `tr_method`
records how the confound was converted to one row per fMRI volume (e.g., `"mean"` if you averaged
signals within each TR), and is required. You can put `None` if it's not applicable (i.e., your confound was
already one row per TR). For each confound, every run must have the same `tr_method` info. Otherwise, `encoding train` 
will complain about inconsistent metadata.

```python
import polars as pl
from hypline import save_confound

df = pl.DataFrame(
    {
        "start_time": [0.0, 1.5, 3.0],
        "confound": [[0.9, 0.01, 0.03], [0.0, 0.2, 0.5], [0.1, 0.2, 0.6]],
    }
)

path = save_confound(
    df,
    bids_root="data/",
    dyad="040",
    ses="1",
    conf="newconf",
    task="conv",
    run="1",
    repetition_time=1.5,
    tr_method="mean",
)
```


## Read data and metadata

Load the complete feature table with `read_feature`:

```python
from hypline import read_feature

df = read_feature(path)
```

To inspect the Parquet footer without loading the feature values, use
`read_feature_metadata`:

```python
from hypline import read_feature_metadata

meta = read_feature_metadata(path)   # feature_name, feature_dim, hypline_version, …
```

The corresponding confound functions are `read_confound` and
`read_confound_metadata`.

Read operations validate the path and cross-check its entities against the
Parquet metadata. A file that passes `read_feature` or `read_confound` therefore
satisfies hypline's feature- or confound-file validation.

## API reference

### Writing

::: hypline.save_feature

::: hypline.save_confound

### Reading

::: hypline.read_feature

::: hypline.read_confound

### Metadata

::: hypline.read_feature_metadata

::: hypline.read_confound_metadata
