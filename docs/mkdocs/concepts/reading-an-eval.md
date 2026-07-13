# Reading an encoding result

When `hypline encoding analyze` scores a model, it writes an **eval**: a small
netCDF file of per-voxel scores. You load it back with
[`load_eval`](../reference/encoding-results.md) and get an
[`xarray.Dataset`](https://docs.xarray.dev/) with a single variable, `corr`.
This page explains what `corr` holds, so you can subset it to the exact scores a
question needs and read them correctly.

If you have not run an analysis yet, the [tutorial](../tutorials/walkthrough.md)
walks through producing one; come back here when you have an eval in hand.

## The shape of `corr`

`corr` is a four-dimensional array. Every score in it sits at one point along
each of these axes:

```python
from hypline.encoding import load_eval

ds = load_eval("data/results/sub-031/encodingEval-selfeval/sub-031_result-encodingEval_desc-selfeval.nc")
ds["corr"].dims      # ('fold', 'band', 'role', 'voxel')
```

| Axis    | What one position along it means                                        |
| ------- | ----------------------------------------------------------------------- |
| `fold`  | One cross-validation fold — the model scored on the data it held out.   |
| `band`  | One part of the model: the task offset, a feature, or the confounds.    |
| `role`  | Which turns were scored: production, comprehension, or either.          |
| `voxel` | One location in the brain.                                              |

Three of the four carry named labels you can select on; `voxel` is a bare
integer index, since an eval has no real voxel identifiers to attach. The rest
of this page takes the labelled axes one at a time.

### `band`: the parts of the model

An encoding model is a **banded ridge**: each feature gets its own band, and
`analyze` reports a separate score for each. The band labels are, in order, a
reserved task band, then one label per feature, then a single confounds band if
the model had any confounds:

```python
ds.coords["band"].values
# array(['screens_band', 'phonemic', 'semantic', 'confounds_band'], dtype=object)
```

- **`screens_band`** is always first and always present. It carries the
  production and comprehension task offsets — the part of the model that soaks
  up the average difference in signal between speaking and listening, so the
  feature bands do not have to.
- **A band per feature**, named exactly as you passed it to `--features`
  (`phonemic`, `semantic-gpt3`, and so on). This is usually what you care about:
  each feature's own contribution to predicting the brain.
- **`confounds_band`** appears only when the model was trained with
  `--confounds`. Every confound shares this one band.

Select a single feature's scores by name:

```python
ds["corr"].sel(band="semantic")
```

!!! note "The production/comprehension split does not add bands"

    Training with the default prod/comp split does not create separate
    production and comprehension bands. It widens each existing band so the model
    can learn different weights for the two states, but the band labels stay the
    same. The production-versus-comprehension split you score on lives on the
    `role` axis below, not here.

### `role`: which turns were scored

A conversation alternates between the target subject speaking and listening, and
a model can predict those two states differently. So every score is also broken
out by **role**, derived from the target subject's own turns:

- **`prod`** — rows where the target was speaking, and only those.
- **`comp`** — rows where the target was listening, and only those.
- **`both`** — either: any row with speech active, from the target or the
  partner.

```python
ds["corr"].sel(role="prod")   # scores during the target's own speech
```

`prod` and `comp` are kept strictly separate. The model's
[FIR delays](how-encoding-works.md) let one turn's signal spill onto the first
rows of the next, and those boundary rows are dropped from both so neither role
is contaminated. `both` keeps them, which is why it is not simply `prod` plus
`comp`. A role with no rows in a fold — a run where the target never listened,
say — scores `NaN`, not zero. That way it can be skipped when you average across
folds rather than dragging the average down.

### `fold`: the cross-validation folds

`fold` is a plain integer index, `0` upward. It does **not** tell you which run
each fold held out; that lives in the dataset's attributes, one list of cells
per fold:

```python
ds.attrs["fold_cells"][0]     # the cells fold 0 was scored on (its held-out run)
```

To collapse the folds into one score per voxel, average across them — and use a
`NaN`-aware mean so an empty role in one fold does not poison the result:

```python
ds["corr"].mean("fold", skipna=True)
```

## The attributes: what analysis this is

The scores alone do not say whose model, whose speech, or whose brain produced
them. That provenance rides along in `ds.attrs`, so an eval file is
self-describing wherever it ends up:

```python
ds.attrs["model_sub"]    # whose trained weights were used
ds.attrs["source_sub"]   # whose speech built the prediction inputs
ds.attrs["target_sub"]   # whose brain the scores are measured against
ds.attrs["test_on"]      # 'OOS' for out-of-sample, else the cells named with --test-on
ds.attrs["delays"]       # the FIR delays the model used, in TRs
ds.attrs["bold_space"]   # the BOLD space it was scored in
```

The three subject roles are what fix an eval's meaning. The same model file says
very different things depending on how `source`, `model`, and `target` line up,
and that choice has its own page:
[Choosing source and model](how-encoding-works.md#choosing-source-and-model).

## What the scores are, and are not

The values in `corr` are himalaya **split scores**: each band's own share of the
joint prediction's accuracy. Two things follow from that, and both are easy to
get wrong:

- A split score is **not a plain Pearson correlation** and need not land in
  `[-1, 1]`. A single band's value can even be negative. Read it as that band's
  relative encoding strength, not as a fraction of variance explained.
- The scores are a decomposition of the joint model's correlation, so **the bands
  do sum to it**: `r_joint = r_screens + r_semantic + … + r_confounds`. That makes
  the sum the joint model's score — but a single band's share is not itself a
  correlation, so do not read one band's value on that scale.

!!! success "A first look at your eval"

    Loaded, a typical within-subject eval subsets like this — the production
    score for one feature, averaged over folds:

    ```python
    ds["corr"].sel(band="semantic", role="prod").mean("fold", skipna=True)
    ```

    From here, [Choosing source and model](how-encoding-works.md#choosing-source-and-model)
    explains how the subject wiring behind an eval changes what a score like this
    tells you.
