# Reading encoding results

The [`encoding`](encoding.md) CLI produces two kinds of results:

- **evals** from `analyze`, containing the per-voxel encoding scores used 
  for downstream analysis;
- **model artifacts** from `train`, 
  containing fitted models and the recipe used to construct them.

Load both from `hypline.encoding`:

```python
from hypline.encoding import load_eval, load_artifact
```

## Load an eval

An eval (`analyze`'s output) loads as an
[`xarray.Dataset`](https://docs.xarray.dev/). This is the usual starting 
point for downstream analysis because it contains the per-voxel encoding scores.

```python
ds = load_eval("data/results/sub-031/encodingEval-selfeval/sub-031_result-encodingEval_desc-selfeval.nc")
```

The main variable is `corr`, a four-dimensional array:

```python
ds["corr"].dims      # ('fold', 'band', 'role', 'voxel')
```

| Axis    | What one position along it means                                        |
| ------- | ----------------------------------------------------------------------- |
| `fold`  | One cross-validation fold — the model scored on the data it held out.   |
| `band`  | One part of the model: the task offset, a feature, or the confounds.    |
| `role`  | Which turns were scored: production, comprehension, or either.          |
| `voxel` | One location in the brain.                                              |

Three of the four carry named labels you can select on; `voxel` is a bare
integer index, since an eval has no real voxel identifiers to attach. The sections 
below explain these dimensions and how to select the scores needed for an analysis.

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
[FIR delays](../FAQ/how-encoding-works.md) let one turn's signal spill onto the first
rows of the next, and those boundary rows are dropped from both so neither role
is contaminated. `both` keeps them, which is why it is not simply `prod` plus
`comp`. A role with no rows in a fold (a run where the target never listened,
say) scores `NaN`, not zero. That way it can be skipped when you average across
folds rather than dragging the average down.

### `fold`: the cross-validation folds

`fold` is a plain integer index, `0` upward. It does not tell you which run
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

### The attributes: what analysis this is

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

The identities of `source`, `model`, and `target` determine what an eval means. 
The same model file says very different things depending on how `source`, `model`, 
and `target` line up, and that choice has its own page:
[Choosing source and model](../FAQ/how-encoding-works.md#choosing-source-and-model).

### What the scores are, and are not

The values in `corr` are himalaya **split scores**: each band's own share of the
joint prediction's accuracy. Two things follow from that, and both are easy to
get wrong:

- A split score is not a plain Pearson correlation and need not land in
  `[-1, 1]`. A single band's value can even be negative. Read it as that band's
  relative encoding strength, not as a fraction of variance explained.
- The scores are a decomposition of the joint model's correlation, so the bands
  do sum to it: `r_joint = r_screens + r_semantic + … + r_confounds`. That makes
  the sum the joint model's score — but a single band's share is not itself a
  correlation, so do not read one band's value on that scale.

!!! success "A first look at your eval"

    Loaded, a typical within-subject eval subsets like this — the production
    score for one feature, averaged over folds:

    ```python
    ds["corr"].sel(band="semantic", role="prod").mean("fold", skipna=True)
    ```

    From here, [Choosing source and model](../FAQ/how-encoding-works.md#choosing-source-and-model)
    explains how the subject wiring behind an eval changes what a score like this
    tells you.

To compare within-brain, cross-brain, and pseudo-dyad evals, see
[Compare cross-brain fits with a baseline](../FAQ/analyze-evals.md).


## Load a model artifact

A model artifact (`train`'s output) loads as an `EncodingArtifact`, which 
contains the fitted weights and the recipe needed to inspect or reuse the model:

```python
artifact = load_artifact("data/results/sub-031/encodingModel-v1/sub-031_result-encodingModel_desc-v1.joblib")

artifact.recipe      # the XRecipe: features, confounds, delays, alphas, split, …
artifact.models      # one FittedModel per fold (its pipeline + the cells it was fit on)
artifact.fold        # the FoldSpec, or None for a single unfolded model
```

`load_artifact` warns, but does not fail, if the artifact was 
written by a different hypline version. Treat this as provenance 
information rather than a hard incompatibility.

## API reference

API documentation for loading, inspecting, and saving encoding results.

### Loading

::: hypline.encoding.load_eval

::: hypline.encoding.load_artifact

### Result types

The artifact structure and its parts.

::: hypline.encoding.EncodingArtifact

::: hypline.encoding.XRecipe

::: hypline.encoding.FittedModel

::: hypline.encoding.FoldSpec

### Saving

The CLI commands write results for you. Use these functions directly 
if you need to save an eval or model artifact yourself.

::: hypline.encoding.save_eval

::: hypline.encoding.save_artifact
