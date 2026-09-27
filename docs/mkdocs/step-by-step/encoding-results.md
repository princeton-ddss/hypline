# Encoding results

The [`encoding`](encoding.md) CLI produces two kinds of results; (1) 
**evals** from `analyze`, containing the per-voxel correlations used 
for downstream analysis; (2) **model artifacts** from `train`, 
containing fitted models and the recipe used to construct them.

Load both from `hypline.encoding`:

```python
from hypline.encoding import load_eval, load_artifact
```

## Load an eval

An eval (`analyze`'s output) loads as an
[`xarray.Dataset`](https://docs.xarray.dev/). This is the usual starting 
point for downstream analysis because it contains the per-voxel encoding correlations.

```python
ds = load_eval("data/results/sub-031/encodingEval-selfeval/sub-031_result-encodingEval_desc-selfeval.nc")

# corr is indexed by fold / band / role / voxel — subset by name:
prod_corr = ds["corr"].sel(role="prod")

# provenance rides in the dataset attributes:
ds.attrs["model_sub"], ds.attrs["target_sub"], ds.attrs["delays"]
```

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

## Reference

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

The CLI commands write results for you; these are the underlying seams, in case
you produce an eval or artifact yourself.

::: hypline.encoding.save_eval

::: hypline.encoding.save_artifact
