# Encoding results

The [`encoding`](encoding.md) CLI writes results; hypline's `encoding` package
reads them back for downstream analysis — the encoding-side parallel to
[`read_feature`](python-api.md). Imports come from `hypline.encoding` (not
top-level `hypline`), which pulls in the encoding stack:

```python
from hypline.encoding import load_eval, load_artifact
```

**An eval** (`analyze`'s output) loads as an
[`xarray.Dataset`](https://docs.xarray.dev/) — the usual downstream target,
since it holds the per-voxel correlations you analyze:

```python
ds = load_eval("data/results/sub-01/encodingEval-self-eval/sub-01_result-encodingEval_desc-self-eval.nc")

# corr is indexed by fold / band / role / voxel — subset by name:
prod_corr = ds["corr"].sel(role="prod")

# provenance rides in the dataset attributes:
ds.attrs["model_sub"], ds.attrs["target_sub"], ds.attrs["delays"]
```

**A model artifact** (`train`'s output) loads as an `EncodingArtifact` — the
fitted weights and the recipe, for reusing or inspecting the model:

```python
artifact = load_artifact("data/results/sub-01/encodingModel-v1/sub-01_result-encodingModel_desc-v1.joblib")

artifact.recipe      # the XRecipe: features, confounds, delays, alphas, split, …
artifact.models      # one FittedModel per fold (its pipeline + the cells it was fit on)
artifact.fold        # the FoldSpec, or None for a single unfolded model
```

`load_artifact` warns (does not fail) if the artifact was written by a different
hypline version — a provenance signal, not a hard incompatibility.

## Reference

Full signatures and docstrings for the encoding results API — the loaders above,
the types an artifact is made of, and the write seams behind them.

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
