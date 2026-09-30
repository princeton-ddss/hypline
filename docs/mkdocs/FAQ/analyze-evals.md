# Compare cross-brain fits with a baseline

You have run [`encoding analyze`](../step-by-step/encoding.md) a few times 
and have a handful of eval files. does a model trained on one partner's brain predict 
the other partner's brain better than a model from an unrelated subject? 

We compare a within-subject fit, a cross-brain fit, and a 
pseudo-dyad baseline side by side.

This is the applied companion to [How to work with encoding results](../step-by-step/encoding-results.md),
which explains how to load a single eval, interpret its axes, and select the scores you need. Here we compare across several evals.

## Produce the evals to compare

This guide assumes that the target, their partner, and each out-of-dyad subject 
used below have a **folded** model trained with the semantic feature and tagged v1:

```bash
hypline encoding train data/ \
  --data-filters task-conv \
  --features semantic \
  --desc v1 \
  --fold-by run
```

Two details carry into everything that follows. 

- The band is named after the feature reference. Training with --features semantic produces a band called semantic. If you instead train with --features semantic-gpt3, select band="semantic-gpt3" below.
- The model must be folded because the following analyses use out-of-sample scoring by default. An unfolded model has no held-out runs to score and raises an error.

Run each comparison against the same target brain, changing only the source and model:

```bash
# within-brain: the subject's own model and speech
hypline encoding analyze data/ \
  --target-sub 031 --source-sub self --model-sub self \
  --model-desc v1 --desc within

# cross-brain, self-driven: own speech, partner's model
hypline encoding analyze data/ \
  --target-sub 031 --source-sub self --model-sub partner \
  --model-desc v1 --desc crossself

# pseudo-dyad: own speech and brain, an out-of-dyad model (named by ID)
hypline encoding analyze data/ \
  --target-sub 031 --source-sub self --model-sub 045 \
  --model-desc v1 --desc pseudodyad
```

The resulting directories appear under `results/sub-031/`:
```bash
encodingEval-within
encodingEval-crossself
encodingEval-pseudodyad
```

See [Choosing source and model](how-encoding-works.md#choosing-source-and-model) for the meaning of other source/model combinations.

## Reduce each eval to one score per voxel

Load each eval, select the same feature band and role, and average across folds. 
Because the source is the target subject's own speech here, select the prod role: 

```python
from hypline.encoding import load_eval

def prod_semantic(desc):
    ds = load_eval(f"data/results/sub-031/encodingEval-{desc}/sub-031_result-encodingEval_desc-{desc}.nc")
    return ds["corr"].sel(band="semantic", role="prod").mean("fold", skipna=True)

within = prod_semantic("within")
crossself = prod_semantic("crossself")
pseudo = prod_semantic("pseudodyad")
```

Each result now contains one semantic-band score per voxel.
`skipna=True` matters because a fold is scored as NaN when the selected role has no rows 
in that fold. A regular mean would propagate that missing value. See
[the `fold` axis](../step-by-step/encoding-results.md#fold-the-cross-validation-folds) for details.

## Compare the cross-brain and pseudo-dyad fits

The pseudo-dyad eval provides a mismatched-model baseline. Compare 
the self-driven cross-brain score against it:

```python
# per-voxel margin of the cross-brain fit over the pseudo-dyad baseline
cross_margin = crossself - pseudo

# whole-brain summary: fraction of scored voxels where the cross-brain fit
# exceeds the pseudo-dyad baseline
valid = cross_margin.notnull()
(cross_margin.where(valid) > 0).sum().item() / valid.sum().item()
```

A positive value means that the matched partner's model predicts the target brain 
better than this out-of-dyad model at that voxel.  The within-subject score is not 
part of this subtraction. It provides a reference for how well the semantic feature 
predicts the target brain when the model is trained on that same subject.

!!! note "Compare the feature band you are testing"

    These are himalaya *split scores*. Split scores describe each band's contribution 
    to the joint correlation, and summing across bands recovers the whole model's score. See —
    [What the scores are](../step-by-step/encoding-results.md#what-the-scores-are-and-are-not).
    Here, the question concerns the `semantic` band specifically. Compare that band across conditions 
    rather than adding confound or other feature bands that are not part of the hypothesis.

!!! important "This is a descriptive comparison"
    A positive margin—or a large fraction of voxels with positive margins—is not by itself a statistical 
    test. Voxels within a brain are not independent observations, so they
    should not be treated as independent samples.
    For population-level inference, repeat the comparison across target subjects or dyads and use a 
    group-level test or permutation procedure that preserves the dyadic structure of the data.
    

## Use multiple pseudo-dyads

One out-of-dyad subject gives only one mismatched comparison. For a more 
informative baseline, run the pseudo-dyad analysis against multiple out-of-dyad 
subjects and compare the cross-brain score against the resulting *distribution*.

```python
import numpy as np

pseudo_ids = ["045", "047", "052", "058"]   # subjects outside 031's dyad
for sid in pseudo_ids:
    ...  # analyze with --model-sub <sid> --desc pseudo<sid>, then load and reduce

baseline = np.stack([prod_semantic(f"pseudo{sid}") for sid in pseudo_ids])
baseline_mean = baseline.mean(0)   # per-voxel chance level across the null set
```

Use as many eligible out-of-dyad models as practical, apply the same inclusion criteria to all of them, 
and record how the comparison set was constructed. 
