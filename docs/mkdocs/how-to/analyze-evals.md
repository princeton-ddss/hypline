# Test the cross-brain effect from evals

You have run [`encoding analyze`](../reference/encoding.md) a few times and have a
handful of eval files. This guide turns them into an answer to the question
hypline is built for: does one partner's model, driven by the shared
conversation, predict the *other* partner's brain better than a mismatched model
would? That comparison is a within-subject fit, a cross-brain fit, and a
pseudo-dyad baseline, read side by side.

This is the applied companion to [Reading an encoding result](../concepts/reading-an-eval.md),
which explains what a single eval's axes mean. Here we compare several.

## Produce the evals to compare

This guide assumes each subject you score has a **folded** model trained on a
semantic feature, tagged `v1` — for the target and its partner, and for the
out-of-dyad subjects used as a baseline below:

```bash
hypline encoding train data/ \
  --tasks conv --features semantic --desc v1 \
  --fold-by run --n-folds loo
```

Two details carry into everything that follows. The band is named after the
feature ref, so training `--features semantic` gives a band called `semantic`; if
you train `--features semantic-gpt3`, select `band="semantic-gpt3"` below instead.
And the model must be folded, because the analyses use the default out-of-sample
scoring — an unfolded model has no held-out runs to score and raises.

Each condition is then one `analyze` run against the same target brain, varying
only the source and model:

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

The three land under `results/sub-031/` as `encodingEval-within`,
`encodingEval-crossself`, and `encodingEval-pseudodyad`. Which source/model
pairings mean what is laid out in
[Choosing source and model](../concepts/how-encoding-works.md#choosing-source-and-model).

## Reduce each eval to one number per voxel

Load each eval and collapse it to a comparable summary: pick the feature band and
the role, then average over folds. For a cross-brain question during the target's
own speech, that is the `semantic` band on the `prod` role:

```python
from hypline.encoding import load_eval

def prod_semantic(desc):
    ds = load_eval(f"data/results/sub-031/encodingEval-{desc}/sub-031_result-encodingEval_desc-{desc}.nc")
    return ds["corr"].sel(band="semantic", role="prod").mean("fold", skipna=True)

within = prod_semantic("within")
crossself = prod_semantic("crossself")
pseudo = prod_semantic("pseudodyad")
```

Each result is one score per voxel. `skipna=True` matters here: a fold where the
role had no rows scored `NaN`, and a plain mean would propagate it — see
[the `fold` axis](../concepts/reading-an-eval.md#fold-the-cross-validation-folds).

## Compare against the baseline

The pseudo-dyad eval is the reference for chance. The cross-brain effect is real
to the extent the self-driven cross-brain score clears it:

```python
# per-voxel margin of the cross-brain fit over the pseudo-dyad baseline
cross_margin = crossself - pseudo

# a whole-brain summary: fraction of scored voxels the cross-brain fit beats
# baseline on (drop voxels that are NaN in either eval first)
valid = cross_margin.notnull()
(cross_margin.where(valid) > 0).sum().item() / valid.sum().item()
```

A within-brain score above both is the sanity check that the feature encodes in
this brain at all; the cross-brain margin over pseudo-dyad is the effect of
interest.

!!! note "Compare one band, don't read the sum as model quality"

    These are himalaya *split scores*: they decompose the joint correlation, so
    summing the feature and confound bands does recover the whole model's score —
    see
    [What the scores are](../concepts/reading-an-eval.md#what-the-scores-are-and-are-not).
    But that joint number is not the comparison here. The cross-brain question is
    about the *semantic* band specifically, so compare that one band across
    conditions, as above, rather than folding in bands that are not what you are
    testing.

## Strengthen the baseline

One out-of-dyad subject is one draw. For a real null you want a *distribution*:
run the pseudo-dyad analysis against several out-of-dyad subjects and compare the
cross-brain score against that spread rather than a single value.

```python
import numpy as np

pseudo_ids = ["045", "047", "052", "058"]   # subjects outside 031's dyad
for sid in pseudo_ids:
    ...  # analyze with --model-sub <sid> --desc pseudo<sid>, then load and reduce

baseline = np.stack([prod_semantic(f"pseudo{sid}") for sid in pseudo_ids])
baseline_mean = baseline.mean(0)   # per-voxel chance level across the null set
```

The wider the null set, the more trustworthy the claim that a cross-brain score
sits above chance. Log how many out-of-dyad subjects you drew, since the strength
of the baseline is part of the result.
