# How the encoding model works

The [`encoding`](../reference/encoding.md) commands are the point of the whole
pipeline: everything before them prepares the two sides of one fit. This page
explains what that fit *is* — the model `train` builds and `analyze` scores — and
then the choice that gives an analysis its meaning: whose speech, whose model,
and whose brain you line up.

You do not need this page to run the commands; the [reference](../reference/encoding.md)
covers every option. Read it to understand what the numbers mean.

## What an encoding model predicts

An encoding model predicts the brain from the stimulus. For each voxel, it learns
a weighted sum of the speech features a participant heard that best reproduces
that voxel's BOLD signal over time. Features are the predictors (**X**), the
denoised BOLD is the target (**Y**), and the fit finds the weights. A model that
predicts held-out BOLD well has captured something real about how that brain
responds to speech.

Two facts about the BOLD signal shape the model:

- **The response is delayed and smeared.** A voxel's answer to a word arrives
  over the next several seconds, not at the instant the word is heard. So each
  feature enters the model at a range of **delays** (`--delays`, in TRs by
  default `0,1,2,3,4,5`), letting the fit weight the word's echo across the
  seconds that follow.
- **The fit must not overfit.** A whole-brain model has far more weights than
  timepoints, so it is a **ridge** regression: a penalty (`--alphas`) shrinks the
  weights, and the strength is chosen by cross-validation.

## Bands: one ridge strength per feature

The model is a **banded ridge**. Each feature you pass to `--features` becomes
its own **band** with its own penalty strength, so a slow-varying feature and a
sharp one are each regularized on their own terms rather than forced to share.
This is also why `analyze` can report a separate score per band: each feature's
contribution is fit, and measured, in its own right.

Two bands are special:

- **The task band** is always present. It carries two intercept-like signals that
  mark when the participant is producing speech and when they are comprehending,
  and it absorbs the average difference in BOLD between those two states so the
  feature bands do not have to. It is the one band left unstandardized, since
  standardizing would destroy the flat-zero pattern that lets it do this.
- **The confound band** appears when you pass `--confounds`. Every confound
  shares this single band, and it is fit *alongside* the features so a feature
  band cannot take credit for signal a confound already explains.

!!! note "Confounds are not the denoising step"

    The confound band is stimulus-derived nuisance — speech onset and rate, say —
    partialled out *inside* the encoding fit. That is separate from
    [`denoise`](../reference/denoise.md), which cleans run-level nuisance (motion,
    drift) out of the BOLD before encoding ever sees it. The same stimulus-derived
    signal can be a feature band in one fit and a confound band in another; the
    role is a choice you make per fit, not a fixed property.

## The production/comprehension split

A conversation has the participant speaking on some turns and listening on
others, and one word can drive a different brain response in the two states. By
default `train` **splits** every feature so the model can learn separate weights
for production and for comprehension. The split doubles the width of each band —
a speaking copy and a listening copy of every feature — but does not add bands.
Pass `--no-split` to fit one shared set of weights instead.

This is worth keeping distinct from the roles you *score* on. The split is about
how the model is *fit*; the [`prod`/`comp`/`both` roles](reading-an-eval.md#role-which-turns-were-scored)
in an eval are about which rows are *scored* afterward. Both come from the same
speaking turns, but they are separate mechanisms.

## Choosing source and model

`train` fits one model per subject. `analyze` then scores a model's predictions
against a subject's real brain, and it takes **three subjects that need not be
the same person**:

- **target** (`--target-sub`) — whose real BOLD the scores are measured against,
  and whose speaking turns define the `prod`/`comp` roles.
- **model** (`--model-sub`) — whose trained weights make the prediction.
- **source** (`--source-sub`) — whose speech builds the prediction's inputs.

The `self` and `partner` shortcuts resolve against the target through
`participants.tsv`, so `source: self` means the source *is* the target subject,
and `partner` is their dyad partner. Holding the target fixed, the pairing of
source and model is what turns the same machinery into different analyses:

| source | model | What it is |
| ------ | ----- | ---------- |
| self | self | **Within-brain.** A subject's own speech and own model predict their own brain — the baseline "does this feature encode in this brain at all" fit. |
| partner | partner | **Cross-brain, partner-driven.** The partner's speech and the partner's model predict the subject's brain. This is the analysis in Zada et al. (2026). |
| self | partner | **Cross-brain, self-driven.** The subject's *own* speech, but the *partner's* model, predicts the subject's brain — holding the stimulus fixed and testing only whether the partner's learned mapping transfers. `--source-sub` defaults to `self` for this reason. |
| self | out-of-dyad | **Pseudo-dyad.** The subject's own speech and brain, scored with a model trained on someone they never spoke with — a baseline for what a mismatched model scores by chance, to compare the real effects against. |

The pseudo-dyad case names an out-of-dyad subject by ID, since `self`/`partner`
only resolve within the target's own dyad:

```bash
hypline encoding analyze data/ \
  --target-sub 031 \
  --source-sub self \
  --model-sub 045 \
  --model-desc v1 \
  --desc pseudodyad
```

All four of these keep the source in the target's own dyad, so they run without
warning. A fifth arrangement does warn: pointing source and target at
*different* dyads pairs one conversation's speech against another's brain, aligned
only by matching run — not a shared conversation. That is a scramble control,
mechanically valid but not a fit, and `analyze` warns to flag it.

!!! note "These readings are one team's, not the tool's"

    The interpretations above are how this project's researchers think about each
    pairing, offered as a starting point. The tool allows other mechanically valid
    combinations, and not every one is scientifically meaningful. Which pairing to
    run, and how to read its scores, is your call.

## Where to go next

- **Read the scores an analysis produces** —
  [Reading an encoding result](reading-an-eval.md).
- **Every option in full** — the [`encoding` reference](../reference/encoding.md).
- **See it run once** — the [tutorial](../tutorials/walkthrough.md) fits a model
  and scores it within and across brains.
