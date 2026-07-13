# Glossary

Hypline gives a handful of ordinary words a specific meaning. This page defines
them in one place; the rest of the docs link here rather than re-explaining. Terms
are grouped by where you meet them.

## Dataset structure

**Dataset root**
: The single directory every command takes as its argument. Hypline finds all
  inputs and writes all outputs by following a fixed layout beneath it, so you
  never pass file paths. See [The hypline dataset layout](layout.md).

**Dyad**
: The pair of subjects who held one conversation while both were scanned. Hypline
  is a hyperscanning pipeline, so the conversation belongs to the dyad, not to
  either partner.

**Sub-keyed / dyad-keyed**
: Which identity a file leads with. A **sub-keyed** file is derived from one brain
  (raw BOLD, denoised BOLD, a fitted model); a **dyad-keyed** file is derived from
  the shared conversation (audio, transcripts, features, confounds). The two are
  bridged through `participants.tsv`. See
  [Subject vs. dyad](layout.md#subject-vs-dyad).

**Segment**
: A named time window within a run — a trial, block, or condition — declared in
  the run's `events.tsv`. Segments are what let you generate per-trial features
  and filter down to specific conditions. See [Segments and metadata](segments.md).

**`desc` variant**
: A named alternative derivation of the same source, tagged with `--desc` and kept
  in its own subdirectory so variants sit side by side. Two phonemic-confound
  flavors (`phonemic-onset`, `phonemic-rate`) are `desc` variants of one kind.

## Features

**Feature**
: A stimulus-derived predictor for the encoding model — a per-word or per-TR
  vector describing the speech a participant heard. The four families are
  **phonemic**, **semantic**, **spectral**, and **syntactic**; see
  [Feature families](feature-families.md).

**Confound**
: A stimulus-derived nuisance regressor (speech onset, speech rate) partialled out
  *inside* the encoding fit, so a feature cannot claim signal a confound explains.
  Distinct from denoising nuisance, which is removed from the BOLD beforehand.

**Downsample**
: Binning a per-word feature onto the BOLD TR grid, since features are timed to
  words but BOLD is sampled per TR. Controlled by `--downsample` (`mean` or `sum`).

## The encoding model

**Encoding model**
: A model that predicts a voxel's BOLD signal from a weighted sum of speech
  features. Hypline fits one per subject with [`encoding train`](../reference/encoding.md).

**Band**
: One part of a banded-ridge model with its own regularization strength. Each
  feature is a band, all confounds share one band, and a reserved task band
  absorbs the production-versus-comprehension signal offset. Also the axis an eval
  scores on. See [How the encoding model works](how-encoding-works.md#bands-one-ridge-strength-per-feature).

**Delays**
: The set of time lags (in TRs) at which each feature enters the model, since a
  voxel's response to a word is spread over the following seconds. Set with
  `--delays`.

**Split**
: Fitting separate weights for when the subject is producing speech and when they
  are comprehending, by duplicating each feature into a speaking and a listening
  copy within its band. On by default; `--no-split` fits one shared set.

## Analysis

**Source / model / target**
: The three independent subject roles in [`encoding analyze`](../reference/encoding.md).
  **Source** drives the prediction's inputs (whose speech), **model** supplies the
  weights (whose trained model), **target** is the brain being predicted (whose
  BOLD). See [Choosing source and model](how-encoding-works.md#choosing-source-and-model).

**Within-brain analysis**
: Scoring a subject's own model, driven by their own speech, against their own
  brain (`source: self, model: self`). The baseline case.

**Cross-brain analysis**
: Predicting one partner's brain using the shared conversation and the other
  partner's model — the analysis hypline is built for. It comes in a
  partner-driven form (`source: partner, model: partner`) and a self-driven form
  (`source: self, model: partner`).

**Pseudo-dyad**
: A baseline where a subject's own speech and brain are scored with a model
  trained on someone they never conversed with. Because source and target are
  still the same subject, it runs without warning; the mismatch is in the model.

**Scramble control**
: A control where source and target belong to *different* dyads, so one
  conversation's speech is paired against another's brain, aligned only by
  matching run rather than a shared conversation. Mechanically valid but not a
  fit; `analyze` warns when it detects this.

**Role (`prod` / `comp` / `both`)**
: The turn subset a score covers, from the target's own turns: `prod` (target
  speaking), `comp` (target listening), `both` (either). An eval reports every
  band's score for each role. See [Reading an encoding result](reading-an-eval.md#role-which-turns-were-scored).

**Eval**
: The output of `analyze` — a netCDF file of per-voxel scores indexed by fold,
  band, role, and voxel, loaded back with `load_eval`. See
  [Reading an encoding result](reading-an-eval.md).

**Split score**
: The value stored in an eval: one band's own share of the joint model's
  correlation, from himalaya. A decomposition, so the bands sum to the joint
  score — but a single band's share is not itself a Pearson correlation and can
  be negative.
