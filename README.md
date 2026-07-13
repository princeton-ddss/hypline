# Hypline

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/princeton-ddss/hypline/actions/workflows/ci.yml/badge.svg)](https://github.com/princeton-ddss/hypline/actions/workflows/ci.yml)
[![CD](https://github.com/princeton-ddss/hypline/actions/workflows/cd.yml/badge.svg)](https://github.com/princeton-ddss/hypline/actions/workflows/cd.yml)

Hypline is a command-line toolbox for cleaning and analyzing data from
hyperscanning studies involving dyadic conversations. Its commands are modular:
each does one job — transcribe audio, generate features, denoise
[fMRIPrep](https://fmriprep.org/en/stable/index.html) BOLD, fit an **encoding
model** — and runs on its own, all inside one
[BIDS](https://bids.neuroimaging.io/)-style dataset.

An encoding model predicts the brain's BOLD response from features of the speech
a participant heard. Hypline prepares both sides of that fit — the stimulus
features and the denoised BOLD — then fits and scores the model with `encoding`.
You bring two inputs: the **stimulus audio** and the **fMRIPrep-preprocessed
BOLD**. Hypline starts from fMRIPrep's outputs; it does not preprocess the BOLD
itself.

Hypline implements the encoding-model approach of Zada et al. (2026), *Neuron*
([10.1016/j.neuron.2025.11.004](https://doi.org/10.1016/j.neuron.2025.11.004)),
which used fMRI hyperscanning and language-model features to study the shared
neural systems for speech production and comprehension in real-time dyadic
conversations.

## Installation

```bash
pip install hypline
```

Also installable with [`uv`](https://docs.astral.sh/uv/) (`uv add hypline`) or
[`poetry`](https://python-poetry.org/) (`poetry add hypline`). This installs the
`hypline` command:

```bash
hypline --help
```

> [!NOTE]
> `hypline transcribe` decodes audio through [FFmpeg](https://ffmpeg.org/),
> which must be installed separately and on your `PATH`. Other commands do not
> need it.

## The pipeline

Hypline's commands compose into a pipeline. Every command reads from a shared
dataset root and writes back into the same tree. Most fall into two independent
branches — a **stimulus branch** and an **fMRIPrep branch** — that prepare the two
sides the **encoding branch** then joins:

| Command                | Branch   | Reads                                  | Writes                          |
| ---------------------- | -------- | -------------------------------------- | ------------------------------- |
| `transcribe`           | stimulus | stimulus audio                         | word-level transcripts          |
| `featuregen phonemic`  | stimulus | transcripts                            | phonemic features (+ confounds) |
| `featuregen semantic`  | stimulus | transcripts                            | semantic features (+ confounds) |
| `featuregen spectral`  | stimulus | stimulus audio                         | spectral features (TR-aligned)  |
| `featuregen syntactic` | stimulus | transcripts                            | syntactic features              |
| `confoundgen phonemic` | stimulus | phonemic features                      | `conf-phonemic` confounds       |
| `confoundgen semantic` | stimulus | semantic features                      | `conf-semantic` confounds       |
| `denoise`              | fMRIPrep | preprocessed BOLD, fMRIPrep confounds  | denoised BOLD (`desc-denoised`) |
| `encoding train`       | encoding | features, confounds, denoised BOLD     | fitted models (`results/`)      |
| `encoding analyze`     | encoding | fitted models, features, denoised BOLD | eval correlations (`results/`)  |

`featuregen phonemic` also generates the matching phonemic confounds by default,
so you rarely call `confoundgen phonemic` directly. And you needn't run every
step: any command works on its own once its inputs exist — run `transcribe` alone
for transcripts, or `denoise` alone to clean fMRIPrep BOLD.

## Quick start

Once your files sit where hypline expects (see
[the dataset layout](https://princeton-ddss.github.io/hypline/latest/concepts/layout/)),
you only ever point a command at the dataset root — it finds its own inputs from
there, so you never pass file paths. End to end, the whole pipeline is four
commands:

```bash
# stimulus branch: audio → transcripts → features (+ phonemic confounds, auto)
hypline transcribe data/ --audio-ext .wav
hypline featuregen phonemic data/

# fMRIPrep branch: clean the BOLD with a motion + drift model, read straight
# from fMRIPrep's confounds table
hypline denoise data/ \
  --columns trans_x,trans_y,trans_z,rot_x,rot_y,rot_z,cosine

# encoding branch: fit the model that maps features onto the denoised BOLD
hypline encoding train data/ \
  --tasks conv \
  --features phonemic \
  --desc v1 \
  --fold-by none
```

After this, `data/` holds phonemic features plus `desc-denoised` BOLD — the two
sides the encoding model needs — and a fitted model under `results/`.

## Documentation

Full guides and per-command reference live at the project
[documentation](https://princeton-ddss.github.io/hypline/latest/). New to
hypline? Walk through
[a full run on the example dataset](https://princeton-ddss.github.io/hypline/latest/tutorials/walkthrough/),
or read
[The hypline dataset layout](https://princeton-ddss.github.io/hypline/latest/concepts/layout/) —
every command depends on it. To load a fitted model or eval back into Python, see
the [encoding results API](https://princeton-ddss.github.io/hypline/latest/reference/encoding-results/).

## License

Released under the [MIT License](LICENSE).
