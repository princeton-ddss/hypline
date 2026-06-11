# Hypline

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/princeton-ddss/hypline/actions/workflows/ci.yml/badge.svg)](https://github.com/princeton-ddss/hypline/actions/workflows/ci.yml)
[![CD](https://github.com/princeton-ddss/hypline/actions/workflows/cd.yml/badge.svg)](https://github.com/princeton-ddss/hypline/actions/workflows/cd.yml)

Hypline is a command-line toolbox for cleaning and analyzing data from
hyperscanning studies involving dyadic conversations. It turns stimulus audio
and [fMRIPrep](https://fmriprep.org/en/stable/index.html) outputs into the
inputs an **encoding model** needs — transcripts, features, confounds, and
denoised BOLD — all organized in one [BIDS](https://bids.neuroimaging.io/)-style
dataset.

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

Hypline commands form a pipeline. Each reads from a shared dataset root and
writes its outputs back into the same tree, along two independent branches — a
**stimulus branch** and an **fMRIPrep branch** — that prepare the two sides an
encoding model later joins:

| Command                | Branch   | Reads                                 | Writes                          |
| ---------------------- | -------- | ------------------------------------- | ------------------------------- |
| `transcribe`           | stimulus | stimulus audio                        | word-level transcripts          |
| `featuregen phonemic`  | stimulus | transcripts                           | phonemic features (+ confounds) |
| `confoundgen phonemic` | stimulus | phonemic features                     | `conf-phonemic` confounds       |
| `denoise`              | fMRIPrep | preprocessed BOLD, fMRIPrep confounds | denoised BOLD (`desc-denoised`) |

`featuregen phonemic` also generates the matching phonemic confounds by default,
so you rarely call `confoundgen phonemic` directly. You do not have to run every
step — each works on its own as long as its inputs exist.

## Quick start

Once your files sit where hypline expects (see
[the dataset layout](https://princeton-ddss.github.io/hypline/latest/concepts/layout/)),
every command takes the dataset root and discovers its inputs from there — you
never pass file paths. End to end, the whole pipeline is three commands:

```bash
# stimulus branch: audio → transcripts → features (+ phonemic confounds, auto)
hypline transcribe data/ --audio-ext .wav
hypline featuregen phonemic data/

# fMRIPrep branch: clean the BOLD with a motion + drift model, read straight
# from fMRIPrep's confounds table
hypline denoise data/ \
  --columns trans_x,trans_y,trans_z,rot_x,rot_y,rot_z,cosine
```

After this, `data/` holds phonemic features plus `desc-denoised` BOLD — the two
sides an encoding model needs.

## Documentation

Full guides and per-command reference live at the project
[documentation](https://princeton-ddss.github.io/hypline/latest/). New to
hypline? Walk through
[a full run on the example dataset](https://princeton-ddss.github.io/hypline/latest/tutorials/walkthrough/),
or read
[The hypline dataset layout](https://princeton-ddss.github.io/hypline/latest/concepts/layout/) —
every command depends on it.

## License

Released under the [MIT License](LICENSE).
