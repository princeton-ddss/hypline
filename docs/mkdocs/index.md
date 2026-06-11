# Hypline

Hypline is a command-line toolbox for cleaning and analyzing data from
hyperscanning studies involving dyadic conversations. It turns stimulus audio
and [fMRIPrep](https://fmriprep.org/en/stable/index.html) outputs into the
inputs an **encoding model** needs — transcripts, features, confounds, and
cleaned BOLD — all organized in one [BIDS](https://bids.neuroimaging.io/)-style
dataset. An encoding model predicts the brain's BOLD response from features of
the speech a participant heard; hypline prepares both sides of that fit, the
stimulus features (the predictors) and the denoised BOLD (the target).

Hypline implements the encoding-model approach of Zada et al. (2026),[^zada]
which used fMRI hyperscanning and language-model features to study the shared
neural systems for speech production and comprehension in real-time dyadic
conversations.

[^zada]: Zada, Z., Nastase, S. A., Speer, S., Mwilambwe-Tshilobo, L., Tsoi, L.,
    Burns, S. M., Falk, E., Hasson, U., & Tamir, D. I. (2026). Linguistic
    coupling between neural systems for speech production and comprehension
    during real-time dyadic conversations. *Neuron*, *114*, 1–14.
    [https://doi.org/10.1016/j.neuron.2025.11.004](https://doi.org/10.1016/j.neuron.2025.11.004)

## Installation

=== "pip"

    ```bash
    pip install hypline
    ```

=== "uv"

    ```bash
    uv add hypline
    ```

=== "poetry"

    ```bash
    poetry add hypline
    ```

This installs the `hypline` command. Confirm it works:

```bash
hypline --help
```

!!! note "FFmpeg required for transcription"

    `hypline transcribe` decodes audio through [FFmpeg](https://ffmpeg.org/),
    which must be installed separately and available on your `PATH`. Other
    commands do not need it.

## The pipeline

Hypline commands form a pipeline. Each one reads from a shared dataset root and
writes its outputs back into the same tree. The steps fall into two independent
branches — a **stimulus branch** and an **fMRIPrep branch** — that prepare the
two sides an encoding model later joins:

| Command                | Branch   | Reads                                  | Writes                          |
| ---------------------- | -------- | -------------------------------------- | ------------------------------- |
| `transcribe`           | stimulus | stimulus audio                         | word-level transcripts          |
| `featuregen phonemic`  | stimulus | transcripts                            | phonemic features (+ confounds) |
| `confoundgen phonemic` | stimulus | phonemic features                      | `conf-phonemic` confounds       |
| `denoise`              | fMRIPrep | preprocessed BOLD, fMRIPrep confounds  | denoised BOLD (`desc-denoised`) |

The branches never meet inside hypline: stimulus commands build the encoding
model's predictors, while `denoise` cleans the BOLD target from fMRIPrep's own
confounds table (and any custom `nuisance/` regressors). The encoding model joins
them downstream.

!!! tip "Features and their confounds in one step"

    `featuregen phonemic` also generates the matching phonemic confounds by
    default, so you rarely call `confoundgen phonemic` directly. See the
    [featuregen reference](reference/featuregen.md).

You do not have to run every step. Each command works on its own as long as its
inputs exist — for example, you can run `denoise` on fMRIPrep outputs without
ever transcribing audio.

## Run the pipeline

Once your files sit where hypline expects (see
[the dataset layout](concepts/layout.md)), every command takes the dataset root
and discovers its inputs from there — you never pass file paths. End to end, the
whole pipeline is three commands. Each reads what the previous ones wrote, so
order matters only where one step's output is the next step's input:

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
sides an encoding model needs. You can also start with `transcribe` alone and
follow the table above step by step. Re-run any single step with `--force` to
overwrite its outputs; without it, hypline skips work it has already done.

!!! tip "If a command seems to do nothing"

    Two harmless cases look like failures. **`No dyads found`** (stimulus
    commands) or **`No subjects found`** (`denoise`) means the area that command
    reads is empty or your `--dyad-ids` / `--sub-ids` / `--data-filters` excluded
    everything — widen the filter or check the files are in place. A command that
    exits instantly with no log means its outputs already exist and were skipped;
    re-run with `--force` to regenerate. Per-command failure modes are listed
    under **Common errors** on each reference page.

## Where to go next

- **Want to try it now?** Follow the [Tutorial](tutorials/walkthrough.md) — a full
  pipeline run on a downloadable example dataset, one command at a time.
- **New to hypline?** Start with [The hypline dataset layout](concepts/layout.md)
  to learn how a dataset is organized — every command depends on it.
- **Want command details?** See the [Reference](reference/transcribe.md) for each
  command's arguments and options.
