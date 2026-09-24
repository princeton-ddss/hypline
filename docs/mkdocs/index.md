# Hypline

Hypline is a command-line toolbox for running encoding analyses on fMRI
hyperscanning data collected during dyadic conversation. It is designed for
researchers who want to relate features of an ongoing conversation—such as its
phonemic, semantic, syntactic, or acoustic content—to the BOLD responses of the
people producing and comprehending that speech.

An encoding model learns a mapping from features of a stimulus or behavior
to measured brain activity and tests how well those features predict responses
in held-out fMRI data. Encoding models are widely used in naturalistic fMRI to
study what information is represented across the brain during continuous
experiences such as listening to speech, watching movies, or engaging in
conversation. In hyperscanning studies, the same framework can be used to
examine the neural representations associated with speech production and
comprehension while two people interact in real time. To learn more about how 
encoding model works, see [How the encoding model works](FAQ/how-encoding-works.md).

Hypline supports the main steps needed to run this analysis on dyadic
conversation data: transcribing recorded speech, generating stimulus features
and their confounds, denoising [fMRIPrep](https://fmriprep.org/en/stable/index.html) 
BOLD data, and fitting and evaluating encoding models. Its commands are modular: 
each performs one step and can be run independently, while all commands operate 
within the same [BIDS](https://bids.neuroimaging.io/)-style dataset.

Hypline implements the encoding-model approach of Zada et al. (2026),[^zada]
which combined fMRI hyperscanning with language-model features to study the shared
neural systems for speech production and comprehension during real-time dyadic
conversation.

[^zada]: Zada, Z., Nastase, S. A., Speer, S., Mwilambwe-Tshilobo, L., Tsoi, L.,
    Burns, S. M., Falk, E., Hasson, U., & Tamir, D. I. (2026). Linguistic
    coupling between neural systems for speech production and comprehension
    during real-time dyadic conversations. *Neuron*, *114*, 1–14.
    [https://doi.org/10.1016/j.neuron.2025.11.004](https://doi.org/10.1016/j.neuron.2025.11.004)

## Installation

Hypline currently supports Python 3.11. We recommend installing it in a
dedicated Python 3.11 environment.

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
    which must be installed separately. Other commands do not need it.

For example: 

=== "macOS"

    ```bash
    brew install ffmpeg
    ```

=== "Ubuntu / Debian" 

    ```bash 
    sudo apt update
    sudo apt install ffmpeg
    ```
    
=== "Conda" 

    ```bash 
    conda install -c conda-forge ffmpeg
    ```

After installation, confirm that FFmpeg is available: 
    ```bash
    ffmpeg -version
    ```

Only `hypline transcribe` requires FFmpeg; other Hypline commands do not.

## The commands

Hypline's commands form a modular pipeline. Each command reads from the same
dataset root and writes its outputs back into that dataset.

The workflow has three parts: (1) the **stimulus branch** prepares features 
describing the conversation; (2) the **fMRIPrep branch** prepares denoised BOLD responses; 
and (3) the **encoding branch** combines these inputs to fit and evaluate encoding models.

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

The stimulus and fMRIPrep branches can be run independently. Stimulus commands
construct the predictors used by the encoding model, while `denoise` prepares
the BOLD responses that serve as its targets. These inputs come together only
when you run the [`encoding`](how-to/encoding.md) commands.

!!! tip "Features and their confounds in one step"

    `featuregen phonemic` generates the corresponding phonemic confounds by
    default, so you usually do not need to call `confoundgen phonemic` separately. 
    See the [featuregen guide](how-to/featuregen.md).

You do not need to run the entire pipeline. Each command can be used
independently as long as its required inputs already exist. For example, you
can run `transcribe` only to generate transcripts, or `denoise` only to clean
fMRIPrep BOLD data.

## Run the pipeline

Once your files follow the 
[hypline dataset layout](data-prep/layout.md), each command takes the dataset root
as its main input and automatically finds the files it needs.

For example, a basic analysis using phonemic features looks like this:

```bash
# stimulus branch: audio → transcripts → features 
hypline transcribe data/ --audio-ext .wav
hypline featuregen phonemic data/

# fMRIPrep branch: preprocessed BOLD → denoised BOLD
hypline denoise data/ \
  --columns trans_x,trans_y,trans_z,rot_x,rot_y,rot_z,cosine

# encoding branch: fit the encoding model
hypline encoding train data/ \
  --data-filters task-conv \
  --features phonemic \
  --desc v1 \
  --fold-by none
```

After these steps, `data/` contains the phonemic predictors, denoised BOLD responses
and fitted encoding models under `results/`. To evaluate fitted models,
use `hypline encoding analyze`. You can then load the resulting evaluation outputs
in Python with [`load_eval` / `load_artifact`](how-to/encoding-results.md).

You can also run any step on its own. Hypline skips outputs that already exist;
use `--force` when you want to regenerate them.

!!! tip "When a command produces no output"

    `No dyads found` for stimulus commands or `No subjects found` for 
    `denoise` and `encoding` usually means that the command could not
    find matching inputs, or that `--dyad-ids` / `--sub-ids` excluded
    all available data. Check the dataset layout and the IDs you supplied. (If
    `--data-filters` matches no data, see [Filter to specific runs or
    conditions](FAQ/filter.md#when-a-filter-matches-nothing).)

## Where to go next

- **Want to try it now?** Follow the [Tutorial](tutorials/walkthrough.md) for a complete
  run on a downloadable example dataset.
- **New to hypline?** Start with [The hypline dataset layout](data-prep/layout.md)
  to learn how a dataset is organized.
- **Want command details?** See the [How-to guides](how-to/transcribe.md) for each
  command's arguments and options.
- **Want to understand the analysis better?** [How the encoding model works](FAQ/how-encoding-works.md)
  and [Feature families](FAQ/feature-families.md) cover what you can predict and how the analysis is set up.
