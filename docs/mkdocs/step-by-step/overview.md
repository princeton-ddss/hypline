# Overview

## The pipeline

Hypline's commands form a modular pipeline. Once your files follow the 
[hypline dataset layout](layout.md), each command takes the dataset root
as its main input and automatically finds the files it needs.

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
when you run the [`encoding`](encoding.md) commands.

!!! tip "Features and their confounds in one step"

    `featuregen phonemic` generates the corresponding phonemic confounds by
    default, so you usually do not need to call `confoundgen phonemic` separately. 
    See the [featuregen guide](featuregen.md).

You do not need to run the entire pipeline. Each command can be used
independently as long as its required inputs already exist. For example, you
can run `transcribe` only to generate transcripts, or `denoise` only to clean
fMRIPrep BOLD data.

## Example command set

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
in Python with [`load_eval` / `load_artifact`](encoding-results.md).

You can also run any step on its own. Hypline skips outputs that already exist;
use `--force` when you want to regenerate them.

