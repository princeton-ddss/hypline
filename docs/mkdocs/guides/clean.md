# Cleaning BOLD Data

## Overview

Hypline supports confound regression to clean BOLD outputs from [fMRIPrep](https://fmriprep.org/en/stable/index.html).

Specifically, the `clean` subcommand supports this, and its details can be checked by running:

```bash
hypline clean --help
```

The `clean` subcommand takes in three required arguments:

| Argument        | Description                                                  |
| --------------- | ------------------------------------------------------------ |
| `fmriprep_dir`  | Directory containing fMRIPrep outputs                        |
| `config_file`   | Configuration file containing model specs, etc.              |
| `model_name`    | Confound regression model to run (defined in configuration)  |

and several options:

| Option                    | Description                             | Default                   |
| ------------------------- | --------------------------------------- | ------------------------- |
| `--output-dir`            | Directory to store cleaned BOLD data    | `<fmriprep_dir>_cleaned`  |
| `--custom-confounds-dir`  | Directory containing custom confounds   | None                      |
| `--subject-ids`           | Target subject IDs (comma-separated)    | All                       |
| `--session-name`          | Target session name                     | All                       |
| `--task-name`             | Target task name                        | All                       |
| `--data-space-name`       | Target BOLD data space                  | `MNI152NLin2009cAsym`     |
| `--n-processes`           | Number of processes to run in parallel  | `1`                       |

!!! info

    Hypline supports the following data spaces at the moment:

    - `fsaverage5`
    - `fsaverage6`
    - `MNI152NLin6Asym`
    - `MNI152NLin2009cAsym`

## Example

For illustration, consider we have the fMRIPrep outputs stored in the following location:

``` hl_lines="3-6"
data/
└── derivatives/
    └── fmriprep/
        ├── logs/
        ├── sub-003/
        └── ...
```

### Setting up configuration

We can start by defining the desired confound regression model(s) in the configuration file, like so:

```yaml title="config.yaml"
model_specs:

  default:
    confounds:
      - "trans_x"
      - "trans_y"
      - "trans_z"
      - "rot_x"
      - "rot_y"
      - "rot_z"
      - "cosine"
    aCompCor:
      - n_comps: 5
        mask: "CSF"
      - n_comps: 5
        mask: "WM"

  default_task:
    confounds:
      - "trans_x"
      - "trans_y"
      - "trans_z"
      - "rot_x"
      - "rot_y"
      - "rot_z"
      - "cosine"
    custom_confounds:
      - "trial_boxcar"
      - "prompt_boxcar"
      - "speech_boxcar"
      - "listen_boxcar"
      - "button_press"
      - "receive_press"
      - "screen_change"
    aCompCor:
      - n_comps: 5
        mask: "CSF"
      - n_comps: 5
        mask: "WM"
```

!!! info

    The configuration file above defines two models, named `default` and `default_task`, respectively.
    The `default` model is defined to use standard and CompCor confounds from fMRIPrep outputs (i.e.,
    `confounds` and `aCompCor` sections above, respectively). The `default_task` model is defined to
    use [custom confounds](#using-custom-confounds) (i.e., `custom_confounds` section above) in addition
    to confounds from fMRIPrep outputs.

!!! note

    Configuration should be provided as a valid YAML file. Check this
    [tutorial](https://www.datacamp.com/blog/what-is-yaml) to learn the basics of YAML.

We can store the configuration file anywhere we want. Let's say we stored it as follows:

``` hl_lines="3-4"
data/
└── derivatives/
    ├── hypline/
    │   └── config.yaml
    └── fmriprep/
        ├── logs/
        ├── sub-003/
        └── ...
```

### Using custom confounds

If the model involves custom confounds, we should provide the data as subject-level TSV files that each contain a given subject’s confound values per TR. Hence, each TSV file will be a tabular data of shape $n \times m$, where $n$ equals the number of total TRs in a run and $m$ equals the number of custom confounds. With the current directory structure, we may prepare custom confounds as follows:

``` hl_lines="5-9"
data/
└── derivatives/
    ├── hypline/
    │   ├── config.yaml
    │   └── custom_confounds/
    │       ├── sub-003/ses-1/func/
    │       │   ├── sub-003_ses-1_task-Conv_run-1_desc-customConfounds_timeseries.tsv
    │       │   └── ...
    │       └── ...
    │
    └── fmriprep/
        ├── logs/
        ├── sub-003/ses-1/func/
        │   ├── sub-003_ses-1_task-Conv_run-1_desc-confounds_timeseries.tsv
        │   ├── sub-003_ses-1_task-Conv_run-1_space-fsaverage6_hemi-L_bold.func.gii
        │   ├── sub-003_ses-1_task-Conv_run-1_space-fsaverage6_hemi-R_bold.func.gii
        │   └── ...
        └── ...
```

!!! note

    The custom confound TSV file should have the same name as its corresponding standard confound TSV file, except for the description (i.e., `desc` entity).

### Running the command

With this setup, we can run the following to clean surface-level BOLD data for all subjects:

```bash
hypline  clean  \
  data/derivatives/fmriprep/  \
  data/derivatives/hypline/config.yml  \
  default_task  \
  --custom-confounds-dir  data/derivatives/hypline/custom_confounds/  \
  --data-space  fsaverage6  \
  --n-processes  8
```

Note that we are here leveraging multiprocessing (i.e., 8 parallel processes) to speed up the cleaning.

### Outputs

The cleaned data will then be stored in the designated location[^1] as follows:

``` hl_lines="20-28"
data/
└── derivatives/
    ├── hypline/
    │   ├── config.yaml
    │   └── custom_confounds/ses-1/func/
    │       ├── sub-003/
    │       │   ├── sub-003_ses-1_task-Conv_run-1_desc-customConfounds_timeseries.tsv
    │       │   └── ...
    │       └── ...
    │
    ├── fmriprep/
    │   ├── logs/
    │   ├── sub-003/ses-1/func/
    │   │   ├── sub-003_ses-1_task-Conv_run-1_desc-confounds_timeseries.tsv
    │   │   ├── sub-003_ses-1_task-Conv_run-1_space-fsaverage6_hemi-L_bold.func.gii
    │   │   ├── sub-003_ses-1_task-Conv_run-1_space-fsaverage6_hemi-R_bold.func.gii
    │   │   └── ...
    │   └── ...
    │
    └── fmriprep_cleaned/
        ├── logs/
        │   ├── sub-003.log
        │   └── ...
        ├── sub-003/ses-1/func/
        │   ├── sub-003_ses-1_task-Conv_run-1_space-fsaverage6_hemi-L_desc-clean_bold.func.gii
        │   ├── sub-003_ses-1_task-Conv_run-1_space-fsaverage6_hemi-R_desc-clean_bold.func.gii
        │   └── ...
        └── ...
```

As shown above, the cleaned data files will be of the same format as their corresponding originals. In fact, their data dimension and other metadata will be the same as the originals, with the only difference being values in the data.

Note that log files are automatically generated to record the details of the cleaning process, which will be useful for reproducibility.

[^1]: In this case, the default value (`<fmriprep_dir>_cleaned`) is used because the given command does not explicitly specify the output directory (`--output-dir`).
