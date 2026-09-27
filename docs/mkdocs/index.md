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


## Where to go next

- **Want to try it now?** Follow the [Tutorial](tutorials/walkthrough.md) for a complete
  run on a downloadable example dataset.
- **New to hypline?** Start with [Hypline dataset layout](step-by-step/layout.md)
  to learn how a dataset is organized.
- **Want command details?** See the [Step-by-step guides](step-by-step/transcribe.md) for each
  command's arguments and options.
- **Want to understand the analysis better?** [How the encoding model works](FAQ/how-encoding-works.md)
  and [Feature families](FAQ/feature-families.md) cover what you can predict and how the analysis is set up.
