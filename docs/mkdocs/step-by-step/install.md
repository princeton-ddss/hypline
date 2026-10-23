# Installation guide

Hypline supports Python 3.11 and later; testing covers 3.11 through 3.13. We recommend
installing it in a dedicated environment.

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

## Where to go next

- **Want to try it now?** Follow the [Tutorial](tutorials/walkthrough.md) for a complete
  run on a downloadable example dataset.
- **New to hypline?** Start with [Hypline dataset layout](step-by-step/layout.md)
  to learn how a dataset is organized.
- **Want command details?** See the [Step-by-step guides](step-by-step/overview.md) for each
  command's arguments and options.
- **Want to understand the analysis better?** [How the encoding model works](FAQ/how-encoding-works.md)
  and [Feature families](FAQ/feature-families.md) cover what you can predict and how the analysis is set up.
