# Welcome to Hypline

Hypline is a Python package that provides a CLI tool for cleaning and analyzing data from hyperscanning studies involving dyadic conversations.

## Installation

Hypline can be installed using `pip` or other package managers such as [`uv`](https://docs.astral.sh/uv/) and [`poetry`](https://python-poetry.org/docs/).

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

## Quick Start

Once the package is installed, `hypline` command will be available, like so:

```bash
hypline --help
```

Running the above will display an overview of the tool, including supported subcommands.

For instance, `clean` is a subcommand for performing confound regression to clean BOLD outputs from [fMRIPrep](https://fmriprep.org/en/stable/index.html), and its details can be viewed by running:

```bash
hypline clean --help
```

## What Next

Please check out user [guides](guides/clean.md) for more detailed instructions and examples.
