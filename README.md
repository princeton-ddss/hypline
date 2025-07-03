# Hypline

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/princeton-ddss/hypline/actions/workflows/ci.yml/badge.svg)](https://github.com/princeton-ddss/hypline/actions/workflows/ci.yml)
[![CD](https://github.com/princeton-ddss/hypline/actions/workflows/cd.yml/badge.svg)](https://github.com/princeton-ddss/hypline/actions/workflows/cd.yml)

Hypline is a Python package that provides a CLI tool for cleaning and analyzing data from hyperscanning studies involving dyadic conversations.

## Installation

Hypline can be installed using `pip`:

```bash
pip install hypline
```

It can also be installed using other package managers such as [`uv`](https://docs.astral.sh/uv/) and [`poetry`](https://python-poetry.org/docs/).

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

If you want to learn more about Hypline, please check out the official project [documentation](https://princeton-ddss.github.io/hypline/latest/).
