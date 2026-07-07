import sys
from typing import Annotated

import typer
from loguru import logger

from .confoundgen import app as confoundgen_app
from .denoise import denoise as denoise_func
from .encoding import app as encoding_app
from .featuregen import app as featuregen_app
from .transcribe import transcribe as transcribe_func

app = typer.Typer()
app.command(name="denoise")(denoise_func)
app.command(name="transcribe")(transcribe_func)
app.add_typer(featuregen_app, name="featuregen")
app.add_typer(confoundgen_app, name="confoundgen")
app.add_typer(encoding_app, name="encoding")


def _version_callback(value: bool):
    if value:
        from hypline._version import __version__

        typer.echo(__version__)
        raise typer.Exit()


def _console_format(_record):
    # callable format (not a string) stops loguru from appending the
    # traceback here, keeping it in the file logs only
    return "{time:YYYY-MM-DD HH:mm:ss} | <level>{level: <7}</level> | {message}\n"


@app.callback()
def callback(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show version and exit",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", help="Show debug-level output"),
    ] = False,
):
    """
    An opinionated framework/toolbox for conducting
    data cleaning and analysis in hyperscanning studies
    involving dyadic conversations.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if verbose else "INFO",
        colorize=sys.stderr.isatty(),
        format=_console_format,
    )
