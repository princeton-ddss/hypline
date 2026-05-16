import sys
from typing import Annotated

import typer
from loguru import logger

from .clean import clean as clean_func
from .featuregen import app as featuregen_app
from .transcribe import transcribe as transcribe_func

app = typer.Typer()
app.command(name="clean")(clean_func)
app.command(name="transcribe")(transcribe_func)
app.add_typer(featuregen_app, name="featuregen")


def _version_callback(value: bool):
    if value:
        from hypline import __version__

        typer.echo(__version__)
        raise typer.Exit()


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
        colorize=True,
    )
