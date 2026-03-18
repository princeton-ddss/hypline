import typer

from .clean import clean as clean_func
from .transcribe import transcribe as transcribe_func

app = typer.Typer()
app.command(name="clean")(clean_func)
app.command(name="transcribe")(transcribe_func)


@app.callback()
def callback():
    """
    An opinionated framework/toolbox for conducting
    data cleaning and analysis in hyperscanning studies
    involving dyadic conversations.
    """
