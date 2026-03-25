import typer

from .clean import clean as clean_func
from .featuregen import app as featuregen_app
from .transcribe import transcribe as transcribe_func

app = typer.Typer()
app.command(name="clean")(clean_func)
app.command(name="transcribe")(transcribe_func)
app.add_typer(featuregen_app, name="featuregen")


@app.callback()
def callback():
    """
    An opinionated framework/toolbox for conducting
    data cleaning and analysis in hyperscanning studies
    involving dyadic conversations.
    """
