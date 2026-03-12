import typer

from .clean import clean as clean_func

app = typer.Typer()
app.command(name="clean")(clean_func)


@app.callback()
def callback():
    """
    An opinionated framework/toolbox for conducting
    data cleaning and analysis in hyperscanning studies
    involving dyadic conversations.
    """
