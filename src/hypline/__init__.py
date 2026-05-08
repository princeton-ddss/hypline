from importlib.metadata import version

from .cli import app

__version__ = version("hypline")


def main():
    app()
