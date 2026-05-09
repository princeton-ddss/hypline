from multiprocessing import Process

import typer


def split_csv(value: str | None, param_hint: str | None = None) -> list[str] | None:
    if value is None:
        return None
    if any(c.isspace() for c in value):
        raise typer.BadParameter(
            "must not contain whitespace (e.g., 01,02,03)", param_hint=param_hint
        )
    items = value.split(",")
    if any(not v for v in items):
        raise typer.BadParameter("empty value between commas", param_hint=param_hint)
    if len(items) != len(set(items)):
        raise typer.BadParameter("duplicate values not allowed", param_hint=param_hint)
    return items


class DillProcess(Process):
    """
    Extend the `Process` class to support serialization
    of closures and local functions.

    Notes
    -----
    Adapted from https://stackoverflow.com/a/72776044.
    """

    def __init__(self, *args, **kwargs):
        import dill

        super().__init__(*args, **kwargs)
        self._target = dill.dumps(self._target)
        self._args, self._kwargs = self._args, self._kwargs  # For type checker

    def run(self):
        import dill

        if self._target:
            self._target = dill.loads(self._target)
            self._target(*self._args, **self._kwargs)
