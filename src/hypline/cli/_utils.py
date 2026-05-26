from collections.abc import Callable, Iterable
from contextlib import contextmanager
from multiprocessing import Process
from pathlib import Path

import typer
from loguru import logger


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


@contextmanager
def subject_log(bids_root: Path, *command_parts: str, sub_id: str):
    log_path = bids_root / "logs" / Path(*command_parts) / f"sub-{sub_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sink_id = logger.add(log_path, level="DEBUG", rotation="5 MB", retention=1)
    try:
        yield
    finally:
        logger.remove(sink_id)


def run_per_subject(
    bids_root: Path,
    *command_parts: str,
    sub_ids: Iterable[str],
    task: Callable[[str], None],
):
    sub_ids = list(sub_ids)
    failed = []
    for sub_id in sub_ids:
        with subject_log(bids_root, *command_parts, sub_id=sub_id):
            try:
                task(sub_id)
            except Exception:
                logger.exception("sub-{} failed", sub_id)
                failed.append(sub_id)

    if failed:
        logger.error(
            "{}/{} subjects failed: {}", len(failed), len(sub_ids), ",".join(failed)
        )
        raise typer.Exit(code=1)


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
