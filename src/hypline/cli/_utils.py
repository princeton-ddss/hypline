from collections.abc import Callable, Iterable
from contextlib import contextmanager
from pathlib import Path

import typer
from loguru import logger

from hypline.bids import Identity


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
def id_log(bids_root: Path, *command_parts: str, id_key: Identity, id_value: str):
    log_path = bids_root / "logs" / Path(*command_parts) / f"{id_key}-{id_value}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sink_id = logger.add(
        log_path,
        level="DEBUG",
        rotation="5 MB",
        retention=1,
        format="{time:YYYY-MM-DD HH:mm:ss} {level: <7} {message}",
    )
    try:
        yield
    finally:
        logger.remove(sink_id)


def run_per_id(
    bids_root: Path,
    *command_parts: str,
    id_key: Identity,
    id_values: Iterable[str],
    task: Callable[[str], None],
):
    """Run `task` per identity, logging and collecting failures.

    `id_key` is the identity prefix (`sub` or `dyad`) — it labels both the
    per-id log file and the failure summary, so dyad-keyed generators report
    `dyad-XX` rather than `sub-XX`.
    """
    id_values = list(id_values)
    failed = []
    for id_val in id_values:
        with id_log(bids_root, *command_parts, id_key=id_key, id_value=id_val):
            try:
                task(id_val)
            except Exception:
                logger.exception("{}-{} failed", id_key, id_val)
                failed.append(id_val)

    if failed:
        logger.error(
            "{}/{} {}s failed: {}",
            len(failed),
            len(id_values),
            id_key,
            ",".join(failed),
        )
        raise typer.Exit(code=1)
