import os
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

BIDS_ENTITY_KEY_RE = re.compile(r"^[a-z]+$")
BIDS_ENTITY_VALUE_RE = re.compile(r"^[a-zA-Z0-9]+$")
BIDS_ENTITY_RE = re.compile(r"^[a-z]+-[a-zA-Z0-9]+$")
BIDS_SUFFIX_RE = re.compile(r"^[a-zA-Z0-9]+$")
EXTENSION_RE = re.compile(r"^\.[a-zA-Z0-9]+(\.[a-zA-Z0-9]+)*$")

# Entities that may appear in raw BOLD run filenames
RAW_BOLD_ENTITIES = (
    "sub",
    "ses",
    "task",
    "acq",
    "ce",
    "rec",
    "dir",
    "run",
    "echo",
    "part",
    "chunk",
)


def validate_bids_entities(*entities: str) -> None:
    for entity in entities:
        if not BIDS_ENTITY_RE.match(entity):
            raise ValueError(f"Invalid BIDS entity: {entity!r}")


def validate_suffix(suffix: str) -> None:
    if not BIDS_SUFFIX_RE.match(suffix):
        raise ValueError(f"Invalid BIDS suffix: {suffix!r}")


def validate_extension(ext: str) -> None:
    if not EXTENSION_RE.match(ext):
        raise ValueError(f"Invalid extension: {ext!r}")


class BIDSPath:
    _entities: dict[str, str]
    _suffix: str | None
    _ext: str
    _path: Path

    def __init__(self, path: str | os.PathLike[str]):
        self._path = Path(path)
        self._entities = {}
        self._suffix = None

        name = self._path.name
        dot_idx = name.find(".")
        if dot_idx == -1:
            stem = name
            self._ext = ""
        else:
            stem = name[:dot_idx]
            self._ext = name[dot_idx:]

        segments = stem.split("_")
        for i, segment in enumerate(segments):
            if "-" in segment:
                key, _, value = segment.partition("-")
                validate_bids_entities(segment)
                if key in self._entities:
                    raise ValueError(f"Duplicate BIDS entity key: {key!r}")
                self._entities[key] = value
            elif i == len(segments) - 1:
                validate_suffix(segment)
                self._suffix = segment
            else:
                raise ValueError(
                    f"Non-entity segment {segment!r} must be the last segment (suffix)"
                )

        if not self._entities:
            raise ValueError(f"No BIDS entities found in: {name!r}")

    @property
    def entities(self) -> dict[str, str]:
        return dict(self._entities)

    @property
    def suffix(self) -> str | None:
        return self._suffix

    @property
    def ext(self) -> str:
        return self._ext

    @property
    def path(self) -> Path:
        return self._path

    def __getattr__(self, name: str) -> str:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._entities[name]
        except KeyError:
            raise AttributeError(
                f"No BIDS entity {name!r} in {self._path.name!r}"
            ) from None

    def with_entity(self, key: str, value: str) -> "BIDSPath":
        """Return a new BIDSPath with `key` set to `value`.

        Existing keys keep their position; new keys are appended. Order is
        preserved in the filename stem of derived paths.
        """
        validate_bids_entities(f"{key}-{value}")

        entities = dict(self._entities)
        entities[key] = value

        parts = [f"{k}-{v}" for k, v in entities.items()]
        if self._suffix:
            parts.append(self._suffix)
        name = "_".join(parts) + self._ext

        return BIDSPath(self._path.with_name(name))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BIDSPath):
            return NotImplemented
        return self._path == other._path

    def __hash__(self) -> int:
        return hash(self._path)

    def __lt__(self, other: "BIDSPath") -> bool:
        if not isinstance(other, BIDSPath):
            return NotImplemented
        return self._path < other._path

    def __repr__(self) -> str:
        return f"BIDSPath({str(self._path)!r})"


def validate_entity_invariance(
    paths: Sequence[BIDSPath], entities: Iterable[str]
) -> None:
    """Raise if any entity has more than one distinct value across paths."""
    for entity in entities:
        values = {p.entities.get(entity) for p in paths}
        if len(values) > 1:
            display = sorted(v if v is not None else "(none)" for v in values)
            raise ValueError(f"Inconsistent {entity!r} across files: found {display}.")


def normalize_bids_filters(
    filters: list[str] | None,
    *,
    reserved: Iterable[str] = (),
) -> list[str]:
    filters = list(filters or [])
    validate_bids_entities(*filters)
    reserved_set = frozenset(reserved)
    for entity in filters:
        key = entity.split("-", 1)[0]
        if key in reserved_set:
            raise ValueError(
                f"bids_filters cannot contain {key!r} "
                "— use the dedicated argument instead"
            )
    return filters


def find_bids_files(
    directory: str | os.PathLike[str],
    ext: str,
    *,
    suffix: str | None = None,
    bids_filters: list[str] | None = None,
    recursive: bool = False,
) -> list[BIDSPath]:
    """Find BIDS files under `directory` matching `suffix` and `ext`.

    Non-parseable filenames are silently skipped. Filters sharing a key
    are OR'd; filters with different keys are AND'd. `recursive=True`
    descends into subdirectories. Results are sorted by path.
    """
    validate_extension(ext)
    if bids_filters:
        validate_bids_entities(*bids_filters)
    directory = Path(directory)
    ends_with = f"_{suffix}{ext}" if suffix is not None else ext

    groups: dict[str, list[str]] = {}
    for entity in bids_filters or ():
        key, _, value = entity.partition("-")
        groups.setdefault(key, []).append(value)

    files = directory.rglob("*") if recursive else directory.iterdir()
    results: list[BIDSPath] = []
    for f in files:
        if not (f.is_file() and f.name.endswith(ends_with)):
            continue
        try:
            bp = BIDSPath(f)
        except ValueError:
            continue
        if all(bp._entities.get(k) in vals for k, vals in groups.items()):
            results.append(bp)

    return sorted(results)
