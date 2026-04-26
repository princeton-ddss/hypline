import os
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

BIDS_ENTITY_RE = re.compile(r"^[a-z]+-[a-zA-Z0-9]+$")
_BIDS_SUFFIX_RE = re.compile(r"^[a-zA-Z0-9]+$")


class BIDSPath:
    _entities: dict[str, str]
    _suffix: str | None
    _extension: str
    _path: Path

    def __init__(self, path: str | os.PathLike[str]):
        self._path = Path(path)
        self._entities = {}
        self._suffix = None

        name = self._path.name
        dot_idx = name.find(".")
        if dot_idx == -1:
            stem = name
            self._extension = ""
        else:
            stem = name[:dot_idx]
            self._extension = name[dot_idx:]

        segments = stem.split("_")
        for i, segment in enumerate(segments):
            if "-" in segment:
                key, _, value = segment.partition("-")
                if not BIDS_ENTITY_RE.match(segment):
                    raise ValueError(f"Invalid BIDS entity: {segment!r}")
                if key in self._entities:
                    raise ValueError(f"Duplicate BIDS entity key: {key!r}")
                self._entities[key] = value
            elif i == len(segments) - 1:
                if not _BIDS_SUFFIX_RE.match(segment):
                    raise ValueError(f"Invalid BIDS suffix: {segment!r}")
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
    def extension(self) -> str:
        return self._extension

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
        entity = f"{key}-{value}"
        if not BIDS_ENTITY_RE.match(entity):
            raise ValueError(f"Invalid BIDS entity: {entity!r}")

        entities = dict(self._entities)
        entities[key] = value

        parts = [f"{k}-{v}" for k, v in entities.items()]
        if self._suffix:
            parts.append(self._suffix)
        name = "_".join(parts) + self._extension

        return BIDSPath(self._path.with_name(name))

    def __repr__(self) -> str:
        return f"BIDSPath({str(self._path)!r})"


def validate_bids_entities(*entities: str) -> None:
    for entity in entities:
        if not BIDS_ENTITY_RE.match(entity):
            raise ValueError(f"Invalid BIDS entity: {entity!r}")


def validate_entity_invariance(
    paths: Sequence[BIDSPath], entities: Iterable[str]
) -> None:
    """Raise if any entity has more than one distinct value across paths."""
    for entity in entities:
        values = {p.entities.get(entity) for p in paths}
        if len(values) > 1:
            display = sorted(v if v is not None else "(none)" for v in values)
            raise ValueError(f"Inconsistent {entity!r} across files: found {display}.")
