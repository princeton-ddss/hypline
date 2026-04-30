from pathlib import Path

import pytest

SUB = "001"
TASK = "conv"
SPACE = "MNI152NLin6Asym"


class BIDSTree:
    """Minimal on-disk fixture for BIDS-derivatives tests.

    Files are empty (touch-only); naming follows fMRIPrep BIDS-derivatives
    convention. All three dirs are pre-created under a single tmp_path root.
    """

    def __init__(self, root: Path):
        self.features_dir = root / "features"
        self.bold_dir = root / "bold"
        self.output_dir = root / "output"
        for dir in (self.features_dir, self.bold_dir, self.output_dir):
            dir.mkdir(parents=True)

    def _stem(self, entities: dict[str, str]) -> str:
        return "_".join(f"{k}-{v}" for k, v in entities.items())

    def _write(
        self,
        directory: Path,
        entities: dict[str, str],
        *,
        suffix: str | None = None,
        ext: str,
        sidecar_json: str | None = None,
    ) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        stem = self._stem(entities)
        name = stem if suffix is None else f"{stem}_{suffix}"
        path = directory / f"{name}{ext}"
        path.touch()
        if sidecar_json is not None:
            (directory / f"{name}.json").write_text(sidecar_json)
        return path

    def _identity_entities(
        self,
        sub: str,
        ses: str | None,
        task: str | None,
        run: str | None,
    ) -> dict[str, str]:
        entities: dict[str, str] = {"sub": sub}
        if ses is not None:
            entities["ses"] = ses
        if task is not None:
            entities["task"] = task
        if run is not None:
            entities["run"] = run
        return entities

    def _bold_entities(
        self,
        sub: str,
        ses: str | None,
        task: str | None,
        run: str | None,
        space: str,
    ) -> dict[str, str]:
        return {
            **self._identity_entities(sub, ses, task, run),
            "space": space,
            "desc": "preproc",
        }

    def add_feature(
        self,
        feature: str,
        *,
        sub: str = SUB,
        ses: str | None = None,
        task: str | None = TASK,
        run: str | None = None,
        subdir: str | None = None,
        **extra_entities: str,
    ) -> Path:
        entities = self._identity_entities(sub, ses, task, run)
        entities.update(extra_entities)
        entities["feature"] = feature
        directory = self.features_dir if subdir is None else self.features_dir / subdir
        return self._write(directory, entities, ext=".parquet")

    def add_bold(
        self,
        *,
        sub: str = SUB,
        ses: str | None = None,
        task: str | None = TASK,
        run: str | None = None,
        tr: float = 2.0,
        space: str = SPACE,
        subdir: str | None = None,
    ) -> Path:
        entities = self._bold_entities(sub, ses, task, run, space)
        directory = self.bold_dir if subdir is None else self.bold_dir / subdir
        return self._write(
            directory,
            entities,
            suffix="bold",
            ext=".nii.gz",
            sidecar_json=f'{{"RepetitionTime": {tr}}}',
        )

    def add_events(
        self,
        *,
        sub: str = SUB,
        ses: str | None = None,
        task: str | None = TASK,
        run: str | None = None,
        rows: list[dict[str, str | float]] | None = None,
        events_json: dict | None = None,
    ) -> Path:
        """Write an events.tsv file, optionally with a colocated events.json sidecar.

        rows is a list of dicts with trial_type, onset, duration, e.g.:
            [{"trial_type": "block-1", "onset": 0.0, "duration": 100.0}]

        events_json, if provided, is written to the matching events.json sidecar
        as raw JSON content (e.g., {"Segments": [{"block": "1", "cond": "R"}, ...]}).
        """
        import json

        identity = self._identity_entities(sub, ses, task, run)
        stem = self._stem(identity)

        tsv_rows = []
        if rows:
            for row in rows:
                tsv_rows.append(
                    f"{row['trial_type']}\t{row['onset']}\t{row['duration']}"
                )
        events_path = self.bold_dir / f"{stem}_events.tsv"
        events_path.write_text("trial_type\tonset\tduration\n" + "\n".join(tsv_rows))

        if events_json is not None:
            (self.bold_dir / f"{stem}_events.json").write_text(json.dumps(events_json))

        return events_path

    def add_boldref(
        self,
        *,
        sub: str = SUB,
        ses: str | None = None,
        task: str | None = TASK,
        run: str | None = None,
        space: str = SPACE,
    ) -> Path:
        return self._write(
            self.bold_dir,
            self._bold_entities(sub, ses, task, run, space),
            suffix="boldref",
            ext=".nii.gz",
        )

    def add_brain_mask(
        self,
        *,
        sub: str = SUB,
        ses: str | None = None,
        task: str | None = TASK,
        run: str | None = None,
        space: str = SPACE,
    ) -> Path:
        entities = {**self._bold_entities(sub, ses, task, run, space), "desc": "brain"}
        return self._write(
            self.bold_dir, entities, suffix="mask", ext=".nii.gz", sidecar_json="{}"
        )

    def add_dseg(
        self,
        *,
        sub: str = SUB,
        ses: str | None = None,
        task: str | None = TASK,
        run: str | None = None,
        space: str = SPACE,
        label: str = "aparcaseg",
    ) -> Path:
        entities = {**self._bold_entities(sub, ses, task, run, space), "desc": label}
        return self._write(self.bold_dir, entities, suffix="dseg", ext=".nii.gz")

    def add_confounds(
        self,
        *,
        sub: str = SUB,
        ses: str | None = None,
        task: str | None = TASK,
        run: str | None = None,
    ) -> Path:
        entities = {**self._identity_entities(sub, ses, task, run), "desc": "confounds"}
        return self._write(
            self.bold_dir, entities, suffix="timeseries", ext=".tsv", sidecar_json="{}"
        )

    def add_xfm(
        self,
        *,
        sub: str = SUB,
        ses: str | None = None,
        task: str | None = TASK,
        run: str | None = None,
    ) -> Path:
        entities = {
            **self._identity_entities(sub, ses, task, run),
            "from": "T1w",
            "to": "scanner",
            "mode": "image",
        }
        return self._write(self.bold_dir, entities, suffix="xfm", ext=".txt")

    def add_bold_siblings(
        self,
        *,
        sub: str = SUB,
        ses: str | None = None,
        task: str | None = TASK,
        run: str | None = None,
        space: str = SPACE,
    ) -> None:
        """Drop the full set of realistic fMRIPrep sibling files for a BOLD run."""
        self.add_boldref(sub=sub, ses=ses, task=task, run=run, space=space)
        self.add_brain_mask(sub=sub, ses=ses, task=task, run=run, space=space)
        self.add_dseg(
            sub=sub, ses=ses, task=task, run=run, space=space, label="aparcaseg"
        )
        self.add_dseg(sub=sub, ses=ses, task=task, run=run, space=space, label="aseg")
        self.add_confounds(sub=sub, ses=ses, task=task, run=run)
        self.add_xfm(sub=sub, ses=ses, task=task, run=run)


@pytest.fixture()
def tree(tmp_path: Path) -> BIDSTree:
    return BIDSTree(tmp_path)
