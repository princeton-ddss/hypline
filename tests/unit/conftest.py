import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest


class BIDSTree:
    """Minimal on-disk fixture mirroring the hypline BIDS tree.

    Layout (`ses` is optional everywhere):
        stimuli/sub-XX/[ses-YY/]<kind>/
        features/sub-XX/[ses-YY/]<kind>/
        derivatives/fmriprep/sub-XX/[ses-YY/]func/

    All helpers require identity entities (`sub`, optional `ses`, `task`, `run`)
    specified explicitly so the data shape stays readable at the callsite.
    """

    def __init__(self, root: Path):
        self.root = root

    @property
    def stimuli_dir(self) -> Path:
        return self.root / "stimuli"

    @property
    def features_dir(self) -> Path:
        return self.root / "features"

    @property
    def fmriprep_dir(self) -> Path:
        return self.root / "derivatives" / "fmriprep"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    def func_dir(self, *, sub: str, ses: str | None = None) -> Path:
        return self._sub_ses_dir(self.fmriprep_dir, sub, ses) / "func"

    def _stem(self, entities: dict[str, str]) -> str:
        return "_".join(f"{k}-{v}" for k, v in entities.items())

    def _identity(
        self,
        sub: str,
        ses: str | None,
        task: str | None,
        run: str | None,
        **extra_entities: str,
    ) -> dict[str, str]:
        entities: dict[str, str] = {"sub": sub}
        if ses is not None:
            entities["ses"] = ses
        if task is not None:
            entities["task"] = task
        if run is not None:
            entities["run"] = run
        entities.update(extra_entities)
        return entities

    def _sub_ses_dir(self, area_root: Path, sub: str, ses: str | None) -> Path:
        sub_dir = area_root / f"sub-{sub}"
        return sub_dir / f"ses-{ses}" if ses is not None else sub_dir

    def _touch(self, path: Path, *, sidecar_json: dict | None = None) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        if sidecar_json is not None:
            stem = path.name.partition(".")[0]  # handle compound exts like `.nii.gz`
            (path.parent / f"{stem}.json").write_text(json.dumps(sidecar_json))
        return path

    def add_stimulus(
        self,
        *,
        sub: str,
        ses: str | None = None,
        task: str | None = None,
        run: str | None = None,
        kind: str,
        ext: str,
        **extra_entities: str,
    ) -> Path:
        entities = self._identity(sub, ses, task, run, **extra_entities)
        entities["stim"] = kind
        kind_dir = self._sub_ses_dir(self.stimuli_dir, sub, ses) / kind
        return self._touch(kind_dir / f"{self._stem(entities)}{ext}")

    def add_feature(
        self,
        *,
        sub: str,
        ses: str | None = None,
        task: str | None = None,
        run: str | None = None,
        kind: str,
        metadata: dict[str, Any] | None = None,
        **extra_entities: str,
    ) -> Path:
        from hypline.features import save_feature

        entities = self._identity(sub, ses, task, run, **extra_entities)
        entities["feat"] = kind
        kind_dir = self._sub_ses_dir(self.features_dir, sub, ses) / kind
        kind_dir.mkdir(parents=True, exist_ok=True)
        path = kind_dir / f"{self._stem(entities)}.parquet"
        df = pl.DataFrame(
            {"start_time": [0.0], "feature": [[0.0]]},
            schema={"start_time": pl.Float64, "feature": pl.Array(pl.Float64, 1)},
        )
        save_feature(df, path, metadata=metadata)
        return path

    def _add_fmriprep(
        self,
        *,
        sub: str,
        ses: str | None,
        task: str | None,
        run: str | None,
        suffix: str,
        ext: str,
        sidecar_json: dict | None = None,
        **extra_entities: str,
    ) -> Path:
        entities = self._identity(sub, ses, task, run, **extra_entities)
        path = self.func_dir(sub=sub, ses=ses) / f"{self._stem(entities)}_{suffix}{ext}"
        return self._touch(path, sidecar_json=sidecar_json)

    def add_bold(
        self,
        *,
        sub: str,
        ses: str | None = None,
        task: str | None = None,
        run: str | None = None,
        space: str,
        desc: str = "preproc",
        tr: float = 2.0,
        **extra_entities: str,
    ) -> Path:
        return self._add_fmriprep(
            sub=sub,
            ses=ses,
            task=task,
            run=run,
            suffix="bold",
            ext=".nii.gz",
            sidecar_json={"RepetitionTime": tr},
            space=space,
            desc=desc,
            **extra_entities,
        )

    def add_events(
        self,
        *,
        sub: str,
        ses: str | None = None,
        task: str | None = None,
        run: str | None = None,
        rows: list[dict[str, str | float]] | None = None,
        events_json: dict | None = None,
    ) -> Path:
        """Write events.tsv (and optionally events.json) colocated with BOLD."""
        stem = self._stem(self._identity(sub, ses, task, run))
        func = self.func_dir(sub=sub, ses=ses)
        func.mkdir(parents=True, exist_ok=True)

        tsv_rows = [
            f"{row['trial_type']}\t{row['onset']}\t{row['duration']}"
            for row in (rows or [])
        ]
        events_path = func / f"{stem}_events.tsv"
        events_path.write_text("trial_type\tonset\tduration\n" + "\n".join(tsv_rows))

        if events_json is not None:
            (func / f"{stem}_events.json").write_text(json.dumps(events_json))

        return events_path

    def add_boldref(
        self,
        *,
        sub: str,
        ses: str | None = None,
        task: str | None = None,
        run: str | None = None,
        space: str,
    ) -> Path:
        return self._add_fmriprep(
            sub=sub,
            ses=ses,
            task=task,
            run=run,
            suffix="boldref",
            ext=".nii.gz",
            space=space,
            desc="preproc",
        )

    def add_brain_mask(
        self,
        *,
        sub: str,
        ses: str | None = None,
        task: str | None = None,
        run: str | None = None,
        space: str,
    ) -> Path:
        return self._add_fmriprep(
            sub=sub,
            ses=ses,
            task=task,
            run=run,
            suffix="mask",
            ext=".nii.gz",
            sidecar_json={},
            space=space,
            desc="brain",
        )

    def add_dseg(
        self,
        *,
        sub: str,
        ses: str | None = None,
        task: str | None = None,
        run: str | None = None,
        space: str,
        desc: str = "aparcaseg",
    ) -> Path:
        return self._add_fmriprep(
            sub=sub,
            ses=ses,
            task=task,
            run=run,
            suffix="dseg",
            ext=".nii.gz",
            space=space,
            desc=desc,
        )

    def add_confounds(
        self,
        *,
        sub: str,
        ses: str | None = None,
        task: str | None = None,
        run: str | None = None,
    ) -> Path:
        return self._add_fmriprep(
            sub=sub,
            ses=ses,
            task=task,
            run=run,
            suffix="timeseries",
            ext=".tsv",
            sidecar_json={},
            desc="confounds",
        )

    def add_xfm(
        self,
        *,
        sub: str,
        ses: str | None = None,
        task: str | None = None,
        run: str | None = None,
    ) -> Path:
        return self._add_fmriprep(
            sub=sub,
            ses=ses,
            task=task,
            run=run,
            suffix="xfm",
            ext=".txt",
            sidecar_json=None,
            **{"from": "T1w", "to": "scanner", "mode": "image"},  # `from` is reserved
        )

    def add_bold_siblings(
        self,
        *,
        sub: str,
        ses: str | None = None,
        task: str | None = None,
        run: str | None = None,
        space: str,
    ) -> None:
        """Drop the full set of realistic fMRIPrep sibling files for a BOLD run."""
        self.add_boldref(sub=sub, ses=ses, task=task, run=run, space=space)
        self.add_brain_mask(sub=sub, ses=ses, task=task, run=run, space=space)
        self.add_dseg(
            sub=sub, ses=ses, task=task, run=run, space=space, desc="aparcaseg"
        )
        self.add_dseg(sub=sub, ses=ses, task=task, run=run, space=space, desc="aseg")
        self.add_confounds(sub=sub, ses=ses, task=task, run=run)
        self.add_xfm(sub=sub, ses=ses, task=task, run=run)


@pytest.fixture()
def tree(tmp_path: Path) -> BIDSTree:
    return BIDSTree(tmp_path)
