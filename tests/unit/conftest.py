import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from hypline.bids import RAW_BOLD_ENTITIES


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

    def raw_func_dir(self, *, sub: str, ses: str | None = None) -> Path:
        return self._sub_ses_dir(self.root, sub, ses) / "func"

    def _entities(
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

    def _stem(self, entities: dict[str, str]) -> str:
        return "_".join(f"{k}-{v}" for k, v in entities.items())

    def _sub_ses_dir(self, area_root: Path, sub: str, ses: str | None) -> Path:
        sub_dir = area_root / f"sub-{sub}"
        return sub_dir / f"ses-{ses}" if ses is not None else sub_dir

    def _write(self, path: Path, *, content: str | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if content is None:
            path.touch()
        else:
            path.write_text(content)

    def _add(
        self,
        *,
        dir: Path,
        sub: str,
        ses: str | None,
        task: str | None,
        run: str | None,
        suffix: str,
        ext: str,
        content: str | None = None,
        sidecar_json: dict | None = None,
        extra_entities: dict[str, str] | None = None,
    ) -> Path:
        entities = self._entities(sub, ses, task, run, **(extra_entities or {}))
        path = dir / f"{self._stem(entities)}_{suffix}{ext}"
        self._write(path, content=content)
        if sidecar_json is not None:
            stem = path.name.partition(".")[0]  # handle compound exts like `.nii.gz`
            self._write(path.parent / f"{stem}.json", content=json.dumps(sidecar_json))
        return path

    def _add_raw(
        self,
        *,
        sub: str,
        ses: str | None,
        task: str | None,
        run: str | None,
        suffix: str,
        ext: str,
        content: str | None = None,
        sidecar_json: dict | None = None,
        extra_entities: dict[str, str] | None = None,
    ) -> Path:
        extras = extra_entities or {}
        invalid = set(extras) - set(RAW_BOLD_ENTITIES)
        if invalid:
            raise ValueError(
                f"Raw BIDS func/ disallows non-identity entities: {sorted(invalid)}"
            )
        return self._add(
            dir=self.raw_func_dir(sub=sub, ses=ses),
            sub=sub,
            ses=ses,
            task=task,
            run=run,
            suffix=suffix,
            ext=ext,
            content=content,
            sidecar_json=sidecar_json,
            extra_entities=extras,
        )

    def _add_fmriprep(
        self,
        *,
        sub: str,
        ses: str | None,
        task: str | None,
        run: str | None,
        suffix: str,
        ext: str,
        content: str | None = None,
        sidecar_json: dict | None = None,
        extra_entities: dict[str, str] | None = None,
    ) -> Path:
        return self._add(
            dir=self.func_dir(sub=sub, ses=ses),
            sub=sub,
            ses=ses,
            task=task,
            run=run,
            suffix=suffix,
            ext=ext,
            content=content,
            sidecar_json=sidecar_json,
            extra_entities=extra_entities,
        )

    def add_stimulus(
        self,
        *,
        sub: str,
        ses: str | None = None,
        task: str | None = None,
        run: str | None = None,
        kind: str,
        ext: str,
        extra_entities: dict[str, str] | None = None,
    ) -> Path:
        entities = self._entities(sub, ses, task, run, **(extra_entities or {}))
        entities["stim"] = kind
        kind_dir = self._sub_ses_dir(self.stimuli_dir, sub, ses) / kind
        path = kind_dir / f"{self._stem(entities)}{ext}"
        self._write(path)
        return path

    def add_feature(
        self,
        *,
        sub: str,
        ses: str | None = None,
        task: str | None = None,
        run: str | None = None,
        kind: str,
        metadata: dict[str, Any] | None = None,
        extra_entities: dict[str, str] | None = None,
    ) -> Path:
        from hypline.features import save_feature

        entities = self._entities(sub, ses, task, run, **(extra_entities or {}))
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
        extra_entities: dict[str, str] | None = None,
    ) -> Path:
        extras = extra_entities or {}
        reserved = {"space", "desc"} & set(extras)
        if reserved:
            raise ValueError(
                f"add_bold disallows overriding reserved entities: {sorted(reserved)}"
            )
        self._add_raw(
            sub=sub,
            ses=ses,
            task=task,
            run=run,
            suffix="bold",
            ext=".nii.gz",
            sidecar_json={"RepetitionTime": tr},
            extra_entities={k: v for k, v in extras.items() if k in RAW_BOLD_ENTITIES},
        )
        return self._add_fmriprep(
            sub=sub,
            ses=ses,
            task=task,
            run=run,
            suffix="bold",
            ext=".nii.gz",
            sidecar_json={"RepetitionTime": tr},
            extra_entities={"space": space, "desc": desc, **extras},
        )

    def add_events(
        self,
        *,
        sub: str,
        ses: str | None = None,
        task: str | None = None,
        run: str | None = None,
        rows: list[dict[str, str | float]] | None = None,
        sidecar_json: dict | None = None,
    ) -> Path:
        """Write events.tsv (and optionally events.json) under raw BIDS func/."""
        tsv_rows = [
            f"{row['trial_type']}\t{row['onset']}\t{row['duration']}"
            for row in (rows or [])
        ]
        return self._add_raw(
            sub=sub,
            ses=ses,
            task=task,
            run=run,
            suffix="events",
            ext=".tsv",
            sidecar_json=sidecar_json,
            content="trial_type\tonset\tduration\n" + "\n".join(tsv_rows),
        )

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
            extra_entities={"space": space, "desc": "preproc"},
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
            extra_entities={"space": space, "desc": "brain"},
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
            extra_entities={"space": space, "desc": desc},
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
            extra_entities={"desc": "confounds"},
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
            extra_entities={"from": "T1w", "to": "scanner", "mode": "image"},
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
