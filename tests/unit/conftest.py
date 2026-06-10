import json
from functools import cache
from pathlib import Path
from typing import Any, Literal

import polars as pl
import pytest

from hypline.bids import BOLD_IDENTITY_ENTITIES, Identity

DEFAULT_BOLD_N_TRS = 10

# Header TR of minimal_nifti_gz: its identity affine makes zoom[3] (the TR) 1.0
HEADER_TR = 1.0


@cache
def minimal_nifti_gz(n_trs: int = DEFAULT_BOLD_N_TRS) -> bytes:
    import gzip

    import nibabel as nib
    import numpy as np

    data = np.arange(1, n_trs + 1, dtype=np.int16).reshape(1, 1, 1, n_trs)
    return gzip.compress(nib.Nifti1Image(data, np.eye(4)).to_bytes())


class BIDSTree:
    """Minimal on-disk fixture mirroring the hypline BIDS tree.

    Layout (`ses` is optional everywhere):
        stimuli/dyad-XX/[ses-YY/]<kind>[-<desc>]/
        features/dyad-XX/[ses-YY/]<kind>[-<desc>]/
        confounds/dyad-XX/[ses-YY/]<kind>[-<desc>]/
        derivatives/fmriprep/sub-XX/[ses-YY/]func/
        derivatives/hypline/sub-XX/[ses-YY/]func/

    The shared-conversation areas (stimuli/features/confounds) are dyad-keyed;
    BOLD-derived areas are sub-keyed. All helpers require identity entities (`sub`
    or `dyad`, optional `ses`, `task`, `run`) specified explicitly so the data
    shape stays readable at the callsite.
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
    def confounds_dir(self) -> Path:
        return self.root / "confounds"

    @property
    def nuisance_dir(self) -> Path:
        return self.root / "nuisance"

    @property
    def fmriprep_dir(self) -> Path:
        return self.root / "derivatives" / "fmriprep"

    @property
    def hypline_dir(self) -> Path:
        return self.root / "derivatives" / "hypline"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    def fmriprep_func_dir(self, *, sub: str, ses: str | None = None) -> Path:
        id_dir = self._identity_ses_dir(
            area_root=self.fmriprep_dir,
            id_key="sub",
            id_value=sub,
            ses=ses,
        )
        return id_dir / "func"

    def denoised_func_dir(self, *, sub: str, ses: str | None = None) -> Path:
        id_dir = self._identity_ses_dir(
            area_root=self.hypline_dir,
            id_key="sub",
            id_value=sub,
            ses=ses,
        )
        return id_dir / "func"

    def raw_func_dir(self, *, sub: str, ses: str | None = None) -> Path:
        id_dir = self._identity_ses_dir(
            area_root=self.root,
            id_key="sub",
            id_value=sub,
            ses=ses,
        )
        return id_dir / "func"

    def _identity_entities(
        self,
        *,
        id_key: Identity,
        id_value: str,
        ses: str | None,
        task: str | None,
        run: str | None,
        **extra_entities: str,
    ) -> dict[str, str]:
        entities: dict[str, str] = {id_key: id_value}
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

    def _identity_ses_dir(
        self,
        *,
        area_root: Path,
        id_key: Identity,
        id_value: str,
        ses: str | None,
    ) -> Path:
        id_dir = area_root / f"{id_key}-{id_value}"
        return id_dir / f"ses-{ses}" if ses is not None else id_dir

    def _write(self, path: Path, *, content: str | bytes | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if content is None:
            path.touch()
        elif isinstance(content, bytes):
            path.write_bytes(content)
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
        content: str | bytes | None = None,
        sidecar_json: dict | None = None,
        extra_entities: dict[str, str] | None = None,
    ) -> Path:
        entities = self._identity_entities(
            id_key="sub",
            id_value=sub,
            ses=ses,
            task=task,
            run=run,
            **(extra_entities or {}),
        )
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
        content: str | bytes | None = None,
        sidecar_json: dict | None = None,
        extra_entities: dict[str, str] | None = None,
    ) -> Path:
        extras = extra_entities or {}
        invalid = set(extras) - BOLD_IDENTITY_ENTITIES
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
        content: str | bytes | None = None,
        sidecar_json: dict | None = None,
        extra_entities: dict[str, str] | None = None,
    ) -> Path:
        return self._add(
            dir=self.fmriprep_func_dir(sub=sub, ses=ses),
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

    def _add_hypline(
        self,
        *,
        sub: str,
        ses: str | None,
        task: str | None,
        run: str | None,
        suffix: str,
        ext: str,
        content: str | bytes | None = None,
        sidecar_json: dict | None = None,
        extra_entities: dict[str, str] | None = None,
    ) -> Path:
        return self._add(
            dir=self.denoised_func_dir(sub=sub, ses=ses),
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

    def add_participants(self, mapping: dict[str, str]) -> Path:
        """Write `participants.tsv` from a bare `sub -> dyad` mapping.

        Stores `participant_id` with the BIDS `sub-` prefix and `dyad_id` with a
        `dyad-` prefix, matching what `read_participants` strips back off.
        """
        rows = ["participant_id\tdyad_id"]
        rows += [f"sub-{sub}\tdyad-{dyad}" for sub, dyad in mapping.items()]
        path = self.root / "participants.tsv"
        self._write(path, content="\n".join(rows) + "\n")
        return path

    def add_stimulus(
        self,
        *,
        dyad: str,
        ses: str | None = None,
        task: str,
        run: str | None = None,
        kind: str,
        ext: str,
        extra_entities: dict[str, str] | None = None,
    ) -> Path:
        extras = extra_entities or {}
        if "desc" in extras:
            raise ValueError("stimuli have no desc variants")
        entities = self._identity_entities(
            id_key="dyad",
            id_value=dyad,
            ses=ses,
            task=task,
            run=run,
            **extras,
        )
        id_dir = self._identity_ses_dir(
            area_root=self.stimuli_dir,
            id_key="dyad",
            id_value=dyad,
            ses=ses,
        )
        kind_dir = id_dir / kind
        path = kind_dir / f"{self._stem(entities)}_{kind}{ext}"
        self._write(path)
        return path

    def add_feature(
        self,
        *,
        dyad: str,
        ses: str | None = None,
        task: str,
        run: str | None = None,
        kind: str,
        desc: str | None = None,
        df: pl.DataFrame | None = None,
        metadata: dict[str, Any] | None = None,
        extra_entities: dict[str, str] | None = None,
    ) -> Path:
        from hypline.io import write_feature

        extras = extra_entities or {}
        if "desc" in extras:
            raise ValueError("pass desc as an explicit argument, not in extra_entities")
        entities = self._identity_entities(
            id_key="dyad",
            id_value=dyad,
            ses=ses,
            task=task,
            run=run,
            **extras,
        )
        entities["feat"] = kind
        if desc is not None:
            entities["desc"] = desc
        subdir = f"{kind}-{desc}" if desc else kind
        id_dir = self._identity_ses_dir(
            area_root=self.features_dir,
            id_key="dyad",
            id_value=dyad,
            ses=ses,
        )
        kind_dir = id_dir / subdir
        kind_dir.mkdir(parents=True, exist_ok=True)
        path = kind_dir / f"{self._stem(entities)}.parquet"
        if df is None:
            df = pl.DataFrame(
                {"start_time": [0.0], "feature": [[0.0]]},
                schema={"start_time": pl.Float64, "feature": pl.Array(pl.Float64, 1)},
            )
        write_feature(df, path, metadata=metadata)
        return path

    def add_confound(
        self,
        *,
        dyad: str,
        ses: str | None = None,
        task: str,
        run: str | None = None,
        kind: str,
        desc: str | None = None,
        df: pl.DataFrame | None = None,
        metadata: dict[str, Any] | None = None,
        extra_entities: dict[str, str] | None = None,
    ) -> Path:
        from hypline.io import write_confound

        extras = extra_entities or {}
        if "desc" in extras:
            raise ValueError("pass desc as an explicit argument, not in extra_entities")
        entities = self._identity_entities(
            id_key="dyad",
            id_value=dyad,
            ses=ses,
            task=task,
            run=run,
            **extras,
        )
        entities["conf"] = kind
        if desc is not None:
            entities["desc"] = desc
        subdir = f"{kind}-{desc}" if desc else kind
        id_dir = self._identity_ses_dir(
            area_root=self.confounds_dir,
            id_key="dyad",
            id_value=dyad,
            ses=ses,
        )
        kind_dir = id_dir / subdir
        kind_dir.mkdir(parents=True, exist_ok=True)
        path = kind_dir / f"{self._stem(entities)}.parquet"
        if df is None:
            df = pl.DataFrame(
                {"start_time": [0.0], "confound": [[0.0]]},
                schema={"start_time": pl.Float64, "confound": pl.Array(pl.Float64, 1)},
            )
        write_confound(df, path, repetition_time=2.0, tr_method=None, metadata=metadata)
        return path

    def add_nuisance(
        self,
        *,
        sub: str,
        ses: str | None = None,
        task: str,
        run: str | None = None,
        kind: str,
        desc: str | None = None,
        df: pl.DataFrame | None = None,
        extra_entities: dict[str, str] | None = None,
    ) -> Path:
        """Write a wide nuisance TSV under nuisance/sub-XX/[ses-YY/]<kind>[-<desc>]/.

        `df` columns become the regressors (one default scalar column if None).
        Unlike feature/confound, the file carries the `_timeseries` suffix and a
        `.tsv` extension, and is plain wide TSV (no metadata footer).
        """
        extras = extra_entities or {}
        if "desc" in extras:
            raise ValueError("pass desc as an explicit argument, not in extra_entities")
        entities = self._identity_entities(
            id_key="sub",
            id_value=sub,
            ses=ses,
            task=task,
            run=run,
            **extras,
        )
        entities["nuis"] = kind
        if desc is not None:
            entities["desc"] = desc
        subdir = f"{kind}-{desc}" if desc else kind
        id_dir = self._identity_ses_dir(
            area_root=self.nuisance_dir,
            id_key="sub",
            id_value=sub,
            ses=ses,
        )
        kind_dir = id_dir / subdir
        kind_dir.mkdir(parents=True, exist_ok=True)
        path = kind_dir / f"{self._stem(entities)}_timeseries.tsv"
        if df is None:
            df = pl.DataFrame({"reg0": [0.0]})
        self._write(path, content=df.write_csv(separator="\t"))
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
        write_raw: bool = True,
        area: Literal["fmriprep", "hypline"] | None = None,
        extra_entities: dict[str, str] | None = None,
    ) -> Path:
        extras = extra_entities or {}
        reserved = {"space", "desc"} & set(extras)
        if reserved:
            raise ValueError(
                f"add_bold disallows overriding reserved entities: {sorted(reserved)}"
            )
        if write_raw:
            self._add_raw(
                sub=sub,
                ses=ses,
                task=task,
                run=run,
                suffix="bold",
                ext=".nii.gz",
                content=minimal_nifti_gz(),
                sidecar_json={"RepetitionTime": tr},
                extra_entities={
                    k: v for k, v in extras.items() if k in BOLD_IDENTITY_ENTITIES
                },
            )
        # Mirror Encoding._discover_bold: preproc in fmriprep, else hypline;
        # `area` overrides for off-convention descs
        _area = area or ("fmriprep" if desc == "preproc" else "hypline")
        _add_func = self._add_fmriprep if _area == "fmriprep" else self._add_hypline
        return _add_func(
            sub=sub,
            ses=ses,
            task=task,
            run=run,
            suffix="bold",
            ext=".nii.gz",
            content=minimal_nifti_gz(),
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

    def add_confounds_timeseries(
        self,
        *,
        sub: str,
        ses: str | None = None,
        task: str | None = None,
        run: str | None = None,
        df: pl.DataFrame | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Write desc-confounds_timeseries.tsv (+ .json sidecar) under fmriprep func/.

        `df` columns become the confound regressors (empty if None); `metadata`
        is the JSON sidecar of CompCor component descriptions (`{}` if None).
        """
        content = df.write_csv(separator="\t") if df is not None else None
        return self._add_fmriprep(
            sub=sub,
            ses=ses,
            task=task,
            run=run,
            suffix="timeseries",
            ext=".tsv",
            content=content,
            sidecar_json=metadata or {},
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
        self.add_confounds_timeseries(sub=sub, ses=ses, task=task, run=run)
        self.add_xfm(sub=sub, ses=ses, task=task, run=run)


@pytest.fixture()
def tree(tmp_path: Path) -> BIDSTree:
    return BIDSTree(tmp_path)
