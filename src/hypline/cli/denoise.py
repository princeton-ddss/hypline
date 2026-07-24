from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from hypline.enums import BoldSpace

from ._utils import run_per_id, split_csv

app = typer.Typer()


@app.command(name="denoise")
def denoise(
    bids_root: Annotated[
        Path,
        typer.Argument(
            help="BIDS dataset root (contains stimuli/, features/, derivatives/)",
            show_default=False,
        ),
    ],
    columns: Annotated[
        str | None,
        typer.Option(
            help="""
            Comma-separated fmriprep confound columns: exact tsv names (e.g.,
            trans_x,rot_x) plus group prefixes that expand to all matches
            (cosine, motion_outlier). When no nuisance channel
            (--columns/--compcor/--custom-sources) is given, defaults to the
            Speer et al. 2024 set: motion and WM/CSF signal (each with squared
            and derivative expansions) plus cosine drift.
            """,
            show_default=False,
        ),
    ] = None,
    compcor: Annotated[
        str | None,
        typer.Option(
            "--compcor",
            help="""
            Comma-separated CompCor selectors <type>:<mask>:<n> (e.g., a:CSF:5 = top-5
            aCompCor in CSF; t::10 = top-10 tCompCor). <type> a=anatomical (mask
            required), t=temporal (no mask); <n> = top-N int or variance fraction (0-1)
            """,
            show_default=False,
        ),
    ] = None,
    custom_sources: Annotated[
        str | None,
        typer.Option(
            help="""
            Comma-separated nuisance/ sources as <kind>[-<desc>] (e.g.,
            physio-v1,resp); selects run-level regressor files from nuisance/.
            Requires --custom-columns
            """,
            show_default=False,
        ),
    ] = None,
    custom_columns: Annotated[
        str | None,
        typer.Option(
            help="""
            Comma-separated column names to select from the --custom-sources
            sources (selected from the horizontal concat of all sources)
            """,
            show_default=False,
        ),
    ] = None,
    space: Annotated[
        BoldSpace,
        typer.Option(
            help="BOLD data space to denoise",
        ),
    ] = BoldSpace.FS_AVERAGE_6,
    sub_ids: Annotated[
        str | None,
        typer.Option(
            help="Comma-separated subject IDs to process (e.g., 01,02); omit for all",
            show_default=False,
        ),
    ] = None,
    bids_filters: Annotated[
        str | None,
        typer.Option(
            "--data-filters",
            help="""
            Comma-separated BIDS entity filters; same-entity values OR'd, different
            entities AND'd (e.g., run-2,run-4,cond-G → (run=2 OR run=4) AND cond=G)
            """,
            show_default=False,
        ),
    ] = None,
    desc: Annotated[
        str,
        typer.Option(
            help="""
            Output desc entity tag (alphanumeric); output lands as desc-<desc>.
            A distinct value keeps separate nuisance-config variants from
            overwriting (e.g., --desc motionOnly).
            """,
        ),
    ] = "denoised",
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Overwrite existing outputs (default skips them)",
        ),
    ] = False,
):
    """Regress fmriprep confounds out of preprocessed BOLD, writing desc-denoised."""
    from hypline.denoise import DEFAULT_CONFOUND_COLUMNS, Denoiser
    from hypline.layout import BIDSLayout

    _columns = split_csv(columns, param_hint="--columns") or []
    _compcor = split_csv(compcor, param_hint="--compcor") or []
    _custom_sources = split_csv(custom_sources, param_hint="--custom-sources") or []
    _custom_columns = split_csv(custom_columns, param_hint="--custom-columns") or []
    if not _columns and not _compcor and not _custom_sources:
        # No channel given: fall back to the default confound set. An explicit
        # --compcor/--custom-sources means the user is picking their own model,
        # so leave the fallback out rather than composing the default onto it.
        _columns = list(DEFAULT_CONFOUND_COLUMNS)
    if bool(_custom_sources) != bool(_custom_columns):
        raise typer.BadParameter(
            "--custom-sources and --custom-columns must be given together",
            param_hint="--custom-sources/--custom-columns",
        )

    _sub_ids = split_csv(sub_ids, param_hint="--sub-ids")
    _bids_filters = split_csv(bids_filters, param_hint="--data-filters")

    _sub_ids = _sub_ids or BIDSLayout(bids_root).list.subjects(area="fmriprep")
    if not _sub_ids:
        logger.warning("No subjects found — nothing to denoise")
        return

    denoiser = Denoiser(
        bids_root=bids_root,
        space=space.value,
        columns=_columns,
        compcor=_compcor,
        custom_sources=_custom_sources,
        custom_columns=_custom_columns,
        bids_filters=_bids_filters,
        desc=desc,
        force=force,
    )

    run_per_id(
        bids_root, "denoise", id_key="sub", id_values=_sub_ids, task=denoiser.denoise
    )
