import multiprocessing as mp
from pathlib import Path

import typer
from rich import print
from typing_extensions import Annotated

from .regression import ConfoundRegression
from .utils import DillProcess

app = typer.Typer()


@app.command()
def clean(
    fmriprep_dir: Annotated[
        str,
        typer.Argument(
            help="Directory containing fMRIPrep outputs",
            show_default=False,
        ),
    ],
    config_file: Annotated[
        str,
        typer.Argument(
            help="Configuration file",
            show_default=False,
        ),
    ],
    model_name: Annotated[
        str,
        typer.Argument(
            help="Confound regression model to run (defined in config)",
            show_default=False,
        ),
    ],
    output_dir: Annotated[
        str | None,
        typer.Option(
            help="Directory to store cleaned BOLD data",
            show_default="<fmriprep_dir>_cleaned",
        ),
    ] = None,
    custom_confounds_dir: Annotated[
        str | None,
        typer.Option(help="Directory containing custom confounds"),
    ] = None,
    subject_ids_input: Annotated[
        str,
        typer.Option(
            "--subject-ids",
            help="Target subject IDs (comma-separated, with no spaces)",
            show_default="all",
        ),
    ] = "*",
    session_name: Annotated[
        str,
        typer.Option(
            help="Target session name",
            show_default="all",
        ),
    ] = "*",
    task_name: Annotated[
        str,
        typer.Option(
            help="Target task name",
            show_default="all",
        ),
    ] = "*",
    data_space_name: Annotated[
        str,
        typer.Option(help="Target BOLD data space"),
    ] = "MNI152NLin2009cAsym",
    n_processes: Annotated[
        int,
        typer.Option(help="Number of processes to run in parallel"),
    ] = 1,
):
    """
    Clean BOLD data in the given fMRIPrep outputs.
    """
    if subject_ids_input == "*":
        subject_ids = [
            path.name[4:] for path in Path(fmriprep_dir).glob("sub-*") if path.is_dir()
        ]
    else:
        subject_ids = subject_ids_input.split(",")

    def clean_bold(subject_ids: list[str]):
        reg = ConfoundRegression(
            config_file=config_file,
            fmriprep_dir=fmriprep_dir,
            output_dir=output_dir,
            custom_confounds_dir=custom_confounds_dir,
        )

        reg.clean_bold(
            model_name=model_name,
            subject_ids=subject_ids,
            session_name=session_name,
            task_name=task_name,
            data_space_name=data_space_name,
        )

    if n_processes < 2:
        clean_bold(subject_ids)
    else:
        if mp.current_process().name == "MainProcess":
            max_processes = mp.cpu_count()
            if n_processes > max_processes:
                n_processes = max_processes
                print(
                    "[bold yellow]Warning:[/bold yellow] "
                    "Requested processes exceed available CPU "
                    "cores. Resetting to match available cores."
                )

            processes: list[DillProcess] = []
            for i in range(n_processes):
                p = DillProcess(target=clean_bold, args=(subject_ids[i::n_processes],))
                processes.append(p)
                p.start()
            for p in processes:
                p.join()


@app.callback()
def callback():
    """
    An opinionated framework/toolbox for conducting
    data cleaning and analysis in hyperscanning studies
    involving dyadic conversations.
    """
