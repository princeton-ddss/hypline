from typing import Literal

from hypline.encoding import EncodingConfig, EncodingTrainer

from ..conftest import BIDSTree

SUB = "001"
DYAD = "101"
TASK = "conv"
SPACE = "MNI152NLin6Asym"


def _make_encoding(
    tree: BIDSTree,
    features: list[str],
    *,
    tasks: list[str] | None = None,
    bold_space: str = SPACE,
    bold_desc: str = "denoised",
    bids_filters: list[str] | None = None,
    fold_by: str | None = None,
    n_folds: int | Literal["loo"] | None = None,
) -> EncodingTrainer:
    return EncodingTrainer(
        config=EncodingConfig(),
        bids_root=tree.root,
        features=features,
        tasks=tasks if tasks is not None else [TASK],
        bold_space=bold_space,
        bold_desc=bold_desc,
        bids_filters=bids_filters,
        fold_by=fold_by,
        n_folds=n_folds,
    )
