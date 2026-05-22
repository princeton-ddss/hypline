from loguru import logger

from hypline.bids import BIDSPath
from hypline.bold import BoldMeta
from hypline.events import segment_tr_slice


def collapse_desc_variants(feature_files: list[BIDSPath]) -> list[BIDSPath]:
    """Collapse `desc-*` variants sharing all non-desc entities."""
    groups: dict[tuple, list[BIDSPath]] = {}
    for feat_file in feature_files:
        key = tuple((k, v) for k, v in feat_file.entities.items() if k != "desc")
        groups.setdefault(key, []).append(feat_file)

    kept = []
    for variants in groups.values():
        kept.append(variants[0])
        if len(variants) > 1:
            dropped = ", ".join(v.path.name for v in variants[1:])
            logger.debug(
                "Collapsing desc-* variants — using {}, skipping {}",
                variants[0].path.name,
                dropped,
            )
    return kept


def segment_n_trs(feat_file: BIDSPath, bold_meta: BoldMeta) -> int:
    """Resolve TR count for the feature file's segment span.

    Feature files are one-per-segment; the segment entity on the filename
    names the segment within `bold_meta.segments`. Unsegmented runs use
    the full BOLD `n_trs`.
    """
    if not bold_meta.segments:
        return bold_meta.n_trs

    segment_entity = bold_meta.segments[0].entity
    segment_value = feat_file.entities.get(segment_entity)
    if segment_value is None:
        raise ValueError(
            f"Feature file {feat_file.path.name!r} is missing segment entity "
            f"{segment_entity!r} declared in events"
        )

    for seg in bold_meta.segments:
        if seg.value == segment_value:
            tr_slice = segment_tr_slice(seg, bold_meta.repetition_time)
            return tr_slice.stop - tr_slice.start

    valid = sorted(seg.value for seg in bold_meta.segments)
    raise ValueError(
        f"Segment value {segment_entity}-{segment_value} from "
        f"{feat_file.path.name!r} not found in events — valid values: {valid}"
    )
