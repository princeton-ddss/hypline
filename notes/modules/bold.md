# BOLD metadata extraction

Domain semantics behind `hypline.bold` utilities that extract per-run
metadata without loading voxel data.

## TR vs n_trs: acquisition-level vs file-level

- **TR (`repetition_time`)** is an *acquisition* property — the physical
  sampling interval set on the scanner. It is identical across all
  representations of a run (raw NIfTI, fmriprep volume output, surface
  GIfTI). `get_repetition_time` accepts any `bids` carrying the run's identity
  entities. It does **not** require raw imaging files: derivative-only datasets
  (fmriprep output without the bulky raw BOLD tree copied alongside) are the
  common case, so resolution is tiered by trust, raw images last:

  1. raw `*_bold.json` sidecar — exact, BIDS-declared, tiny, often retained
     even when raw images are not.
  2. any fmriprep BOLD `.nii.gz` for the run (any `space`/`desc`/... variant —
     TR is acquisition-level, so all agree). Restricted to `.nii.gz`: surface
     `.func.gii` carries TR unreliably and is never consulted for it.
  3. raw BOLD image header — last resort; the file most often absent.

  Within tiers 1–2 the declared sidecar value is preferred over the image
  header (tier 3 is header-only), because the NIfTI header zoom is a float32
  prone to drift (e.g. a true `1.5` can read back imprecisely) while the JSON
  value is exact.
- **n_trs** is a *file* property — the volume count of the specific image.
  `get_n_trs` requires a BOLD `bids` and reads its own header. fmriprep *can*
  drop dummy scans (`--dummy-scans` / auto non-steady-state trimming) and
  shorten the derivative count, but hypline rejects such runs — see the
  invariant below.

Callers that start from a non-BOLD `bids` (e.g. confound/feature generators
holding a feature file) and need a present BOLD image for `n_trs` resolve it
with `resolve_bold_image`, not a path constructed via `layout.path.raw`: a
constructed raw path may not exist on disk — derivative-only trees are the
common case. n_trs is preserved across variants, so whichever image it returns
answers. (TR needs no such resolver — `get_repetition_time` prefers a tiny
sidecar, touching a BOLD image only as a fallback.)

## Enforced invariant: derivative n_trs must equal raw

`load_bold_meta` raises if a derivative BOLD's `n_trs` differs from the raw
image. events.tsv onsets are raw-relative; hypline does not shift them, so a
trimmed derivative would misalign and is refused outright rather than
silently re-indexed.

Because the counts are guaranteed equal, callers producing TR-aligned
artifacts (e.g. confound files) may anchor `n_trs` on either the raw or the
preprocessed BOLD — both are correct, and a mismatch can only surface as the
invariant raising. `n_trs` is also space-invariant across fmriprep outputs of
the same run, so any `space-*` variant works.

For the downstream events-timeline reasoning behind the raw-relative onset
rule, see [../decisions/feature-files.md](../decisions/feature-files.md).

## Header reads are header-only

Both `get_repetition_time` (header reads) and `get_n_trs` use
`nibabel.load`, which is lazy: it reads the header and memory-maps the data.
Accessing `img.header.get_zooms()`, `img.header.get_data_shape()`, or
`len(img.darrays)` never touches voxel data. Voxel load happens only on
`.get_fdata()` or `darray.data`.
