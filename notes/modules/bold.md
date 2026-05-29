# BOLD metadata extraction

Domain semantics behind `hypline.bold` utilities that extract per-run
metadata without loading voxel data.

## TR vs n_trs: acquisition-level vs file-level

- **TR (`repetition_time`)** is an *acquisition* property — the physical
  sampling interval set on the scanner. It is identical across all
  representations of a run (raw NIfTI, fmriprep volume output, surface
  GIfTI). The raw `*_bold.json` sidecar is authoritative. `get_repetition_time`
  accepts any `bids` carrying the run's identity entities and resolves to the
  raw tree via `layout.path.raw`.
- **n_trs** is a *file* property — the volume count of the specific image.
  `get_n_trs` requires a BOLD `bids` and reads its own header. fmriprep *can*
  drop dummy scans (`--dummy-scans` / auto non-steady-state trimming) and
  shorten the derivative count, but hypline rejects such runs — see the
  invariant below.

## Enforced invariant: derivative n_trs must equal raw

`get_bold_meta` raises if a derivative BOLD's `n_trs` differs from the raw
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

Both `get_repetition_time` (header fallback) and `get_n_trs` use
`nibabel.load`, which is lazy: it reads the header and memory-maps the data.
Accessing `img.header.get_zooms()`, `img.header.get_data_shape()`,
`img.darrays[0].meta`, or `len(img.darrays)` never touches voxel data. Voxel
load happens only on `.get_fdata()` or `darray.data`.
