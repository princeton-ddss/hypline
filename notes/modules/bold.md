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
  Raw and derivative files may differ: fmriprep can drop dummy scans, which
  changes the volume count but not the sampling interval between remaining
  volumes. `get_n_trs` therefore requires a BOLD `bids` and reads its own
  header — resolving to raw would give the wrong answer for derivative
  callers.

## Header reads are header-only

Both `get_repetition_time` (header fallback) and `get_n_trs` use
`nibabel.load`, which is lazy: it reads the header and memory-maps the data.
Accessing `img.header.get_zooms()`, `img.header.get_data_shape()`,
`img.darrays[0].meta`, or `len(img.darrays)` never touches voxel data. Voxel
load happens only on `.get_fdata()` or `darray.data`.
