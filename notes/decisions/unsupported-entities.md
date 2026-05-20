# Unsupported BIDS entities

Hypline rejects a subset of BIDS entities at `BIDSPath` construction. Any path
carrying one of these entities raises `ValueError` on construction.

## Rejected entities

`acq`, `ce`, `rec`, `dir`, `echo`, `part`, `chunk`.

Defined as `UNSUPPORTED_ENTITIES` in [`hypline.bids`](../../src/hypline/bids.py).

## Rationale

These entities mark **methodological variation** within a project — different
acquisition parameters, contrast agents, reconstruction pipelines,
phase-encoding directions, echoes, magnitude/phase splits, or stitched-run
chunks. Hyperscanning, hypline's target domain, fixes a single acquisition
protocol across all runs and participants by construction: comparing brain
activity across simultaneously-scanned participants only makes sense when the
acquisition is held constant. Methodological variants have no place in a
valid hypline project.

Disallowing these entities at the path-construction boundary lets downstream
code assume a single coherent acquisition without per-call invariance checks.

## Enforcement

Single chokepoint: `BIDSPath.__init__`. Every path hypline handles is wrapped
in `BIDSPath`, so the rejection catches raw, derivative, feature, and
confound paths uniformly.

## Revisit conditions

Lift the ban (entirely or per-entity) if hypline grows beyond the
hyperscanning domain — e.g. a methods-comparison study comparing acquisition
variants. Re-adding an entity requires restoring per-call invariance
validation in `encoding._discover_bold` so a single training call still sees
one coherent acquisition.
