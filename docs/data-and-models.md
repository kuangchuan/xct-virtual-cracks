# Data and model availability

## Included

- Four V45 synthetic-only development checkpoints in `models/v45/`, with SHA-256 values in `models/manifest.json`.
- Full-five mIoU summary statistics and all 100 individual run scores from the frozen vn46 Table 5 study.
- Fitting/validation identities and a compact experiment protocol.
- Manuscript visual examples and a newly plotted full-five comparison.
- Original source snapshots and a portable interface for new demonstrations.

## External experimental data

- **GFRP:** [Ultra-fast time-lapse synchrotron radiation CT imaging of compressive failure in unidirectional glass fibre-epoxy composite, Zenodo](https://doi.org/10.5281/zenodo.13348028).
- **SiC composite:** [Data from: Detecting damage in composites using volume decomposition analysis of tomographic data, Dryad](https://doi.org/10.5061/dryad.9cnp5hqsm).
- **Mortar and Recycled-aggregate mortar:** data-access arrangements will accompany the manuscript. The full volumes are not included in this repository.

The binary human-reference masks, full synthetic training arrays and historical native caches are not bundled in this snapshot. Access to the displayed results does not imply that all inputs needed to reproduce the complete 100-fit experiment are included.

## Checkpoint correspondence

| Checkpoint family | Qualitative figures | Quantitative comparison |
|---|---|---|
| Supplied V45 models | Original Figures 7–8 and associated supplementary examples | Original development results in Table 4 |
| Separate updated-generator models | No claim that the original figures use these weights | vn46 Table 5, five schemes × four materials × five seeds |

The portable MPS inference script loads the supplied V45 checkpoints with the corresponding input representation. New training produces a separate run and does not overwrite those checkpoints.

Experimental images retain their original measurement content. Figure assets are presentation exports, rather than substitutes for the source volumes or training arrays. Original data providers retain their applicable data terms. This private snapshot has no open-source or data redistribution licence; see `NOTICE`.
