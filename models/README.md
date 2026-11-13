# V45 development checkpoints

The four checkpoint files in `v45/` are byte-identical to the frozen V45 models used for the original manuscript development figures. File hashes are recorded in `manifest.json`.

Use `scripts/infer_mps.py` with the matching material name. GFRP requires its internal three-channel local-contrast representation; the inference wrapper selects it automatically.

These files are not the separate Table 5 weights. Do not use the Table 5 scores as benchmark scores for these checkpoints.
