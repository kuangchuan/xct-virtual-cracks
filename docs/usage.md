# Installation and use

## Environment

The research environment uses Python 3.10. The pinned dependencies support the package's declared Python range of 3.10–3.12; use Python 3.10 for the closest environment match.

```bash
conda create -n xct-cracks python=3.10
conda activate xct-cracks
python -m pip install -e .
```

Geometry generation runs without a neural accelerator. Training and neural inference require native macOS on Apple Silicon, an MPS-enabled PyTorch build, and access to Metal from the executing process. If a restricted execution context cannot access Metal, run the command in a normal macOS Terminal. Neural commands stop instead of falling back to CPU.

## Generate one paired example

```bash
python scripts/generate_demo.py --output runs/demo --seed 271
```

The default background is a reproducible procedural illustration at 256 × 256 pixels, rather than a measured XCT image. Output directories must be new, which prevents accidental replacement of earlier runs.

For a native two-dimensional XCT TIFF:

```bash
python scripts/generate_demo.py \
  --input path/to/slice.tiff \
  --mask path/to/specimen-mask.tiff \
  --mode core \
  --output runs/slice-demo \
  --seed 271
```

The specimen mask is an external support mask, not a crack reference. Nonzero mask pixels identify specimen support. The image must be finite, two-dimensional, and at least 256 pixels along each axis. With no mask, the original texture-based envelope estimator is used. Measured intensities are robustly normalised without geometric resizing.

Modes:

| Mode | Meaning |
|---|---|
| `core` | Injected geometry from the resolved-width bands |
| `challenge` | Narrower and weaker-contrast injected damage |
| `control` | No new injection; existing source defects may remain |

Outputs:

| File | Meaning |
|---|---|
| `original.tiff` | Normalised input at native resolution |
| `synthetic.tiff` | Input with the rendered perturbation |
| `coverage.tiff` | Float32 subpixel area fraction, in [0,1] |
| `width.tiff` | Local injected width in native pixels |
| `decrement.tiff` | The rendered intensity decrement |
| `validity.tiff` | Binary validity; zero is excluded from the loss |
| `specimen.tiff` | External support mask |
| `metadata.json` | Seed, configuration, paths and rendering records |
| `preview.png` | Input, synthetic image and area coverage |

A single generated pair is not a training dataset. Its floating coverage and binary validity use a simpler output encoding than the historical prepared-dataset format below.

## Infer with a V45 checkpoint

```bash
python scripts/infer_mps.py \
  --input path/to/slice.tiff \
  --checkpoint models/v45/mortar.pt \
  --material mortar \
  --output runs/mortar-inference \
  --batch-size 8
```

Material identifiers are `mortar`, `recycled_mortar`, `gfrp` and `sic`. The checkpoint must match the material. GFRP uses the supplied local-contrast channel representation internally. Checkpoint loading uses PyTorch's restricted `weights_only=True` loader.

For a raw image, an optional `--mask` controls the normalisation support. Use `--normalised` only if the input already contains the intended finite [0,1] values; this preserves those values and cannot be combined with `--mask`.

Inference uses native 256-pixel windows, a 128-pixel stride and Hann aggregation. It writes a float32 `probability.tiff`, a binary `prediction.tiff`, and an `inference.json` record with the checkpoint hash and processing choices. No resizing or morphological postprocessing is applied.

The original README figures use the supplied V45 checkpoints. The separate Table 5 scores must not be assigned to these weights.

## Train on a prepared dataset

```bash
python scripts/train_mps.py \
  --dataset path/to/prepared-dataset \
  --material mortar \
  --output runs/mortar-training \
  --batch-size 8 \
  --seed 271
```

The command uses the original synthetic training engine, including fractional-coverage supervision and grouped development validation, with explicit MPS execution. It does not implement the complete Table 5 real-image adaptation workflow or its plateau schedule. Those historical study sources are retained separately.

The dataset directory needs:

- `manifest.csv`, containing the original sample records and relative array paths.
- `configuration.json`, including `generation.raster_supersampling`.
- `COMPLETE.json`, the prepared dataset's completion record.
- Per-sample image, mask, coverage and confidence TIFFs, plus any arrays required by dataset verification.

The manifest fields read by sampling are `id`, `background_group`, `split`, `mode`, `image`, `mask`, `coverage` and `confidence`. Use separate `train` and `validation` groups, with all variants of a source background kept together. The training sampler expects `core`, `challenge` and `control` examples.

In the original prepared format, images are normalised to [0,1], coverage is an integer count out of `raster_supersampling ** 2`, the mask identifies nonzero geometric support, and confidence encodes binary validity as 0 or 255. The default supersampling factor is 4. Convert demo outputs to the declared dataset format explicitly; do not rename a single demo directory as a prepared dataset.

The default training example uses batch size 8 and up to 120 epochs. Options include `--max-epochs`, `--min-epochs`, `--patience` and `--learning-rate`. Table 5 used batch size 4 and its separately documented stage-specific schedule. Choose concurrency from measured memory and throughput on the target Mac; launching more workers does not guarantee faster aggregate training.

## Checks

```bash
python -m unittest discover -s tests -v
```

The included checks cover geometry, coverage, reproducibility and explicit backend rejection without starting neural training. An MPS allocation/forward/backward/optimiser probe runs when a neural command is invoked.
