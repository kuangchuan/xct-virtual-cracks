# Method overview

## From a measured slice to paired training data

The method starts from a reconstructed XCT slice and its external specimen support. An image-derived structural field attracts routes towards local features without treating grey value as a calibrated mechanical property. Variable apertures and optional branches turn routes into a geometric support. Supersampling estimates how much of each native pixel is occupied by the injected damage.

A simple image-domain rendering model changes appearance while preserving the acquired background:

```text
I_virtual = clip(I_measured - PSF * [coverage × mixing × max(I_local - I_dark, 0)], 0, 1)
```

Here `*` denotes convolution. The point-spread function acts on the injected decrement. The existing reconstructed image is not blurred again. The dark reference and mixing factor are image-domain heuristics, rather than a calibrated physical crack-opening model.

The coverage target is defined before grey rendering. Optical spread can therefore extend beyond geometric support without expanding the target. Ambiguous overlap with existing source defects is handled through the original validity convention. A no-new-injection control has no added synthetic damage; it does not certify that the measured slice contains no native defects.

## Learning

Each material uses a compact MONAI BasicUNet with five encoder levels and features `(8, 8, 16, 32, 64, 8)`. The loss combines soft cross-entropy and soft Dice, with validity masking. GFRP adds local contrast channels at 9- and 31-pixel neighbourhoods. Native 256-pixel windows retain the acquired spatial sampling.

Inference aggregates overlapping windows with a nonzero Hann weight and a 128-pixel stride. A probability threshold of 0.5 yields the binary output. The reported real-image validation uses complete images without reference-dependent clipping or morphological postprocessing.

## Original models and adaptation models

The supplied V45 checkpoints generated the original synthetic and real qualitative examples in the README. Their training and development selection used synthetic data.

The separate Table 5 study uses five independently initialised virtual models per material, each followed by its own seed-matched adaptation runs on 14, 7 or 4 real fitting images. Twenty matched real-only MPS fits were reused unchanged and 80 generator-related fits were added. The same six real images per material select all study checkpoints.

The fixed updated virtual sampler uses 35% original no-new-injection controls. Within positive examples, the mixture is 80% original V45, 10% original geometry with extended grey rendering, and 10% extended geometry with extended grey rendering. Fitting-label-selected real backgrounds are excluded from this pretraining sampler. This recipe differs from separate annotation-assisted exploratory studies, which are not presented here as the Table 5 experiment.

Fine-tuning uses up to 260 epochs, compared with 120 in the preceding comparison. Improvements between those tables cannot be attributed to geometry changes alone.

## Metric

The displayed metric is mIoU, computed from confusion counts pooled over the six complete images and averaged over the two segmentation classes. Each run produces one value per material and training scheme. Tables and plots report the arithmetic mean and sample standard deviation (`ddof=1`) over all five runs. Individual scores remain available for audit.

Validation selection, same-acquisition image correlation and missing matched real-only 4-/7-image controls limit claims about independent generalisation and annotation efficiency. The strongest supported statement is that controlled virtual training provides an initialisation that can be adapted with few real fitting images, with outcomes that depend on material.
