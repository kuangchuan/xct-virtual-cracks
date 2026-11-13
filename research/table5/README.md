# Frozen Table 5 study

This directory documents the completed updated-generator experiment used in manuscript revision vn46. It is distinct from the original V45 development-model figures.

## Design

- Four materials; 20 annotated real images per material.
- Fixed split: 14 fitting images and six common validation images.
- Five schemes: real-only 14, virtual-only, and virtual plus 14/7/4 real fitting images.
- Seeds: 271, 811, 1459, 2027 and 4099.
- Five independent virtual initialisations per material; each adaptation starts from its own seed's virtual checkpoint.
- Native 256 × 256 windows; batch size 4; Apple MPS.
- 80 new fits plus 20 unchanged matched real-only fits.
- Full-image validation with 256-pixel windows, stride 128, Hann aggregation and threshold 0.5.
- All five scores retained; arithmetic mean and sample SD throughout the repository.

The same six real images select checkpoints. The result is validation performance, rather than an independent held-out test. Fitting budgets count images used for gradient updates and do not include the six selection images.

## Files

- `protocol.json`: curated protocol without workstation paths or superseded reporting variants.
- `fixed_split.csv`: exact fixed image identities and nested 4-/7-image membership.
- `source_manifest.json`: original and packaged hashes of the source snapshots.
- `code_snapshot/`: historical implementation, retained for audit.
- `../../results/validation_miou_all_five.csv`: the 20 reported cells.
- `../../results/seedwise_miou.csv`: all 100 individual run scores.

## Reproduction boundary

The source snapshots preserve historical imports, dated cache layouts and dependency-hash checks. Ten files are byte-identical to their originals; one has only its status-page text translated into English, as recorded in the manifest. They are provenance records, not installed modules or standalone commands. Exact replay additionally requires the original prepared source manifests, real-reference masks, native arrays, generation records, validation windows and the 20 reused real-only results.

For new data, use the portable package and MPS scripts at the repository root. The single-pair generation demo is a software example and is not a training dataset or a replay of this study.

## Interpretation

The pure-virtual models already approach the real-only reference most closely in the two mortar materials. Few-image fine-tuning improves the mean over virtual-only training in all four materials. Mortar with four fitting images is numerically close to and slightly above the real-only 14-image reference; GFRP also narrows its gap substantially. SiC remains less responsive to the present adaptation recipe.

These results describe material-dependent adaptation potential. They do not establish statistical equivalence, universal superiority, or a proven percentage reduction in annotation needs.
