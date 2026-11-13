#!/usr/bin/env python3
"""Train on an engine-format dataset using Apple MPS only.

This is a portable example of the original synthetic training engine. It is not
the exact Table 5 schedule, real-image adaptation or five-seed study launcher.
"""

import argparse
from dataclasses import replace
from pathlib import Path
import os
import sys

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xct_cracks.mps import require_mps
from xct_cracks import Optimized_Crack_Engine as engine
from xct_cracks.GFRP_Context_Study import ContrastUNet
from xct_cracks.dataset import validate_training_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Prepared engine-format dataset directory")
    parser.add_argument("--material", choices=engine.MATERIALS, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=271)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--min-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; use a new directory")
    if (args.batch_size < 1 or args.max_epochs < 1 or args.min_epochs < 1 or
            args.min_epochs > args.max_epochs or args.patience < 1 or args.learning_rate <= 0):
        parser.error("Batch size, epochs, patience and learning rate must be positive; min epochs <= max epochs")
    try:
        preflight = validate_training_dataset(args.dataset)
    except (ValueError, KeyError, OSError) as error:
        parser.error(f"{error}; the single-pair demonstration is not a training dataset")
    runtime = require_mps()
    cfg = replace(engine.TrainingConfig(), device="mps", num_threads=1,
                  patch_size=256, batch_size=args.batch_size, seed=args.seed,
                  max_epochs=args.max_epochs, min_epochs=args.min_epochs,
                  patience=args.patience, learning_rate=args.learning_rate,
                  export_real_count=0, evaluate_test=False)
    factory = ContrastUNet if args.material == "gfrp" else engine.make_model
    engine.train_material(args.material, args.dataset, args.output, cfg, model_factory=factory)
    engine.write_json(args.output / "mps_runtime.json", runtime)
    engine.write_json(args.output / "dataset_preflight.json", preflight)


if __name__ == "__main__":
    main()
