#!/usr/bin/env python3
"""Run a trusted local checkpoint on a native 2D TIFF using Apple MPS only."""

import argparse
from dataclasses import fields, replace
from pathlib import Path
import os
import sys

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import tifffile
import torch
from xct_cracks.mps import require_mps
from xct_cracks import Optimized_Crack_Engine as engine
from xct_cracks.GFRP_Context_Study import ContrastUNet
from xct_cracks.demo import load_background


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mask", type=Path, help="Optional specimen-support TIFF; not a crack reference")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--material", choices=engine.MATERIALS, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--normalised", action="store_true", help="Input is already normalised to [0,1]; preserve its values")
    args = parser.parse_args()
    if args.output.exists() or args.batch_size < 1:
        parser.error("Use a new output directory and a positive batch size")
    if args.normalised:
        if args.mask is not None:
            parser.error("--mask is used only when normalising a raw TIFF")
        image = np.asarray(tifffile.imread(args.input), np.float32)
        if (image.ndim != 2 or min(image.shape) < 256 or not np.isfinite(image).all()
                or image.min() < 0 or image.max() > 1):
            parser.error("Normalised input must be finite 2D, at least 256 pixels per axis, and within [0,1]")
        source = {"source": args.input.name, "normalisation": "user-supplied [0,1] values"}
    else:
        image, _, source = load_background(args.input, args.mask)
    runtime = require_mps()
    checkpoint = torch.load(args.checkpoint, map_location="mps", weights_only=True)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        parser.error("Checkpoint must contain a state_dict")
    if checkpoint.get("material", args.material) != args.material:
        parser.error("Checkpoint material does not match --material")
    known = {f.name for f in fields(engine.TrainingConfig)}
    configuration = {k: v for k, v in checkpoint.get("config", {}).items() if k in known}
    cfg = engine.TrainingConfig(**configuration)
    if cfg.patch_size != 256:
        parser.error("This release example accepts native-256 checkpoints only")
    cfg = replace(cfg, device="mps", batch_size=args.batch_size, num_threads=1,
                  export_real_count=0, evaluate_test=False)
    model = (ContrastUNet(cfg) if args.material == "gfrp" else engine.make_model(cfg)).to("mps")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    probability = engine.predict_native(model, image, "mps", cfg)
    args.output.mkdir(parents=True, exist_ok=False)
    tifffile.imwrite(args.output / "probability.tiff", probability.astype(np.float32), compression="zlib")
    tifffile.imwrite(args.output / "prediction.tiff", (probability >= 0.5).astype(np.uint8), compression="zlib")
    engine.write_json(args.output / "inference.json", {
        "runtime": runtime, "input": source, "checkpoint": args.checkpoint.name,
        "checkpoint_sha256": engine.sha(args.checkpoint), "material": args.material,
        "native_window": 256, "stride": 128, "threshold": 0.5,
        "pixels_resized": False, "postprocessing": None,
    })
    print(f"Saved native-image probability and prediction to {args.output}")


if __name__ == "__main__":
    main()
