#!/usr/bin/env python3
"""Generate a native-pixel illustrative pair without any neural-network work."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xct_cracks.demo import generate_pair, load_background, save_demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Optional 2D TIFF; no resizing is performed")
    parser.add_argument("--mask", type=Path, help="Optional matching specimen-support TIFF")
    parser.add_argument("--output", type=Path, default=Path("results/demo"), help="New output directory")
    parser.add_argument("--seed", type=int, default=271)
    parser.add_argument("--mode", choices=("core", "challenge", "control"), default="core")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new directory to retain previous results")
    image, mask, source = load_background(args.input, args.mask, args.seed)
    result = generate_pair(image, mask, args.seed, args.mode)
    output = save_demo(result, args.output, source)
    print(f"Saved native {image.shape[1]} x {image.shape[0]} demonstration to {output}")
    print("No model was trained or run. The default texture is procedural illustration.")


if __name__ == "__main__":
    main()
