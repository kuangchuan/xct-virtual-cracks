"""Cheap dataset preflight before allocating any neural work on MPS."""

from pathlib import Path
import json
import tifffile

from . import Optimized_Crack_Engine as engine


def validate_training_dataset(directory, patch_size=256):
    """Check required arrays, native dimensions, modes and background-group split."""
    directory = Path(directory)
    for name in ("manifest.csv", "configuration.json", "COMPLETE.json"):
        if not (directory / name).is_file():
            raise ValueError(f"Dataset is missing {name}")
    configuration = json.loads((directory / "configuration.json").read_text())
    supersampling = configuration.get("generation", {}).get("raster_supersampling", 0)
    if not isinstance(supersampling, int) or supersampling < 1:
        raise ValueError("Configuration requires a positive integer generation.raster_supersampling")
    rows = engine.read_rows(directory / "manifest.csv")
    if not rows or len({r.get("id") for r in rows}) != len(rows):
        raise ValueError("Manifest must contain nonempty, unique sample identifiers")
    groups = {}
    for row in rows:
        split, group = row.get("split"), row.get("background_group")
        if split not in {"train", "validation", "test"} or not group:
            raise ValueError("Every sample requires a valid split and background_group")
        groups.setdefault(group, set()).add(split)
    if any(len(splits) > 1 for splits in groups.values()):
        raise ValueError("Background-group leakage across dataset splits")
    for split in ("train", "validation"):
        part = [r for r in rows if r["split"] == split]
        if not part or not {"core", "challenge", "control"} <= {r.get("mode") for r in part}:
            raise ValueError(f"The {split} split requires core, challenge and control samples")
        for row in part:
            shapes = set()
            for key in ("image", "coverage", "confidence", "mask", "ignore", "original"):
                if not row.get(key) or not (directory / row[key]).is_file():
                    raise ValueError(f"Sample {row['id']} is missing {key}")
                with tifffile.TiffFile(directory / row[key]) as file:
                    shape = tuple(file.series[0].shape)
                if len(shape) != 2 or min(shape) < patch_size:
                    raise ValueError(f"Sample {row['id']} requires native 2D arrays of at least {patch_size} pixels")
                shapes.add(shape)
            if len(shapes) != 1:
                raise ValueError(f"Sample {row['id']} has mismatched array dimensions")
    try:
        selected = engine.selected_validation_rows(directory)
    except (IndexError, StopIteration) as error:
        raise ValueError("Validation groups require the original three core-width variants plus challenge/control") from error
    if any(not r.get("width_band") for r in selected):
        raise ValueError("Validation samples require width_band labels")
    return {"samples": len(rows), "background_groups": len(groups),
            "train": sum(r["split"] == "train" for r in rows),
            "validation": sum(r["split"] == "validation" for r in rows),
            "native_patch_size": patch_size, "background_group_leakage": False}
