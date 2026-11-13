"""A self-contained geometry demonstration using the original V45 routines."""

from dataclasses import asdict, replace
from pathlib import Path
import json
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "xct-virtual-cracks-mpl"))

import numpy as np
from scipy import ndimage as ndi
import tifffile
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from . import Optimized_Crack_Engine as engine


def procedural_background(seed=271, size=256):
    """Make an illustrative texture; this is not an experimental XCT image."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    mask = ((yy - (size - 1) / 2) ** 2 +
            (xx - (size - 1) / 2) ** 2) < (size * 0.46) ** 2
    texture = ndi.gaussian_filter(rng.normal(size=(size, size)), 2)
    texture /= max(float(texture.std()), 1e-6)
    image = 0.62 + 0.055 * texture
    for _ in range(28):
        cy, cx = rng.uniform(size * 0.15, size * 0.85, 2)
        radius = rng.uniform(2, 12)
        image -= rng.uniform(0.05, 0.22) * np.exp(
            -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * radius ** 2))
    image += rng.normal(0, 0.009, image.shape)
    return np.where(mask, np.clip(image, 0, 1), 0).astype(np.float32), mask


def load_background(input_path=None, mask_path=None, seed=271):
    """Keep native pixels and use the same robust normalisation for both pairs."""
    if input_path is None:
        if mask_path is not None:
            raise ValueError("--mask requires --input")
        image, mask = procedural_background(seed)
        return image, mask, {"source": "procedural illustration, not experimental data"}
    raw = np.asarray(tifffile.imread(input_path), np.float32)
    if raw.ndim != 2 or min(raw.shape) < 256 or not np.isfinite(raw).all():
        raise ValueError("Input must be a finite 2D TIFF of at least 256 x 256 pixels")
    if mask_path is None:
        mask, support = engine.xio.specimen_envelope(raw, "TIFF")
    else:
        mask = np.asarray(tifffile.imread(mask_path))
        if mask.shape != raw.shape or not np.isfinite(mask).all():
            raise ValueError("Mask must match the input shape and contain finite values")
        mask = mask > 0
        support = {"method": "user-supplied support mask"}
    image, normalisation = engine.normalize_image(raw, mask)
    return image, mask, {
        "source": Path(input_path).name,
        "normalisation": normalisation,
        "specimen": support,
        "pixels_resized": False,
    }


def generate_pair(image, mask, seed=271, mode="core"):
    """Generate one pair; control means no new injection, not certified healthy."""
    if mode not in {"core", "challenge", "control"}:
        raise ValueError("Mode must be core, challenge or control")
    image, mask = np.asarray(image, np.float32), np.asarray(mask, bool)
    if image.ndim != 2 or image.shape != mask.shape or min(image.shape) < 256:
        raise ValueError("A matching 2D image/support of at least 256 pixels is required")
    if not np.isfinite(image).all() or mask.sum() < 64:
        raise ValueError("Image must be finite and support must contain at least 64 pixels")
    if image.min() < 0 or image.max() > 1:
        raise ValueError("Image must be normalised to [0, 1]")
    cfg = replace(engine.GenerationConfig(), seed=seed)
    if mode == "control":
        coverage = np.zeros_like(image)
        width = np.zeros_like(image)
        synthetic = image.copy()
        decrement = np.zeros_like(image)
        ignored = np.zeros_like(mask)
        paths, rendering = [], {"mode": "no new injection"}
    else:
        paths = engine.route_paths(image, mask, seed, mode, cfg)
        coverage, _, width = engine.rasterize_tiled(paths, image.shape, mask, cfg)
        core = ndi.binary_erosion(mask, iterations=6)
        if core.sum() < 64:
            core = mask
        dark = float(np.percentile(image[core], cfg.dark_percentile))
        synthetic, decrement, _, rendering = engine.render_crack(
            image, mask, coverage, seed, mode, cfg, dark)
        ignored, ambiguity = engine.supervision_ambiguity(
            image, mask, synthetic, decrement, coverage,
            engine.cached_native_field(image, mask))
        rendering["ambiguity_diagnostics"] = ambiguity
    return {
        "original": image,
        "synthetic": synthetic,
        "coverage": coverage,
        "width": width,
        "decrement": decrement,
        "validity": (~ignored).astype(np.uint8),
        "specimen": mask.astype(np.uint8),
        "metadata": {
            "seed": seed,
            "mode": mode,
            "shape_yx": list(image.shape),
            "generation": asdict(cfg),
            "rendering": rendering,
            "paths": [{"kind": p["kind"],
                       "points_yx": p["points"].tolist(),
                       "width_px": p["widths"].tolist()} for p in paths],
            "meaning": "Coverage describes injected geometry only. Original defects are not exhaustively labelled.",
        },
    }


def save_demo(result, output, source):
    """Write native arrays and a compact, reproducible preview."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for key in ("original", "synthetic", "coverage", "width", "decrement", "validity", "specimen"):
        tifffile.imwrite(output / f"{key}.tiff", result[key], compression="zlib")
    metadata = dict(result["metadata"], input=source,
                    coverage_encoding="float32 area fraction in [0,1]",
                    validity_encoding="uint8 0 or 1; exclude zero from the loss")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    fig = Figure(figsize=(12, 3.8))
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 3)
    for ax, key, label in zip(axes, ("original", "synthetic", "coverage"),
                              ("(a) Input texture", "(b) Injected damage", "(c) Area coverage")):
        ax.imshow(result[key], cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.set_xlabel(label)
    fig.subplots_adjust(left=0.025, right=0.975, top=0.98, bottom=0.11, wspace=0.06)
    fig.savefig(output / "preview.png", dpi=160)
    return output
