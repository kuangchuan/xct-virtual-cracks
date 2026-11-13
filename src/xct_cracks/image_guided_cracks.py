"""Image-informed crack augmentation for XCT segmentation, without phase labels.

The structural prior encodes image boundaries, not fracture strength. Geometry,
fractional pixel coverage, binary supervision, and rendered contrast are distinct
outputs. All material groups use the same synthesis parameters.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, graph, measure, morphology, draw
import tifffile


@dataclass(frozen=True)
class Settings:
    scales_px: tuple = (0.8, 1.6, 3.2)
    baseline: float = 0.5
    boundary_band_px: float = 5.0
    crop_margin_px: int = 6
    n_paths: tuple = (1, 2)
    length_fraction: tuple = (0.28, 0.65)
    branch_probability: float = 0.5
    width_range_px: tuple = (0.7, 4.0)
    width_variation: float = 0.2
    psf_sigma_px: float = 0.8
    attenuation_range: tuple = (0.15, 0.4)
    noise_std_range: tuple = (0.0, 0.006)
    supersampling: int = 4


def normalize_image(image, specimen):
    """Preserve local intensity ordering with a recorded robust linear mapping."""
    image = np.asarray(image, np.float32)
    specimen = np.asarray(specimen, bool)
    if image.ndim != 2 or image.shape != specimen.shape:
        raise ValueError('Image and specimen envelope must be matching 2D arrays.')
    if not np.isfinite(image).all() or specimen.sum() < 64:
        raise ValueError('A finite image and nonempty specimen envelope are required.')
    low, high = np.percentile(image[specimen], (1, 99.5))
    if high <= low:
        return np.zeros_like(image), dict(low=float(low), high=float(high), constant=True)
    norm = np.clip((image - low) / (high - low), 0, 1).astype(np.float32)
    return norm, dict(low=float(low), high=float(high))


def structural_prior(image, specimen, settings=Settings()):
    """One continuous, bounded structural prior for display and route cost."""
    norm, normalization = normalize_image(image, specimen)
    mask = np.asarray(specimen, bool)
    # Nearest-interior extension avoids inserting an artificial image step at M.
    _, nearest = ndi.distance_transform_edt(~mask, return_indices=True)
    extended = norm[nearest[0], nearest[1]]
    distance = ndi.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    boundary_weight = np.clip(distance / settings.boundary_band_px, 0, 1)
    core = mask & (distance >= settings.boundary_band_px)
    if core.sum() < 64:
        core = mask
    response = np.zeros_like(norm)
    scales = []
    for sigma in settings.scales_px:
        dy = ndi.gaussian_filter(extended, sigma, order=(1, 0), mode='nearest')
        dx = ndi.gaussian_filter(extended, sigma, order=(0, 1), mode='nearest')
        magnitude = (sigma * np.hypot(dy, dx)).astype(np.float32)
        scale = max(float(np.percentile(magnitude[core], 95)), 1e-6)
        response = np.maximum(response, np.clip(magnitude / scale, 0, 1))
        scales.append(scale)
    response *= boundary_weight * mask
    prior = np.where(mask, settings.baseline + (1 - settings.baseline) * response, 0).astype(np.float32)
    cost = np.full(norm.shape, np.inf, np.float32)
    cost[mask] = 1.0 / prior[mask]
    return dict(original=norm, specimen=mask, boundary_response=response, prior=prior,
                cost=cost, normalization=normalization, gradient_scales=scales)


def _sample_coordinate(weights, valid, rng):
    coords = np.column_stack(np.nonzero(valid))
    if not len(coords):
        return None
    probabilities = np.asarray(weights[valid], float)
    probabilities /= probabilities.sum()
    return coords[rng.choice(len(coords), p=probabilities)]


def _route(cost, domain, start, target):
    try:
        points, _ = graph.route_through_array(cost, tuple(start), tuple(target),
                                             fully_connected=True, geometric=True)
    except (ValueError, RuntimeError):
        return None
    points = np.asarray(points, np.float32)
    if len(points) < 8:
        return None
    smooth = ndi.gaussian_filter1d(points, 1.0, axis=0, mode='nearest')
    ij = np.rint(smooth).astype(int)
    # Smoothing is a cosmetic geometric operation, never permission to leave M.
    return smooth if domain[ij[:, 0], ij[:, 1]].all() else points


def _widths(n, settings, rng, branch=False):
    low, high = settings.width_range_px
    nominal = np.exp(rng.uniform(np.log(low), np.log(high)))
    nominal *= 0.65 if branch else 1.0
    fluctuation = ndi.gaussian_filter1d(rng.normal(size=n), max(1, n / 12))
    if fluctuation.std() > 0:
        fluctuation /= fluctuation.std()
    width = nominal * np.clip(1 + settings.width_variation * fluctuation, 0.5, 1.5)
    edge = max(2, int(0.12 * n))
    taper = np.ones(n)
    taper[:edge] = np.linspace(0.35, 1, edge)
    taper[-edge:] = np.linspace(1, 0.35, edge)
    return np.maximum(0.25, width * taper).astype(np.float32)


def generate_geometry(field, settings, rng):
    mask, prior = field['specimen'], field['prior']
    domain = mask.copy()
    m = settings.crop_margin_px
    domain[:m] = False; domain[-m:] = False
    domain[:, :m] = False; domain[:, -m:] = False
    components = measure.label(domain)
    occupied = np.zeros(mask.shape, bool)
    paths = []
    yy, xx = np.indices(mask.shape)
    diagonal = np.hypot(*mask.shape)
    for path_index in range(int(rng.integers(settings.n_paths[0], settings.n_paths[1] + 1))):
        for attempt in range(16):
            start = _sample_coordinate(prior, domain & ~occupied, rng)
            if start is None:
                break
            distance = np.hypot(yy - start[0], xx - start[1])
            candidates = (components == components[tuple(start)]) & domain
            candidates &= (distance >= settings.length_fraction[0] * diagonal)
            candidates &= (distance <= settings.length_fraction[1] * diagonal)
            if candidates.sum() < 16:
                continue
            target = _sample_coordinate(prior, candidates, rng)
            cost = field['cost'].copy()
            cost[~domain] = np.inf
            if occupied.any():
                cost += 4 * np.exp(-ndi.distance_transform_edt(~occupied) / 4)
            points = _route(cost, domain, start, target)
            if points is None:
                continue
            direct = np.linalg.norm(target - start)
            length = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
            if length > 1.9 * direct:
                continue
            paths.append(dict(points=points, widths=_widths(len(points), settings, rng), kind='main'))
            ij = np.rint(points).astype(int); occupied[ij[:, 0], ij[:, 1]] = True
            if rng.random() < settings.branch_probability:
                start_b = np.rint(points[len(points) // 2]).astype(int)
                dist_b = np.hypot(yy - start_b[0], xx - start_b[1])
                band = domain & (components == components[tuple(start_b)])
                band &= (dist_b > 0.08 * diagonal) & (dist_b < 0.22 * diagonal)
                band &= ndi.distance_transform_edt(~occupied) > 6
                target_b = _sample_coordinate(prior, band, rng)
                if target_b is not None:
                    branch = _route(cost, domain, start_b, target_b)
                    if branch is not None:
                        paths.append(dict(points=branch, widths=_widths(len(branch), settings, rng, True), kind='branch'))
            break
    return paths


def rasterize(paths, shape, specimen, settings):
    """Derive coverage, touched-pixel labels and nominal width from one geometry."""
    s = settings.supersampling
    high_shape = (shape[0] * s, shape[1] * s)
    center = np.zeros(high_shape, bool)
    widths = np.zeros(high_shape, np.float32)
    for path in paths:
        points = path['points'] * s + (s - 1) / 2
        for i in range(len(points) - 1):
            rr, cc = draw.line(*np.rint(points[i]).astype(int), *np.rint(points[i + 1]).astype(int))
            width = float((path['widths'][i] + path['widths'][i + 1]) / 2)
            center[rr, cc] = True
            widths[rr, cc] = np.maximum(widths[rr, cc], width)
    if not center.any():
        zeros = np.zeros(shape, np.float32)
        return zeros, zeros.astype(bool), zeros
    distance, indices = ndi.distance_transform_edt(~center, return_indices=True)
    local_width = widths[indices[0], indices[1]]
    support = distance <= local_width * s / 2
    coverage = support.reshape(shape[0], s, shape[1], s).mean(axis=(1, 3)).astype(np.float32)
    coverage *= specimen
    target = coverage > 0  # Explicit touched-pixel convention, including subpixel cracks.
    mid = s // 2
    nominal_width = local_width[mid::s, mid::s] * target
    return coverage, target, nominal_width.astype(np.float32)


def synthesize(image, specimen, seed, settings=Settings()):
    """Synthesize one pair; no hidden retries, manual phase labels or crack labels."""
    rng = np.random.default_rng(seed)
    field = structural_prior(image, specimen, settings)
    paths = generate_geometry(field, settings, rng)
    coverage, target, width = rasterize(paths, field['original'].shape, specimen, settings)
    alpha = ndi.gaussian_filter(coverage, settings.psf_sigma_px, mode='constant') * specimen
    delta = float(rng.uniform(*settings.attenuation_range))
    noise_std = float(rng.uniform(*settings.noise_std_range))
    noise = rng.normal(0, noise_std, alpha.shape).astype(np.float32) * np.sqrt(alpha)
    virtual = np.clip(field['original'] - delta * alpha + noise, 0, 1).astype(np.float32)
    return dict(**field, paths=paths, coverage=coverage, target=target, width=width,
                virtual=virtual, alpha=alpha, seed=int(seed), delta=delta, noise_std=noise_std)


def export_sample(sample, folder, stem):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    arrays = {key: sample[key] for key in ['original', 'virtual', 'prior', 'boundary_response',
                                         'coverage', 'width', 'specimen', 'target']}
    for key, array in arrays.items():
        tifffile.imwrite(folder / f'{stem}_{key}.tiff', array.astype(np.uint8) if array.dtype == bool else array)
    geometry = [dict(kind=p['kind'], points_yx=p['points'].tolist(), nominal_width_px=p['widths'].tolist())
                for p in sample['paths']]
    (folder / f'{stem}_geometry.json').write_text(json.dumps(geometry))


def settings_dict(settings=Settings()):
    return asdict(settings)
