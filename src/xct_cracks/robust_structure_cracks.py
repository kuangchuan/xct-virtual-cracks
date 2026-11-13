"""Noise-compensated, locally normalized structural guidance for XCT augmentation.

This field measures image structure; it is neither mechanical weakness nor a
crack probability. The formula is a proposed combination of standard image
operations, not an implementation claimed to be established by another paper.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy import ndimage as ndi
from .image_guided_cracks import generate_geometry, rasterize, export_sample


@dataclass(frozen=True)
class Settings:
    scales_px: tuple = (1.0, 2.0, 4.0)
    noise_k: float = 3.0
    context_ratio: float = 4.0
    boundary_exclusion_px: float = 12.0
    boundary_taper_px: float = 12.0
    guidance_strength: float = 3.0
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
    # Explicit opt-in keeps earlier notebooks and dependent modules reproducible.
    rendering_mode: str = 'decrement'
    dark_percentile: float = 1.0
    dark_mixing_range: tuple = (0.85, 0.95)
    width_cap_px: float | None = None


def normalize_image(image, specimen):
    """Use symmetric robust percentiles; retain the same map for each pair."""
    raw, mask = np.asarray(image, np.float64), np.asarray(specimen, bool)
    if raw.ndim != 2 or raw.shape != mask.shape or mask.sum() < 64 or not np.isfinite(raw).all():
        raise ValueError('A finite 2D image and matching nonempty specimen mask are required.')
    low, high = np.percentile(raw[mask], (0.5, 99.5))
    norm = np.zeros_like(raw) if high <= low else np.clip((raw - low) / (high - low), 0, 1)
    return norm.astype(np.float32), dict(low=float(low), high=float(high), percentiles=[0.5, 99.5])


def structural_prior(image, specimen, settings=Settings()):
    """Compute interpretable evidence W in [0,1] and cost 1/(1+lambda W).

    Noise is estimated by a unit-energy diagonal high-pass filter and MAD.
    White Gaussian noise is a working approximation: correlated reconstruction
    noise and fine texture can bias this estimate. Diagnostics retain its value.
    """
    s = settings
    if min(s.scales_px) <= 0 or s.noise_k < 0 or s.context_ratio <= 0 or s.boundary_taper_px <= 0 or s.guidance_strength < 0:
        raise ValueError('Invalid scale, noise, boundary or guidance setting.')
    norm, normalization = normalize_image(image, specimen)
    mask = np.asarray(specimen, bool)
    distance = ndi.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    core = mask & (distance > s.boundary_exclusion_px + s.boundary_taper_px)
    if core.sum() < 64:
        core = ndi.binary_erosion(mask, iterations=1)
    if core.sum() < 16:
        raise ValueError('Insufficient specimen interior for noise estimation.')
    # HH has unit L2 norm, so white-noise variance is preserved by the filter.
    hh_kernel = np.array([[1.0, -1.0], [-1.0, 1.0]]) / 2
    highpass = ndi.convolve(norm.astype(float), hh_kernel, mode='reflect')
    residual = highpass[core]
    noise_std = float(np.median(np.abs(residual - np.median(residual))) / 0.6744897501960817)
    _, nearest = ndi.distance_transform_edt(~mask, return_indices=True)
    extended = norm[nearest[0], nearest[1]].astype(np.float64)
    epsilon = max(float(np.ptp(norm[mask])) ** 2 * 1e-12, np.finfo(float).tiny)
    responses, thresholds = [], []
    for sigma in s.scales_px:
        # Compute the exact norm of the discrete derivative kernel used below.
        radius = int(4 * sigma + 0.5)
        impulse = np.zeros((2 * radius + 1,) * 2)
        impulse[radius, radius] = 1
        derivative = ndi.gaussian_filter(impulse, sigma, order=(1, 0), mode='constant')
        threshold = s.noise_k * noise_std * np.linalg.norm(derivative)
        gy = ndi.gaussian_filter(extended, sigma, order=(1, 0), mode='reflect')
        gx = ndi.gaussian_filter(extended, sigma, order=(0, 1), mode='reflect')
        amplitude = sigma * np.maximum(np.hypot(gy, gx) - threshold, 0)
        smooth = ndi.gaussian_filter(extended, sigma, mode='reflect')
        context = s.context_ratio * sigma
        mean = ndi.gaussian_filter(smooth, context, mode='reflect')
        variance = np.maximum(ndi.gaussian_filter(smooth ** 2, context, mode='reflect') - mean ** 2, 0)
        response = amplitude ** 2 / (amplitude ** 2 + variance + (sigma * threshold) ** 2 + epsilon)
        responses.append(response.astype(np.float32))
        thresholds.append(float(threshold))
    gate = np.clip((distance - s.boundary_exclusion_px) / s.boundary_taper_px, 0, 1)
    prior = (np.mean(responses, axis=0) * gate * mask).astype(np.float32)
    cost = np.full(norm.shape, np.inf, np.float32)
    cost[mask] = 1 / (1 + s.guidance_strength * prior[mask])
    info = dict(noise_std=noise_std, gradient_noise_thresholds=thresholds,
                noise_model='white_Gaussian_approximation', noise_estimator='HH_MAD',
                noise_statistics_pixels=int(core.sum()))
    return dict(original=norm, specimen=mask, prior=prior, cost=cost,
                boundary_response=np.mean(responses, axis=0), scale_responses=responses,
                normalization=normalization, structure_info=info)


def render(original, specimen, coverage, seed, settings=Settings(), delta=None,
           noise_std=None, mixing=None):
    """Separate appearance from geometry; use the same rule for every material.

    Dark-phase mixing removes a fraction of the available local intensity above
    a specimen-derived dark reference. Unlike a fixed decrement, bright fibres
    do not retain their full texture contrast through a void. Existing darker
    pixels are not brightened. The dark reference is a heuristic, not measured air.
    """
    s = settings
    specimen = np.asarray(specimen, bool)
    if s.rendering_mode not in ('decrement', 'dark_phase'):
        raise ValueError('Unknown rendering mode.')
    if not 0 <= s.dark_percentile <= 100 or not 0 <= s.dark_mixing_range[0] <= s.dark_mixing_range[1] <= 1:
        raise ValueError('Invalid dark reference percentile or mixing range.')
    rng = np.random.default_rng(seed)
    alpha = ndi.gaussian_filter(coverage, s.psf_sigma_px, mode='constant') * specimen
    noise_std = float(rng.uniform(*s.noise_std_range)) if noise_std is None else noise_std
    if s.rendering_mode == 'dark_phase':
        mixing = float(rng.uniform(*s.dark_mixing_range)) if mixing is None else mixing
        distance = ndi.distance_transform_edt(np.pad(specimen, 1))[1:-1, 1:-1]
        core = specimen & (distance > s.boundary_exclusion_px)
        core = core if core.sum() >= 64 else specimen
        dark = float(np.percentile(original[core], s.dark_percentile))
        decrement = mixing * alpha * np.maximum(original - dark, 0)
        info = dict(mode='dark_phase', dark_reference=dark, mixing=mixing,
                    dark_percentile=s.dark_percentile)
    else:
        delta = float(rng.uniform(*s.attenuation_range)) if delta is None else delta
        decrement = delta * alpha
        info = dict(mode='decrement', delta=delta)
    clean = np.clip(original - decrement, 0, 1).astype(np.float32)
    noise = rng.normal(0, noise_std, alpha.shape).astype(np.float32) * np.sqrt(alpha)
    virtual = np.clip(clean + noise, 0, 1).astype(np.float32)
    return dict(virtual=virtual, alpha=alpha, clean_virtual=clean,
                clean_decrement=original - clean, rendering_info=info)


def synthesize(image, specimen, seed, settings=Settings()):
    if settings.width_cap_px is not None and settings.width_cap_px <= 0:
        raise ValueError('Width cap must be positive.')
    rng = np.random.default_rng(seed)
    field = structural_prior(image, specimen, settings)
    # Endpoint selection is independent of feature density. Weak/no structure
    # yields a uniform positive route cost, so injection remains possible.
    paths = generate_geometry(dict(field, prior=np.asarray(specimen, np.float32)), settings, rng)
    if settings.width_cap_px is not None:
        for path in paths:
            path['widths'] = np.minimum(path['widths'], settings.width_cap_px)
    coverage, target, width = rasterize(paths, field['original'].shape, specimen, settings)
    delta = float(rng.uniform(*settings.attenuation_range))
    noise_std = float(rng.uniform(*settings.noise_std_range))
    # Preserve the complete legacy random sequence when that mode is requested.
    if settings.rendering_mode == 'decrement':
        alpha = ndi.gaussian_filter(coverage, settings.psf_sigma_px, mode='constant') * specimen
        noise = rng.normal(0, noise_std, alpha.shape).astype(np.float32) * np.sqrt(alpha)
        virtual = np.clip(field['original'] - delta * alpha + noise, 0, 1).astype(np.float32)
        return dict(**field, paths=paths, coverage=coverage, target=target, width=width,
                    virtual=virtual, alpha=alpha, seed=int(seed), delta=delta, noise_std=noise_std)
    appearance = render(field['original'], specimen, coverage,
                        np.random.SeedSequence([int(seed), 812]), settings,
                        noise_std=noise_std)
    return dict(**field, paths=paths, coverage=coverage, target=target, width=width,
                **appearance, seed=int(seed), delta=delta, noise_std=noise_std)


def settings_dict(settings=Settings()):
    return asdict(settings)
