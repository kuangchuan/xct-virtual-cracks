"""Acquisition-aware input handling; separate from crack synthesis."""
from pathlib import Path
import struct
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology
import tifffile
import olefile


def read_plane(record):
    path = Path(record['source'])
    if path.suffix.lower() != '.txm':
        image = tifffile.imread(path)
        return image, dict(kind='TIFF', dtype=str(image.dtype))
    with olefile.OleFileIO(path) as ole:
        header = {key: struct.unpack('<I', ole.openstream('ImageInfo/' + key).read(4))[0]
                  for key in ['ImageWidth', 'ImageHeight', 'NoOfImages', 'DataType']}
        if header['DataType'] != 5:
            raise ValueError(f'Unsupported TXM datatype: {header["DataType"]}')
        number = int(record['slice_index_zero_based']) + 1
        stream = f'ImageData{(number - 1) // 100 + 1}/Image{number}'
        raw = ole.openstream(stream).read()
        shape = (header['ImageHeight'], header['ImageWidth'])
        if len(raw) != int(np.prod(shape)) * 2:
            raise ValueError('TXM byte count does not match the image dimensions.')
        image = np.frombuffer(raw, dtype='<u2').reshape(shape).copy()
    return image, dict(kind='TXM', stream=stream, **header)


def _largest_filled(mask):
    labels = measure.label(ndi.binary_fill_holes(mask))
    regions = measure.regionprops(labels)
    if not regions:
        raise ValueError('Automatic specimen-envelope estimation produced an empty mask.')
    return ndi.binary_fill_holes(labels == max(regions, key=lambda region: region.area).label)


def specimen_envelope(image, acquisition_kind):
    """Estimate external support from a complete slice, without phase labels.

    TXM scans have an explicit zero-padded reconstruction FOV. Synchrotron TIFF
    slices may instead have gray exterior and phase contrast; texture support
    avoids classifying individual dark phases as exterior.
    """
    image = np.asarray(image, np.float32)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError('A finite two-dimensional source slice is required.')
    low, high = np.percentile(image, [1, 99.5])
    norm = np.clip((image - low) / max(float(high - low), 1e-6), 0, 1)
    smooth = ndi.gaussian_filter(norm, 1.2)
    if acquisition_kind == 'TXM':
        acquired = ndi.binary_erosion(image > 0, iterations=5)
        if acquired.sum() < 64:
            raise ValueError('Missing nonzero reconstructed field of view.')
        threshold = float(filters.threshold_otsu(smooth[acquired]))
        mask = (smooth > threshold) & acquired
        mask = ndi.binary_closing(mask, structure=morphology.disk(3))
        method = dict(method='within_FOV_intensity_envelope', threshold=threshold,
                      fov_erosion_px=5, closing_radius_px=3)
    else:
        mean = ndi.gaussian_filter(smooth, 4)
        local_std = np.sqrt(np.maximum(ndi.gaussian_filter(smooth * smooth, 4) - mean * mean, 0))
        high_threshold = float(filters.threshold_otsu(local_std))
        low_threshold = 0.5 * high_threshold
        mask = filters.apply_hysteresis_threshold(local_std, low_threshold, high_threshold)
        mask = ndi.binary_closing(mask, structure=morphology.disk(5))
        method = dict(method='full_slice_texture_envelope', high_threshold=high_threshold,
                      low_threshold=low_threshold, local_std_sigma_px=4, closing_radius_px=5)
    mask = _largest_filled(mask)
    return mask, dict(**method, full_specimen_fraction=float(mask.mean()))


def prepare_patch(record):
    """Keep v3 slice selection and crop coordinates; correct the full-slice support."""
    image, header = read_plane(record)
    full_mask, information = specimen_envelope(image, header['kind'])
    y, x = record['crop_origin_yx']
    height, width = record['patch_shape']
    patch = image[y:y + height, x:x + width].copy()
    mask = full_mask[y:y + height, x:x + width].copy()
    if patch.shape != (height, width) or mask.sum() < 64:
        raise ValueError('The recorded crop is invalid or misses the specimen.')
    return patch, mask, information
