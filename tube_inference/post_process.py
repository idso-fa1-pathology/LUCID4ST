import os
import numpy as np
import cv2
from scipy import ndimage as ndi


# -----------------------------------------------------------------------------
# Post-processing hyperparameters (hardcoded; python port of post_process.m).
# Only the input/output paths are configurable -- everything below is fixed.
# -----------------------------------------------------------------------------
# Tissue classes in the TME mask (RGB) that are excluded from the tumor bed.
ALVEOLI_RGB = (0, 128, 0)
MUSCLE_RGB = (0, 0, 128)
FAT_RGB = (128, 128, 0)

CLOSE_STRUCT = np.ones((3, 3), np.uint8)   # strel('square', 3)
CONNECTIVITY_STRUCT = np.ones((3, 3), np.uint8)  # bwconncomp 8-connectivity
MIN_COMPONENT_AREA = 10000                 # drop bed islands smaller than this (pixels)
GAUSS_SIGMA = 30                           # imgaussfilt sigma
GAUSS_THRESHOLD = 0.5                      # re-binarize after smoothing

# How a TME mask file is named for a given slide stem (first match wins).
TME_SUFFIXES = ('.tif_Ss1.png', '_Ss1.png', '.png')


def find_tme_mask(tme_dir, stem):
    """Locate the TME mask for a slide stem, trying the known naming conventions."""
    for suffix in TME_SUFFIXES:
        path = os.path.join(tme_dir, stem + suffix)
        if os.path.exists(path):
            return path
    return None


def _color_eq(img_rgb, rgb):
    return (img_rgb[:, :, 0] == rgb[0]) & (img_rgb[:, :, 1] == rgb[1]) & (img_rgb[:, :, 2] == rgb[2])


def post_process(tbed_path, tme_path, out_path):
    """Refine a raw tumor-bed mask (<stem>_tbed.png) using the TME tissue mask.

    Steps (from post_process.m):
      1. align the raw bed mask to the TME mask size,
      2. keep only tissue that is not alveoli / muscle / fat, fill + close it,
      3. mask the bed to that tissue,
      4. remove connected bed components smaller than MIN_COMPONENT_AREA,
      5. Gaussian-smooth and re-threshold the bed,
      6. apply it to the raw RGB mask and write <stem>_tme_tbed.png.
    """
    mask_raw = cv2.cvtColor(cv2.imread(tbed_path), cv2.COLOR_BGR2RGB)
    mask_tme = cv2.cvtColor(cv2.imread(tme_path), cv2.COLOR_BGR2RGB)

    # 1. align the raw bed mask to the TME mask size (top-left aligned, zero padded/cropped)
    m, n = mask_tme.shape[:2]
    aligned = np.zeros((m, n, 3), dtype=mask_raw.dtype)
    mm, nn = min(m, mask_raw.shape[0]), min(n, mask_raw.shape[1])
    aligned[:mm, :nn, :] = mask_raw[:mm, :nn, :]
    mask_raw = aligned

    # 2. tissue = TME mask minus alveoli/muscle/fat, then fill + close + fill
    remove = _color_eq(mask_tme, ALVEOLI_RGB) | _color_eq(mask_tme, MUSCLE_RGB) | _color_eq(mask_tme, FAT_RGB)
    tissue = mask_tme.copy()
    tissue[remove] = 0
    tissue_bin = np.any(tissue > 0, axis=2)
    tissue_bin = ndi.binary_fill_holes(tissue_bin)
    tissue_bin = ndi.binary_closing(tissue_bin, structure=CLOSE_STRUCT)
    tissue_bin = ndi.binary_fill_holes(tissue_bin)

    # 3. bed within tissue
    bed = (mask_raw[:, :, 0] > 0).astype(np.uint8)
    bed = bed * tissue_bin.astype(np.uint8)

    # 4. remove small connected components (8-connectivity)
    labeled, num = ndi.label(bed, structure=CONNECTIVITY_STRUCT)
    if num > 0:
        sizes = np.bincount(labeled.ravel())
        small_labels = np.where(sizes < MIN_COMPONENT_AREA)[0]
        small_labels = small_labels[small_labels != 0]  # never touch background label 0
        if small_labels.size:
            bed[np.isin(labeled, small_labels)] = 0

    # 5. smooth and re-binarize
    bed = ndi.gaussian_filter(bed.astype(np.float64), GAUSS_SIGMA)
    bed = (bed > GAUSS_THRESHOLD).astype(np.uint8)

    # 6. apply the refined bed to the raw RGB mask and save
    mask_final = mask_raw * bed[:, :, None]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(mask_final, cv2.COLOR_RGB2BGR))
    return out_path
