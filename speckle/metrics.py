"""Image quality metrics for despeckling.

Full-reference (simulation, where a speckle-free ground truth exists):
    PSNR, SSIM, Pratt's Figure Of Merit (FOM), EPI (edge preservation index)
No-reference (real images, using ROIs):
    SNR = mean/std in a homogeneous ROI, ENL = SNR^2,
    CNR, gCNR (generalized CNR, Rodriguez-Molares 2020),
    boundary sharpness ratio (BSR) along an annotated lesion contour.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_laplace, distance_transform_edt, binary_dilation, binary_erosion, sobel
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.feature import canny


def psnr(ref, img):
    return float(peak_signal_noise_ratio(ref, img, data_range=1.0))


def ssim(ref, img):
    return float(structural_similarity(ref, img, data_range=1.0))


def epi(ref, img, sigma=2.0):
    """Edge Preservation Index (Sattar et al. 1997): correlation of the
    high-pass (Laplacian of Gaussian) components of reference and filtered
    image (1 = perfect). A small Gaussian scale keeps it from scoring
    pixel-level speckle as 'edges'."""
    a = gaussian_laplace(ref, sigma)
    b = gaussian_laplace(img, sigma)
    a -= a.mean()
    b -= b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


def pratt_fom(ref, img, sigma=3.0, alpha=1 / 9):
    """Pratt's figure of merit between Canny edge maps (1 = ideal edges)."""
    e_ref = canny(ref, sigma=sigma)
    e_img = canny(img, sigma=sigma)
    n_ref, n_img = e_ref.sum(), e_img.sum()
    if n_ref == 0 or n_img == 0:
        return 0.0
    d = distance_transform_edt(~e_ref)
    return float((1.0 / (1 + alpha * d[e_img] ** 2)).sum() / max(n_ref, n_img))


def snr_roi(img, mask):
    v = img[mask]
    return float(v.mean() / (v.std() + 1e-12))


def enl(img, mask):
    return snr_roi(img, mask) ** 2


def cnr(img, m_in, m_out):
    a, b = img[m_in], img[m_out]
    return float(abs(a.mean() - b.mean()) / np.sqrt(a.var() + b.var() + 1e-12))


def gcnr(img, m_in, m_out, bins=256):
    """Generalized CNR: 1 - overlap of the two intensity histograms.
    Bounded in [0, 1], invariant to monotonic transforms (e.g. log compression)."""
    h1, edges = np.histogram(img[m_in], bins=bins, range=(0, 1), density=True)
    h2, _ = np.histogram(img[m_out], bins=bins, range=(0, 1), density=True)
    dx = edges[1] - edges[0]
    return float(1 - np.minimum(h1, h2).sum() * dx)


def lesion_rois(mask, gap=5, ring=20):
    """Lesion interior (eroded) and a background ring around the lesion."""
    mask = mask.astype(bool)
    inner = binary_erosion(mask, iterations=gap)
    outer = binary_dilation(mask, iterations=gap + ring) & ~binary_dilation(mask, iterations=gap)
    return inner, outer


def boundary_sharpness(img, mask, width=2):
    """Mean gradient magnitude on the annotated lesion boundary."""
    mask = mask.astype(bool)
    edge = binary_dilation(mask, iterations=width) & ~binary_erosion(mask, iterations=width)
    g = np.hypot(sobel(img, 0), sobel(img, 1))
    return float(g[edge].mean())


def full_reference(ref, img):
    return {"PSNR": psnr(ref, img), "SSIM": ssim(ref, img),
            "FOM": pratt_fom(ref, img), "EPI": epi(ref, img)}
