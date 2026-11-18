"""Physics-based B-mode ultrasound speckle simulator.

Model (linear, space-invariant, Born approximation):
    RF(x, z) = T(x, z) * h(x, z)
where T is a field of randomly located point scatterers whose amplitude
variance follows a tissue echogenicity map E(x, z), and h is the pulse-echo
point spread function. We work directly at baseband (IQ), so the envelope is
|RF_IQ|, and the B-mode image is the log-compressed envelope.

Because many scatterers fall inside each resolution cell, the IQ signal is a
circular complex Gaussian process and the envelope is Rayleigh distributed:
this is fully developed speckle. The speckle-free reference is the expected
intensity E[|RF_IQ|^2] = E * |h|^2, log-compressed with the same settings,
which gives us a ground truth that real scanners never provide.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

C_SOUND = 1540.0  # m/s


def make_psf(f0_mhz=5.0, frac_bw=0.6, lateral_fwhm_mm=0.9, dz_mm=0.05, dx_mm=0.05):
    """Complex baseband pulse-echo PSF (Gaussian envelope, carrier removed but
    the round-trip phase term kept so axial interference produces speckle)."""
    lam = C_SOUND / (f0_mhz * 1e6) * 1e3  # wavelength in mm
    # axial: Gaussian pulse, -6 dB fractional bandwidth -> pulse length
    sigma_f = frac_bw * f0_mhz * 1e6 / (2 * np.sqrt(2 * np.log(2)))
    sigma_t = 1.0 / (2 * np.pi * sigma_f)
    sigma_z = sigma_t * C_SOUND / 2 * 1e3  # mm (round trip)
    sigma_x = lateral_fwhm_mm / (2 * np.sqrt(2 * np.log(2)))
    nz = int(np.ceil(4 * sigma_z / dz_mm))
    nx = int(np.ceil(4 * sigma_x / dx_mm))
    z = np.arange(-nz, nz + 1) * dz_mm
    x = np.arange(-nx, nx + 1) * dx_mm
    X, Z = np.meshgrid(x, z)
    env = np.exp(-Z**2 / (2 * sigma_z**2) - X**2 / (2 * sigma_x**2))
    phase = np.exp(1j * 4 * np.pi * Z / lam)
    return env * phase


def phantom(kind="cyst", shape=(512, 384), seed=0, texture=0.35,
            texture_scale=3.0, n_septa=6):
    """Echogenicity map (linear intensity scale, 1.0 = background tissue)."""
    H, W = shape
    zz, xx = np.mgrid[0:H, 0:W]
    E = np.ones(shape)
    if kind == "cyst":
        # layered tissue + anechoic cysts + hyper/hypo-echoic inclusions
        E[zz < 0.12 * H] = 2.0                      # bright skin/fat layer
        E[(zz > 0.12 * H) & (zz < 0.16 * H)] = 0.5  # thin hypoechoic fascia
        circles = [  # (cz, cx, r, value)
            (0.30, 0.28, 0.07, 0.0),   # anechoic cyst (large)
            (0.30, 0.70, 0.045, 0.0),  # anechoic cyst (small)
            (0.55, 0.30, 0.08, 4.0),   # hyperechoic +6 dB
            (0.55, 0.72, 0.08, 0.25),  # hypoechoic -6 dB
            (0.80, 0.50, 0.10, 0.5),   # subtle lesion -3 dB
        ]
        for cz, cx, r, v in circles:
            m = (zz - cz * H) ** 2 + (xx - cx * W) ** 2 < (r * min(H, W) * 1.6) ** 2
            E[m] = v
        # thin bright wire targets (resolution) near the bottom
        for k in range(5):
            E[int(0.93 * H), int((0.15 + 0.17 * k) * W)] = 400.0
    elif kind == "organ":
        rng = np.random.default_rng(seed)
        # smooth texture + a lobulated "organ" boundary + vessel
        from scipy.ndimage import gaussian_filter
        tex = gaussian_filter(rng.standard_normal(shape), 25)
        tex = 1 + 0.6 * tex / np.abs(tex).max()
        r0 = 0.30 * min(H, W) * 1.4
        ang = np.arctan2(zz - 0.5 * H, xx - 0.5 * W)
        rr = np.hypot(zz - 0.5 * H, xx - 0.5 * W)
        organ = rr < r0 * (1 + 0.12 * np.sin(5 * ang) + 0.05 * np.cos(9 * ang))
        E = np.where(organ, 1.8 * tex, 0.8 * tex)
        vessel = np.abs((zz - 0.62 * H) - 0.25 * (xx - 0.5 * W)) < 0.025 * H
        E[vessel & organ] = 0.02
        E[zz < 0.08 * H] = 2.5
    elif kind == "edges":
        # vertical + horizontal + oblique step edges for edge-preservation tests
        E[:, : W // 3] = 0.3
        E[:, 2 * W // 3:] = 3.0
        E[H // 2:, W // 3: 2 * W // 3] = 0.08
        diag = (zz - 0.2 * H) > 1.6 * (xx - 0.2 * W)
        E[(zz < 0.45 * H) & diag & (xx > W // 3) & (xx < 2 * W // 3)] = 2.0
    else:
        raise ValueError(kind)
    if texture:
        # fine-scale tissue microstructure (fibro-glandular texture, small
        # ducts/septa). It is part of the ground truth, so a filter that just
        # blurs everything is penalised, as it would be clinically.
        from scipy.ndimage import gaussian_filter
        rng = np.random.default_rng(seed + 1000)
        t = gaussian_filter(rng.standard_normal(shape), texture_scale)
        t /= t.std() + 1e-12
        E = E * np.exp(texture * t)
        for _ in range(n_septa):  # thin bright curvilinear septa
            z0, x0 = rng.uniform(0.1, 0.9) * H, rng.uniform(0, W)
            ang = rng.uniform(0, np.pi)
            curv = rng.uniform(-0.004, 0.004)
            s = np.linspace(-0.15, 0.15, 400) * min(H, W)
            zl = (z0 + s * np.sin(ang) + curv * s**2).astype(int)
            xl = (x0 + s * np.cos(ang)).astype(int)
            ok = (zl >= 0) & (zl < H) & (xl >= 0) & (xl < W)
            line = np.zeros(shape)
            line[zl[ok], xl[ok]] = 1
            line = gaussian_filter(line, 1.0) > 0.05
            E[line & (E > 0.05)] *= 2.5
    return E


def log_compress(env, dyn_range_db=60.0, ref=None):
    ref = env.max() if ref is None else ref
    db = 20 * np.log10(np.maximum(env, 1e-12) / ref)
    return np.clip((db + dyn_range_db) / dyn_range_db, 0, 1)


def simulate_bmode(E, f0_mhz=5.0, lateral_fwhm_mm=0.9, dyn_range_db=50.0,
                   snr_db=40.0, seed=0, pixel_mm=0.1, oversample=2):
    """Return (bmode, reference, envelope).

    bmode      : log-compressed speckled B-mode image in [0, 1]
    reference  : log-compressed speckle-free expected image in [0, 1]
    envelope   : linear envelope (Rayleigh speckle) before compression
    """
    rng = np.random.default_rng(seed)
    E_out = E
    # simulate on a finer grid (so the carrier phase is properly sampled),
    # then decimate to the display pixel size, like a scanner's beamformer
    E = np.kron(E, np.ones((oversample, oversample)))
    d = pixel_mm / oversample
    h = make_psf(f0_mhz, lateral_fwhm_mm=lateral_fwhm_mm, dz_mm=d, dx_mm=d)
    # dense scatterers: circular complex Gaussian amplitudes per grid point
    scat = (rng.standard_normal(E.shape) + 1j * rng.standard_normal(E.shape)) / np.sqrt(2)
    scat *= np.sqrt(E)
    iq = fftconvolve(scat, h, mode="same")
    # electronic (thermal) noise
    p_sig = np.mean(np.abs(iq) ** 2)
    noise_p = p_sig / 10 ** (snr_db / 10)
    iq += np.sqrt(noise_p / 2) * (rng.standard_normal(E.shape) + 1j * rng.standard_normal(E.shape))
    env = np.abs(iq)[::oversample, ::oversample]
    # speckle-free expected intensity
    exp_int = fftconvolve(E, np.abs(h) ** 2, mode="same")[::oversample, ::oversample] + noise_p
    exp_env = np.sqrt(np.maximum(exp_int, 0))
    # place the median tissue level at mid-grey, as a sonographer's gain would
    ref_level = np.median(exp_env) * 10 ** (dyn_range_db / 2 / 20)
    bmode = log_compress(env, dyn_range_db, ref=ref_level)
    reference = log_compress(exp_env, dyn_range_db, ref=ref_level)
    return bmode, reference, env


def add_multiplicative_speckle(img, sigma=0.2, model="loupas", seed=0):
    """Synthetic speckle on an (already displayed) image.

    loupas : I = R + sqrt(R) * n,  n ~ N(0, sigma^2)   (display-domain model,
             used by Loupas 1989 and Coupe 2009 OBNLM)
    gamma  : I = R * g, g ~ Gamma(L, 1/L) with L = 1/sigma^2 (intensity, L looks)
    """
    rng = np.random.default_rng(seed)
    if model == "loupas":
        out = img + np.sqrt(np.maximum(img, 0)) * rng.normal(0, sigma, img.shape)
    elif model == "gamma":
        L = 1.0 / sigma**2
        out = img * rng.gamma(L, 1.0 / L, img.shape)
    else:
        raise ValueError(model)
    return np.clip(out, 0, 1)
