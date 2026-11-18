"""Speckle reduction filters.

All filters take a float image in [0, 1] (log-compressed B-mode, as displayed
by a scanner) and return a float image in [0, 1].

Classical local-statistics filters : median, lee, kuan, frost
Diffusion filters                  : perona_malik, srad, dpad
Variational / edge-aware           : tv, bilateral
Non-local                          : obnlm (Pearson distance, Coupe et al. 2009)
Proposed                           : asnlm (Adaptive Speckle-statistics NLM)
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter, median_filter as _median
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral

EPS = 1e-6


# ---------------------------------------------------------------- helpers
def local_stats(u, size):
    m = uniform_filter(u, size, mode="reflect")
    m2 = uniform_filter(u * u, size, mode="reflect")
    v = np.maximum(m2 - m * m, 0.0)
    return m, v


def estimate_cu2(u, size=7):
    """Squared coefficient of variation of pure speckle, Cu^2.

    Estimated as the median of the local CV^2 map. Most pixels in an
    ultrasound image belong to homogeneous speckle, so the median lands on
    the speckle mode and is robust to edges (Aja-Fernandez & Alberola 2006).
    """
    m, v = local_stats(u, size)
    ci2 = v / np.maximum(m, EPS) ** 2
    return float(np.median(ci2[m > 0.05]))


def estimate_sigma2_loupas(u, size=7):
    """Noise level sigma^2 for the display-domain model I = R + sqrt(R) n."""
    m, v = local_stats(u, size)
    r = v / np.maximum(m, EPS)
    return float(np.median(r[m > 0.05]))


# ---------------------------------------------------------------- classical
def median(u, size=5):
    return _median(u, size=size, mode="reflect")


def lee(u, size=7, cu2=None):
    """Lee (1980) adaptive filter: I_hat = m + k (I - m), k = 1 - Cu^2/Ci^2."""
    cu2 = estimate_cu2(u, size) if cu2 is None else cu2
    m, v = local_stats(u, size)
    ci2 = v / np.maximum(m, EPS) ** 2
    k = np.clip(1.0 - cu2 / np.maximum(ci2, EPS), 0.0, 1.0)
    return np.clip(m + k * (u - m), 0, 1)


def kuan(u, size=7, cu2=None):
    """Kuan (1985) LMMSE filter for multiplicative noise."""
    cu2 = estimate_cu2(u, size) if cu2 is None else cu2
    m, v = local_stats(u, size)
    ci2 = v / np.maximum(m, EPS) ** 2
    k = np.clip((1.0 - cu2 / np.maximum(ci2, EPS)) / (1.0 + cu2), 0.0, 1.0)
    return np.clip(m + k * (u - m), 0, 1)


def frost(u, size=7, damping=2.0, cu2=None):
    """Frost (1982): exponentially weighted kernel, decay ~ K * (Ci/Cu)^2 * |t|."""
    cu2 = estimate_cu2(u, size) if cu2 is None else cu2
    m, v = local_stats(u, size)
    ci2 = v / np.maximum(m, EPS) ** 2
    alpha = damping * ci2 / max(cu2, EPS)
    r = size // 2
    up = np.pad(u, r, mode="reflect")
    num = np.zeros_like(u)
    den = np.zeros_like(u)
    H, W = u.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            d = np.hypot(dy, dx)
            w = np.exp(-alpha * d)
            num += w * up[r + dy: r + dy + H, r + dx: r + dx + W]
            den += w
    return np.clip(num / den, 0, 1)


# ---------------------------------------------------------------- diffusion
def _grads(u):
    up = np.pad(u, 1, mode="edge")
    n = up[:-2, 1:-1] - u
    s = up[2:, 1:-1] - u
    w = up[1:-1, :-2] - u
    e = up[1:-1, 2:] - u
    return n, s, w, e


def perona_malik(u, n_iter=30, kappa=0.08, dt=0.15):
    """Perona-Malik (1990) anisotropic diffusion, g = 1 / (1 + (|grad|/kappa)^2)."""
    u = u.copy()
    for _ in range(n_iter):
        n, s, w, e = _grads(u)
        g = lambda d: 1.0 / (1.0 + (d / kappa) ** 2)
        u = u + dt * (g(n) * n + g(s) * s + g(w) * w + g(e) * e)
    return np.clip(u, 0, 1)


def srad(u, n_iter=200, dt=0.2, q0_window=7):
    """Speckle Reducing Anisotropic Diffusion (Yu & Acton, IEEE TIP 2002).

    The speckle scale q0 is re-estimated every iteration as the median of the
    local CV map (automatic version, no manual homogeneous ROI needed).
    """
    I = np.maximum(u.copy(), 1e-3)
    for _ in range(n_iter):
        q02 = estimate_cu2(I, q0_window)
        n, s, w, e = _grads(I)
        grad2 = (n**2 + s**2 + w**2 + e**2) / np.maximum(I, EPS) ** 2
        lap = (n + s + w + e) / np.maximum(I, EPS)
        q2 = (0.5 * grad2 - (1 / 16) * lap**2) / (1 + 0.25 * lap) ** 2
        c = 1.0 / (1.0 + (q2 - q02) / (q02 * (1 + q02) + EPS))
        c = np.clip(c, 0, 1)
        cp = np.pad(c, 1, mode="edge")
        cs, ce = cp[2:, 1:-1], cp[1:-1, 2:]
        div = cs * s + c * n + ce * e + c * w
        I = I + dt * div  # explicit scheme, stable for dt <= 0.25
    return np.clip(I, 0, 1)


def dpad(u, n_iter=40, dt=0.2, window=5):
    """Detail Preserving Anisotropic Diffusion (Aja-Fernandez & Alberola 2006).

    Like SRAD, but the diffusion coefficient uses the Kuan filter's local CV
    estimated over a window, which is more robust than SRAD's instantaneous q.
    """
    I = np.maximum(u.copy(), 1e-3)
    for _ in range(n_iter):
        m, v = local_stats(I, window)
        ci2 = np.maximum(v / np.maximum(m, EPS) ** 2, EPS)
        cu2 = float(np.median(ci2))
        c = np.clip((1 + 1 / ci2) / (1 + 1 / cu2), 0, 1)
        n, s, w, e = _grads(I)
        cp = np.pad(c, 1, mode="edge")
        cn, cs, cw, ce = cp[:-2, 1:-1], cp[2:, 1:-1], cp[1:-1, :-2], cp[1:-1, 2:]
        # face coefficients = mean of the two cells
        I = I + dt * (0.5 * (c + cn) * n + 0.5 * (c + cs) * s
                      + 0.5 * (c + cw) * w + 0.5 * (c + ce) * e)
    return np.clip(I, 0, 1)


# ---------------------------------------------------------------- variational
def tv(u, weight=0.08):
    return np.clip(denoise_tv_chambolle(u, weight=weight), 0, 1)


def bilateral(u, sigma_color=0.08, sigma_spatial=3):
    return np.clip(denoise_bilateral(u, sigma_color=sigma_color,
                                     sigma_spatial=sigma_spatial), 0, 1)


# ---------------------------------------------------------------- non-local
def _nlm_core(u, h2_map, patch=5, search=11):
    """Pixelwise NLM with Pearson distance (Bayesian NLM for the Loupas model,
    gamma = 0.5). h2_map may be a scalar or a per-pixel array: that is what
    lets the proposed method change the smoothing strength locally.

    Distance between patches P(x) and P(y):
        d(x, y) = mean_p (u(x+p) - u(y+p))^2 / u(y+p)
    Weight: exp(-d / h^2). The centre pixel gets the max weight of the others.
    """
    r = search // 2
    H, W = u.shape
    up = np.pad(u, r, mode="reflect")
    num = np.zeros_like(u)
    den = np.zeros_like(u)
    wmax = np.zeros_like(u)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy == 0 and dx == 0:
                continue
            S = up[r + dy: r + dy + H, r + dx: r + dx + W]
            D = (u - S) ** 2 / np.maximum(S, 0.02)
            Dm = uniform_filter(D, patch, mode="reflect")
            w = np.exp(-Dm / h2_map)
            num += w * S
            den += w
            np.maximum(wmax, w, out=wmax)
    wmax = np.where(wmax == 0, 1.0, wmax)
    num += wmax * u
    den += wmax
    return np.clip(num / den, 0, 1)


def obnlm(u, beta=1.0, patch=5, search=11, sigma2=None):
    """Optimized Bayesian NLM (Coupe et al., IEEE TIP 2009), pixelwise version.
    h^2 = 2 * beta * sigma^2 under the Loupas model."""
    sigma2 = estimate_sigma2_loupas(u) if sigma2 is None else sigma2
    return _nlm_core(u, 2.0 * beta * sigma2, patch, search)


def speckle_index(u, sigma2, window=15):
    """Local speckle-statistics index r(x) = local_var / (local_mean * sigma^2).

    Under the Loupas model (I = R + sqrt(R) n), a window containing only fully
    developed speckle has var = sigma^2 * mean, so r is close to 1. Edges,
    boundaries and real texture add variance that speckle cannot explain, so
    r rises well above 1. This is a per-pixel hypothesis test: "is what I see
    here explained by speckle alone?"
    """
    m, v = local_stats(u, window)
    return v / (np.maximum(m, 0.02) * sigma2)


def structure_map(u, sigma2, r_lo=1.5, r_hi=3.0, window=15, smooth=2.0):
    """Soft structure weight w(x) in [0, 1]: 0 = pure speckle, 1 = structure."""
    r = speckle_index(u, sigma2, window)
    w = np.clip((r - r_lo) / (r_hi - r_lo), 0, 1)
    if smooth:
        w = gaussian_filter(w, smooth)
    return r, w


def gated_diffusion(u, c, n_iter=60, dt=0.2, kappa=0.03):
    """Diffusion steered by the speckle-statistics structure map.

    Face conductance = c_gate + (1 - c_gate) * g(|grad|), where
      c_gate = (1 - w)^p, fixed from Stage 1 (min of the two cells, so the
               gate cannot leak across a detected boundary), and
      g      = 1 / (1 + (|grad| / kappa)^2), the Perona-Malik edge stop.
    In pure speckle (w ~ 0) this is plain linear diffusion: strong, and free
    of the false edges Perona-Malik creates in noise. Inside the structure
    zone (w ~ 1) it becomes edge-stopping diffusion, smoothing along a
    boundary but not across it. kappa=None disables the edge-stop term.
    """
    u = u.copy()
    cp = np.pad(c, 1, mode="edge")
    gates = [np.minimum(c, cp[:-2, 1:-1]), np.minimum(c, cp[2:, 1:-1]),
             np.minimum(c, cp[1:-1, :-2]), np.minimum(c, cp[1:-1, 2:])]
    for _ in range(n_iter):
        flux = 0.0
        for d, cg in zip(_grads(u), gates):
            if kappa:
                cg = cg + (1 - cg) / (1 + (d / kappa) ** 2)
            flux = flux + cg * d
        u = u + dt * flux
    return u


def asnlm(u, beta_min=0.5, beta_max=3.0, r_lo=1.5, r_hi=3.0, window=15,
          smooth=2.0, patch=5, search=11, refine_window=5, refine_lo=3.0,
          refine_hi=7.5, diff_iter=150, gate_power=2.0, kappa_rel=0.2,
          noise_window=7, sigma2=None, return_maps=False):
    """PROPOSED: Adaptive Speckle-statistics Non-Local Means (AS-NLM).

    Stage 1  Speckle statistics
             Estimate the speckle level sigma^2 (Loupas display-domain model)
             and the per-pixel speckle index r(x) = var / (mean * sigma^2).
             Map it to a soft structure weight w1(x) in [0, 1].
    Stage 2  Adaptive Bayesian NLM
             NLM with Pearson patch distance and a per-pixel strength
                 h^2(x) = 2 sigma^2 beta(x),
                 beta(x) = beta_max - (beta_max - beta_min) w1(x)
             Strong averaging where the statistics say "only speckle here",
             gentle averaging where they say "real structure here".
    Stage 3  Re-test on the NLM estimate
             Speckle variance is now much lower, so weak (low-contrast)
             boundaries that were hidden in Stage 1 stand out. A second index
             on the NLM output, relative to its own median, gives w2(x);
             the final structure map is w = max(w1, w2).
    Stage 4  Structure-gated diffusion
             NLM leaves low-frequency blotches in homogeneous tissue because
             speckle grains are larger than its patches. Linear diffusion with
             gate (1 - w)^p removes them; inside the structure zone a
             Perona-Malik edge stop with kappa tied to the noise level lets it
             smooth along boundaries but not across them.

    Every threshold is relative to the estimated speckle level, so the
    method adapts to gain, frequency and dynamic range without retuning.

    noise_window: window used to estimate sigma^2. v1 (as tuned and reported
    in the main results) uses 7. The post-hoc variant AS-NLM-c sets it equal
    to `window`, so the index r has median ~1 by construction; see
    experiments/08_posthoc_window.py.
    """
    if sigma2 is None:
        sigma2 = estimate_sigma2_loupas(u, noise_window or window)
    r, w1 = structure_map(u, sigma2, r_lo, r_hi, window, smooth)
    beta = beta_max - (beta_max - beta_min) * w1
    nlm = _nlm_core(u, 2.0 * beta * sigma2, patch, search)
    m, v = local_stats(nlm, refine_window)
    r2 = v / np.maximum(m, 0.02)
    r2 = r2 / (np.median(r2) + EPS)
    w2 = gaussian_filter(np.clip((r2 - refine_lo) / (refine_hi - refine_lo), 0, 1), 1.0)
    w = np.maximum(w1, w2)
    kappa = kappa_rel * np.sqrt(sigma2 * np.median(u))
    out = gated_diffusion(nlm, (1 - w) ** gate_power, diff_iter, kappa=kappa) if diff_iter else nlm
    out = np.clip(out, 0, 1)
    if return_maps:
        return out, {"r": r, "w1": w1, "w2": w2, "w": w, "beta": beta,
                     "sigma2": sigma2, "kappa": kappa, "nlm": nlm}
    return out


# registry used by experiments and the CLI
FILTERS = {
    "Median": median,
    "Lee": lee,
    "Kuan": kuan,
    "Frost": frost,
    "Perona-Malik": perona_malik,
    "SRAD": srad,
    "DPAD": dpad,
    "TV": tv,
    "Bilateral": bilateral,
    "OBNLM": obnlm,
    "AS-NLM (proposed)": asnlm,
}
