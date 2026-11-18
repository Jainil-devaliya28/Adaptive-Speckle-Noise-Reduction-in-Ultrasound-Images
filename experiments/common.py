"""Shared experiment settings: parameter grids, tuned parameters, data loaders."""
from __future__ import annotations

import glob
import json
import os
import sys
from functools import partial

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from speckle import filters as F  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(RESULTS, "figures")
os.makedirs(FIGS, exist_ok=True)
BUSI_DIR = os.environ.get("BUSI_DIR", os.path.join(os.path.dirname(ROOT), "data_src", "busi", "Dataset_BUSI", "data_with_GT"))

# Every method gets a fair search over its main parameters on a separate
# tuning set (different phantoms and seeds from the test set).
GRIDS = {
    "Median": {"size": [3, 5, 7, 9, 11, 15, 19]},
    "Lee": {"size": [9, 15, 21, 31, 41]},
    "Kuan": {"size": [9, 15, 21, 31, 41]},
    "Frost": {"size": [7, 11, 15], "damping": [0.1, 0.25, 0.5, 1.0]},
    "Perona-Malik": {"kappa": [0.02, 0.03, 0.05, 0.08], "n_iter": [50, 100, 200, 400]},
    "SRAD": {"n_iter": [100, 200, 400, 800, 1500]},
    "DPAD": {"n_iter": [100, 200, 400, 800], "window": [7, 9, 13]},
    "TV": {"weight": [0.1, 0.2, 0.3, 0.45, 0.6, 0.9]},
    "Bilateral": {"sigma_color": [0.1, 0.2, 0.3, 0.5], "sigma_spatial": [2, 4, 6]},
    "OBNLM": {"beta": [3, 5, 8, 12, 20], "patch": [5, 7, 9]},
    "AS-NLM (proposed)": {"beta_max": [3, 5, 8], "diff_iter": [80, 150, 250], "kappa_rel": [0.15, 0.2, 0.3]},
}

SHORT = {"AS-NLM (proposed)": "AS-NLM"}


def tuned_params():
    p = os.path.join(RESULTS, "tuned_params.json")
    return json.load(open(p)) if os.path.exists(p) else {k: {} for k in F.FILTERS}


def tuned_filters():
    tp = tuned_params()
    return {name: partial(fn, **tp.get(name, {})) for name, fn in F.FILTERS.items()}


def load_busi(classes=("benign", "malignant"), n_per_class=None, seed=0):
    """Return list of (name, image[0..1] float, mask bool)."""
    from skimage import io
    rng = np.random.default_rng(seed)
    out = []
    for c in classes:
        files = sorted(f for f in glob.glob(os.path.join(BUSI_DIR, c, "*.png")) if "_mask" not in f)
        if n_per_class:
            files = list(rng.choice(files, size=min(n_per_class, len(files)), replace=False))
        for f in files:
            img = io.imread(f)
            if img.ndim == 3:
                img = img[..., :3].mean(-1)
            img = img.astype(float) / 255.0
            mf = f[:-4] + "_mask.png"
            mask = None
            if os.path.exists(mf):
                m = io.imread(mf).astype(float)
                m = m if m.ndim == 2 else m[..., 0]
                mask = m > 0.5 * max(m.max(), 1e-9)
            out.append((os.path.basename(f)[:-4], img, mask))
    return out
