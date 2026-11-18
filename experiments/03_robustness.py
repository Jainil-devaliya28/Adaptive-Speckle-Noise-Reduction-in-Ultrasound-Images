"""Step 3: robustness to acquisition settings the filters were NOT tuned for.

All parameters stay fixed at the values tuned for 5 MHz / 50 dB / 40 dB SNR.
We then change one acquisition setting at a time:
  * transmit frequency (speckle grain size scales with wavelength)
  * display dynamic range (changes speckle contrast in the displayed image)
  * electronic SNR
A truly adaptive filter should hold its quality; a filter with absolute
thresholds (e.g. Perona-Malik's kappa, TV weight) is expected to drift.
"""
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import RESULTS, tuned_filters
from speckle import simulate as S, metrics as M

CONDITIONS = (
    [("f0 (MHz)", f, dict(f0_mhz=f, lateral_fwhm_mm=0.9 * 5.0 / f)) for f in (3.5, 5.0, 7.5, 10.0)]
    + [("Dynamic range (dB)", d, dict(dyn_range_db=d)) for d in (35, 50, 65, 80)]
    + [("SNR (dB)", s, dict(snr_db=s)) for s in (15, 25, 40)]
)
PHANTOMS = [("cyst", 7), ("organ", 27), ("edges", 37)]


def run(job):
    (factor, level, kw), (kind, seed) = job
    b, ref, _ = S.simulate_bmode(S.phantom(kind, seed=seed), seed=seed, **kw)
    rows = []
    for name, fn in {"Noisy": lambda x: x, **tuned_filters()}.items():
        o = fn(b)
        rows.append({"factor": factor, "level": level, "phantom": kind, "method": name,
                     **M.full_reference(ref, o)})
    return rows


if __name__ == "__main__":
    jobs = [(c, p) for c in CONDITIONS for p in PHANTOMS]
    with Pool(os.cpu_count()) as pool:
        res = pool.map(run, jobs, chunksize=1)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv(os.path.join(RESULTS, "robustness.csv"), index=False)
    t = df.groupby(["factor", "level", "method"], sort=False)[["SSIM", "FOM"]].mean().unstack("method")
    print(t["SSIM"].round(3).to_string())
    # spread across conditions = robustness (lower is better)
    g = df.groupby(["factor", "level", "method"])["SSIM"].mean().groupby("method").agg(["mean", "min", "std"])
    print(g.sort_values("mean", ascending=False).round(3))
