"""Step 2: main benchmark on simulated phantoms (held-out test seeds).

12 test images: 4 cyst, 4 organ, 4 edge phantoms (seeds never used in tuning).
Full-reference metrics (PSNR, SSIM, FOM, EPI) on all, plus lesion
detectability (CNR, gCNR) for the anechoic, hyper- and hypo-echoic targets of
the cyst phantom. Writes results/sim_per_image.csv and results/sim_summary.csv,
and saves one example set of outputs for the figures.
"""
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import RESULTS, tuned_filters
from speckle import simulate as S, metrics as M

TEST_SET = [("cyst", s) for s in (1, 2, 3, 4)] + [("organ", s) for s in (21, 22, 23, 24)] \
    + [("edges", s) for s in (31, 32, 33, 34)]

# cyst phantom target geometry (mirrors simulate.phantom)
TARGETS = {"anechoic": (0.30, 0.28, 0.07), "hyperechoic": (0.55, 0.30, 0.08),
           "hypoechoic": (0.55, 0.72, 0.08), "subtle -3dB": (0.80, 0.50, 0.10)}


def target_masks(shape):
    H, W = shape
    zz, xx = np.mgrid[0:H, 0:W]
    out = {}
    for name, (cz, cx, r) in TARGETS.items():
        R = r * min(H, W) * 1.6
        d = np.hypot(zz - cz * H, xx - cx * W)
        out[name] = (d < 0.8 * R, (d > 1.2 * R) & (d < 1.7 * R))
    return out


def run_one(job):
    kind, seed = job
    b, ref, _ = S.simulate_bmode(S.phantom(kind, seed=seed), seed=seed)
    rows, outs = [], {"B-mode (input)": b, "Reference": ref}
    filters = {"Noisy": lambda x: x, **tuned_filters()}
    for name, fn in filters.items():
        t0 = time.time()
        o = fn(b)
        dt = time.time() - t0
        row = {"phantom": kind, "seed": seed, "method": name, "time_s": dt, **M.full_reference(ref, o)}
        if kind == "cyst":
            for tname, (m_in, m_out) in target_masks(b.shape).items():
                row[f"CNR {tname}"] = M.cnr(o, m_in, m_out)
                row[f"gCNR {tname}"] = M.gcnr(o, m_in, m_out)
        rows.append(row)
        outs[name] = o
    return rows, (outs if (kind, seed) in (("cyst", 1), ("organ", 21), ("edges", 31)) else None)


if __name__ == "__main__":
    with Pool(os.cpu_count()) as pool:
        res = pool.map(run_one, TEST_SET, chunksize=1)
    rows = [r for rr, _ in res for r in rr]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "sim_per_image.csv"), index=False)
    metrics = [c for c in df.columns if c not in ("phantom", "seed", "method")]
    summ = df.groupby("method", sort=False)[metrics].agg(["mean", "std"])
    summ.columns = [f"{a} {b}" for a, b in summ.columns]
    summ.to_csv(os.path.join(RESULTS, "sim_summary.csv"))
    for (kind, seed), (_, outs) in zip(TEST_SET, res):
        if outs is not None:
            np.savez_compressed(os.path.join(RESULTS, f"sim_example_{kind}.npz"), **{k: v.astype(np.float32) for k, v in outs.items()})
    show = ["PSNR", "SSIM", "FOM", "EPI", "gCNR anechoic", "gCNR hypoechoic", "time_s"]
    print(df.groupby("method", sort=False)[show].mean().round(3).to_string())
