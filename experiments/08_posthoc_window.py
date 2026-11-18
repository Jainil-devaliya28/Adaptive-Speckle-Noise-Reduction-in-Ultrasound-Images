"""Step 8 (POST-HOC): consistent noise-estimation window, AS-NLM-c.

Found after the BUSI results: v1 estimates sigma^2 with a 7x7 window but
computes the speckle index r with a 15x15 window. On simulated speckle the
two agree; on real scanner images (vendor-smoothed speckle, fine tissue
texture) the larger window sees extra variance, so about half of every BUSI
image is flagged as 'structure' (vs 5% on phantoms) and v1 smooths too little.
AS-NLM-c uses the same 15x15 window for both, so median(r) is ~1 by design.

Because this change was chosen after seeing the BUSI results, it is reported
separately and should be confirmed on new data. No other parameter changes.
"""
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import wilcoxon

from common import RESULTS, F, load_busi, tuned_params
from speckle import simulate as S, metrics as M

P = tuned_params()["AS-NLM (proposed)"]
V = {"AS-NLM v1": dict(P), "AS-NLM-c": {**P, "noise_window": None}}
SIM = [("cyst", s) for s in (1, 2, 3, 4)] + [("organ", s) for s in (21, 22, 23, 24)] + [("edges", s) for s in (31, 32, 33, 34)]


def sim_job(job):
    kind, seed = job
    b, ref, _ = S.simulate_bmode(S.phantom(kind, seed=seed), seed=seed)
    return [{"phantom": kind, "variant": n, **M.full_reference(ref, F.asnlm(b, **p))} for n, p in V.items()]


def rob_job(job):
    (factor, level, kw), (kind, seed) = job
    b, ref, _ = S.simulate_bmode(S.phantom(kind, seed=seed), seed=seed, **kw)
    return [{"factor": factor, "level": level, "variant": n, "SSIM": M.ssim(ref, F.asnlm(b, **p))} for n, p in V.items()]


def busi_job(item):
    import importlib
    seg = importlib.import_module("05_busi_real")
    name, img, mask = item
    inner, ring = M.lesion_rois(mask)
    if inner.sum() < 50 or ring.sum() < 50:
        return []
    band = ndi.binary_dilation(mask, iterations=2) & ~ndi.binary_erosion(mask, iterations=2)
    rows = []
    for n, p in V.items():
        o, mp = F.asnlm(img, return_maps=True, **p)
        g = np.hypot(ndi.sobel(o, 0), ndi.sobel(o, 1))
        rows.append({"image": name, "method": n, "CNR": M.cnr(o, inner, ring), "gCNR": M.gcnr(o, inner, ring),
                     "SNR_bg": M.snr_roi(o, ring), "BCR": float(g[band].mean() / (g[ring & ~band].mean() + 1e-9)),
                     "Dice": seg.dice(seg.segment(o, mask), mask), "structure_frac": float((mp["w"] > 0.5).mean())})
    return rows


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from importlib import import_module
    R = import_module("03_robustness")
    with Pool(os.cpu_count()) as pool:
        sim = pd.DataFrame([r for rr in pool.map(sim_job, SIM) for r in rr])
        rob = pd.DataFrame([r for rr in pool.map(rob_job, [(c, p) for c in R.CONDITIONS for p in R.PHANTOMS]) for r in rr])
        data = [d for d in load_busi(n_per_class=int(os.environ.get("N_PER_CLASS", 50)), seed=0) if d[2] is not None]
        busi = pd.DataFrame([r for rr in pool.map(busi_job, data) for r in rr])
    sim.to_csv(os.path.join(RESULTS, "posthoc_sim.csv"), index=False)
    rob.to_csv(os.path.join(RESULTS, "posthoc_robustness.csv"), index=False)
    busi.to_csv(os.path.join(RESULTS, "posthoc_busi.csv"), index=False)
    print(sim.groupby("variant")[["PSNR", "SSIM", "FOM", "EPI"]].mean().round(3))
    g = rob.groupby(["factor", "level", "variant"])["SSIM"].mean().unstack()
    print(g.round(3)); print(g.agg(["mean", "min"]).round(3))
    print(busi.groupby("method")[["CNR", "gCNR", "SNR_bg", "BCR", "Dice", "structure_frac"]].mean().round(3))
    # paired tests of AS-NLM-c against every method from the main BUSI run
    main = pd.read_csv(os.path.join(RESULTS, "busi_per_image.csv"))
    c = busi[busi["method"] == "AS-NLM-c"].set_index("image")
    tests = []
    for meth in main["method"].unique():
        o = main[main["method"] == meth].set_index("image").loc[c.index]
        row = {"vs": meth}
        for m in ("gCNR", "CNR", "BCR", "Dice", "SNR_bg"):
            d = c[m] - o[m]
            row[f"{m} diff"] = d.mean()
            row[f"{m} p"] = wilcoxon(c[m], o[m]).pvalue if (d != 0).any() else 1.0
        tests.append(row)
    t = pd.DataFrame(tests)
    t.to_csv(os.path.join(RESULTS, "posthoc_busi_wilcoxon.csv"), index=False)
    print(t.round(4).to_string())
