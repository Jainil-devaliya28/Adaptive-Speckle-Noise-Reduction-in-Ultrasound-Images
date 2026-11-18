"""Step 5: evaluation on real clinical images (BUSI breast ultrasound dataset).

Real scans have no speckle-free ground truth, so we use the radiologist
lesion masks that ship with BUSI to define regions and measure:
  * CNR and gCNR     lesion vs a surrounding ring of tissue (detectability)
  * SNR_bg           mean/std in the ring (speckle suppression)
  * BCR              boundary contrast ratio: mean gradient on the annotated
                     lesion boundary / mean gradient in the ring. Above 1 means
                     the true boundary stands out from residual texture.
  * Dice             a simple, fixed segmentation (Otsu inside a box around the
                     lesion) run on each filtered image, scored against the
                     radiologist mask. Shows whether despeckling helps a
                     downstream task, not only how the image looks.
Parameters are the ones tuned on simulation: no tuning on real data.
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import wilcoxon
from skimage.filters import threshold_otsu

from common import RESULTS, load_busi, tuned_filters
from speckle import metrics as M

N_PER_CLASS = int(os.environ.get("N_PER_CLASS", 50))
EXAMPLES = 4


def segment(img, mask, margin=0.25):
    """Box-prompted Otsu segmentation of a hypoechoic lesion."""
    zz, xx = np.where(mask)
    z0, z1, x0, x1 = zz.min(), zz.max(), xx.min(), xx.max()
    dz, dx = int((z1 - z0) * margin) + 5, int((x1 - x0) * margin) + 5
    z0, x0 = max(z0 - dz, 0), max(x0 - dx, 0)
    z1, x1 = min(z1 + dz, img.shape[0] - 1), min(x1 + dx, img.shape[1] - 1)
    crop = img[z0:z1 + 1, x0:x1 + 1]
    seg = crop < threshold_otsu(crop)
    seg = ndi.binary_opening(seg, iterations=2)
    lab, n = ndi.label(seg)
    if n == 0:
        return np.zeros_like(mask)
    # keep the component closest to the box centre (the prompt's intent)
    cz, cx = (crop.shape[0] - 1) / 2, (crop.shape[1] - 1) / 2
    best, bd = 0, np.inf
    for k in range(1, n + 1):
        pz, px = np.where(lab == k)
        if len(pz) < 30:
            continue
        d = np.min(np.hypot(pz - cz, px - cx))
        if d < bd:
            best, bd = k, d
    out = np.zeros_like(mask)
    if best:
        out[z0:z1 + 1, x0:x1 + 1] = ndi.binary_fill_holes(lab == best)
    return out


def dice(a, b):
    return 2 * (a & b).sum() / (a.sum() + b.sum() + 1e-9)


def run(item):
    name, img, mask = item
    inner, ring = M.lesion_rois(mask)
    if inner.sum() < 50 or ring.sum() < 50:
        return [], None
    band = ndi.binary_dilation(mask, iterations=2) & ~ndi.binary_erosion(mask, iterations=2)
    rows, outs = [], {"Input": img}
    for meth, fn in {"Noisy": lambda x: x, **tuned_filters()}.items():
        t0 = time.time()
        o = fn(img)
        dt = time.time() - t0
        g = np.hypot(ndi.sobel(o, 0), ndi.sobel(o, 1))
        rows.append({
            "image": name, "class": name.split(" ")[0], "method": meth, "time_s": dt,
            "CNR": M.cnr(o, inner, ring), "gCNR": M.gcnr(o, inner, ring),
            "SNR_bg": M.snr_roi(o, ring),
            "BCR": float(g[band].mean() / (g[ring & ~band].mean() + 1e-9)),
            "Dice": dice(segment(o, mask), mask),
        })
        outs[meth] = o
    return rows, outs


if __name__ == "__main__":
    data = load_busi(n_per_class=N_PER_CLASS, seed=0)
    data = [d for d in data if d[2] is not None]
    print(f"{len(data)} images", flush=True)
    with Pool(os.cpu_count()) as pool:
        res = pool.map(run, data, chunksize=1)
    df = pd.DataFrame([r for rr, _ in res for r in rr])
    df.to_csv(os.path.join(RESULTS, "busi_per_image.csv"), index=False)
    # example images for figures (two benign, two malignant)
    picked = {"benign": 0, "malignant": 0}
    for (name, img, mask), (_, outs) in zip(data, res):
        c = name.split(" ")[0]
        if outs is not None and picked[c] < EXAMPLES // 2 and 1500 < mask.sum() < 40000:
            np.savez_compressed(os.path.join(RESULTS, f"busi_example_{name.replace(' ', '_')}.npz"),
                                mask=mask, **{k: v.astype(np.float32) for k, v in outs.items()})
            picked[c] += 1
    cols = ["CNR", "gCNR", "SNR_bg", "BCR", "Dice", "time_s"]
    summ = df.groupby("method", sort=False)[cols].agg(["mean", "std"])
    summ.columns = [f"{a} {b}" for a, b in summ.columns]
    summ.to_csv(os.path.join(RESULTS, "busi_summary.csv"))
    print(df.groupby("method", sort=False)[cols].mean().round(3).to_string())
    # paired Wilcoxon tests: proposed vs every other method
    prop = "AS-NLM (proposed)"
    tests = []
    piv = {m: df.pivot(index="image", columns="method", values=m) for m in ("gCNR", "Dice", "BCR")}
    for meth in df["method"].unique():
        if meth == prop:
            continue
        row = {"vs": meth}
        for m, p in piv.items():
            d = p[prop] - p[meth]
            row[f"{m} diff"] = d.mean()
            row[f"{m} p"] = wilcoxon(p[prop], p[meth]).pvalue if (d != 0).any() else 1.0
        tests.append(row)
    tdf = pd.DataFrame(tests)
    tdf.to_csv(os.path.join(RESULTS, "busi_wilcoxon.csv"), index=False)
    print(tdf.round(4).to_string())
