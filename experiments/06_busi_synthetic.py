"""Step 6: real anatomy + controlled synthetic speckle.

Take BUSI images as the 'reference', add extra speckle with the Loupas
display-domain model I = R + sqrt(R) n at three noise levels, filter, and
measure how close we get back to the original. This is the protocol used in
many despeckling papers. Caveat (discussed in the report): the reference
already contains the scanner's own speckle, so it rewards fidelity to the
original image rather than true speckle removal.
"""
import os
import zlib
from multiprocessing import Pool

import pandas as pd

from common import RESULTS, load_busi, tuned_filters
from speckle import simulate as S, metrics as M

SIGMAS = (0.1, 0.2, 0.3)


def run(job):
    (name, img, _), sigma = job
    noisy = S.add_multiplicative_speckle(img, sigma=sigma, model="loupas", seed=zlib.crc32(name.encode()) % 1000)
    return [{"image": name, "sigma": sigma, "method": meth, **M.full_reference(img, fn(noisy))}
            for meth, fn in {"Noisy": lambda x: x, **tuned_filters()}.items()]


if __name__ == "__main__":
    data = load_busi(n_per_class=6, seed=1)
    jobs = [(d, s) for d in data for s in SIGMAS]
    with Pool(os.cpu_count()) as pool:
        res = pool.map(run, jobs, chunksize=1)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv(os.path.join(RESULTS, "busi_synthetic.csv"), index=False)
    print(df.groupby(["sigma", "method"], sort=False)[["PSNR", "SSIM", "EPI"]].mean().unstack(0).round(3).to_string())
