"""Step 4: ablation study of the proposed AS-NLM.

Removes or replaces one component at a time to show what each stage adds.
Uses the same 12 held-out test phantoms as the main benchmark.
"""
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import F, RESULTS, tuned_params
from speckle import simulate as S, metrics as M

TEST_SET = [("cyst", s) for s in (1, 2, 3, 4)] + [("organ", s) for s in (21, 22, 23, 24)] \
    + [("edges", s) for s in (31, 32, 33, 34)]
P = tuned_params()["AS-NLM (proposed)"]
BMAX = P.get("beta_max", 3.0)


def variants():
    full = dict(P)
    return {
        "A. Fixed-strength NLM (beta = beta_max)": lambda u: F.obnlm(u, beta=BMAX),
        "B. + adaptive beta(x) [Stage 1-2]": lambda u: F.asnlm(u, **{**full, "diff_iter": 0}),
        "C. + ungated linear diffusion": lambda u: _ungated(u, full),
        "D. + gated diffusion, no re-test [no Stage 3]": lambda u: F.asnlm(u, **{**full, "refine_lo": 1e6, "refine_hi": 2e6}),
        "E. + gated diffusion, no edge-stop (kappa off)": lambda u: F.asnlm(u, **{**full, "kappa_rel": 0}),
        "F. Fixed beta + full gating (no adaptive NLM)": lambda u: F.asnlm(u, **{**full, "beta_min": BMAX}),
        "G. Full AS-NLM": lambda u: F.asnlm(u, **full),
    }


def _ungated(u, p):
    s2 = F.estimate_sigma2_loupas(u)
    _, w = F.structure_map(u, s2)
    beta = BMAX - (BMAX - 0.5) * w
    nlm = F._nlm_core(u, 2 * beta * s2)
    return np.clip(F.gated_diffusion(nlm, np.ones_like(u), p.get("diff_iter", 150), kappa=None), 0, 1)


def run(job):
    kind, seed = job
    b, ref, _ = S.simulate_bmode(S.phantom(kind, seed=seed), seed=seed)
    return [{"phantom": kind, "seed": seed, "variant": n, **M.full_reference(ref, fn(b))}
            for n, fn in variants().items()]


if __name__ == "__main__":
    with Pool(os.cpu_count()) as pool:
        res = pool.map(run, TEST_SET, chunksize=1)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv(os.path.join(RESULTS, "ablation.csv"), index=False)
    print(df.groupby("variant", sort=False)[["PSNR", "SSIM", "FOM", "EPI"]].mean().round(3).to_string())
