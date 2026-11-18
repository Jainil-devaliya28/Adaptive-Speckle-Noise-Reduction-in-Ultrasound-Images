"""Step 1: tune every filter on a held-out tuning set of simulated phantoms.

Selection criterion: 0.5 * SSIM + 0.5 * FOM (balances speckle suppression
against edge fidelity, so no method wins by simply blurring everything).
Writes results/tuned_params.json and results/tuning_log.csv.
"""
import itertools
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import F, GRIDS, RESULTS
from speckle import simulate as S, metrics as M

TUNE_SET = [("organ", 101), ("organ", 102), ("edges", 103), ("cyst", 104)]
_DATA = None


def _init():
    global _DATA
    _DATA = []
    for kind, seed in TUNE_SET:
        b, ref, _ = S.simulate_bmode(S.phantom(kind, seed=seed), seed=seed)
        _DATA.append((b, ref))


def _eval(job):
    name, params = job
    fn = F.FILTERS[name]
    scores = []
    t0 = time.time()
    for b, ref in _DATA:
        out = fn(b, **params)
        scores.append(M.full_reference(ref, out))
    row = {k: float(np.mean([s[k] for s in scores])) for k in scores[0]}
    row.update(method=name, params=json.dumps(params), time_s=(time.time() - t0) / len(_DATA))
    row["objective"] = 0.5 * row["SSIM"] + 0.5 * row["FOM"]
    return row


if __name__ == "__main__":
    import sys
    only = sys.argv[1:]  # optionally retune just some methods
    jobs = []
    for name, grid in GRIDS.items():
        if only and name not in only:
            continue
        keys = list(grid)
        for vals in itertools.product(*grid.values()):
            jobs.append((name, dict(zip(keys, vals))))
    print(f"{len(jobs)} configurations")
    with Pool(os.cpu_count(), initializer=_init) as pool:
        rows = pool.map(_eval, jobs, chunksize=1)
    df = pd.DataFrame(rows)
    log = os.path.join(RESULTS, "tuning_log.csv")
    if only and os.path.exists(log):
        old = pd.read_csv(log)
        df_all = pd.concat([old[~old["method"].isin(only)], df])
    else:
        df_all = df
    df_all.to_csv(log, index=False)
    p = os.path.join(RESULTS, "tuned_params.json")
    best = json.load(open(p)) if (only and os.path.exists(p)) else {}
    for name, g in df.groupby("method", sort=False):
        r = g.loc[g["objective"].idxmax()]
        best[name] = json.loads(r["params"])
        print(f"{name:20s} {r['params']:45s} SSIM={r['SSIM']:.3f} FOM={r['FOM']:.3f}")
    json.dump(best, open(os.path.join(RESULTS, "tuned_params.json"), "w"), indent=2)
