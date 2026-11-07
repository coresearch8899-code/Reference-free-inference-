# -*- coding: utf-8 -*-
"""Step F11: (a) two ways of handling a level shift, on the same scenarios; (b) the whole procedure repeated in each replicate:
season and thresholds learned from that replicate's development data only, then frozen before its evaluation data are scored."""
import os
import numpy as np
import pandas as pd
from detector import loo, remove_season, step_search, pipeline, calibrate, decide
from injection import RES, SUF, DEV, X, noise, thresholds, run

rng = np.random.default_rng(20261111)
NREP = int(os.environ.get("NREP", 1000)); BOOT = int(os.environ.get("BOOT", 1000))
FP = int(os.environ.get("FP", 500)); FPB = int(os.environ.get("FPB", 300))

# ---- (a) level shifts, N = 3, 18 months of evaluation data
test = 78; sc = np.ones(3); q = thresholds(test, 3, sc, BOOT, rng); cache = {}
SC = [("Step 0.2, no drift, 48 weeks after it", None, (1, 30, 0.2)), ("Step 0.1, no drift, 48 weeks after it", None, (1, 30, 0.1)), ("Step 0.2, no drift, 18 weeks after it", None, (1, 60, 0.2)),
      ("Drift 5 % per year, no step", {0: -5}, None), ("Drift 2 % per year, no step", {0: -2}, None), ("Step 0.2 and drift 5 % per year, same sensor", {1: -5}, (1, 30, 0.2)), ("Step 0.2 and drift 5 % per year, different sensors", {0: -5}, (1, 30, 0.2))]
rows = []
for lab, dr, sh in SC:
    for meth, name in (("remove", "Step subtracted, then slope"), ("joint", "Step and slope estimated jointly")):
        rows.append(dict(scenario=lab, method=name, **run(test, 3, sc, q, rng, NREP, dr, sh, method=meth, cache=cache)))
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
pd.DataFrame(rows).to_csv(os.path.join(RES, f"step_comparison{SUF}.csv"), index=False)

# ---- (b) whole procedure in each replicate, N = 3, 18 months of development data
TESTS = (78, 104); DR = (0, 2, 5); N = 3; rec = []
for i in range(FP):
    L = noise(DEV + max(TESTS), N, sc, rng)
    Cd, _ = step_search(remove_season(loo(L[:DEV]), X[:DEV], DEV))
    E = Cd - Cd.mean(0)
    for test in TESTS:
        qi = calibrate(E, DEV, test, X, rng, B=FPB)
        ramp = np.zeros(DEV + test); ramp[DEV:] = np.arange(test) / 52.18
        for D in DR:
            Lx = L[:DEV + test].copy(); Lx[:, 0] += -D / 100 * ramp
            r = pipeline(loo(Lx), DEV, test, X)
            st, kk = decide(r, qi)
            truth = {0} if D > 0 else set()                   # with no injected drift nothing can be named correctly
            rec.append(dict(rep=i, test=test, drift=D, threshold=qi["s"], state=st, inconsistency=int(st > 0 or len(r["inwin"]) > 0), right=int(st == 2 and kk in truth), wrong=int(st == 2 and kk not in truth)))
    if i % 50 == 0:
        print(i, flush=True)
R = pd.DataFrame(rec); R.to_csv(os.path.join(RES, f"full_process_runs{SUF}.csv"), index=False)
out = []; GRID = pd.read_csv(os.path.join(RES, f"calibrated_grid{SUF}.csv"))     # noise-model thresholds: the ones of the main injection table
for (t_, D), g in R.groupby(["test", "drift"]):
    p = (g.state > 0).mean()
    q0 = float(GRID[(GRID.N == N) & (GRID.months == round(t_ / 4.348)) & (GRID.drift == 0)].s_threshold.iloc[0]) if D == 0 else np.nan
    out.append(dict(months=round(t_ / 4.348), drift=D, replicates=len(g), detected=p, inconsistency=g.inconsistency.mean(), se=np.sqrt(p * (1 - p) / len(g)), ambiguous=(g.state == 1).mean(), right=g.right.mean(), wrong=g.wrong.mean(),
                    threshold_mean=g.threshold.mean(), threshold_sd=g.threshold.std(), threshold_p5=g.threshold.quantile(.05), threshold_p95=g.threshold.quantile(.95), oracle_threshold=q0))
S = pd.DataFrame(out); S.to_csv(os.path.join(RES, f"full_process{SUF}.csv"), index=False)
print(S.round(3).to_string())
