# -*- coding: utf-8 -*-
"""Step F8: walk-forward evaluation on the NIST networks. For each fold, the season and all thresholds are fixed on the development weeks
and the detector is scored on the following weeks, which played no part in calibration. The folds overlap, so counts are not independent."""
import json, os, re
import numpy as np
import pandas as pd
from detector import harmonics, loo, signature, remove_season, step_search, pipeline, calibrate, conditional, decide, nominal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
for p in (os.path.join(ROOT, "data", "nist_daily.csv"), os.path.join(ROOT, "data", "nist_daily.csv")):
    if os.path.exists(p):
        D = pd.read_csv(p, parse_dates=["date"])
NAME = {4901: "C", 4902: "G", 4903: "R"}
fr = {}
for s, g in D.groupby("system"):
    g = g.set_index("date").sort_index()
    for c in g.columns:
        m = re.match(r"irradiance_(?:poa|ghi)_o_(\d+)__kwh$", c)
        if m:
            v = g[c].where(g[c.replace("__kwh", "__cov")] > 0.95)
            fr[f"{NAME[s]}{m.group(1)}"] = v.where(v > 0.15 * v.quantile(0.9))
LOG = np.log(pd.DataFrame(fr).sort_index())
GROUPS = {"Roof, six identical pyranometers": ("co-located", ["R2206", "R2207", "R2209", "R2210", "R2212", "R2213"]),
          "Roof, three sensor types": ("co-located", ["R2209", "R2211", "R2215"]),
          "Campus, one pyranometer per array": ("separated", ["C2196", "G2202", "R2206"])}
B = int(os.environ.get("BOOT", 1000))
rng = np.random.default_rng(20261108)
rows, inj = [], []
for name, (kind, g) in GROUPS.items():
    W = (LOG[g] - LOG[g].mean()).resample("7D").mean().interpolate(limit_direction="both")
    doy = W.index.dayofyear.values
    X = harmonics(doy, 1)
    L = W.values
    C0 = loo(L)
    n = len(C0); N = len(g); F = signature(N)
    for test in (52, 78, 104):
        for dev in range(52, n - test + 1, 13):
            Cdev = remove_season(C0[:dev], X[:dev], dev)
            Cdev, st_dev = step_search(Cdev)
            E = Cdev - Cdev.mean(0)
            q = calibrate(E, dev, test, X, rng, B=B); cache = {}
            base = pipeline(C0, dev, test, X)
            q_b = conditional(base, q, E, dev, test, X, rng, B, cache)
            s_c, k_c = decide(base, q_b); s_n, k_n = nominal(base)
            old = pipeline(C0, dev, test, X, method="remove")          # same fold, step subtracted first, for comparison
            rows.append(dict(network=name, kind=kind, N=N, test_weeks=test, dev_weeks=dev, s_threshold=q["s"], S=base["S"], state_cal=s_c, sensor_cal=g[k_c] if k_c >= 0 else "",
                             state_nom=s_n, sensor_nom=g[k_n] if k_n >= 0 else "", inconsistency_cal=int(s_c > 0 or len(base["inwin"]) > 0), inconsistency_nom=int(s_n > 0 or len(base["inwin"]) > 0), state_cal_subtract=decide(old, q)[0], state_nom_subtract=nominal(old)[0], unresolved=int(base["unresolved"]), steps_in_window=len(base["inwin"]), steps=len(base["steps"]), steps_detail=str([(g[k], str(W.index[t].date()), round(d, 2)) for k, t, d in base["steps"]])))
            ramp = np.zeros(n); ramp[dev:dev + test] = np.arange(test) / 52.18
            for rate in (3, 5, 10):
                for k in range(N):
                    Ci = C0 - (rate / 100) * np.outer(ramp, F[:, k])
                    Ci[dev + test:] = C0[dev + test:]
                    r = pipeline(Ci, dev, test, X)
                    for meth, (st, kk) in (("calibrated", decide(r, conditional(r, q, E, dev, test, X, rng, B, cache))), ("nominal", nominal(r))):
                        inj.append(dict(network=name, kind=kind, N=N, test_weeks=test, dev_weeks=dev, rate=rate, sensor=k, method=meth, state=st, unresolved=int(r["unresolved"] and meth == "calibrated"), inconsistency=int(st > 0 or len(r["inwin"]) > 0), correct=int(st == 2 and kk == k), wrong=int(st == 2 and kk != k)))
    print(name, flush=True)
R = pd.DataFrame(rows); I = pd.DataFrame(inj)
R.to_csv(os.path.join(RES, "nist_walkforward_null.csv"), index=False); I.to_csv(os.path.join(RES, "nist_walkforward_inj.csv"), index=False)
summ = []
for (nm, kd, tw), q in R.groupby(["network", "kind", "test_weeks"], sort=False):
    for meth, col in (("Nominal", "state_nom"), ("Calibrated", "state_cal")):
        ii = I[(I.network == nm) & (I.test_weeks == tw) & (I.method == meth.lower())]
        row = dict(network=nm, kind=kd, test_weeks=tw, method=meth, folds=len(q), alarm_count=int((q[col] > 0).sum()), alarm_count_subtract=int((q[col.replace("state_nom", "state_nom_subtract").replace("state_cal", "state_cal_subtract")] > 0).sum()), unresolved_count=int(q.unresolved.sum()) if meth == "Calibrated" else 0, folds_step_in_window=int((q.steps_in_window > 0).sum()), inconsistency_count=int(q["inconsistency_" + ("cal" if meth == "Calibrated" else "nom")].sum()), alarms_no_injection=float((q[col] > 0).mean()), median_threshold=float(q.s_threshold.median()) if meth == "Calibrated" else np.nan)
        for rate in (3, 5, 10):
            j = ii[ii.rate == rate]
            row.update({f"gradual_{rate}": float(((j.state > 0) & (j.unresolved == 0)).mean()), f"unresolved_{rate}": float(j.unresolved.mean()), f"inconsistency_{rate}": float(j.inconsistency.mean()), f"detected_{rate}": float((j.state > 0).mean()), f"correct_{rate}": float(j.correct.mean()), f"wrong_{rate}": float(j.wrong.mean()), f"ambiguous_{rate}": float((j.state == 1).mean())})
        summ.append(row)
S = pd.DataFrame(summ); S.to_csv(os.path.join(RES, "nist_walkforward_summary.csv"), index=False)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
print(S.round(2).to_string())
