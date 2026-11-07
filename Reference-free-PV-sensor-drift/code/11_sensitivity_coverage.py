# -*- coding: utf-8 -*-
"""Step F10: sensitivity of the recovery ratio (leave-one-event-out, open final segment, incomplete cleaning, real unequal spacing),
empirical coverage of the state-space intervals by stratum, and hyper-parameter table."""
import json, os, pickle, sys
import numpy as np
import pandas as pd


from ss_data import load_panel
from ss_model import harm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results"); R14RES = RES
rng = np.random.default_rng(20261110)
P = load_panel(); n = P["n"]; H = np.stack([harm(x) for x in P["doy"]]); h, y = P["h"], P["y"]; ev = P["events"]
tt = np.arange(n, dtype=float)
CL = ["C2", "C3", "C4", "C5", "C6"]


def fit(series, resets, last=None):
    ok = np.isfinite(series) & (tt <= (last if last is not None else n))
    t = tt[ok]; r = np.sort(resets)
    seg = np.searchsorted(r, t, side="right"); K = len(r) + 1
    D = np.zeros((len(t), K)); D[np.arange(len(t)), seg] = 1
    X = np.column_stack([D, t / 365.25, H[ok]])          # same design as f4_reset_decision.py
    b = np.linalg.lstsq(X, series[ok], rcond=None)[0]
    span = (t[-1] - t[0]) / 365.25
    G = b[K - 1] - b[0]
    return dict(within=float(b[K] * 100), recovered=float(G * 100), net=float((b[K] + G / span) * 100), ratio=float(G / (-b[K] * span)))


rows = []
for lab, s in (("Sensor contrast", h[:, 0] - h[:, 2]), ("Yield ratio", y[:, 0] - y[:, 2])):
    rk = [ev[c] for c in CL]
    rows.append(dict(signal=lab, case="All five cleanings", **fit(s, rk)))
    for c in CL:
        rows.append(dict(signal=lab, case=f"Without {c}", **fit(s, [ev[x] for x in CL if x != c])))
    rows.append(dict(signal=lab, case="Open final segment removed", **fit(s, rk, last=ev["C6"])))
R = pd.DataFrame(rows); R.to_csv(os.path.join(RES, "reset_sensitivity.csv"), index=False)
print(R.round(2).to_string())

# traceable inputs of the cleaning scenarios: slope of the sensor-free log yield ratio between campaigns, with Newey-West intervals
def seg_slope(a, b):
    s = (y[:, 0] - y[:, 2]).copy(); ok = np.isfinite(s) & (tt > a) & (tt < b)
    x = tt[ok] - tt[ok].mean(); u0 = s[ok] - s[ok].mean(); sxx = (x ** 2).sum(); bb = (x * u0).sum() / sxx
    g = x * (u0 - bb * x); S = (g ** 2).sum()
    for k in range(1, 31):
        S += 2 * (1 - k / 31) * (g[k:] * g[:-k]).sum()
    se = np.sqrt(S) / sxx
    return dict(slope_pct_day=float(bb * 100), ci=[float((bb - 1.96 * se) * 100), float((bb + 1.96 * se) * 100)], n=int(ok.sum()))


UNIT = 10000 / 31490                                   # THB per W per campaign, Mongkoldhumrongkul (2023)
json.dump(dict(dry_C5_C6=seg_slope(ev["C5"], ev["C6"]), monsoon_C4_C5=seg_slope(ev["C4"], ev["C5"]), unit_cost_thb_per_w=UNIT, zone1_kwp=2256.3, campaign_cost_kthb=UNIT * 2256.3),
          open(os.path.join(RES, "soiling_inputs.json"), "w"), indent=1)
print(open(os.path.join(RES, "soiling_inputs.json")).read())

# incomplete cleaning, with the real unequal spacing of the logged cleanings
pool = pd.read_csv(os.path.join(RES, "residual_pools.csv"), index_col=0, parse_dates=True)
x = pool[pool.index < "2025-01-01"]["sensor_contrast_h1_h3"].values; x = x[np.argmax(np.isfinite(x)):]
m = len(x); reset = np.array([ev[c] for c in CL], float)
sim = []
for L in (3, 5, 10):
    for eta in (1.0, 0.75, 0.5, 0.25):
        for frac_drift in (0.0, 0.5):
            d = frac_drift * L / 100 / 365.25; r = (1 - frac_drift) * L / 100 / 365.25
            s = np.zeros(n); s[0] = 0
            for t in range(1, n):
                s[t] = s[t - 1] - r
                if t in reset:
                    s[t] += eta * (-(s[t - 1]))              # a fraction eta of the accumulated reversible loss is removed
            sig = s - d * tt
            res = []
            for _ in range(300):
                idx = (rng.integers(0, m) + np.arange(n)) % m
                st = np.where(rng.random(n) < 1 / 60)[0]
                e = x[idx].copy()
                for a in st:
                    e[a:a + 60] = x[(rng.integers(0, m) + np.arange(min(60, n - a))) % m][:len(e[a:a + 60])]
                res.append(fit(sig + e, reset.astype(int)))
            q = pd.DataFrame(res)
            sim.append(dict(loss=L, eta=eta, true_drift=frac_drift * L, ratio=q.ratio.median(), net=q.net.median(), expected_ratio_eq6=eta * (1 - frac_drift), expected_net_eq6=-(frac_drift * L + (1 - eta) * (1 - frac_drift) * L)))
S = pd.DataFrame(sim); S.to_csv(os.path.join(RES, "reset_incomplete.csv"), index=False)
print(S.round(2).to_string())

# empirical coverage of the state-space intervals, by stratum (hyper-parameters held at their estimates)
V = pd.read_csv(os.path.join(R14RES, "validation_semisynthetic.csv"))
V["abs_drift"] = pd.qcut(V.drift_true.abs(), 3, labels=["smallest third", "middle third", "largest third"])
V["clip"] = pd.qcut(V.trunc_share_z1_post, 3, labels=["fewest clipped days", "middle third", "most clipped days"])
cov = []
for by, lab in (("abs_drift", "Size of the injected drift"), ("clip", "Share of clipped days")):
    for k, g in V.groupby(by, observed=True):
        cov.append(dict(stratum=lab, level=str(k), n=len(g), drift_coverage=float(g.cover_tobit.mean()), final_sensitivity_coverage=float(g.sigend_cover_tobit.mean()), clipped_share=float(g.trunc_share_z1_post.mean())))
cov.append(dict(stratum="All", level="", n=len(V), drift_coverage=float(V.cover_tobit.mean()), final_sensitivity_coverage=float(V.sigend_cover_tobit.mean()), clipped_share=float(V.trunc_share_z1_post.mean())))
C = pd.DataFrame(cov); C.to_csv(os.path.join(RES, "coverage_strata.csv"), index=False)
print(C.round(2).to_string())
th = pickle.load(open(os.path.join(R14RES, "mssm_main.pkl"), "rb"))["theta"]
json.dump(th, open(os.path.join(RES, "hyperparameters.json"), "w"), indent=1)
