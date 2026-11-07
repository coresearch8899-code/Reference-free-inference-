# -*- coding: utf-8 -*-
"""Step F4: (a) reset criterion: recovery ratio separating permanent loss (drift) from reversible loss (soiling);
(b) application to the case study; (c) independent satellite check; (d) bias of the optimal cleaning interval."""
import json, os, sys
import numpy as np
import pandas as pd

from ss_data import load_panel
from ss_model import harm
from ss_data import load_power

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
rng = np.random.default_rng(20261107)
NREP = int(os.environ.get("NREP", 2000))
pool = pd.read_csv(os.path.join(RES, "residual_pools.csv"), index_col=0, parse_dates=True)
pool = pool[pool.index < "2025-01-01"]


def sbb(x, n, reps, block=60):
    x = x[np.argmax(np.isfinite(x)):]
    m = len(x); idx = np.empty((reps, n), dtype=np.int32); idx[:, 0] = rng.integers(0, m, reps)
    for t in range(1, n):
        idx[:, t] = np.where(rng.random(reps) < 1 / block, rng.integers(0, m, reps), (idx[:, t - 1] + 1) % m)
    return x[idx]


def weekly(a):
    n = a.shape[-1] // 7
    w = a[..., :n * 7].reshape(a.shape[:-1] + (n, 7))
    c = np.sum(np.isfinite(w), -1)
    m = np.where(c > 0, np.nansum(w, -1) / np.maximum(c, 1), np.nan)
    return np.where(np.isfinite(m), m, np.nanmean(m, -1, keepdims=True))


def design(t, resets, extra=None):
    """segment intercepts + common slope; returns X and the contrast giving the sum of steps"""
    seg = np.searchsorted(np.asarray(resets), t, side="right")
    K = len(resets) + 1
    D = np.zeros((len(t), K)); D[np.arange(len(t)), seg] = 1
    X = np.column_stack([D, t / 365.25] + ([extra] if extra is not None else []))
    w_steps = np.zeros(X.shape[1]); w_steps[0] = -1; w_steps[K - 1] = 1          # sum of steps = last - first intercept
    return X, w_steps, K


def recovery(y, t, resets, extra=None):
    """returns slope within segments, total recovered, net slope (% /yr) and recovery ratio for rows of y (days along columns)"""
    ok = np.isfinite(y).all(0) if y.ndim == 2 else np.isfinite(y)
    X, w, K = design(t[ok], resets, None if extra is None else extra[ok])
    pin = np.linalg.pinv(X)
    b = (pin @ (y[:, ok].T if y.ndim == 2 else y[ok])).T
    slope = b[..., K]
    span = (t[ok][-1] - t[ok][0]) / 365.25
    steps = b @ w
    net = slope + steps / span
    return slope * 100, steps * 100, net * 100, steps / np.where(np.abs(slope * span) > 1e-9, -slope * span, np.nan)


# ---- (a) simulation: can the ratio tell drift from soiling?
r0 = pool["sensor_contrast_h1_h3"].values
rows = []
for Tm in (12, 24, 36):
    T = int(Tm * 30.44); t = np.arange(T, dtype=float)
    for per_year in (2, 3, 4):
        gap = 365.25 / per_year
        resets = np.arange(gap, T, gap)
        tw = 3.0 + 7 * np.arange(T // 7)
        for L in (2, 3, 5, 10):                      # total loss rate, % per year
            E = sbb(r0, T, NREP)
            d = -L / 100 / 365.25
            drift = d * t
            saw = d * (t - np.concatenate([[0], resets])[np.searchsorted(resets, t, side="right")])
            res = {}
            for name, sig in (("drift", drift), ("soiling", saw), ("half", 0.5 * drift + 0.5 * saw)):
                s, st, n, rho = recovery(weekly(E + sig), tw, resets)
                res[name] = (n, rho)
            perm_d = res["drift"][0]; perm_s = res["soiling"][0]
            acc = 0.5 * (np.mean(perm_d < -L / 2) + np.mean(perm_s > -L / 2))
            rows.append(dict(T_months=Tm, resets_per_year=per_year, loss_pct_yr=L, accuracy=float(acc),
                             rho_drift=float(np.nanmedian(res["drift"][1])), rho_soiling=float(np.nanmedian(res["soiling"][1])),
                             net_half=float(np.median(res["half"][0]))))
rc = pd.DataFrame(rows); rc.to_csv(os.path.join(RES, "reset_classifier.csv"), index=False)
print(rc.groupby(["T_months", "resets_per_year"]).accuracy.mean().unstack().round(3))

# ---- (b) case study
P = load_panel(); d = P["date"]; n = P["n"]
H = np.stack([harm(x) for x in P["doy"]])
h, y = P["h"], P["y"]
ev = P["events"]
clean = [ev[k] for k in ("C2", "C3", "C4", "C5", "C6")]             # logged cleanings with data on both sides
tt = np.arange(n, dtype=float)
out = {}


def boot(series, resets, tag, B=2000):
    ok = np.isfinite(series)
    X, w, K = design(tt[ok], resets, H[ok])
    pin = np.linalg.pinv(X)
    b = pin @ series[ok]
    fit = X @ b; u = series[ok] - fit
    span = (tt[ok][-1] - tt[ok][0]) / 365.25
    res_est = dict(within_pct_yr=float(b[K] * 100), recovered_pct=float(b @ w * 100), net_pct_yr=float((b[K] + b @ w / span) * 100),
                   recovery_ratio=float(b @ w / (-b[K] * span)))
    # moving-block bootstrap of the residuals
    m = len(u); bl = 30
    st = rng.integers(0, m - bl, (B, m // bl + 1))
    idx = (st[..., None] + np.arange(bl)).reshape(B, -1)[:, :m]
    ys = fit + u[idx]
    bb = (pin @ ys.T).T
    nets = (bb[:, K] + bb @ w / span) * 100
    rat = (bb @ w) / (-bb[:, K] * span)
    res_est.update(net_ci=[float(x) for x in np.percentile(nets, [2.5, 97.5])],
                   ratio_ci=[float(x) for x in np.percentile(rat, [2.5, 97.5])], n=int(ok.sum()))
    out[tag] = res_est
    print(tag, {k: (np.round(v, 2).tolist() if not isinstance(v, int) else v) for k, v in res_est.items()})


boot(h[:, 0] - h[:, 2], clean, "sensor_contrast_1_3_all_cleanings")
boot(y[:, 0] - y[:, 2], clean, "yield_ratio_1_3_all_cleanings")
late = d >= np.datetime64("2025-01-01")
c = (h[:, 0] - h[:, 2]).copy(); c[~late] = np.nan
boot(c, [ev["C4"], ev["C5"], ev["C6"]], "sensor_contrast_1_3_from_2025")
c = (y[:, 0] - y[:, 2]).copy(); c[~late] = np.nan
boot(c, [ev["C4"], ev["C5"], ev["C6"]], "yield_ratio_1_3_from_2025")

# ---- (c) independent satellite reference, same days for both sensors
pw = load_power(d)
sat = np.log(pw.ALLSKY_SFC_SW_DWN.values)
for i in (0, 2):
    for lab, m in (("pre2025", ~late), ("from2025", late)):
        s = np.where(m, h[:, i] - sat, np.nan); ok = np.isfinite(s)
        X = np.column_stack([np.ones(n), H, tt / 365.25])[ok]
        b = np.linalg.lstsq(X, s[ok], rcond=None)[0]; u = s[ok] - X @ b
        XtXi = np.linalg.inv(X.T @ X)
        g = X * u[:, None]; S = g.T @ g
        for k in range(1, 31):
            S = S + (1 - k / 31) * (g[k:].T @ g[:-k] + g[:-k].T @ g[k:])
        se = np.sqrt((XtXi @ S @ XtXi)[-1, -1])
        out[f"sat_sensor{i+1}_{lab}"] = dict(slope_pct_yr=float(b[-1] * 100), ci=[float((b[-1] - 1.96 * se) * 100), float((b[-1] + 1.96 * se) * 100)], n=int(ok.sum()))
        print(f"sensor {i+1} vs CERES {lab}: {b[-1]*100:+.1f} ({(b[-1]-1.96*se)*100:+.1f}, {(b[-1]+1.96*se)*100:+.1f}) %/yr, n={ok.sum()}")
ann = pd.Series(pw.ALLSKY_SFC_SW_DWN.values / pw.CLRSKY_SFC_SW_DWN.values, index=d)
ann = np.log(ann[(ann.index.month.isin([12, 1, 2, 3]))]).groupby(lambda x: x.year if x.month > 6 else x.year - 1).mean()
out["ceres_dry_season_clearness_by_year"] = {str(k): float(v) for k, v in ann.items()}

# ---- (d) bias of the cleaning interval: T*^ / T* = sqrt(r / (r - d)), cost excess = (x + 1/x)/2 - 1 with x = sqrt(rhat / r)
rate = {"R13 dry": 0.31, "R13 wet": 0.044, "MSSM dry": 0.180, "MSSM wet": 0.039}
T0 = {"R13 dry": 110, "R13 wet": 285}
tb = []
for dr_yr in (2, 5, 8.9, 12.9):
    dd = dr_yr / 365.25
    for k, r in rate.items():
        rh = r - dd
        x = np.sqrt(max(rh, 1e-12) / r)
        tb.append(dict(drift_pct_yr=dr_yr, season=k, r_pct_d=r, r_apparent=rh, masked=bool(rh <= 0),
                       T_ratio=float(1 / x) if rh > 0 else np.inf, T_hat=float(T0[k] / x) if (k in T0 and rh > 0) else np.nan,
                       cost_excess_pct=float(((x + 1 / x) / 2 - 1) * 100) if rh > 0 else np.nan))
tb = pd.DataFrame(tb); tb.to_csv(os.path.join(RES, "cleaning_bias.csv"), index=False)
print(tb.round(3).to_string())
json.dump(out, open(os.path.join(RES, "reset_case_study.json"), "w"), indent=1)
