# -*- coding: utf-8 -*-
"""Step F2: power, false-alarm rate and isolation of the reference-free drift detector.
Real residuals of the stable period are resampled with a stationary block bootstrap (they are NOT assumed AR(1)),
known drift is injected, and the leave-one-out network contrast is tested for a linear trend on weekly means with a
Newey-West (HAC) standard error. Grid: drift x record length x number of sensors; baselines; dependence scenarios."""
import json, os, sys
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
rng = np.random.default_rng(20261106)
MODE = os.environ.get("MODE", "detrended")     # "detrended" = lower noise bound, "raw" = upper noise bound
SUF = "" if MODE == "detrended" else "_raw"
pool = pd.read_csv(os.path.join(RES, f"residual_pools{SUF}.csv"), index_col=0, parse_dates=True)
pool = pool[pool.index < "2025-01-01"]
noise = json.load(open(os.path.join(RES, "noise_structure.json")))
NREP = int(os.environ.get("NREP", 1000))
BLOCK = 60                    # mean block length (days) of the stationary bootstrap
ALPHA = 0.05
Z = norm.ppf(1 - ALPHA / 2) + norm.ppf(0.8)


def sbb(series, n, reps, rng=rng, block=BLOCK):
    """stationary block bootstrap: returns reps x n (NaN pattern of the source travels with the blocks)"""
    x = series[np.arange(np.argmax(np.isfinite(series)), len(series))]
    m = len(x)
    idx = np.empty((reps, n), dtype=np.int32)
    idx[:, 0] = rng.integers(0, m, reps)
    for t in range(1, n):
        restart = rng.random(reps) < 1 / block
        idx[:, t] = np.where(restart, rng.integers(0, m, reps), (idx[:, t - 1] + 1) % m)
    return x[idx]


def weekly(a):
    n = a.shape[-1] // 7
    w = a[..., :n * 7].reshape(a.shape[:-1] + (n, 7))
    c = np.sum(np.isfinite(w), -1)
    m = np.where(c > 0, np.nansum(w, -1) / np.maximum(c, 1), np.nan)
    return np.where(np.isfinite(m), m, np.nanmean(m, -1, keepdims=True))


def hac_t(w, naive=False):
    """OLS slope and t statistic per row; Newey-West with Bartlett weights (weekly data)"""
    n = w.shape[-1]
    x = np.arange(n) - (n - 1) / 2
    sxx = np.sum(x ** 2)
    wc = w - w.mean(-1, keepdims=True)
    b = (wc * x).sum(-1) / sxx
    u = wc - b[..., None] * x
    g = x * u
    S = (g ** 2).sum(-1)
    if naive:                                   # AR(1)-corrected SE, the formula used in R14
        a, c = u[..., :-1], u[..., 1:]
        phi = (a * c).sum(-1) / (u ** 2).sum(-1)
        se = np.sqrt((u ** 2).sum(-1) / (n - 2) / sxx * (1 + phi) / (1 - phi))
        return b, b / se
    L = max(4, int(1.5 * n ** (1 / 3)))
    for k in range(1, L + 1):
        S = S + 2 * (1 - k / (L + 1)) * (g[..., k:] * g[..., :-k]).sum(-1)
    return b, b / np.sqrt(np.maximum(S, 1e-20) / sxx ** 2)


def network(delta_day, T_days, N, reps, sd_extra=0.0, common=0.0, miss=0.0, pool_key="sensor_contrast_h1_h3"):
    """N sensors, sensor 0 drifts by delta_day. Idiosyncratic noise = contrast residual / sqrt(2).
    common: fraction of the drift shared by all sensors. Returns detection, isolation and FAR-style rates."""
    r = pool[pool_key].values / np.sqrt(2)
    t = np.arange(T_days)
    E = np.stack([sbb(r, T_days, reps) for _ in range(N)], 1)        # reps x N x days
    if sd_extra > 0:
        E = E + rng.normal(0, sd_extra, E.shape)
    if miss > 0:
        E = np.where(rng.random(E.shape) < miss, np.nan, E)
    E = E + common * delta_day * t
    E[:, 0, :] += (1 - common) * delta_day * t
    W = weekly(E)
    C = W - (W.sum(1, keepdims=True) - W) / (N - 1)                  # sensor minus mean of the others
    b, tt = hac_t(C)
    crit = norm.ppf(1 - ALPHA / (2 * N))
    flagged = np.abs(tt) > crit
    return dict(detect=float(flagged.any(1).mean()),
                isolate=float((flagged.any(1) & ((np.abs(tt) + 1e-9 * rng.random(tt.shape)).argmax(1) == 0)).mean()),   # ties (N = 2) broken at random
                tmax=np.abs(tt).max(1))


def predicted_power(delta_yr, T_days, N, lrv_pair):
    """Eq. (7b): closed form with the long-run variance of the leave-one-out contrast"""
    lrv = lrv_pair / 2 * N / (N - 1)
    sd = np.sqrt(12 * lrv / T_days ** 3)
    a = norm.ppf(1 - ALPHA / (2 * N))
    return float(norm.cdf(abs(delta_yr) / 365.25 / sd - a) + norm.cdf(-abs(delta_yr) / 365.25 / sd - a))


lrv = noise["sensor_contrast_h1_h3"]["lrv_ar1" if MODE == "detrended" else "lrv_raw"]
GRID_D = [0.5, 1, 2, 3, 5, 8, 10]           # % per year
GRID_T = [6, 12, 18, 24, 36, 60]            # months
GRID_N = [2, 3, 4, 6]
rows = []
for N in GRID_N:
    for Tm in GRID_T:
        T = int(Tm * 30.44)
        null = network(0.0, T, N, NREP)
        far = null["detect"]
        crit_cal = float(np.quantile(null["tmax"], 0.95))              # size-adjusted threshold: 5 % false alarms by construction
        for D in GRID_D:
            res = network(-D / 100 / 365.25, T, N, NREP)
            rows.append(dict(N=N, T_months=Tm, drift_pct_yr=D, power=res["detect"], power_adj=float((res["tmax"] > crit_cal).mean()), crit_cal=crit_cal, isolation=res["isolate"], far=far,
                             power_pred=predicted_power(D / 100, T, N, lrv)))
        print(f"N={N} T={Tm} FAR={far:.3f}", flush=True)
pw = pd.DataFrame(rows)
pw.to_csv(os.path.join(RES, f"power_grid{SUF}.csv"), index=False)

# ---- SE choice: the AR(1) correction of R14 versus HAC, under no drift, N = 3
far_se = {}
for Tm in (12, 24, 36):
    T = int(Tm * 30.44)
    r = pool["sensor_contrast_h1_h3"].values
    W = weekly(sbb(r, T, 4 * NREP))
    far_se[Tm] = dict(hac=float((np.abs(hac_t(W)[1]) > 1.96).mean()), ar1=float((np.abs(hac_t(W, naive=True)[1]) > 1.96).mean()))
print("FAR by SE:", far_se)

# ---- baselines: single-reference designs, one sensor against an external series, plus the own PR
BASE = {"Sensor contrast (N = 2)": ("sensor_contrast_h1_h3", 1.0),
        "Clear-sky ratio, all days": ("clear_sky_all_days_h1", 1.0),
        "Clear-sky ratio, clearest 40 % of days": ("clear_sky_clear_days_h1", 1.0),
        "Satellite ratio (CERES)": ("satellite_CERES_all_days_h1", 1.0)}
INTERANNUAL = 0.02            # assumed s.d. of the yearly level of an external reference (scenario)
brows = []
for lab, (key, _) in BASE.items():
    r = pool[key].values
    for Tm in (12, 24, 36, 60):
        T = int(Tm * 30.44); t = np.arange(T)
        for ia in (0.0, INTERANNUAL):
            if ia > 0 and key.startswith("sensor"):
                continue
            E = sbb(r, T, NREP)
            if ia > 0:
                yr = (t // 365).astype(int)
                E = E + rng.normal(0, ia, (NREP, yr.max() + 1))[:, yr]
            t0 = np.abs(hac_t(weekly(E))[1]); far = float((t0 > 1.96).mean()); ca = float(np.quantile(t0, 0.95))
            pw_, pa_ = {}, {}
            for D in (2, 5, 8.9):
                tt_ = np.abs(hac_t(weekly(E - D / 100 / 365.25 * t))[1])
                pw_[D] = float((tt_ > 1.96).mean()); pa_[D] = float((tt_ > ca).mean())
            brows.append(dict(method=lab, T_months=Tm, interannual_sd=ia, far=far, power_2=pw_[2], power_5=pw_[5], power_8_9=pw_[8.9],
                              adj_2=pa_[2], adj_5=pa_[5], adj_8_9=pa_[8.9]))
# own PR: array losses and soiling are confounded with the sensor
r = pool["own_PR_y1_h1"].values
for Tm in (12, 24, 36, 60):
    T = int(Tm * 30.44); t = np.arange(T)
    E = sbb(r, T, NREP)
    plr = -0.01 / 365.25 * t                        # -1 %/yr array loss, a typical value
    saw = -0.03 * ((t % 120) / 120)                 # 3 % soiling sawtooth, cleaned every 120 days
    arr = plr + saw
    far = float((hac_t(weekly(E + arr))[1] > 1.96).mean())
    pw_ = {D: float((hac_t(weekly(E + arr + D / 100 / 365.25 * t))[1] > 1.96).mean()) for D in (2, 5, 8.9)}
    brows.append(dict(method="Own-sensor PR trend (rising PR)", T_months=Tm, interannual_sd=0.0, far=far, power_2=pw_[2], power_5=pw_[5], power_8_9=pw_[8.9]))
bs = pd.DataFrame(brows); bs.to_csv(os.path.join(RES, f"baselines{SUF}.csv"), index=False)

# ---- dependence between zones (N = 3, 24 months, 5 %/yr)
lam_gap = 0.08; sd_a = 0.295                        # fitted loading gap of Zone 2 and common-sky s.d. (R14 Table 4)
SC = {"Reference case": dict(),
      "Half of the drift shared by all sensors": dict(common=0.5),
      "All sensors drift alike": dict(common=1.0),
      "Different orientation (sky loading differs by 0.08)": dict(sd_extra=lam_gap * sd_a),
      "50 % of sensor days censored or missing": dict(miss=0.5)}
drows = []
for lab, kw in SC.items():
    res = network(-0.05 / 365.25, int(24 * 30.44), 3, NREP, **kw)
    n0 = network(0.0, int(24 * 30.44), 3, NREP, **{k: v for k, v in kw.items() if k != "common"})
    drows.append(dict(scenario=lab, power=res["detect"], isolation=res["isolate"], far=n0["detect"]))
dp = pd.DataFrame(drows); dp.to_csv(os.path.join(RES, f"dependence{SUF}.csv"), index=False)
json.dump(dict(far_by_se=far_se, nrep=NREP, block=BLOCK), open(os.path.join(RES, f"power_meta{SUF}.json"), "w"), indent=1)
print(bs.to_string()); print(dp.to_string())
