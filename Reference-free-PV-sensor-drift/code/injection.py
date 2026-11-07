# -*- coding: utf-8 -*-
"""Noise model and run loop shared by the injection studies: real residuals, stationary block bootstrap by day, weekly means."""
import os
import numpy as np
import pandas as pd
from detector import harmonics, loo, signature, pipeline, decide, nominal, block_rows

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
MODE = os.environ.get("MODE", "detrended"); SUF = "" if MODE == "detrended" else "_raw"
_p = pd.read_csv(os.path.join(RES, f"residual_pools{SUF}.csv"), index_col=0, parse_dates=True)
_p = _p[_p.index < "2025-01-01"]["sensor_contrast_h1_h3"].values / np.sqrt(2)
pool = _p[np.argmax(np.isfinite(_p)):]
DEV = 78
X = harmonics(1 + (7 * np.arange(400)) % 365, 1)


def noise(weeks, N, scale, rng, block=60):
    """independent sensor noise from real residuals: stationary block bootstrap by day, then weekly means"""
    out = np.empty((weeks, N))
    for k in range(N):
        idx = block_rows(len(pool), weeks * 7, rng, block)
        w = (pool[idx] * scale[k]).reshape(weeks, 7)
        c = np.isfinite(w).sum(1)
        m = np.where(c > 0, np.nansum(w, 1) / np.maximum(c, 1), np.nan)
        out[:, k] = np.where(np.isfinite(m), m, np.nanmean(m))
    return out


def thresholds(test, N, scale, B, rng, add=()):
    """thresholds from the noise model; add: steps (sensor, week, size) placed in every run, for the conditional case"""
    S, R, D = np.empty(B), np.empty(B), np.empty(B); F = signature(N)
    for i in range(B):
        C = loo(noise(DEV + test, N, scale, rng))
        for k, t, d in add:
            C[t:] += d * F[:, k]
        r = pipeline(C, DEV, test, X)
        S[i], R[i], D[i] = r["S"], r["Rall"][rng.integers(N)], r["D"]
    return dict(s=float(np.quantile(S, .95)), r=float(np.quantile(R, .95)), d=float(np.quantile(D, .95)))


def run(test, N, scale, q, rng, reps, drifts=None, shift=None, method="joint", B_cond=300, cache=None):
    """drifts: dict sensor -> % per year during the test period; shift: (sensor, week within test, size).
    With method joint, a run with a step in the window uses thresholds for that step (cached by sensor, week and rounded size)."""
    res = dict(no=0, amb=0, right=0, wrong=0, unresolved=0, inconsistency=0, nom_alarm=0, nom_right=0, nom_wrong=0, step_right=0, step_any=0, drift_reported=0)
    truth = set(k for k, v in (drifts or {}).items() if v != 0); bias = []
    t = np.zeros(DEV + test); t[DEV:] = np.arange(test) / 52.18
    cache = {} if cache is None else cache
    for _ in range(reps):
        L = noise(DEV + test, N, scale, rng)
        for k, v in (drifts or {}).items():
            L[:, k] += v / 100 * t
        if shift:
            L[DEV + shift[1]:, shift[0]] += shift[2]
        r = pipeline(loo(L), DEV, test, X, method=method)
        q_ = q
        if method == "joint" and r["inwin"]:
            key = tuple(sorted((k, t_, round(d, 1)) for k, t_, d in r["inwin"]))
            if key not in cache:
                cache[key] = thresholds(test, N, scale, B_cond, rng, add=r["inwin"])
            q_ = cache[key]
        st, kk = decide(r, q_)
        res["step_any"] += len(r["steps"]) > 0
        if shift and any(k_ == shift[0] and abs(t_ - (DEV + shift[1])) <= 6 for k_, t_, _ in r["steps"]):
            res["step_right"] += 1
        res["unresolved"] += bool(r["unresolved"]); res["drift_reported"] += st > 0 and not r["unresolved"]
        res["inconsistency"] += st > 0 or len(r["inwin"]) > 0
        if st == 0: res["no"] += 1
        elif st == 1: res["amb"] += 1
        elif kk in truth: res["right"] += 1
        else: res["wrong"] += 1
        sn, kn = nominal(r)
        if sn:
            res["nom_alarm"] += 1
            res["nom_right" if kn in truth else "nom_wrong"] += 1
        if len(truth) == 1:
            k0 = next(iter(truth)); bias.append(r["b"][k0] - drifts[k0])
    out = {k: v / reps for k, v in res.items()}
    out["bias"] = float(np.mean(bias)) if bias else np.nan
    out["bias_sd"] = float(np.std(bias)) if bias else np.nan
    return out
