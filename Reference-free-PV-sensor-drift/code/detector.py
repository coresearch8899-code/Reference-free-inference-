# -*- coding: utf-8 -*-
"""Calibrated three-state network detector.
Every step that is applied to real data (seasonal fit on the development period only, step search, slope estimation, choice of sensor)
is repeated inside each bootstrap surrogate, so that thresholds reflect the whole procedure.
States: 0 no detected inconsistency; 1 detected, attribution ambiguous; 2 detected and attributed."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import norm

WEEKS_PER_YEAR = 52.18
MIN_POST = 26                     # weeks needed after a step before a drift statement is made; fixed before the simulations were run


def harmonics(doy, nh):
    w = 2 * np.pi * np.asarray(doy) / 365.25
    return np.column_stack([np.ones(len(w))] + [f(k * w) for k in range(1, nh + 1) for f in (np.sin, np.cos)])


def loo(L):
    """leave-one-out contrasts, rows = weeks, columns = sensors"""
    N = L.shape[1]
    return L - (L.sum(1, keepdims=True) - L) / (N - 1)


def signature(N):
    F = -np.ones((N, N)) / (N - 1)
    np.fill_diagonal(F, 1.0)
    return F                                          # column k: pattern of the contrasts when sensor k alone changes


def remove_season(C, X, dev):
    b = np.linalg.lstsq(X[:dev], C[:dev], rcond=None)[0]
    return C - X @ b


def step_search(C, half=8, min_size=0.05, zmin=6.0, maxit=6):
    """largest difference between the medians of the following and preceding `half` weeks; accepted if it exceeds min_size and
    zmin robust standard deviations of the weekly differences; the single-sensor pattern is removed and the search repeats"""
    C = C.copy(); n, N = C.shape; steps = []
    if n < 2 * half + 2:
        return C, steps
    F = signature(N)
    for _ in range(maxit):
        med = np.median(sliding_window_view(C, half, axis=0), axis=2)
        d = med[half:] - med[:-half]
        i, k = np.unravel_index(np.abs(d).argmax(), d.shape)
        t, dd = i + half, d[i, k]
        sig = 1.4826 * np.median(np.abs(np.diff(C[:, k]))) / np.sqrt(2)
        if abs(dd) < min_size or abs(dd) < zmin * sig:
            break
        C[t:] -= dd * F[:, k]
        steps.append((int(k), int(t), float(dd)))
    return C, steps


def slope_t(C, taus=(), keep=None):
    """slopes (% per year) and Newey-West t statistics of each column, jointly with one level term per week in taus.
    keep: optional boolean mask of usable weeks. Without steps and without a mask this is plain OLS on the window."""
    n0 = C.shape[0]
    idx = np.arange(n0) if keep is None else np.where(keep)[0]
    x = idx.astype(float); Y = C[idx]; n = len(idx)
    Z = [np.ones(n)]
    for tau in taus:
        d = (x >= tau).astype(float)
        if 2 <= d.sum() <= n - 2:
            Z.append(d)
    Q = np.linalg.qr(np.column_stack(Z))[0]
    xr = x - Q @ (Q.T @ x); Yr = Y - Q @ (Q.T @ Y)
    sxx = (xr ** 2).sum()
    b = (xr[:, None] * Yr).sum(0) / sxx
    u = Yr - xr[:, None] * b
    g = xr[:, None] * u
    S = (g ** 2).sum(0)
    L = max(4, int(1.5 * n ** (1 / 3)))
    for k in range(1, L + 1):
        S = S + 2 * (1 - k / (L + 1)) * (g[k:] * g[:-k]).sum(0)
    return b * WEEKS_PER_YEAR * 100, b / np.sqrt(np.maximum(S, 1e-20) / sxx ** 2)


def single_fault_fit(b):
    """residual of the best and second-best single-sensor explanation of the slope pattern"""
    N = len(b); F = signature(N)
    dlt = (F.T @ b) / (F ** 2).sum(0)
    R = ((b[:, None] - F * dlt) ** 2).sum(0)
    o = np.argsort(R)
    return int(o[0]), float(R[o[0]]), float(R[o[1]] - R[o[0]]), R


def pipeline(C_all, dev, test, X, method="joint", keep=None):
    """C_all: contrasts over dev + test weeks. Season from dev only; steps searched on data available at the end of test; only the test window is tested.
    method "joint": a step found inside the window gets its own level term, estimated with the slope. method "remove": the step estimate is subtracted first."""
    C = remove_season(C_all[:dev + test], X[:dev + test], dev)
    C2, steps = step_search(C)
    inwin = [(k, t, d) for k, t, d in steps if dev < t < dev + test]
    taus = []
    for t_ in sorted(t - dev for _, t, _ in inwin):
        if not taus or t_ - taus[-1] >= 4:
            taus.append(t_)
    if method == "joint":
        b, t = slope_t(C[dev:dev + test], taus, keep)
    else:
        b, t = slope_t(C2[dev:dev + test], (), keep)
    k, R1, D, Rall = single_fault_fit(b)
    post = test - max(taus) if taus else None
    return dict(b=b, t=t, S=float(np.abs(b).max()), k=k, R1=R1, D=D, Rall=Rall, steps=steps, inwin=inwin, taus=taus, post=post,
                unresolved=bool(method == "joint" and taus and post < MIN_POST))


def block_rows(m, n, rng, block=13):
    idx = np.empty(n, dtype=int); idx[0] = rng.integers(m)
    rs = rng.random(n) < 1 / block; nw = rng.integers(0, m, n)
    for t in range(1, n):
        idx[t] = nw[t] if rs[t] else (idx[t - 1] + 1) % m
    return idx


def calibrate(E, dev, test, X, rng, B=500, block=13, alpha=0.05, keep=None, add=()):
    """E: residual contrasts of the development period (rows resampled jointly, so that cross-sensor dependence is kept).
    add: steps (sensor, week, size) put into each surrogate, so that the step search and the joint fit are part of the calibration."""
    S, R1, D = np.empty(B), np.empty(B), np.empty(B)
    F = signature(E.shape[1])
    for i in range(B):
        Cs = E[block_rows(len(E), dev + test, rng, block)]
        for k, t, d in add:
            Cs[t:] += d * F[:, k]
        r = pipeline(Cs, dev, test, X, keep=keep)
        # consistency limit: residual of a fixed sensor, which is what a true single fault leaves; the minimum over sensors would be too strict
        S[i], R1[i], D[i] = r["S"], r["Rall"][rng.integers(len(r["Rall"]))], r["D"]
    return dict(s=float(np.quantile(S, 1 - alpha)), r=float(np.quantile(R1, 1 - alpha)), d=float(np.quantile(D, 1 - alpha)))


def conditional(r, q, E, dev, test, X, rng, B, cache, keep=None):
    """thresholds for this result: the plain ones when no step lies in the window, otherwise those for a surrogate with the same step(s)"""
    if not r["inwin"]:
        return q
    key = (dev, test, tuple(sorted((k, t, round(d, 1)) for k, t, d in r["inwin"])))
    if key not in cache:
        cache[key] = calibrate(E, dev, test, X, rng, B=B, keep=keep, add=r["inwin"])
    return cache[key]


def decide(r, q):
    """Attribution needs (i) an alarm, (ii) a single-fault pattern that fits, (iii) a clear margin over the next sensor.
    A step with too few weeks after it gives state 1 with r["unresolved"]: an inconsistency is detected and the drift question stays open."""
    if r.get("unresolved"):
        return 1, -1
    if r["S"] <= q["s"]:
        return 0, -1
    if len(r["b"]) >= 3 and r["R1"] <= q["r"] and r["D"] >= q["d"]:
        return 2, r["k"]
    return 1, -1


def nominal(r):
    """baseline: Newey-West t, Bonferroni over sensors, always picks the largest |t|"""
    N = len(r["t"])
    if (np.abs(r["t"]) > norm.ppf(1 - 0.05 / (2 * N))).any():
        return 2, int(np.abs(r["t"]).argmax())
    return 0, -1
