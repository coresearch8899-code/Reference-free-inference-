# -*- coding: utf-8 -*-
"""Step F6: external validation on the NIST campus arrays (PVDAQ 4901/4902/4903, CC BY 4.0), every third day.
Groups come from instrument metadata, not from the data. There is no ground truth for these sensors, so an alarm without injection
is 'an alarm on well-kept instruments', not necessarily false. Results are given with and without a minimum effect size."""
import json, os, re
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
D = pd.read_csv(os.path.join(ROOT, "data", "nist_daily.csv"), parse_dates=["date"])
NAME = {4901: "C", 4902: "G", 4903: "R"}
frames = {}
for s, g in D.groupby("system"):
    g = g.set_index("date").sort_index()
    for c in g.columns:
        m = re.match(r"irradiance_(?:poa|ghi)_o_(\d+)__kwh$", c)
        if not m:
            continue
        v = g[c].where(g[c.replace("__kwh", "__cov")] > 0.95)
        v = v.where(v > 0.15 * v.quantile(0.9))                 # dark days carry no usable ratio; pyranometer channels are stored on another scale
        frames[f"{NAME[s]}{m.group(1)}"] = v
X = pd.DataFrame(frames).sort_index()
L = np.log(X)
GROUPS = {"Roof, six Eppley PSP": ["R2206", "R2207", "R2209", "R2210", "R2212", "R2213"],
          "Campus, one Eppley PSP per array": ["C2196", "G2202", "R2206"],
          "Roof, three sensor types": ["R2209", "R2211", "R2215"]}
FLOORS = (0.0, 2.0, 3.0, 5.0)
w = 2 * np.pi * X.index.dayofyear.values / 365.25
HM = np.column_stack([np.ones(len(X)), np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w)])


def contrasts(Lg):
    """deseasonalised leave-one-out contrasts, weekly means"""
    W = {}
    for c in Lg.columns:
        others = Lg.drop(columns=c); others = others - others.mean()
        cn = (Lg[c] - Lg[c].mean() - others.mean(axis=1)).values
        ok = np.isfinite(cn)
        r = np.full(len(cn), np.nan); r[ok] = cn[ok] - HM[ok] @ np.linalg.lstsq(HM[ok], cn[ok], rcond=None)[0]
        W[c] = pd.Series(r, index=Lg.index).resample("7D").mean()
    return pd.DataFrame(W)


def hac(y):
    ok = np.isfinite(y); x = np.where(ok)[0].astype(float); x -= x.mean(); yy = y[ok] - y[ok].mean()
    sxx = (x ** 2).sum(); b = (x * yy).sum() / sxx; u = yy - b * x; g = x * u; S = (g ** 2).sum()
    L_ = max(4, int(1.5 * ok.sum() ** (1 / 3)))
    for k in range(1, L_ + 1):
        S += 2 * (1 - k / (L_ + 1)) * (g[k:] * g[:-k]).sum()
    return b * 52.18 * 100, b / np.sqrt(max(S, 1e-20) / sxx ** 2)


def windows(W, win):
    crit = norm.ppf(1 - 0.05 / (2 * W.shape[1])); out = []
    for st in range(0, len(W) - win + 1, 13):
        a = np.array([hac(W[c].values[st:st + win]) for c in W.columns])
        out.append((a[:, 0], a[:, 1], crit))
    return out


def rate(out, floor, k=None):
    """share of windows with an alarm; with k, share in which sensor k is flagged and named"""
    n = 0
    for sl, tv, crit in out:
        fl = (np.abs(tv) > crit) & (np.abs(sl) >= floor)
        if k is None:
            n += fl.any()
        else:
            n += fl.any() and int(np.argmax(np.abs(tv) * fl)) == k
    return n / len(out)


def find_steps(Lg, half=8, min_size=0.05, zmin=6.0, max_steps=8):
    """iteratively locate level shifts: difference of medians of the 8 weeks after and before each week, on the leave-one-out contrasts.
    The sensor with the largest shift is the one named; it is level-corrected and the search repeats."""
    Lg = Lg.copy(); steps = []
    for _ in range(max_steps):
        W = contrasts(Lg)
        best = None
        for c in W.columns:
            v = W[c].values
            for t in range(half, len(v) - half):
                a, b = v[t - half:t], v[t:t + half]
                if np.isfinite(a).sum() < 4 or np.isfinite(b).sum() < 4:
                    continue
                d = np.nanmedian(b) - np.nanmedian(a)
                if best is None or abs(d) > abs(best[2]):
                    best = (c, t, d)
        if best is None:
            break
        c, t, d = best
        mad = 1.4826 * np.nanmedian(np.abs(np.diff(W[c].values))) / np.sqrt(2)
        if abs(d) < min_size or abs(d) < zmin * mad:
            break
        date = W.index[t]
        # a shift in one of N sensors moves its own leave-one-out contrast by d = size, the others by -size/(N-1)
        Lg.loc[Lg.index >= date, c] -= d
        steps.append(dict(sensor=c, date=str(date.date()), size_log=round(float(d), 3)))
    return Lg, steps


rows, full = [], []
for name, g in GROUPS.items():
    raw = L[g]
    Lg, steps = find_steps(raw)
    print(name, "steps:", steps, flush=True)
    W0 = contrasts(Lg)
    Wraw = contrasts(raw)
    Wraw.to_csv(os.path.join(RES, f"nist_contrast_{list(GROUPS).index(name)}.csv"))
    base_raw = {m: windows(Wraw, int(m * 4.348)) for m in (18, 24, 36)}
    pw = W0.apply(lambda s: hac(s.values)[0])                     # whole-record slopes after step removal
    full.append(dict(group=name, N=len(g), steps=steps, days=int(Lg.dropna(how="all").shape[0]), slopes={k: round(float(v), 2) for k, v in pw.items()},
                     median_abs=float(pw.abs().median()), max_abs=float(pw.abs().max()), sd_contrast=float(np.nanstd(W0.values))))
    print(name, full[-1], flush=True)
    for win_m in (18, 24, 36):
        wn = int(win_m * 4.348)
        base = windows(W0, wn)
        row = dict(group=name, N=len(g), window_months=win_m, n_windows=len(base))
        row["max_abs_slope"] = float(max(np.abs(b[0]).max() for b in base))
        row["alarm_before_step_removal"] = rate(base_raw[win_m], 0.0)
        for fl in FLOORS:
            row[f"alarm_floor{fl:g}"] = rate(base, fl)
        for inj in (2, 5, 10):
            for fl in FLOORS:
                det, iso = [], []
                for k, c in enumerate(g):
                    Li = Lg.copy()
                    Li[c] = Li[c] - inj / 100 * (Li.index - Li.index[0]).days.values / 365.25
                    o = windows(contrasts(Li), wn)
                    det.append(rate(o, fl)); iso.append(rate(o, fl, k))
                row[f"power{inj}_floor{fl:g}"] = float(np.mean(det)); row[f"iso{inj}_floor{fl:g}"] = float(np.mean(iso))
        rows.append(row); print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items() if k.startswith(("group", "window", "n_w", "max", "alarm", "iso5", "power5"))}, flush=True)
pd.DataFrame(rows).to_csv(os.path.join(RES, "nist_validation.csv"), index=False)
json.dump(dict(full=full, n_days=int(len(X)), first=str(X.index.min().date()), last=str(X.index.max().date())), open(os.path.join(RES, "nist_meta.json"), "w"), indent=1)
