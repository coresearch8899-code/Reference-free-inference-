# -*- coding: utf-8 -*-
"""Step F1: empirical noise structure of every monitored signal in the stable period (before 2025).
Reports s, lag-1 phi, ACF at longer lags and the long-run variance, which is what really governs trend detection."""
import json, os, sys
import numpy as np
import pandas as pd

from ss_data import load_panel, load_power
from ss_model import harm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def deseason(s, H, mask, trend=True):
    ok = mask & np.isfinite(s)
    X = np.column_stack([np.ones(len(s)), H] + ([np.arange(len(s)) / 365.25] if trend else []))
    b = np.linalg.lstsq(X[ok], s[ok], rcond=None)[0]
    r = np.full(len(s), np.nan)
    r[ok] = s[ok] - X[ok] @ b
    return r


def acf(r, lag):
    a, b = r[:-lag], r[lag:]
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 30 else np.nan


def long_run_var(r, L=90):
    """Bartlett-kernel long-run variance on a gappy daily series (pairwise autocovariances)."""
    x = r - np.nanmean(r)
    g0 = np.nanmean(x * x)
    lrv = g0
    for k in range(1, L + 1):
        a, b = x[:-k], x[k:]
        ok = np.isfinite(a) & np.isfinite(b)
        lrv += 2 * (1 - k / (L + 1)) * np.mean(a[ok] * b[ok])
    return float(lrv)


if __name__ == "__main__":
    P = load_panel()
    d = P["date"]; H = np.stack([harm(x) for x in P["doy"]])
    pre = d < np.datetime64("2025-01-01")
    pw = load_power(d)
    h, y = P["h"], P["y"]
    lnH = np.log(P["Hcs"])
    clear = (h[:, 2] - lnH) > np.nanpercentile((h[:, 2] - lnH)[pre], 60)      # clearest 40 % by the stable sensor
    S = {"sensor_contrast_h1_h3": h[:, 0] - h[:, 2],
         "yield_ratio_y1_y3": y[:, 0] - y[:, 2],
         "clear_sky_all_days_h1": h[:, 0] - lnH,
         "clear_sky_clear_days_h1": np.where(clear, h[:, 0] - lnH, np.nan),
         "satellite_CERES_all_days_h1": h[:, 0] - np.log(pw.ALLSKY_SFC_SW_DWN.values),
         "own_PR_y1_h1": y[:, 0] - h[:, 0]}
    out, pool, raw = {}, {}, {}
    for k, s in S.items():
        r = deseason(s, H, pre)
        r0 = deseason(s, H, pre, trend=False)        # upper bound: slow variation kept in the noise
        pool[k], raw[k] = r, r0
        sd = float(np.nanstd(r, ddof=1)); phi = acf(r, 1); lrv = long_run_var(r)
        out[k] = dict(sd_raw=float(np.nanstd(r0, ddof=1)), lrv_raw=long_run_var(r0), n=int(np.isfinite(r).sum()), sd=sd, phi1=phi, acf7=acf(r, 7), acf30=acf(r, 30), acf90=acf(r, 90),
                      lrv=lrv, lrv_ar1=sd ** 2 * (1 + phi) / (1 - phi), vif_true=lrv / sd ** 2, vif_ar1=(1 + phi) / (1 - phi))
        print(f"{k:32s} n={out[k]['n']:4d} sd={sd:.4f} phi1={phi:.2f} acf7={out[k]['acf7']:.2f} acf30={out[k]['acf30']:.2f} "
              f"acf90={out[k]['acf90']:.2f} VIF_true={out[k]['vif_true']:.1f} VIF_AR1={out[k]['vif_ar1']:.1f}")
    json.dump(out, open(os.path.join(RES, "noise_structure.json"), "w"), indent=1)
    pd.DataFrame(pool, index=d).to_csv(os.path.join(RES, "residual_pools.csv"))
    pd.DataFrame(raw, index=d).to_csv(os.path.join(RES, "residual_pools_raw.csv"))
