# -*- coding: utf-8 -*-
"""Step 2: semi-synthetic validation. The real sky (smoothed a + L of the main fit) and the fitted seasonal terms
are kept; sensor and array states with known truth are injected, AR(1) and white noise with the fitted parameters
are added, and the 90 % PR truncation of the reporting layer is applied. Estimators compared at the fitted
hyper-parameters:
  MSSM-Tobit (proposed), MSSM-drop (truncated days discarded), MSSM-naive (bound used as value),
  RATIO: OLS of ln(H1/H3) on harmonics + broken-stick trend with the true break (R13-type), uncensored days only,
  WINDOW: post-minus-pre median of ln(Yf1/Yf3) (R13 event-study estimator) for cleaning jumps."""
import json, os, pickle
import numpy as np
import pandas as pd
from ss_data import load_panel, ROOT
from ss_model import MSSM, SIG, DLT, S, A, L, M0, MH, GY, GH1, GH2, KAP, BT, harm

RES = os.path.join(ROOT, "results")
rng = np.random.default_rng(20261026)
P = load_panel()
fitm = pickle.load(open(os.path.join(RES, "mssm_main.pkl"), "rb"))
th = fitm["theta"]
XS = fitm["XS"]
n = P["n"]
H = np.stack([harm(d) for d in P["doy"]])
lnHcs = np.log(P["Hcs"])
Tt = (P["tamb"] - 25) / 10
EV = ["C2", "C3", "C4", "C5", "C6"]
evk = [P["events"][e] for e in EV]
BREAK = int(np.searchsorted(P["date"].values, np.datetime64("2025-03-01")))
NREP = int(os.environ.get("NREP", 100))

SKY = XS[:, A] + XS[:, L]
M = XS[-1, M0] + H @ XS[-1, MH]
GYc = H @ XS[-1, GY]
GHi = [H @ XS[-1, GH1], H @ XS[-1, GH2], np.zeros(n)]
KAPv, BTv = XS[-1, KAP], XS[-1, BT]
LAMH = [1.0, th["lam_h2"], 1.0]


def ar1(phi, sd, size):
    e = np.empty(size)
    e[0] = rng.normal(0, sd / np.sqrt(1 - phi ** 2))
    w = rng.normal(0, sd, size)
    for k in range(1, size):
        e[k] = phi * e[k - 1] + w[k]
    return e


def truth(scale, jump_scale):
    """truth built on the fitted paths, so that truncation occurs as in the record:
    sensor-1 divergence after BREAK scaled by `scale`; array jumps of zone 1 at cleanings scaled by `jump_scale`."""
    sig = XS[:, SIG].copy()
    sig[BREAK:, 0] = sig[BREAK, 0] + scale * (sig[BREAK:, 0] - sig[BREAK, 0])
    s = XS[:, S].copy()
    ds = np.diff(s[:, 0], prepend=s[0, 0])
    for e in evk:
        w = slice(e, e + 7)
        ds[w] = ds[w] * jump_scale[evk.index(e)]
    s[:, 0] = np.cumsum(ds)
    return sig, s


def simulate(sig, s):
    y = np.full((n, 3), np.nan); h = np.full((n, 3), np.nan); hc = np.full((n, 3), np.nan)
    mask_y = np.isfinite(P["y"])
    for i in range(3):
        yi = (lnHcs + M + th["lam_y"] * SKY + GYc + KAPv[i] + s[:, i] + BTv * Tt
              + ar1(th["phi_y"], th["sd_ary"], n) + rng.normal(0, th["sd_y%d" % (i + 1)], n))
        hi = lnHcs + M + LAMH[i] * SKY + GHi[i] + sig[:, i] + ar1(th["phi_h"], th["sd_arh"], n) + rng.normal(0, th["sd_h"], n)
        yi[~mask_y[:, i]] = np.nan
        y[:, i] = yi
        ok = np.isfinite(yi)
        trunc = (yi - hi) >= np.log(0.90)
        h[ok & ~trunc, i] = hi[ok & ~trunc]
        hc[ok & trunc, i] = yi[ok & trunc] - np.log(0.90)
    return y, h, hc


def ratio_ols(h):
    ok = np.isfinite(h[:, 0]) & np.isfinite(h[:, 2])
    t = np.arange(n) / 365.25
    X = np.column_stack([np.ones(n), H, t, np.maximum(t - BREAK / 365.25, 0)])
    b = np.linalg.lstsq(X[ok], (h[ok, 0] - h[ok, 2]), rcond=None)[0]
    return b[-1] + b[-2]


def window_jumps(y):
    r = y[:, 0] - y[:, 2]
    out = []
    for e in evk:
        pre = r[e - 14:e]; post = r[e + 4:e + 18]
        out.append(np.nanmedian(post) - np.nanmedian(pre))
    return np.array(out)


models = {c: MSSM(P, censor=c) for c in ("tobit", "drop", "naive")}
rows = []
for rep in range(NREP):
    sig, s = truth(rng.uniform(0.5, 1.5), rng.uniform(0.0, 2.0, len(evk)))
    c13 = sig[:, 0] - sig[:, 2]
    drift = (c13[n - 1] - c13[BREAK + 30]) / ((n - 1 - BREAK - 30) / 365.25)
    y, h, hc = simulate(sig, s)
    true_rel = np.array([(s[e + 6, 0] - s[e + 6, 2]) - (s[e - 1, 0] - s[e - 1, 2]) for e in evk])
    r = dict(rep=rep, drift_true=drift, ratio=ratio_ols(h),
             trunc_share_z1=float(np.isfinite(hc[:, 0]).sum() / np.isfinite(y[:, 0]).sum()),
             trunc_share_z1_post=float(np.isfinite(hc[BREAK:, 0]).sum() / np.isfinite(y[BREAK:, 0]).sum()))
    r["jump_rmse_window"] = float(np.sqrt(np.mean((window_jumps(y) - true_rel) ** 2)))
    for c, m in models.items():
        m.set_data(y, h, hc)
        XS_, PS_ = m.smooth(th)
        post = slice(BREAK + 30, n - 1)
        r[f"drift_{c}"] = float(np.mean(XS_[post, DLT[0]]) * 365.25)
        # drift over the post-break period from the smoothed level (robust to LLT wiggles)
        w = np.zeros(PS_.shape[1]); w[SIG[0]] = 1; w[SIG[2]] = -1
        r[f"drift_lvl_{c}"] = float((w @ XS_[n - 1] - w @ XS_[BREAK + 30]) / ((n - 1 - BREAK - 30) / 365.25))
        sdl = np.sqrt(w @ PS_[n - 1] @ w + w @ PS_[BREAK + 30] @ w) / ((n - 1 - BREAK - 30) / 365.25)
        r[f"cover_{c}"] = bool(abs(r[f"drift_lvl_{c}"] - drift) < 1.96 * sdl)
        r[f"sigend_err_{c}"] = float(XS_[-1, SIG[0]] - sig[-1, 0])
        r[f"sigend_cover_{c}"] = bool(abs(r[f"sigend_err_{c}"]) < 1.96 * np.sqrt(PS_[-1, SIG[0], SIG[0]]))
        je = np.array([(XS_[e + 6, S[0]] - XS_[e + 6, S[2]]) - (XS_[e - 1, S[0]] - XS_[e - 1, S[2]]) for e in evk])
        r[f"jump_rmse_{c}"] = float(np.sqrt(np.mean((je - true_rel) ** 2)))
    rows.append(r)
    print(rep, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "validation_semisynthetic.csv"), index=False)
summ = {}
for c in ("tobit", "drop", "naive"):
    err = df[f"drift_lvl_{c}"] - df.drift_true
    summ[c] = dict(drift_bias_pct_yr=float(err.mean() * 100), drift_rmse_pct_yr=float(np.sqrt((err ** 2).mean()) * 100),
                   drift_cover95=float(df[f"cover_{c}"].mean()),
                   sigend_bias_pct=float(df[f"sigend_err_{c}"].mean() * 100),
                   sigend_rmse_pct=float(np.sqrt((df[f"sigend_err_{c}"] ** 2).mean()) * 100),
                   sigend_cover95=float(df[f"sigend_cover_{c}"].mean()),
                   jump_rmse_logpct=float(df[f"jump_rmse_{c}"].mean() * 100))
err = df.ratio - df.drift_true
summ["ratio_ols"] = dict(drift_bias_pct_yr=float(err.mean() * 100), drift_rmse_pct_yr=float(np.sqrt((err ** 2).mean()) * 100))
summ["window_median"] = dict(jump_rmse_logpct=float(df.jump_rmse_window.mean() * 100))
summ["n_rep"] = NREP
summ["trunc_share_z1_mean"] = float(df.trunc_share_z1.mean())
summ["trunc_share_z1_post_mean"] = float(df.trunc_share_z1_post.mean())
json.dump(summ, open(os.path.join(RES, "validation_summary.json"), "w"), indent=2)
print(json.dumps(summ, indent=2))
