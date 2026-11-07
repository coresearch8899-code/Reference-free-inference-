# -*- coding: utf-8 -*-
"""Step 4: results of the fitted MSSM: sensor trajectories, drift rates, array condition, soiling rates,
intervention effects, net PLR, auxiliary-residual screening for unlogged events, variant comparison."""
import json, os, pickle
import numpy as np
import pandas as pd
from scipy.stats import norm, chi2
from ss_data import load_panel, ROOT
from ss_model import MSSM, SIG, DLT, S, RD, RW, KAP, PNAMES, NX

RES = os.path.join(ROOT, "results")
P = load_panel()
n = P["n"]
dates = P["date"]
fit = pickle.load(open(os.path.join(RES, "mssm_main.pkl"), "rb"))
th = fit["theta"]
m = MSSM(P)
m.smooth(th)
XS, PS, J = m.XS, m.PS, m.J
S_LAB = 11.513
yrs = lambda a, b: (b - a) / 365.25
out = {"theta": th, "ll": fit["ll"], "n_days": n, "n_env_obs": m.n_env}


def cov_between(t1, t2):
    """Cov(x_t1, x_t2 | Y) for t1 < t2 = J_t1 J_t1+1 ... J_t2-1 P_s[t2]"""
    M = np.eye(NX)
    for t in range(t1, t2):
        M = M @ J[t]
    return M @ PS[t2]


def diff_ci(idx_a, t1, t2, idx_b=None, scale=1.0):
    """posterior mean and 95 % CI of [x_a(t2) - x_b(t2)] - [x_a(t1) - x_b(t1)] (idx_b optional)"""
    w2 = np.zeros(NX); w1 = np.zeros(NX)
    w2[idx_a] += 1; w1[idx_a] += 1
    if idx_b is not None:
        w2[idx_b] -= 1; w1[idx_b] -= 1
    mean = w2 @ XS[t2] - w1 @ XS[t1]
    C = cov_between(t1, t2)
    var = w2 @ PS[t2] @ w2 + w1 @ PS[t1] @ w1 - 2 * w1 @ C @ w2
    sd = np.sqrt(max(var, 0))
    return [float(mean * scale), float((mean - 1.96 * sd) * scale), float((mean + 1.96 * sd) * scale)]


def tidx(s):
    return int(np.searchsorted(dates.values, np.datetime64(s)))


# ---- sensors ----
marks = {"start": 0, "2025-01-01": tidx("2025-01-01"), "2025-09-01": tidx("2025-09-01"),
         "2026-03-01": tidx("2026-03-01"), "end": n - 1}
sens = {}
for i in range(3):
    z = {}
    for k, t in marks.items():
        mu, sd = XS[t, SIG[i]], np.sqrt(PS[t, SIG[i], SIG[i]])
        z[f"sig_{k}"] = [float(mu * 100), float((mu - 1.96 * sd) * 100), float((mu + 1.96 * sd) * 100)]
        z[f"Seff_{k}"] = [float(S_LAB * np.exp(mu)), float(S_LAB * np.exp(mu - 1.96 * sd)), float(S_LAB * np.exp(mu + 1.96 * sd))]
    t0, t1, t2 = 30, marks["2025-01-01"], n - 1
    z["drift_pre2025_pct_yr"] = diff_ci(SIG[i], t0, t1, scale=100 / yrs(t0, t1))
    z["drift_2025on_pct_yr"] = diff_ci(SIG[i], t1, t2, scale=100 / yrs(t1, t2))
    z["drift_record_pct_yr"] = diff_ci(SIG[i], t0, t2, scale=100 / yrs(t0, t2))
    # relative to sensor 3 (gauge-invariant contrast)
    if i < 2:
        z["rel3_drift_pre2025_pct_yr"] = diff_ci(SIG[i], t0, t1, SIG[2], scale=100 / yrs(t0, t1))
        z["rel3_drift_2025on_pct_yr"] = diff_ci(SIG[i], t1, t2, SIG[2], scale=100 / yrs(t1, t2))
    sens[f"zone{i+1}"] = z
# onset: smoothed drift-rate state of sensor 1 credibly negative relative to sensor 3
d13 = (XS[:, DLT[0]] - XS[:, DLT[2]]) * 36525
sd13 = np.sqrt(PS[:, DLT[0], DLT[0]] + PS[:, DLT[2], DLT[2]] - 2 * PS[:, DLT[0], DLT[2]]) * 36525
neg = np.where(d13 + 1.96 * sd13 < 0)[0]
sens["onset_rel3_first_credible_negative"] = str(dates[neg[0]].date()) if len(neg) else None
sens["dlt13_min_pct_yr"] = [float(d13.min()), str(dates[int(np.argmin(d13))].date())]
out["sensors"] = sens
np.save(os.path.join(RES, "smoothed_main.npy"), XS)
np.save(os.path.join(RES, "smoothed_main_var.npy"), np.array([np.diag(p) for p in PS]))
np.save(os.path.join(RES, "smoothed_dlt13.npy"), np.column_stack([d13, sd13]))

# ---- arrays ----
arr = {}
for i in range(3):
    z = {}
    for nm, idx in (("r_dry", RD[i]), ("r_wet", RW[i])):
        mu, sd = XS[-1, idx], np.sqrt(PS[-1, idx, idx])
        z[nm + "_pct_d"] = [float(mu * 100), float((mu - 1.96 * sd) * 100), float((mu + 1.96 * sd) * 100)]
    if i < 2:
        for nm, ia, ib in (("rel3_r_dry_pct_d", RD[i], RD[2]), ("rel3_r_wet_pct_d", RW[i], RW[2])):
            w = np.zeros(NX); w[ia] = 1; w[ib] = -1
            mu, sd = w @ XS[-1], np.sqrt(w @ PS[-1] @ w)
            z[nm] = [float(mu * 100), float((mu - 1.96 * sd) * 100), float((mu + 1.96 * sd) * 100)]
    t0 = 30
    z["net_PLR_pct_yr"] = diff_ci(S[i], t0, n - 1, scale=100 / yrs(t0, n - 1))
    if i < 2:
        z["net_PLR_rel3_pct_yr"] = diff_ci(S[i], t0, n - 1, S[2], scale=100 / yrs(t0, n - 1))
    frac_dry = P["dry"].mean()
    w = np.zeros(NX); w[RD[i]] = frac_dry; w[RW[i]] = 1 - frac_dry
    mu, sd = w @ XS[-1] * 36525, np.sqrt(w @ PS[-1] @ w) * 36525
    z["underlying_loss_rate_pct_yr"] = [float(mu), float(mu - 1.96 * sd), float(mu + 1.96 * sd)]
    arr[f"zone{i+1}"] = z
out["arrays"] = arr

# ---- interventions ----
ev = {}
for e in ("C1", "C2", "C3", "M1", "C4", "C5", "C6", "T1"):
    k = P["events"][e]
    t1, t2 = k - 1, min(k + 6, n - 1)
    ev[e] = {"date": str(dates[k].date())}
    for i in range(3):
        ev[e][f"zone{i+1}_abs_logpct"] = diff_ci(S[i], t1, t2, scale=100)
    for i in range(2):
        ev[e][f"zone{i+1}_rel3_logpct"] = diff_ci(S[i], t1, t2, S[2], scale=100)
        ev[e][f"sens{i+1}_rel3_logpct"] = diff_ci(SIG[i], t1, t2, SIG[2], scale=100)
out["events"] = ev

# ---- inter-event slopes of the relative array condition (posterior, per day) ----
seg = {}
order = [("C4", "C5"), ("C5", "C6"), ("C6", "T1"), ("C2", "C3"), ("C3", "C4")]
for a, b in order:
    t1, t2 = P["events"][a] + 7, P["events"][b] - 1
    for i in range(2):
        seg[f"{a}-{b}_zone{i+1}_rel3_pct_d"] = diff_ci(S[i], t1, t2, S[2], scale=100 / (t2 - t1))
    seg[f"{a}-{b}_zone1_abs_pct_d"] = diff_ci(S[0], t1, t2, scale=100 / (t2 - t1))
    seg[f"{a}-{b}_days"] = t2 - t1
    seg[f"{a}-{b}_dry_share"] = float(P["dry"][t1:t2].mean())
out["segments"] = seg

# ---- identifiable PR trends of Zone 1 (contrasts): own sensor (s1 - sig1) and Zone 3 sensor (s1 - sig3) ----
t0 = 30
w_own = np.zeros(NX); w_own[S[0]] = 1; w_own[SIG[0]] = -1
w_r3 = np.zeros(NX); w_r3[S[0]] = 1; w_r3[SIG[2]] = -1


def lin_ci(w, t1, t2, scale):
    C = cov_between(t1, t2)
    mean = w @ XS[t2] - w @ XS[t1]
    sd = np.sqrt(max(w @ PS[t2] @ w + w @ PS[t1] @ w - 2 * w @ C @ w, 0))
    return [float(mean * scale), float((mean - 1.96 * sd) * scale), float((mean + 1.96 * sd) * scale)]


out["pr_trends"] = dict(own=lin_ci(w_own, t0, n - 1, 100 / yrs(t0, n - 1)), ref3=lin_ci(w_r3, t0, n - 1, 100 / yrs(t0, n - 1)))

# ---- auxiliary residuals: unlogged level shifts ----
aux = {}
logged = set()
for e, k in P["events"].items():
    if k is not None:
        logged.update(range(k - 2, k + 8))
thr = norm.ppf(1 - 0.025 / (3 * n))          # Bonferroni over zones x days
AUX = {}
for nm, idxs in (("array", S), ("sensor", SIG)):
    for i in range(3):
        r = m.aux_residuals(th, idxs[i])
        AUX[f"{nm}{i+1}"] = r
        hits = np.where(np.abs(r) > thr)[0]
        aux[f"{nm}{i+1}"] = [dict(date=str(dates[t].date()), t_stat=float(r[t]), near_logged_event=bool(t in logged)) for t in hits]
aux["threshold_abs_t"] = float(thr)
out["aux_residuals"] = aux
pd.DataFrame(AUX, index=dates).to_csv(os.path.join(RES, "aux_residuals.csv"))

# ---- variants ----
import glob
fits = {}
for fn in glob.glob(os.path.join(RES, "mssm_fit_*.json")):
    fits.update(json.load(open(fn)))
var = {}
for name in fits:
    f = pickle.load(open(os.path.join(RES, f"mssm_{name}.pkl"), "rb"))
    X, Pd = f["XS"], f["PS_diag"]
    t1 = marks["2025-01-01"]
    var[name] = dict(ll=fits[name]["ll"],
                     sig1_end=float(X[-1, SIG[0]] * 100), sig1_end_sd=float(np.sqrt(Pd[-1, SIG[0]]) * 100),
                     sig3_end=float(X[-1, SIG[2]] * 100), sig3_end_sd=float(np.sqrt(Pd[-1, SIG[2]]) * 100),
                     sig1_drift_2025on=float((X[-1, SIG[0]] - X[t1, SIG[0]]) / yrs(t1, n - 1) * 100),
                     sig13_drift_2025on=float(((X[-1, SIG[0]] - X[-1, SIG[2]]) - (X[t1, SIG[0]] - X[t1, SIG[2]])) / yrs(t1, n - 1) * 100),
                     s1_net_PLR=float(X[-1, S[0]] / yrs(30, n - 1) * 100), s3_net_PLR=float(X[-1, S[2]] / yrs(30, n - 1) * 100))
if "sensor_jump" in fits:
    lr = 2 * (fits["sensor_jump"]["ll"] - fits["main"]["ll"])
    var["LR_sensor_jump"] = dict(LR=float(lr), p_chi2_1=float(chi2.sf(max(lr, 0), 1)))
out["variants"] = var
json.dump(out, open(os.path.join(RES, "results_main.json"), "w"), indent=2, default=float)
print(json.dumps({k: out[k] for k in ("theta", "sensors", "arrays", "events", "aux_residuals", "variants")}, indent=1, default=float)[:12000])
