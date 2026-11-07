# -*- coding: utf-8 -*-
"""Step 3: theory-driven quantities.
 (a) Minimum detectable drift (MDD) and detection time for a contrast versus a clear-sky anchor (Prop. 3),
     with a Monte-Carlo check of the closed form.
 (b) Real-time (filtered) detection date of the Zone 1 divergence.
 (c) Redundancy-aware recalibration: posterior variance of an absolute sensitivity under stand-alone,
     synchronous-network and staggered-network calibration policies (Riccati recursion, Prop. 4)."""
import json, os, pickle
import numpy as np
from scipy.stats import norm
from ss_data import load_panel, ROOT
from ss_model import MSSM, DLT, SIG, harm

RES = os.path.join(ROOT, "results")
rng = np.random.default_rng(7)
P = load_panel()
fitm = pickle.load(open(os.path.join(RES, "mssm_main.pkl"), "rb"))
th = fitm["theta"]
n = P["n"]
H = np.stack([harm(d) for d in P["doy"]])
out = {}

# ---------- (a) MDD ----------
pre = P["date"] < np.datetime64("2025-01-01")
lnHcs = np.log(P["Hcs"])


def ar1_resid(series, mask):
    ok = mask & np.isfinite(series)
    X = np.column_stack([np.ones(n), H])[ok]
    r = series[ok] - X @ np.linalg.lstsq(X, series[ok], rcond=None)[0]
    # lag-1 on consecutive days only
    idx = np.where(ok)[0]
    cons = np.diff(idx) == 1
    phi = np.corrcoef(r[:-1][cons], r[1:][cons])[0, 1]
    return float(np.std(r, ddof=1)), float(phi)


def n_detect(delta_per_day, sd, phi, alpha=0.05, power=0.8):
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return (z ** 2 * 12 * sd ** 2 * (1 + phi) / ((1 - phi) * delta_per_day ** 2)) ** (1 / 3)


h, y = P["h"], P["y"]
series = {"contrast_h1_h3": h[:, 0] - h[:, 2],
          "anchor_h1_clearsky": h[:, 0] - lnHcs,
          "anchor_h3_clearsky": h[:, 2] - lnHcs,
          "yield_ratio_y1_y3": y[:, 0] - y[:, 2]}
mdd = {}
for k, s in series.items():
    sd, phi = ar1_resid(s, pre)
    mdd[k] = dict(sd=sd, phi=phi)
    for rate in (0.02, 0.05, 0.089):
        mdd[k][f"days_to_detect_{int(rate*1000)}permil_yr"] = float(n_detect(rate / 365.25, sd, phi))
out["mdd"] = mdd

# Monte-Carlo check of the closed form (power at n_d should be ~0.80)
mc = {}
for k in ("contrast_h1_h3", "anchor_h1_clearsky"):
    sd, phi = mdd[k]["sd"], mdd[k]["phi"]
    d = 0.05 / 365.25
    nd = int(round(n_detect(d, sd, phi)))
    rej = 0
    nsim = 2000
    t = np.arange(nd)
    for _ in range(nsim):
        e = np.empty(nd); e[0] = rng.normal(0, sd)
        innov = rng.normal(0, sd * np.sqrt(1 - phi ** 2), nd)
        for j in range(1, nd):
            e[j] = phi * e[j - 1] + innov[j]
        yy = d * t + e
        X = np.column_stack([np.ones(nd), t])
        b = np.linalg.lstsq(X, yy, rcond=None)[0]
        r = yy - X @ b
        ph = np.corrcoef(r[:-1], r[1:])[0, 1]
        se = np.sqrt(np.var(r, ddof=2) / np.sum((t - t.mean()) ** 2) * (1 + ph) / (1 - ph))
        rej += abs(b[1] / se) > 1.96
    mc[k] = dict(n_days=nd, empirical_power=rej / nsim)
out["mdd_mc_check"] = mc

# ---------- (b) real-time detection ----------
m = MSSM(P)
m.filter(th, store=True)
dl = m.XF[:, DLT[0]] * 365.25
sd = np.sqrt(m.PF[:, DLT[0], DLT[0]]) * 365.25
flag = (dl + 1.96 * sd) < 0
run = 0; det = None
for t in range(n):
    run = run + 1 if flag[t] else 0
    if run >= 14 and det is None:
        det = t - 13
out["online_detection"] = dict(first_day=str(P["date"][det].date()) if det is not None else None,
                               filtered_drift_at_detection_pct_yr=float(dl[det] * 100) if det is not None else None)
np.save(os.path.join(RES, "online_drift.npy"), np.column_stack([dl, sd]))

# ---------- (c) recalibration policies ----------
q_s, q_d = th["q_sig"] ** 2, th["q_dlt"] ** 2
v_c = th["sd_h"] ** 2 + th["sd_arh"] ** 2 / (1 - th["phi_h"]) ** 2      # long-run variance of a daily sensor deviation
v_K = th["sd_b"] ** 2 + 0.5 * (th["sd_env"] ** 2 + th["sd_envw"] ** 2)  # one envelope pseudo-observation
u_c = 0.01                      # standard uncertainty of a field recalibration (1 %, k = 1)


def policy_var(N, Tcal, mode, anchor, years=12, q_d=q_d):
    """posterior s.d. of sensor-1 absolute log-sensitivity (mean, max over steady cycles).
    mode: 'standalone' (one sensor, no contrasts), 'sync' (N sensors calibrated together every Tcal),
    'stagger' (one of N sensors calibrated every Tcal/N). Contrasts: daily h_k = c(t) + sig_k + e, c diffuse.
    anchor: monthly clear-sky envelope pseudo-observation of every sensor (common block effect + noise)."""
    days = int(years * 365.25)
    Ns = 1 if mode == "standalone" else N
    nx = 2 * Ns + 2                              # (sig, dlt) per sensor, c(t), block effect b
    Tm = np.eye(nx)
    for k in range(Ns):
        Tm[2 * k, 2 * k + 1] = 1
    Tm[-2, -2] = 0; Tm[-1, -1] = 0
    Q = np.zeros(nx)
    Q[0:2 * Ns:2] = q_s; Q[1:2 * Ns:2] = q_d; Q[-2] = 1e4; Q[-1] = th["sd_b"] ** 2
    Pm = np.diag(np.r_[np.tile([u_c ** 2, (1e-5) ** 2], Ns), Q[-2], Q[-1]])
    v = np.zeros(days)
    for t in range(days):
        Pm = Tm @ Pm @ Tm.T + np.diag(Q)
        Z, R = [], []
        if Ns > 1:
            for k in range(Ns):
                z = np.zeros(nx); z[2 * k] = 1; z[-2] = 1; Z.append(z); R.append(v_c)
        if anchor and t % 30 == 15:
            for k in range(Ns):
                z = np.zeros(nx); z[2 * k] = 1; z[-1] = 1; Z.append(z); R.append(v_K - th["sd_b"] ** 2)
        for k in range(Ns):
            due = (t % Tcal == 0) if mode != "stagger" else ((t - (k * Tcal) // N) % Tcal == 0)
            if due and t > 0:
                z = np.zeros(nx); z[2 * k] = 1; Z.append(z); R.append(u_c ** 2)
        if Z:
            Z = np.array(Z); Sm = Z @ Pm @ Z.T + np.diag(R)
            K = Pm @ Z.T @ np.linalg.inv(Sm)
            Pm = Pm - K @ Z @ Pm
            Pm = 0.5 * (Pm + Pm.T)
        v[t] = Pm[0, 0]
    last = v[-int(2 * Tcal):]        # steady cycles
    return float(np.sqrt(last.mean())), float(np.sqrt(last.max()))


q_d_nom = 1e-4 * 3 / 365.25 ** 3      # healthy sensor: 1 % (1 s.d.) drift-induced error after one year
pol_nom = {}
for Tcal_y in (1, 2, 3, 5):
    Tc = int(Tcal_y * 365.25)
    pol_nom[f"T{Tcal_y}y"] = {"standalone": policy_var(1, Tc, "standalone", False, q_d=q_d_nom),
                              "standalone_clearsky": policy_var(1, Tc, "standalone", True, q_d=q_d_nom),
                              "sync_N3_noanchor": policy_var(3, Tc, "sync", False, q_d=q_d_nom),
                              "stagger_N3_noanchor": policy_var(3, Tc, "stagger", False, q_d=q_d_nom),
                              "stagger_N3_anchor": policy_var(3, Tc, "stagger", True, q_d=q_d_nom)}
    print("nominal", Tcal_y, {k: tuple(round(100 * x, 2) for x in v) for k, v in pol_nom[f"T{Tcal_y}y"].items()}, flush=True)
out["policy_nominal_sd_mean_max"] = pol_nom
pol = {}
for Tcal_y in (1, 2, 3, 5):
    Tc = int(Tcal_y * 365.25)
    row = {}
    row["standalone"] = policy_var(1, Tc, "standalone", anchor=False)
    row["standalone_clearsky"] = policy_var(1, Tc, "standalone", anchor=True)
    for N in (2, 3, 5):
        row[f"sync_N{N}_noanchor"] = policy_var(N, Tc, "sync", anchor=False)
        row[f"stagger_N{N}_noanchor"] = policy_var(N, Tc, "stagger", anchor=False)
        row[f"stagger_N{N}_anchor"] = policy_var(N, Tc, "stagger", anchor=True)
    pol[f"T{Tcal_y}y"] = row
    print(Tcal_y, {k: tuple(round(100 * x, 2) for x in v) for k, v in row.items()}, flush=True)
out["policy_sd_mean_max"] = pol


# closed-form stand-alone admissible interval (Prop. 4): u_c^2 + q_s T + q_d T^3/3 = (tau/1.96)^2
def T_tau(tau):
    from scipy.optimize import brentq
    f = lambda T: u_c ** 2 + q_s * T + q_d * T ** 3 / 3 - (tau / 1.96) ** 2
    return brentq(f, 1e-6, 1e6) if f(1e6) > 0 else np.inf


out["admissible_interval_days_standalone"] = {f"tau_{tau}": float(T_tau(tau)) for tau in (0.02, 0.03, 0.05)}
out["params"] = dict(q_sig=th["q_sig"], q_dlt=th["q_dlt"], v_c=v_c, v_K=v_K, u_c=u_c, q_d_nom=q_d_nom)
out["admissible_interval_days_nominal"] = {}
for tau in (0.02, 0.03, 0.05):
    from scipy.optimize import brentq
    f = lambda T: u_c ** 2 + q_s * T + q_d_nom * T ** 3 / 3 - (tau / 1.96) ** 2
    out["admissible_interval_days_nominal"][f"tau_{tau}"] = float(brentq(f, 1e-6, 1e6)) if f(1e6) > 0 else None
json.dump(out, open(os.path.join(RES, "theory.json"), "w"), indent=2, default=float)
print(json.dumps({k: out[k] for k in ("mdd", "mdd_mc_check", "online_detection", "admissible_interval_days_standalone", "params")}, indent=2, default=float))
