# -*- coding: utf-8 -*-
"""Step F9: (a) calibrated power and attribution by injection; (b) fault scenarios; (c) probability of at least one false alarm during monitoring;
(d) truly online monitoring of the case study: season and thresholds are frozen at the end of the development period, and every look uses only data available then;
(e) sensitivity of (d) to the treatment of censored weeks."""
import json, os, sys
import numpy as np
import pandas as pd
from detector import harmonics, loo, remove_season, step_search, pipeline, calibrate, conditional, decide, block_rows
from injection import RES, MODE, SUF, DEV, X, noise, thresholds, run

rng = np.random.default_rng(20261109)
NREP = int(os.environ.get("NREP", 1000)); BOOT = int(os.environ.get("BOOT", 1000))

# ---- (a) power and attribution
rows = []
for N in (2, 3, 6):
    sc = np.ones(N)
    for test in (52, 78, 104, 156):
        q = thresholds(test, N, sc, BOOT, rng)
        for D in (0, 1, 2, 3, 5, 10):
            r = run(test, N, sc, q, rng, NREP, {0: -D} if D else None)
            rows.append(dict(N=N, months=round(test / 4.348), drift=D, s_threshold=q["s"], **r))
        print(N, test, q, flush=True)
G = pd.DataFrame(rows); G.to_csv(os.path.join(RES, f"calibrated_grid{SUF}.csv"), index=False)

# ---- (b) scenarios, 5 % per year, 18 months of test data
test = 78; rows = []
SCEN = {"No fault": (3, np.ones(3), None, None),
        "One sensor drifts": (3, np.ones(3), {0: -5}, None),
        "One sensor drifts, it is the noisiest (noise 1 : 1.5 : 2)": (3, np.array([2, 1, 1.5]), {0: -5}, None),
        "One sensor drifts, it is the quietest (noise 1 : 1.5 : 2)": (3, np.array([1, 1.5, 2]), {0: -5}, None),
        "Two sensors drift together": (3, np.ones(3), {0: -5, 1: -5}, None),
        "Two sensors drift together, five sensors": (5, np.ones(5), {0: -5, 1: -5}, None),
        "All sensors drift alike": (3, np.ones(3), {0: -5, 1: -5, 2: -5}, None)}
for lab, (N, sc, dr, sh) in SCEN.items():
    q = thresholds(test, N, sc, BOOT, rng)
    rows.append(dict(scenario=lab, N=N, **run(test, N, sc, q, rng, NREP, dr, sh)))
    print(rows[-1], flush=True)
Sc = pd.DataFrame(rows); Sc.to_csv(os.path.join(RES, f"scenarios{SUF}.csv"), index=False)

# ---- (c) repeated looks: probability of at least one false alarm during 24 months of monthly looks, N = 3
looks = list(range(26, 105, 4)); N = 3; sc = np.ones(3)
qs = {w: thresholds(w, N, sc, max(300, BOOT // 2), rng)["s"] for w in looks}
mx = np.empty(NREP); anyl = np.empty(NREP); stepruns = 0; cc = {}
for i in range(NREP):
    C = loo(noise(DEV + looks[-1], N, sc, rng)); vals = []; found = False
    for w in looks:                                     # step search and conditional thresholds are repeated at every look, as in real monitoring
        r = pipeline(C[:DEV + w], DEV, w, X); q_ = qs[w]
        if r["inwin"]:
            found = True
            key = (w, tuple(sorted((k, t_, round(d, 1)) for k, t_, d in r["inwin"])))
            if key not in cc:
                cc[key] = thresholds(w, N, sc, 200, rng, add=r["inwin"])["s"]
            q_ = cc[key]
        vals.append(0.0 if r["unresolved"] else r["S"] / q_)
    vals = np.array(vals); stepruns += found
    mx[i] = vals.max(); anyl[i] = (vals > 1).any()
fw = float(np.quantile(mx, .95))
json.dump(dict(n_looks=len(looks), per_look_alpha=0.05, prob_any_false_alarm=float(anyl.mean()), family_multiplier=fw, share_runs_with_step=stepruns / NREP), open(os.path.join(RES, f"monitoring_false_alarm{SUF}.json"), "w"), indent=1)
print("P(any false alarm)", anyl.mean(), "multiplier", fw, flush=True)

if MODE == "detrended":
    # ---- (d) online monitoring of the case study (no future data: season, residuals and thresholds frozen at the end of development)
    
    from ss_data import load_panel
    try:
        from ss_data import load_power
    except ImportError:
        from ss_data import load_power
    P = load_panel(); d = pd.DatetimeIndex(P["date"])
    Sg = pd.DataFrame({"sensor 1": P["h"][:, 0], "sensor 3": P["h"][:, 2], "satellite": np.log(load_power(P["date"]).ALLSKY_SFC_SW_DWN.values)}, index=d)
    Wr = (Sg - Sg.mean()).resample("7D")
    W = Wr.mean().interpolate(limit_direction="both")
    days = Wr.count()
    devend = int(W.index.searchsorted(pd.Timestamp("2024-07-01")))
    XX = harmonics(W.index.dayofyear.values, 1)
    C0 = loo(W.values)
    Cd, _ = step_search(remove_season(C0[:devend], XX[:devend], devend))
    E = Cd - Cd.mean(0)
    lens = list(range(26, len(C0) - devend + 1, 4))

    def monitor(lens, mask=None, seed=7):
        """lens: evaluation lengths of the looks; mask: weeks usable in the slope fit; a look that reaches censored weeks without a mask uses held values"""
        rngb = np.random.default_rng(seed)
        kp = (lambda w: None) if mask is None else (lambda w: mask[devend:devend + w])
        Q = {w: calibrate(E, devend, w, XX, rngb, B=max(400, BOOT // 2), keep=kp(w)) for w in lens}
        mxb = np.empty(NREP); any_b = np.empty(NREP); cache = {}; stepruns = 0
        for i in range(NREP):                               # the sequence of looks repeats step search and conditional recalibration
            Cs = E[block_rows(len(E), devend + lens[-1], rngb)]; v = []; found = False
            for w in lens:
                r = pipeline(Cs[:devend + w], devend, w, XX, keep=kp(w))
                found = found or bool(r["inwin"])
                qw = conditional(r, Q[w], E, devend, w, XX, rngb, 200, cache, keep=kp(w))
                v.append(0.0 if r["unresolved"] else r["S"] / qw["s"])
            v = np.array(v); stepruns += found
            mxb[i] = v.max(); any_b[i] = (v > 1).any()
        mult = float(np.quantile(mxb, .95)); out = []; cache = {}
        for w in lens:
            r = pipeline(C0[:devend + w], devend, w, XX, keep=kp(w))
            qw = conditional(r, Q[w], E, devend, w, XX, rngb, max(400, BOOT // 2), cache, keep=kp(w))
            st, kk = decide(r, qw)
            q2 = dict(qw); q2["s"] = mult * qw["s"]
            st2, kk2 = decide(r, q2)
            out.append(dict(week_end=str(W.index[devend + w - 1].date()), slopes=[round(float(x), 1) for x in r["b"]], threshold=round(qw["s"], 1), family_threshold=round(q2["s"], 1),
                            state=st, sensor=list(W.columns)[kk] if kk >= 0 else "", state_family=st2, sensor_family=list(W.columns)[kk2] if kk2 >= 0 else "", steps_in_window=len(r["inwin"]), unresolved=int(r["unresolved"])))
        return out, mult, float(any_b.mean()), stepruns / NREP

    def summary(out):
        return dict(looks=len(out), first_alarm=next((o["week_end"] for o in out if o["state"] > 0), None), first_attrib=next((o["week_end"] for o in out if o["state"] == 2), None),
                    first_family_alarm=next((o["week_end"] for o in out if o["state_family"] > 0), None), first_family_attrib=next((o["week_end"] for o in out if o["state_family"] == 2), None),
                    attributed_sensor1=sum(o["state_family"] == 2 and o["sensor_family"] == "sensor 1" for o in out), attributed_other=sum(o["state_family"] == 2 and o["sensor_family"] != "sensor 1" for o in out),
                    ambiguous=sum(o["state_family"] == 1 for o in out), last_slopes=out[-1]["slopes"], last_look=out[-1]["week_end"])

    # main analysis: no value is imputed; reporting stops when a look ends on censored weeks
    ok = (days.values >= 3).all(1)
    print("weeks after development with fewer than 3 usable days in a signal:", [str(x.date()) for x in W.index[devend:][~ok[devend:]]], flush=True)
    bad = {w: not ok[devend + w - 4:devend + w].all() for w in lens}
    valid = [w for w in lens if not bad[w]]
    out, mult, pany, srun = monitor(valid, seed=10)
    O = pd.DataFrame(out); O.to_csv(os.path.join(RES, "online_calibrated.csv"), index=False)
    json.dump(dict(dev_end="2024-06-30", dev_weeks=devend, prob_any_false_alarm=pany, family_multiplier=mult, share_runs_with_step=srun, **summary(out)), open(os.path.join(RES, "online_calibrated.json"), "w"), indent=1)
    print(open(os.path.join(RES, "online_calibrated.json")).read()); print(O.to_string())

    # ---- (e) sensitivity to censored weeks
    rows = [dict(variant="Reporting stopped when a look ends on censored weeks (main analysis)", **summary(out), family_multiplier=mult)]
    o2, m2, _, _ = monitor(lens, mask=ok, seed=8)
    rows.append(dict(variant="Only weeks with at least 3 usable days", **summary(o2), family_multiplier=m2))
    o3, m3, _, _ = monitor(lens, seed=7)
    rows.append(dict(variant="Last value held", **summary(o3), family_multiplier=m3))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "online_sensitivity.csv"), index=False)
    print(pd.DataFrame(rows).to_string())
