# -*- coding: utf-8 -*-
"""Step 1: MLE of the MSSM (main specification) and of the comparison variants; smoothed states."""
import json, os, pickle, time
import numpy as np
from ss_data import load_panel, ROOT
from ss_model import MSSM, fit, th_from, PNAMES, SIG, DLT, S, RD, RW, KAP, BT

RES = os.path.join(ROOT, "results")
P = load_panel()
VARIANTS = {
    "main": dict(),
    "drop_truncated": dict(censor="drop"),
    "naive_truncated": dict(censor="naive"),
    "sensor_jump": dict(sensor_jump=True),
    "mean_anchor": dict(anchor="mean"),
    "no_anchor": dict(anchor="none"),
}
ONLY = os.environ.get("ONLY")
if ONLY:
    VARIANTS = {k: v for k, v in VARIANTS.items() if k in ONLY.split(",")}
out = {}
v_main = None
if os.path.exists(os.path.join(RES, "mssm_main.pkl")) and "main" not in VARIANTS:
    v_main = pickle.load(open(os.path.join(RES, "mssm_main.pkl"), "rb"))["v"]
for name, kw in VARIANTS.items():
    t0 = time.time()
    m = MSSM(P, **kw)
    v, ll = fit(m, v0=v_main)
    if name == "main":
        v_main = v
    th = th_from(v)
    m.smooth(th)
    k = len(PNAMES)
    out[name] = dict(ll=ll, aic=-2 * ll + 2 * k, theta=th, sec=time.time() - t0)
    with open(os.path.join(RES, f"mssm_{name}.pkl"), "wb") as f:
        pickle.dump(dict(theta=th, v=v, ll=ll, XS=m.XS, PS_diag=np.array([np.diag(p) for p in m.PS]),
                         PS_sig_dlt=m.PS[:, :9, :9], kw=kw), f)
    xs, ps = m.XS[-1], m.PS[-1]
    print(f"[{name}] ll={ll:.2f} ({time.time()-t0:.0f}s) theta=" + ", ".join(f"{a}={b:.3g}" for a, b in th.items()))
    for i in range(3):
        print(f"   zone{i+1}: sig_end={xs[SIG[i]]:+.4f}±{np.sqrt(ps[SIG[i],SIG[i]]):.4f}  "
              f"r_dry={xs[RD[i]]*100:+.3f}±{np.sqrt(ps[RD[i],RD[i]])*100:.3f} %/d  "
              f"r_wet={xs[RW[i]]*100:+.3f}±{np.sqrt(ps[RW[i],RW[i]])*100:.3f} %/d")
for k, v in out.items():
    json.dump({k: dict(ll=v["ll"], aic=v["aic"], theta=v["theta"])}, open(os.path.join(RES, f"mssm_fit_{k}.json"), "w"), indent=2)
