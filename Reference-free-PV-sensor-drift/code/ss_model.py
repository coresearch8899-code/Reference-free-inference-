# -*- coding: utf-8 -*-
"""Metrological-soiling state-space model (MSSM) for a redundant multi-zone PV network.

Observation model (day t, zone i, log space; H = seasonal harmonics of day of year):
  h_i(t) = ln Hcs + m0 + mH.H + lam_i [a(t) + L(t)] + gH_i.H + sig_i(t) + u^h_i(t) + e^h_i       (own-sensor insolation)
  y_i(t) = ln Hcs + m0 + mH.H + lam_y [a(t) + L(t)] + gY.H + kap_i + s_i(t) + bT T~(t) + u^y_i(t) + e^y_i   (final yield)
  K_ib   = c_env + gE.H + gH_i.H + sig_i(t_b) + B_b + e^K_ib     (clear-sky envelope anchor, 30-day blocks)
a(t) iid common sky; L(t) random-walk atmospheric level (interannual cloudiness); u = AR(1) short-memory deviations;
B_b iid common block effect of the envelope. Gauges: sig_3(0) = 0, s_i(0) = 0, L(0) = 0, lam_3 = 1.
State dynamics:
  sig_i(t+1) = sig_i(t) + dlt_i(t) + w^sig,   dlt_i(t+1) = dlt_i(t) + w^dlt               (local linear trend)
  s_i(t+1)   = s_i(t) + r^dry_i D(t+1) + r^wet_i (1 - D(t+1)) + w^s + J_i(t+1)              (array condition)
J_i: intervention disturbance with large variance on logged-event windows.
Truncated PR days enter as censored observations h_i <= ln(Y_f,i / 0.90) (Tobit/assumed-density Kalman update)."""
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

# ---- state layout ----
SIG, DLT, S = [0, 1, 2], [3, 4, 5], [6, 7, 8]
A, L, B = 9, 10, 11
M0 = 12
MH = list(range(13, 17))
GY = list(range(17, 21))
GH1, GH2 = list(range(21, 25)), list(range(25, 29))
KAP = [29, 30, 31]
RD, RW = [32, 33, 34], [35, 36, 37]
BT = 38
ARH, ARY = [39, 40, 41], [42, 43, 44]
ENV = 45
GE = list(range(46, 50))
NX = 50
STATIC = [M0] + MH + GY + GH1 + GH2 + KAP + RD + RW + [BT, ENV] + GE
PNAMES = ["q_dlt", "q_s", "q_L", "sd_a", "sd_h", "sd_y1", "sd_y2", "sd_y3", "sd_b", "sd_env", "sd_envw",
          "phi_h", "sd_arh", "phi_y", "sd_ary", "lam_y", "lam_h2"]
POSITIVE = ["q_dlt", "q_s", "q_L", "sd_a", "sd_h", "sd_y1", "sd_y2", "sd_y3", "sd_b", "sd_env", "sd_envw",
            "sd_arh", "sd_ary"]
BLOCK = 30
ENV_Q = 0.95


def harm(doy):
    w = 2 * np.pi * doy / 365.25
    return np.array([np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w)])


def envelope_blocks(P, h, hc, y, block=BLOCK, q=ENV_Q, max_cens=0.5, min_days=15):
    """P95 of h_i - ln Hcs per 30-day block; censored days count as -inf; blocks with >50 % censoring skipped."""
    n = P["n"]
    lnHcs = np.log(P["Hcs"])
    out = []                                        # (t_mid, zone, K)
    for b0 in range(0, n - block + 1, block):
        sl = slice(b0, b0 + block)
        for i in range(3):
            valid = np.isfinite(y[sl, i])
            nv = valid.sum()
            if nv < min_days:
                continue
            unc = np.isfinite(h[sl, i])
            cen = np.isfinite(hc[sl, i])
            if cen.sum() > max_cens * nv or unc.sum() < min_days // 2:
                continue
            v = np.where(unc, h[sl, i] - lnHcs[sl], -np.inf)[valid]
            out.append((b0 + block // 2, i, float(np.quantile(v, q))))
    return out


class MSSM:
    def __init__(self, P, events=("C1", "C2", "C3", "C4", "C5", "C6", "T1"), win=5, int_sd=0.15,
                 censor="tobit", sensor_jump=False, anchor="envelope", burn=30, extra_events=()):
        self.P = P
        self.n = P["n"]
        self.censor = censor
        self.anchor = anchor            # 'envelope' (default) | 'mean' (stationary mean clearness) | 'none'
        self.burn = burn
        self.int_var = np.zeros((self.n, 3))
        self.sint_var = np.zeros((self.n, 3))
        ev = [P["events"].get(e) for e in events] + list(extra_events)
        for k in ev:
            if k is None:
                continue
            for j in range(k, min(k + win, self.n)):
                self.int_var[j, :] = int_sd ** 2 / win
        if sensor_jump:
            for e in events:
                k = P["events"].get(e)
                if k is None or not e.startswith("C"):
                    continue
                for j in range(k, min(k + win, self.n)):
                    self.sint_var[j, :] = int_sd ** 2 / win
        self.H = np.stack([harm(d) for d in P["doy"]])
        self.lnHcs = np.log(P["Hcs"])
        self.Tt = (P["tamb"] - 25.0) / 10.0
        self._build_obs()

    # ---- observation rows (loadings of a and L filled in _load) ----
    def _row_h(self, t, i):
        z = np.zeros(NX)
        z[M0] = 1; z[MH] = self.H[t]; z[SIG[i]] = 1; z[ARH[i]] = 1
        if i == 0: z[GH1] = self.H[t]
        if i == 1: z[GH2] = self.H[t]
        return z

    def _row_y(self, t, i):
        z = np.zeros(NX)
        z[M0] = 1; z[MH] = self.H[t]; z[GY] = self.H[t]; z[KAP[i]] = 1; z[S[i]] = 1
        z[BT] = self.Tt[t]; z[ARY[i]] = 1
        return z

    def _row_k(self, t, i):
        z = np.zeros(NX)
        z[ENV] = 1; z[GE] = self.H[t]; z[SIG[i]] = 1; z[B] = 1
        if i == 0: z[GH1] = self.H[t]
        if i == 1: z[GH2] = self.H[t]
        return z

    def _load(self, Z, kinds, th):
        Z = Z.copy()
        lam_h = (1.0, th["lam_h2"], 1.0)
        for r, (k, i) in enumerate(kinds):
            if k in ("k", "kw"):
                continue
            lam = th["lam_y"] if k == "y" else lam_h[i]
            Z[r, A] = lam
            Z[r, L] = lam if self.anchor != "mean" else 0.0
        return Z

    def _build_obs(self, y=None, h=None, hc=None):
        P = self.P
        y = P["y"] if y is None else y
        h = P["h"] if h is None else h
        hc = P["hc"] if hc is None else hc
        K = {}
        if self.anchor == "envelope":
            for t, i, val in envelope_blocks(P, h, hc, y):
                K.setdefault(t, []).append((i, val))
        self.n_env = sum(len(v) for v in K.values())
        self.obs = []
        for t in range(self.n):
            Zs, vs, kinds, cz, cb, czk = [], [], [], [], [], []
            for i in range(3):
                if np.isfinite(y[t, i]):
                    Zs.append(self._row_y(t, i)); vs.append(y[t, i] - self.lnHcs[t]); kinds.append(("y", i))
            for i in range(3):
                if np.isfinite(h[t, i]):
                    Zs.append(self._row_h(t, i)); vs.append(h[t, i] - self.lnHcs[t]); kinds.append(("h", i))
                elif np.isfinite(hc[t, i]):
                    if self.censor == "tobit":
                        cz.append(self._row_h(t, i)); cb.append(hc[t, i] - self.lnHcs[t]); czk.append(("h", i))
                    elif self.censor == "naive":
                        Zs.append(self._row_h(t, i)); vs.append(hc[t, i] - self.lnHcs[t]); kinds.append(("h", i))
            for i, val in K.get(t, []):
                wet = self.P["date"][t].month in (5, 6, 7, 8, 9, 10)
                Zs.append(self._row_k(t, i)); vs.append(val); kinds.append(("kw" if wet else "k", i))
            self.obs.append((np.array(Zs) if Zs else None, np.array(vs), kinds,
                             np.array(cz) if cz else None, np.array(cb), czk))

    def set_data(self, y, h, hc):
        self._build_obs(y, h, hc)

    # ---- system matrices ----
    def _Tth(self, t, th):
        T = np.eye(NX)
        for i in range(3):
            T[SIG[i], DLT[i]] = 1.0
            T[S[i], RD[i]] = self.P["dry"][t]
            T[S[i], RW[i]] = 1.0 - self.P["dry"][t]
        T[A, A] = 0.0
        T[B, B] = 0.0
        T[ARH, ARH] = th["phi_h"]
        T[ARY, ARY] = th["phi_y"]
        return T

    def _Q(self, t, th):
        q = np.zeros(NX)
        q[SIG] = th["q_sig"] ** 2 + self.sint_var[t]
        q[DLT] = th["q_dlt"] ** 2
        q[S] = th["q_s"] ** 2 + self.int_var[t]
        q[A] = th["sd_a"] ** 2
        q[L] = th["q_L"] ** 2 if self.anchor == "envelope" else (1e2 if self.anchor == "none" else 0.0)
        q[B] = th["sd_b"] ** 2
        q[ARH] = th["sd_arh"] ** 2
        q[ARY] = th["sd_ary"] ** 2
        return q

    def _Rdiag(self, kinds, th):
        r = []
        for k, i in kinds:
            r.append(th["sd_h"] ** 2 if k == "h" else th["sd_env"] ** 2 if k == "k" else th["sd_envw"] ** 2 if k == "kw"
                     else th["sd_y%d" % (i + 1)] ** 2)
        return np.array(r)

    def init(self, th):
        x = np.zeros(NX)
        Pd = np.zeros(NX)
        Pd[STATIC] = 1e2
        Pd[[SIG[0], SIG[1]]] = 1e2           # constant sensor offsets vs Zone 3 (diffuse)
        Pd[DLT] = (1e-3) ** 2
        Pd[A] = th["sd_a"] ** 2
        Pd[B] = th["sd_b"] ** 2
        Pd[RD + RW] = (1e-2) ** 2
        Pd[ARH] = th["sd_arh"] ** 2 / max(1 - th["phi_h"] ** 2, 1e-3)
        Pd[ARY] = th["sd_ary"] ** 2 / max(1 - th["phi_y"] ** 2, 1e-3)
        return x, np.diag(Pd)

    # ---- filter ----
    def filter(self, th, store=False):
        n = self.n
        x, Pm = self.init(th)
        ll = 0.0
        if store:
            XP = np.zeros((n, NX)); PP = np.zeros((n, NX, NX)); XF = np.zeros((n, NX)); PF = np.zeros((n, NX, NX))
        for t in range(n):
            if t > 0:
                T = self._Tth(t, th)
                x = T @ x
                Pm = T @ Pm @ T.T
                Pm[np.diag_indices(NX)] += self._Q(t, th)
            if store:
                XP[t] = x; PP[t] = Pm
            Z, v, kinds, cz, cb, czk = self.obs[t]
            if Z is not None:
                Z = self._load(Z, kinds, th)
                R = self._Rdiag(kinds, th)
                PZ = Pm @ Z.T
                Sm = Z @ PZ + np.diag(R)
                e = v - Z @ x
                try:
                    Lc = np.linalg.cholesky(Sm)
                except np.linalg.LinAlgError:
                    return -1e10
                Le = np.linalg.solve(Lc, e)
                if t >= self.burn:
                    ll += -0.5 * (Le @ Le) - np.log(np.diag(Lc)).sum() - 0.5 * len(e) * np.log(2 * np.pi)
                K = np.linalg.solve(Sm, PZ.T).T
                x = x + K @ e
                Pm = Pm - K @ PZ.T
                Pm = 0.5 * (Pm + Pm.T)
            if cz is not None:                                   # Tobit update, one censored obs at a time
                for zrow, c in zip(self._load(cz, czk, th), cb):
                    mu = zrow @ x
                    pz = Pm @ zrow
                    s2 = zrow @ pz + th["sd_h"] ** 2
                    s = np.sqrt(s2)
                    al = (c - mu) / s
                    Phi = max(norm.cdf(al), 1e-300)
                    lam = norm.pdf(al) / Phi
                    Eh = mu - s * lam
                    Vh = s2 * (1 - al * lam - lam ** 2)
                    if t >= self.burn:
                        ll += np.log(Phi)
                    k = pz / s2
                    x = x + k * (Eh - mu)
                    Pm = Pm - np.outer(k, pz) + np.outer(k, k) * Vh
                    Pm = 0.5 * (Pm + Pm.T)
            if store:
                XF[t] = x; PF[t] = Pm
        if store:
            self.XP, self.PP, self.XF, self.PF = XP, PP, XF, PF
        return ll

    # ---- RTS smoother ----
    def smooth(self, th):
        self.filter(th, store=True)
        n = self.n
        XS = self.XF.copy(); PS = self.PF.copy(); self.J = np.zeros((n, NX, NX))
        for t in range(n - 2, -1, -1):
            T = self._Tth(t + 1, th)
            Pp = self.PP[t + 1]
            J = np.linalg.solve(Pp + 1e-12 * np.eye(NX), (self.PF[t] @ T.T).T).T
            self.J[t] = J
            XS[t] = self.XF[t] + J @ (XS[t + 1] - self.XP[t + 1])
            PS[t] = self.PF[t] + J @ (PS[t + 1] - Pp) @ J.T
            PS[t] = 0.5 * (PS[t] + PS[t].T)
        self.XS, self.PS = XS, PS
        return XS, PS

    def aux_residuals(self, th, idx):
        """Standardised smoothed state disturbances (auxiliary residuals) of state idx, transition t-1 -> t."""
        n = self.n
        out = np.full(n, np.nan)
        for t in range(n - 1):
            T = self._Tth(t + 1, th)
            q = self._Q(t + 1, th)[idx]
            eta = self.XS[t + 1, idx] - (T @ self.XS[t])[idx]
            C = self.PS[t + 1] @ self.J[t].T                      # Cov(x_{t+1}, x_t | Y)
            V = self.PS[t + 1] + T @ self.PS[t] @ T.T - C @ T.T - T @ C.T
            d = q - V[idx, idx]
            if d > 1e-14:
                out[t + 1] = eta / np.sqrt(d)
        return out


Q_SIG_FIX = 1e-5      # sensitivity level noise fixed: sig is (near) integrated random walk = smooth trend


def th_from(v):
    th = {"q_sig": Q_SIG_FIX}
    for k, x in zip(PNAMES, v):
        if k in POSITIVE:
            th[k] = float(np.exp(x))
        elif k.startswith("phi"):
            th[k] = float(np.tanh(x))
        else:
            th[k] = float(x)
    return th


def v_from(th):
    return np.array([np.log(th[k]) if k in POSITIVE else (np.arctanh(th[k]) if k.startswith("phi") else th[k])
                     for k in PNAMES])


V0 = v_from(dict(q_dlt=1e-5, q_s=1.1e-3, q_L=1e-3, sd_a=0.30, sd_h=0.056, sd_y1=0.017, sd_y2=0.056,
                 sd_y3=0.014, sd_b=0.03, sd_env=0.03, sd_envw=0.08, phi_h=0.87, sd_arh=0.03, phi_y=0.70, sd_ary=0.04,
                 lam_y=0.9, lam_h2=0.92))


def fit(model, v0=None, maxiter=1800, verbose=False):
    if v0 is None:
        v0 = V0
    f = lambda v: -model.filter(th_from(v))
    res = minimize(f, v0, method="Nelder-Mead", options=dict(maxiter=maxiter, xatol=1e-3, fatol=1e-2, disp=verbose,
                                                             adaptive=True))
    res2 = minimize(f, res.x, method="L-BFGS-B", options=dict(maxiter=30))
    best = res2 if res2.fun <= res.fun else res
    return best.x, -best.fun
