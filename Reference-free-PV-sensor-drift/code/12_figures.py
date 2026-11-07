# -*- coding: utf-8 -*-
"""Step F7: figures of the framework paper (16 cm wide, 8 pt, 600 dpi, Okabe-Ito colours)."""
import ast, json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from ss_data import load_panel
from ss_model import SIG, DLT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results"); FIG = os.path.join(ROOT, "figs")
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.linewidth": 0.6, "lines.linewidth": 1.1, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
CM = 1 / 2.54
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "verm": "#D55E00", "sky": "#56B4E9", "pink": "#CC79A7", "grey": "0.45"}


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=600, bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")          # vector file for submission
    fig.savefig(os.path.join(FIG, name + ".tif"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def panel(ax, s):
    """panel letter only, centred; the description belongs in the caption"""
    ax.set_title(s.split(")")[0] + ")", fontsize=9, pad=3, loc="center")


def fig_framework():
    fig, ax = plt.subplots(figsize=(16 * CM, 7.6 * CM)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 50)

    def box(x, y, w, h, title, body, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0", fc=fc, ec="0.3", lw=0.6))
        ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=7.3, weight="bold")
        ax.text(x + w / 2, y + h - 5.4, body, ha="center", va="top", fontsize=7.2, linespacing=1.4)
    box(0.5, 27, 22, 22, "Inputs", "chain indications\nclipped PR bounds\nsatellite series\ncleaning dates", "#eef3fa")
    box(26, 27, 23, 22, "Conditions", "identifiable quantities\n3 signals, one fault\nno absolute values", "#f6efe6")
    box(52.5, 27, 22, 22, "Decision rules", "frozen rules\nbootstrap thresholds\nrecovery ratio\ninterval bias", "#eaf5ee")
    box(78, 27, 21.5, 22, "Outcomes", "no inconsistency\nno attribution\nsensor attributed\nratio d/r", "#f2f2f2")
    box(13, 2, 34, 19, "Main evidence", "known faults in real noise\nthresholds learned per replicate\nforward test, public network", "#f2f2f2")
    box(53, 2, 34, 19, "Demonstration", "three-zone rooftop system\nmonitoring with frozen rules\nsatellite comparison series", "#f2f2f2")
    for x0, x1 in ((23.2, 25.6), (49.6, 52.1), (75.2, 77.6)):
        ax.annotate("", (x1, 38), (x0, 38), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.3"))
    ax.annotate("", (30, 21.6), (63, 26.6), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.3"))
    ax.annotate("", (70, 21.6), (70, 26.6), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.3"))
    save(fig, "Figure_1")


def fig_power():
    g = pd.read_csv(os.path.join(RES, "calibrated_grid.csv")); u = pd.read_csv(os.path.join(RES, "calibrated_grid_raw.csv"))
    fig, ax = plt.subplots(2, 2, figsize=(16 * CM, 10.5 * CM)); ax = ax.ravel()
    sty = {1: (C["sky"], ":", "v"), 2: (C["blue"], "--", "s"), 3: (C["green"], "-.", "^"), 5: (C["orange"], "-", "o"), 10: (C["verm"], "-", "D")}
    for a_, df, tt in ((ax[0], g, "(a) detection, lower noise bound"), (ax[1], u, "(b) detection, upper noise bound")):
        for D, (c_, ls, mk) in sty.items():
            s = df[(df.N == 3) & (df.drift == D)].sort_values("months")
            a_.plot(s.months, 1 - s.no, color=c_, ls=ls, marker=mk, ms=3, label=f"{D} % yr⁻¹")
        a_.axhline(0.05, color="k", lw=0.4, ls=":"); a_.set_ylim(0, 1.03); a_.set_xticks([12, 18, 24, 36]); a_.set_xlabel("Evaluation period (months)"); a_.set_ylabel("Runs with a drift detected"); panel(a_, tt)
    ax[0].legend(fontsize=7, loc="lower right")
    s = g[(g.N == 3) & (g.months == 18)].sort_values("drift"); x = np.arange(len(s)); bot = np.zeros(len(s))
    for col, lab, c_, h_ in (("right", "correct sensor named", C["green"], None), ("amb", "detected, ambiguous", C["sky"], "///"), ("wrong", "wrong sensor named", C["verm"], "xxx"), ("no", "no detection", "0.85", None)):
        ax[2].bar(x, s[col], 0.68, bottom=bot, color=c_, hatch=h_, edgecolor="w", lw=0.4, label=lab); bot += s[col].values
    ax[2].set_xticks(x); ax[2].set_xticklabels([f"{d:g}" for d in s.drift]); ax[2].set_ylim(0, 1); ax[2].set_xlabel("Injected drift (% yr⁻¹)"); ax[2].set_ylabel("Share of runs"); ax[2].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2); panel(ax[2], "(c) outcomes, 18 months, lower bound")
    for df, lab, ls, mk in ((g, "lower bound", "-", "o"), (u, "upper bound", "--", "s")):
        s = df[(df.N == 3) & (df.drift == 0)].sort_values("months")
        ax[3].plot(s.months, s.nom_alarm, color=C["verm"], ls=ls, marker=mk, ms=3, label=f"nominal, {lab}")
        ax[3].plot(s.months, 1 - s.no, color=C["blue"], ls=ls, marker=mk, ms=3, label=f"bootstrap, {lab}")
    ax[3].axhline(0.05, color="k", lw=0.4, ls=":"); ax[3].set_xticks([12, 18, 24, 36]); ax[3].set_ylim(0, None); ax[3].set_xlabel("Evaluation period (months)"); ax[3].set_ylabel("False-alarm rate"); ax[3].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2); panel(ax[3], "(d) false alarms, N = 3")
    fig.tight_layout(w_pad=1.2, h_pad=1.0); save(fig, "Figure_2")


def fig_baselines():
    b = pd.read_csv(os.path.join(RES, "baselines.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(16 * CM, 6.6 * CM))
    M = [("Sensor contrast (N = 2)", "Sensor contrast", C["blue"], "-", "o"), ("Satellite ratio (CERES)", "Satellite ratio", C["orange"], "--", "s"),
         ("Clear-sky ratio, clearest 40 % of days", "Clear-sky, clear days", C["green"], "-.", "^"), ("Clear-sky ratio, all days", "Clear-sky, all days", C["verm"], ":", "v")]
    for key, lab, c_, ls, mk in M:
        s = b[(b.method == key) & (b.interannual_sd == 0)].sort_values("T_months")
        ax[0].plot(s.T_months, s.adj_2, color=c_, ls=ls, marker=mk, ms=2.5, label=lab)
        s2 = b[(b.method == key) & (b.interannual_sd > 0)].sort_values("T_months")
        ax[1].plot((s2 if len(s2) else s).T_months, (s2 if len(s2) else s).adj_2, color=c_, ls=ls, marker=mk, ms=2.5)
        ax[2].plot((s2 if len(s2) else s).T_months, (s2 if len(s2) else s).far, color=c_, ls=ls, marker=mk, ms=2.5)
    fig.legend(*ax[0].get_legend_handles_labels(), fontsize=7, loc="lower center", ncol=4); ax[0].set_ylabel("Power at 2 % yr⁻¹")
    for a, t in zip(ax[:2], ("(a) stable reference", "(b) varying reference")):
        a.set_ylim(0, 1.03); a.set_xlabel("Record length (months)"); a.axhline(0.8, color="k", lw=0.4, ls="--"); panel(a, t)
    ax[2].axhline(0.05, color="k", lw=0.4, ls=":"); ax[2].set_ylim(0, 0.65); ax[2].set_xlabel("Record length (months)"); ax[2].set_ylabel("False-alarm rate"); panel(ax[2], "(c) false alarms in (b)")
    fig.tight_layout(w_pad=1.0, rect=(0, 0.09, 1, 1)); save(fig, "Figure_3")


def fig_case():
    P = load_panel(); d = P["date"]
    X = np.load(os.path.join(RES, "smoothed_main.npy")); V = np.load(os.path.join(RES, "smoothed_main_var.npy"))
    O = pd.read_csv(os.path.join(RES, "online_calibrated.csv")); RS = json.load(open(os.path.join(RES, "reset_case_study.json")))
    fig = plt.figure(figsize=(16 * CM, 11.5 * CM)); gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1])
    a0 = fig.add_subplot(gs[0, :2]); a1 = fig.add_subplot(gs[0, 2]); a2 = fig.add_subplot(gs[1, :2]); a3 = fig.add_subplot(gs[1, 2])
    zc = [C["blue"], C["orange"], C["green"]]; ls = ["-", "--", ":"]
    for i in range(3):
        mu, sd = X[:, SIG[i]] * 100, np.sqrt(V[:, SIG[i]]) * 100
        a0.fill_between(d, mu - 1.96 * sd, mu + 1.96 * sd, color=zc[i], alpha=0.15, lw=0); a0.plot(d, mu, color=zc[i], ls=ls[i], label=f"Sensor {i+1}")
    a0.set_ylabel("Log sensitivity (%)"); a0.legend(ncol=3, fontsize=6, loc="lower left"); a0.xaxis.set_major_locator(mdates.YearLocator()); a0.xaxis.set_major_formatter(mdates.DateFormatter("%Y")); panel(a0, "(a) state-space estimate")
    for k, i, lab in (("sat_sensor1", 0, "Sensor 1"), ("sat_sensor3", 2, "Sensor 3")):
        for j, per in enumerate(("pre2025", "from2025")):
            v = RS[f"{k}_{per}"]; x = j + (0.12 if i == 2 else -0.12)
            a1.errorbar(x, v["slope_pct_yr"], yerr=[[v["slope_pct_yr"] - v["ci"][0]], [v["ci"][1] - v["slope_pct_yr"]]], fmt="o" if i == 0 else "s", ms=3, capsize=2, color=zc[i], lw=0.8, label=lab if j == 0 else None)
    a1.axhline(0, color="k", lw=0.4); a1.set_xticks([0, 1]); a1.set_xticklabels(["before\n2025", "from\n2025"]); a1.set_xlim(-0.5, 1.5); a1.set_ylabel("Trend vs CERES (% yr⁻¹)"); a1.set_ylim(-29, 5); a1.legend(fontsize=7, loc="lower left"); panel(a1, "(b) satellite check")
    m = pd.to_datetime(O.week_end); S = O.slopes.map(lambda s: max(abs(v) for v in ast.literal_eval(s)))
    a2.plot(m, S, color="0.55", lw=0.8, zorder=1)
    a2.plot(m, O.threshold, color=C["grey"], ls=":", label="per-look threshold"); a2.plot(m, O.family_threshold, color="k", ls="--", lw=0.8, label="family-wise threshold")
    for st_, lab, mk, c_ in ((0, "no inconsistency", "o", "0.75"), (1, "detected, ambiguous", "s", C["sky"]), (2, "attributed to sensor 1", "D", C["verm"])):
        q = O.state_family == st_
        a2.scatter(m[q], S[q], marker=mk, s=9, color=c_, label=lab, zorder=2, lw=0)
    a2.set_yscale("log"); a2.set_yticks([2, 5, 10, 20]); a2.set_yticklabels(["2", "5", "10", "20"]); a2.minorticks_off(); a2.set_ylabel("Largest |slope| (% yr⁻¹)"); a2.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17), columnspacing=0.9, handletextpad=0.4); a2.xaxis.set_major_formatter(mdates.DateFormatter("%b %y")); a2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7])); panel(a2, "(c) monitoring with frozen rules")
    RS2 = pd.read_csv(os.path.join(RES, "reset_sensitivity.csv")); lim = (-1.2, 1.5)
    labs = ["All five", "Without C2", "Without C3", "Without C4", "Without C5", "Without C6", "No final seg."]
    for sig, c_, mk, dy in (("Sensor contrast", C["blue"], "o", 0.14), ("Yield ratio", C["orange"], "s", -0.14)):
        v = RS2[RS2.signal == sig].ratio.values; yy = np.arange(len(v))[::-1] + dy
        a3.scatter(np.clip(v, *lim), yy, marker=mk, s=11, color=c_, label=sig, lw=0)
        for vi, yi in zip(v, yy):
            if vi > lim[1]:
                a3.text(lim[1] - 0.03, yi + 0.3, f"{vi:.1f}", ha="right", va="center", fontsize=6.5)
    a3.axvline(0, color="k", lw=0.4); a3.axvline(1, color="k", lw=0.4, ls=":"); a3.set_xlim(-1.3, 1.6); a3.set_yticks(range(7)); a3.set_yticklabels(labs[::-1], fontsize=7); a3.set_xlabel("Recovery ratio"); a3.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.2, -0.4), ncol=2, handletextpad=0.1, columnspacing=0.6, borderaxespad=0); panel(a3, "(d) reset sensitivity")
    fig.tight_layout(h_pad=0.8, w_pad=0.8); save(fig, "Figure_5")


def fig_bias():
    fig, ax = plt.subplots(1, 2, figsize=(16 * CM, 5.4 * CM))
    x = np.linspace(0.001, 0.97, 300)
    ax[0].plot(x, 1 / np.sqrt(1 - x), color=C["blue"]); ax[0].set_ylim(1, 6); ax[0].set_xlim(0, 1)
    ax[0].set_xlabel("Hidden drift / true soiling rate, d/r"); ax[0].set_ylabel("Chosen / optimal interval"); panel(ax[0], "(a) interval bias")
    u = np.sqrt(1 - x); ax[1].plot(x, ((u + 1 / u) / 2 - 1) * 100, color=C["verm"]); ax[1].set_yscale("log"); ax[1].set_ylim(0.01, 300); ax[1].set_xlim(0, 1)
    ax[1].set_xlabel("Hidden drift / true soiling rate, d/r"); ax[1].set_ylabel("Excess daily cost (%)"); panel(ax[1], "(b) cost penalty")
    t = pd.read_csv(os.path.join(RES, "cleaning_bias.csv"))
    for _, r_ in t[t.drift_pct_yr.isin([8.9, 12.9]) & t.season.isin(["R13 dry", "R13 wet"])].iterrows():
        q = (r_.r_pct_d - r_.r_apparent) / r_.r_pct_d
        mk = "o" if "dry" in r_.season else "s"
        ax[0].plot(q, 1 / np.sqrt(1 - q), mk, ms=3.5, color=C["orange"] if "dry" in r_.season else C["green"])
        ax[1].plot(q, r_.cost_excess_pct, mk, ms=3.5, color=C["orange"] if "dry" in r_.season else C["green"])
    ax[0].plot([], [], "o", color=C["orange"], ms=3.5, label="illustrative, dry-season rate"); ax[0].plot([], [], "s", color=C["green"], ms=3.5, label="illustrative, wet-season rate"); ax[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout(w_pad=1.5); save(fig, "Figure_6")


def fig_nist():
    W = pd.read_csv(os.path.join(RES, "nist_contrast_0.csv"), index_col=0, parse_dates=True)
    fig, ax = plt.subplots(1, 2, figsize=(16 * CM, 6 * CM), gridspec_kw=dict(width_ratios=[1.5, 1]))
    pal = [C["verm"], C["blue"], C["green"], C["orange"], C["pink"], C["sky"]]
    order = ["R2207"] + [c for c in W.columns if c != "R2207"]
    for k, c in enumerate(order):
        ax[0].plot(W.index, W[c].rolling(4, min_periods=2).median(), color=pal[k], lw=1.3 if c == "R2207" else 0.7, label="sensor with shifts" if c == "R2207" else ("five neighbours" if k == 1 else None))
    ax[0].set_ylim(-0.22, 0.34); ax[0].set_ylabel("Leave-one-out contrast (log)"); ax[0].legend(fontsize=7, loc="upper left"); ax[0].xaxis.set_major_locator(mdates.YearLocator()); ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y")); panel(ax[0], "(a) six identical roof pyranometers")
    F = pd.read_csv(os.path.join(RES, "nist_walkforward_summary.csv"))
    net = [("Roof, six identical pyranometers", "roof, identical", -0.18), ("Roof, three sensor types", "roof, three types", 0.0), ("Campus, one pyranometer per array", "campus, separated", 0.18)]
    weeks = {52: 0, 78: 1, 104: 2}
    for meth, c_, ls, dx in (("Nominal", C["verm"], "--", -0.05), ("Calibrated", C["blue"], "-", 0.05)):
        q = F[F.method == meth]
        for n_, lab, off in net:
            s = q[q.network == n_].sort_values("test_weeks")
            for w_, k_, n__ in zip(s.test_weeks, s.alarm_count - s.unresolved_count, s.folds):
                ax[1].annotate(f"{k_:.0f}/{n__:.0f}", (weeks[w_] + off + dx, k_ / n__), xytext=(0, 5 if meth == "Nominal" else -9), textcoords="offset points", ha="center", fontsize=5.5, color="0.25")
            ax[1].plot([weeks[w] + off + dx for w in s.test_weeks], (s.alarm_count - s.unresolved_count) / s.folds, ls="", marker={"roof, identical": "o", "roof, three types": "s", "campus, separated": "^"}[lab], ms=3.6, color=c_, mfc=(c_ if meth == "Calibrated" else "none"))
    for lab, mk in (("roof, identical", "o"), ("roof, three types", "s"), ("campus, separated", "^")):
        ax[1].plot([], [], mk, color="0.35", ms=3.5, label=lab)
    ax[1].plot([], [], "o", color=C["verm"], mfc="none", ms=3.5, label="nominal"); ax[1].plot([], [], "o", color=C["blue"], ms=3.5, label="bootstrap")
    ax[1].axhline(0.05, color="k", lw=0.4, ls=":"); ax[1].set_xticks([0, 1, 2]); ax[1].set_xticklabels(["52", "78", "104"]); ax[1].set_xlabel("Evaluation weeks"); ax[1].set_ylabel("Folds with a drift report"); ax[1].set_ylim(-0.13, 1.05); ax[1].legend(fontsize=6.3, ncol=2, loc="upper center", columnspacing=0.8, handletextpad=0.2); panel(ax[1], "(b) forward evaluation")
    fig.tight_layout(w_pad=1.2); save(fig, "Figure_4")


def graphical_abstract():
    """plain language, left to right, colour-blind-safe"""
    RS = json.load(open(os.path.join(RES, "reset_case_study.json")))
    W = pd.read_csv(os.path.join(RES, "nist_contrast_0.csv"), index_col=0, parse_dates=True)
    fig = plt.figure(figsize=(16 * CM, 6.6 * CM))
    a0 = fig.add_axes([0.0, 0.0, 0.27, 1.0]); a0.axis("off"); a0.set_xlim(0, 10); a0.set_ylim(0, 10)
    a0.text(5, 9.5, "1  Sensors check\none another", ha="center", va="top", fontsize=7.5, weight="bold")
    for k, (x, c_) in enumerate(zip((1.9, 5.0, 8.1), (C["blue"], C["orange"], C["green"]))):
        a0.add_patch(FancyBboxPatch((x - 1.2, 3.0), 2.4, 1.3, boxstyle="round,pad=0.1", fc=c_, lw=0))
        a0.text(x, 3.65, f"signal {k+1}", ha="center", va="center", color="w", fontsize=6.3, weight="bold")
    a0.text(5, 6.3, "same sky", ha="center", fontsize=6.8)
    for x in (1.9, 5.0, 8.1):
        a0.annotate("", (x, 4.55), (5, 6.1), arrowprops=dict(arrowstyle="-|>", lw=0.5, color="0.45"))
    a0.text(5, 2.2, "no calibrated instrument needed\nto compare them", ha="center", va="top", fontsize=6.2)
    a1 = fig.add_axes([0.35, 0.18, 0.27, 0.56])
    for c_ in W.columns:
        s = W[c_].rolling(4, min_periods=2).median()
        a1.plot(W.index, s, color=C["verm"] if c_ == "R2207" else "0.65", lw=1.2 if c_ == "R2207" else 0.5, ls="-" if c_ == "R2207" else "--")
    a1.set_ylabel("Sensor minus the others", fontsize=6.5); a1.tick_params(labelsize=6); a1.xaxis.set_major_locator(mdates.YearLocator()); a1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.text(0.48, 0.95, "2  One sensor stands out\n(public test network)", ha="center", va="top", fontsize=7.5, weight="bold")
    for x0 in (0.265, 0.635):
        fig.patches.append(FancyArrowPatch((x0, 0.86), (x0 + 0.04, 0.86), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=9, lw=0.8, color="0.3"))
    a2 = fig.add_axes([0.70, 0.0, 0.30, 1.0]); a2.axis("off")
    a2.text(0, 0.95, "3  What operators learn", fontsize=7.5, weight="bold", va="top")
    a2.text(0, 0.80, ("• Three answers: all consistent,\n  something is wrong, or this sensor\n  is wrong\n\n• Naming one sensor needs three\n  signals and one fault at a time\n\n"
                      "• Limits must be set on each system,\n  from its own earlier data\n\n• Hidden drift matters most where\n  panels soil slowly"), fontsize=6.2, va="top", linespacing=1.3)
    fig.savefig(os.path.join(FIG, "Graphical_Abstract.png"), dpi=600)
    fig.savefig(os.path.join(FIG, "Graphical_Abstract.tif"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


if __name__ == "__main__":
    for f in (fig_framework, fig_power, fig_baselines, fig_nist, fig_case, fig_bias, graphical_abstract):
        try:
            f(); print("ok", f.__name__)
        except Exception as e:
            print("FAILED", f.__name__, repr(e))
