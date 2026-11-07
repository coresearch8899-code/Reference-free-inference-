# -*- coding: utf-8 -*-
"""Data panel for the metrological-soiling state-space model (R14).
Builds, per day and zone: y = ln Y_f, h = ln H_own (uncensored), censoring bound for truncated PR days,
clear-sky insolation H_cs (Ineichen-Perez, climatological Linke turbidity), regime and event indicators."""
import os
import numpy as np
import pandas as pd
import pvlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "data")
DATA = os.path.join(SRC, "daily_zone_data.csv")
OMLOG = os.path.join(SRC, "om_log.csv")
LAT, LON = 20.0, 100.0
ZONES = ["Zone 1", "Zone 2", "Zone 3"]
PR_CAP = 0.90
START = "2023-04-26"          # after commissioning (rule R3)


def clear_sky_daily(dates):
    loc = pvlib.location.Location(LAT, LON, tz="Asia/Bangkok", altitude=400)
    idx = pd.date_range(dates.min(), dates.max() + pd.Timedelta(days=1), freq="10min", tz="Asia/Bangkok",
                        inclusive="left")
    cs = loc.get_clearsky(idx, model="ineichen")["ghi"]
    h = cs.resample("D").sum() * (10 / 60) / 1000.0           # kWh m-2
    h.index = h.index.tz_localize(None)
    return h.reindex(dates).values


def load_panel():
    d = pd.read_csv(DATA, parse_dates=["date"])
    d = d[d.date >= START]
    days = pd.date_range(d.date.min(), d.date.max(), freq="D")
    n = len(days)
    P = {"date": days, "n": n}
    P["Hcs"] = clear_sky_daily(days)
    doy = days.dayofyear.values
    P["doy"] = doy
    P["t_year"] = (days - days[0]).days.values / 365.25
    m = days.month.values
    P["dry"] = np.isin(m, [12, 1, 2, 3]).astype(float)     # haze/dry season Dec-Mar, as in R13
    y = np.full((n, 3), np.nan)
    h = np.full((n, 3), np.nan)
    hc = np.full((n, 3), np.nan)                             # upper bound of h on truncated days
    tamb = np.full(n, np.nan)
    pos = {dt: k for k, dt in enumerate(days)}
    for zi, z in enumerate(ZONES):
        dz = d[d.zone == z]
        for _, r in dz.iterrows():
            k = pos[r.date]
            if not bool(r.include_trend_event):
                continue
            yf = r.equivalent_hours
            if not (yf > 0.5):
                continue
            y[k, zi] = np.log(yf)
            pr = r.pr_percent / 100.0
            if bool(r.flag_pr_truncated):
                hc[k, zi] = np.log(yf / PR_CAP)              # true PR >= 0.90  ->  H <= Yf/0.90
            elif pr > 0.3:
                h[k, zi] = np.log(yf / pr)
        t = dz.set_index("date").ambient_temp_mean_c
        for dt, v in t.items():
            if dt in pos and np.isfinite(v):
                tamb[pos[dt]] = v if np.isnan(tamb[pos[dt]]) else 0.5 * (tamb[pos[dt]] + v)
    tamb = pd.Series(tamb).interpolate(limit_direction="both").values
    P.update(y=y, h=h, hc=hc, tamb=tamb)
    log = pd.read_csv(OMLOG, parse_dates=["date_gregorian"])
    P["events"] = {r.code: pos.get(r.date_gregorian, None) for _, r in log.iterrows()}
    return P


def load_power(dates):
    """daily CERES irradiance for the site (NASA POWER), site coordinates removed"""
    d = pd.read_csv(os.path.join(SRC, "satellite_ceres_daily.csv")).replace(-999, np.nan)
    d.index = pd.to_datetime(dict(year=d.YEAR, month=d.MO, day=d.DY))
    return d.reindex(dates)


if __name__ == "__main__":
    P = load_panel()
    print(P["n"], P["date"][0].date(), P["date"][-1].date())
    print("obs y:", np.isfinite(P["y"]).sum(0), " h:", np.isfinite(P["h"]).sum(0), " censored:", np.isfinite(P["hc"]).sum(0))
    print("Hcs range", np.nanmin(P["Hcs"]), np.nanmax(P["Hcs"]))
    print(P["events"])
