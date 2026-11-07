# -*- coding: utf-8 -*-
"""Step F3: daily aggregates of the three NIST campus arrays from the public PVDAQ archive (CC BY 4.0, doi:10.25984/1846021).
Each 1-min daily file is read, reduced to daily irradiance sums and energies (daylight minutes only) and discarded."""
import io, os, sys, time, datetime as dt, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "nist_daily.csv")
BASE = "https://oedi-data-lake.s3.amazonaws.com/pvdaq/csv/pvdata/system_id={s}/year={y}/month={m}/day={d}/system_{s}__date_{y}_{m:02d}_{d:02d}.csv"
SYS = {4901: "canopy", 4902: "ground", 4903: "roof"}
UA = {"User-Agent": "pv-research-fetch/1.0"}


def one(args):
    s, day = args
    url = BASE.format(s=s, y=day.year, m=day.month, d=day.day)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                raw = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            time.sleep(2)
        except Exception:
            time.sleep(2)
    else:
        return None
    d = pd.read_csv(io.BytesIO(raw), low_memory=False)
    tc = d.columns[0]
    irr = [c for c in d.columns if c.startswith("irradiance_")]
    meters = [c for c in d.columns if c.startswith("ac_power_meter_")]
    if not irr:
        return None
    d[irr + meters] = d[irr + meters].apply(pd.to_numeric, errors="coerce")
    rec = {"system": s, "date": day.isoformat(), "n_min": len(d)}
    for c in irr:
        v = d[c].clip(lower=0)
        rec[f"{c}__kwh"] = v.sum() / 60000.0                      # W m-2 min -> kWh m-2
        rec[f"{c}__cov"] = v.notna().mean()
    for c in meters:
        rec[f"{c}__sum"] = d[c].sum() / 60.0                      # as stored, units x h
        rec[f"{c}__cov"] = d[c].notna().mean()
    return rec


if __name__ == "__main__":
    days = [dt.date(2014, 10, 1) + dt.timedelta(days=i) for i in range((dt.date(2018, 12, 31) - dt.date(2014, 10, 1)).days + 1)]
    days = days[::3]                      # every third day: weekly trend tests do not need all days, and the archive is slow
    jobs = [(s, x) for x in days for s in SYS]
    print(len(jobs), "files", flush=True)
    rows, t0 = [], time.time()
    with ThreadPoolExecutor(10) as ex:
        for i, r in enumerate(ex.map(one, jobs)):
            if r:
                rows.append(r)
            if i % 300 == 0:
                print(i, len(rows), f"{time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("saved", OUT, len(rows))
