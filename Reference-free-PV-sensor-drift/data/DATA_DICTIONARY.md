# Data dictionary

Only variables read by the analysis are included. Site coordinates are omitted.

**daily_zone_data.csv** – one row per zone and day, 3 March 2023 – 7 September 2026.
`date`; `zone` (Zone 1–3); `equivalent_hours` final yield (kWh/kWp); `pr_percent` reported PR, stored as min(PR, 90.00);
`ambient_temp_mean_c`; `flag_pr_truncated` True when PR is clipped; `include_trend_event` True for days used.

**om_log.csv** – operator's log: `code` (C cleaning, M meter calibration, T transformer maintenance), `date_gregorian`, `intervention`.

**satellite_ceres_daily.csv** – `YEAR, MO, DY`, `ALLSKY_SFC_SW_DWN`, `CLRSKY_SFC_SW_DWN` (kWh m⁻² d⁻¹), CERES SYN1deg from NASA POWER.

**nist_daily.csv** – daily sums for the NIST campus arrays (PVDAQ systems 4901, 4902, 4903; CC BY 4.0, doi:10.25984/1846021), every third day,
from `optional_fetch_nist.py`. `irradiance_*__kwh`: daily sum as stored (pyranometer channels are on another scale, which does not affect log ratios);
`*__cov`: share of non-missing minutes; `ac_power_meter_*__sum`: as stored.
