import pandas as pd
from pathlib import Path
from datetime import date, timedelta

m = pd.read_csv("gdex_outputs/результаты-алдан/profile_metrics.csv")
print("metrics", len(m))
print("date range", m["datetime_utc"].min(), "->", m["datetime_utc"].max())

dec = m[(m["year"] == 1999) & (m["month"] == 12)].copy()
dec["day"] = pd.to_datetime(dec["datetime_utc"]).dt.day
print("1999-12 profiles", len(dec))
print("days present", sorted(dec["day"].unique().tolist()))
print(dec.groupby("day").size())

raw = Path("gdex_data/raw/1999")
files = sorted(raw.glob("gdas.adpupa.t*z.199912*.bufr"))
print("bufr on disk", len(files))
srcs = {Path(s).name for s in m["source_file"].astype(str)}
miss = [f.name for f in files if f.name not in srcs]
print("missing from metrics", len(miss))
print("missing names:", miss)
print("missing yyyymmdd:", sorted({n.split(".")[-2] for n in miss}))

# all 1999-10..12 coverage summary
for month in (10, 11, 12):
    sub = m[(m["year"] == 1999) & (m["month"] == month)].copy()
    days = sorted(pd.to_datetime(sub["datetime_utc"]).dt.day.unique().tolist()) if len(sub) else []
    print(f"1999-{month:02d}: profiles={len(sub)} days={days}")
