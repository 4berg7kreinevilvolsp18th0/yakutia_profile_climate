"""Full-range gap check: metrics days vs BUFR 00/12 on disk."""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "gdex_outputs/результаты-алдан/profile_metrics.csv"
RAW = ROOT / "gdex_data/raw"
PAT = re.compile(r"gdas\.adpupa\.t(00|12)z\.(\d{8})\.bufr$")


def main() -> None:
    metrics = pd.read_csv(METRICS)
    time_col = "datetime_utc" if "datetime_utc" in metrics.columns else "obs_time"
    metrics[time_col] = pd.to_datetime(metrics[time_col], errors="coerce")
    metrics = metrics.dropna(subset=[time_col])
    print("metrics profiles", len(metrics))
    print("date range", metrics[time_col].min(), "->", metrics[time_col].max())

    days = set(metrics[time_col].dt.normalize().dt.date)
    start = metrics[time_col].min().normalize().date()
    end = metrics[time_col].max().normalize().date()
    all_days = pd.date_range(start, end, freq="D").date
    missing_cal = [d for d in all_days if d not in days]
    print("calendar days in span", len(all_days))
    print("days with >=1 profile", len(days))
    print("calendar gaps (no profile any cycle)", len(missing_cal))

    srcs = {Path(s).name for s in metrics.source_file.astype(str)}
    bufr: list[tuple[str, str, str]] = []
    for p in RAW.rglob("gdas.adpupa.t*z.*.bufr"):
        m = PAT.match(p.name)
        if m:
            bufr.append((p.name, m.group(1), m.group(2)))
    print("bufr 00/12 on disk", len(bufr))
    pending = [b for b in bufr if b[0] not in srcs]
    print("pending (not in metrics source)", len(pending))

    y0 = start.strftime("%Y%m%d")
    y1 = end.strftime("%Y%m%d")
    pending_in_span = [b for b in pending if y0 <= b[2] <= y1]
    print("pending inside metrics span", len(pending_in_span))

    by_month: dict[str, dict] = defaultdict(lambda: {"00": 0, "12": 0, "dates": set()})
    for name, cyc, ymd in pending_in_span:
        ym = ymd[:6]
        by_month[ym][cyc] += 1
        by_month[ym]["dates"].add(ymd)

    print("\n=== PENDING by month (BUFR 00/12 not in metrics) ===")
    for ym in sorted(by_month):
        d = by_month[ym]
        print(
            f"{ym}: pending_files={d['00'] + d['12']} "
            f"(00={d['00']}, 12={d['12']}), unique_dates={len(d['dates'])}"
        )

    print("\n=== METRICS coverage by month (sparse only) ===")
    g = metrics.copy()
    g["ym"] = g[time_col].dt.strftime("%Y-%m")
    g["day"] = g[time_col].dt.day
    sparse = []
    for ym, sub in g.groupby("ym"):
        ndays = int(pd.Period(ym).days_in_month)
        present = sorted(int(x) for x in sub.day.unique())
        miss = [d for d in range(1, ndays + 1) if d not in present]
        if miss:
            sparse.append((ym, len(sub), len(present), ndays, len(miss), miss))

    print("months with missing calendar days", len(sparse), "/", g["ym"].nunique())
    for r in sparse:
        miss_s = ",".join(map(str, r[5][:25])) + ("..." if len(r[5]) > 25 else "")
        print(f"{r[0]}: profiles={r[1]} days={r[2]}/{r[3]} miss={r[4]} [{miss_s}]")

    print("\n=== longest calendar gap runs (no profile at all) ===")
    miss_set = set(missing_cal)
    runs: list[list] = []
    run: list = []
    for d in all_days:
        if d in miss_set:
            run.append(d)
        else:
            if run:
                runs.append(run)
                run = []
    if run:
        runs.append(run)
    runs.sort(key=len, reverse=True)
    for run in runs[:30]:
        print(f"{run[0]} .. {run[-1]}  ({len(run)} days)")
    print("total gap runs", len(runs))

    ymd_pending_12 = {ymd for name, cyc, ymd in pending_in_span if cyc == "12"}
    ymd_pending_00 = {ymd for name, cyc, ymd in pending_in_span if cyc == "00"}
    bufr_ymd_12 = {ymd for name, cyc, ymd in bufr if cyc == "12"}
    bufr_ymd_00 = {ymd for name, cyc, ymd in bufr if cyc == "00"}
    recoverable = []
    no_bufr = []
    for d in missing_cal:
        ymd = d.strftime("%Y%m%d")
        has12 = ymd in ymd_pending_12
        has00 = ymd in ymd_pending_00
        exists12 = ymd in bufr_ymd_12
        exists00 = ymd in bufr_ymd_00
        if has12 or has00:
            recoverable.append((d, has00, has12))
        elif not exists12 and not exists00:
            no_bufr.append(d)

    print("\n=== calendar-missing days classification ===")
    print("missing days with pending BUFR (likely incomplete decode)", len(recoverable))
    print("missing days with no 00/12 BUFR on disk", len(no_bufr))
    print("note: files without Aldan usually stay pending forever")

    print("\n=== pending files by year ===")
    by_year: dict[str, int] = defaultdict(int)
    for name, cyc, ymd in pending_in_span:
        by_year[ymd[:4]] += 1
    for y in sorted(by_year):
        print(f"{y}: {by_year[y]}")

    # Focus: missing calendar days that have pending t12 (high chance of real data loss)
    t12_recoverable_days = sorted({d for d, has00, has12 in recoverable if has12})
    print("\n=== missing days WITH pending t12 (likely real gaps) ===")
    print("count", len(t12_recoverable_days))
    # group by month
    by_ym: dict[str, list] = defaultdict(list)
    for d in t12_recoverable_days:
        by_ym[d.strftime("%Y-%m")].append(d.day)
    for ym in sorted(by_ym):
        days_list = by_ym[ym]
        print(f"{ym}: {len(days_list)} days -> {days_list}")


if __name__ == "__main__":
    main()
