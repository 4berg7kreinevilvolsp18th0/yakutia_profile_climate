"""Сравнение нового metrics CSV со старым + gap_report по индексу BUFR."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path


def _norm_dt(value: str) -> str:
    dt = (value or "").replace("Z", "")
    if len(dt) >= 16:
        return dt[:16]
    if len(dt) >= 13:
        return dt[:13] + ":00"
    return dt


def _load_metrics_keys(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            key = _norm_dt(row.get("datetime_utc") or "")
            if not key:
                continue
            # keep best by n_levels_to_500
            prev = out.get(key)
            n = int(float(row.get("n_levels_to_500") or 0))
            if prev is None or n >= int(float(prev.get("n_levels_to_500") or 0)):
                out[key] = row
    return out


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--old-metrics", default="gdex_outputs/результаты-алдан/profile_metrics.csv")
    p.add_argument("--new-metrics", default="gdex_outputs/результаты-алдан-полный/profile_metrics.csv")
    p.add_argument("--index", default="gdex_outputs/результаты-алдан-полный/aldan_bufr_index.csv")
    p.add_argument("--output-dir", default="gdex_outputs/результаты-алдан-полный")
    p.add_argument("--station-id", default="31004")
    p.add_argument("--start-date", default="1999-10-01")
    p.add_argument("--end-date", default="2026-07-19")
    args = p.parse_args(argv)

    out_dir = Path(args.output_dir)
    old = _load_metrics_keys(Path(args.old_metrics))
    new = _load_metrics_keys(Path(args.new_metrics))

    # filter station if present
    def _filter_station(d: dict[str, dict]) -> dict[str, dict]:
        return {
            k: v for k, v in d.items()
            if not args.station_id or str(v.get("station_id") or args.station_id) == args.station_id
        }

    old = _filter_station(old)
    new = _filter_station(new)

    only_new = sorted(set(new) - set(old))
    only_old = sorted(set(old) - set(new))
    both = sorted(set(new) & set(old))

    compare_rows = (
        [{"obs_datetime": k, "status": "only_new", "source_file": new[k].get("source_file", "")} for k in only_new]
        + [{"obs_datetime": k, "status": "only_old", "source_file": old[k].get("source_file", "")} for k in only_old]
        + [{"obs_datetime": k, "status": "both", "source_file": new[k].get("source_file", "")} for k in both]
    )
    _write_csv(
        out_dir / "compare_new_vs_old_metrics.csv",
        compare_rows,
        ["obs_datetime", "status", "source_file"],
    )

    # index obs
    index_keys: set[str] = set()
    index_by_day: dict[str, set[str]] = defaultdict(set)
    index_path = Path(args.index)
    if index_path.exists():
        with index_path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                key = _norm_dt(row.get("obs_datetime") or "")
                if not key:
                    continue
                index_keys.add(key)
                index_by_day[key[:10]].add(key)

    start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    new_days = {k[:10] for k in new}
    gap_rows = []
    d = start
    while d <= end:
        day = d.isoformat()
        in_index = day in index_by_day
        in_new = day in new_days
        if in_new:
            klass = "present_in_new"
        elif in_index:
            klass = "in_bufr_missing_in_new_csv"
        else:
            klass = "absent_in_archive"
        gap_rows.append({
            "date": day,
            "class": klass,
            "index_obs_count": len(index_by_day.get(day, ())),
            "new_obs_count": sum(1 for k in new if k.startswith(day)),
            "old_obs_count": sum(1 for k in old if k.startswith(day)),
        })
        d += timedelta(days=1)

    _write_csv(
        out_dir / "gap_report.csv",
        gap_rows,
        ["date", "class", "index_obs_count", "new_obs_count", "old_obs_count"],
    )

    # sort/dedupe new metrics written as cleaned copy
    if Path(args.new_metrics).exists():
        rows = []
        with Path(args.new_metrics).open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames or []
            for row in reader:
                if args.station_id and str(row.get("station_id") or "") not in ("", args.station_id):
                    continue
                rows.append(row)
        rows.sort(key=lambda r: (r.get("datetime_utc") or "", r.get("profile_id") or ""))
        # dedupe profile_id keeping first after sort
        seen: set[str] = set()
        deduped = []
        for row in rows:
            pid = row.get("profile_id") or ""
            if pid in seen:
                continue
            seen.add(pid)
            deduped.append(row)
        out_metrics = out_dir / "profile_metrics_sorted.csv"
        _write_csv(out_metrics, deduped, list(fieldnames))
    else:
        out_metrics = None

    summary = {
        "old_unique_obs": len(old),
        "new_unique_obs": len(new),
        "only_new": len(only_new),
        "only_old": len(only_old),
        "both": len(both),
        "index_unique_obs": len(index_keys),
        "gap_absent_in_archive": sum(1 for r in gap_rows if r["class"] == "absent_in_archive"),
        "gap_in_bufr_missing_csv": sum(1 for r in gap_rows if r["class"] == "in_bufr_missing_in_new_csv"),
        "gap_present": sum(1 for r in gap_rows if r["class"] == "present_in_new"),
        "sorted_metrics": str(out_metrics) if out_metrics else "",
    }
    (out_dir / "compare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
