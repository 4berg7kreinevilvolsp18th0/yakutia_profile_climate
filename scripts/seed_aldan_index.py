"""Быстрый индекс: seed из metrics + скан всех циклов по календарным дырам."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--old-metrics", default="gdex_outputs/результаты-алдан/profile_metrics.csv")
    p.add_argument("--raw-root", default="gdex_data/raw")
    p.add_argument("--output-dir", default="gdex_outputs/результаты-алдан-полный")
    p.add_argument("--start-date", default="1999-10-01")
    p.add_argument("--end-date", default="2026-07-19")
    p.add_argument("--workers", type=int, default=14)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    raw_root = Path(args.raw_root)

    days_with: set[str] = set()
    seed_rows: list[dict[str, str]] = []
    with Path(args.old_metrics).open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            dt = (row.get("datetime_utc") or "").replace("Z", "")
            day = dt[:10]
            if day:
                days_with.add(day)
            src = row.get("source_file") or ""
            name = Path(src).name
            ymd = ""
            cycle = ""
            for part in name.split("."):
                if part.startswith("t") and part.endswith("z") and len(part) == 4:
                    cycle = part[1:3]
                if part.isdigit() and len(part) == 8:
                    ymd = part
            year = ymd[:4] if ymd else ""
            path = raw_root / year / name
            if not path.exists() and Path(src).exists():
                path = Path(src)
            seed_rows.append({
                "source_file": name,
                "source_path": str(path.resolve()) if path.exists() else str(path),
                "cycle": cycle or str(row.get("cycle") or "").zfill(2),
                "file_ymd": ymd,
                "obs_datetime": dt[:19],
                "subset_index": str(row.get("subset_index") or ""),
                "delayed": "0",
            })

    gap_dates = []
    d = start
    while d <= end:
        if d.isoformat() not in days_with:
            gap_dates.append(d)
        d += timedelta(days=1)

    gap_files: list[Path] = []
    for gd in gap_dates:
        ymd = gd.strftime("%Y%m%d")
        year_dir = raw_root / str(gd.year)
        if not year_dir.is_dir():
            continue
        for cycle in ("00", "06", "12", "18"):
            path = year_dir / f"gdas.adpupa.t{cycle}z.{ymd}.bufr"
            if path.exists():
                gap_files.append(path)

    gap_list = out_dir / "_gap_files.txt"
    gap_list.write_text("\n".join(str(p.resolve()) for p in gap_files), encoding="utf-8")
    print(json.dumps({
        "seed_from_metrics": len(seed_rows),
        "calendar_gap_days": len(gap_dates),
        "gap_bufr_files": len(gap_files),
        "gap_list": str(gap_list),
    }, ensure_ascii=False, indent=2), flush=True)

    gap_hits: list[dict[str, str]] = []
    gap_scan_dir = out_dir / "_gap_scan"
    if gap_files:
        rc = subprocess.call([
            sys.executable,
            str(ROOT / "scripts" / "scan_aldan_index.py"),
            "--file-list", str(gap_list),
            "--workers", str(args.workers),
            "--output-dir", str(gap_scan_dir),
            "--old-metrics", args.old_metrics,
            "--start-date", args.start_date,
            "--end-date", args.end_date,
        ], cwd=str(ROOT))
        if rc != 0:
            print(f"gap scan failed rc={rc}", file=sys.stderr)
            return rc
        gap_index = gap_scan_dir / "aldan_bufr_index.csv"
        if gap_index.exists():
            with gap_index.open(encoding="utf-8", newline="") as fh:
                gap_hits = list(csv.DictReader(fh))

    columns = [
        "source_file", "source_path", "cycle", "file_ymd",
        "obs_datetime", "subset_index", "delayed",
    ]
    merged: dict[tuple, dict] = {}
    for row in seed_rows + gap_hits:
        key = (row.get("source_file"), row.get("obs_datetime"), row.get("subset_index"))
        merged[key] = row
    rows = sorted(merged.values(), key=lambda r: (r.get("obs_datetime") or "", r.get("source_file") or ""))

    index_path = out_dir / "aldan_bufr_index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # compare vs old
    old_keys = {(r.get("obs_datetime") or "")[:16] for r in seed_rows if r.get("obs_datetime")}
    index_keys = {(r.get("obs_datetime") or "")[:16] for r in rows if r.get("obs_datetime")}
    only_bufr = sorted(index_keys - old_keys)
    compare_path = out_dir / "compare_index_vs_old.csv"
    with compare_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["obs_datetime", "status", "note"])
        writer.writeheader()
        for key in only_bufr:
            writer.writerow({"obs_datetime": key, "status": "in_bufr_not_in_old_csv", "note": "from_gap_scan"})
        for row in rows:
            if row.get("delayed") == "1":
                writer.writerow({
                    "obs_datetime": row.get("obs_datetime", ""),
                    "status": "delayed_file",
                    "note": f"file={row.get('source_file')}",
                })

    summary = {
        "mode": "seed_metrics_plus_gap_scan",
        "index_rows": len(rows),
        "gap_hits": len(gap_hits),
        "gap_days": len(gap_dates),
        "in_bufr_not_old": len(only_bufr),
        "index_csv": str(index_path),
    }
    (out_dir / "aldan_index_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
