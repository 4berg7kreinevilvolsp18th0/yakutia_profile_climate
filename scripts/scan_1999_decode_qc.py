"""Быстрый QC-скан локальных BUFR за 1999 (Алдан)."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _init(tables_dir: str, export_dir: str) -> None:
    global REG
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from gdex_bufr.bufr_adapter import init_decoder_tables

    REG = init_decoder_tables({
        "directory": tables_dir,
        "wmo_version": "latest",
        "master_table_version": 43,
        "export_dir": export_dir,
        "export_on_update": False,
    })


def _check(path_str: str) -> dict:
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from gdex_bufr.bufr_adapter import decode_bufr_file
    from gdex_bufr.pybufrkit_check import profile_decode_qc

    path = Path(path_str)
    try:
        profiles = decode_bufr_file(
            path,
            station_id="31004",
            max_profiles=1,
            registry=REG,
        )
    except Exception as exc:  # noqa: BLE001
        return {"file": path.name, "status": "decode_error", "error": str(exc)}

    if not profiles:
        return {"file": path.name, "status": "no_aldan"}

    profile = profiles[0]
    qc = profile_decode_qc(profile)
    thermo = [
        (lv.pressure_hpa, lv.air_temperature_c)
        for lv in profile.levels
        if lv.pressure_hpa is not None and lv.air_temperature_c is not None
    ]
    cold = [(p, t) for p, t in thermo if p >= 950 and t < -30]
    return {
        "file": path.name,
        "status": "ok" if qc["ok"] and not cold else "qc_fail",
        "qc_ok": qc["ok"],
        "jumps": qc["suspicious_tropospheric_jumps"],
        "cold_near_1000": cold[:5],
        "surface": thermo[0] if thermo else None,
        "n_levels": len(profile.levels),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=1999)
    parser.add_argument("--months", default="10,11,12")
    parser.add_argument("--cycles", default="00,12")
    parser.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    args = parser.parse_args()

    months = {int(x) for x in args.months.split(",") if x.strip()}
    cycles = {c.strip().zfill(2) for c in args.cycles.split(",") if c.strip()}
    year_dir = ROOT / "gdex_data" / "raw" / str(args.year)
    files: list[Path] = []
    for path in sorted(year_dir.glob("gdas.adpupa.t*z.*.bufr")):
        name = path.name
        # gdas.adpupa.t12z.19991004.bufr
        parts = name.split(".")
        if len(parts) < 4:
            continue
        cycle = parts[2][1:3]
        date_token = parts[3]
        if cycle not in cycles:
            continue
        if not date_token.startswith(str(args.year)):
            continue
        month = int(date_token[4:6])
        if month not in months:
            continue
        files.append(path)

    tables_dir = str((ROOT / "gdex_data" / "bufr_tables").resolve())
    export_dir = str((ROOT / "gdex_data" / "bufr_tables_export").resolve())

    print(json.dumps({"files": len(files), "workers": args.workers}, ensure_ascii=False), flush=True)
    stats: Counter[str] = Counter()
    bad: list[dict] = []

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init,
        initargs=(tables_dir, export_dir),
    ) as pool:
        futures = {pool.submit(_check, str(path)): path for path in files}
        done = 0
        for future in as_completed(futures):
            result = future.result()
            done += 1
            stats[result["status"]] += 1
            if result["status"] != "ok" and result["status"] != "no_aldan":
                bad.append(result)
            elif result.get("cold_near_1000"):
                stats["cold_near_1000"] += 1
                bad.append(result)
            if done % 20 == 0 or done == len(files):
                print(f"progress {done}/{len(files)} {dict(stats)}", flush=True)

    print(json.dumps({"summary": dict(stats), "bad_count": len(bad), "bad_sample": bad[:10]}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
