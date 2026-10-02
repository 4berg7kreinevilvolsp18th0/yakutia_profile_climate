"""Сравнение режимов на одних BUFR с проверкой полного совпадения рабочих данных.

Пример: py -3 scripts/benchmark_decode.py --years 1999,2010,2025 --output report.json
Это измерение последовательного декодирования, не прогноз полного ночного прогона.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gdex_bufr.bufr_adapter import _make_decoder, decode_bufr_file, init_decoder_tables
from gdex_bufr.profile_climate.extract import process_profile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--years", default="1999,2010,2025")
    p.add_argument("--station", default="31004")
    p.add_argument("--region", help="Сравнить все включённые станции региона")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    station_ids = args.station
    if args.region:
        from gdex_bufr.profile_climate.config import load_profile_climate_config
        config = load_profile_climate_config(ROOT / "profile_climate_config.yaml")
        station_ids = {s.station_id for s in config.stations_in_region(args.region)}
        if not station_ids:
            raise ValueError("В регионе нет включённых станций")
    registry = init_decoder_tables({"directory": str(ROOT / "gdex_data/bufr_tables"),
                                   "wmo_version": "latest", "master_table_version": 43,
                                   "export_on_update": False})
    results = []
    reference = {}
    for year in args.years.split(","):
        files = sorted((ROOT / "gdex_data/raw" / year).glob("*.bufr"))
        if not files:
            raise ValueError(f"Нет BUFR за {year}")
        path = files[len(files) // 2]
        for mode in ("audit", "climate"):
            decoder = _make_decoder(registry)
            started = time.perf_counter()
            profiles = decode_bufr_file(path, station_id=station_ids, registry=registry,
                                       decoder=decoder, strict=True, collect_elements=mode == "audit",
                                       fast_station_filter=mode != "audit")
            decoded_at = time.perf_counter()
            outputs = [process_profile(prof, registry=registry, output_mode=mode) for prof in profiles]
            done = time.perf_counter()
            core = [(long, metric) for long, metric, _, _ in outputs]
            digest = hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()
            if mode == "audit":
                reference[str(path)] = digest
            row = {"file": str(path), "mode": mode, "profiles": len(profiles),
                   "decode_seconds": decoded_at - started, "extract_seconds": done - decoded_at,
                   "seconds": done - started, "result_pickle_bytes": len(pickle.dumps(outputs)),
                   "working_sha256": digest, "matches_audit": digest == reference[str(path)]}
            results.append(row)
            print(json.dumps(row), flush=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if all(r["matches_audit"] and r["profiles"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
