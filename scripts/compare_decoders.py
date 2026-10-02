"""Три декодера, одинаковые файлы, полное сравнение профилей и научных результатов.

Каждый замер — отдельный процесс: нативный сбой не уничтожает весь отчёт, а
локальные таблицы соседних файлов не смешиваются. Импорт/запуск процесса замерен
отдельно. Быстрый вариант с расхождениями никогда не объявляется победителем.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def code_fingerprint():
    digest = hashlib.sha256()
    for path in sorted((ROOT / "gdex_bufr").rglob("*.py")) + [Path(__file__).resolve()]:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def compare_values(left, right, location="", differences=None):
    """Числа: абсолютная погрешность 1e-8; пропуск не равен нулю."""
    if differences is None:
        differences = []
    # В CSV одна метрика содержит JSON-массив. Сравниваем числа внутри него
    # тем же допуском, а не строки с разным представлением последнего бита.
    if location.endswith("/inversion_from_top_tops") and isinstance(left, str) and isinstance(right, str):
        return compare_values(json.loads(left), json.loads(right), location + "/json", differences)
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(left.keys() | right.keys()):
            if key not in left or key not in right:
                differences.append({"field": location + "/" + key, "reason": "missing key"})
            else:
                compare_values(left[key], right[key], location + "/" + key, differences)
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            differences.append({"field": location, "lengths": [len(left), len(right)]})
        for i, (a, b) in enumerate(zip(left, right)):
            compare_values(a, b, f"{location}/{i}", differences)
    elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not (math.isclose(left, right, rel_tol=0, abs_tol=1e-8)
                or (math.isnan(left) and math.isnan(right))):
            differences.append({"field": location, "reference": left, "candidate": right})
    elif left != right:
        differences.append({"field": location, "reference": left, "candidate": right})
    return differences


def worker(args):
    from gdex_bufr.bufr_adapter import init_decoder_tables
    from gdex_bufr.decoder_backends import decode
    from gdex_bufr.profile_climate.config import load_profile_climate_config
    from gdex_bufr.profile_climate.extract import process_profile

    result = {"backend": args.backend, "file": str(Path(args.file).resolve()), "status": "error"}
    try:
        registry = init_decoder_tables({"directory": str(ROOT / "gdex_data/bufr_tables"),
                                       "master_table_version": 43, "export_on_update": False})
        config = load_profile_climate_config(ROOT / "profile_climate_config.yaml")
        stations = {s.station_id for s in config.stations_in_region(args.region)} if args.region else args.station
        start = time.perf_counter()
        profiles = decode(Path(args.file).resolve(), backend=args.backend, station_id=stations,
                          registry=registry, strict=True, collect_elements=False)
        decoded = time.perf_counter()
        normalized = []
        for profile in profiles:
            long, metrics, _, _ = process_profile(profile, registry=registry, output_mode="climate")
            normalized.append({"station_id": profile.station_id, "subset_index": profile.subset_index,
                               "datetime": profile.report_datetime_utc,
                               "latitude": profile.latitude_deg, "longitude": profile.longitude_deg,
                               "elevation": profile.station_elevation_m,
                               "levels": [asdict(level) for level in profile.levels],
                               "coded_metadata": profile.metadata.get("coded_metadata", {}),
                               "long": long, "metrics": metrics})
        finished = time.perf_counter()
        normalized.sort(key=lambda p: (p["station_id"] or "", p["datetime"] or "", p["subset_index"]))
        from gdex_bufr.decoder_backends import backend_version
        version = backend_version(args.backend)
        peak = None
        if sys.platform != "win32":
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        result.update(status="ok", version=version, profiles=len(profiles),
                      levels=sum(len(p.levels) for p in profiles), decode_seconds=decoded-start,
                      processing_seconds=finished-decoded, total_seconds=finished-start,
                      peak_rss_mib=peak, data=normalized)
    except Exception:
        result["error"] = traceback.format_exc()
    Path(args.result).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if result["status"] == "ok" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", default="1999,2010,2025")
    parser.add_argument("--station", default="31004")
    parser.add_argument("--region")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", default="gdex_outputs/decoder_comparison")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=("pybufrkit", "eccodes", "ncepbufr"))
    parser.add_argument("--file")
    parser.add_argument("--result")
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    if args.repeats < 1:
        parser.error("--repeats должен быть положительным")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Свежие имена исключают чтение результата старого запуска после native STOP.
    attempt = uuid.uuid4().hex[:12]
    code_sha256 = code_fingerprint()
    sources = {}
    runs = []
    for year in args.years.split(","):
        paths = sorted((ROOT / "gdex_data/raw" / year).glob("*.bufr"))
        if not paths:
            raise ValueError(f"Нет файлов за {year}")
        source = paths[len(paths)//2]
        sources[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        reference = None
        for repeat in range(args.repeats):
            # Меняем порядок: один движок не получает постоянно прогретый диск.
            order = ["pybufrkit", "eccodes", "ncepbufr"]
            if repeat % 2:
                order.reverse()
            for backend in order:
                result_path = output / f"{attempt}_{year}_{repeat}_{backend}.json"
                command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--backend", backend,
                           "--file", str(source), "--result", str(result_path), "--station", args.station]
                if args.region:
                    command += ["--region", args.region]
                start = time.perf_counter()
                try:
                    process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                             errors="replace", timeout=args.timeout)
                    if process.returncode == 0 and result_path.exists():
                        result = json.loads(result_path.read_text(encoding="utf-8"))
                    else:
                        result = {"backend": backend, "status": "error", "file": str(source),
                                  "exit_code": process.returncode, "error": process.stderr[-6000:]}
                        if result_path.exists():
                            result["error"] += json.loads(result_path.read_text(encoding="utf-8")).get("error", "")
                except subprocess.TimeoutExpired:
                    result = {"backend": backend, "status": "timeout", "file": str(source)}
                result.update(year=year, repeat=repeat, wall_seconds=time.perf_counter()-start)
                data = result.pop("data", None)
                if backend == "pybufrkit" and reference is None and data:
                    reference = data
                if reference is not None and data is not None:
                    differences = compare_values(reference, data)
                    result["differences"] = len(differences)
                    result["difference_examples"] = differences[:15]
                    result["matches_reference"] = not differences
                else:
                    result["matches_reference"] = False
                runs.append(result)
                print(json.dumps(result, ensure_ascii=True), flush=True)
                (output / "progress.json").write_text(json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {}
    for backend in ("pybufrkit", "eccodes", "ncepbufr"):
        subset = [r for r in runs if r["backend"] == backend]
        eligible = all(r["status"] == "ok" and r["matches_reference"] and r["profiles"] > 0 for r in subset)
        summary[backend] = {"eligible": eligible, "runs": len(subset),
                            "matched": sum(r["matches_reference"] for r in subset),
                            "median_seconds": statistics.median(r["total_seconds"] for r in subset)
                            if all("total_seconds" in r for r in subset) else None}
        per_file = {year: statistics.median(r["total_seconds"] for r in subset if r["year"] == year)
                    for year in args.years.split(",")
                    if all("total_seconds" in r for r in subset if r["year"] == year)}
        summary[backend]["per_year_median_seconds"] = per_file
        summary[backend]["sample_seconds"] = sum(per_file.values()) if eligible else None
    eligible = [name for name in summary if summary[name]["eligible"]]
    winner = min(eligible, key=lambda name: summary[name]["sample_seconds"]) if eligible else None
    unchanged = code_sha256 == code_fingerprint()
    if not unchanged:
        winner = None
    packages = {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "metpy", "pybufrkit", "eccodes")
                if importlib.util.find_spec(p) is not None}
    report = {"environment": {"python": sys.version, "platform": platform.platform(), "packages": packages},
              "code_sha256": code_sha256, "code_unchanged": unchanged,
              "comparison": "all decoded levels, metadata fields, working rows and metrics; abs_tol=1e-8",
              "sources_sha256": sources, "attempt": attempt,
              "selection": {"region": args.region, "station": args.station if not args.region else None},
              "summary": summary, "winner_on_test_sample": winner, "runs": runs}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "winner_on_test_sample": winner}, indent=2))
    return 0 if unchanged and all(s["eligible"] for s in summary.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
