"""Индекс BUFR с станцией Aldan (31004) по всем циклам + сверка со старым metrics."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

STATION_ID = "31004"
INDEX_COLUMNS = [
    "source_file",
    "source_path",
    "cycle",
    "file_ymd",
    "obs_datetime",
    "subset_index",
    "delayed",
]

_WORKER: dict[str, Any] = {}


def _ensure_scan_worker(tables_dir: str, export_dir: str, project_root: str) -> None:
    if _WORKER.get("ready"):
        return
    root = str(Path(project_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)

    from gdex_bufr.meteo_parser_bridge import ensure_meteo_parser_import

    ensure_meteo_parser_import((Path(project_root) / "../meteo_parser").resolve())
    while root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    for key in list(sys.modules):
        if key == "gdex_bufr" or key.startswith("gdex_bufr."):
            del sys.modules[key]

    from gdex_bufr.bufr_adapter import _make_decoder, init_decoder_tables

    registry = init_decoder_tables({
        "directory": tables_dir,
        "wmo_version": "latest",
        "master_table_version": 43,
        "export_dir": export_dir,
        "export_on_update": False,
    })
    _WORKER.update({
        "registry": registry,
        "decoder": _make_decoder(registry),
        "ready": True,
    })


def _scan_one(payload: tuple[str, str, str, str, str]) -> list[dict[str, str]]:
    bufr_path, station_id, tables_dir, export_dir, project_root = payload
    root = str(Path(project_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    _ensure_scan_worker(tables_dir, export_dir, project_root)

    from gdex_bufr.bufr_adapter import (
        DESC_DAY,
        DESC_HOUR,
        DESC_MONTH,
        DESC_WMO_BLOCK,
        DESC_WMO_STATION,
        DESC_YEAR,
        _is_observation_message,
        _iter_messages,
        _query_values,
        _subset_station_id,
        _subset_value,
    )
    from pybufrkit.mdquery import MetadataExprParser, MetadataQuerent

    path = Path(bufr_path)
    name = path.name
    parts = name.split(".")
    cycle = ""
    file_ymd = ""
    for part in parts:
        if part.startswith("t") and part.endswith("z") and len(part) == 4:
            cycle = part[1:3]
        if part.isdigit() and len(part) == 8:
            file_ymd = part

    hits: list[dict[str, str]] = []
    try:
        raw = path.read_bytes()
        for message in _iter_messages(_WORKER["decoder"], raw):
            if not _is_observation_message(message):
                continue
            n_subsets = int(MetadataQuerent(MetadataExprParser()).query(message, "%n_subsets") or 0)
            block_map = _query_values(message, DESC_WMO_BLOCK)
            station_map = _query_values(message, DESC_WMO_STATION)
            year_map = _query_values(message, DESC_YEAR)
            month_map = _query_values(message, DESC_MONTH)
            day_map = _query_values(message, DESC_DAY)
            hour_map = _query_values(message, DESC_HOUR)
            for idx in range(n_subsets):
                sid = _subset_station_id(
                    {DESC_WMO_BLOCK: block_map, DESC_WMO_STATION: station_map},
                    idx,
                )
                if sid != station_id:
                    continue
                y = _subset_value(year_map, idx)
                m = _subset_value(month_map, idx)
                d = _subset_value(day_map, idx)
                h = _subset_value(hour_map, idx)
                if None in (y, m, d, h):
                    obs_dt = ""
                    delayed = ""
                else:
                    obs_dt = f"{int(y):04d}-{int(m):02d}-{int(d):02d}T{int(h):02d}:00:00"
                    obs_ymd = f"{int(y):04d}{int(m):02d}{int(d):02d}"
                    delayed = "1" if file_ymd and obs_ymd != file_ymd else "0"
                hits.append({
                    "source_file": name,
                    "source_path": str(path),
                    "cycle": cycle,
                    "file_ymd": file_ymd,
                    "obs_datetime": obs_dt,
                    "subset_index": str(idx),
                    "delayed": delayed,
                })
    except Exception as exc:  # noqa: BLE001
        return [{"source_file": name, "source_path": str(path), "cycle": cycle,
                 "file_ymd": file_ymd, "obs_datetime": "", "subset_index": "",
                 "delayed": f"ERROR:{exc!r}"}]
    return hits


def _list_bufr(raw_root: Path, start: date, end: date, cycles: list[str]) -> list[Path]:
    files: list[Path] = []
    cycle_set = {c.zfill(2)[-2:] for c in cycles}
    for year in range(start.year, end.year + 1):
        year_dir = raw_root / str(year)
        if not year_dir.is_dir():
            continue
        for cycle in sorted(cycle_set):
            for path in year_dir.glob(f"gdas.adpupa.t{cycle}z.*.bufr"):
                ymd = path.name.split(".")[-2]
                if len(ymd) != 8 or not ymd.isdigit():
                    continue
                obs = date(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:8]))
                if obs < start or obs > end:
                    continue
                files.append(path)
    return sorted(files)


def _list_from_file_list(path: Path) -> list[Path]:
    files: list[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = Path(line)
        if p.exists():
            files.append(p)
    return files


def _write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _compare_with_old(index_rows: list[dict[str, str]], old_metrics: Path, out_path: Path) -> dict[str, int]:
    old_keys: set[str] = set()
    if old_metrics.exists():
        with old_metrics.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                dt = (row.get("datetime_utc") or "").replace("Z", "")
                if len(dt) >= 16:
                    old_keys.add(dt[:16])
                elif len(dt) >= 13:
                    old_keys.add(dt[:13] + ":00")

    index_keys: set[str] = set()
    clean_rows = [r for r in index_rows if not str(r.get("delayed", "")).startswith("ERROR")]
    for row in clean_rows:
        dt = row.get("obs_datetime") or ""
        if len(dt) >= 16:
            index_keys.add(dt[:16])
        elif len(dt) >= 13:
            index_keys.add(dt[:13] + ":00")

    only_bufr = sorted(index_keys - old_keys)
    only_old = sorted(old_keys - index_keys)
    both = sorted(index_keys & old_keys)
    delayed = [r for r in clean_rows if r.get("delayed") == "1"]

    rows = (
        [{"obs_datetime": k, "status": "in_bufr_not_in_old_csv", "note": ""} for k in only_bufr]
        + [{"obs_datetime": k, "status": "in_old_csv_not_in_bufr_index", "note": ""} for k in only_old]
        + [{
            "obs_datetime": r.get("obs_datetime", ""),
            "status": "delayed_file",
            "note": f"file={r.get('source_file')}",
        } for r in delayed]
    )
    _write_csv(out_path, rows, ["obs_datetime", "status", "note"])
    return {
        "index_hits": len(clean_rows),
        "unique_obs_in_index": len(index_keys),
        "unique_obs_in_old": len(old_keys),
        "in_bufr_not_old": len(only_bufr),
        "in_old_not_bufr": len(only_old),
        "both": len(both),
        "delayed_rows": len(delayed),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Индекс BUFR с Aldan 31004")
    p.add_argument("--config", default="gdex_config.yaml")
    p.add_argument("--station-id", default=STATION_ID)
    p.add_argument("--start-date", default="1999-10-01")
    p.add_argument("--end-date", default="2026-07-30")
    p.add_argument("--cycles", default="00,06,12,18")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--output-dir", default="gdex_outputs/результаты-алдан-полный")
    p.add_argument("--old-metrics", default="gdex_outputs/результаты-алдан/profile_metrics.csv")
    p.add_argument("--limit-files", type=int)
    p.add_argument("--file-list", help="Текстовый список путей BUFR (по одному на строку)")
    return p


def _resolve_paths(cfg, args: argparse.Namespace) -> tuple[date, date, list[str], Path, Path, Path, Path]:
    start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    cycles = [c.strip() for c in args.cycles.split(",") if c.strip()]
    raw_root = Path(cfg.data_dir)
    if not raw_root.is_absolute():
        raw_root = (ROOT / raw_root).resolve()
    tables = Path(cfg.bufr_tables.directory)
    if not tables.is_absolute():
        tables = (ROOT / tables).resolve()
    export = Path(cfg.bufr_tables.export_dir)
    if not export.is_absolute():
        export = (ROOT / export).resolve()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return start, end, cycles, raw_root, tables, export, out_dir


def _collect_files(args: argparse.Namespace, raw_root: Path, start: date, end: date, cycles: list[str]) -> list[Path]:
    if args.file_list:
        files = _list_from_file_list(Path(args.file_list))
    else:
        files = _list_bufr(raw_root, start, end, cycles)
    if args.limit_files:
        files = files[: args.limit_files]
    return files


def _scan_files(
    files: list[Path],
    *,
    station_id: str,
    tables: Path,
    export: Path,
    workers: int,
) -> list[dict[str, str]]:
    all_hits: list[dict[str, str]] = []
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _scan_one,
                (str(path), station_id, str(tables), str(export), str(ROOT)),
            ): path
            for path in files
        }
        for fut in as_completed(futures):
            path = futures[fut]
            try:
                hits = fut.result()
            except Exception as exc:  # noqa: BLE001
                logger.warning("worker %s: %s", path.name, exc)
                hits = []
            all_hits.extend(hits)
            done += 1
            if done % 200 == 0 or done == len(files):
                logger.info("[%s/%s] hits_so_far=%s last=%s", done, len(files), len(all_hits), path.name)
    return all_hits


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from gdex_bufr.config import load_config
    from gdex_bufr.meteo_parser_bridge import ensure_meteo_parser_import

    cfg = load_config(args.config)
    ensure_meteo_parser_import(cfg.meteo_parser_path)
    sys.path.insert(0, str(ROOT))

    # 1) пути и список BUFR
    start, end, cycles, raw_root, tables, export, out_dir = _resolve_paths(cfg, args)
    files = _collect_files(args, raw_root, start, end, cycles)
    logger.info("Скан Aldan %s: файлов=%s workers=%s", args.station_id, len(files), args.workers)

    # 2) параллельный скан
    all_hits = _scan_files(
        files,
        station_id=args.station_id,
        tables=tables,
        export=export,
        workers=args.workers,
    )

    # 3) индекс + сверка со старым metrics
    index_path = out_dir / "aldan_bufr_index.csv"
    clean = [r for r in all_hits if not str(r.get("delayed", "")).startswith("ERROR")]
    clean.sort(key=lambda r: (r.get("obs_datetime") or "", r.get("source_file") or ""))
    _write_csv(index_path, clean, INDEX_COLUMNS)

    compare_path = out_dir / "compare_index_vs_old.csv"
    stats = _compare_with_old(clean, Path(args.old_metrics), compare_path)
    summary = {
        "station_id": args.station_id,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "cycles": cycles,
        "files_scanned": len(files),
        "index_csv": str(index_path),
        "compare_csv": str(compare_path),
        **stats,
    }
    (out_dir / "aldan_index_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
