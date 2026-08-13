"""Индекс BUFR: какие файлы содержат станции каталога (region / список WMO)."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

INDEX_COLUMNS = [
    "station_id",
    "station_slug",
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


def _parse_file_meta(name: str) -> tuple[str, str]:
    cycle = ""
    file_ymd = ""
    for part in name.split("."):
        if part.startswith("t") and part.endswith("z") and len(part) == 4:
            cycle = part[1:3]
        if part.isdigit() and len(part) == 8:
            file_ymd = part
    return cycle, file_ymd


def _scan_one(payload: tuple[str, str, str, str, str]) -> list[dict[str, str]]:
    bufr_path, station_ids_csv, tables_dir, export_dir, project_root = payload
    wanted = {part.strip().zfill(5)[-5:] for part in station_ids_csv.split(",") if part.strip()}
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
    cycle, file_ymd = _parse_file_meta(name)
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
                if not sid or sid not in wanted:
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
                    "station_id": sid,
                    "source_file": name,
                    "source_path": str(path),
                    "cycle": cycle,
                    "file_ymd": file_ymd,
                    "obs_datetime": obs_dt,
                    "subset_index": str(idx),
                    "delayed": delayed,
                })
    except Exception as exc:  # noqa: BLE001
        return [{
            "station_id": "",
            "source_file": name,
            "source_path": str(path),
            "cycle": cycle,
            "file_ymd": file_ymd,
            "obs_datetime": "",
            "subset_index": "",
            "delayed": f"ERROR:{exc!r}",
        }]
    return hits


def list_bufr(raw_root: Path, start: date, end: date, cycles: Iterable[str]) -> list[Path]:
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


def write_index_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def summarize_hits(
    rows: list[dict[str, str]],
    *,
    station_ids: list[str],
    slug_by_id: dict[str, str],
) -> dict[str, Any]:
    clean = [r for r in rows if not str(r.get("delayed", "")).startswith("ERROR")]
    by_station: dict[str, int] = defaultdict(int)
    files_by_station: dict[str, set[str]] = defaultdict(set)
    for row in clean:
        sid = row.get("station_id") or ""
        by_station[sid] += 1
        if row.get("source_file"):
            files_by_station[sid].add(row["source_file"])
    return {
        "hits_total": len(clean),
        "errors": sum(1 for r in rows if str(r.get("delayed", "")).startswith("ERROR")),
        "per_station": {
            sid: {
                "slug": slug_by_id.get(sid, ""),
                "hits": by_station.get(sid, 0),
                "files": len(files_by_station.get(sid, ())),
            }
            for sid in station_ids
        },
    }


def scan_files(
    files: list[Path],
    *,
    station_ids: list[str],
    tables_dir: Path,
    export_dir: Path,
    project_root: Path,
    workers: int,
) -> list[dict[str, str]]:
    ids_csv = ",".join(station_ids)
    all_hits: list[dict[str, str]] = []
    done = 0
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(
                _scan_one,
                (str(path), ids_csv, str(tables_dir), str(export_dir), str(project_root)),
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


def cmd_station_index(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Индекс BUFR → станции региона")
    parser.add_argument("--config", default="gdex_config.yaml")
    parser.add_argument("--catalog", default="stations_catalog.yaml")
    parser.add_argument("--region", default="far_east")
    parser.add_argument("--start-date", default="1999-10-01")
    parser.add_argument("--end-date", default="2026-07-30")
    parser.add_argument("--cycles", default="00,06,12,18")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-files", type=int)
    parser.add_argument("--input-dir", default="", help="Корень BUFR (иначе data_dir из config)")
    parser.add_argument(
        "--output-dir",
        default="",
        help="По умолчанию gdex_outputs/far_east/regions/<region>",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from gdex_bufr.config import load_config
    from gdex_bufr.meteo_parser_bridge import ensure_meteo_parser_import
    from gdex_bufr.profile_climate.config import load_stations_catalog
    from gdex_bufr.profile_climate.paths import region_dir

    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(args.config)
    ensure_meteo_parser_import(cfg.meteo_parser_path)

    catalog = load_stations_catalog(args.catalog)
    stations = catalog.stations_in_region(args.region)
    if not stations:
        logger.error("В каталоге нет станций region=%s", args.region)
        return 2
    station_ids = [s.station_id for s in stations]
    slug_by_id = {s.station_id: s.slug for s in stations}

    start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    cycles = [c.strip() for c in args.cycles.split(",") if c.strip()]

    raw_root = Path(args.input_dir) if args.input_dir else Path(cfg.data_dir)
    if not raw_root.is_absolute():
        raw_root = (project_root / raw_root).resolve()
    tables = Path(cfg.bufr_tables.directory)
    if not tables.is_absolute():
        tables = (project_root / tables).resolve()
    export = Path(cfg.bufr_tables.export_dir)
    if not export.is_absolute():
        export = (project_root / export).resolve()

    out_dir = Path(args.output_dir) if args.output_dir else region_dir(args.region)
    if not out_dir.is_absolute():
        out_dir = (project_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list_bufr(raw_root, start, end, cycles)
    if args.limit_files:
        files = files[: args.limit_files]
    logger.info(
        "station-index region=%s stations=%s files=%s workers=%s out=%s",
        args.region, ",".join(station_ids), len(files), args.workers, out_dir,
    )

    hits = scan_files(
        files,
        station_ids=station_ids,
        tables_dir=tables,
        export_dir=export,
        project_root=project_root,
        workers=args.workers,
    )
    for row in hits:
        row["station_slug"] = slug_by_id.get(row.get("station_id") or "", "")
    clean = [r for r in hits if not str(r.get("delayed", "")).startswith("ERROR")]
    clean.sort(key=lambda r: (r.get("station_id") or "", r.get("obs_datetime") or "", r.get("source_file") or ""))
    index_path = write_index_csv(out_dir / "station_index.csv", clean)
    stats = summarize_hits(hits, station_ids=station_ids, slug_by_id=slug_by_id)
    summary = {
        "region": args.region,
        "station_ids": station_ids,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "cycles": cycles,
        "files_scanned": len(files),
        "index_csv": str(index_path),
        **stats,
    }
    (out_dir / "station_index_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0
