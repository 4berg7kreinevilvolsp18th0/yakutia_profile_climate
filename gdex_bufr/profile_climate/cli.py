"""CLI-команды profile_climate."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

from gdex_bufr.batch_render import FILENAME_RE, list_bufr_files
from gdex_bufr.bufr_adapter import decode_bufr_file
from gdex_bufr.bufr_tables import get_registry
from gdex_bufr.config import AppConfig
from gdex_bufr.profile_climate.config import ProfileClimateConfig, load_profile_climate_config
from gdex_bufr.profile_climate.export import export_all
from gdex_bufr.profile_climate.extract import normalize_station_id, process_profile
from gdex_bufr.profile_climate.plots import render_all_monthly_plots

logger = logging.getLogger(__name__)


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def _parse_cycles(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    return [part.strip().zfill(2)[-2:] for part in value.split(",") if part.strip()]


def _resolve_station(pc_cfg: ProfileClimateConfig, station_arg: str | None) -> tuple[str, str, str]:
    if not station_arg:
        if len(pc_cfg.stations) == 1:
            station = pc_cfg.stations[0]
            return station.station_id, station.slug, station.name
        raise SystemExit("Укажите --station или задайте одну станцию в profile_climate_config.yaml")

    if station_arg.isdigit():
        station = pc_cfg.station_by_id(station_arg)
        if station:
            return station.station_id, station.slug, station.name
        return normalize_station_id(station_arg), station_arg, station_arg

    station = pc_cfg.station_by_slug(station_arg)
    if station:
        return station.station_id, station.slug, station.name
    return station_arg, station_arg, station_arg


def _load_csv_rows(path: Path) -> list[dict]:
    import csv

    if not path.exists():
        return []
    rows: list[dict] = []
    float_fields = {
        "pressure_hpa", "temperature_c", "height_m",
        "n_levels_total", "n_levels_to_500", "p_surface_hpa", "t_surface_c",
        "p_top_hpa", "t_top_c", "delta_t_top_surface_c",
        "inversion_top_pressure_hpa", "inversion_top_height_m",
        "inversion_top_temp_c", "inversion_delta_t_c",
    }
    int_fields = {"year", "month", "level_index"}
    bool_fields = {"inversion_detected"}

    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed = dict(row)
            for key in float_fields:
                if parsed.get(key) not in (None, ""):
                    parsed[key] = float(parsed[key])
            for key in int_fields:
                if parsed.get(key) not in (None, ""):
                    parsed[key] = int(parsed[key])
            for key in bool_fields:
                value = parsed.get(key)
                if isinstance(value, str):
                    parsed[key] = value.lower() in {"true", "1", "yes"}
            rows.append(parsed)
    return rows


def _filter_by_cycles(rows: list[dict], cycles: list[str]) -> list[dict]:
    cycle_set = set(cycles)
    return [row for row in rows if str(row.get("cycle", "")).zfill(2)[-2:] in cycle_set]


def cmd_station_profiles(
    app_cfg: AppConfig,
    pc_cfg: ProfileClimateConfig,
    args: argparse.Namespace,
) -> int:
    station_id, station_slug, station_name = _resolve_station(pc_cfg, args.station)
    start = _parse_date(args.start_date) or pc_cfg.start_date
    end = _parse_date(args.end_date) or pc_cfg.end_date
    pressure_top = float(args.pressure_top or pc_cfg.pressure_top_hpa)
    cycles = _parse_cycles(args.cycles, pc_cfg.cycles)
    output_dir = Path(args.output or "gdex_outputs/profile_climate")

    files = list_bufr_files(
        app_cfg,
        start_date=start,
        end_date=end,
        limit=args.limit_files,
        only_completed_downloads=not args.include_all_files,
    )
    files = [path for path in files if _file_cycle(path) in cycles]

    long_rows: list[dict] = []
    metrics_rows: list[dict] = []
    registry = get_registry()

    for bufr_path in files:
        profiles = decode_bufr_file(
            bufr_path,
            station_id=station_id,
            registry=registry,
            decode_mode=app_cfg.decode_mode,
        )
        for profile in profiles:
            rows, metric = process_profile(
                profile,
                station_name=station_name,
                pressure_top_hpa=pressure_top,
                min_levels_to_500=pc_cfg.min_levels_to_500,
                min_inversion_delta_c=pc_cfg.min_inversion_delta_c,
            )
            long_rows.extend(rows)
            metrics_rows.append(metric)

    paths = export_all(
        long_rows,
        metrics_rows,
        output_dir,
        config_info={
            "station_id": station_id,
            "station_slug": station_slug,
            "station_name": station_name,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "pressure_top_hpa": pressure_top,
            "cycles": cycles,
        },
    )
    print(json.dumps({
        "station_id": station_id,
        "station_slug": station_slug,
        "files_processed": len(files),
        "profiles": len(metrics_rows),
        "levels": len(long_rows),
        "outputs": paths,
    }, ensure_ascii=False, indent=2))
    return 0 if metrics_rows else 1


def cmd_monthly_profile_plots(
    pc_cfg: ProfileClimateConfig,
    args: argparse.Namespace,
) -> int:
    station_id, station_slug, station_name = _resolve_station(pc_cfg, args.station)
    start = _parse_date(args.start_date) or pc_cfg.start_date
    end = _parse_date(args.end_date) or pc_cfg.end_date
    pressure_top = float(args.pressure_top or pc_cfg.pressure_top_hpa)

    input_path = Path(args.input or "gdex_outputs/profile_climate/profiles_long.csv")
    metrics_path = Path(args.metrics or "gdex_outputs/profile_climate/profile_metrics.csv")
    output_root = Path(args.output or "gdex_outputs/monthly_temperature_profiles")

    long_rows = [r for r in _load_csv_rows(input_path) if normalize_station_id(r.get("station_id")) == station_id]
    metrics_rows = [r for r in _load_csv_rows(metrics_path) if normalize_station_id(r.get("station_id")) == station_id]

    if not long_rows:
        print(f"Нет данных в {input_path} для станции {station_id}", file=sys.stderr)
        return 1

    written = render_all_monthly_plots(
        station_slug=station_slug,
        station_name=station_name,
        long_rows=long_rows,
        metrics_rows=metrics_rows,
        output_root=output_root,
        start_year=start.year,
        end_year=end.year,
        start_month=start.month,
        end_month=end.month,
        pressure_top_hpa=pressure_top,
        max_surface_pressure_hpa=pc_cfg.max_surface_pressure_hpa,
        plot_only_good=pc_cfg.plot_only_good,
        plot_min_levels=pc_cfg.plot_min_levels,
        min_profiles_per_month=pc_cfg.min_profiles_per_month,
    )
    print(json.dumps({
        "station_id": station_id,
        "plots_written": len(written),
        "outputs": written[:20],
    }, ensure_ascii=False, indent=2))
    return 0 if written else 1


def cmd_discover_stations(app_cfg: AppConfig, args: argparse.Namespace) -> int:
    start = _parse_date(args.start_date)
    end = _parse_date(args.end_date)
    files = list_bufr_files(
        app_cfg,
        start_date=start,
        end_date=end,
        limit=args.limit_files,
        only_completed_downloads=not args.include_all_files,
    )

    registry = get_registry()
    stations: dict[str, dict] = defaultdict(lambda: {"count": 0, "lat": None, "lon": None, "names": set()})

    for bufr_path in files:
        profiles = decode_bufr_file(
            bufr_path,
            max_profiles=args.max_profiles_per_file,
            registry=registry,
            decode_mode=app_cfg.decode_mode,
        )
        for profile in profiles:
            sid = normalize_station_id(profile.station_id)
            if not sid:
                continue
            entry = stations[sid]
            entry["count"] += 1
            if profile.latitude_deg is not None:
                entry["lat"] = profile.latitude_deg
            if profile.longitude_deg is not None:
                entry["lon"] = profile.longitude_deg

    rows = [
        {
            "station_id": sid,
            "profiles_seen": data["count"],
            "latitude_deg": data["lat"],
            "longitude_deg": data["lon"],
        }
        for sid, data in sorted(stations.items())
    ]
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


def _file_cycle(path: Path) -> str:
    match = FILENAME_RE.search(path.name)
    return match.group("cycle") if match else ""
