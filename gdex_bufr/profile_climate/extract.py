"""Извлечение температурных профилей из RadiosondeProfile."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from gdex_bufr.meteo_parser_bridge import RadiosondeProfile, VerticalLevel
from gdex_bufr.profile_climate.metrics import compute_profile_metrics

FILENAME_RE = re.compile(
    r"gdas\.(?P<obs_type>[a-z]+)\.t(?P<cycle>\d{2})z\.(?P<obs_date>\d{8})\.bufr$"
)


def normalize_station_id(station_id: str | None) -> str:
    return str(station_id or "").zfill(5)[-5:]


def parse_cycle_from_source(source_file: str, report_datetime_utc: str | None) -> str:
    match = FILENAME_RE.search(source_file)
    if match:
        return match.group("cycle")
    if report_datetime_utc:
        try:
            dt = datetime.fromisoformat(report_datetime_utc.replace("Z", "+00:00"))
            return f"{dt.hour:02d}"
        except ValueError:
            pass
    return "00"


def make_profile_id(station_id: str, datetime_utc: str, cycle: str) -> str:
    dt_token = datetime_utc.replace(":", "").replace("-", "").replace("T", "_")[:13]
    return f"{normalize_station_id(station_id)}_{dt_token}_{cycle}"


def extract_temperature_levels(
    profile: RadiosondeProfile,
    *,
    pressure_top_hpa: float = 500.0,
) -> list[dict[str, Any]]:
    """Возвращает уровни с T и P от нижнего уровня до pressure_top_hpa."""
    thermo: list[VerticalLevel] = [
        lv for lv in profile.levels
        if lv.pressure_hpa is not None and lv.air_temperature_c is not None and lv.pressure_hpa > 0
    ]
    if not thermo:
        return []

    thermo.sort(key=lambda lv: lv.pressure_hpa, reverse=True)

    # ADPUPA: в одном subset несколько секций (SFC/WXPR) с повторяющимся давлением
    by_pressure: dict[float, VerticalLevel] = {}
    for level in thermo:
        key = round(level.pressure_hpa, 1)
        prev = by_pressure.get(key)
        if prev is None or (level.air_temperature_c is not None and prev.air_temperature_c is None):
            by_pressure[key] = level
    thermo = sorted(by_pressure.values(), key=lambda lv: lv.pressure_hpa, reverse=True)

    trimmed = [lv for lv in thermo if lv.pressure_hpa >= pressure_top_hpa]
    if not trimmed:
        return []

    rows: list[dict[str, Any]] = []
    for index, level in enumerate(trimmed):
        rows.append({
            "level_index": index,
            "pressure_hpa": level.pressure_hpa,
            "temperature_c": level.air_temperature_c,
            "height_m": level.geopotential_height_m,
            "qc_flag": "",
        })
    return rows


def profile_datetime_parts(report_datetime_utc: str | None) -> tuple[str, int, int]:
    if not report_datetime_utc:
        return "", 0, 0
    try:
        dt = datetime.fromisoformat(report_datetime_utc.replace("Z", "+00:00"))
        return dt.isoformat().replace("+00:00", "Z"), dt.year, dt.month
    except ValueError:
        return report_datetime_utc, 0, 0


def process_profile(
    profile: RadiosondeProfile,
    *,
    station_name: str | None = None,
    pressure_top_hpa: float = 500.0,
    min_levels_to_500: int = 5,
    min_inversion_delta_c: float = 0.2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Извлекает long-format уровни и метрики для одного профиля."""
    station_id = normalize_station_id(profile.station_id)
    datetime_utc, year, month = profile_datetime_parts(profile.report_datetime_utc)
    cycle = parse_cycle_from_source(profile.source_file, profile.report_datetime_utc)
    profile_id = make_profile_id(station_id, datetime_utc or "unknown", cycle)

    levels = extract_temperature_levels(profile, pressure_top_hpa=pressure_top_hpa)
    metrics = compute_profile_metrics(
        levels,
        pressure_top_hpa=pressure_top_hpa,
        min_levels_to_500=min_levels_to_500,
        min_inversion_delta_c=min_inversion_delta_c,
        n_levels_total=len(profile.levels),
    )

    long_rows: list[dict[str, Any]] = []
    for row in levels:
        long_rows.append({
            "station_id": station_id,
            "station_name": station_name or "",
            "datetime_utc": datetime_utc,
            "year": year,
            "month": month,
            "cycle": cycle,
            "profile_id": profile_id,
            "level_index": row["level_index"],
            "pressure_hpa": row["pressure_hpa"],
            "temperature_c": row["temperature_c"],
            "height_m": row["height_m"],
            "source_file": profile.source_file,
            "qc_flag": row["qc_flag"],
        })

    metric_row = {
        "profile_id": profile_id,
        "station_id": station_id,
        "station_name": station_name or "",
        "datetime_utc": datetime_utc,
        "year": year,
        "month": month,
        "cycle": cycle,
        "source_file": profile.source_file,
        **metrics,
    }
    return long_rows, metric_row
