"""Извлечение температурных профилей из RadiosondeProfile."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from gdex_bufr.bufr_tables import BufrTablesRegistry, get_registry
from gdex_bufr.meteo_parser_bridge import RadiosondeProfile, VerticalLevel
from gdex_bufr.profile_climate.field_types import level_type_annotations
from gdex_bufr.profile_climate.metrics import compute_profile_metrics
from gdex_bufr.xlsx_export import _level_row

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


def _thermo_levels(profile: RadiosondeProfile) -> list[VerticalLevel]:
    thermo: list[VerticalLevel] = [
        lv for lv in profile.levels
        if lv.pressure_hpa is not None and lv.air_temperature_c is not None and lv.pressure_hpa > 0
    ]
    if not thermo:
        return []

    # ADPUPA: в одном subset несколько секций (SFC/WXPR) с повторяющимся давлением
    by_pressure: dict[float, VerticalLevel] = {}
    for level in thermo:
        key = round(level.pressure_hpa, 1)
        prev = by_pressure.get(key)
        if prev is None or (level.air_temperature_c is not None and prev.air_temperature_c is None):
            by_pressure[key] = level
    return sorted(by_pressure.values(), key=lambda lv: lv.pressure_hpa, reverse=True)


def _level_climate_fields(level: VerticalLevel) -> dict[str, Any]:
    """Поля уровня в стиле оригинального дешифровщика + алиасы для климатических метрик."""
    height = level.geopotential_height_m
    if height is None and level.geopotential_m2s2 is not None:
        from gdex_bufr.meteo_parser_bridge import geopotential_to_height_m

        height = round(geopotential_to_height_m(level.geopotential_m2s2), 1)
    return {
        "SEQ": level.seq,
        "VSIG": level.vertical_significance,
        "vertical_significance_code": level.vertical_significance_code,
        "replication_index": level.replication_index,
        "PRES": level.pressure_hpa,
        "pressure_hpa": level.pressure_hpa,
        "GEOPOT": level.geopotential_m2s2,
        "geopotential_m2s2": level.geopotential_m2s2,
        "FLVL": height,
        "geopotential_height_m": height,
        "height_m": height,
        "AIR": level.air_temperature_c,
        "air_temperature_c": level.air_temperature_c,
        "temperature_c": level.air_temperature_c,
        "DEW-": level.dew_point_temperature_c,
        "dew_point_temperature_c": level.dew_point_temperature_c,
        "REL": level.relative_humidity_percent,
        "relative_humidity_percent": level.relative_humidity_percent,
        "WIND": level.wind_direction_deg,
        "wind_direction_deg": level.wind_direction_deg,
        "WIND.1": level.wind_speed,
        "wind_speed": level.wind_speed,
        "qc_flag": "",
    }


def extract_temperature_levels(
    profile: RadiosondeProfile,
    *,
    pressure_top_hpa: float = 500.0,
) -> list[dict[str, Any]]:
    """Возвращает уровни с T и P от нижнего уровня до pressure_top_hpa."""
    thermo = _thermo_levels(profile)
    if not thermo:
        return []

    trimmed = [
        lv for lv in thermo
        if pressure_top_hpa <= lv.pressure_hpa <= 1000.0
    ]
    if not trimmed:
        return []

    rows: list[dict[str, Any]] = []
    for index, level in enumerate(trimmed):
        rows.append({"level_index": index, **_level_climate_fields(level)})
    return rows


def extract_decoded_levels(
    profile: RadiosondeProfile,
    *,
    registry: BufrTablesRegistry | None = None,
    type_ann: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Все уровни профиля + type-аннотации (FXY/unit/kind) как в debufr."""
    annotations = type_ann if type_ann is not None else level_type_annotations(registry)
    return [{**_level_row(profile, level), **annotations} for level in profile.levels]


def extract_debufr_elements(
    profile: RadiosondeProfile,
    *,
    profile_id: str,
    station_name: str = "",
) -> list[dict[str, Any]]:
    """Полный element-dump из metadata (собран при декодировании)."""
    elements = profile.metadata.get("debufr_elements") or []
    rows: list[dict[str, Any]] = []
    for item in elements:
        rows.append({
            "profile_id": profile_id,
            "station_name": station_name,
            "source_file": profile.source_file,
            "station_id": normalize_station_id(profile.station_id),
            "subset_index": profile.subset_index,
            "report_datetime_utc": profile.report_datetime_utc,
            **item,
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
    registry: BufrTablesRegistry | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Извлекает long, метрики, decoded_levels (+types) и debufr_elements."""
    reg = registry or get_registry()
    type_ann = level_type_annotations(reg)

    station_id = normalize_station_id(profile.station_id)
    datetime_utc, year, month = profile_datetime_parts(profile.report_datetime_utc)
    cycle = parse_cycle_from_source(profile.source_file, profile.report_datetime_utc)
    profile_id = make_profile_id(station_id, datetime_utc or "unknown", cycle)
    meta = profile.metadata or {}

    levels = extract_temperature_levels(profile, pressure_top_hpa=pressure_top_hpa)
    metrics = compute_profile_metrics(
        levels,
        pressure_top_hpa=pressure_top_hpa,
        min_levels_to_500=min_levels_to_500,
        min_inversion_delta_c=min_inversion_delta_c,
        n_levels_total=len(profile.levels),
    )

    profile_meta = {
        "station_id": station_id,
        "station_name": station_name or "",
        "datetime_utc": datetime_utc,
        "year": year,
        "month": month,
        "cycle": cycle,
        "profile_id": profile_id,
        "subset_index": profile.subset_index,
        "latitude_deg": profile.latitude_deg,
        "longitude_deg": profile.longitude_deg,
        "data_status": profile.data_status,
        "data_status_reason": profile.data_status_reason,
        "source_file": profile.source_file,
    }

    long_rows: list[dict[str, Any]] = []
    for row in levels:
        long_rows.append({**profile_meta, **row})

    metric_row = {
        **profile_meta,
        "table_edition": meta.get("table_edition"),
        "n_pressure_raw": meta.get("n_pressure_raw"),
        "n_temp_raw": meta.get("n_temp_raw"),
        "n_wind_raw": meta.get("n_wind_raw"),
        **metrics,
    }

    decoded_rows: list[dict[str, Any]] = []
    for row in extract_decoded_levels(profile, registry=reg, type_ann=type_ann):
        decoded_rows.append({
            "profile_id": profile_id,
            "station_name": station_name or "",
            **row,
        })

    element_rows = extract_debufr_elements(
        profile,
        profile_id=profile_id,
        station_name=station_name or "",
    )
    return long_rows, metric_row, decoded_rows, element_rows
