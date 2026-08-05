"""Экспорт декодированных BUFR-профилей в Excel (XLSX)."""
from __future__ import annotations

import threading
from pathlib import Path

import pandas as pd

from gdex_bufr.meteo_parser_bridge import RadiosondeProfile

PROFILE_COLUMNS = [
    "source_file",
    "station_id",
    "subset_index",
    "latitude_deg",
    "longitude_deg",
    "report_datetime_utc",
    "n_levels",
    "data_status",
    "data_status_reason",
    "table_edition",
    "n_pressure_raw",
    "n_temp_raw",
    "n_wind_raw",
]

LEVEL_COLUMNS = [
    "source_file",
    "station_id",
    "subset_index",
    "report_datetime_utc",
    "data_status",
    "REC",
    "OBS",
    "REPORT TIME",
    "WMO/STATION/SATELLITE ID",
    "LATI-",
    "LONGI-",
    "STN",
    "SEQ",
    "VSIG",
    "PRES",
    "GEOPOT",
    "FLVL",
    "AIR",
    "DEW-",
    "REL",
    "WIND",
    "WIND.1",
    "replication_index",
    "pressure_hpa",
    "geopotential_height_m",
    "height_010009_m",
    "height_007007_m",
    "height_phi_m",
    "geopotential_m2s2",
    "air_temperature_c",
    "dew_point_temperature_c",
    "wind_direction_deg",
    "wind_speed",
    "relative_humidity_percent",
    "vertical_significance_code",
]

_write_lock = threading.Lock()


def _report_time_compact(report_dt: str | None) -> str | None:
    if not report_dt:
        return None
    digits = "".join(ch for ch in report_dt if ch.isdigit())
    if len(digits) >= 12:
        return digits[:12]
    return digits or None


def _level_row(profile: RadiosondeProfile, level) -> dict:
    coded = profile.metadata.get("coded_metadata", {})
    obs_type = coded.get("002001", {}).get("value_text") or "RAOBF"
    stn_elev = coded.get("001012", {}).get("value")
    return {
        "source_file": profile.source_file,
        "station_id": profile.station_id or "",
        "subset_index": profile.subset_index,
        "report_datetime_utc": profile.report_datetime_utc,
        "data_status": profile.data_status,
        "REC": "ADPUPA",
        "OBS": obs_type,
        "REPORT TIME": _report_time_compact(profile.report_datetime_utc),
        "WMO/STATION/SATELLITE ID": profile.station_id or "",
        "LATI-": profile.latitude_deg,
        "LONGI-": profile.longitude_deg,
        "STN": stn_elev,
        "SEQ": level.seq,
        "VSIG": level.vertical_significance,
        "PRES": level.pressure_hpa,
        "GEOPOT": level.geopotential_m2s2,
        "FLVL": level.geopotential_height_m,
        "AIR": level.air_temperature_c,
        "DEW-": level.dew_point_temperature_c,
        "REL": level.relative_humidity_percent,
        "WIND": level.wind_direction_deg,
        "WIND.1": level.wind_speed,
        "replication_index": level.replication_index,
        "pressure_hpa": level.pressure_hpa,
        "geopotential_height_m": level.geopotential_height_m,
        "height_010009_m": level.height_010009_m,
        "height_007007_m": level.height_007007_m,
        "height_phi_m": level.height_phi_m,
        "geopotential_m2s2": level.geopotential_m2s2,
        "air_temperature_c": level.air_temperature_c,
        "dew_point_temperature_c": level.dew_point_temperature_c,
        "wind_direction_deg": level.wind_direction_deg,
        "wind_speed": level.wind_speed,
        "relative_humidity_percent": level.relative_humidity_percent,
        "vertical_significance_code": level.vertical_significance_code,
    }


def profiles_to_frames(profiles: list[RadiosondeProfile]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict] = []
    level_rows: list[dict] = []

    for profile in profiles:
        meta = profile.metadata
        summary_rows.append({
            "source_file": profile.source_file,
            "station_id": profile.station_id or "",
            "subset_index": profile.subset_index,
            "latitude_deg": profile.latitude_deg,
            "longitude_deg": profile.longitude_deg,
            "report_datetime_utc": profile.report_datetime_utc,
            "n_levels": len(profile.levels),
            "data_status": profile.data_status,
            "data_status_reason": profile.data_status_reason,
            "table_edition": meta.get("table_edition"),
            "n_pressure_raw": meta.get("n_pressure_raw"),
            "n_temp_raw": meta.get("n_temp_raw"),
            "n_wind_raw": meta.get("n_wind_raw"),
        })
        for level in profile.levels:
            level_rows.append(_level_row(profile, level))

    summary = pd.DataFrame(summary_rows, columns=PROFILE_COLUMNS)
    levels = pd.DataFrame(level_rows, columns=LEVEL_COLUMNS)
    return summary, levels


def write_profiles_xlsx(path: Path, profiles: list[RadiosondeProfile]) -> Path | None:
    """Записывает профили в XLSX (листы profiles и levels), перезаписывая файл."""
    if not profiles:
        return None
    summary, levels = profiles_to_frames(profiles)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="profiles", index=False)
        levels.to_excel(writer, sheet_name="levels", index=False)
    return path


def append_profiles_xlsx(path: Path, profiles: list[RadiosondeProfile]) -> Path | None:
    """Добавляет профили в существующий XLSX или создаёт новый."""
    if not profiles:
        return None
    path = Path(path)
    new_summary, new_levels = profiles_to_frames(profiles)

    with _write_lock:
        if path.exists():
            with pd.ExcelFile(path) as book:
                existing_summary = pd.read_excel(book, sheet_name="profiles")
                existing_levels = pd.read_excel(book, sheet_name="levels")
            summary = pd.concat([existing_summary, new_summary], ignore_index=True)
            levels = pd.concat([existing_levels, new_levels], ignore_index=True)
        else:
            summary = new_summary
            levels = new_levels

        path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            summary.to_excel(writer, sheet_name="profiles", index=False)
            levels.to_excel(writer, sheet_name="levels", index=False)
    return path
