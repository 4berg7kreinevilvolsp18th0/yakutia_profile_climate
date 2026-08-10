"""Извлечение температурных профилей из RadiosondeProfile."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from gdex_bufr.bufr_adapter import vsig_legacy_label
from gdex_bufr.bufr_tables import BufrTablesRegistry, get_registry
from gdex_bufr.meteo_parser_bridge import (
    RadiosondeProfile,
    VerticalLevel,
    estimate_geopotential_height_m,
    geopotential_to_height_m,
)
from gdex_bufr.profile_climate.field_types import level_type_annotations
from gdex_bufr.profile_climate.height_fill import (
    STATION_ELEVATION_M,
    fill_profile_level_heights,
    station_elevation_m,
)
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

    # ADPUPA: несколько секций (sig SFC ≈ высота станции, затем manl с P~1000 и H≪z_st).
    # Поверхность станции — SFC с высотой ближе к elevation станции; уровни с P выше неё отбрасываем.
    surface = _pick_station_surface(
        profile.levels,
        station_id=profile.station_id,
        bufr_station_elevation_m=profile.station_elevation_m,
    )
    if surface is not None and surface.pressure_hpa is not None:
        p_cap = float(surface.pressure_hpa) + 2.0
        thermo = [lv for lv in thermo if float(lv.pressure_hpa) <= p_cap]
        if not any(
            abs(float(lv.pressure_hpa) - float(surface.pressure_hpa)) < 0.15
            and lv.air_temperature_c is not None
            for lv in thermo
        ):
            # поверхность могла быть без T в sig-секции — добавим её явно
            thermo.append(surface)

    # Дедуп по давлению: при равном P предпочитаем уровень с T, затем SFC
    by_pressure: dict[float, VerticalLevel] = {}
    for level in thermo:
        key = round(float(level.pressure_hpa), 1)
        prev = by_pressure.get(key)
        if prev is None:
            by_pressure[key] = level
            continue
        prev_score = (
            (2 if prev.air_temperature_c is not None else 0)
            + (1 if (prev.vertical_significance or "").upper() == "SFC" else 0)
        )
        cur_score = (
            (2 if level.air_temperature_c is not None else 0)
            + (1 if (level.vertical_significance or "").upper() == "SFC" else 0)
        )
        if cur_score >= prev_score:
            by_pressure[key] = level
    return sorted(by_pressure.values(), key=lambda lv: lv.pressure_hpa, reverse=True)


def _pick_station_surface(
    levels: list[VerticalLevel],
    *,
    station_id: str | None,
    bufr_station_elevation_m: float | None = None,
) -> VerticalLevel | None:
    """Выбирает SFC ближе к высоте станции из BUFR (0-07-001) или справочника."""
    elev = (
        bufr_station_elevation_m
        or station_elevation_m(station_id)
        or STATION_ELEVATION_M.get("31004", 679.0)
    )
    sfc = [
        lv
        for lv in levels
        if lv.pressure_hpa is not None
        and (lv.vertical_significance or "").upper() == "SFC"
    ]
    if not sfc:
        with_p = [lv for lv in levels if lv.pressure_hpa is not None]
        return max(with_p, key=lambda lv: float(lv.pressure_hpa)) if with_p else None

    with_h = [lv for lv in sfc if lv.geopotential_height_m is not None]
    if with_h:
        return min(with_h, key=lambda lv: abs(float(lv.geopotential_height_m) - float(elev)))
    # без высоты — первый SFC в порядке шаблона (обычно sig-секция)
    return min(sfc, key=lambda lv: lv.seq if lv.seq is not None else 10**9)


def _level_climate_fields(level: VerticalLevel) -> dict[str, Any]:
    """Поля уровня в стиле оригинального дешифровщика + алиасы для климатических метрик."""
    height = level.geopotential_height_m
    if height is None and level.geopotential_m2s2 is not None:
        height = round(geopotential_to_height_m(level.geopotential_m2s2), 1)
    direct_height = (
        level.height_010009_m
        if level.height_010009_m is not None
        else level.height_007007_m
    )
    height_phi = level.height_phi_m
    if height_phi is None and level.geopotential_m2s2 is not None:
        height_phi = round(geopotential_to_height_m(level.geopotential_m2s2), 1)
    vsig_wmo = level.vertical_significance
    return {
        "SEQ": level.seq,
        "VSIG": vsig_wmo,
        "VSIG_wmo": vsig_wmo,
        "VSIG_legacy": vsig_legacy_label(vsig_wmo),
        "vertical_significance_code": level.vertical_significance_code,
        "replication_index": level.replication_index,
        "PRES": level.pressure_hpa,
        "pressure_hpa": level.pressure_hpa,
        "GEOPOT": level.geopotential_m2s2,
        "geopotential_m2s2": level.geopotential_m2s2,
        "height_010009_m": level.height_010009_m,
        "height_007007_m": level.height_007007_m,
        "height_bufr_m": direct_height,
        "height_phi_m": height_phi,
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


def _direct_bufr_height_m(level: VerticalLevel) -> float | None:
    """Высота из прямых BUFR-дескрипторов (010009 / 007007)."""
    if level.height_010009_m is not None:
        return level.height_010009_m
    return level.height_007007_m


def _phi_height_m(level: VerticalLevel) -> float | None:
    """Высота из геопотенциала Φ (м²/с² → м)."""
    if level.height_phi_m is not None:
        return level.height_phi_m
    if level.geopotential_m2s2 is not None:
        return round(geopotential_to_height_m(level.geopotential_m2s2), 1)
    return None


def _resolve_height_msl(
    level: VerticalLevel,
    *,
    surface: VerticalLevel | None,
    station_z: float | None,
    p_surface: float | None,
    below_by_pressure: bool,
) -> tuple[float | None, str | None]:
    """Выбирает высоту MSL по приоритету источников.

    Порядок: прямая BUFR → Φ → барометрия ниже станции → enriched → высота станции.
    """
    direct = _direct_bufr_height_m(level)
    if direct is not None:
        return direct, "direct_bufr"

    height_phi = _phi_height_m(level)
    if height_phi is not None:
        return height_phi, "phi"

    if (
        below_by_pressure
        and level.pressure_hpa is not None
        and p_surface is not None
        and station_z is not None
    ):
        height_msl = round(
            float(station_z)
            + estimate_geopotential_height_m(
                float(level.pressure_hpa),
                surface_pressure_hpa=float(p_surface),
            ),
            1,
        )
        return height_msl, "baro_below_station"

    if level.geopotential_height_m is not None:
        return level.geopotential_height_m, "enriched"

    # Для самой поверхности станции берём справочную высоту.
    if (
        (level.vertical_significance or "").upper() == "SFC"
        and station_z is not None
        and level is surface
    ):
        return float(station_z), "station_007001"

    return None, None


def extract_decoded_levels(
    profile: RadiosondeProfile,
    *,
    registry: BufrTablesRegistry | None = None,
    type_ann: dict[str, Any] | None = None,
    pressure_top_hpa: float = 500.0,
) -> list[dict[str, Any]]:
    """Все BUFR-уровни, включая нижние секции, с MSL/AGL и QC."""
    annotations = type_ann if type_ann is not None else level_type_annotations(registry)
    station_z = (
        profile.station_elevation_m
        if profile.station_elevation_m is not None
        else station_elevation_m(profile.station_id)
    )
    surface = _pick_station_surface(
        profile.levels,
        station_id=profile.station_id,
        bufr_station_elevation_m=profile.station_elevation_m,
    )
    p_surface = surface.pressure_hpa if surface is not None else None
    rows: list[dict[str, Any]] = []
    for level in profile.levels:
        row = {**_level_row(profile, level), **annotations}
        direct_height = _direct_bufr_height_m(level)
        height_phi = _phi_height_m(level)
        below_by_pressure = (
            level.pressure_hpa is not None
            and p_surface is not None
            and float(level.pressure_hpa) > float(p_surface) + 2.0
        )
        height_msl, height_msl_source = _resolve_height_msl(
            level,
            surface=surface,
            station_z=station_z,
            p_surface=p_surface,
            below_by_pressure=below_by_pressure,
        )
        below_by_height = (
            height_msl is not None
            and station_z is not None
            and float(height_msl) < float(station_z) - 100.0
        )
        below_station = below_by_pressure or below_by_height
        in_working = (
            not below_station
            and level.pressure_hpa is not None
            and level.air_temperature_c is not None
            and pressure_top_hpa <= float(level.pressure_hpa) <= 1000.0
        )
        row.update({
            "height_010009_m": level.height_010009_m,
            "height_007007_m": level.height_007007_m,
            "height_bufr_m": direct_height,
            "height_phi_m": height_phi,
            "height_decoded_m": level.geopotential_height_m,
            "height_msl_m": height_msl,
            "height_msl_source": height_msl_source,
            "height_agl_m": (
                None
                if height_msl is None or station_z is None
                else round(float(height_msl) - float(station_z), 1)
            ),
            "station_elevation_m": station_z,
            "below_station": below_station,
            "in_working_profile": in_working,
            "qc_flag": "below_station" if below_station else "",
        })
        rows.append(row)
    return rows


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

    station_z = (
        profile.station_elevation_m
        if profile.station_elevation_m is not None
        else station_elevation_m(station_id)
    )
    surface = _pick_station_surface(
        profile.levels,
        station_id=station_id,
        bufr_station_elevation_m=profile.station_elevation_m,
    )
    surface_pressure = surface.pressure_hpa if surface is not None else None
    levels = extract_temperature_levels(profile, pressure_top_hpa=pressure_top_hpa)
    levels = fill_profile_level_heights(
        levels,
        surface_pressure_hpa=surface_pressure,
        station_id=station_id,
        station_elevation_override_m=station_z,
    )
    # Инверсия должна видеть уже окончательную высоту уровня.
    metrics = compute_profile_metrics(
        levels,
        pressure_top_hpa=pressure_top_hpa,
        min_levels_to_500=min_levels_to_500,
        min_inversion_delta_c=min_inversion_delta_c,
        n_levels_total=len(profile.levels),
    )
    metrics["station_elevation_m"] = station_z

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
    for row in extract_decoded_levels(
        profile,
        registry=reg,
        type_ann=type_ann,
        pressure_top_hpa=pressure_top_hpa,
    ):
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
