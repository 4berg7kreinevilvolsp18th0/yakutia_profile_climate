"""Мост к существующему meteo_parser: общие имена полей и QC-конвенции."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Iterable
from typing import Any


@dataclass
class VerticalLevel:
    pressure_hpa: float | None = None
    geopotential_height_m: float | None = None
    air_temperature_c: float | None = None
    dew_point_temperature_c: float | None = None
    wind_direction_deg: float | None = None
    wind_speed: float | None = None
    relative_humidity_percent: float | None = None
    replication_index: int | None = None
    seq: int | None = None
    vertical_significance: str | None = None
    vertical_significance_code: int | None = None
    geopotential_m2s2: float | None = None


DATA_STATUS_OK = "OK"
DATA_STATUS_PARTIAL = "PARTIAL"
DATA_STATUS_NO_THERMO = "NO_THERMO"
DATA_STATUS_EMPTY = "EMPTY"

THERMO_PLOT_TYPES = frozenset({
    "skewt", "profile", "rh", "thermo", "height", "theta_e", "ttd", "composite",
})
WIND_PLOT_TYPES = frozenset({"wind", "hodograph", "wind_shear"})


def _thermo_levels(levels: list[VerticalLevel]) -> list[VerticalLevel]:
    return [
        lv
        for lv in levels
        if lv.pressure_hpa is not None and lv.air_temperature_c is not None
    ]


def _wind_levels(levels: list[VerticalLevel]) -> list[VerticalLevel]:
    return [
        lv
        for lv in levels
        if lv.pressure_hpa is not None and lv.wind_speed is not None
    ]


def _surface_pressure_hpa(levels: list[VerticalLevel]) -> float | None:
    pressures = [lv.pressure_hpa for lv in levels if lv.pressure_hpa is not None]
    return max(pressures) if pressures else None


def estimate_geopotential_height_m(
    pressure_hpa: float,
    *,
    surface_pressure_hpa: float,
) -> float:
    """Барометрическая оценка геопотенциальной высоты (стандартная атмосфера)."""
    ratio = max(pressure_hpa / surface_pressure_hpa, 1e-6)
    return 44330.0 * (1.0 - ratio**0.1903)


# g0 и радиус Земли как в MetPy (geopotential_to_height)
_G0_M_S2 = 9.80665
_EARTH_RADIUS_M = 6_371_229.0


def geopotential_to_height_m(geopotential_m2s2: float) -> float:
    """Геопотенциал Φ [м²/с²] → геометрическая высота [м].

    Использует MetPy ``geopotential_to_height`` (учёт изменения g с высотой):
    ``z = Φ · R_e / (g0 · R_e − Φ)``.
    При недоступности MetPy — та же аналитическая формула.
    """
    phi = float(geopotential_m2s2)
    try:
        import metpy.calc as mpcalc
        from metpy.units import units

        height = mpcalc.geopotential_to_height(units.Quantity(phi, "m^2/s^2"))
        return float(height.magnitude)
    except Exception:
        denom = _G0_M_S2 * _EARTH_RADIUS_M - phi
        if abs(denom) < 1e-9:
            return float("nan")
        return (phi * _EARTH_RADIUS_M) / denom


def _calculate_rh_percent(air_c: float | None, dewpoint_c: float | None) -> float | None:
    if air_c is None or dewpoint_c is None:
        return None
    try:
        from decoders import calculate_relative_humidity

        return calculate_relative_humidity(air_c, dewpoint_c).get("relative_humidity_percent")
    except Exception:
        pass
    # Magnus (как в meteo_parser.decoders.calculate_relative_humidity)
    try:
        from math import exp

        es_t = 6.112 * exp((17.62 * air_c) / (243.12 + air_c))
        es_td = 6.112 * exp((17.62 * dewpoint_c) / (243.12 + dewpoint_c))
        return round(max(0.0, min(100.0, 100.0 * es_td / es_t)), 1)
    except Exception:
        return None


def enrich_vertical_level(
    level: VerticalLevel,
    *,
    surface_pressure_hpa: float | None,
    station_elevation_m: float | None = None,
) -> VerticalLevel:
    """Дополняю уровень: RH из T/Td и высота, если в BUFR на уровне её нет."""
    rh = level.relative_humidity_percent
    if rh is None:
        rh = _calculate_rh_percent(level.air_temperature_c, level.dew_point_temperature_c)

    height = level.geopotential_height_m
    vsig = (level.vertical_significance or "").upper()
    # 0) поверхность станции: Height of station (0-07-001)
    if height is None and vsig == "SFC" and station_elevation_m is not None:
        height = round(float(station_elevation_m), 1)
    # 1) Φ → z (MetPy), если высота отсутствует, но есть геопотенциал
    if height is None and level.geopotential_m2s2 is not None:
        height = round(geopotential_to_height_m(level.geopotential_m2s2), 1)
    # 2) иначе оценка от высоты станции + барометрия от P_sfc
    if height is None and level.pressure_hpa is not None and surface_pressure_hpa is not None:
        above = estimate_geopotential_height_m(
            level.pressure_hpa,
            surface_pressure_hpa=surface_pressure_hpa,
        )
        base = 0.0 if station_elevation_m is None else float(station_elevation_m)
        height = round(base + above, 1)

    if rh == level.relative_humidity_percent and height == level.geopotential_height_m:
        return level

    return VerticalLevel(
        pressure_hpa=level.pressure_hpa,
        geopotential_height_m=height,
        air_temperature_c=level.air_temperature_c,
        dew_point_temperature_c=level.dew_point_temperature_c,
        wind_direction_deg=level.wind_direction_deg,
        wind_speed=level.wind_speed,
        relative_humidity_percent=rh,
        replication_index=level.replication_index,
        seq=level.seq,
        vertical_significance=level.vertical_significance,
        vertical_significance_code=level.vertical_significance_code,
        geopotential_m2s2=level.geopotential_m2s2,
    )


def enrich_profile_levels(profile: "RadiosondeProfile") -> "RadiosondeProfile":
    """Обогащаю профиль BUFR полями, как в decoded-слое meteo_parser."""
    if not profile.levels:
        return profile

    station_z = profile.station_elevation_m
    if station_z is None:
        station_z = profile.metadata.get("station_elevation_m")

    # P_sfc: давление SFC (предпочтительно), иначе max P
    sfc_ps = [
        lv.pressure_hpa
        for lv in profile.levels
        if lv.pressure_hpa is not None and (lv.vertical_significance or "").upper() == "SFC"
    ]
    if sfc_ps and station_z is not None:
        # если несколько SFC — берём тот, чья будущая H ближе к station_z после назначения
        # (давление первого/ближайшего по шаблону часто вернее max P)
        sfc_levels = [
            lv for lv in profile.levels
            if lv.pressure_hpa is not None and (lv.vertical_significance or "").upper() == "SFC"
        ]
        # без высоты на уровне — предпочитаем меньший seq (sig-секция)
        surface_p = min(sfc_levels, key=lambda lv: lv.seq if lv.seq is not None else 10**9).pressure_hpa
    else:
        surface_p = _surface_pressure_hpa(profile.levels)

    enriched_levels = [
        enrich_vertical_level(
            level,
            surface_pressure_hpa=surface_p,
            station_elevation_m=None if station_z is None else float(station_z),
        )
        for level in profile.levels
    ]
    profile.levels = enriched_levels

    flags = profile.metadata.setdefault("enrichment", {})
    if station_z is not None:
        flags["station_elevation_from_bufr"] = True
    if any(lv.relative_humidity_percent is not None for lv in enriched_levels):
        flags["rh_from_t_td"] = True
    if any(
        lv.geopotential_height_m is not None and lv.geopotential_m2s2 is not None
        for lv in enriched_levels
    ):
        flags["height_from_geopotential"] = True
    if any(lv.geopotential_height_m is not None for lv in enriched_levels):
        flags["height_from_pressure_or_station"] = True
    return profile

# функция для оценки статуса данных профиля
def assess_profile_data(profile: "RadiosondeProfile", *, min_levels: int = 5) -> tuple[str, str]:
    """Статус данных профиля и краткая причина для экспорта/графиков."""
    levels = profile.levels
    if not levels:
        return DATA_STATUS_EMPTY, "no_levels"

    thermo = _thermo_levels(levels)
    wind = _wind_levels(levels)

    if not thermo:
        if wind:
            return DATA_STATUS_NO_THERMO, "wind_only"
        return DATA_STATUS_EMPTY, "no_thermo_no_wind"

    if len(thermo) < min_levels:
        return DATA_STATUS_PARTIAL, f"few_thermo_levels:{len(thermo)}"

    return DATA_STATUS_OK, ""


def profile_plot_types(profile: "RadiosondeProfile", plot_types: Iterable[str], *, min_levels: int = 2) -> list[str]:
    """Какие типы графиков можно строить без ошибок для данного профиля."""
    status = str(profile.metadata.get("data_status", DATA_STATUS_OK))
    thermo_n = len(_thermo_levels(profile.levels))
    wind_n = len(_wind_levels(profile.levels))
    allowed: list[str] = []
    for plot_type in plot_types:
        if plot_type == "map":
            allowed.append(plot_type)
        elif plot_type in THERMO_PLOT_TYPES:
            if status in {DATA_STATUS_OK, DATA_STATUS_PARTIAL} and thermo_n >= min_levels:
                allowed.append(plot_type)
        elif plot_type in WIND_PLOT_TYPES:
            if wind_n >= min_levels:
                allowed.append(plot_type)
    return allowed


@dataclass
class RadiosondeProfile:
    """Профиль radiosonde в терминах, близких к decoded_observations.csv meteo_parser."""

    source_file: str
    subset_index: int
    station_id: str | None = None
    latitude_deg: float | None = None
    longitude_deg: float | None = None
    report_datetime_utc: str | None = None
    station_elevation_m: float | None = None
    levels: list[VerticalLevel] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def data_status(self) -> str:
        return str(self.metadata.get("data_status", DATA_STATUS_OK))

    @property
    def data_status_reason(self) -> str:
        return str(self.metadata.get("data_status_reason", ""))

    def surface_row(self) -> dict[str, Any]:
        """Плоская строка для совместимости с meteo_parser decoded-слоем."""
        surface = self.levels[0] if self.levels else VerticalLevel()
        return {
            "source_file": self.source_file,
            "report_type": "BUFR_ADPUPA",
            "station_id": self.station_id,
            "latitude_deg": self.latitude_deg,
            "longitude_deg": self.longitude_deg,
            "report_datetime_utc": self.report_datetime_utc,
            "station_elevation_m": self.station_elevation_m,
            "air_temperature_c": surface.air_temperature_c,
            "dew_point_temperature_c": surface.dew_point_temperature_c,
            "wind_direction_deg": surface.wind_direction_deg,
            "wind_speed": surface.wind_speed,
            "station_pressure_hpa": surface.pressure_hpa,
            "data_status": self.data_status,
            "missing_reason": self.data_status_reason,
            "quality_flags": "",
        }


def ensure_meteo_parser_import(meteo_parser_path: Path | None) -> Path | None:
    """Добавляю meteo_parser в sys.path, если путь задан и существует."""
    if meteo_parser_path is None:
        return None
    path = Path(meteo_parser_path).resolve()
    if not path.exists():
        return None
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    return path


def load_meteo_parser_field_names(meteo_parser_path: Path | None) -> list[str]:
    """Читаю DECODED_COLUMNS из meteo_parser для согласованности имён."""
    path = ensure_meteo_parser_import(meteo_parser_path)
    if path is None:
        return []
    try:
        from parser import DECODED_COLUMNS  # type: ignore

        return list(DECODED_COLUMNS)
    except Exception:
        return []
