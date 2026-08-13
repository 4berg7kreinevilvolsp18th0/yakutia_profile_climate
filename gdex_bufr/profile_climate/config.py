"""Конфигурация profile_climate из YAML."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import yaml


@dataclass
class StationConfig:
    id: str
    slug: str
    name: str
    elevation_m: float | None = None
    region: str = ""
    latitude_deg: float | None = None
    longitude_deg: float | None = None
    enabled: bool = True

    @property
    def station_id(self) -> str:
        return str(self.id).zfill(5)[-5:]


@dataclass
class StationsCatalog:
    stations: list[StationConfig] = field(default_factory=list)
    default_region: str = "far_east"
    default_station: str = "aldan"

    def station_by_id(self, station_id: str) -> StationConfig | None:
        normalized = str(station_id).zfill(5)[-5:]
        for station in self.stations:
            if station.station_id == normalized:
                return station
        return None

    def station_by_slug(self, slug: str) -> StationConfig | None:
        for station in self.stations:
            if station.slug == slug:
                return station
        return None

    def stations_in_region(self, region: str, *, enabled_only: bool = True) -> list[StationConfig]:
        wanted = str(region or "").strip().lower()
        rows = [s for s in self.stations if str(s.region or "").strip().lower() == wanted]
        if enabled_only:
            rows = [s for s in rows if s.enabled]
        return rows

    def unique_by_slug(self, stations: list[StationConfig] | None = None) -> list[StationConfig]:
        seen: set[str] = set()
        out: list[StationConfig] = []
        for station in stations if stations is not None else self.stations:
            if station.slug in seen:
                continue
            seen.add(station.slug)
            out.append(station)
        return out


def _station_from_raw(item: dict) -> StationConfig:
    elev = item.get("elevation_m")
    lat = item.get("latitude_deg", item.get("lat"))
    lon = item.get("longitude_deg", item.get("lon"))
    enabled = item.get("enabled", True)
    return StationConfig(
        id=str(item["id"]),
        slug=str(item["slug"]),
        name=str(item.get("name", item["slug"])),
        elevation_m=None if elev in (None, "", "null") else float(elev),
        region=str(item.get("region") or ""),
        latitude_deg=None if lat in (None, "", "null") else float(lat),
        longitude_deg=None if lon in (None, "", "null") else float(lon),
        enabled=bool(enabled) if enabled not in ("", "null") else True,
    )


def load_stations_catalog(path: str | Path = "stations_catalog.yaml") -> StationsCatalog:
    config_path = Path(path)
    if not config_path.exists():
        return StationsCatalog()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    items = raw.get("stations", raw if isinstance(raw, list) else [])
    return StationsCatalog(
        stations=[_station_from_raw(item) for item in items],
        default_region=str(raw.get("default_region") or "far_east"),
        default_station=str(raw.get("default_station") or "aldan"),
    )


@dataclass
class ProfileClimateConfig:
    stations: list[StationConfig] = field(default_factory=list)
    start_date: date = date(1999, 1, 1)
    end_date: date = date(2026, 12, 31)
    pressure_top_hpa: float = 500.0
    variable: str = "temperature"
    min_levels_to_500: int = 5
    min_inversion_delta_c: float = 0.2
    confirm_drop_levels: int = 2
    confirm_depth_hpa: float = 30.0
    min_drop_delta_c: float = 0.2
    min_profiles_per_month: int = 5
    max_surface_pressure_hpa: float = 1000.0
    plot_only_good: bool = False
    plot_min_levels: int = 3
    cycles: list[str] = field(default_factory=lambda: ["00", "12"])
    default_region: str = "far_east"
    default_station: str = "aldan"
    # gap-v3 (параллельно legacy v2; климатические поля v2 не заменяет)
    inversion_v3_max_embedded_gap_m: float = 100.0
    inversion_v3_min_strength_c: float = 0.3
    inversion_v3_min_depth_m: float | None = None
    inversion_v3_he_threshold_m: float = 250.0
    inversion_v3_max_gap_drop_c: float | None = None

    def station_by_id(self, station_id: str) -> StationConfig | None:
        normalized = str(station_id).zfill(5)[-5:]
        for station in self.stations:
            if station.station_id == normalized:
                return station
        return None

    def station_by_slug(self, slug: str) -> StationConfig | None:
        for station in self.stations:
            if station.slug == slug:
                return station
        return None

    def stations_in_region(self, region: str, *, enabled_only: bool = True) -> list[StationConfig]:
        wanted = str(region or "").strip().lower()
        rows = [s for s in self.stations if str(s.region or "").strip().lower() == wanted]
        if enabled_only:
            rows = [s for s in rows if s.enabled]
        return rows

    def unique_by_slug(self, stations: list[StationConfig] | None = None) -> list[StationConfig]:
        seen: set[str] = set()
        out: list[StationConfig] = []
        for station in stations if stations is not None else self.stations:
            if station.slug in seen:
                continue
            seen.add(station.slug)
            out.append(station)
        return out


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def load_profile_climate_config(path: str | Path = "profile_climate_config.yaml") -> ProfileClimateConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    section = raw.get("profile_climate", raw)

    catalog_ref = section.get("stations_catalog", "stations_catalog.yaml")
    catalog_path = Path(catalog_ref)
    if not catalog_path.is_absolute():
        catalog_path = (config_path.parent / catalog_path).resolve() if config_path.exists() else Path(catalog_ref)
    catalog = load_stations_catalog(catalog_path) if catalog_path.exists() else StationsCatalog()
    if catalog.stations:
        stations = catalog.stations
    else:
        stations = [_station_from_raw(item) for item in section.get("stations", [])]

    return ProfileClimateConfig(
        stations=stations,
        start_date=_parse_date(section.get("start_date", "1999-01-01")),
        end_date=_parse_date(section.get("end_date", "2026-12-31")),
        pressure_top_hpa=float(section.get("pressure_top_hpa", 500)),
        variable=str(section.get("variable", "temperature")),
        min_levels_to_500=int(section.get("min_levels_to_500", 5)),
        min_inversion_delta_c=float(section.get("min_inversion_delta_c", 0.2)),
        confirm_drop_levels=int(section.get("confirm_drop_levels", 2)),
        confirm_depth_hpa=float(section.get("confirm_depth_hpa", 30)),
        min_drop_delta_c=float(section.get("min_drop_delta_c", 0.2)),
        min_profiles_per_month=int(section.get("min_profiles_per_month", 5)),
        max_surface_pressure_hpa=float(section.get("max_surface_pressure_hpa", 1000)),
        plot_only_good=bool(section.get("plot_only_good", False)),
        plot_min_levels=int(section.get("plot_min_levels", 3)),
        cycles=[str(c).zfill(2)[-2:] for c in section.get("cycles", ["00", "12"])],
        default_region=str(section.get("default_region") or catalog.default_region or "far_east"),
        default_station=str(section.get("default_station") or catalog.default_station or "aldan"),
        inversion_v3_max_embedded_gap_m=float(
            section.get("inversion_v3_max_embedded_gap_m", 100.0)
        ),
        inversion_v3_min_strength_c=float(section.get("inversion_v3_min_strength_c", 0.3)),
        inversion_v3_min_depth_m=(
            None
            if section.get("inversion_v3_min_depth_m", None) in (None, "", "null")
            else float(section["inversion_v3_min_depth_m"])
        ),
        inversion_v3_he_threshold_m=float(section.get("inversion_v3_he_threshold_m", 250.0)),
        inversion_v3_max_gap_drop_c=(
            None
            if section.get("inversion_v3_max_gap_drop_c", None) in (None, "", "null")
            else float(section["inversion_v3_max_gap_drop_c"])
        ),
    )
