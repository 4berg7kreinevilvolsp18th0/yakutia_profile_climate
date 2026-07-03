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

    @property
    def station_id(self) -> str:
        return str(self.id).zfill(5)[-5:]


@dataclass
class ProfileClimateConfig:
    stations: list[StationConfig] = field(default_factory=list)
    start_date: date = date(1999, 1, 1)
    end_date: date = date(2026, 12, 31)
    pressure_top_hpa: float = 500.0
    variable: str = "temperature"
    min_levels_to_500: int = 5
    min_inversion_delta_c: float = 0.2
    min_profiles_per_month: int = 5
    cycles: list[str] = field(default_factory=lambda: ["00", "12"])

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


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def load_profile_climate_config(path: str | Path = "profile_climate_config.yaml") -> ProfileClimateConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    section = raw.get("profile_climate", raw)

    stations = [
        StationConfig(
            id=str(item["id"]),
            slug=str(item["slug"]),
            name=str(item.get("name", item["slug"])),
        )
        for item in section.get("stations", [])
    ]

    return ProfileClimateConfig(
        stations=stations,
        start_date=_parse_date(section.get("start_date", "1999-01-01")),
        end_date=_parse_date(section.get("end_date", "2026-12-31")),
        pressure_top_hpa=float(section.get("pressure_top_hpa", 500)),
        variable=str(section.get("variable", "temperature")),
        min_levels_to_500=int(section.get("min_levels_to_500", 5)),
        min_inversion_delta_c=float(section.get("min_inversion_delta_c", 0.2)),
        min_profiles_per_month=int(section.get("min_profiles_per_month", 5)),
        cycles=[str(c).zfill(2)[-2:] for c in section.get("cycles", ["00", "12"])],
    )
