"""Пути регионального дерева far_east (эталоны актуальное/прогон не трогать)."""
from __future__ import annotations

from pathlib import Path

FAR_EAST_ROOT = Path("gdex_outputs") / "far_east"
LEGACY_ACTUAL_DIR = Path("gdex_outputs") / "актуальное"
LEGACY_ALDAN_DIR = Path("gdex_outputs") / "результаты-алдан"


def station_dir(slug: str, *, root: Path | None = None) -> Path:
    return (root or FAR_EAST_ROOT) / "stations" / slug


def catalog_station_dir(slug: str | None = None, *, root: Path | None = None) -> Path:
    """Каталог станции: slug из аргумента или default_station из YAML."""
    from gdex_bufr.profile_climate.config import load_stations_catalog

    catalog = load_stations_catalog()
    return station_dir(slug or catalog.default_station or "aldan", root=root)
