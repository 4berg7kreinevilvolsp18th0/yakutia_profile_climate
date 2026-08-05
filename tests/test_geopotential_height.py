"""Тесты перевода геопотенциала в высоту (MetPy)."""
from __future__ import annotations

import pytest

from gdex_bufr.meteo_parser_bridge import (
    VerticalLevel,
    enrich_vertical_level,
    geopotential_to_height_m,
)


def test_geopotential_to_height_matches_metpy_examples():
    # Пример из документации MetPy
    assert geopotential_to_height_m(0.0) == pytest.approx(0.0, abs=1e-6)
    assert geopotential_to_height_m(9805.0) == pytest.approx(999.98867965, abs=1e-4)
    assert geopotential_to_height_m(19607.0) == pytest.approx(1999.98521653, abs=1e-4)
    assert geopotential_to_height_m(29406.0) == pytest.approx(2999.98947022, abs=1e-4)


def test_enrich_fills_height_from_geopotential():
    level = VerticalLevel(
        pressure_hpa=925.0,
        air_temperature_c=-5.0,
        geopotential_m2s2=9805.0,
        geopotential_height_m=None,
    )
    enriched = enrich_vertical_level(level, surface_pressure_hpa=1000.0)
    assert enriched.geopotential_height_m == pytest.approx(999.99, abs=0.05)


def test_enrich_sfc_uses_station_height_from_bufr():
    level = VerticalLevel(
        pressure_hpa=927.0,
        air_temperature_c=12.0,
        vertical_significance="SFC",
        geopotential_height_m=None,
    )
    enriched = enrich_vertical_level(
        level,
        surface_pressure_hpa=927.0,
        station_elevation_m=680.0,
    )
    assert enriched.geopotential_height_m == pytest.approx(680.0)
