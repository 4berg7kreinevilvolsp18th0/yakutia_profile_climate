"""Тесты заполнения высоты (interp + baro)."""
from __future__ import annotations

import pytest

from gdex_bufr.profile_climate.height_fill import (
    STATION_ELEVATION_M,
    barometric_height_m,
    fill_profile_level_heights,
    interpolate_heights_on_pressure,
)


def test_aldan_station_elevation():
    assert STATION_ELEVATION_M["31004"] == pytest.approx(679.0)


def test_interpolate_linear_on_pressure():
    pressures = [900.0, 850.0, 800.0, 700.0]
    heights = [1000.0, None, None, 3000.0]
    out = interpolate_heights_on_pressure(pressures, heights)
    assert out[0] == 1000.0
    assert out[3] == 3000.0
    assert out[1] is not None and 1000.0 < out[1] < 3000.0
    assert out[2] is not None and out[1] < out[2] < 3000.0


def test_baro_uses_station_elevation():
    # у поверхности прирост 0 → высота станции
    z = barometric_height_m(930.0, surface_pressure_hpa=930.0, station_elevation_m=679.0)
    assert z == pytest.approx(679.0, abs=0.1)
    z2 = barometric_height_m(500.0, surface_pressure_hpa=930.0, station_elevation_m=679.0)
    assert z2 > 4000.0


def test_fill_prefers_obs_then_interp_then_baro():
    levels = [
        {"pressure_hpa": 930.0, "temperature_c": -10.0, "height_m": 679.0},
        {"pressure_hpa": 850.0, "temperature_c": -15.0, "height_m": None},
        {"pressure_hpa": 700.0, "temperature_c": -25.0, "height_m": 3000.0},
        {"pressure_hpa": 600.0, "temperature_c": -30.0, "height_m": None},  # вне якорей → baro/extr? no extrap
    ]
    filled = fill_profile_level_heights(
        levels,
        surface_pressure_hpa=930.0,
        station_id="31004",
    )
    assert filled[0]["height_source"] == "observed_or_geopot"
    assert filled[1]["height_source"] == "interp"
    assert filled[1]["height_interp_m"] is not None
    assert filled[1]["height_baro_m"] is not None
    # 600 гПа ниже 700 — экстраполяции нет → baro
    assert filled[3]["height_interp_m"] is None
    assert filled[3]["height_source"] == "baro"
