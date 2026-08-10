"""Тесты заполнения высоты (interp + baro) и сверки Алдана 2026-01-01."""
from __future__ import annotations

import pytest

from gdex_bufr.bufr_adapter import _adpupa_vsig_label, vsig_legacy_label
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


def test_baro_aldan_sverka_916_940_zst680():
    """Сверка: baro(916) при Psfc=940, zst=680 → 897.6."""
    z = barometric_height_m(916.0, surface_pressure_hpa=940.0, station_elevation_m=680.0)
    assert z == pytest.approx(897.6, abs=0.05)


def test_linear_interp_916_between_925_and_850():
    """Сверка: линейная H(916) между 925/797.5 и 850/1442.3 → 874.9."""
    out = interpolate_heights_on_pressure(
        [925.0, 916.0, 850.0],
        [797.5, None, 1442.3],
    )
    # interpolate_heights_on_pressure округляет до 0.1
    assert out[1] == pytest.approx(874.9, abs=0.01)
    raw = 797.5 + (1442.3 - 797.5) * (916.0 - 925.0) / (850.0 - 925.0)
    assert raw == pytest.approx(874.876, abs=0.001)


@pytest.mark.parametrize(
    "pressure,expected",
    [
        (866.0, 1304.7),
        (844.0, 1501.1),
        (797.0, 1961.7),
        (783.0, 2098.9),
        (736.0, 2559.5),
        (612.0, 3989.7),
        (512.0, 5214.0),
    ],
)
def test_linear_interp_aldan_sverka_sig_levels(pressure, expected):
    """Промежуточные SIGT между якорями MANL из сверки (округление до 0.1)."""
    anchors_p = [925.0, 850.0, 700.0, 500.0]
    anchors_h = [797.5, 1442.3, 2912.3, 5360.9]
    pressures = anchors_p + [pressure]
    heights = anchors_h + [None]
    out = interpolate_heights_on_pressure(pressures, heights)
    got = out[-1]
    assert got is not None
    assert got == pytest.approx(expected, abs=0.05)


def test_fill_prefers_obs_then_interp_then_baro():
    levels = [
        {"pressure_hpa": 930.0, "temperature_c": -10.0, "height_010009_m": 679.0},
        {"pressure_hpa": 850.0, "temperature_c": -15.0},
        {"pressure_hpa": 700.0, "temperature_c": -25.0, "height_010009_m": 3000.0},
        {"pressure_hpa": 600.0, "temperature_c": -30.0},  # вне якорей → baro
    ]
    filled = fill_profile_level_heights(
        levels,
        surface_pressure_hpa=930.0,
        station_id="31004",
    )
    assert filled[0]["height_source"] == "level"
    assert filled[1]["height_source"] == "interp"
    assert filled[1]["height_interp_m"] is not None
    assert filled[1]["height_baro_m"] is not None
    assert filled[3]["height_interp_m"] is None
    assert filled[3]["height_source"] == "baro"


def test_fill_sigt_uses_interp_not_baro_when_manl_phi_anchors():
    """Профиль с Φ на MANL и пустыми SIGT: 916 → interp, не baro/observed."""
    levels = [
        {
            "pressure_hpa": 940.0,
            "VSIG": "SFC",
            "temperature_c": -20.0,
        },
        {
            "pressure_hpa": 925.0,
            "VSIG": "MANL",
            "height_phi_m": 797.5,
            "temperature_c": -18.0,
        },
        {
            "pressure_hpa": 916.0,
            "VSIG": "SIGT",
            "temperature_c": -17.0,
        },
        {
            "pressure_hpa": 850.0,
            "VSIG": "MANL",
            "height_phi_m": 1442.3,
            "temperature_c": -15.0,
        },
        {
            "pressure_hpa": 700.0,
            "VSIG": "MANL",
            "height_phi_m": 2912.3,
            "temperature_c": -25.0,
        },
        {
            "pressure_hpa": 512.0,
            "VSIG": "SIGT",
            "temperature_c": -30.0,
        },
        {
            "pressure_hpa": 500.0,
            "VSIG": "MANL",
            "height_phi_m": 5360.9,
            "temperature_c": -32.0,
        },
    ]
    filled = fill_profile_level_heights(
        levels,
        surface_pressure_hpa=940.0,
        station_elevation_override_m=680.0,
        station_id="31004",
    )
    by_p = {row["pressure_hpa"]: row for row in filled}
    row916 = by_p[916.0]
    assert row916["height_obs_m"] is None
    assert row916["height_source"] == "interp"
    assert row916["height_interp_m"] == pytest.approx(874.9, abs=0.05)
    assert row916["height_m"] == pytest.approx(874.9, abs=0.05)
    assert row916["height_baro_m"] == pytest.approx(897.6, abs=0.1)
    assert by_p[512.0]["height_m"] == pytest.approx(5214.0, abs=0.1)
    assert by_p[500.0]["height_m"] > by_p[512.0]["height_m"]


def test_contaminated_geopotential_height_m_is_not_observed_anchor():
    """Baro, уже записанный в geopotential_height_m, не должен стать observed."""
    levels = [
        {
            "pressure_hpa": 925.0,
            "VSIG": "MANL",
            "height_phi_m": 797.5,
            "geopotential_height_m": 797.5,
        },
        {
            "pressure_hpa": 916.0,
            "VSIG": "SIGT",
            "geopotential_height_m": 897.6,  # contamination
            "height_m": 897.6,
        },
        {
            "pressure_hpa": 850.0,
            "VSIG": "MANL",
            "height_phi_m": 1442.3,
            "geopotential_height_m": 1442.3,
        },
    ]
    filled = fill_profile_level_heights(
        levels,
        surface_pressure_hpa=940.0,
        station_elevation_override_m=680.0,
    )
    row = filled[1]
    assert row["height_obs_m"] is None
    assert row["height_source"] == "interp"
    assert row["height_m"] == pytest.approx(874.9, abs=0.2)


def test_vsig_code_4_is_sigt_and_8_is_maxw():
    assert _adpupa_vsig_label(4) == "SIGT"
    assert _adpupa_vsig_label(8) == "MAXW"
    assert vsig_legacy_label("SIGT") == "TXPR"
    assert vsig_legacy_label("TROP") == "TXPR"
    assert vsig_legacy_label("SIGT+TROP") == "TXPR"
    assert vsig_legacy_label("MANL") == "MANL"
