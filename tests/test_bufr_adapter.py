"""Regression tests for ADPUPA BUFR level decoding."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from gdex_bufr.bufr_adapter import _decode_adpupa_flat_levels, _normalize_pressure
from gdex_bufr.meteo_parser_bridge import RadiosondeProfile, VerticalLevel
from gdex_bufr.pybufrkit_check import profile_decode_qc


class _Registry:
    def __init__(self, unit: str = "Pa") -> None:
        self.unit = unit

    def lookup_descriptor(self, _fxy: str) -> SimpleNamespace:
        return SimpleNamespace(unit=self.unit)


class _TemplateData:
    def __init__(self, descriptor_ids: list[int], values: list[float | int | None]) -> None:
        self.decoded_descriptors_all_subsets = [
            [SimpleNamespace(id=descriptor_id) for descriptor_id in descriptor_ids]
        ]
        self.decoded_values_all_subsets = [values]

    def wire(self) -> None:
        return None


def _message(descriptor_ids: list[int], values: list[float | int | None]) -> SimpleNamespace:
    template_data = _TemplateData(descriptor_ids, values)
    return SimpleNamespace(template_data=SimpleNamespace(value=template_data))


@pytest.mark.parametrize(
    ("pressure_pa", "expected_hpa"),
    [
        (100000.0, 1000.0),
        (92500.0, 925.0),
        (2000.0, 20.0),
        (1000.0, 10.0),
        (700.0, 7.0),
    ],
)
def test_pressure_descriptor_pa_is_always_converted_to_hpa(
    pressure_pa: float,
    expected_hpa: float,
) -> None:
    assert _normalize_pressure(pressure_pa, _Registry("Pa")) == pytest.approx(expected_hpa)


def test_pressure_descriptor_hpa_is_not_scaled_again() -> None:
    assert _normalize_pressure(925.0, _Registry("hPa")) == pytest.approx(925.0)


def test_adpupa_record_keeps_post_pressure_fields_on_same_level() -> None:
    # Mirrors the relevant order in NCEP ADPUPA: VSIG, pressure, geopotential,
    # temperature/dewpoint and wind. The final raw pressure is 1000 Pa (10 hPa),
    # which caused the historical false -46 C point at 1000 hPa.
    descriptor_ids = [
        8001,
        7004,
        12225,
        12227,
        11001,
        11002,
        8001,
        7004,
        10008,
        8001,
        7004,
        10008,
        12225,
        12227,
        11001,
        11002,
        8001,
        7004,
        10008,
        12225,
        12227,
        11001,
        11002,
    ]
    values = [
        64,
        92800.0,
        265.85,
        264.95,
        30,
        1.0,
        32,
        100000.0,
        784.0,
        32,
        92500.0,
        6860.0,
        265.65,
        264.85,
        275,
        6.0,
        32,
        1000.0,
        304682.0,
        227.25,
        214.25,
        225,
        14.0,
    ]

    levels = _decode_adpupa_flat_levels(
        _message(descriptor_ids, values),
        0,
        registry=_Registry("Pa"),
    )

    assert [level.pressure_hpa for level in levels] == pytest.approx(
        [928.0, 1000.0, 925.0, 10.0]
    )
    assert levels[0].air_temperature_c == pytest.approx(-7.3)
    assert levels[0].dew_point_temperature_c == pytest.approx(-8.2)
    assert levels[0].wind_direction_deg == pytest.approx(30.0)
    assert levels[0].wind_speed == pytest.approx(1.0)

    assert levels[1].air_temperature_c is None
    assert levels[1].geopotential_height_m == pytest.approx(79.9)

    assert levels[2].air_temperature_c == pytest.approx(-7.5)
    assert levels[2].geopotential_m2s2 == pytest.approx(6860.0)
    assert levels[2].wind_direction_deg == pytest.approx(275.0)

    assert levels[3].pressure_hpa == pytest.approx(10.0)
    assert levels[3].air_temperature_c == pytest.approx(-45.9)
    assert not any(
        level.pressure_hpa == pytest.approx(1000.0)
        and level.air_temperature_c == pytest.approx(-45.9)
        for level in levels
    )


def test_profile_decode_qc_flags_false_surface_temperature_tooth() -> None:
    profile = RadiosondeProfile(
        source_file="sample.bufr",
        subset_index=0,
        levels=[
            VerticalLevel(pressure_hpa=1000.0, air_temperature_c=-45.5, seq=1),
            VerticalLevel(pressure_hpa=928.0, air_temperature_c=-7.3, seq=2),
            VerticalLevel(pressure_hpa=850.0, air_temperature_c=-9.5, seq=3),
        ],
    )

    qc = profile_decode_qc(profile)

    assert not qc["ok"]
    assert len(qc["suspicious_tropospheric_jumps"]) == 1


def test_profile_decode_qc_accepts_smooth_profile() -> None:
    profile = RadiosondeProfile(
        source_file="sample.bufr",
        subset_index=0,
        levels=[
            VerticalLevel(pressure_hpa=928.0, air_temperature_c=-7.3, seq=1),
            VerticalLevel(pressure_hpa=850.0, air_temperature_c=-9.5, seq=2),
            VerticalLevel(pressure_hpa=700.0, air_temperature_c=-16.9, seq=3),
            VerticalLevel(pressure_hpa=10.0, air_temperature_c=-45.9, seq=4),
        ],
    )

    qc = profile_decode_qc(profile)

    assert qc["ok"]
    assert qc["pressure_min_hpa"] == pytest.approx(10.0)
