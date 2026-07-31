"""Проверка полного набора полей профиля как в оригинальном дешифровщике."""
from __future__ import annotations

import pytest

from gdex_bufr.meteo_parser_bridge import RadiosondeProfile, VerticalLevel
from gdex_bufr.profile_climate.export import (
    DECODED_LEVEL_COLUMNS,
    PROFILE_METRICS_COLUMNS,
    PROFILES_LONG_COLUMNS,
)
from gdex_bufr.profile_climate.extract import process_profile
from gdex_bufr.xlsx_export import LEVEL_COLUMNS


def test_process_profile_exports_original_decoder_fields() -> None:
    profile = RadiosondeProfile(
        source_file="gdas.adpupa.t12z.19991002.bufr",
        subset_index=0,
        station_id="31004",
        latitude_deg=58.37,
        longitude_deg=125.22,
        report_datetime_utc="1999-10-02T12:00:00Z",
        metadata={
            "data_status": "OK",
            "data_status_reason": "",
            "table_edition": 13,
            "n_pressure_raw": 12,
            "n_temp_raw": 10,
            "n_wind_raw": 8,
            "coded_metadata": {"002001": {"value_text": "RAOBF"}, "001012": {"value": 682}},
        },
        levels=[
            VerticalLevel(
                pressure_hpa=925.0,
                geopotential_height_m=700.0,
                geopotential_m2s2=6864.0,
                air_temperature_c=-5.1,
                dew_point_temperature_c=-7.0,
                relative_humidity_percent=86.0,
                wind_direction_deg=240.0,
                wind_speed=5.0,
                seq=1,
                replication_index=0,
                vertical_significance="SFC",
                vertical_significance_code=32,
            ),
            VerticalLevel(
                pressure_hpa=850.0,
                geopotential_height_m=1400.0,
                air_temperature_c=-8.0,
                dew_point_temperature_c=-12.0,
                wind_direction_deg=250.0,
                wind_speed=8.0,
                seq=2,
                replication_index=1,
                vertical_significance="SIG",
                vertical_significance_code=4,
            ),
            VerticalLevel(
                pressure_hpa=500.0,
                geopotential_height_m=5400.0,
                air_temperature_c=-24.0,
                seq=3,
                replication_index=2,
                vertical_significance="SIG",
                vertical_significance_code=4,
            ),
        ],
    )

    long_rows, metric, decoded, elements = process_profile(profile, station_name="Aldan")
    assert len(long_rows) == 3
    row = long_rows[0]
    for col in PROFILES_LONG_COLUMNS:
        assert col in row, col
    assert row["DEW-"] == -7.0
    assert row["dew_point_temperature_c"] == -7.0
    assert row["REL"] == 86.0
    assert row["WIND"] == 240.0
    assert row["WIND.1"] == 5.0
    assert row["VSIG"] == "SFC"
    assert row["temperature_c"] == -5.1
    assert row["height_m"] == 700.0

    for col in PROFILE_METRICS_COLUMNS:
        assert col in metric, col
    assert metric["latitude_deg"] == 58.37
    assert metric["n_temp_raw"] == 10
    assert metric["data_status"] == "OK"

    assert len(decoded) == 3
    decoded_row = decoded[0]
    for col in LEVEL_COLUMNS:
        assert col in decoded_row, col
    for col in DECODED_LEVEL_COLUMNS:
        assert col in decoded_row, col
    assert decoded_row["AIR"] == pytest.approx(-5.1)
    assert decoded_row["PRES_fxy"] == "007004"
    assert decoded_row["profile_id"] == metric["profile_id"]
    assert elements == []
