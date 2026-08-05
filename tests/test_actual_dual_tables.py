"""Две актуальные таблицы: все BUFR-уровни и физический рабочий профиль."""
from pathlib import Path

import pandas as pd

from gdex_bufr.meteo_parser_bridge import RadiosondeProfile, VerticalLevel
from gdex_bufr.profile_climate.export import export_all
from gdex_bufr.profile_climate.extract import process_profile


def _profile() -> RadiosondeProfile:
    return RadiosondeProfile(
        source_file="gdas.adpupa.t12z.20000914.bufr",
        subset_index=2,
        station_id="31004",
        report_datetime_utc="2000-09-14T12:00:00",
        station_elevation_m=680.0,
        levels=[
            VerticalLevel(
                pressure_hpa=927.0,
                air_temperature_c=-30.0,
                geopotential_height_m=680.0,
                height_010009_m=680.0,
                vertical_significance="SFC",
                seq=1,
            ),
            VerticalLevel(
                pressure_hpa=900.0,
                air_temperature_c=-27.0,
                geopotential_height_m=930.0,
                height_010009_m=930.0,
                vertical_significance="MANL",
                seq=2,
            ),
            VerticalLevel(
                pressure_hpa=850.0,
                air_temperature_c=-24.0,
                geopotential_height_m=1_428.0,
                height_phi_m=1_428.0,
                geopotential_m2s2=14_000.0,
                vertical_significance="SIGT",
                seq=3,
            ),
            VerticalLevel(
                pressure_hpa=820.0,
                air_temperature_c=-26.0,
                geopotential_height_m=1_850.0,
                height_010009_m=1_850.0,
                vertical_significance="MANL",
                seq=4,
            ),
            VerticalLevel(
                pressure_hpa=790.0,
                air_temperature_c=-28.0,
                geopotential_height_m=2_150.0,
                height_010009_m=2_150.0,
                vertical_significance="SIGT",
                seq=5,
            ),
            VerticalLevel(
                pressure_hpa=500.0,
                air_temperature_c=-45.0,
                geopotential_height_m=5_600.0,
                height_010009_m=5_600.0,
                vertical_significance="MANL",
                seq=6,
            ),
            # Начало повторной manl-секции, физически ниже станции Алдан.
            VerticalLevel(
                pressure_hpa=992.0,
                air_temperature_c=-20.0,
                geopotential_height_m=68.0,
                height_010009_m=68.0,
                vertical_significance="SFC",
                seq=38,
            ),
        ],
    )


def test_working_profile_excludes_below_station_but_all_levels_keeps_it():
    working, metric, decoded, _ = process_profile(_profile(), station_name="Aldan")

    assert working[0]["pressure_hpa"] == 927.0
    assert working[0]["height_msl_m"] == 680.0
    assert working[0]["height_agl_m"] == 0.0
    assert all(float(row["pressure_hpa"]) <= 929.0 for row in working)

    low = next(row for row in decoded if row["pressure_hpa"] == 992.0)
    assert low["height_bufr_m"] == 68.0
    assert low["height_agl_m"] == -612.0
    assert low["below_station"] is True
    assert low["in_working_profile"] is False
    assert low["qc_flag"] == "below_station"

    phi = next(row for row in decoded if row["pressure_hpa"] == 850.0)
    assert phi["height_bufr_m"] is None
    assert phi["geopotential_m2s2"] == 14_000.0
    assert phi["height_phi_m"] is not None

    assert metric["station_elevation_m"] == 680.0
    assert metric["inversion_detected"] is True
    assert metric["inversion_top_pressure_hpa"] == 850.0
    assert metric["inversion_top_height_m"] is not None


def test_actual_export_writes_both_csv_and_named_xlsx(tmp_path: Path):
    working, metric, decoded, _ = process_profile(_profile(), station_name="Aldan")
    output = tmp_path / "актуальное"

    paths = export_all(
        working,
        [metric],
        output,
        decoded_rows=decoded,
    )

    assert (output / "profiles_long.csv").exists()
    assert (output / "profiles_working.csv").exists()
    assert (output / "decoded_all_levels.csv").exists()
    assert (output / "profile_metrics.csv").exists()
    assert Path(paths["xlsx"]).name == "aldan_actual.xlsx"
    sheets = set(pd.ExcelFile(paths["xlsx"]).sheet_names)
    assert {"profiles_working", "profile_metrics", "decoded_all_levels"} <= sheets
