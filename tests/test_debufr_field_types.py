"""Типы данных колонок (эталон NCEPLIBS-bufr debufr/ufdump) и полный element-dump."""
from __future__ import annotations

from pathlib import Path

import pytest

from gdex_bufr.bufr_adapter import decode_bufr_file, init_decoder_tables
from gdex_bufr.bufr_tables import get_registry
from gdex_bufr.meteo_parser_bridge import RadiosondeProfile, VerticalLevel
from gdex_bufr.profile_climate.export import (
    DECODED_LEVEL_COLUMNS,
    DEBUFR_ELEMENT_COLUMNS,
    export_all,
)
from gdex_bufr.profile_climate.extract import process_profile
from gdex_bufr.profile_climate.field_types import (
    LEVEL_FIELD_FXY,
    build_field_types_rows,
    level_type_annotations,
)


def test_field_types_pres_has_unit_and_kind() -> None:
    rows = build_field_types_rows(get_registry())
    by_col = {r["column"]: r for r in rows}
    assert by_col["PRES"]["fxy"] == "007004"
    assert by_col["PRES"]["unit"]
    assert by_col["PRES"]["kind"] in {"numeric", "code", "flag", "unknown"}
    assert by_col["AIR"]["fxy"] == LEVEL_FIELD_FXY["AIR"]
    assert by_col["VSIG"]["kind"] in {"code", "flag", "numeric", "unknown"}


def test_decoded_levels_include_type_suffixes() -> None:
    profile = RadiosondeProfile(
        source_file="gdas.adpupa.t12z.19991002.bufr",
        subset_index=0,
        station_id="31004",
        latitude_deg=58.37,
        longitude_deg=125.22,
        report_datetime_utc="1999-10-02T12:00:00Z",
        metadata={"data_status": "OK"},
        levels=[
            VerticalLevel(
                pressure_hpa=925.0,
                geopotential_height_m=700.0,
                air_temperature_c=-5.1,
                dew_point_temperature_c=-7.0,
                seq=1,
                vertical_significance="SFC",
                vertical_significance_code=32,
            ),
            VerticalLevel(
                pressure_hpa=500.0,
                geopotential_height_m=5400.0,
                air_temperature_c=-24.0,
                seq=2,
            ),
        ],
    )
    long_rows, metric, decoded, elements = process_profile(profile, station_name="Aldan")
    assert long_rows
    assert metric["station_id"] == "31004"
    assert decoded
    row = decoded[0]
    assert "PRES_fxy" in row and row["PRES_fxy"] == "007004"
    assert row["PRES_unit"]
    assert row["PRES_kind"]
    assert row["AIR_fxy"] == "012101"
    assert all(col in DECODED_LEVEL_COLUMNS for col in ("PRES_unit", "AIR_kind", "VSIG_fxy"))
    # без реального template dump элементов нет
    assert elements == []


def test_real_adpupa_debufr_elements_have_pressure_and_types(tmp_path: Path) -> None:
    bufr = Path("gdex_data/raw/1999/gdas.adpupa.t12z.19991002.bufr")
    if not bufr.exists():
        pytest.skip("нет локального BUFR 1999-10-02")

    registry = init_decoder_tables({
        "directory": "gdex_data/bufr_tables",
        "wmo_version": "latest",
        "master_table_version": 43,
        "export_dir": str(tmp_path / "tables_export"),
        "export_on_update": False,
    })
    profiles = decode_bufr_file(
        bufr,
        station_id="31004",
        max_profiles=1,
        registry=registry,
        decode_mode="adpupa",
    )
    assert profiles
    long_rows, metrics, decoded, elements = process_profile(
        profiles[0],
        station_name="Aldan",
        registry=registry,
    )
    assert long_rows
    assert metrics["profile_status"] in {"good", "short", "no_500"}
    assert decoded[0]["PRES"] is not None
    assert decoded[0]["PRES_unit"]
    assert elements
    fxys = {row["fxy"] for row in elements}
    assert "007004" in fxys
    assert "012225" in fxys or "012101" in fxys
    assert all(row.get("unit") not in (None,) for row in elements if row["fxy"] == "007004")
    assert all(col in elements[0] for col in ("fxy", "unit", "kind", "scale", "nbits"))

    out = tmp_path / "out"
    paths = export_all(
        long_rows,
        [metrics],
        out,
        decoded_rows=decoded,
        element_rows=elements,
    )
    assert Path(paths["field_types"]).exists()
    assert Path(paths["debufr_elements"]).exists()
    assert Path(paths["decoded_levels"]).exists()
    header = Path(paths["decoded_levels"]).read_text(encoding="utf-8").splitlines()[0]
    assert "PRES_unit" in header
    assert "AIR_kind" in header


def test_level_type_annotations_stable() -> None:
    ann = level_type_annotations(get_registry())
    assert ann["WIND.1_fxy"] == "011002"
    assert set(DEBUFR_ELEMENT_COLUMNS) >= {"fxy", "unit", "kind", "scale"}
