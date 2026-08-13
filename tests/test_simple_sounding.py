"""Тесты автономного контура Алдана SFC/MANL/TXPR."""
import sys
from pathlib import Path

from gdex_bufr.meteo_parser_bridge import RadiosondeProfile, VerticalLevel

_OLD_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "старое"
if str(_OLD_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_OLD_SCRIPTS))

from aldan_simple_pipeline import (  # noqa: E402
    detect_inversion,
    normalized_level_type,
    pick_preferred_surface,
    process_profile,
)


def _profile() -> RadiosondeProfile:
    return RadiosondeProfile(
        source_file="gdas.adpupa.t12z.20000914.bufr",
        subset_index=0,
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
                vertical_significance="TROP",
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
            # Повторный SFC manl-секции: не поверхность станции.
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


def test_level_labels_are_reduced_to_three_required_types():
    assert normalized_level_type("SFC") == "SFC"
    assert normalized_level_type("MANL") == "MANL"
    assert normalized_level_type("SIGT") == "TXPR"
    assert normalized_level_type("TROP") == "TXPR"
    assert normalized_level_type("SIGW") is None


def test_preferred_surface_rejects_low_height_repeated_sfc():
    profile = _profile()
    surface = pick_preferred_surface(profile.levels, 680.0)
    assert surface is not None
    assert surface.pressure_hpa == 927.0
    assert surface.geopotential_height_m == 680.0


def test_process_preserves_bufr_height_and_calculates_inversion_height():
    rows, metrics, raw_sfc = process_profile(_profile())

    assert {row["VSIG"] for row in rows} == {"SFC", "MANL", "TXPR"}
    assert rows[0]["pressure_hpa"] == 927.0
    assert all(row["pressure_hpa"] <= 929.0 for row in rows)
    assert rows[0]["height_bufr_m"] == 680.0
    assert rows[0]["height_m"] == 680.0
    assert rows[0]["height_source"] == "level"

    txpr = next(row for row in rows if row["pressure_hpa"] == 850.0)
    assert txpr["height_bufr_m"] is None
    assert txpr["geopotential_m2s2"] == 14_000.0
    assert txpr["height_phi_m"] is not None
    assert txpr["height_m"] == txpr["height_phi_m"]
    assert txpr["height_source"] == "phi"

    assert metrics["profile_status"] == "good"
    assert metrics["inversion_detected"] is True
    assert metrics["inversion_top_pressure_hpa"] == 850.0
    assert metrics["inversion_top_height_m"] == txpr["height_m"]
    assert len(raw_sfc) == 2
    assert sum(bool(row["is_preferred"]) for row in raw_sfc) == 1


def test_inversion_v2_requires_sustained_lapse():
    levels = [
        {"pressure_hpa": 927.0, "temperature_c": -30.0, "height_m": 680.0},
        {"pressure_hpa": 900.0, "temperature_c": -27.0, "height_m": 930.0},
        {"pressure_hpa": 850.0, "temperature_c": -24.0, "height_m": 1_450.0},
        {"pressure_hpa": 820.0, "temperature_c": -26.0, "height_m": 1_850.0},
        {"pressure_hpa": 790.0, "temperature_c": -28.0, "height_m": 2_150.0},
    ]
    result = detect_inversion(levels)
    assert result.inversion_detected is True
    assert result.inversion_top_height_m == 1_450.0
