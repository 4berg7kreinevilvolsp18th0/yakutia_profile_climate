"""Тесты выбора SFC ближе к высоте станции Алдан."""
from gdex_bufr.meteo_parser_bridge import RadiosondeProfile, VerticalLevel
from gdex_bufr.profile_climate.extract import _pick_station_surface, extract_temperature_levels


def test_pick_sfc_near_station_elevation_not_spurious_low_height():
    levels = [
        VerticalLevel(
            pressure_hpa=927.0,
            air_temperature_c=12.0,
            geopotential_height_m=635.0,
            vertical_significance="SFC",
            seq=1,
        ),
        VerticalLevel(
            pressure_hpa=850.0,
            air_temperature_c=15.0,
            geopotential_height_m=1420.0,
            vertical_significance="MANL",
            seq=2,
        ),
        VerticalLevel(
            pressure_hpa=992.0,
            air_temperature_c=12.0,
            geopotential_height_m=68.0,
            vertical_significance="SFC",
            seq=38,
        ),
        VerticalLevel(
            pressure_hpa=1000.0,
            air_temperature_c=11.0,
            geopotential_height_m=40.0,
            vertical_significance="MANL",
            seq=39,
        ),
    ]
    surface = _pick_station_surface(levels, station_id="31004")
    assert surface is not None
    assert surface.pressure_hpa == 927.0
    assert surface.geopotential_height_m == 635.0

    profile = RadiosondeProfile(source_file="x.bufr", subset_index=0, station_id="31004", levels=levels)
    climate = extract_temperature_levels(profile, pressure_top_hpa=500.0)
    assert climate[0]["pressure_hpa"] == 927.0
    assert all(row["pressure_hpa"] <= 929.0 for row in climate)
    assert not any(row["pressure_hpa"] >= 990.0 for row in climate)
