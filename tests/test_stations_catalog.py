"""Тесты каталога станций ДВ."""
from gdex_bufr.profile_climate.config import (
    load_profile_climate_config,
    load_stations_catalog,
)
from gdex_bufr.profile_climate.height_fill import station_elevation_m
from gdex_bufr.profile_climate.paths import catalog_station_dir


def test_catalog_loads_far_east_stations():
    catalog = load_stations_catalog("stations_catalog.yaml")
    assert catalog.station_by_id("31004").slug == "aldan"
    assert catalog.station_by_slug("yakutsk").station_id == "24959"
    region = catalog.stations_in_region("far_east")
    ids = {s.station_id for s in region}
    assert {"31004", "24959", "25913", "31736", "31977", "32540", "32150"} <= ids
    assert len(catalog.unique_by_slug(region)) >= 50
    assert catalog.station_by_id("31004").elevation_m == 679.0
    assert catalog.default_region == "far_east"
    assert catalog.default_station == "aldan"


def test_legacy_wmo_ids_share_slug():
    catalog = load_stations_catalog("stations_catalog.yaml")
    assert catalog.station_by_id("24122").slug == catalog.station_by_id("24125").slug == "olenek"
    assert catalog.station_by_id("24944").slug == catalog.station_by_id("24947").slug == "olekminsk"


def test_profile_config_uses_catalog(tmp_path):
    cfg = load_profile_climate_config("profile_climate_config.yaml")
    assert cfg.station_by_id("31004").region == "far_east"
    assert cfg.default_station == "aldan"
    assert len(cfg.stations_in_region("far_east")) > 2


def test_elevation_prefers_catalog():
    assert station_elevation_m("31004") == 679.0
    assert station_elevation_m("24959") == 103.0
    assert station_elevation_m("25913") == 115.0
    assert station_elevation_m("99999") is None


def test_catalog_station_dir_follows_yaml():
    assert catalog_station_dir().as_posix().endswith("stations/aldan")
    assert catalog_station_dir("yakutsk").as_posix().endswith("stations/yakutsk")


def test_catalog_missing_file_is_empty(tmp_path):
    catalog = load_stations_catalog(tmp_path / "nope.yaml")
    assert catalog.stations == []
