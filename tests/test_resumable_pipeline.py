"""Проверяем незавершённую запись, повтор файла и разделение станций."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gdex_bufr.profile_climate.run_store import RunStore, fingerprint, source_key, write_part, read_part
from gdex_bufr.profile_climate.stream_export import export_store
from gdex_bufr.profile_climate.profile_averaging import compute_method_b_on_year_month_profiles
from gdex_bufr.profile_climate.article_figures.metrics import recurrence_percent_table
from gdex_bufr.profile_climate.fast_pipeline import select_stations
from gdex_bufr.profile_climate.config import load_profile_climate_config


def payload(station="31004"):
    row = {"profile_id": "p1", "station_id": station, "station_name": "Test",
           "year": 2000, "month": 1, "cycle": "00", "profile_status": "good",
           "temperature_c": -5., "pressure_hpa": 900.}
    return {"long": [row], "metrics": [row], "decoded": [], "elements": []}


def test_only_committed_unchanged_sources_are_done(tmp_path):
    source = tmp_path / "input.bufr"
    source.write_bytes(b"BUFR")
    output = tmp_path / "run"
    store = RunStore(output, {"mode": "climate"})
    try:
        part = write_part(output / "parts", source_key(source), payload())
        assert not store.completed(source)  # Сбой между записью порции и подтверждением.
        store.record(source, fingerprint(source), {"part": str(part), "profiles": 1})
        assert store.completed(source)
        store.record(source, fingerprint(source), {"error": "broken message"})
        assert not store.completed(source)
        assert list(store.results()) == []
        store.record(source, fingerprint(source), {"part": str(part), "profiles": 1})
        assert len(list(store.results())) == 1  # Повтор не дублирует результат.
        source.write_bytes(b"BUFR changed")
        assert not store.completed(source)
    finally:
        store.close()
    with pytest.raises(ValueError, match="Параметры"):
        RunStore(output, {"mode": "audit"})


def test_same_basename_different_folders(tmp_path):
    assert source_key(tmp_path / "a" / "x.bufr") != source_key(tmp_path / "b" / "x.bufr")


def test_diagnostic_bytes_round_trip_without_loss(tmp_path):
    data = payload()
    data["elements"] = [{"value": b"\x00\xff station\x80"}]
    part = write_part(tmp_path, "sample.bufr", data)
    assert read_part(part) == data


def test_stream_export_splits_within_requested_output(tmp_path):
    store = RunStore(tmp_path / "out", {})
    try:
        for station in ("31004", "24959"):
            source = tmp_path / (station + ".bufr")
            source.write_bytes(b"BUFR")
            part = write_part(store.output / "parts", source_key(source), payload(station))
            store.record(source, fingerprint(source), {"part": str(part), "profiles": 1})
        result = export_store(store, store.output, config={}, station_slugs={"31004": "aldan", "24959": "yakutsk"},
                              split=True, output_mode="climate")
        assert result["profiles"] == 2
        for slug, station in (("aldan", 31004), ("yakutsk", 24959)):
            table = pd.read_csv(store.output / "stations" / slug / "profiles_long.csv")
            assert table.station_id.tolist() == [station]
        assert not (store.output / "debufr_elements.csv").exists()
        assert not (store.output / "profiles_working.csv").exists()
    finally:
        store.close()


def test_month_counts_and_bands_respect_missing_levels():
    stats = compute_method_b_on_year_month_profiles(
        [np.array([1., np.nan, np.nan]), np.array([3., 8., np.nan])], min_year_months=2)
    assert stats[-1].tolist() == [2, 1, 0]
    for values in stats[:-1]:
        assert np.isfinite(values[0])
        assert np.isnan(values[1:]).all()


def test_recurrence_counts_unique_profiles_and_empty_denominator():
    layers = pd.DataFrame({"profile_id": ["a", "a", "bad"], "month": [1, 1, 1],
                           "position_type": ["G"] * 3, "top_height_agl_m": [20., 30., 25.]})
    qc = pd.DataFrame({"profile_id": ["a", "b", "bad"], "month": [1, 1, 1],
                       "eligible_article": [True, True, False]})
    result = recurrence_percent_table(layers, qc, bin_edges=[0., 100.], value_col="top_height_agl_m", by_month=True)
    hit = result[(result.month == 1) & (result.position_type == "G") & (result.bin_left == 0)].iloc[0]
    assert hit.profiles == 1
    assert hit.recurrence_percent == 50.
    assert result[result.month == 2].recurrence_percent.isna().all()


def test_slug_includes_historic_wmo_ids():
    config = load_profile_climate_config()
    assert {s.station_id for s in select_stations(config, "olenek", "")} == {"24122", "24125"}
    assert {s.station_id for s in select_stations(config, "24122", "")} == {"24122"}
