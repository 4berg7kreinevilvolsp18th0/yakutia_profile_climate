"""Тесты сборки данных дашборда без предварительной фильтрации."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_daily_profiles.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_daily_profiles", BUILDER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_raw_mode_keeps_spikes_bad_status_singletons_and_missing_height(tmp_path):
    long_csv = tmp_path / "profiles_long.csv"
    metrics_csv = tmp_path / "profile_metrics.csv"
    pd.DataFrame([
        {
            "profile_id": "p1", "station_id": "31004", "station_name": "Aldan",
            "datetime_utc": "2020-01-01T00:00:00", "cycle": "00",
            "pressure_hpa": pressure, "temperature_c": temp, "height_m": height,
        }
        for pressure, temp, height in [
            (900.0, -10.0, 1000.0),
            (898.0, 30.0, 1020.0),
            (850.0, -14.0, 1500.0),
            (800.0, -16.0, 2000.0),
        ]
    ] + [{
        "profile_id": "p2", "station_id": "31004", "station_name": "Aldan",
        "datetime_utc": "2020-01-02T12:00:00", "cycle": "12",
        "pressure_hpa": 850.0, "temperature_c": -20.0, "height_m": None,
    }]).to_csv(long_csv, index=False)
    pd.DataFrame([
        {
            "profile_id": "p1", "profile_status": "good",
            "t_surface_c": -10.0, "inversion_detected": True,
            "inversion_top_height_m": 1020.0,
            "inversion_top_pressure_hpa": 898.0,
            "inversion_top_temp_c": 30.0,
            "inversion_delta_t_c": 40.0,
        },
        {
            "profile_id": "p2", "profile_status": "bad_pressure",
            "datetime_utc": "2020-01-02T12:00:00", "cycle": "12",
            "t_surface_c": -20.0, "inversion_detected": False,
        },
        {
            "profile_id": "p3", "profile_status": "no_temp",
            "datetime_utc": "2020-01-03T00:00:00", "cycle": "00",
            "t_surface_c": None, "inversion_detected": False,
        },
    ]).to_csv(metrics_csv, index=False)

    payload = _load_builder().build_daily_profiles(long_csv, metrics_csv)
    observations = [
        obs
        for month in payload["months"].values()
        for day in month["days"]
        for obs in day["observations"]
    ]
    by_id = {obs["profile_id"]: obs for obs in observations}

    assert payload["level_mode"] == "raw"
    assert payload["plot_min_levels"] == 1
    assert payload["n_levels"] == 5
    assert set(by_id) == {"p1", "p2", "p3"}
    assert by_id["p1"]["temperature_c"] == [-10.0, 30.0, -14.0, -16.0]
    assert by_id["p1"]["pressure_hpa"] == [900.0, 898.0, 850.0, 800.0]
    assert by_id["p1"]["inversion_detected"] is True
    assert by_id["p1"]["inversion_top_height_m"] == 1020.0
    assert by_id["p1"]["inversion_top_pressure_hpa"] == 898.0
    # нет H → барометрия от станции Алдан (679 м) при P≈P_sfc
    assert by_id["p2"]["heights_m"] == [679.0]
    assert by_id["p2"]["heights_baro_m"] == [679.0]
    assert by_id["p3"]["n_levels"] == 0
    assert by_id["p3"]["missing_levels"] is True


def test_clean_mode_keeps_legacy_status_filter(tmp_path):
    long_csv = tmp_path / "profiles_long.csv"
    metrics_csv = tmp_path / "profile_metrics.csv"
    pd.DataFrame([{
        "profile_id": "bad", "station_id": "31004", "station_name": "Aldan",
        "datetime_utc": "2020-01-01T00:00:00", "cycle": "00",
        "pressure_hpa": 850.0, "temperature_c": -20.0, "height_m": 1500.0,
    }]).to_csv(long_csv, index=False)
    pd.DataFrame([{
        "profile_id": "bad", "profile_status": "bad_pressure",
        "t_surface_c": -20.0, "inversion_detected": False,
    }]).to_csv(metrics_csv, index=False)

    payload = _load_builder().build_daily_profiles(
        long_csv,
        metrics_csv,
        level_mode="clean",
    )

    assert payload["level_mode"] == "clean"
    assert payload["n_observations"] == 0
    