"""Тесты фильтров дашборда (без Streamlit UI)."""
from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "scripts" / "profile_dashboard.py"


def _load_dashboard():
    spec = importlib.util.spec_from_file_location("profile_dashboard", DASHBOARD)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_filter_observations_cycle_and_range():
    mod = _load_dashboard()
    observations = [
        {"profile_id": "a", "date": "2020-01-01", "cycle": "00", "inversion_detected": False},
        {"profile_id": "b", "date": "2020-01-02", "cycle": "12", "inversion_detected": True},
        {"profile_id": "c", "date": "2020-01-15", "cycle": "00", "inversion_detected": True},
        {"profile_id": "d", "date": "2020-01-20", "cycle": "12", "inversion_detected": False},
    ]
    only12 = mod.filter_observations(
        observations,
        cycle_mode="12",
        day_from=date(2020, 1, 1),
        day_to=date(2020, 1, 31),
        inversion_only=False,
    )
    assert {o["profile_id"] for o in only12} == {"b", "d"}

    ranged = mod.filter_observations(
        observations,
        cycle_mode="00+12",
        day_from=date(2020, 1, 1),
        day_to=date(2020, 1, 10),
        inversion_only=False,
    )
    assert {o["profile_id"] for o in ranged} == {"a", "b"}

    inv = mod.filter_observations(
        observations,
        cycle_mode="00+12",
        day_from=date(2020, 1, 1),
        day_to=date(2020, 1, 31),
        inversion_only=True,
    )
    assert {o["profile_id"] for o in inv} == {"b", "c"}


def test_filter_by_type_and_v2_v3():
    mod = _load_dashboard()
    observations = [
        {
            "profile_id": "g",
            "date": "2020-01-01",
            "cycle": "00",
            "inversion_detected": True,
            "n_inversion_layers_v3": 1,
            "has_G_v3": True,
            "has_E_v3": False,
            "has_HE_v3": False,
            "inversion_layers_v3": [
                {
                    "position_type": "G",
                    "base_height_agl_m": 0.0,
                    "top_height_agl_m": 200.0,
                    "depth_m": 200.0,
                    "delta_t_c": 2.0,
                    "mean_gradient_c_100m": 1.0,
                }
            ],
        },
        {
            "profile_id": "he",
            "date": "2020-01-01",
            "cycle": "12",
            "inversion_detected": False,
            "n_inversion_layers_v3": 1,
            "has_G_v3": False,
            "has_E_v3": False,
            "has_HE_v3": True,
            "inversion_layers_v3": [
                {
                    "position_type": "HE",
                    "base_height_agl_m": 400.0,
                    "top_height_agl_m": 900.0,
                    "depth_m": 500.0,
                    "delta_t_c": 1.5,
                    "mean_gradient_c_100m": 0.3,
                }
            ],
        },
        {
            "profile_id": "none",
            "date": "2020-01-02",
            "cycle": "00",
            "inversion_detected": False,
            "n_inversion_layers_v3": 0,
        },
    ]
    only_g = mod.filter_observations(
        observations,
        cycle_mode="00+12",
        day_from=date(2020, 1, 1),
        day_to=date(2020, 1, 31),
        inversion_only=False,
        types=["G"],
    )
    assert [o["profile_id"] for o in only_g] == ["g"]

    only_v3 = mod.filter_observations(
        observations,
        cycle_mode="00+12",
        day_from=date(2020, 1, 1),
        day_to=date(2020, 1, 31),
        inversion_only=False,
        v2_v3_mode="only_v3",
    )
    assert [o["profile_id"] for o in only_v3] == ["he"]

    tall = mod.filter_observations(
        observations,
        cycle_mode="00+12",
        day_from=date(2020, 1, 1),
        day_to=date(2020, 1, 31),
        inversion_only=False,
        top_agl_range=(800.0, 1200.0),
    )
    assert [o["profile_id"] for o in tall] == ["he"]

    assert mod._obs_primary_type(observations[0]) == "G"
    assert mod._obs_profile_color(observations[0], color_by_class=True, fallback="#111") == mod.V3_TYPE_COLORS["G"]


def test_filter_00_plus_12_excludes_other_cycles():
    mod = _load_dashboard()
    observations = [
        {"profile_id": "c00", "date": "2020-01-01", "cycle": "00", "inversion_detected": False},
        {"profile_id": "c06", "date": "2020-01-01", "cycle": "06", "inversion_detected": False},
        {"profile_id": "c12", "date": "2020-01-01", "cycle": "12", "inversion_detected": False},
        {"profile_id": "c18", "date": "2020-01-01", "cycle": "18", "inversion_detected": False},
    ]
    only_main = mod.filter_observations(
        observations,
        cycle_mode="00+12",
        day_from=date(2020, 1, 1),
        day_to=date(2020, 1, 31),
        inversion_only=False,
    )
    assert {o["profile_id"] for o in only_main} == {"c00", "c12"}

    all_cycles = mod.filter_observations(
        observations,
        cycle_mode="Все сроки",
        day_from=date(2020, 1, 1),
        day_to=date(2020, 1, 31),
        inversion_only=False,
    )
    assert {o["profile_id"] for o in all_cycles} == {"c00", "c06", "c12", "c18"}


def test_layer_endpoint_y_matches_profile_height():
    mod = _load_dashboard()
    obs = {
        "pressure_hpa": [900.0, 850.0, 800.0],
        "heights_m": [1000.0, 1500.0, 2000.0],
        "temperature_c": [-10.0, -5.0, -8.0],
    }
    layer = {
        "base_pressure_hpa": 900.0,
        "top_pressure_hpa": 850.0,
        "base_height_m": 9999.0,
        "top_height_m": 9999.0,
        "base_temperature_c": -10.0,
        "top_temperature_c": -5.0,
        "position_type": "G",
    }
    assert mod._layer_endpoint_y(obs, layer, which="base", y_axis="pressure") == 900.0
    assert mod._layer_endpoint_y(obs, layer, which="top", y_axis="pressure") == 850.0
    assert mod._layer_endpoint_y(obs, layer, which="base", y_axis="height") == 1000.0
    assert mod._layer_endpoint_y(obs, layer, which="top", y_axis="height") == 1500.0


def test_primary_type_from_layers():
    mod = _load_dashboard()
    assert mod._primary_type_from_layers([{"position_type": "E"}]) == "E"
    assert mod._primary_type_from_layers([]) is None
