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
