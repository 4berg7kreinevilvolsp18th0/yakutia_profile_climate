"""Тесты фильтров и среднего профиля в дашборде."""
from __future__ import annotations

import importlib.util
import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "scripts" / "profile_dashboard.py"

pytest.importorskip("streamlit")
pytest.importorskip("plotly")


def _load_dashboard():
    spec = importlib.util.spec_from_file_location("profile_dashboard", DASHBOARD)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _observation(profile_id: str, **overrides) -> dict:
    obs = {
        "profile_id": profile_id,
        "date": "2020-01-01",
        "cycle": "00",
        "pressure_hpa": [900.0, 850.0, 800.0],
        "temperature_c": [-10.0, -12.0, -16.0],
        "heights_m": [1000.0, 1500.0, 2000.0],
        "inversion_detected": False,
        "inversion_candidate": False,
        "inversion_quality": "none",
    }
    obs.update(overrides)
    return obs


def test_has_levels_separates_metrics_only_profiles():
    dash = _load_dashboard()
    assert dash.has_levels(_observation("p1")) is True
    metrics_only = _observation(
        "p2",
        pressure_hpa=[],
        temperature_c=[],
        heights_m=[],
        missing_levels=True,
    )
    assert dash.has_levels(metrics_only) is False


def test_filter_by_inversion_quality_and_missing_levels():
    dash = _load_dashboard()
    observations = [
        _observation("confirmed", inversion_detected=True, inversion_quality="confirmed"),
        _observation("rejected", inversion_candidate=True, inversion_quality="rejected_no_lapse"),
        _observation(
            "empty",
            pressure_hpa=[],
            temperature_c=[],
            heights_m=[],
            missing_levels=True,
        ),
    ]
    common = {
        "cycle_mode": "00+12",
        "day_from": date(2020, 1, 1),
        "day_to": date(2020, 1, 31),
        "inversion_only": False,
    }

    everything = dash.filter_observations(observations, **common)
    assert {o["profile_id"] for o in everything} == {"confirmed", "rejected", "empty"}

    only_confirmed = dash.filter_observations(
        observations, **common, inversion_quality="confirmed"
    )
    assert [o["profile_id"] for o in only_confirmed] == ["confirmed"]

    with_levels = dash.filter_observations(observations, **common, hide_missing_levels=True)
    assert {o["profile_id"] for o in with_levels} == {"confirmed", "rejected"}


def test_month_mean_follows_curve_preparation_mode():
    """Среднее должно считаться по тем же массивам, что и нарисованные кривые."""
    dash = _load_dashboard()
    # Дубль давления 850 гПа с аномальной T: подготовка кривых его убирает, сырой режим — нет.
    obs = _observation(
        "dup",
        pressure_hpa=[900.0, 850.0, 850.0, 800.0],
        temperature_c=[-10.0, -12.0, -30.0, -16.0],
        heights_m=[1000.0, 1500.0, 1500.0, 2000.0],
    )
    enabled = {"dup"}

    raw = dash.month_mean([obs], enabled, y_axis="pressure", apply_plot_qc=False)
    prepared = dash.month_mean([obs], enabled, y_axis="pressure", apply_plot_qc=True)
    assert raw is not None and prepared is not None

    assert np.nanmin(raw[1]) < -20.0, "сырой режим сохраняет выброс −30 °C"
    assert np.nanmin(prepared[1]) > -17.0, "подготовка кривых убирает дублирующий уровень"


def test_month_mean_returns_none_without_enabled():
    dash = _load_dashboard()
    assert dash.month_mean([_observation("p1")], set(), y_axis="pressure") is None


def _build_payload(tmp_path: Path) -> Path:
    """Готовит daily_profiles.json тем же сборщиком, что и рабочий конвейер."""
    import pandas as pd

    spec = importlib.util.spec_from_file_location(
        "build_daily_profiles", ROOT / "scripts" / "build_daily_profiles.py"
    )
    assert spec and spec.loader
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)

    long_csv = tmp_path / "profiles_long.csv"
    metrics_csv = tmp_path / "profile_metrics.csv"
    pd.DataFrame([
        {
            "profile_id": "p1", "station_id": "31004", "station_name": "Aldan",
            "datetime_utc": "2020-01-01T00:00:00", "cycle": "00",
            "pressure_hpa": pressure, "temperature_c": temp, "height_m": height,
        }
        for pressure, temp, height in [
            (920.0, -28.0, 700.0),
            (900.0, -24.0, 900.0),
            (850.0, -26.0, 1400.0),
            (700.0, -35.0, 3000.0),
        ]
    ]).to_csv(long_csv, index=False)
    pd.DataFrame([{
        "profile_id": "p1", "profile_status": "good",
        "t_surface_c": -28.0, "inversion_detected": True,
        "inversion_candidate": True, "inversion_quality": "confirmed",
        "inversion_top_height_m": 900.0, "inversion_top_pressure_hpa": 900.0,
        "inversion_top_temp_c": -24.0, "inversion_delta_t_c": 4.0,
        "p_surface_hpa": 920.0, "station_elevation_m": 679.0,
    }]).to_csv(metrics_csv, index=False)

    payload = builder.build_daily_profiles(long_csv, metrics_csv)
    out = tmp_path / "daily_profiles.json"
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return out


def test_dashboard_renders_payload_from_builder(tmp_path):
    """Смоук: приложение целиком отрабатывает на свежесобранном JSON."""
    from streamlit.testing.v1 import AppTest

    data_path = _build_payload(tmp_path)
    app = AppTest.from_file(str(DASHBOARD), default_timeout=60).run()
    app.sidebar.text_input[0].set_value(str(data_path)).run()

    assert not app.exception
    assert not app.error
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["На графике"] == "1 / 1"
    assert metrics["Без уровней"] == "0 / 0"
    inv_metric = next(
        (v for k, v in metrics.items() if k.startswith("С инверсией")),
        None,
    )
    assert inv_metric == "1"


def test_dashboard_reports_broken_json(tmp_path):
    from streamlit.testing.v1 import AppTest

    broken = tmp_path / "daily_profiles.json"
    broken.write_text("{ это не json", encoding="utf-8")
    app = AppTest.from_file(str(DASHBOARD), default_timeout=60).run()
    app.sidebar.text_input[0].set_value(str(broken)).run()

    assert not app.exception
    assert any("Не удалось прочитать" in e.value for e in app.error)
