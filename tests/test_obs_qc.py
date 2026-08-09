"""Тесты QC наблюдений."""
from __future__ import annotations

import numpy as np

from gdex_bufr.profile_climate.obs_qc import (
    clean_observation_levels,
    form_rmse,
    interp_on_pressure_grid,
    is_spike_outlier,
    max_abs_dt,
    month_median_shape,
    prepare_plot_arrays,
    raw_plot_arrays,
    remove_hampel_spike_levels,
    remove_temperature_spikes_by_pressure,
    spike_scores,
    suggest_outliers_form,
    suggest_outliers_spike,
)


def test_interp_on_pressure_grid_no_extrapolation():
    p = np.array([900.0, 800.0, 700.0])
    t = np.array([-10.0, -20.0, -30.0])
    grid = np.array([950.0, 850.0, 650.0])
    out = interp_on_pressure_grid(p, t, grid)
    assert np.isnan(out[0])
    assert abs(out[1] - (-15.0)) < 1e-9
    assert np.isnan(out[2])


def test_remove_spike_by_pressure():
    levels = [
        {"pressure_hpa": 900.0, "temperature_c": -10.0, "height_m": 1000.0},
        {"pressure_hpa": 898.0, "temperature_c": 20.0, "height_m": 1020.0},  # spike
        {"pressure_hpa": 850.0, "temperature_c": -15.0, "height_m": 1500.0},
    ]
    cleaned = remove_temperature_spikes_by_pressure(levels, max_delta_c=10.0, max_dp_hpa=5.0)
    assert len(cleaned) == 2
    assert cleaned[1]["pressure_hpa"] == 850.0


def test_clean_observation_levels_temp_range():
    levels = [
        {"pressure_hpa": 900.0, "temperature_c": -10.0, "height_m": 1000.0},
        {"pressure_hpa": 850.0, "temperature_c": -100.0, "height_m": 1500.0},  # too cold
        {"pressure_hpa": 800.0, "temperature_c": -20.0, "height_m": 2000.0},
    ]
    cleaned = clean_observation_levels(levels, pressure_top_hpa=500.0, max_surface_pressure_hpa=1000.0)
    assert all(lv["temperature_c"] >= -90 for lv in cleaned)
    assert len(cleaned) == 2


def _profile(temps: list[float], profile_id: str = "p") -> dict:
    n = len(temps)
    # равномерная сетка 900 → 500
    pressures = [900.0 - i * (400.0 / (n - 1)) for i in range(n)]
    heights = [1000.0 + i * 200.0 for i in range(n)]
    return {
        "profile_id": profile_id,
        "pressure_hpa": pressures,
        "temperature_c": temps,
        "heights_m": heights,
        "n_levels": n,
        "t_surface_c": temps[0],
    }


def test_hampel_flags_single_tooth():
    temps = [-10.0, -12.0, -14.0, 25.0, -18.0, -20.0, -22.0, -24.0]  # зуб
    obs = _profile(temps, "bad")
    assert is_spike_outlier(obs)
    max_r, n_spike = spike_scores(obs)
    assert n_spike >= 1
    assert max_r >= 8.0
    assert "bad" in suggest_outliers_spike([obs], {"bad"})


def test_hampel_keeps_smooth_inversion():
    # плавная инверсия: шаги ≤4°C, без одиночного зуба
    temps = [-30.0, -27.0, -24.0, -21.0, -19.0, -22.0, -26.0, -30.0, -34.0, -38.0]
    obs = _profile(temps, "inv")
    assert not is_spike_outlier(obs)


def test_remove_hampel_spike_levels_drops_tooth():
    levels = [
        {"pressure_hpa": 900.0 - i * 50.0, "temperature_c": t, "height_m": 1000.0 + i * 200.0}
        for i, t in enumerate([-10.0, -12.0, -14.0, 25.0, -18.0, -20.0, -22.0, -24.0])
    ]
    cleaned = remove_hampel_spike_levels(levels)
    assert len(cleaned) < len(levels)
    assert all(abs(lv["temperature_c"] - 25.0) > 0.1 for lv in cleaned)


def test_form_rmse_flags_crooked_shape():
    normal = []
    for i in range(10):
        # одинаковая форма T−Ts: 0, -10, -20, -30
        base = -5.0 - i  # разный Ts
        normal.append(_profile(
            [base, base - 10.0, base - 20.0, base - 30.0, base - 35.0, base - 40.0],
            f"n{i}",
        ))
    # та же Ts, но «кривая» середина
    bad = _profile(
        [-10.0, 15.0, -5.0, -40.0, -50.0, -55.0],
        "crooked",
    )
    pool = normal + [bad]
    flagged = suggest_outliers_form(pool, {o["profile_id"] for o in pool})
    assert "crooked" in flagged

    stats = month_median_shape(pool)
    assert stats is not None
    grid, median_anom = stats
    assert form_rmse(bad, median_anom, grid) > form_rmse(normal[0], median_anom, grid)
    assert max_abs_dt(bad) >= 0.0


def test_prepare_plot_arrays_removes_height_spiral():
    obs = {
        "temperature_c": [-10.0, -12.0, -14.0, -16.0],
        "pressure_hpa": [900.0, 850.0, 800.0, 750.0],
        # провал H при подъёме по P → уровень отбрасываем, порядок T как на оси гПа
        "heights_m": [1000.0, 1500.0, 1200.0, 2000.0],
    }
    prepared = prepare_plot_arrays(obs, "height")
    assert prepared is not None
    t, h = prepared
    assert np.all(np.diff(h) > 0)
    assert list(h) == [1000.0, 1500.0, 2000.0]
    assert list(t) == [-10.0, -12.0, -16.0]


def test_raw_plot_arrays_keeps_pressure_order_on_height_axis():
    """На оси метров тот же порядок уровней, что и по гПа (даже при кривом H)."""
    obs = {
        "temperature_c": [-10.0, -12.0, -14.0],
        "pressure_hpa": [900.0, 850.0, 800.0],
        "heights_m": [1500.0, 1000.0, 2000.0],
    }
    t, h = raw_plot_arrays(obs, "height")
    assert list(t) == [-10.0, -12.0, -14.0]
    assert list(h) == [1500.0, 1000.0, 2000.0]
    t_p, _p = raw_plot_arrays(obs, "pressure")
    assert list(t_p) == list(t)


def test_prepare_plot_arrays_pressure_strictly_decreasing():
    obs = {
        "temperature_c": [-10.0, -20.0, -30.0],
        "pressure_hpa": [800.0, 900.0, 700.0],  # не по порядку
        "heights_m": [2000.0, 1000.0, 3000.0],
    }
    prepared = prepare_plot_arrays(obs, "pressure")
    assert prepared is not None
    t, p = prepared
    assert np.all(np.diff(p) < 0)


def test_raw_plot_arrays_same_temperatures_on_both_axes():
    obs = {
        "temperature_c": [-10.0, 25.0, -14.0, -16.0],
        "pressure_hpa": [900.0, 850.0, 850.0, 800.0],
        "heights_m": [1000.0, 1500.0, 1200.0, 2000.0],
    }
    temps_h, heights = raw_plot_arrays(obs, "height")
    temps_p, press = raw_plot_arrays(obs, "pressure")
    assert press.tolist() == [900.0, 850.0, 850.0, 800.0]
    assert list(temps_h) == list(temps_p)
    assert heights.tolist() == [1000.0, 1500.0, 1200.0, 2000.0]


def test_clean_drops_decreasing_height():
    levels = [
        {"pressure_hpa": 900.0, "temperature_c": -10.0, "height_m": 1000.0},
        {"pressure_hpa": 850.0, "temperature_c": -12.0, "height_m": 1500.0},
        {"pressure_hpa": 800.0, "temperature_c": -14.0, "height_m": 1300.0},  # ниже предыдущей
        {"pressure_hpa": 750.0, "temperature_c": -16.0, "height_m": 2000.0},
    ]
    cleaned = clean_observation_levels(levels, pressure_top_hpa=500.0, max_surface_pressure_hpa=1000.0)
    heights = [lv["height_m"] for lv in cleaned]
    assert heights == sorted(heights)
    assert 1300.0 not in heights
