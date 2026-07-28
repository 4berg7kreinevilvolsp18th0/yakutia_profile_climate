"""Тесты QC наблюдений."""
from __future__ import annotations

import numpy as np

from gdex_bufr.profile_climate.obs_qc import (
    clean_observation_levels,
    interp_on_pressure_grid,
    mad_outlier_fraction,
    max_abs_dt,
    month_median_mad,
    remove_temperature_spikes_by_pressure,
    suggest_outliers_mad,
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


def test_mad_flags_outlier_profile():
    normal = []
    for i in range(8):
        normal.append({
            "profile_id": f"n{i}",
            "pressure_hpa": [900.0, 800.0, 700.0, 600.0],
            "temperature_c": [-10.0, -20.0, -30.0, -40.0],
            "heights_m": [1000.0, 2000.0, 3000.0, 4000.0],
            "n_levels": 4,
        })
    outlier = {
        "profile_id": "bad",
        "pressure_hpa": [900.0, 800.0, 700.0, 600.0],
        "temperature_c": [40.0, 30.0, 20.0, 10.0],
        "heights_m": [1000.0, 2000.0, 3000.0, 4000.0],
        "n_levels": 4,
    }
    pool = normal + [outlier]
    flagged = suggest_outliers_mad(pool, {o["profile_id"] for o in pool}, k=5.0, fraction=0.25)
    assert "bad" in flagged

    stats = month_median_mad(pool)
    assert stats is not None
    grid, median, mad = stats
    frac = mad_outlier_fraction(outlier, median, mad, grid, k=5.0)
    assert frac >= 0.25
    assert max_abs_dt(outlier) >= 0.0
