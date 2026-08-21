"""Тесты усреднения профилей (Method A / B / C)."""
from __future__ import annotations

import numpy as np
import pytest

from gdex_bufr.profile_climate.profile_averaging import (
    AveragingConfig,
    AveragingFilters,
    compute_method_a_on_stack,
    compute_method_b_on_year_month_profiles,
    compute_profile_average,
    cycle_matches,
    filter_observations_for_averaging,
    interpolate_observation,
)
from gdex_bufr.profile_climate.profile_interpolation import (
    DEFAULT_PRESSURE_GRID_HPA,
    default_target_grid,
    interpolate_temperature_profile,
)


def _obs(
    profile_id: str,
    date: str,
    *,
    cycle: str = "00",
    pressures: list[float] | None = None,
    temps: list[float] | None = None,
) -> dict:
    p = pressures or [900.0, 850.0, 800.0]
    t = temps or [-10.0, -12.0, -14.0]
    h = [1000.0 + i * 500.0 for i in range(len(p))]
    return {
        "profile_id": profile_id,
        "date": date,
        "cycle": cycle,
        "pressure_hpa": p,
        "temperature_c": t,
        "heights_m": h,
    }


def test_method_a_equal_grid_arithmetic_mean():
    grid = default_target_grid(coordinate="pressure")
    p = np.array([900.0, 850.0, 800.0])
    t1 = np.array([-10.0, -12.0, -14.0])
    t2 = np.array([-8.0, -10.0, -12.0])
    s1 = interpolate_temperature_profile(p, None, t1, grid, coordinate="pressure")
    s2 = interpolate_temperature_profile(p, None, t2, grid, coordinate="pressure")
    stack = np.vstack([s1, s2])
    mean, *_ = compute_method_a_on_stack(stack, min_samples=1)
    expected = np.nanmean(stack, axis=0)
    np.testing.assert_allclose(mean, expected, equal_nan=True)


def test_method_a_vs_b_year_weights():
    grid = np.array([850.0])
    # year 1: one profile T=0; year 2: nine profiles T=10
    obs = [_obs("y1", "2001-03-01", temps=[0.0, 0.0, 0.0])]
    for i in range(9):
        obs.append(_obs(f"y2_{i}", "2002-03-01", temps=[10.0, 10.0, 10.0]))

    filt = AveragingFilters(year_start=2001, year_end=2002, selected_months=frozenset([3]))
    cfg_a = AveragingConfig(method="A", target_grid=grid, min_samples_a=1, min_samples_b=1)
    cfg_b = AveragingConfig(method="B", target_grid=grid, min_samples_a=1, min_samples_b=1)

    ra = compute_profile_average(obs, filt, cfg_a).months[0].central[0]
    rb = compute_profile_average(obs, filt, cfg_b).months[0].central[0]
    assert ra == pytest.approx(9.0)  # 1*0 + 9*10 = 90 / 10
    assert rb == pytest.approx(5.0)  # mean(0, 10)


def test_method_c_surface_anomaly_aligns_profiles():
    """Профили с разной Ts, но одинаковым лапсом → одинаковые аномалии."""
    grid = np.array([900.0, 850.0, 800.0])
    obs = [
        {
            "profile_id": "cold",
            "date": "2001-01-01",
            "cycle": "00",
            "pressure_hpa": [900.0, 850.0, 800.0],
            "temperature_c": [-30.0, -28.0, -26.0],
            "heights_m": [0.0, 500.0, 1000.0],
            "t_surface_c": -30.0,
        },
        {
            "profile_id": "warm",
            "date": "2001-01-02",
            "cycle": "00",
            "pressure_hpa": [900.0, 850.0, 800.0],
            "temperature_c": [-10.0, -8.0, -6.0],
            "heights_m": [0.0, 500.0, 1000.0],
            "t_surface_c": -10.0,
        },
    ]
    filt = AveragingFilters(year_start=2001, year_end=2001, selected_months=frozenset([1]))
    cfg = AveragingConfig(
        method="C",
        target_grid=grid,
        min_samples_a=1,
        min_samples_b=1,
        keep_individual_profiles=True,
    )
    out = compute_profile_average(obs, filt, cfg)
    res = out.months[0]
    # Обе аномалии: 0, +2, +4 → среднее то же
    np.testing.assert_allclose(res.central, [0.0, 2.0, 4.0], atol=1e-9)
    assert len(res.individual_profiles) == 2
    for prof in res.individual_profiles:
        np.testing.assert_allclose(prof, [0.0, 2.0, 4.0], atol=1e-9)
    assert out.metadata["quantity"] == "delta_t_surface_c"


def test_to_surface_anomaly_helper():
    from gdex_bufr.profile_climate.profile_averaging import to_surface_anomaly

    row = np.array([-20.0, -18.0, np.nan])
    out = to_surface_anomaly(row, -20.0)
    np.testing.assert_allclose(out[:2], [0.0, 2.0])
    assert np.isnan(out[2])



def test_no_extrapolation():
    grid = np.array([925.0, 850.0, 500.0])
    p = np.array([900.0, 800.0])
    t = np.array([-5.0, -15.0])
    out = interpolate_temperature_profile(p, None, t, grid, coordinate="pressure")
    assert np.isnan(out[0])
    assert np.isnan(out[2])
    assert not np.isnan(out[1])


def test_nan_not_converted_to_zero():
    grid = np.array([850.0])
    p = np.array([900.0, 800.0])
    t = np.array([-5.0, -15.0])
    out = interpolate_temperature_profile(p, None, t, np.array([500.0]), coordinate="pressure")
    assert np.isnan(out[0])
    assert out[0] != 0.0


def test_excluding_march_2008_keeps_january_2008():
    obs = [
        _obs("j", "2008-01-15"),
        _obs("m", "2008-03-15"),
    ]
    filt = AveragingFilters(
        year_start=2008,
        year_end=2008,
        selected_months=frozenset([1, 3]),
        excluded_year_months=frozenset([(2008, 3)]),
    )
    pool = filter_observations_for_averaging(obs, filt)
    assert {o["profile_id"] for o in pool} == {"j"}


def test_cycle_00_12_only():
    obs = [
        _obs("a", "2020-01-01", cycle="00"),
        _obs("b", "2020-01-02", cycle="12"),
        _obs("c", "2020-01-03", cycle="06"),
    ]
    filt = AveragingFilters(year_start=2020, year_end=2020, cycle_mode="00+12")  # type: ignore[arg-type]
    pool = filter_observations_for_averaging(obs, filt)
    assert {o["profile_id"] for o in pool} == {"a", "b"}
    assert cycle_matches("06", "00+12") is False
    assert cycle_matches("12", "00+12") is True


def test_row_order_independent():
    grid = default_target_grid(coordinate="pressure")
    obs1 = _obs("a", "2020-03-01")
    obs2 = _obs("b", "2020-03-02")
    obs1_rev = {
        **obs1,
        "pressure_hpa": list(reversed(obs1["pressure_hpa"])),
        "temperature_c": list(reversed(obs1["temperature_c"])),
        "heights_m": list(reversed(obs1["heights_m"])),
    }
    i1 = interpolate_observation(obs1, grid, coordinate="pressure")
    i2 = interpolate_observation(obs1_rev, grid, coordinate="pressure")
    assert i1 is not None and i2 is not None
    np.testing.assert_allclose(i1, i2, equal_nan=True)

    filt = AveragingFilters(year_start=2020, year_end=2020, selected_months=frozenset([3]))
    cfg = AveragingConfig(min_samples_a=1)
    r1 = compute_profile_average([obs1, obs2], filt, cfg).months[0].central
    r2 = compute_profile_average([obs1_rev, obs2], filt, cfg).months[0].central
    np.testing.assert_allclose(r1, r2, equal_nan=True)


def test_different_level_counts_after_interpolation():
    grid = np.array([850.0, 800.0])
    shallow = _obs("s", "2020-03-01", pressures=[900.0, 850.0], temps=[-5.0, -7.0])
    deep = _obs("d", "2020-03-02", pressures=[900.0, 850.0, 750.0], temps=[-5.0, -7.0, -12.0])
    s = interpolate_observation(shallow, grid, coordinate="pressure")
    d = interpolate_observation(deep, grid, coordinate="pressure")
    assert s is not None and d is not None
    assert not np.isnan(s[0]) and not np.isnan(d[0])
    assert not np.isnan(d[1])
    assert np.isnan(s[1])


def test_method_b_q_range_from_year_month_profiles():
    grid = np.array([850.0])
    ym_profiles = [
        np.array([-20.0]),
        np.array([-10.0]),
        np.array([0.0]),
        np.array([10.0]),
    ]
    _, _, q25, q75, _, _ = compute_method_b_on_year_month_profiles(ym_profiles, min_year_months=1)
    assert q25[0] == pytest.approx(-12.5)
    assert q75[0] == pytest.approx(2.5)

    stack_individual = np.array([[-20.0], [-10.0], [0.0], [10.0], [-100.0]])
    _, _, q25_a, q75_a, _, _ = compute_method_a_on_stack(stack_individual, min_samples=1)
    assert q25_a[0] != pytest.approx(q25[0])
