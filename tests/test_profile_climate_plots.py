"""Тесты месячных графиков profile_climate."""
from pathlib import Path

import numpy as np

from gdex_bufr.profile_climate.plots import (
    _daily_mean_profiles,
    _group_profiles_by_day,
    render_monthly_temperature_profiles,
)
from gdex_bufr.profile_climate.plot_filter import filter_plot_levels, is_profile_plot_eligible


def _height_for_pressure(pressure_hpa: float) -> float:
    # грубая шкала для тестов
    return (1000.0 - pressure_hpa) * 8.0


def _append_profile(long_rows, metrics_rows, *, day: int, cycle: str, temp_offset: float):
    profile_id = f"31004_199901{day:02d}{cycle}_{cycle}"
    levels = [
        (1000, -30 + day * 0.1 + temp_offset),
        (900, -25 + day * 0.1 + temp_offset),
        (800, -28 + day * 0.1 + temp_offset),
        (700, -32 + day * 0.1 + temp_offset),
        (600, -36 + day * 0.1 + temp_offset),
        (500, -40 + day * 0.1 + temp_offset),
    ]
    for index, (pressure, temp) in enumerate(levels):
        long_rows.append({
            "station_id": "31004",
            "station_name": "Aldan",
            "datetime_utc": f"1999-01-{day:02d}T{cycle}:00:00Z",
            "year": 1999,
            "month": 1,
            "cycle": cycle,
            "profile_id": profile_id,
            "level_index": index,
            "pressure_hpa": pressure,
            "temperature_c": temp,
            "height_m": _height_for_pressure(pressure),
            "source_file": "test.bufr",
            "qc_flag": "",
        })
    metrics_rows.append({
        "profile_id": profile_id,
        "station_id": "31004",
        "station_name": "Aldan",
        "datetime_utc": f"1999-01-{day:02d}T{cycle}:00:00Z",
        "year": 1999,
        "month": 1,
        "cycle": cycle,
        "profile_status": "good",
        "inversion_detected": day % 2 == 0,
    })


def _sample_rows():
    long_rows = []
    metrics_rows = []
    for day in range(1, 6):
        _append_profile(long_rows, metrics_rows, day=day, cycle="00", temp_offset=0.0)
        _append_profile(long_rows, metrics_rows, day=day, cycle="12", temp_offset=1.0)
    return long_rows, metrics_rows


def test_filter_rejects_pressure_above_1000():
    rows = [
        {"pressure_hpa": 1010.0, "temperature_c": -5.0, "height_m": -80.0},
        {"pressure_hpa": 950.0, "temperature_c": -6.0, "height_m": 400.0},
        {"pressure_hpa": 850.0, "temperature_c": -8.0, "height_m": 1200.0},
    ]
    filtered = filter_plot_levels(rows, pressure_top_hpa=500.0, max_surface_pressure_hpa=1000.0)
    assert len(filtered) == 2
    assert all(float(r["pressure_hpa"]) <= 1000.0 for r in filtered)


def test_daily_mean_averages_cycles_per_day():
    long_rows, metrics_rows = _sample_rows()
    profiles = {}
    profile_metrics = {row["profile_id"]: row for row in metrics_rows}
    for profile_id in profile_metrics:
        profiles[profile_id] = [
            row for row in long_rows if row["profile_id"] == profile_id
        ]

    by_day = _group_profiles_by_day(profiles, profile_metrics)
    assert len(by_day) == 5
    assert all(len(day_profiles) == 2 for day_profiles in by_day.values())

    daily = _daily_mean_profiles(by_day, grid_points=6)
    day_key = "1999-01-01"
    _, mean_t, n_profiles = daily[day_key]
    assert n_profiles == 2
    assert np.isclose(mean_t[0], -29.4, atol=0.05)


def test_render_monthly_png(tmp_path: Path):
    long_rows, metrics_rows = _sample_rows()
    output_path = tmp_path / "aldan_1999_01_temperature_profiles_to_500hpa.png"
    result = render_monthly_temperature_profiles(
        station_slug="aldan",
        station_name="Aldan",
        year=1999,
        month=1,
        long_rows=long_rows,
        metrics_rows=metrics_rows,
        output_path=output_path,
        min_profiles_per_month=3,
    )
    assert result is not None
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_bad_profile_excluded(tmp_path: Path):
    long_rows, metrics_rows = _sample_rows()
    bad_id = "31004_19990199_12"
    long_rows.append({
        "station_id": "31004",
        "station_name": "Aldan",
        "datetime_utc": "1999-01-99T12:00:00Z",
        "year": 1999,
        "month": 1,
        "cycle": "12",
        "profile_id": bad_id,
        "level_index": 0,
        "pressure_hpa": 1015.0,
        "temperature_c": -50.0,
        "height_m": -100.0,
        "source_file": "test.bufr",
        "qc_flag": "",
    })
    metrics_rows.append({
        "profile_id": bad_id,
        "profile_status": "bad_pressure",
        "datetime_utc": "1999-01-99T12:00:00Z",
        "year": 1999,
        "month": 1,
        "cycle": "12",
        "inversion_detected": False,
    })
    assert not is_profile_plot_eligible(
        metrics_rows[-1],
        filter_plot_levels([long_rows[-1]]),
    )


def test_empty_month_does_not_crash(tmp_path: Path):
    output_path = tmp_path / "empty.png"
    result = render_monthly_temperature_profiles(
        station_slug="aldan",
        station_name="Aldan",
        year=1999,
        month=2,
        long_rows=[],
        metrics_rows=[],
        output_path=output_path,
    )
    assert result is None
    assert not output_path.exists()
