"""Тесты метрик profile_climate."""
from gdex_bufr.profile_climate.metrics import (
    PROFILE_STATUS_GOOD,
    PROFILE_STATUS_NO_500,
    PROFILE_STATUS_SHORT,
    compute_profile_metrics,
)


def _levels_from_pairs(pairs: list[tuple[float, float]]) -> list[dict]:
    return [
        {"pressure_hpa": p, "temperature_c": t, "height_m": None}
        for p, t in pairs
    ]


def test_n_levels_to_500():
    levels = _levels_from_pairs([
        (1000, -20),
        (850, -15),
        (700, -25),
        (600, -30),
        (500, -35),
        (400, -40),
    ])
    metrics = compute_profile_metrics(levels, pressure_top_hpa=500, min_levels_to_500=5)
    assert metrics["n_levels_to_500"] == 5
    assert metrics["profile_status"] == PROFILE_STATUS_GOOD


def test_delta_t_top_surface_c():
    levels = _levels_from_pairs([
        (1000, -25.0),
        (850, -20.0),
        (700, -30.0),
        (600, -35.0),
        (500, -40.0),
    ])
    metrics = compute_profile_metrics(levels, pressure_top_hpa=500, min_levels_to_500=5)
    assert metrics["t_surface_c"] == -25.0
    assert metrics["t_top_c"] == -40.0
    assert metrics["delta_t_top_surface_c"] == -15.0


def test_short_profile_status():
    levels = _levels_from_pairs([
        (1000, -20),
        (800, -18),
        (600, -16),
        (500, -14),
    ])
    metrics = compute_profile_metrics(levels, pressure_top_hpa=500, min_levels_to_500=5)
    assert metrics["profile_status"] == PROFILE_STATUS_SHORT


def test_no_500_profile_status():
    levels = _levels_from_pairs([
        (1000, -20),
        (900, -18),
        (800, -16),
        (700, -14),
        (600, -12),
    ])
    metrics = compute_profile_metrics(levels, pressure_top_hpa=500, min_levels_to_500=5)
    assert metrics["profile_status"] == PROFILE_STATUS_NO_500
