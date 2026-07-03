"""Тесты поиска инверсии profile_climate."""
from gdex_bufr.profile_climate.inversion import detect_surface_inversion


def _levels_from_pairs(pairs: list[tuple[float, float]], heights: list[float] | None = None) -> list[dict]:
    rows = []
    for index, (pressure, temp) in enumerate(pairs):
        rows.append({
            "pressure_hpa": pressure,
            "temperature_c": temp,
            "height_m": heights[index] if heights else None,
        })
    return rows


def test_inversion_detected_on_synthetic_profile():
    levels = _levels_from_pairs([
        (1000, -30.0),
        (950, -28.0),
        (900, -25.0),
        (850, -24.0),
        (800, -26.0),
    ], heights=[100, 200, 300, 400, 500])
    result = detect_surface_inversion(levels, min_inversion_delta_c=0.2)
    assert result.inversion_detected is True
    assert result.inversion_top_pressure_hpa == 850
    assert result.inversion_top_temp_c == -24.0
    assert result.inversion_delta_t_c == 6.0
    assert result.inversion_top_height_m == 400


def test_no_inversion_when_temperature_decreases():
    levels = _levels_from_pairs([
        (1000, -20.0),
        (900, -22.0),
        (800, -25.0),
        (700, -30.0),
    ])
    result = detect_surface_inversion(levels, min_inversion_delta_c=0.2)
    assert result.inversion_detected is False


def test_noise_below_min_delta_ignored():
    levels = _levels_from_pairs([
        (1000, -20.0),
        (900, -19.9),
        (800, -21.0),
    ])
    result = detect_surface_inversion(levels, min_inversion_delta_c=0.2)
    assert result.inversion_detected is False
