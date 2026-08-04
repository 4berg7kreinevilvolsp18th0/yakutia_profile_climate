"""Тесты поиска инверсии profile_climate (v2)."""
from gdex_bufr.profile_climate.inversion import (
    QUALITY_CONFIRMED,
    QUALITY_NONE,
    QUALITY_REJECTED_NO_LAPSE,
    detect_surface_inversion,
)


def _levels_from_pairs(pairs: list[tuple[float, float]], heights: list[float] | None = None) -> list[dict]:
    rows = []
    for index, (pressure, temp) in enumerate(pairs):
        rows.append({
            "pressure_hpa": pressure,
            "temperature_c": temp,
            "height_m": heights[index] if heights else None,
        })
    return rows


def test_inversion_confirmed_with_sustained_lapse():
    # рост до 850; выше — два шага падения и глубина ≥30 гПа
    levels = _levels_from_pairs([
        (1000, -30.0),
        (950, -28.0),
        (900, -25.0),
        (850, -24.0),
        (820, -26.0),
        (800, -28.0),
    ], heights=[100, 200, 300, 400, 500, 600])
    result = detect_surface_inversion(levels, min_inversion_delta_c=0.2)
    assert result.inversion_detected is True
    assert result.inversion_candidate is True
    assert result.inversion_quality == QUALITY_CONFIRMED
    assert result.inversion_top_pressure_hpa == 850
    assert result.inversion_top_temp_c == -24.0
    assert result.inversion_delta_t_c == 6.0
    assert result.inversion_top_height_m == 400
    assert result.inversion_confirm_drop_c == -2.0  # окно ≥30 гПа закрывается на 820: -26 - (-24)


def test_pocket_rejected_when_lapse_not_sustained():
    # рост до 850, лёгкое падение останавливает этап1; выше снова рост («загон»)
    levels = _levels_from_pairs([
        (1000, -30.0),
        (950, -28.0),
        (900, -25.0),
        (850, -24.0),
        (820, -26.0),  # первый шаг вниз
        (800, -25.0),  # снова вверх — нет устойчивого падения
        (770, -23.0),
    ])
    result = detect_surface_inversion(levels, min_inversion_delta_c=0.2)
    assert result.inversion_candidate is True
    assert result.inversion_detected is False
    assert result.inversion_quality == QUALITY_REJECTED_NO_LAPSE
    assert result.inversion_top_pressure_hpa == 850


def test_no_inversion_when_temperature_decreases():
    levels = _levels_from_pairs([
        (1000, -20.0),
        (900, -22.0),
        (800, -25.0),
        (700, -30.0),
    ])
    result = detect_surface_inversion(levels, min_inversion_delta_c=0.2)
    assert result.inversion_detected is False
    assert result.inversion_candidate is False
    assert result.inversion_quality == QUALITY_NONE


def test_noise_below_min_delta_ignored():
    levels = _levels_from_pairs([
        (1000, -20.0),
        (900, -19.9),
        (800, -21.0),
    ])
    result = detect_surface_inversion(levels, min_inversion_delta_c=0.2)
    assert result.inversion_detected is False
    assert result.inversion_quality == QUALITY_NONE


def test_growth_at_top_without_levels_above_rejected():
    levels = _levels_from_pairs([
        (1000, -30.0),
        (950, -28.0),
        (900, -25.0),
    ])
    result = detect_surface_inversion(levels, min_inversion_delta_c=0.2)
    assert result.inversion_candidate is True
    assert result.inversion_detected is False
    assert result.inversion_quality == QUALITY_REJECTED_NO_LAPSE
    assert result.inversion_top_pressure_hpa == 900
