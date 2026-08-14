"""Тесты поиска инверсии profile_climate (v2 + from_top)."""
import json

from gdex_bufr.profile_climate.inversion import (
    QUALITY_CONFIRMED,
    QUALITY_NONE,
    QUALITY_REJECTED_NO_LAPSE,
    detect_inversions_from_top,
    detect_surface_inversion,
    detect_surface_inversion_v2_legacy,
    v2_inversion_path,
)
from gdex_bufr.profile_climate.metrics import compute_profile_metrics


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
    _res, path = v2_inversion_path(levels, min_inversion_delta_c=0.2)
    assert [lv["pressure_hpa"] for lv in path] == [1000, 950, 900, 850]

    from_top = detect_inversions_from_top(levels, min_inversion_delta_c=0.2)
    confirmed = [h for h in from_top if h.quality == QUALITY_CONFIRMED]
    assert len(confirmed) == 1
    assert confirmed[0].pressure_hpa == 850
    assert confirmed[0].temperature_c == -24.0


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
    _res, path = v2_inversion_path(levels, min_inversion_delta_c=0.2)
    assert [lv["pressure_hpa"] for lv in path] == [1000, 950, 900, 850]

    from_top = detect_inversions_from_top(levels, min_inversion_delta_c=0.2)
    assert all(h.quality != QUALITY_CONFIRMED for h in from_top)
    assert any(
        h.quality == QUALITY_REJECTED_NO_LAPSE and h.pressure_hpa == 850 for h in from_top
    )


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
    _res, path = v2_inversion_path(levels, min_inversion_delta_c=0.2)
    assert path == []
    assert detect_inversions_from_top(levels) == []


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


def test_v2_legacy_alias_matches_detect_surface_inversion():
    """Контрольный алиас должен давать тот же результат, что и основная функция v2."""
    cases = [
        [  # confirmed
            (1000, -30.0), (950, -28.0), (900, -25.0),
            (850, -24.0), (820, -26.0), (800, -28.0),
        ],
        [  # rejected_no_lapse
            (1000, -30.0), (950, -28.0), (900, -25.0),
            (850, -24.0), (820, -26.0), (800, -25.0), (770, -23.0),
        ],
        [  # none
            (1000, -20.0), (900, -22.0), (800, -25.0), (700, -30.0),
        ],
    ]
    for pairs in cases:
        levels = _levels_from_pairs(pairs)
        a = detect_surface_inversion(levels, min_inversion_delta_c=0.2)
        b = detect_surface_inversion_v2_legacy(levels, min_inversion_delta_c=0.2)
        assert a.as_dict() == b.as_dict()


def test_v2_gold_fixture_regression():
    from pathlib import Path

    gold = json.loads(
        (Path(__file__).parent / "fixtures" / "inversion_v2_gold.json").read_text(encoding="utf-8")
    )
    defaults = gold["defaults"]
    for case in gold["cases"]:
        result = detect_surface_inversion_v2_legacy(case["levels"], **defaults)
        expect = case["expect"]
        assert result.inversion_quality == expect["inversion_quality"], case["id"]
        assert result.inversion_detected is expect["inversion_detected"], case["id"]
        if "inversion_top_pressure_hpa" in expect:
            assert result.inversion_top_pressure_hpa == expect["inversion_top_pressure_hpa"], case["id"]
        if "inversion_top_temp_c" in expect:
            assert result.inversion_top_temp_c == expect["inversion_top_temp_c"], case["id"]


def test_elevated_inversion_from_top_while_surface_v2_none():
    # У земли lapse; выше elevated-слой роста с confirm.
    levels = _levels_from_pairs([
        (1000, -20.0),
        (950, -22.0),
        (900, -24.0),
        (850, -22.0),  # начало elevated
        (800, -20.0),  # верх elevated
        (770, -22.0),
        (740, -24.0),
        (700, -26.0),
    ], heights=[0, 400, 800, 1200, 1600, 1900, 2200, 2600])
    v2 = detect_surface_inversion(levels)
    assert v2.inversion_quality == QUALITY_NONE
    assert v2.inversion_detected is False

    from_top = detect_inversions_from_top(levels)
    confirmed = [h for h in from_top if h.quality == QUALITY_CONFIRMED]
    assert len(confirmed) == 1
    assert confirmed[0].pressure_hpa == 800
    assert confirmed[0].temperature_c == -20.0


def test_two_layers_from_top_ordered_upper_first():
    # Приземный слой + elevated; оба с confirm.
    levels = _levels_from_pairs([
        (1000, -30.0),
        (950, -28.0),
        (900, -26.0),  # верх приземного
        (870, -28.0),
        (840, -30.0),
        (800, -28.0),  # начало elevated
        (760, -26.0),  # верх elevated
        (730, -28.0),
        (700, -30.0),
        (670, -32.0),
    ], heights=[0, 400, 800, 1100, 1400, 1800, 2200, 2500, 2800, 3100])
    v2 = detect_surface_inversion(levels)
    assert v2.inversion_detected is True
    assert v2.inversion_top_pressure_hpa == 900

    from_top = detect_inversions_from_top(levels)
    confirmed = [h for h in from_top if h.quality == QUALITY_CONFIRMED]
    assert len(confirmed) == 2
    # сверху вниз: сначала elevated 760, потом приземный 900
    assert confirmed[0].pressure_hpa == 760
    assert confirmed[1].pressure_hpa == 900


def test_metrics_exports_from_top_json_fields():
    levels = _levels_from_pairs([
        (1000, -30.0),
        (950, -28.0),
        (900, -25.0),
        (850, -24.0),
        (820, -26.0),
        (800, -28.0),
        (500, -35.0),
    ], heights=[100, 200, 300, 400, 500, 600, 5000])
    metrics = compute_profile_metrics(levels, pressure_top_hpa=500.0, min_levels_to_500=3)
    assert metrics["inversion_detected"] is True
    assert metrics["inversion_from_top_count"] == 1
    assert isinstance(metrics["inversion_from_top_tops"], str)
    tops = json.loads(metrics["inversion_from_top_tops"])
    assert tops[0]["pressure_hpa"] == 850
    assert tops[0]["quality"] == QUALITY_CONFIRMED
