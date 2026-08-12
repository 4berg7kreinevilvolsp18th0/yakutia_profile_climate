"""Unit-тесты gap-merge инверсий v3."""
from __future__ import annotations

import numpy as np

from gdex_bufr.profile_climate.inversion import detect_surface_inversion
from gdex_bufr.profile_climate.inversion_layers import (
    detect_inversion_layers_gap_v3,
    merge_short_gaps,
    positive_runs,
    summarize_inversion_layers,
)


def test_positive_runs_basic():
    # интервалы: + + - +
    grad = np.array([1.0, 2.0, -1.0, 0.5])
    assert positive_runs(grad) == [(0, 2), (3, 4)]


def test_surface_g_layer():
    z = np.array([680.0, 780.0, 880.0, 1080.0])
    t = np.array([-30.0, -28.0, -25.0, -27.0])
    p = np.array([930.0, 910.0, 890.0, 850.0])
    layers = detect_inversion_layers_gap_v3(z, t, p, min_strength_c=0.3)
    assert len(layers) == 1
    assert layers[0].position_type == "G"
    assert layers[0].base_idx == 0
    assert abs(layers[0].delta_t_c - 5.0) < 1e-9
    assert abs(layers[0].depth_m - 200.0) < 1e-9


def test_elevated_e_and_he():
    # поверхность без инверсии, затем E (база ~100 м AGL), затем HE (база ~400 м)
    z = np.array([700.0, 750.0, 800.0, 900.0, 1100.0, 1200.0, 1400.0])
    t = np.array([-20.0, -21.0, -19.0, -17.0, -18.5, -16.0, -17.0])
    p = np.array([930.0, 920.0, 910.0, 890.0, 850.0, 830.0, 800.0])
    layers = detect_inversion_layers_gap_v3(
        z, t, p, min_strength_c=0.3, max_embedded_gap_m=50.0, he_threshold_m=250.0,
    )
    types = [ly.position_type for ly in layers]
    assert "E" in types
    assert "HE" in types
    assert "G" not in types


def test_gap_below_threshold_merges():
    # два положительных сегмента с gap 60 м
    z = np.array([0.0, 100.0, 160.0, 260.0])
    t = np.array([-10.0, -8.0, -8.5, -6.0])
    layers = detect_inversion_layers_gap_v3(
        z, t, max_embedded_gap_m=100.0, min_strength_c=0.3,
    )
    assert len(layers) == 1
    assert layers[0].embedded_gap_count == 1
    assert abs(layers[0].embedded_gap_depth_total_m - 60.0) < 1e-9
    assert abs(layers[0].delta_t_c - 4.0) < 1e-9


def test_gap_above_threshold_keeps_two_layers():
    z = np.array([0.0, 100.0, 220.0, 320.0])
    t = np.array([-10.0, -8.0, -8.5, -6.0])
    # gap = 120 м >= 100 → не merge
    layers = detect_inversion_layers_gap_v3(
        z, t, max_embedded_gap_m=100.0, min_strength_c=0.3,
    )
    assert len(layers) == 2
    assert all(ly.embedded_gap_count == 0 for ly in layers)


def test_weak_layer_filtered():
    z = np.array([0.0, 50.0, 150.0])
    t = np.array([-10.0, -9.9, -11.0])  # ΔT=0.1 < 0.3
    layers = detect_inversion_layers_gap_v3(z, t, min_strength_c=0.3)
    assert layers == []


def test_min_depth_filter():
    z = np.array([0.0, 20.0, 120.0])
    t = np.array([-10.0, -8.0, -12.0])
    layers = detect_inversion_layers_gap_v3(
        z, t, min_strength_c=0.3, min_depth_m=50.0,
    )
    assert layers == []
    layers2 = detect_inversion_layers_gap_v3(
        z, t, min_strength_c=0.3, min_depth_m=None,
    )
    assert len(layers2) == 1


def test_jagged_profile_v3_keeps_layers_where_v2_stops_early():
    """Зубец: v2 останавливается на первом обрыве роста; v3 видит приподнятый слой."""
    levels = [
        {"pressure_hpa": 1000, "temperature_c": -30.0, "height_m": 100},
        {"pressure_hpa": 950, "temperature_c": -28.0, "height_m": 200},
        {"pressure_hpa": 920, "temperature_c": -29.0, "height_m": 280},  # короткий провал
        {"pressure_hpa": 880, "temperature_c": -26.0, "height_m": 360},
        {"pressure_hpa": 850, "temperature_c": -24.0, "height_m": 450},
        {"pressure_hpa": 820, "temperature_c": -27.0, "height_m": 550},
        {"pressure_hpa": 800, "temperature_c": -29.0, "height_m": 650},
    ]
    v2 = detect_surface_inversion(levels, min_inversion_delta_c=0.2)
    # v2: рост до 950, затем падение → верх на 950; confirm может rejected
    assert v2.inversion_candidate is True
    assert v2.inversion_top_pressure_hpa == 950

    z = [lv["height_m"] for lv in levels]
    t = [lv["temperature_c"] for lv in levels]
    p = [lv["pressure_hpa"] for lv in levels]
    layers = detect_inversion_layers_gap_v3(
        z, t, p, max_embedded_gap_m=100.0, min_strength_c=0.3,
    )
    assert len(layers) >= 1
    # gap 80 м между 200 и 280 → merge в один G-слой до 450
    assert layers[0].position_type == "G"
    assert layers[0].top_height_m == 450
    assert layers[0].embedded_gap_count >= 1


def test_summarize_pattern_g_plus_e():
    z = np.array([0.0, 100.0, 250.0, 350.0, 500.0])
    t = np.array([-10.0, -7.0, -9.0, -6.0, -8.0])
    layers = detect_inversion_layers_gap_v3(
        z, t, max_embedded_gap_m=50.0, min_strength_c=0.3, he_threshold_m=250.0,
    )
    summary = summarize_inversion_layers("pid", layers, z0=0.0)
    assert summary["n_inversion_layers"] == 2
    assert summary["has_G"] is True
    assert summary["has_E"] is True
    assert summary["pattern"] == "G+E"


def test_merge_short_gaps_metadata():
    z = np.array([0.0, 50.0, 80.0, 150.0])
    runs = [(0, 1), (2, 3)]
    merged = merge_short_gaps(runs, z, max_gap_depth_m=100.0)
    assert len(merged) == 1
    assert merged[0].embedded_gap_count == 1
    assert abs(merged[0].embedded_gap_depth_total_m - 30.0) < 1e-9
