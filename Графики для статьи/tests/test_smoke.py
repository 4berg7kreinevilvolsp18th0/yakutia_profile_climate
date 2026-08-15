from __future__ import annotations

import pandas as pd

from gdex_bufr.profile_climate.article_figures.config import InversionConfig
from gdex_bufr.profile_climate.article_figures.metrics import detect_surface_inversion_v2


def test_confirmed_inversion():
    levels = pd.DataFrame(
        {
            "pressure_hpa": [1000, 950, 900, 850, 820, 800],
            "temperature_c": [-30, -28, -25, -24, -26, -28],
            "height_m": [0, 400, 800, 1300, 1600, 1800],
        }
    )
    result = detect_surface_inversion_v2(levels, InversionConfig())
    assert result.detected
    assert result.quality == "confirmed"
    assert result.delta_t_c == 6


def test_rejected_inversion():
    levels = pd.DataFrame(
        {
            "pressure_hpa": [1000, 950, 900, 850, 820, 800],
            "temperature_c": [-30, -28, -25, -24, -26, -25],
            "height_m": [0, 400, 800, 1300, 1600, 1800],
        }
    )
    result = detect_surface_inversion_v2(levels, InversionConfig())
    assert not result.detected
    assert result.candidate
    assert result.quality == "rejected_no_lapse"


def test_typed_layers_and_gamma():
    from gdex_bufr.profile_climate.article_figures.config import AnalysisConfig, LayerClassConfig
    from gdex_bufr.profile_climate.article_figures.metrics import (
        _detect_confirmed_layers_arrays,
        frequency_matrix_by_type,
        gamma_count_table,
        height_count_table,
        profile_type_flags,
    )
    import numpy as np

    p = np.array([1000, 950, 900, 850, 820, 800], dtype=float)
    t = np.array([-30, -28, -25, -24, -26, -28], dtype=float)
    h = np.array([0, 400, 800, 1300, 1600, 1800], dtype=float)
    layers = _detect_confirmed_layers_arrays(p, t, h, InversionConfig(), LayerClassConfig())
    assert len(layers) == 1
    assert layers[0]["position_type"] == "G"
    assert layers[0]["gamma_c_per_100m"] > 0

    # elevated: рост начинается не у поверхности
    p2 = np.array([1000, 950, 900, 850, 800, 770, 740, 700], dtype=float)
    t2 = np.array([-20, -22, -24, -22, -20, -22, -24, -26], dtype=float)
    h2 = np.array([0, 400, 800, 1200, 1600, 1900, 2200, 2600], dtype=float)
    layers2 = _detect_confirmed_layers_arrays(p2, t2, h2, InversionConfig(), LayerClassConfig())
    assert any(ly["position_type"] in {"E", "HE"} for ly in layers2)

    df_layers = pd.DataFrame(
        [
            {
                "profile_id": "a",
                "datetime_utc": "2000-01-01",
                "year": 2000,
                "month": 1,
                "cycle": "12",
                "layer_index": 0,
                **layers[0],
            }
        ]
    )
    qc = pd.DataFrame(
        [
            {
                "profile_id": "a",
                "year": 2000,
                "month": 1,
                "cycle": "12",
                "eligible_article": True,
            },
            {
                "profile_id": "b",
                "year": 2000,
                "month": 1,
                "cycle": "12",
                "eligible_article": True,
            },
        ]
    )
    flags = profile_type_flags(df_layers, qc)
    matrix = frequency_matrix_by_type(flags, inversion_type="G")
    assert float(matrix.loc[2000, 1]) == 50.0
    heights = height_count_table(df_layers, bin_edges=AnalysisConfig().layers.height_bin_edges_m, by_month=False)
    assert int(heights["count"].sum()) == 1
    gammas = gamma_count_table(df_layers, bin_edges=AnalysisConfig().layers.gamma_bin_edges_c_per_100m)
    assert int(gammas["days"].sum()) == 1


def test_height_bins_keep_overflow_and_empty_slots():
    from gdex_bufr.profile_climate.article_figures.config import AnalysisConfig
    from gdex_bufr.profile_climate.article_figures.metrics import height_count_table

    layers = pd.DataFrame(
        [
            {"top_height_agl_m": 10.0, "month": 1, "position_type": "G"},
            {"top_height_agl_m": 5000.0, "month": 1, "position_type": "HE"},
        ]
    )
    edges = AnalysisConfig().layers.height_bin_edges_m
    table = height_count_table(layers, bin_edges=edges, by_month=False)
    assert int(table["count"].sum()) == 2
    assert table["count"].iloc[0] == 1
    assert table["count"].iloc[-1] == 1
    assert len(table) == len(edges)


def test_height_primary_rejects_negative_depth():
    from gdex_bufr.profile_climate.article_figures.config import AnalysisConfig
    from gdex_bufr.profile_climate.article_figures.metrics import (
        compute_inversion_layers,
        compute_inversion_layers_pressure_order,
        layer_geometry_qc,
    )
    from gdex_bufr.profile_climate.article_figures.data import build_profile_qc

    rows = []
    for pressure, temp, height in [
        (1000, -20.0, 0.0),
        (950, -18.0, 800.0),
        (900, -16.0, 400.0),  # высота не монотонна относительно P
        (850, -18.0, 1200.0),
        (800, -20.0, 1600.0),
    ]:
        rows.append(
            {
                "profile_id": "p1",
                "station_id": "31004",
                "datetime_utc": "2000-01-01T00:00:00",
                "cycle": "00",
                "pressure_hpa": pressure,
                "temperature_c": temp,
                "height_m": height,
                "year": 2000,
                "month": 1,
            }
        )
    df = pd.DataFrame(rows)
    cfg = AnalysisConfig(strict_surface_qc=False)
    qc = build_profile_qc(df, cfg)
    qc["eligible_article"] = True
    old = compute_inversion_layers_pressure_order(df, qc, cfg)
    new = compute_inversion_layers(df, qc, cfg)
    new_qc = layer_geometry_qc(new)
    assert new_qc["negative_depth"] == 0
    assert new_qc["top_below_base"] == 0
    _ = old


def test_equal_bar_widths():
    from gdex_bufr.profile_climate.article_figures.config import FigureStyle
    from gdex_bufr.profile_climate.article_figures.plots import plot_height_counts_bar

    table = pd.DataFrame(
        {
            "month": [0, 0, 0],
            "bin_left": [0.0, 50.0, 100.0],
            "bin_right": [50.0, 100.0, 250.0],
            "bin_center": [25.0, 75.0, 175.0],
            "count": [3, 5, 8],
        }
    )
    fig = plot_height_counts_bar(table, FigureStyle(show_title=False))
    widths = [p.get_width() for p in fig.axes[0].patches]
    assert widths
    assert max(widths) - min(widths) < 1e-9
    import matplotlib.pyplot as plt
    plt.close(fig)
