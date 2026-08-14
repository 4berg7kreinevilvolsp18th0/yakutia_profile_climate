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
    assert heights["count"].sum() == 1
    gammas = gamma_count_table(df_layers, bin_edges=AnalysisConfig().layers.gamma_bin_edges_c_per_100m)
    assert gammas["days"].sum() == 1
