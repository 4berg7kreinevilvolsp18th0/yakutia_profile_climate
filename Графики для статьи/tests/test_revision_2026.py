"""Тесты ревизии графиков: геометрия, две γ, знаменатель %, оси."""
from __future__ import annotations

import sys
from pathlib import Path

_ARTICLE = Path(__file__).resolve().parents[1]
_PROJECT = _ARTICLE.parent
for name in list(sys.modules):
    if name == "gdex_bufr" or name.startswith("gdex_bufr."):
        del sys.modules[name]
while str(_PROJECT) in sys.path:
    sys.path.remove(str(_PROJECT))
sys.path.insert(0, str(_ARTICLE))

import numpy as np
import pandas as pd

from gdex_bufr.profile_climate.article_figures.config import AnalysisConfig, FigureStyle
from gdex_bufr.profile_climate.article_figures.data import build_profile_qc
from revision_2026.metrics import (
    compute_local_gammas,
    compute_sfc_level_gamma,
    frequency_percent,
    gamma_local_interval,
    gamma_sfc_to_level,
    interpolate_inside,
    layer_geometry_ok,
    monthly_type_frequency,
    profile_layer_counts,
    shared_abs_limit,
    unique_profile_bin_percent,
    valid_layers,
)
from revision_2026.plots import plot_gamma_sfc_monthly_panels, plot_type01_shared
from revision_2026.style import revision_style


def test_depth_equals_top_minus_base():
    row = {"base_height_agl_m": 10.0, "top_height_agl_m": 40.0, "depth_m": 30.0}
    assert abs(row["depth_m"] - (row["top_height_agl_m"] - row["base_height_agl_m"])) < 1e-9
    assert layer_geometry_ok(row)
    assert row["depth_m"] > 0
    assert row["top_height_agl_m"] > row["base_height_agl_m"]


def test_valid_layers_drops_negative_depth():
    layers = pd.DataFrame(
        [
            {"base_height_agl_m": 0.0, "top_height_agl_m": 80.0, "depth_m": 80.0, "month": 1},
            {"base_height_agl_m": 100.0, "top_height_agl_m": 50.0, "depth_m": -50.0, "month": 1},
        ]
    )
    out = valid_layers(layers)
    assert len(out) == 1
    assert float(out.iloc[0]["depth_m"]) == 80.0


def test_gamma_formulas():
    assert abs(gamma_local_interval(-10.0, -8.0, 0.0, 200.0) - 1.0) < 1e-9
    assert abs(gamma_sfc_to_level(-20.0, -30.0, 680.0, 1680.0) - (-1.0)) < 1e-9
    assert np.isnan(gamma_local_interval(0.0, 1.0, 10.0, 10.0))
    assert np.isnan(gamma_sfc_to_level(0.0, 1.0, 100.0, 50.0))


def test_interpolate_no_extrapolation():
    p = np.array([900.0, 850.0, 800.0])
    t = np.array([-10.0, -12.0, -14.0])
    assert np.isfinite(interpolate_inside(p, t, 850.0))
    assert np.isnan(interpolate_inside(p, t, 500.0))
    assert np.isnan(interpolate_inside(p, t, 950.0))


def test_sfc_gamma_nan_outside_profile():
    rows = []
    for p, t, h in [(930.0, -20.0, 680.0), (900.0, -21.0, 900.0), (850.0, -22.0, 1400.0)]:
        rows.append(
            {
                "profile_id": "p1",
                "station_id": "31004",
                "datetime_utc": "2000-01-01T00:00:00",
                "cycle": "00",
                "pressure_hpa": p,
                "temperature_c": t,
                "height_m": h,
                "year": 2000,
                "month": 1,
            }
        )
    df = pd.DataFrame(rows)
    cfg = AnalysisConfig(strict_surface_qc=False, max_surface_pressure_hpa=1000.0)
    qc = build_profile_qc(df, cfg)
    qc["eligible_article"] = True
    table = compute_sfc_level_gamma(df, qc, cfg)
    assert np.isfinite(table.loc[0, "gamma_sfc_850"])
    assert np.isnan(table.loc[0, "gamma_sfc_500"])
    expected = gamma_sfc_to_level(-20.0, -22.0, 680.0, 1400.0)
    assert abs(table.loc[0, "gamma_sfc_850"] - expected) < 1e-6


def test_local_gamma_uses_all_intervals_including_negative():
    rows = []
    for p, t, h in [
        (930.0, 0.0, 0.0),
        (900.0, 2.0, 100.0),
        (850.0, 0.0, 200.0),
        (800.0, -2.0, 300.0),
    ]:
        rows.append(
            {
                "profile_id": "p1",
                "station_id": "31004",
                "datetime_utc": "2000-01-01T00:00:00",
                "cycle": "00",
                "pressure_hpa": p,
                "temperature_c": t,
                "height_m": h,
                "year": 2000,
                "month": 1,
            }
        )
    df = pd.DataFrame(rows)
    cfg = AnalysisConfig(strict_surface_qc=False, max_surface_pressure_hpa=1000.0)
    qc = build_profile_qc(df, cfg)
    qc["eligible_article"] = True
    local = compute_local_gammas(df, qc, cfg)
    assert (local["gamma_local_c_100m"] > 0).any()
    assert (local["gamma_local_c_100m"] < 0).any()
    assert local["month"].isin(range(1, 13)).all()


def test_frequency_denominator_is_eligible_profiles_not_layers():
    flags = pd.DataFrame(
        {
            "profile_id": ["a", "b", "c", "d"],
            "month": [1, 1, 1, 1],
            "has_G": [True, True, False, False],
            "has_E": [False, False, False, False],
            "has_HE": [False, False, False, False],
        }
    )
    monthly = monthly_type_frequency(flags)
    jan = monthly.loc[monthly["month"] == 1].iloc[0]
    assert jan["n_eligible"] == 4
    assert jan["n_G"] == 2
    assert abs(jan["F_G"] - 50.0) < 1e-9
    layers = pd.DataFrame(
        {
            "profile_id": ["a", "a", "b"],
            "month": [1, 1, 1],
            "base_height_agl_m": [5.0, 8.0, 10.0],
            "position_type": ["G", "G", "G"],
        }
    )
    qc = pd.DataFrame(
        {
            "profile_id": ["a", "b", "c", "d"],
            "month": [1, 1, 1, 1],
            "eligible_article": [True, True, True, True],
        }
    )
    bins = unique_profile_bin_percent(layers, qc, value_col="base_height_agl_m", bin_edges=(0, 50, 100))
    first = bins[(bins["month"] == 1) & (bins["bin_left"] == 0)].iloc[0]
    assert first["n_layers"] == 3
    assert first["n_profiles"] == 2
    assert first["n_eligible"] == 4
    assert abs(first["frequency_percent"] - 50.0) < 1e-9


def test_month_ordering_1_to_12():
    flags = pd.DataFrame(
        {
            "profile_id": [f"p{i}" for i in range(12)],
            "month": list(range(1, 13)),
            "has_G": [False] * 12,
            "has_E": [False] * 12,
            "has_HE": [False] * 12,
        }
    )
    monthly = monthly_type_frequency(flags)
    assert list(monthly["month"]) == list(range(1, 13))


def test_shared_axis_and_color_limits():
    lim = shared_abs_limit(np.array([-2.0, 1.0]), np.array([1.5, -1.8]), q=100)
    assert abs(lim - 2.0) < 1e-9
    style = revision_style(FigureStyle(dpi=80, output_formats=("png",)))
    ym = pd.DataFrame(
        {
            "year": [2000, 2001] * 12,
            "month": list(range(1, 13)) + list(range(1, 13)),
            **{f"median_{k}": np.linspace(-1, 1, 24) for k in (850, 700, 500)},
            **{f"q25_{k}": np.linspace(-1.2, 0.8, 24) for k in (850, 700, 500)},
            **{f"q75_{k}": np.linspace(-0.8, 1.2, 24) for k in (850, 700, 500)},
        }
    )
    fig = plot_gamma_sfc_monthly_panels(ym, style)
    ylims = [ax.get_ylim() for ax in fig.axes if ax.get_ylabel() or True]
    y0 = {round(a.get_ylim()[0], 5) for a in fig.axes if len(a.get_lines())}
    y1 = {round(a.get_ylim()[1], 5) for a in fig.axes if len(a.get_lines())}
    assert len(y0) == 1
    assert len(y1) == 1
    import matplotlib.pyplot as plt

    plt.close(fig)

    years = list(range(2000, 2004))
    mats = {}
    for kind, scale in (("G", 80.0), ("E", 20.0), ("HE", 5.0)):
        mats[kind] = pd.DataFrame(
            np.full((len(years), 12), scale / 2),
            index=years,
            columns=range(1, 13),
        )
        mats[kind].iloc[0, 0] = scale
    fig2 = plot_type01_shared(mats, style)
    images = [im for ax in fig2.axes for im in ax.get_images()]
    vmins = {im.norm.vmin for im in images}
    vmaxs = {im.norm.vmax for im in images}
    assert vmins == {0}
    assert len(vmaxs) == 1
    assert list(vmaxs)[0] == 80.0
    plt.close(fig2)


def test_profile_layer_counts_include_zero():
    layers = pd.DataFrame({"profile_id": ["a", "a"]})
    qc = pd.DataFrame(
        {
            "profile_id": ["a", "b"],
            "year": [2000, 2000],
            "month": [1, 1],
            "cycle": ["00", "00"],
            "eligible_article": [True, True],
        }
    )
    counts = profile_layer_counts(layers, qc)
    assert set(counts["n_inversion_layers"]) == {0, 2}
    cond = counts["n_inversion_layers"] >= 2
    freq = frequency_percent(counts, condition=cond, group_cols=["month"])
    assert abs(freq.loc[0, "frequency_percent"] - 50.0) < 1e-9
