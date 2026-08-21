"""Каноническая master table для всех графиков статьи."""
from __future__ import annotations

import numpy as np
import pandas as pd

from gdex_bufr.profile_climate.article_figures.config import AnalysisConfig
from gdex_bufr.profile_climate.article_figures.metrics import (
    SEASON_BY_MONTH,
    compute_inversion_layers,
    profile_type_flags,
)

from .metrics import (
    STANDARD_LEVELS_HPA,
    compute_local_gammas,
    compute_sfc_level_gamma,
    valid_layers,
)


def _assert_layer_geometry(layers: pd.DataFrame) -> None:
    if layers.empty:
        return
    depth = pd.to_numeric(layers["depth_m"], errors="coerce")
    base = pd.to_numeric(layers["base_height_agl_m"], errors="coerce")
    top = pd.to_numeric(layers["top_height_agl_m"], errors="coerce")
    assert bool((depth > 0).all()), "depth_m must be > 0"
    assert bool((top > base).all()), "top_height_agl_m must exceed base_height_agl_m"
    recomputed = top - base
    assert bool(np.allclose(depth, recomputed, rtol=0, atol=1e-3)), "depth_m != top - base"


def build_article_graphics_master_table(
    df: pd.DataFrame,
    profile_qc: pd.DataFrame,
    config: AnalysisConfig,
    *,
    layers: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Возвращает (layers_table, profiles_table)."""
    layers_raw = layers if layers is not None else compute_inversion_layers(df, profile_qc, config)
    layers_ok = valid_layers(layers_raw)
    _assert_layer_geometry(layers_ok)

    sfc = compute_sfc_level_gamma(df, profile_qc, config, levels_hpa=STANDARD_LEVELS_HPA)
    local = compute_local_gammas(df, profile_qc, config)
    flags = profile_type_flags(layers_ok, profile_qc)

    eligible = profile_qc[profile_qc["eligible_article"]].copy()
    eligible["profile_id"] = eligible["profile_id"].astype(str)

    layer_cols = [
        "profile_id",
        "datetime_utc",
        "year",
        "month",
        "cycle",
        "position_type",
        "base_height_agl_m",
        "top_height_agl_m",
        "depth_m",
        "delta_t_c",
        "gamma_c_per_100m",
        "mean_gradient_c_100m",
    ]
    layers_table = layers_ok.copy()
    layers_table["season"] = layers_table["month"].map(SEASON_BY_MONTH)
    layers_table["eligible"] = True
    for col in layer_cols:
        if col not in layers_table.columns:
            layers_table[col] = np.nan
    if "gamma_c_per_100m" not in layers_table.columns and "mean_gradient_c_100m" in layers_table.columns:
        layers_table["gamma_c_per_100m"] = layers_table["mean_gradient_c_100m"]
    layers_table = layers_table[[c for c in layer_cols + ["season", "eligible"] if c in layers_table.columns]]

    profiles = eligible.merge(
        sfc,
        on=["profile_id", "year", "month", "cycle"],
        how="left",
        suffixes=("", "_sfc"),
    )
    profiles["season"] = profiles["month"].map(SEASON_BY_MONTH)
    profiles["eligible"] = True
    profiles["n_layers"] = profiles["profile_id"].map(
        layers_ok.groupby("profile_id").size()
    ).fillna(0).astype(int)

    for level in STANDARD_LEVELS_HPA:
        key = int(level)
        profiles[f"T_{key}"] = profiles.get(f"t_{key}_c")
        profiles[f"H_{key}"] = profiles.get(f"h_{key}_m")
        profiles[f"gamma_{key}"] = profiles.get(f"gamma_sfc_{key}")

    flag_cols = [c for c in ["profile_id", "has_G", "has_E", "has_HE", "has_any_v3"] if c in flags.columns]
    profiles = profiles.merge(
        flags[flag_cols],
        on="profile_id",
        how="left",
    )
    if "has_any_v3" not in profiles.columns and all(c in profiles.columns for c in ["has_G", "has_E", "has_HE"]):
        profiles["has_any_v3"] = profiles[["has_G", "has_E", "has_HE"]].any(axis=1)

    # local gamma count per profile
    if not local.empty:
        n_intervals = local.groupby("profile_id").size().rename("n_local_intervals")
        profiles = profiles.merge(n_intervals, on="profile_id", how="left")
    else:
        profiles["n_local_intervals"] = 0

    assert set(profiles["cycle"].dropna().astype(str).str.zfill(2).str[-2:]).issubset({"00", "12", "06", "18", "00", "12"})

    return layers_table.reset_index(drop=True), profiles.reset_index(drop=True)


def export_master_table_csv(
    layers: pd.DataFrame,
    profiles: pd.DataFrame,
    output_dir,
) -> None:
    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    layers.to_csv(out / "article_graphics_master_layers.csv", index=False, encoding="utf-8-sig")
    profiles.to_csv(out / "article_graphics_master_profiles.csv", index=False, encoding="utf-8-sig")
