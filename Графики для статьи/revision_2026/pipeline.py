"""Сборка ревизии в отдельные папки. Старый pipeline не вызывается для перезаписи sample_output."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gdex_bufr.profile_climate.article_figures.config import AnalysisConfig, FigureStyle
from gdex_bufr.profile_climate.article_figures.data import build_profile_qc, compute_completeness, load_profiles
from gdex_bufr.profile_climate.article_figures.metrics import (
    INVERSION_TYPES,
    annual_inversion_frequency,
    compute_inversion_layers,
    compute_seasonal_climatology,
    frequency_matrix_by_type,
    monthly_inversion_frequency,
    compute_inversion_metrics,
    interpolate_eligible_profiles,
    layer_geometry_qc,
)
from gdex_bufr.profile_climate.article_figures.plots import (
    plot_annual_inversion_variability,
    plot_completeness_heatmap,
    plot_monthly_inversion_frequency,
    plot_seasonal_temperature_profiles,
)
from gdex_bufr.profile_climate.article_figures.pipeline import save_figure

from .metrics import (
    prepare_revision_tables,
    shared_abs_limit,
    year_month_frequency,
)
from .plots import (
    plot_bubble_year_month,
    plot_counts_year_month,
    plot_density_by_type,
    plot_depth_boxplots,
    plot_depth_delta_density,
    plot_ecdf,
    plot_gamma_sfc_annual_cycle,
    plot_gamma_sfc_monthly_panels,
    plot_gehe_summary,
    plot_hexbin,
    plot_joint_depth_vs_base,
    plot_local_gamma_00_12,
    plot_local_gamma_box_month,
    plot_local_gamma_hist,
    plot_local_gamma_hist_seasons,
    plot_mean_layers_month,
    plot_month_height_heatmap,
    plot_month_type_heatmap,
    plot_monthly_depth_vs_base_12,
    plot_multilayer_00_12,
    plot_multilayer_hist,
    plot_multilayer_monthly_stack,
    plot_multilayer_season_stack,
    plot_qc_base_top,
    plot_ridgeline,
    plot_scatter_typed,
    plot_seasonal_gamma_z,
    plot_seasonal_phase,
    plot_simple_hist,
    plot_type01_shared,
    plot_violin,
    plot_year_month_heatmap,
)
from .style import TYPE_COLORS, TYPE_LABELS, MONTHS_RU, add_caption, revision_style, station_caption


def _dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "updated": output_dir / "figures" / "updated",
        "thickness": output_dir / "figures" / "article" / "01_thickness",
        "gamma_sfc": output_dir / "figures" / "article" / "02_gamma_sfc_P",
        "gamma_local": output_dir / "figures" / "article" / "03_gamma_local",
        "multilayer": output_dir / "figures" / "article" / "04_multilayer",
        "gehe": output_dir / "figures" / "article" / "05_GEHE",
        "extra": output_dir / "figures" / "article" / "06_extra",
        "diagnostic": output_dir / "figures" / "diagnostic",
        "tables": output_dir / "tables" / "article_figures",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _save(fig, folder: Path, name: str, style: FigureStyle, saved: dict[str, list[str]]) -> None:
    saved[name] = [str(p) for p in save_figure(fig, folder / name, style)]


def _csv(table: pd.DataFrame, tables_dir: Path, name: str) -> None:
    table.to_csv(tables_dir / f"{name}.csv", index=False, encoding="utf-8-sig")


def build_revision(
    input_csv: str | Path,
    output_dir: str | Path,
    analysis: AnalysisConfig,
    style: FigureStyle,
) -> dict:
    output_dir = Path(output_dir)
    style = revision_style(style)
    dirs = _dirs(output_dir)
    saved: dict[str, list[str]] = {}

    df = load_profiles(input_csv, station_id=analysis.station_id, cycles=analysis.cycles)
    qc = build_profile_qc(df, analysis)
    inversion = compute_inversion_metrics(df, qc, analysis)
    interpolated = interpolate_eligible_profiles(df, qc, analysis.pressure_grid_hpa, cycles=analysis.cycles)
    seasonal = compute_seasonal_climatology(df, qc, analysis, interpolated=interpolated)
    monthly_v2 = monthly_inversion_frequency(inversion)
    annual_v2, trend = annual_inversion_frequency(
        inversion,
        start_year=analysis.trend_start_year,
        end_year=analysis.trend_end_year,
        moving_window=analysis.moving_average_window,
    )
    completeness_long, completeness_matrix = compute_completeness(df, cycles=analysis.cycles)
    layers_v3 = compute_inversion_layers(df, qc, analysis)
    tables = prepare_revision_tables(df, qc, analysis, layers=layers_v3)
    layers = tables["layers"]
    flags = tables["flags"]
    local = tables["local_gamma"]
    sfc = tables["sfc_gamma"]
    counts = tables["layer_counts"]
    year_from = int(qc["year"].min())
    year_to = int(qc["year"].max())
    cap = station_caption(analysis.station_name, analysis.station_id, year_from, year_to)

    for name, table in {
        "profile_qc": qc,
        "inversion_layers_valid": layers,
        "profile_type_flags": flags,
        "gamma_sfc_P": sfc,
        "gamma_sfc_monthly": tables["sfc_gamma_monthly"],
        "gamma_sfc_year_month": tables["sfc_gamma_year_month"],
        "gamma_local_intervals": local,
        "gamma_local_extreme_abs_ge_15": tables["local_gamma_extreme_ge_15"],
        "gamma_sfc_extreme_abs_ge_15": tables["sfc_gamma_extreme_ge_15"],
        "local_median_heatmap": tables["local_median_heatmap"],
        "local_ppos_heatmap": tables["local_ppos_heatmap"],
        "local_pneg_heatmap": tables["local_pneg_heatmap"],
        "profile_layer_counts": counts,
        "monthly_type_frequency": tables["monthly_type_frequency"],
        "type_summary": tables["type_summary"],
        "bin_base_profile_percent": tables["bin_base_profile_percent"],
        "bin_depth_profile_percent": tables["bin_depth_profile_percent"],
        "bin_top_profile_percent": tables["bin_top_profile_percent"],
        "monthly_inversion_frequency_v2": monthly_v2,
        "annual_inversion_frequency_v2": annual_v2,
        "completeness_monthly": completeness_long,
        "gamma_counts_n_intervals": local.groupby(
            pd.cut(local["gamma_local_c_100m"], bins=analysis.layers.gamma_bin_edges_c_per_100m, right=False),
            observed=False,
        ).size().rename("n_intervals").reset_index(),
    }.items():
        _csv(table, dirs["tables"], name)

    # --- updated restyle of core article figures ---
    _save(plot_completeness_heatmap(completeness_matrix, style), dirs["updated"], "fig01_completeness_heatmap", style, saved)
    _save(plot_seasonal_temperature_profiles(seasonal, style), dirs["updated"], "fig02_seasonal_temperature_profiles", style, saved)
    fig = plot_monthly_inversion_frequency(monthly_v2, style)
    add_caption(fig, cap + ". Приземная инверсия v2 (не G/E/HE).")
    _save(fig, dirs["updated"], "fig03_monthly_inversion_frequency_v2", style, saved)
    fig = plot_annual_inversion_variability(annual_v2, trend, style)
    add_caption(fig, cap + ". Приземная инверсия v2 (не G/E/HE).")
    _save(fig, dirs["updated"], "fig04_annual_inversion_variability_v2", style, saved)
    type_mats = {k: frequency_matrix_by_type(flags, inversion_type=k) for k in INVERSION_TYPES}
    _save(plot_type01_shared(type_mats, style, caption=cap + ". Повторяемость профилей с ≥1 слоем v3; общая шкала."), dirs["updated"], "type01_matrix_G_E_HE_shared_scale", style, saved)
    for kind, mat in type_mats.items():
        _csv(mat.reset_index(), dirs["tables"], f"frequency_matrix_{kind}")

    # --- 01 thickness ---
    _save(plot_joint_depth_vs_base(layers, style, caption=cap), dirs["thickness"], "inversion_depth_vs_base_joint", style, saved)
    fig12, xlim, ylim, vmax = plot_monthly_depth_vs_base_12(layers, style, caption=cap)
    _save(fig12, dirs["thickness"], "inversion_depth_vs_base_monthly_12panel", style, saved)
    for month in range(1, 13):
        g = layers[layers["month"] == month]
        _save(
            plot_joint_depth_vs_base(g, style, caption=f"{cap}. {MONTHS_RU[month - 1]}", xlim=xlim, ylim=ylim, vmax=vmax),
            dirs["thickness"],
            f"inversion_depth_vs_base_month_{month:02d}",
            style,
            saved,
        )
    _save(plot_depth_boxplots(layers, style, caption=cap), dirs["thickness"], "inversion_depth_boxplots", style, saved)
    _csv(layers[["profile_id", "year", "month", "cycle", "position_type", "base_height_agl_m", "top_height_agl_m", "depth_m", "delta_t_c", "gamma_c_per_100m"]], dirs["tables"], "joint_depth_vs_base")

    # --- 02 gamma sfc-P ---
    _save(plot_gamma_sfc_annual_cycle(tables["sfc_gamma_monthly"], style, caption=cap + ". γ_sfc-P = 100(T_P−T_sfc)/(H_P−H_sfc)."), dirs["gamma_sfc"], "type03_gamma_annual_cycle_850_700_500", style, saved)
    _save(plot_gamma_sfc_monthly_panels(tables["sfc_gamma_year_month"], style, caption=cap + ". Панели: год; три линии 850/700/500 гПа."), dirs["gamma_sfc"], "type03_gamma_monthly_850_700_500", style, saved)

    # --- 03 local gamma ---
    med_tab = tables["local_median_heatmap"]
    ppos = tables["local_ppos_heatmap"]
    pneg = tables["local_pneg_heatmap"]
    lim = shared_abs_limit(med_tab["value"].to_numpy(float))
    _save(plot_local_gamma_hist(local, style, caption=cap + ". γ_local = 100(T_{i+1}−T_i)/(z_{i+1}−z_i), все интервалы."), dirs["gamma_local"], "gamma_local_histogram", style, saved)
    _save(plot_local_gamma_hist_seasons(local, style, caption=cap), dirs["gamma_local"], "gamma_local_histogram_seasons", style, saved)
    _save(plot_month_height_heatmap(med_tab, style, label="медиана γ_local, °C/100 м", cmap="RdBu_r", vmin=-lim, vmax=lim, center0=True, caption=cap), dirs["gamma_local"], "gamma_local_month_height_median", style, saved)
    _save(plot_month_height_heatmap(ppos, style, label="P(γ_local > 0), %", cmap="YlOrRd", vmin=0, vmax=100, center0=False, caption=cap), dirs["gamma_local"], "gamma_local_month_height_p_positive", style, saved)
    _save(plot_month_height_heatmap(pneg, style, label="P(γ_local < 0), %", cmap="YlGnBu", vmin=0, vmax=100, center0=False, caption=cap), dirs["gamma_local"], "gamma_local_month_height_p_negative", style, saved)
    _save(plot_local_gamma_00_12(local, style, caption=cap), dirs["gamma_local"], "gamma_local_00_vs_12", style, saved)
    _save(plot_local_gamma_box_month(local, style, caption=cap), dirs["gamma_local"], "gamma_local_monthly_box", style, saved)
    _save(plot_seasonal_gamma_z(local, style, caption=cap), dirs["gamma_local"], "gamma_local_seasonal_profiles", style, saved)

    # --- density depth-dT ---
    _save(plot_depth_delta_density(layers, style, caption=cap), dirs["thickness"], "hexbin_depth_vs_delta_t", style, saved)

    # --- 04 multilayer ---
    _save(plot_multilayer_hist(counts, style, caption=cap), dirs["multilayer"], "n_layers_histogram", style, saved)
    _save(plot_multilayer_monthly_stack(counts, style, caption=cap), dirs["multilayer"], "n_layers_monthly_percent", style, saved)
    _save(plot_multilayer_season_stack(counts, style, caption=cap), dirs["multilayer"], "n_layers_seasonal_stack", style, saved)
    multi_mat = (
        counts.groupby(["year", "month"])["multilayer"].mean().mul(100.0).reset_index()
        .pivot(index="year", columns="month", values="multilayer").reindex(columns=range(1, 13)).sort_index()
    )
    _save(plot_year_month_heatmap(multi_mat, style, label="P(n_layers ≥ 2), %", vmin=0, vmax=float(np.nanmax(multi_mat.to_numpy()) or 1), caption=cap), dirs["multilayer"], "heatmap_p_multilayer", style, saved)
    _save(plot_multilayer_00_12(counts, style, caption=cap), dirs["multilayer"], "p_multilayer_00_12", style, saved)
    _save(plot_mean_layers_month(counts, style, caption=cap), dirs["multilayer"], "mean_layers_per_profile_month", style, saved)
    _csv(multi_mat.reset_index(), dirs["tables"], "heatmap_p_multilayer")

    # --- 05 G/E/HE ---
    _save(plot_gehe_summary(layers, tables["monthly_type_frequency"], style, caption=cap), dirs["gehe"], "GEHE_summary_three_columns", style, saved)
    any_mat = year_month_frequency(flags, "any")
    _save(plot_year_month_heatmap(any_mat, style, label="P(есть слой v3), %", vmin=0, vmax=float(np.nanmax(any_mat.to_numpy()) or 1), caption=cap), dirs["gehe"], "heatmap_any_inversion_year_month", style, saved)
    _csv(any_mat.reset_index(), dirs["tables"], "heatmap_any_inversion")

    # --- 06 extra (≥10) ---
    _save(plot_hexbin(layers["base_height_agl_m"], layers["depth_m"], style, "Основание AGL, м", "Толщина, м", cap), dirs["extra"], "hexbin_base_vs_depth", style, saved)
    _save(plot_hexbin(layers["gamma_c_per_100m"], layers["depth_m"], style, "γ слоя, °C/100 м", "Толщина, м", cap), dirs["extra"], "hexbin_layer_gamma_vs_depth", style, saved)
    _save(plot_scatter_typed(layers, style, "base_height_agl_m", "delta_t_c", "Основание AGL, м", "ΔT, °C", cap), dirs["extra"], "scatter_base_vs_delta_t_by_type", style, saved)
    _save(plot_violin(
        [layers.loc[layers["position_type"] == k, "delta_t_c"] for k in INVERSION_TYPES],
        [TYPE_LABELS[k] for k in INVERSION_TYPES], "ΔT, °C", style, cap, [TYPE_COLORS[k] for k in INVERSION_TYPES],
    ), dirs["extra"], "violin_delta_t_by_type", style, saved)
    _save(plot_violin(
        [layers.loc[layers["month"] == m, "depth_m"] for m in range(1, 13)],
        MONTHS_RU, "depth_m, м", style, cap,
    ), dirs["extra"], "violin_depth_by_month", style, saved)
    _save(plot_ridgeline([layers.loc[layers["month"] == m, "depth_m"].to_numpy() for m in range(1, 13)], MONTHS_RU, "depth_m, м", style, cap), dirs["extra"], "ridgeline_depth_by_month", style, saved)
    _save(plot_ridgeline([layers.loc[layers["month"] == m, "delta_t_c"].to_numpy() for m in range(1, 13)], MONTHS_RU, "ΔT, °C", style, cap), dirs["extra"], "ridgeline_delta_t_by_month", style, saved)
    _save(plot_month_type_heatmap(tables["monthly_type_frequency"], style, cap), dirs["extra"], "heatmap_month_type_frequency", style, saved)
    _save(plot_bubble_year_month(flags, layers, style, cap), dirs["extra"], "bubble_year_month_freq_delta_t", style, saved)
    _save(plot_seasonal_phase(layers, counts, style, cap), dirs["extra"], "seasonal_phase_depth_delta_t", style, saved)
    _save(plot_density_by_type(layers, style, cap), dirs["extra"], "density_depth_delta_t_by_type", style, saved)
    _save(plot_ecdf(layers, "depth_m", "depth_m, м", style, cap), dirs["extra"], "ecdf_depth_by_type", style, saved)
    _save(plot_ecdf(layers, "delta_t_c", "ΔT, °C", style, cap), dirs["extra"], "ecdf_delta_t_by_type", style, saved)
    base_m = tables["bin_base_profile_percent"]
    depth_m = tables["bin_depth_profile_percent"]
    _save(plot_month_height_heatmap(base_m.rename(columns={"frequency_percent": "value"}), style, label="% профилей с слоем в бине основания", cmap="YlOrRd", vmin=0, vmax=float(np.nanmax(base_m["frequency_percent"]) or 1), center0=False, caption=cap), dirs["extra"], "heatmap_month_base_bin_frequency", style, saved)
    _save(plot_month_height_heatmap(depth_m.rename(columns={"frequency_percent": "value"}), style, label="% профилей с слоем в бине толщины", cmap="YlOrRd", vmin=0, vmax=float(np.nanmax(depth_m["frequency_percent"]) or 1), center0=False, caption=cap), dirs["extra"], "heatmap_month_depth_bin_frequency", style, saved)

    # --- diagnostic ---
    qc_geo = layer_geometry_qc(layers)
    _save(plot_qc_base_top(layers, style, cap), dirs["diagnostic"], "qc_base_vs_top_yeqx", style, saved)
    _save(plot_simple_hist(layers["depth_m"], "depth_m, м", style, cap, logy=True), dirs["diagnostic"], "qc_depth_histogram", style, saved)
    _save(plot_simple_hist(local["dz_m"], "Δz соседних уровней, м", style, cap, logy=True), dirs["diagnostic"], "qc_dz_histogram", style, saved)
    if "embedded_gap_depth_total_m" in layers.columns:
        _save(plot_simple_hist(layers["embedded_gap_depth_total_m"], "суммарный embedded gap, м", style, cap), dirs["diagnostic"], "qc_embedded_gap_depth", style, saved)
        frac = layers["embedded_gap_depth_total_m"] / layers["depth_m"].replace(0, np.nan)
        _save(plot_simple_hist(frac, "embedded_gap_fraction", style, cap), dirs["diagnostic"], "qc_embedded_gap_fraction", style, saved)
        _csv(pd.DataFrame({"embedded_gap_fraction": frac}), dirs["tables"], "qc_embedded_gap_fraction")
    if "embedded_gap_count" in layers.columns:
        _save(plot_hexbin(layers["embedded_gap_count"] + 1, layers["delta_t_c"], style, "число сегментов (gap_count+1)", "ΔT, °C", cap), dirs["diagnostic"], "qc_delta_t_vs_source_segments", style, saved)
    _save(plot_counts_year_month(qc, style, cap), dirs["diagnostic"], "qc_eligible_counts_year_month", style, saved)
    _csv(pd.DataFrame([qc_geo]), dirs["tables"], "qc_geometry_counts")

    summary = {
        "input": str(input_csv),
        "eligible_profiles": int(qc["eligible_article"].sum()),
        "valid_layers": int(len(layers)),
        "geometry_qc": qc_geo,
        "year_from": year_from,
        "year_to": year_to,
        "figures": saved,
        "note": "γ_sfc-P и γ_local считаются раздельно. Старый sample_output не перезаписывался.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary
