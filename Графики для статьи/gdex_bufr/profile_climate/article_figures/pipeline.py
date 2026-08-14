from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import AnalysisConfig, FigureStyle
from .data import build_profile_qc, compute_completeness, load_profiles
from .metrics import (
    INVERSION_TYPES,
    annual_inversion_frequency,
    compute_inversion_layers,
    compute_inversion_metrics,
    compute_seasonal_climatology,
    frequency_matrix_by_type,
    gamma_count_table,
    height_count_table,
    monthly_inversion_frequency,
    pressure_level_annual_series,
    profile_type_flags,
    interpolate_eligible_profiles,
)
from .plots import (
    plot_annual_inversion_variability,
    plot_completeness_heatmap,
    plot_gamma_by_month_box,
    plot_gamma_counts_bar,
    plot_gamma_counts_hist_step,
    plot_gamma_counts_line,
    plot_height_counts_bar,
    plot_height_counts_by_month_facets,
    plot_height_counts_line,
    plot_height_counts_months_overlay,
    plot_inversion_type_frequency_matrix,
    plot_monthly_inversion_frequency,
    plot_monthly_inversion_intensity,
    plot_monthly_inversion_top_height,
    plot_pressure_level_time_series,
    plot_profile_qc_summary,
    plot_seasonal_temperature_profiles,
)


def save_figure(fig, output_base: Path, style: FigureStyle) -> list[Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in style.output_formats:
        path = output_base.with_suffix(f".{fmt}")
        kwargs = {"bbox_inches": "tight"}
        if fmt.lower() == "png":
            kwargs["dpi"] = style.dpi
        fig.savefig(path, **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def _write_table_csv(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(path, index=False, encoding="utf-8-sig")


def build_all(
    input_csv: str | Path,
    output_dir: str | Path,
    analysis: AnalysisConfig,
    style: FigureStyle,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    df = load_profiles(input_csv, station_id=analysis.station_id, cycles=analysis.cycles)
    qc = build_profile_qc(df, analysis)
    completeness_long, completeness_matrix = compute_completeness(df, cycles=analysis.cycles)
    inversion = compute_inversion_metrics(df, qc, analysis)
    interpolated = interpolate_eligible_profiles(df, qc, analysis.pressure_grid_hpa, cycles=analysis.cycles)
    seasonal = compute_seasonal_climatology(df, qc, analysis, interpolated=interpolated)
    monthly = monthly_inversion_frequency(inversion)
    annual, trend_stats = annual_inversion_frequency(
        inversion,
        start_year=analysis.trend_start_year,
        end_year=analysis.trend_end_year,
        moving_window=analysis.moving_average_window,
    )
    pressure_series = pressure_level_annual_series(df, qc, analysis, interpolated=interpolated)

    layers = compute_inversion_layers(df, qc, analysis)
    type_flags = profile_type_flags(layers, qc)
    height_all = height_count_table(layers, bin_edges=analysis.layers.height_bin_edges_m, by_month=False)
    height_monthly = height_count_table(layers, bin_edges=analysis.layers.height_bin_edges_m, by_month=True)
    gamma_all = gamma_count_table(layers, bin_edges=analysis.layers.gamma_bin_edges_c_per_100m, by_month=False)

    type_matrices = {kind: frequency_matrix_by_type(type_flags, inversion_type=kind) for kind in INVERSION_TYPES}

    tables = {
        "profile_qc": qc,
        "completeness_monthly": completeness_long,
        "inversion_metrics_v2": inversion,
        "inversion_layers_typed": layers,
        "inversion_type_flags": type_flags,
        "seasonal_climatology": seasonal,
        "monthly_inversion_frequency": monthly,
        "annual_inversion_frequency": annual,
        "pressure_level_annual_series": pressure_series,
        "height_counts_all": height_all,
        "height_counts_monthly": height_monthly,
        "gamma_counts_all": gamma_all,
    }
    for kind, matrix in type_matrices.items():
        tables[f"frequency_matrix_{kind}"] = matrix.reset_index()

    for name, table in tables.items():
        _write_table_csv(table, tables_dir / f"{name}.csv")

    figure_builders: list[tuple[str, object]] = [
        ("fig01_completeness_heatmap", lambda: plot_completeness_heatmap(completeness_matrix, style)),
        ("fig02_seasonal_temperature_profiles", lambda: plot_seasonal_temperature_profiles(seasonal, style)),
        ("fig03_monthly_inversion_frequency", lambda: plot_monthly_inversion_frequency(monthly, style)),
        ("fig04_annual_inversion_variability", lambda: plot_annual_inversion_variability(annual, trend_stats, style)),
        ("extra01_monthly_inversion_intensity", lambda: plot_monthly_inversion_intensity(inversion, style)),
        ("extra02_monthly_inversion_top_height", lambda: plot_monthly_inversion_top_height(inversion, style)),
        ("extra03_profile_qc_summary", lambda: plot_profile_qc_summary(qc, style)),
        ("extra04_pressure_level_time_series", lambda: plot_pressure_level_time_series(pressure_series, style)),
        ("type01_matrix_G", lambda: plot_inversion_type_frequency_matrix(type_matrices["G"], style, inversion_type="G")),
        ("type01_matrix_E", lambda: plot_inversion_type_frequency_matrix(type_matrices["E"], style, inversion_type="E")),
        ("type01_matrix_HE", lambda: plot_inversion_type_frequency_matrix(type_matrices["HE"], style, inversion_type="HE")),
        ("type02_height_bar", lambda: plot_height_counts_bar(height_all, style)),
        ("type02_height_line", lambda: plot_height_counts_line(height_all, style)),
        ("type02_height_months_facets", lambda: plot_height_counts_by_month_facets(height_monthly, style)),
        ("type02_height_months_overlay", lambda: plot_height_counts_months_overlay(height_monthly, style)),
        ("type03_gamma_bar", lambda: plot_gamma_counts_bar(gamma_all, style)),
        ("type03_gamma_line", lambda: plot_gamma_counts_line(gamma_all, style)),
        (
            "type03_gamma_hist",
            lambda: plot_gamma_counts_hist_step(
                layers, style, bin_edges=analysis.layers.gamma_bin_edges_c_per_100m
            ),
        ),
        ("type03_gamma_monthly_box", lambda: plot_gamma_by_month_box(layers, style)),
    ]
    for month in range(1, 13):
        figure_builders.append(
            (
                f"type02_height_bar_m{month:02d}",
                lambda m=month: plot_height_counts_bar(height_monthly, style, month=m),
            )
        )

    saved: dict[str, list[str]] = {}
    for name, builder in figure_builders:
        fig = builder()
        saved[name] = [str(p) for p in save_figure(fig, figures_dir / name, style)]

    summary = {
        "input": str(input_csv),
        "rows": int(len(df)),
        "profiles": int(qc.shape[0]),
        "eligible_article_profiles": int(qc["eligible_article"].sum()),
        "confirmed_inversions": int(inversion.loc[inversion["eligible_article"], "inversion_detected"].sum()),
        "confirmed_layers": int(len(layers)),
        "layers_by_type": {
            kind: int((layers["position_type"] == kind).sum()) if not layers.empty else 0
            for kind in INVERSION_TYPES
        },
        "trend": trend_stats,
        "analysis_config": asdict(analysis),
        "figure_style": asdict(style),
        "figures": saved,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary
