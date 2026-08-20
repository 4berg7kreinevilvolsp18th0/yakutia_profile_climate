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
    annual_median_table,
    compute_inversion_layers,
    compute_inversion_layers_pressure_order,
    compute_inversion_metrics,
    compute_seasonal_climatology,
    frequency_matrix_by_type,
    compute_interval_gammas,
    compute_reference_level_gammas,
    gamma_count_table,
    height_count_table,
    layer_geometry_qc,
    monthly_inversion_frequency,
    monthly_median_iqr_table,
    pressure_level_annual_series,
    profile_type_flags,
    interpolate_eligible_profiles,
    reference_pressure_heights_agl,
    recurrence_percent_table,
    year_month_median_matrix,
)
from .plots import (
    plot_annual_inversion_variability,
    plot_annual_median_top,
    plot_base_vs_top_scatter,
    plot_completeness_heatmap,
    plot_gamma_by_month_box,
    plot_gamma_counts_bar,
    plot_gamma_counts_hist_step,
    plot_gamma_counts_line,
    plot_gamma_line_monthly_facets,
    plot_gamma_reference_line_monthly_facets,
    plot_gamma_scatter_hist,
    GAMMA_YEAR_START,
    GAMMA_YEAR_COUNT,
    plot_height_counts_bar,
    plot_height_counts_by_month_facets,
    plot_height_counts_line,
    plot_height_counts_months_overlay,
    plot_height_median_heatmap,
    plot_inversion_type_frequency_matrix,
    plot_monthly_inversion_frequency,
    plot_monthly_inversion_intensity,
    plot_monthly_inversion_top_height,
    plot_monthly_median_iqr,
    plot_pressure_level_time_series,
    plot_profile_qc_summary,
    plot_qc_old_vs_new,
    plot_recurrence_by_type_bars,
    plot_seasonal_quantiles,
    plot_seasonal_temperature_profiles,
    plot_top_height_cdf_by_cycle,
    plot_top_height_vs_depth_boxplots,
    plot_top_height_vs_depth_joint,
    plot_top_height_vs_depth_monthly_facets,
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
    layers_old = compute_inversion_layers_pressure_order(df, qc, analysis)
    type_flags = profile_type_flags(layers, qc)
    height_all = height_count_table(layers, bin_edges=analysis.layers.height_bin_edges_m, by_month=False)
    height_monthly = height_count_table(layers, bin_edges=analysis.layers.height_bin_edges_m, by_month=True)
    height_by_type = height_count_table(
        layers, bin_edges=analysis.layers.height_bin_edges_m, by_month=False, by_type=True,
    )
    height_monthly_by_type = height_count_table(
        layers, bin_edges=analysis.layers.height_bin_edges_m, by_month=True, by_type=True,
    )
    interval_gammas = compute_interval_gammas(df, qc, analysis)
    reference_gammas = compute_reference_level_gammas(df, qc, analysis)
    gamma_year_start = GAMMA_YEAR_START
    gamma_year_end = gamma_year_start + GAMMA_YEAR_COUNT - 1
    year_label = f"{gamma_year_start}–{gamma_year_end}"
    interval_gammas_period = interval_gammas[
        interval_gammas["year"].between(gamma_year_start, gamma_year_end)
    ].copy()
    reference_gammas_850_750_500 = compute_reference_level_gammas(
        df,
        qc,
        analysis,
        reference_levels_hpa=(850.0, 750.0, 500.0),
        year_start=gamma_year_start,
        year_end=gamma_year_end,
    )
    gamma_all = gamma_count_table(interval_gammas, bin_edges=analysis.layers.gamma_bin_edges_c_per_100m, by_month=False)
    gamma_monthly_all = gamma_count_table(
        interval_gammas_period,
        bin_edges=analysis.layers.gamma_bin_edges_c_per_100m,
        by_month=True,
    )
    gamma_monthly_ref = gamma_count_table(
        reference_gammas_850_750_500,
        bin_edges=analysis.layers.gamma_bin_edges_c_per_100m,
        by_month=True,
        by_pressure=True,
    )
    top_recurrence = recurrence_percent_table(
        layers, qc, bin_edges=analysis.layers.height_bin_edges_m, value_col="top_height_agl_m", by_month=False,
    )
    top_recurrence_month = recurrence_percent_table(
        layers, qc, bin_edges=analysis.layers.height_bin_edges_m, value_col="top_height_agl_m", by_month=True,
    )
    base_recurrence = recurrence_percent_table(
        layers, qc, bin_edges=analysis.layers.height_bin_edges_m, value_col="base_height_agl_m", by_month=False,
    )
    depth_recurrence = recurrence_percent_table(
        layers, qc, bin_edges=analysis.layers.height_bin_edges_m, value_col="depth_m", by_month=False,
    )
    monthly_top_iqr = monthly_median_iqr_table(layers, "top_height_agl_m")
    monthly_base_iqr = monthly_median_iqr_table(layers, "base_height_agl_m")
    annual_top = annual_median_table(layers, "top_height_agl_m")
    qc_old = layer_geometry_qc(layers_old)
    qc_new = layer_geometry_qc(layers)

    type_matrices = {kind: frequency_matrix_by_type(type_flags, inversion_type=kind) for kind in INVERSION_TYPES}
    height_heatmaps = {
        kind: year_month_median_matrix(layers, inversion_type=kind, value_col="top_height_agl_m")
        for kind in INVERSION_TYPES
    }
    ref_heights = reference_pressure_heights_agl(analysis)
    ref_heights_table = pd.DataFrame(
        [
            {
                "pressure_hpa": p,
                "height_agl_m": round(h, 1),
                "station_elevation_m": analysis.station_elevation_m,
            }
            for p, h in sorted(ref_heights.items(), reverse=True)
        ]
    )

    tables = {
        "profile_qc": qc,
        "completeness_monthly": completeness_long,
        "inversion_metrics_v2": inversion,
        "inversion_layers_typed": layers,
        "inversion_layers_height_fixed": layers,
        "inversion_layers_pressure_order": layers_old,
        "inversion_type_flags": type_flags,
        "profile_type_flags_height_fixed": type_flags,
        "seasonal_climatology": seasonal,
        "monthly_inversion_frequency": monthly,
        "annual_inversion_frequency": annual,
        "pressure_level_annual_series": pressure_series,
        "height_counts_all": height_all,
        "height_counts_monthly": height_monthly,
        "height_counts_by_type": height_by_type,
        "height_counts_monthly_by_type": height_monthly_by_type,
        "gamma_counts_all": gamma_all,
        "gamma_counts_monthly_1999_2026": gamma_monthly_all,
        "gamma_counts_monthly_850_750_500_1999_2026": gamma_monthly_ref,
        "interval_gammas": interval_gammas,
        "reference_level_gammas": reference_gammas,
        "reference_level_gammas_850_750_500": reference_gammas_850_750_500,
        "top_height_recurrence_percent": top_recurrence,
        "top_height_month_recurrence_percent": top_recurrence_month,
        "base_height_recurrence_percent": base_recurrence,
        "depth_recurrence_percent": depth_recurrence,
        "05_monthly_top_height_median_IQR_fixed": monthly_top_iqr,
        "06_monthly_base_height_median_IQR_fixed": monthly_base_iqr,
        "annual_median_top_height": annual_top,
        "reference_pressure_heights_agl": ref_heights_table,
        "height_qc_old_vs_new": pd.DataFrame(
            [
                {"Проверка": "Толщина ≤ 0", "Старая версия": qc_old["negative_depth"], "Исправленная": qc_new["negative_depth"]},
                {"Проверка": "Верх ≤ основания", "Старая версия": qc_old["top_below_base"], "Исправленная": qc_new["top_below_base"]},
            ]
        ),
    }
    for kind, matrix in type_matrices.items():
        tables[f"frequency_matrix_{kind}"] = matrix.reset_index()

    for name, table in tables.items():
        _write_table_csv(table, tables_dir / f"{name}.csv")

    height_style = FigureStyle(**{**style.__dict__, "show_title": True})
    figure_builders: list[tuple[str, object]] = [
        ("00_height_QC_old_vs_new", lambda: plot_qc_old_vs_new(qc_old, qc_new, height_style)),
        (
            "01_top_height_recurrence_G_E_HE_fixed",
            lambda: plot_recurrence_by_type_bars(
                top_recurrence, height_style,
                value_name="Повторяемость, % от пригодных профилей",
                title="Высота верха инверсий — исправленная геометрия",
            ),
        ),
        (
            "02_base_height_recurrence_G_E_HE_fixed",
            lambda: plot_recurrence_by_type_bars(
                base_recurrence, height_style,
                value_name="Повторяемость, % от пригодных профилей",
                title="Высота основания инверсий — исправленная геометрия",
            ),
        ),
        (
            "03_inversion_depth_recurrence_G_E_HE_fixed",
            lambda: plot_recurrence_by_type_bars(
                depth_recurrence, height_style,
                value_name="Повторяемость, % от пригодных профилей",
                title="Толщина инверсионного слоя — исправленная геометрия",
            ),
        ),
        ("04_base_vs_top_height_QC_fixed", lambda: plot_base_vs_top_scatter(layers, height_style)),
        (
            "05_monthly_top_height_median_IQR_fixed",
            lambda: plot_monthly_median_iqr(
                monthly_top_iqr, height_style,
                ylabel="Высота верха AGL, м",
                title="Сезонный ход высоты верха: медиана и IQR",
            ),
        ),
        (
            "06_monthly_base_height_median_IQR_fixed",
            lambda: plot_monthly_median_iqr(
                monthly_base_iqr, height_style,
                ylabel="Высота основания AGL, м",
                title="Сезонный ход высоты основания: медиана и IQR",
            ),
        ),
        ("08_top_height_cdf_00_12_G_E_HE_fixed", lambda: plot_top_height_cdf_by_cycle(layers, height_style)),
        ("09_annual_median_top_height_G_E_HE_fixed", lambda: plot_annual_median_top(annual_top, height_style)),
        ("10_top_height_seasonal_quantiles_G_E_HE_fixed", lambda: plot_seasonal_quantiles(layers, height_style)),
        (
            "11_top_height_vs_depth_joint_G",
            lambda: plot_top_height_vs_depth_joint(layers, height_style, inversion_type="G"),
        ),
        (
            "11_top_height_vs_depth_joint_E",
            lambda: plot_top_height_vs_depth_joint(layers, height_style, inversion_type="E"),
        ),
        (
            "11_top_height_vs_depth_joint_HE",
            lambda: plot_top_height_vs_depth_joint(layers, height_style, inversion_type="HE"),
        ),
        (
            "12_top_height_vs_depth_monthly_12panel",
            lambda: plot_top_height_vs_depth_monthly_facets(layers, height_style),
        ),
        (
            "13_top_height_vs_depth_boxplots",
            lambda: plot_top_height_vs_depth_boxplots(layers, height_style),
        ),
        (
            "14_gamma_vs_height_joint",
            lambda: plot_gamma_scatter_hist(reference_gammas, ref_heights, height_style),
        ),
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
                interval_gammas, style, bin_edges=analysis.layers.gamma_bin_edges_c_per_100m
            ),
        ),
        ("type03_gamma_monthly_box", lambda: plot_gamma_by_month_box(interval_gammas, style)),
    ]
    for kind in INVERSION_TYPES:
        figure_builders.append(
            (
                f"07_heatmap_top_height_{kind}_fixed",
                lambda k=kind: plot_height_median_heatmap(
                    height_heatmaps[k], height_style, inversion_type=k,
                    title=f"{'Приземная G' if k=='G' else 'Приподнятая E' if k=='E' else 'Высокая приподнятая HE'}: медианная высота верха",
                ),
            )
        )
        subset = height_by_type[height_by_type["position_type"] == kind]
        figure_builders.append(
            (
                f"type02_height_bar_{kind}",
                lambda table=subset: plot_height_counts_bar(table, style),
            )
        )
    for month in range(1, 13):
        figure_builders.append(
            (
                f"type02_height_bar_m{month:02d}",
                lambda m=month: plot_height_counts_bar(height_monthly, style, month=m),
            )
        )
        for kind in INVERSION_TYPES:
            figure_builders.append(
                (
                    f"type02_height_bar_m{month:02d}_{kind}",
                    lambda m=month, k=kind: plot_height_counts_bar(
                        height_monthly_by_type, style, month=m, inversion_type=k,
                    ),
                )
            )

    saved: dict[str, list[str]] = {}
    for name, builder in figure_builders:
        fig = builder()
        saved[name] = [str(p) for p in save_figure(fig, figures_dir / name, style)]

    gamma_monthly_dir = figures_dir / "gamma_monthly"
    gamma_ref_dir = figures_dir / "gamma_monthly_850_750_500"
    gamma_monthly_style = FigureStyle(**{**style.__dict__, "show_title": True, "dpi": 300})
    gamma_extra: list[tuple[Path, object]] = [
        (
            gamma_monthly_dir / "gamma_line_12months_1999-2025",
            lambda: plot_gamma_line_monthly_facets(
                gamma_monthly_all,
                gamma_monthly_style,
                year_label=year_label,
            ),
        ),
        (
            gamma_ref_dir / "gamma_line_850_750_500_12months_1999-2025",
            lambda: plot_gamma_reference_line_monthly_facets(
                gamma_monthly_ref,
                gamma_monthly_style,
                pressures_hpa=(850.0, 750.0, 500.0),
                year_label=year_label,
            ),
        ),
    ]
    for path, builder in gamma_extra:
        fig = builder()
        key = str(path.relative_to(figures_dir)).replace("\\", "/")
        saved[key] = [str(p) for p in save_figure(fig, path, gamma_monthly_style)]

    gamma_by_year_dir = figures_dir / "gamma_by_year"
    gamma_edges = analysis.layers.gamma_bin_edges_c_per_100m
    gamma_year_style = FigureStyle(**{**gamma_monthly_style.__dict__, "output_formats": ("png",)})
    for year in range(gamma_year_start, gamma_year_end + 1):
        year_slice = interval_gammas_period[interval_gammas_period["year"] == year]
        if year_slice.empty:
            continue
        year_table = gamma_count_table(year_slice, bin_edges=gamma_edges, by_month=True)
        year_label_one = str(year)
        fig = plot_gamma_line_monthly_facets(
            year_table,
            gamma_year_style,
            year_label=year_label_one,
            title=f"Распределение γ по месяцам, {year}" if gamma_year_style.language == "ru" else f"γ by month, {year}",
        )
        rel = gamma_by_year_dir / f"gamma_line_{year}"
        key = str(rel.relative_to(figures_dir)).replace("\\", "/")
        saved[key] = [str(p) for p in save_figure(fig, rel, gamma_year_style)]

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
        "method": "height-primary gap-v3",
        "qc_checks": {"old": qc_old, "new": qc_new},
        "reference_pressure_heights_agl": ref_heights_table.to_dict(orient="records"),
        "trend": trend_stats,
        "analysis_config": asdict(analysis),
        "figure_style": asdict(style),
        "figures": saved,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary
