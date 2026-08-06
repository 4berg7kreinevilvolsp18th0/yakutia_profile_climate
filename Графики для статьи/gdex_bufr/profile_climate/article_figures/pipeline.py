from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import AnalysisConfig, FigureStyle
from .data import build_profile_qc, compute_completeness, load_profiles
from .metrics import (
    annual_inversion_frequency,
    compute_inversion_metrics,
    compute_seasonal_climatology,
    monthly_inversion_frequency,
    pressure_level_annual_series,
    interpolate_eligible_profiles,
)
from .plots import (
    plot_annual_inversion_variability,
    plot_completeness_heatmap,
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

    tables = {
        "profile_qc": qc,
        "completeness_monthly": completeness_long,
        "inversion_metrics_v2": inversion,
        "seasonal_climatology": seasonal,
        "monthly_inversion_frequency": monthly,
        "annual_inversion_frequency": annual,
        "pressure_level_annual_series": pressure_series,
    }
    for name, table in tables.items():
        table.to_csv(tables_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    figures = {
        "fig01_completeness_heatmap": plot_completeness_heatmap(completeness_matrix, style),
        "fig02_seasonal_temperature_profiles": plot_seasonal_temperature_profiles(seasonal, style),
        "fig03_monthly_inversion_frequency": plot_monthly_inversion_frequency(monthly, style),
        "fig04_annual_inversion_variability": plot_annual_inversion_variability(annual, trend_stats, style),
        "extra01_monthly_inversion_intensity": plot_monthly_inversion_intensity(inversion, style),
        "extra02_monthly_inversion_top_height": plot_monthly_inversion_top_height(inversion, style),
        "extra03_profile_qc_summary": plot_profile_qc_summary(qc, style),
        "extra04_pressure_level_time_series": plot_pressure_level_time_series(pressure_series, style),
    }
    saved = {name: [str(p) for p in save_figure(fig, figures_dir / name, style)] for name, fig in figures.items()}

    summary = {
        "input": str(input_csv),
        "rows": int(len(df)),
        "profiles": int(qc.shape[0]),
        "eligible_article_profiles": int(qc["eligible_article"].sum()),
        "confirmed_inversions": int(inversion.loc[inversion["eligible_article"], "inversion_detected"].sum()),
        "trend": trend_stats,
        "analysis_config": asdict(analysis),
        "figure_style": asdict(style),
        "figures": saved,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary
