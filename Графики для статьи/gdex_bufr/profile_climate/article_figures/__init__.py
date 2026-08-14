"""Инструменты подготовки графиков для статьи о профилях Алдана."""

from .config import AnalysisConfig, FigureStyle, InversionConfig, LayerClassConfig
from .data import load_profiles, build_profile_qc, compute_completeness
from .metrics import compute_inversion_metrics, compute_seasonal_climatology
from .plots import (
    plot_completeness_heatmap,
    plot_seasonal_temperature_profiles,
    plot_monthly_inversion_frequency,
    plot_annual_inversion_variability,
    plot_monthly_inversion_intensity,
    plot_monthly_inversion_top_height,
    plot_profile_qc_summary,
    plot_monthly_profile_bundle,
    plot_pressure_level_time_series,
    plot_inversion_type_frequency_matrix,
    plot_height_counts_bar,
    plot_height_counts_line,
    plot_gamma_counts_bar,
    plot_gamma_counts_line,
)

__all__ = [
    "AnalysisConfig",
    "FigureStyle",
    "InversionConfig",
    "LayerClassConfig",
    "load_profiles",
    "build_profile_qc",
    "compute_completeness",
    "compute_inversion_metrics",
    "compute_seasonal_climatology",
    "plot_completeness_heatmap",
    "plot_seasonal_temperature_profiles",
    "plot_monthly_inversion_frequency",
    "plot_annual_inversion_variability",
    "plot_monthly_inversion_intensity",
    "plot_monthly_inversion_top_height",
    "plot_profile_qc_summary",
    "plot_monthly_profile_bundle",
    "plot_pressure_level_time_series",
    "plot_inversion_type_frequency_matrix",
    "plot_height_counts_bar",
    "plot_height_counts_line",
    "plot_gamma_counts_bar",
    "plot_gamma_counts_line",
]
