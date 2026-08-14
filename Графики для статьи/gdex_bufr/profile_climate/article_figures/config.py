from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class InversionConfig:
    """Пороговые параметры алгоритма приземной инверсии v2."""

    min_inversion_delta_c: float = 0.2
    confirm_drop_levels: int = 2
    confirm_depth_hpa: float = 30.0
    min_drop_delta_c: float = 0.2


@dataclass(frozen=True)
class LayerClassConfig:
    """Классификация слоёв по высоте основания (как G/E/HE, без изменения V3)."""

    surface_tolerance_m: float = 30.0
    he_threshold_m: float = 250.0
    height_bin_edges_m: tuple[float, ...] = (0.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0)
    gamma_bin_edges_c_per_100m: tuple[float, ...] = (
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0,
    )


@dataclass(frozen=True)
class AnalysisConfig:
    """Правила отбора и агрегации профилей."""

    station_id: str = "31004"
    station_name: str = "Алдан"
    station_elevation_m: float = 679.0
    cycles: tuple[str, ...] = ("00", "12")
    pressure_top_hpa: float = 500.0
    pressure_bottom_hpa: float = 1000.0
    min_levels_to_500: int = 5
    exact_top_tolerance_hpa: float = 0.6

    # Дополнительный фильтр для журнального анализа. Его можно отключить в UI.
    strict_surface_qc: bool = True
    max_surface_pressure_hpa: float = 960.0
    use_surface_height_qc: bool = False
    max_surface_height_deviation_m: float = 250.0

    trend_start_year: int = 2005
    trend_end_year: int = 2025
    moving_average_window: int = 5

    pressure_grid_hpa: tuple[float, ...] = tuple(float(x) for x in range(500, 926, 25))
    standard_pressure_levels_hpa: tuple[float, ...] = (925.0, 850.0, 700.0, 500.0)
    inversion: InversionConfig = field(default_factory=InversionConfig)
    layers: LayerClassConfig = field(default_factory=LayerClassConfig)


@dataclass(frozen=True)
class FigureStyle:
    """Параметры оформления и экспорта рисунков."""

    language: str = "ru"
    journal_mode: bool = True
    show_title: bool = False
    font_family: str = "DejaVu Sans"
    base_font_size: float = 9.5
    title_font_size: float = 12.0
    label_font_size: float = 10.0
    tick_font_size: float = 8.5
    legend_font_size: float = 8.5
    line_width: float = 1.8
    marker_size: float = 4.5
    grid_alpha: float = 0.25
    dpi: int = 600
    completeness_cmap: str = "cividis"
    missing_color: str = "#E6E8EB"
    figure_width_in: float = 7.2
    figure_height_in: float = 4.5
    output_formats: tuple[str, ...] = ("png", "svg")


def load_yaml_config(path: str | Path) -> tuple[AnalysisConfig, FigureStyle]:
    """Загрузить частичную конфигурацию YAML поверх значений по умолчанию."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Для YAML-конфига установите PyYAML") from exc

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    analysis_raw = dict(raw.get("analysis", {}))
    inversion_raw = dict(analysis_raw.pop("inversion", {}))
    layers_raw = dict(analysis_raw.pop("layers", {}))
    if "cycles" in analysis_raw:
        analysis_raw["cycles"] = tuple(str(x).zfill(2) for x in analysis_raw["cycles"])
    if "pressure_grid_hpa" in analysis_raw:
        analysis_raw["pressure_grid_hpa"] = tuple(float(x) for x in analysis_raw["pressure_grid_hpa"])
    if "standard_pressure_levels_hpa" in analysis_raw:
        analysis_raw["standard_pressure_levels_hpa"] = tuple(
            float(x) for x in analysis_raw["standard_pressure_levels_hpa"]
        )
    if "height_bin_edges_m" in layers_raw:
        layers_raw["height_bin_edges_m"] = tuple(float(x) for x in layers_raw["height_bin_edges_m"])
    if "gamma_bin_edges_c_per_100m" in layers_raw:
        layers_raw["gamma_bin_edges_c_per_100m"] = tuple(
            float(x) for x in layers_raw["gamma_bin_edges_c_per_100m"]
        )
    analysis_raw["inversion"] = InversionConfig(**inversion_raw)
    analysis_raw["layers"] = LayerClassConfig(**layers_raw)

    style_raw = dict(raw.get("style", {}))
    if "output_formats" in style_raw:
        style_raw["output_formats"] = tuple(style_raw["output_formats"])
    return AnalysisConfig(**analysis_raw), FigureStyle(**style_raw)
