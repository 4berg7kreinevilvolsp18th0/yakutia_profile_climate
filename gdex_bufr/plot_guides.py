"""Справочник метеодиаграмм GDEX BUFR — метаданные и пути к markdown."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

GUIDES_ROOT = Path(__file__).resolve().parent / "guides"
PLOTS_DIR = GUIDES_ROOT / "plots"
CUSTOM_DIR = GUIDES_ROOT / "custom"


@dataclass(frozen=True)
class PlotGuideInfo:
    plot_type: str
    title_ru: str
    color_slot: str
    output_suffix: str
    guide_file: str
    min_levels: int = 2


PLOT_GUIDES: dict[str, PlotGuideInfo] = {
    "skewt": PlotGuideInfo(
        "skewt", "Skew-T лог-p", "temperature / dewpoint / barbs", "_skewt.png", "skewt.md", 2
    ),
    "profile": PlotGuideInfo(
        "profile", "Вертикальный профиль T/Td", "temperature / dewpoint", "_profile.png", "profile.md", 2
    ),
    "wind": PlotGuideInfo(
        "wind", "Профиль ветра", "wind_speed / wind_direction", "_wind.png", "wind.md", 1
    ),
    "hodograph": PlotGuideInfo(
        "hodograph", "Годограф", "hodograph", "_hodograph.png", "hodograph.md", 2
    ),
    "rh": PlotGuideInfo(
        "rh", "Относительная влажность", "rh", "_rh.png", "rh.md", 2
    ),
    "thermo": PlotGuideInfo(
        "thermo", "Термодинамические индексы", "thermo_box", "_thermo.png", "thermo.md", 5
    ),
    "height": PlotGuideInfo(
        "height", "Геопотенциальная высота", "height", "_height.png", "height.md", 2
    ),
    "theta_e": PlotGuideInfo(
        "theta_e", "Эквивалентная потенциальная температура", "theta_e", "_theta_e.png", "theta_e.md", 2
    ),
    "ttd": PlotGuideInfo(
        "ttd", "Разность T − Td", "ttd", "_ttd.png", "ttd.md", 2
    ),
    "wind_shear": PlotGuideInfo(
        "wind_shear", "Сдвиг ветра", "wind_shear", "_wind_shear.png", "wind_shear.md", 2
    ),
    "composite": PlotGuideInfo(
        "composite", "Сводная панель", "все слоты theme", "_composite.png", "composite.md", 2
    ),
    "map": PlotGuideInfo(
        "map", "Карта станций", "map_stations", "_stations_map.png", "map.md", 1
    ),
}


def guide_path(plot_type: str) -> Path | None:
    info = PLOT_GUIDES.get(plot_type)
    if info is None:
        return None
    path = PLOTS_DIR / info.guide_file
    return path if path.exists() else None


def list_guides() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for key, info in PLOT_GUIDES.items():
        path = PLOTS_DIR / info.guide_file
        rows.append({
            "plot_type": key,
            "title_ru": info.title_ru,
            "color_slot": info.color_slot,
            "guide": str(path),
            "exists": str(path.exists()),
        })
    return rows
