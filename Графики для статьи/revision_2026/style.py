"""Единый стиль ревизии: serif, белый фон, без внутреннего заголовка."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt

from gdex_bufr.profile_climate.article_figures.config import FigureStyle

TYPE_COLORS = {"G": "#B03A2E", "E": "#2471A3", "HE": "#6C3483"}
TYPE_LABELS = {
    "G": "Приземная (G)",
    "E": "Приподнятая (E)",
    "HE": "Высокоприподнятая (HE)",
}
LEVEL_COLORS = {850.0: "#1A5276", 700.0: "#B9770E", 500.0: "#196F3D"}
DIAG_HIST_COLOR = "#2471A3"
DIAG_HEXBIN_CMAP = "YlOrRd"
DIAG_HEATMAP_CMAP = "YlOrRd"
NEUTRAL_LINE = "#7F8C8D"
REFERENCE_LINE = "#1C2833"
LEVEL_LABELS = {850.0: "850 гПа", 700.0: "700 гПа", 500.0: "500 гПа"}
MONTHS_RU = ["Янв", "Фев", "Мар", "Апр", "Май", "Июн", "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"]
SEASONS_RU = {"DJF": "Зима (DJF)", "MAM": "Весна (MAM)", "JJA": "Лето (JJA)", "SON": "Осень (SON)"}
SEASON_ORDER = ("DJF", "MAM", "JJA", "SON")


def pick_serif() -> str:
    available = {f.name for f in mpl.font_manager.fontManager.ttflist}
    for name in ("Times New Roman", "Times", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"):
        if name in available:
            return name
    return "serif"


def revision_style(base: FigureStyle | None = None) -> FigureStyle:
    src = (base.__dict__ if base is not None else FigureStyle().__dict__).copy()
    src.update(
        {
            "language": "ru",
            "journal_mode": True,
            "show_title": False,
            "font_family": pick_serif(),
            "dpi": 600,
            "output_formats": ("png", "svg"),
            "figure_width_in": 7.2,
            "figure_height_in": 4.6,
        }
    )
    return FigureStyle(**src)


@contextmanager
def revision_rc(style: FigureStyle):
    params = {
        "font.family": style.font_family,
        "font.serif": [style.font_family, "DejaVu Serif", "serif"],
        "font.size": style.base_font_size,
        "axes.titlesize": style.title_font_size,
        "axes.labelsize": style.label_font_size,
        "xtick.labelsize": style.tick_font_size,
        "ytick.labelsize": style.tick_font_size,
        "legend.fontsize": style.legend_font_size,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.unicode_minus": False,
    }
    with mpl.rc_context(params):
        yield


def add_caption(fig: plt.Figure, text: str) -> None:
    fig.text(0.5, 0.01, text, ha="center", va="bottom", fontsize=7.5, color="#4D4D4D")


def station_caption(station_name: str, station_id: str, year_from: int, year_to: int) -> str:
    return (
        f"{station_name}, WMO {station_id} (≈680 м н.у.м.); "
        f"{year_from}–{year_to}, сроки 00 и 12 UTC"
    )


def finish(fig: plt.Figure, style: FigureStyle, caption: str | None = None) -> plt.Figure:
    if caption:
        fig.subplots_adjust(bottom=0.14)
        add_caption(fig, caption)
    else:
        fig.tight_layout()
    return fig
