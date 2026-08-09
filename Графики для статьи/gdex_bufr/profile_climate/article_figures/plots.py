from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FigureStyle
from .metrics import SEASON_ORDER

MONTHS_RU = ["Янв", "Фев", "Мар", "Апр", "Май", "Июн", "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"]
MONTHS_EN = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SEASONS_RU = {"DJF": "Зима", "MAM": "Весна", "JJA": "Лето", "SON": "Осень"}
SEASONS_EN = {"DJF": "Winter", "MAM": "Spring", "JJA": "Summer", "SON": "Autumn"}

DEFAULT_ANNOTATE_BELOW_PERCENT = 80.0
COMPLETENESS_VMIN = 0
COMPLETENESS_VMAX = 100


@contextmanager
def article_rc(style: FigureStyle):
    params = {
        "font.family": style.font_family,
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
    }
    with mpl.rc_context(params):
        yield


def _months(style: FigureStyle) -> list[str]:
    return MONTHS_RU if style.language == "ru" else MONTHS_EN


def _finish(fig, style: FigureStyle):
    fig.tight_layout()
    return fig


def plot_completeness_heatmap(
    matrix: pd.DataFrame,
    style: FigureStyle,
    *,
    annotate_below: float | None = DEFAULT_ANNOTATE_BELOW_PERCENT,
    title: str | None = None,
):
    with article_rc(style):
        fig, ax = plt.subplots(figsize=(style.figure_width_in, max(style.figure_height_in, 5.2)))
        values = matrix.to_numpy(float)
        masked = np.ma.masked_invalid(values)
        cmap = mpl.colormaps[style.completeness_cmap].copy()
        cmap.set_bad(style.missing_color)
        image = ax.imshow(
            masked,
            aspect="auto",
            interpolation="nearest",
            vmin=COMPLETENESS_VMIN,
            vmax=COMPLETENESS_VMAX,
            cmap=cmap,
        )
        ax.set_xticks(np.arange(12), labels=_months(style))
        ax.set_yticks(np.arange(len(matrix.index)), labels=[str(x) for x in matrix.index])
        ax.set_xlabel("Месяц" if style.language == "ru" else "Month")
        ax.set_ylabel("Год" if style.language == "ru" else "Year")
        ax.set_xticks(np.arange(-0.5, 12, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(matrix.index), 1), minor=True)
        ax.grid(which="minor", linewidth=0.35, alpha=0.35)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)

        if annotate_below is not None:
            for y in range(values.shape[0]):
                for x in range(values.shape[1]):
                    v = values[y, x]
                    if np.isfinite(v) and v < annotate_below:
                        ax.text(x, y, f"{v:.0f}", ha="center", va="center", fontsize=max(6.0, style.tick_font_size - 1.5))

        cbar = fig.colorbar(image, ax=ax, pad=0.025, fraction=0.035)
        cbar.set_label("Полнота наблюдений, %" if style.language == "ru" else "Observation completeness, %")
        if style.show_title:
            ax.set_title(title or ("Месячная полнота радиозондовых профилей Алдана" if style.language == "ru" else "Monthly completeness of Aldan radiosonde profiles"))
        return _finish(fig, style)


def plot_seasonal_temperature_profiles(
    climatology: pd.DataFrame,
    style: FigureStyle,
    *,
    title: str | None = None,
):
    labels = SEASONS_RU if style.language == "ru" else SEASONS_EN
    with article_rc(style):
        fig, ax = plt.subplots(figsize=(style.figure_width_in * 0.78, style.figure_height_in * 1.22))
        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for idx, season in enumerate(SEASON_ORDER):
            g = climatology[climatology["season"].astype(str) == season].sort_values("pressure_hpa")
            if g.empty:
                continue
            color = cycle[idx % len(cycle)]
            ax.plot(g["median"], g["pressure_hpa"], label=labels[season], linewidth=style.line_width, color=color)
            ax.fill_betweenx(g["pressure_hpa"], g["q25"], g["q75"], alpha=0.15, color=color, linewidth=0)
        ax.invert_yaxis()
        ax.set_xlabel("Температура, °C" if style.language == "ru" else "Temperature, °C")
        ax.set_ylabel("Давление, гПа" if style.language == "ru" else "Pressure, hPa")
        ax.grid(True, alpha=style.grid_alpha, linewidth=0.5)
        ax.legend(frameon=False, loc="best")
        if style.show_title:
            ax.set_title(title or ("Сезонные медианные температурные профили" if style.language == "ru" else "Seasonal median temperature profiles"))
        return _finish(fig, style)


def plot_monthly_inversion_frequency(
    monthly: pd.DataFrame,
    style: FigureStyle,
    *,
    title: str | None = None,
):
    with article_rc(style):
        fig, ax = plt.subplots(figsize=(style.figure_width_in, style.figure_height_in))
        for idx, cycle_value in enumerate(sorted(monthly["cycle"].unique())):
            g = monthly[monthly["cycle"] == cycle_value].set_index("month").reindex(range(1, 13))
            ax.plot(
                range(1, 13), g["frequency_percent"], marker="o", markersize=style.marker_size,
                linewidth=style.line_width, label=f"{cycle_value} UTC",
                linestyle="-" if idx == 0 else "--",
            )
        ax.set_xticks(range(1, 13), labels=_months(style))
        ax.set_xlabel("Месяц" if style.language == "ru" else "Month")
        ax.set_ylabel("Повторяемость, %" if style.language == "ru" else "Frequency, %")
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", alpha=style.grid_alpha, linewidth=0.5)
        ax.legend(frameon=False, ncol=2)
        if style.show_title:
            ax.set_title(title or ("Годовой ход подтверждённых приземных инверсий" if style.language == "ru" else "Annual cycle of confirmed surface inversions"))
        return _finish(fig, style)


def plot_annual_inversion_variability(
    annual: pd.DataFrame,
    stats: dict[str, float],
    style: FigureStyle,
    *,
    title: str | None = None,
):
    with article_rc(style):
        fig, ax = plt.subplots(figsize=(style.figure_width_in, style.figure_height_in))
        ax.plot(annual["year"], annual["frequency_percent"], marker="o", markersize=style.marker_size, linewidth=style.line_width, label="Годовая повторяемость" if style.language == "ru" else "Annual frequency")
        ax.plot(annual["year"], annual["moving_average_percent"], linewidth=style.line_width + 0.3, label="5-летнее скользящее среднее" if style.language == "ru" else "5-year moving average")
        slope = stats.get("sen_slope_pp_per_year", np.nan)
        if annual["sen_trend_percent"].notna().any():
            slope_label = (f"Наклон Сена: {slope:+.2f} п.п./год" if style.language == "ru" else f"Sen slope: {slope:+.2f} pp/year")
            ax.plot(annual["year"], annual["sen_trend_percent"], linestyle=":", linewidth=style.line_width, label=slope_label)
        ax.set_xlabel("Год" if style.language == "ru" else "Year")
        ax.set_ylabel("Повторяемость, %" if style.language == "ru" else "Frequency, %")
        ax.grid(True, axis="y", alpha=style.grid_alpha, linewidth=0.5)
        ax.legend(frameon=False, loc="best")
        if style.show_title:
            ax.set_title(title or ("Межгодовая изменчивость повторяемости инверсий" if style.language == "ru" else "Interannual variability of inversion frequency"))
        return _finish(fig, style)


def _monthly_boxplot(metrics: pd.DataFrame, column: str, ylabel_ru: str, ylabel_en: str, style: FigureStyle, title: str | None = None):
    data = metrics[metrics["eligible_article"] & metrics["inversion_detected"]].copy()
    values = [data.loc[data["month"] == m, column].dropna().to_numpy() for m in range(1, 13)]
    with article_rc(style):
        fig, ax = plt.subplots(figsize=(style.figure_width_in, style.figure_height_in))
        bp = ax.boxplot(values, tick_labels=_months(style), showfliers=False, patch_artist=True, widths=0.62)
        for patch in bp["boxes"]:
            patch.set_alpha(0.35)
        for median in bp["medians"]:
            median.set_linewidth(style.line_width)
        ax.set_xlabel("Месяц" if style.language == "ru" else "Month")
        ax.set_ylabel(ylabel_ru if style.language == "ru" else ylabel_en)
        ax.grid(True, axis="y", alpha=style.grid_alpha, linewidth=0.5)
        if style.show_title and title:
            ax.set_title(title)
        return _finish(fig, style)


def plot_monthly_inversion_intensity(metrics: pd.DataFrame, style: FigureStyle):
    return _monthly_boxplot(metrics, "inversion_delta_t_c", "Интенсивность инверсии, °C", "Inversion strength, °C", style, "Сезонное распределение интенсивности инверсий")


def plot_monthly_inversion_top_height(metrics: pd.DataFrame, style: FigureStyle):
    return _monthly_boxplot(metrics, "inversion_top_height_agl_m", "Высота верха над нижним уровнем, м", "Top height above lowest level, m", style, "Сезонное распределение высоты верха инверсий")


def plot_profile_qc_summary(profile_qc: pd.DataFrame, style: FigureStyle):
    counts = profile_qc["profile_status"].value_counts().sort_values(ascending=True)
    with article_rc(style):
        fig, ax = plt.subplots(figsize=(style.figure_width_in, style.figure_height_in * 0.8))
        bars = ax.barh(counts.index, counts.values)
        ax.bar_label(bars, padding=3, fontsize=style.tick_font_size)
        ax.set_xlabel("Число профилей" if style.language == "ru" else "Number of profiles")
        ax.set_ylabel("Статус профиля" if style.language == "ru" else "Profile status")
        ax.grid(True, axis="x", alpha=style.grid_alpha, linewidth=0.5)
        return _finish(fig, style)


def plot_monthly_profile_bundle(
    df: pd.DataFrame,
    style: FigureStyle,
    *,
    year: int,
    month: int,
    cycles: Iterable[str] = ("00", "12"),
    max_profiles: int | None = None,
):
    use = df[(df["year"] == year) & (df["month"] == month) & df["cycle"].isin([str(x).zfill(2) for x in cycles])]
    groups = list(use.groupby("profile_id", sort=True))
    if max_profiles:
        groups = groups[:max_profiles]
    with article_rc(style):
        fig, ax = plt.subplots(figsize=(style.figure_width_in * 0.8, style.figure_height_in * 1.15))
        for _, g in groups:
            g = g.dropna(subset=["pressure_hpa", "temperature_c"]).sort_values("pressure_hpa", ascending=False)
            ax.plot(g["temperature_c"], g["pressure_hpa"], linewidth=0.65, alpha=0.35)
        ax.invert_yaxis()
        ax.set_xlabel("Температура, °C" if style.language == "ru" else "Temperature, °C")
        ax.set_ylabel("Давление, гПа" if style.language == "ru" else "Pressure, hPa")
        ax.grid(True, alpha=style.grid_alpha, linewidth=0.5)
        if style.show_title:
            ax.set_title(f"{year}-{month:02d}: {len(groups)} профилей" if style.language == "ru" else f"{year}-{month:02d}: {len(groups)} profiles")
        return _finish(fig, style)


def plot_pressure_level_time_series(series: pd.DataFrame, style: FigureStyle):
    with article_rc(style):
        fig, ax = plt.subplots(figsize=(style.figure_width_in, style.figure_height_in))
        for p in sorted(series["pressure_hpa"].unique(), reverse=True):
            g = series[series["pressure_hpa"] == p].sort_values("year")
            ax.plot(g["year"], g["median_temperature_c"], linewidth=style.line_width, label=f"{p:g} гПа" if style.language == "ru" else f"{p:g} hPa")
        ax.set_xlabel("Год" if style.language == "ru" else "Year")
        ax.set_ylabel("Медианная температура, °C" if style.language == "ru" else "Median temperature, °C")
        ax.grid(True, axis="y", alpha=style.grid_alpha, linewidth=0.5)
        ax.legend(frameon=False, ncol=2)
        return _finish(fig, style)
