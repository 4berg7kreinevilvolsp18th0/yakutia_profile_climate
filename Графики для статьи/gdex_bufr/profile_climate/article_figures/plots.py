from __future__ import annotations

from contextlib import contextmanager
from itertools import permutations
from pathlib import Path
from typing import Callable, Iterable, Literal, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

from .config import AnalysisConfig, FigureStyle
from .metrics import SEASON_BY_MONTH, SEASON_ORDER

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


def _bin_tick_labels(left: np.ndarray, right: np.ndarray) -> list[str]:
    labels = []
    for lo, hi in zip(left, right):
        if not np.isfinite(lo):
            labels.append(f"≤{int(hi)}" if float(hi).is_integer() else f"≤{hi:g}")
        elif not np.isfinite(hi):
            labels.append(f"≥{int(lo)}" if float(lo).is_integer() else f"≥{lo:g}")
        else:
            lo_s = str(int(lo)) if float(lo).is_integer() else f"{lo:g}"
            hi_s = str(int(hi)) if float(hi).is_integer() else f"{hi:g}"
            labels.append(f"{lo_s}–{hi_s}")
    return labels


def _equal_width_bars(ax, labels: list[str], values, *, color=None, rotate: bool = True):
    """Столбцы одинаковой ширины; подписи бинов на категориальной оси."""
    x = np.arange(len(labels), dtype=float)
    bars = ax.bar(
        x,
        np.asarray(values, dtype=float),
        width=0.82,
        align="center",
        alpha=0.82,
        color=color or "#2E86C1",
        edgecolor="#1B4F72",
        linewidth=0.4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45 if rotate else 0, ha="right" if rotate else "center")
    ax.set_xlim(-0.6, max(len(labels) - 0.4, 0.6))
    return bars


def _type_title(inversion_type: str, style: FigureStyle) -> str:
    from .metrics import TYPE_LABELS_EN, TYPE_LABELS_RU

    labels = TYPE_LABELS_RU if style.language == "ru" else TYPE_LABELS_EN
    return labels.get(inversion_type, inversion_type)


def plot_inversion_type_frequency_matrix(
    matrix: pd.DataFrame,
    style: FigureStyle,
    *,
    inversion_type: str,
    title: str | None = None,
):
    """Матрица год×месяц: повторяемость (%) одного вида инверсии."""
    with article_rc(style):
        fig, ax = plt.subplots(figsize=(style.figure_width_in, max(style.figure_height_in, 5.2)))
        values = matrix.to_numpy(float)
        masked = np.ma.masked_invalid(values)
        cmap = mpl.colormaps["YlOrRd"].copy()
        cmap.set_bad(style.missing_color)
        vmax = float(np.nanmax(values)) if np.isfinite(values).any() else 100.0
        vmax = max(vmax, 1.0)
        image = ax.imshow(
            masked,
            aspect="auto",
            interpolation="nearest",
            vmin=0,
            vmax=vmax,
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
        cbar = fig.colorbar(image, ax=ax, pad=0.025, fraction=0.035)
        cbar.set_label("Повторяемость, %" if style.language == "ru" else "Frequency, %")
        if style.show_title:
            kind = _type_title(inversion_type, style)
            ax.set_title(
                title
                or (
                    f"Повторяемость инверсий: {kind}"
                    if style.language == "ru"
                    else f"Inversion frequency: {kind}"
                )
            )
        return _finish(fig, style)


def plot_height_counts_bar(
    table: pd.DataFrame,
    style: FigureStyle,
    *,
    month: int | None = None,
    inversion_type: str | None = None,
    title: str | None = None,
):
    """Столбчатый: X — высота, Y — число инверсий."""
    data = table[table["month"] == (month or 0)].copy() if "month" in table.columns else table.copy()
    if inversion_type and "position_type" in data.columns:
        data = data[data["position_type"] == inversion_type]
    data = data.sort_values("bin_left") if not data.empty else data
    color = TYPE_COLORS.get(inversion_type) if inversion_type else None
    with article_rc(style):
        fig, ax = plt.subplots(figsize=(style.figure_width_in, style.figure_height_in))
        if data.empty:
            ax.text(0.5, 0.5, "нет данных", ha="center", va="center", transform=ax.transAxes)
        else:
            labels = _bin_tick_labels(data["bin_left"].to_numpy(float), data["bin_right"].to_numpy(float))
            _equal_width_bars(ax, labels, data["count"].to_numpy(float), color=color)
        ax.set_xlabel("Интервал высоты верха AGL, м" if style.language == "ru" else "Top height AGL bin, m")
        ax.set_ylabel("Число инверсий" if style.language == "ru" else "Inversion count")
        ax.grid(True, axis="y", alpha=style.grid_alpha, linewidth=0.5)
        if style.show_title:
            if title:
                ax.set_title(title)
            else:
                parts = ["Высоты инверсий" if style.language == "ru" else "Inversion heights"]
                if month:
                    parts.append(_months(style)[month - 1])
                if inversion_type:
                    parts.append(_type_title(inversion_type, style))
                ax.set_title(" — ".join(parts) if (month or inversion_type) else (
                    "Распределение высот инверсий" if style.language == "ru" else "Inversion height distribution"
                ))
        return _finish(fig, style)


def plot_height_counts_line(
    table: pd.DataFrame,
    style: FigureStyle,
    *,
    month: int | None = None,
    title: str | None = None,
):
    """Линейный: X — высота, Y — число инверсий."""
    data = table[table["month"] == (month or 0)].copy() if "month" in table.columns else table.copy()
    data = data.sort_values("bin_left") if not data.empty else data
    with article_rc(style):
        fig, ax = plt.subplots(figsize=(style.figure_width_in, style.figure_height_in))
        if not data.empty:
            ax.plot(
                data["bin_center"],
                data["count"],
                marker="o",
                markersize=style.marker_size,
                linewidth=style.line_width,
            )
            ax.fill_between(data["bin_center"], data["count"], alpha=0.15)
        ax.set_xlabel("Высота верха AGL, м" if style.language == "ru" else "Top height AGL, m")
        ax.set_ylabel("Число инверсий" if style.language == "ru" else "Inversion count")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=style.grid_alpha, linewidth=0.5)
        if style.show_title:
            ax.set_title(title or ("Высоты инверсий (линия)" if style.language == "ru" else "Inversion heights (line)"))
        return _finish(fig, style)


def plot_height_counts_by_month_facets(
    monthly_table: pd.DataFrame,
    style: FigureStyle,
    *,
    title: str | None = None,
):
    """12 панелей: высоты инверсий по месяцам (столбцы)."""
    with article_rc(style):
        fig, axes = plt.subplots(3, 4, figsize=(style.figure_width_in * 1.35, style.figure_height_in * 1.6), sharey=True)
        months = _months(style)
        for ax, month in zip(axes.ravel(), range(1, 13)):
            g = monthly_table[monthly_table["month"] == month]
            if not g.empty:
                g = g.sort_values("bin_left")
                labels = _bin_tick_labels(g["bin_left"].to_numpy(float), g["bin_right"].to_numpy(float))
                _equal_width_bars(ax, labels, g["count"].to_numpy(float), rotate=True)
            ax.set_title(months[month - 1], fontsize=style.tick_font_size)
            ax.grid(True, axis="y", alpha=style.grid_alpha, linewidth=0.4)
            if month > 8:
                ax.set_xlabel("H AGL, м" if style.language == "ru" else "H AGL, m", fontsize=style.tick_font_size - 0.5)
            if month in (1, 5, 9):
                ax.set_ylabel("N" if style.language != "ru" else "Число", fontsize=style.tick_font_size - 0.5)
        if style.show_title:
            fig.suptitle(title or ("Высоты инверсий по месяцам" if style.language == "ru" else "Inversion heights by month"))
        fig.tight_layout()
        return fig


def plot_height_counts_months_overlay(
    monthly_table: pd.DataFrame,
    style: FigureStyle,
    *,
    title: str | None = None,
):
    """Все месяцы на одном графике линиями."""
    with article_rc(style):
        fig, ax = plt.subplots(figsize=(style.figure_width_in, style.figure_height_in))
        cmap = mpl.colormaps["turbo"]
        months = _months(style)
        for month in range(1, 13):
            g = monthly_table[monthly_table["month"] == month].sort_values("bin_center")
            if g.empty:
                continue
            ax.plot(
                g["bin_center"],
                g["count"],
                linewidth=style.line_width * 0.85,
                color=cmap((month - 1) / 11.0),
                label=months[month - 1],
            )
        ax.set_xlabel("Высота верха AGL, м" if style.language == "ru" else "Top height AGL, m")
        ax.set_ylabel("Число инверсий" if style.language == "ru" else "Inversion count")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=style.grid_alpha, linewidth=0.5)
        ax.legend(frameon=False, ncol=4, fontsize=style.legend_font_size - 0.5)
        if style.show_title:
            ax.set_title(title or ("Высоты инверсий: все месяцы" if style.language == "ru" else "Inversion heights: all months"))
        return _finish(fig, style)


def plot_gamma_counts_bar(
    table: pd.DataFrame,
    style: FigureStyle,
    *,
    title: str | None = None,
):
    """Столбчатый: X — γ (°C/100 м), Y — число дней."""
    data = table[table["month"] == 0].copy() if "month" in table.columns else table.copy()
    with article_rc(style):
        fig, ax = plt.subplots(figsize=(style.figure_width_in * 1.25, style.figure_height_in))
        if not data.empty:
            data = data.sort_values("bin_left")
            labels = _bin_tick_labels(data["bin_left"].to_numpy(float), data["bin_right"].to_numpy(float))
            _equal_width_bars(ax, labels, data["days"].to_numpy(float), color="#2C7FB8")
            zero_at = np.flatnonzero(data["bin_left"].to_numpy(float) >= 0)
            if zero_at.size:
                ax.axvline(float(zero_at[0]) - 0.5, color="#7F8C8D", linewidth=0.9, linestyle="--")
        ax.set_xlabel("γ, °C/100 м" if style.language == "ru" else "γ, °C/100 m")
        ax.set_ylabel("Число вертикальных интервалов" if style.language == "ru" else "Number of vertical intervals")
        ax.grid(True, axis="y", alpha=style.grid_alpha, linewidth=0.5)
        if style.show_title:
            ax.set_title(title or ("Распределение температурного градиента γ" if style.language == "ru" else "Temperature gradient γ distribution"))
        return _finish(fig, style)


def plot_gamma_counts_line(
    table: pd.DataFrame,
    style: FigureStyle,
    *,
    title: str | None = None,
):
    data = table[table["month"] == 0].copy() if "month" in table.columns else table.copy()
    with article_rc(style):
        fig, ax = plt.subplots(figsize=(style.figure_width_in, style.figure_height_in))
        if not data.empty:
            ax.plot(
                data["bin_center"],
                data["days"],
                marker="s",
                markersize=style.marker_size,
                linewidth=style.line_width,
                color="#117A65",
            )
            ax.fill_between(data["bin_center"], data["days"], alpha=0.18, color="#117A65")
        ax.axvline(0, color="#7F8C8D", linewidth=0.9, linestyle="--")
        ax.set_xlabel("γ, °C/100 м" if style.language == "ru" else "γ, °C/100 m")
        ax.set_ylabel("Число вертикальных интервалов" if style.language == "ru" else "Number of vertical intervals")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=style.grid_alpha, linewidth=0.5)
        if style.show_title:
            ax.set_title(title or ("γ — линейный вид" if style.language == "ru" else "γ — line view"))
        return _finish(fig, style)


def plot_gamma_counts_hist_step(
    layers: pd.DataFrame,
    style: FigureStyle,
    *,
    bin_edges: Iterable[float],
    title: str | None = None,
):
    """Ступенчатая гистограмма всех интервальных γ, включая отрицательные."""
    use = layers.dropna(subset=["gamma_c_per_100m"]).copy()
    with article_rc(style):
        fig, ax = plt.subplots(figsize=(style.figure_width_in, style.figure_height_in))
        if not use.empty:
            vals = use["gamma_c_per_100m"].to_numpy(float)
            edges = np.asarray(tuple(bin_edges), dtype=float)
            vals = np.clip(vals, edges[0], np.nextafter(edges[-1], -np.inf))
            ax.hist(
                vals,
                bins=edges,
                histtype="stepfilled",
                alpha=0.55,
                color="#6C3483",
                edgecolor="#4A235A",
                linewidth=1.2,
            )
        ax.axvline(0, color="#7F8C8D", linewidth=0.9, linestyle="--")
        ax.set_xlabel("γ, °C/100 м" if style.language == "ru" else "γ, °C/100 m")
        ax.set_ylabel("Число вертикальных интервалов" if style.language == "ru" else "Number of vertical intervals")
        ax.grid(True, axis="y", alpha=style.grid_alpha, linewidth=0.5)
        if style.show_title:
            ax.set_title(title or ("γ — гистограмма" if style.language == "ru" else "γ — histogram"))
        return _finish(fig, style)


def plot_gamma_by_month_box(
    layers: pd.DataFrame,
    style: FigureStyle,
    *,
    title: str | None = None,
):
    """Boxplot всех интервальных γ по месяцам (и + и −)."""
    use = layers.dropna(subset=["gamma_c_per_100m"]).copy()
    values = [use.loc[use["month"] == m, "gamma_c_per_100m"].to_numpy() for m in range(1, 13)]
    with article_rc(style):
        fig, ax = plt.subplots(figsize=(style.figure_width_in, style.figure_height_in))
        bp = ax.boxplot(values, tick_labels=_months(style), showfliers=False, patch_artist=True, widths=0.62)
        for patch in bp["boxes"]:
            patch.set_facecolor("#AED6F1")
            patch.set_alpha(0.7)
        ax.axhline(0, color="#7F8C8D", linewidth=0.9, linestyle="--")
        ax.set_xlabel("Месяц" if style.language == "ru" else "Month")
        ax.set_ylabel("γ, °C/100 м" if style.language == "ru" else "γ, °C/100 m")
        ax.grid(True, axis="y", alpha=style.grid_alpha, linewidth=0.5)
        if style.show_title:
            ax.set_title(title or ("Сезонный ход γ" if style.language == "ru" else "Seasonal cycle of γ"))
        return _finish(fig, style)


TYPE_COLORS = {"G": "#2E86C1", "E": "#E67E22", "HE": "#1E8449"}
CYCLE_COLORS = {"00": "#2E86C1", "12": "#E67E22"}
SEASON_COLORS_3D = {"DJF": "#2E86C1", "MAM": "#E67E22", "JJA": "#1E8449", "SON": "#C0392B"}

LAYER_3D_PRESETS: tuple[dict[str, str], ...] = (
    {
        "id": "top_height_depth_gamma",
        "x": "top_height_agl_m",
        "y": "depth_m",
        "z": "gamma_c_per_100m",
        "xlabel_ru": "Высота верха AGL, м",
        "ylabel_ru": "Толщина слоя, м",
        "zlabel_ru": "γ, °C/100 м",
        "xlabel_en": "Top height AGL, m",
        "ylabel_en": "Layer depth, m",
        "zlabel_en": "γ, °C/100 m",
    },
    {
        "id": "base_height_depth_deltaT",
        "x": "base_height_agl_m",
        "y": "depth_m",
        "z": "delta_t_c",
        "xlabel_ru": "Высота основания AGL, м",
        "ylabel_ru": "Толщина слоя, м",
        "zlabel_ru": "ΔT, °C",
        "xlabel_en": "Base height AGL, m",
        "ylabel_en": "Layer depth, m",
        "zlabel_en": "ΔT, °C",
    },
    {
        "id": "base_top_depth",
        "x": "base_height_agl_m",
        "y": "top_height_agl_m",
        "z": "depth_m",
        "xlabel_ru": "Высота основания AGL, м",
        "ylabel_ru": "Высота верха AGL, м",
        "zlabel_ru": "Толщина слоя, м",
        "xlabel_en": "Base height AGL, m",
        "ylabel_en": "Top height AGL, m",
        "zlabel_en": "Layer depth, m",
    },
)

REFERENCE_GAMMA_3D_PRESSURES = (850.0, 750.0, 500.0)

INVERSION_3D_AXIS_COLS = ("top_height_agl_m", "depth_m", "gamma_c_per_100m")
INVERSION_3D_AXIS_LABELS: dict[str, tuple[str, str]] = {
    "top_height_agl_m": ("Высота верха AGL, м", "Top height AGL, m"),
    "depth_m": ("Толщина слоя, м", "Layer depth, m"),
    "gamma_c_per_100m": ("γ, °C/100 м", "γ, °C/100 m"),
}
INVERSION_3D_AXIS_SUFFIX: dict[str, str] = {
    "top_height_agl_m": "Htop",
    "depth_m": "D",
    "gamma_c_per_100m": "G",
}


def plot_qc_old_vs_new(old_qc: dict, new_qc: dict, style: FigureStyle, *, title: str | None = None):
    labels_ru = ["Толщина ≤ 0", "Верх ≤ основания"]
    labels_en = ["Depth ≤ 0", "Top ≤ base"]
    labels = labels_ru if style.language == "ru" else labels_en
    old_vals = [old_qc.get("negative_depth", 0), old_qc.get("top_below_base", 0)]
    new_vals = [new_qc.get("negative_depth", 0), new_qc.get("top_below_base", 0)]
    with article_rc(style):
        fig, ax = plt.subplots(figsize=(style.figure_width_in * 0.85, style.figure_height_in))
        x = np.arange(len(labels), dtype=float)
        width = 0.36
        b1 = ax.bar(x - width / 2, old_vals, width=width, color="#2E86C1", label="Старая версия" if style.language == "ru" else "Previous")
        b2 = ax.bar(x + width / 2, new_vals, width=width, color="#E67E22", label="Исправленная" if style.language == "ru" else "Corrected")
        ax.bar_label(b1, padding=3, fontsize=style.tick_font_size)
        ax.bar_label(b2, padding=3, fontsize=style.tick_font_size)
        ax.set_xticks(x, labels)
        ax.set_ylabel("Число физических противоречий" if style.language == "ru" else "Number of physical inconsistencies")
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", alpha=style.grid_alpha, linewidth=0.5)
        ax.legend(frameon=False)
        if style.show_title:
            ax.set_title(title or ("Контроль высот инверсий: до и после исправления" if style.language == "ru" else "Inversion height QC: before and after"))
        return _finish(fig, style)


def plot_recurrence_by_type_bars(
    table: pd.DataFrame,
    style: FigureStyle,
    *,
    value_name: str,
    title: str | None = None,
):
    """Три панели G/E/HE, одинаковая ширина столбцов, все бины."""
    from .metrics import INVERSION_TYPES

    data = table[table["month"] == 0].copy() if "month" in table.columns else table.copy()
    with article_rc(style):
        fig, axes = plt.subplots(1, 3, figsize=(style.figure_width_in * 1.55, style.figure_height_in * 1.05), sharey=True)
        ycol = "recurrence_percent" if "recurrence_percent" in data.columns else "count"
        for ax, kind in zip(axes, INVERSION_TYPES):
            g = data[data["position_type"] == kind].sort_values("bin_left")
            labels = _bin_tick_labels(g["bin_left"].to_numpy(float), g["bin_right"].to_numpy(float)) if not g.empty else []
            values = g[ycol].to_numpy(float) if not g.empty else []
            bars = _equal_width_bars(ax, labels, values, color=TYPE_COLORS[kind]) if len(labels) else []
            if len(bars):
                for bar, val in zip(bars, values):
                    if val >= 0.15:
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.1f}", ha="center", va="bottom", fontsize=max(5.5, style.tick_font_size - 2.5))
            ax.set_title(_type_title(kind, style), fontsize=style.tick_font_size + 0.5)
            ax.grid(True, axis="y", alpha=style.grid_alpha, linewidth=0.4)
            ax.set_xlabel("H AGL, м" if style.language == "ru" else "H AGL, m", fontsize=style.tick_font_size)
        axes[0].set_ylabel(value_name)
        if style.show_title:
            fig.suptitle(title or "")
        fig.tight_layout()
        return fig


def plot_base_vs_top_scatter(layers: pd.DataFrame, style: FigureStyle, *, title: str | None = None):
    from .metrics import INVERSION_TYPES

    with article_rc(style):
        fig, axes = plt.subplots(1, 3, figsize=(style.figure_width_in * 1.55, style.figure_height_in * 1.05), sharey=True)
        for ax, kind in zip(axes, INVERSION_TYPES):
            g = layers[layers["position_type"] == kind]
            ax.scatter(g["base_height_agl_m"], g["top_height_agl_m"], s=6, alpha=0.22, color=TYPE_COLORS[kind], linewidths=0)
            lim = 0.0
            if not g.empty:
                lim = float(np.nanmax([g["base_height_agl_m"].max(), g["top_height_agl_m"].max()]))
            lim = max(lim, 100.0)
            ax.plot([0, lim], [0, lim], linestyle="--", color="#7F8C8D", linewidth=0.9)
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_title(_type_title(kind, style), fontsize=style.tick_font_size + 0.5)
            ax.grid(True, alpha=style.grid_alpha, linewidth=0.4)
            ax.set_xlabel("Основание AGL, м" if style.language == "ru" else "Base AGL, m")
        axes[0].set_ylabel("Верх AGL, м" if style.language == "ru" else "Top AGL, m")
        if style.show_title:
            fig.suptitle(title or ("QC геометрии: верх каждого слоя выше основания" if style.language == "ru" else "Geometry QC: top above base"))
        fig.tight_layout()
        return fig


def plot_monthly_median_iqr(table: pd.DataFrame, style: FigureStyle, *, ylabel: str, title: str | None = None):
    from .metrics import INVERSION_TYPES

    with article_rc(style):
        fig, ax = plt.subplots(figsize=(style.figure_width_in, style.figure_height_in))
        months = _months(style)
        x = np.arange(1, 13)
        for kind in INVERSION_TYPES:
            g = table[table["position_type"] == kind].set_index("month").reindex(range(1, 13))
            ax.plot(x, g["median"], marker="o", markersize=style.marker_size, linewidth=style.line_width, color=TYPE_COLORS[kind], label=_type_title(kind, style))
            ax.fill_between(x, g["q25"], g["q75"], color=TYPE_COLORS[kind], alpha=0.16, linewidth=0)
        ax.set_xticks(x, labels=months)
        ax.set_xlabel("Месяц" if style.language == "ru" else "Month")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", alpha=style.grid_alpha, linewidth=0.5)
        ax.legend(frameon=False)
        if style.show_title and title:
            ax.set_title(title)
        return _finish(fig, style)


def plot_height_median_heatmap(
    matrix: pd.DataFrame,
    style: FigureStyle,
    *,
    inversion_type: str,
    title: str | None = None,
):
    """Теплокарта медианной высоты: шкала по данному типу, не глобальная 0–5000."""
    with article_rc(style):
        fig, ax = plt.subplots(figsize=(style.figure_width_in, max(style.figure_height_in, 5.2)))
        values = matrix.to_numpy(float)
        masked = np.ma.masked_invalid(values)
        finite = values[np.isfinite(values)]
        vmax = float(np.nanpercentile(finite, 98)) if finite.size else 1.0
        vmax = max(vmax, 50.0)
        cmap = mpl.colormaps["viridis"].copy()
        cmap.set_bad(style.missing_color)
        image = ax.imshow(masked, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax, cmap=cmap)
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
        cbar = fig.colorbar(image, ax=ax, pad=0.025, fraction=0.035)
        cbar.set_label("Медианная высота верха AGL, м" if style.language == "ru" else "Median top height AGL, m")
        if style.show_title:
            ax.set_title(title or f"{_type_title(inversion_type, style)}")
        return _finish(fig, style)


def plot_top_height_cdf_by_cycle(layers: pd.DataFrame, style: FigureStyle, *, title: str | None = None):
    from .metrics import INVERSION_TYPES

    with article_rc(style):
        fig, axes = plt.subplots(1, 3, figsize=(style.figure_width_in * 1.55, style.figure_height_in * 1.05), sharey=True)
        for ax, kind in zip(axes, INVERSION_TYPES):
            g = layers[layers["position_type"] == kind]
            for cycle, color, ls in (("00", "#2E86C1", "-"), ("12", "#E67E22", "--")):
                vals = np.sort(g.loc[g["cycle"] == cycle, "top_height_agl_m"].dropna().to_numpy(float))
                if vals.size == 0:
                    continue
                y = np.linspace(0, 100, vals.size, endpoint=True)
                ax.plot(vals, y, color=color, linestyle=ls, linewidth=style.line_width, label=f"{cycle} UTC")
            ax.set_title(_type_title(kind, style), fontsize=style.tick_font_size + 0.5)
            ax.set_ylim(0, 100)
            ax.set_xlabel("Высота верха AGL, м" if style.language == "ru" else "Top height AGL, m")
            ax.grid(True, alpha=style.grid_alpha, linewidth=0.4)
        axes[0].set_ylabel("Накопленная доля слоёв, %" if style.language == "ru" else "Cumulative share of layers, %")
        axes[-1].legend(frameon=False)
        if style.show_title:
            fig.suptitle(title or ("Распределение высоты верха по срокам" if style.language == "ru" else "Top height CDF by cycle"))
        fig.tight_layout()
        return fig


def plot_annual_median_top(table: pd.DataFrame, style: FigureStyle, *, title: str | None = None):
    from .metrics import INVERSION_TYPES

    with article_rc(style):
        fig, ax = plt.subplots(figsize=(style.figure_width_in, style.figure_height_in))
        for kind in INVERSION_TYPES:
            g = table[table["position_type"] == kind].sort_values("year")
            if g.empty:
                continue
            ax.plot(g["year"], g["median"], marker="o", markersize=style.marker_size, linewidth=style.line_width, color=TYPE_COLORS[kind], label=_type_title(kind, style))
        ax.set_xlabel("Год" if style.language == "ru" else "Year")
        ax.set_ylabel("Медианная высота верха AGL, м" if style.language == "ru" else "Median top height AGL, m")
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", alpha=style.grid_alpha, linewidth=0.5)
        ax.legend(frameon=False)
        if style.show_title:
            ax.set_title(title or ("Межгодовая изменчивость высоты верха" if style.language == "ru" else "Interannual top height"))
        return _finish(fig, style)


def plot_seasonal_quantiles(layers: pd.DataFrame, style: FigureStyle, *, title: str | None = None):
    from .metrics import INVERSION_TYPES, SEASON_BY_MONTH, SEASON_ORDER

    season_colors = {"DJF": "#2E86C1", "MAM": "#E67E22", "JJA": "#1E8449", "SON": "#C0392B"}
    season_labels = SEASONS_RU if style.language == "ru" else SEASONS_EN
    with article_rc(style):
        fig, axes = plt.subplots(1, 3, figsize=(style.figure_width_in * 1.55, style.figure_height_in * 1.05), sharey=True)
        use = layers.dropna(subset=["top_height_agl_m"]).copy()
        use["season"] = use["month"].map(SEASON_BY_MONTH)
        for ax, kind in zip(axes, INVERSION_TYPES):
            g = use[use["position_type"] == kind]
            for season in SEASON_ORDER:
                vals = np.sort(g.loc[g["season"] == season, "top_height_agl_m"].to_numpy(float))
                if vals.size == 0:
                    continue
                y = np.linspace(0, 100, vals.size, endpoint=True)
                ax.plot(vals, y, color=season_colors[season], linewidth=style.line_width, label=season_labels[season])
            ax.set_title(_type_title(kind, style), fontsize=style.tick_font_size + 0.5)
            ax.set_ylim(0, 100)
            ax.set_xlabel("Высота верха AGL, м" if style.language == "ru" else "Top height AGL, m")
            ax.grid(True, alpha=style.grid_alpha, linewidth=0.4)
        axes[0].set_ylabel("Квантиль, %" if style.language == "ru" else "Quantile, %")
        axes[-1].legend(frameon=False)
        if style.show_title:
            fig.suptitle(title or ("Сезонное распределение высоты верха" if style.language == "ru" else "Seasonal top-height distribution"))
        fig.tight_layout()
        return fig


REF_LINE_COLORS = {850: "#C0392B", 750: "#D68910", 700: "#2980B9", 500: "#1E8449"}

GAMMA_YEAR_START = 1999
GAMMA_YEAR_COUNT = 27  # 1999–2025 включительно


def _gamma_count_ylabel(style: FigureStyle) -> str:
    if style.language == "ru":
        return "Число вертикальных интервалов"
    return "Number of vertical intervals"


def _gamma_interval_data_note(style: FigureStyle, *, year_label: str) -> str:
    if style.language == "ru":
        return (
            f"γ = 100·ΔT/Δz на каждой паре соседних уровней профиля (500–1000 гПа); "
            f"профили со статусом eligible_article, запуски 00/12 UTC; период: {year_label}"
        )
    return (
        f"γ = 100·ΔT/Δz for each adjacent level pair (500–1000 hPa); "
        f"eligible_article profiles, 00/12 UTC launches; period: {year_label}"
    )


def _gamma_reference_data_note(style: FigureStyle, *, year_label: str, pressures: Sequence[float]) -> str:
    ptxt = "/".join(str(int(p)) for p in pressures)
    if style.language == "ru":
        return (
            f"γ = 100·ΔT/Δz между соседними стандартными изobar ({ptxt} гПа); "
            f"eligible_article, 00/12 UTC; период: {year_label}"
        )
    return (
        f"γ = 100·ΔT/Δz between standard isobars ({ptxt} hPa); "
        f"eligible_article, 00/12 UTC; period: {year_label}"
    )


def _draw_gamma_line_on_ax(
    ax,
    data: pd.DataFrame,
    style: FigureStyle,
    *,
    color: str = "#117A65",
    label: str | None = None,
) -> None:
    if data.empty:
        return
    g = data.sort_values("bin_center")
    ax.plot(
        g["bin_center"],
        g["days"].to_numpy(float),
        marker="s",
        markersize=max(style.marker_size - 1.5, 2.5),
        linewidth=style.line_width * 0.85,
        color=color,
        label=label,
    )
    ax.fill_between(g["bin_center"], g["days"], alpha=0.12, color=color)


def plot_gamma_line_monthly_facets(
    monthly_table: pd.DataFrame,
    style: FigureStyle,
    *,
    year_label: str = "1999–2025",
    title: str | None = None,
    data_note: str | None = None,
):
    """12 панелей: распределение γ по месяцам (линейная шкала N)."""
    months = _months(style)
    note = data_note or _gamma_interval_data_note(style, year_label=year_label)
    with article_rc(style):
        fig, axes = plt.subplots(
            3, 4,
            figsize=(style.figure_width_in * 2.35, style.figure_height_in * 2.55),
            sharex=True,
        )
        for ax, month in zip(axes.ravel(), range(1, 13)):
            g = monthly_table[monthly_table["month"] == month]
            _draw_gamma_line_on_ax(ax, g, style)
            ax.axvline(0, color="#7F8C8D", linewidth=0.7, linestyle="--", alpha=0.8)
            n = int(g["days"].sum()) if not g.empty else 0
            ax.set_title(f"{months[month - 1]} (n={n:,})".replace(",", " "), fontsize=style.tick_font_size)
            ax.grid(True, alpha=style.grid_alpha, linewidth=0.35)
            ax.set_ylim(bottom=0)
        for ax in axes[2, :]:
            ax.set_xlabel("γ, °C/100 м" if style.language == "ru" else "γ, °C/100 m", fontsize=style.tick_font_size)
        for ax in axes[:, 0]:
            ax.set_ylabel(_gamma_count_ylabel(style), fontsize=style.tick_font_size)
        if style.show_title:
            fig.suptitle(
                title or (
                    f"Распределение γ по месяцам, {year_label}"
                    if style.language == "ru"
                    else f"γ distribution by month, {year_label}"
                ),
                fontsize=style.title_font_size,
            )
            fig.text(0.5, 0.015, note, ha="center", fontsize=style.tick_font_size - 1, color="#566573")
        fig.tight_layout(rect=(0, 0.04, 1, 0.96) if style.show_title else None)
        return fig


def plot_gamma_reference_line_monthly_facets(
    monthly_table: pd.DataFrame,
    style: FigureStyle,
    *,
    pressures_hpa: Sequence[float] = (850.0, 750.0, 500.0),
    year_label: str = "1999–2025",
    title: str | None = None,
    data_note: str | None = None,
):
    """12 панелей: γ на опорных изobar (850/750/500 гПа) — три линии на каждой панели."""
    months = _months(style)
    pressures = [float(p) for p in pressures_hpa]
    note = data_note or _gamma_reference_data_note(style, year_label=year_label, pressures=pressures)
    with article_rc(style):
        fig, axes = plt.subplots(
            3, 4,
            figsize=(style.figure_width_in * 2.35, style.figure_height_in * 2.65),
            sharex=True,
        )
        for ax, month in zip(axes.ravel(), range(1, 13)):
            for pressure in pressures:
                g = monthly_table[
                    (monthly_table["month"] == month)
                    & (monthly_table["pressure_hpa"] == pressure)
                ]
                label = f"{int(pressure)} гПа" if style.language == "ru" else f"{int(pressure)} hPa"
                _draw_gamma_line_on_ax(
                    ax,
                    g,
                    style,
                    color=REF_LINE_COLORS.get(int(pressure), "#34495E"),
                    label=label,
                )
            ax.axvline(0, color="#7F8C8D", linewidth=0.7, linestyle="--", alpha=0.8)
            ax.set_title(months[month - 1], fontsize=style.tick_font_size)
            ax.grid(True, alpha=style.grid_alpha, linewidth=0.35)
            ax.set_ylim(bottom=0)
        for ax in axes[2, :]:
            ax.set_xlabel("γ, °C/100 м" if style.language == "ru" else "γ, °C/100 m", fontsize=style.tick_font_size)
        for ax in axes[:, 0]:
            ax.set_ylabel(_gamma_count_ylabel(style), fontsize=style.tick_font_size)
        handles = [
            mpl.lines.Line2D(
                [0], [0],
                color=REF_LINE_COLORS.get(int(p), "#34495E"),
                linewidth=style.line_width,
                label=f"{int(p)} гПа" if style.language == "ru" else f"{int(p)} hPa",
            )
            for p in pressures
        ]
        axes.ravel()[-1].legend(handles=handles, frameon=False, fontsize=style.legend_font_size - 1, loc="upper right")
        if style.show_title:
            ptxt = "/".join(str(int(p)) for p in pressures)
            fig.suptitle(
                title or (
                    f"γ на изobar {ptxt} гПа по месяцам, {year_label}"
                    if style.language == "ru"
                    else f"γ at {ptxt} hPa isobars by month, {year_label}"
                ),
                fontsize=style.title_font_size,
            )
            fig.text(0.5, 0.015, note, ha="center", fontsize=style.tick_font_size - 1, color="#566573")
        fig.tight_layout(rect=(0, 0.04, 1, 0.96) if style.show_title else None)
        return fig


def _layer_height_depth_frame(layers: pd.DataFrame) -> pd.DataFrame:
    use = layers.dropna(subset=["top_height_agl_m", "depth_m"]).copy()
    use = use[(use["depth_m"] > 0) & (use["top_height_agl_m"] >= 0)]
    return use


def _draw_pressure_ref_lines(
    ax,
    ref_heights: Mapping[int, float],
    *,
    style: FigureStyle,
    label_in_legend: bool = True,
) -> None:
    for pressure_hpa, height_m in sorted(ref_heights.items(), reverse=True):
        color = REF_LINE_COLORS.get(int(pressure_hpa), "#7F8C8D")
        label = f"{int(pressure_hpa)} гПа" if style.language == "ru" else f"{int(pressure_hpa)} hPa"
        ax.axvline(
            float(height_m),
            color=color,
            linestyle="--",
            linewidth=style.line_width * 0.75,
            alpha=0.85,
            label=label if label_in_legend else None,
        )


def _joint_scatter_hist_layers(
    fig: plt.Figure,
    gs: GridSpec,
    use: pd.DataFrame,
    style: FigureStyle,
    *,
    scatter_type: str | None = None,
    xlabel: str,
    ylabel: str,
    title: str | None = None,
    bins_x: int = 36,
    bins_y: int = 28,
) -> None:
    """Joint scatter (высота vs толщина) с боковыми гистограммами по типам G/E/HE."""
    from .metrics import INVERSION_TYPES

    ax_sc = fig.add_subplot(gs[1:4, 0:3])
    ax_hx = fig.add_subplot(gs[0, 0:3], sharex=ax_sc)
    ax_hy = fig.add_subplot(gs[1:4, 3], sharey=ax_sc)

    if scatter_type is not None:
        scatter_df = use[use["position_type"] == scatter_type]
        color = TYPE_COLORS.get(scatter_type, "#34495E")
        ax_sc.scatter(
            scatter_df["top_height_agl_m"],
            scatter_df["depth_m"],
            s=8,
            alpha=0.42,
            color=color,
            linewidths=0,
        )
    elif "position_type" in use.columns:
        for kind in INVERSION_TYPES:
            g = use[use["position_type"] == kind]
            if g.empty:
                continue
            ax_sc.scatter(
                g["top_height_agl_m"],
                g["depth_m"],
                s=8,
                alpha=0.42,
                color=TYPE_COLORS[kind],
                linewidths=0,
                label=_type_title(kind, style),
            )
    else:
        ax_sc.scatter(
            use["top_height_agl_m"],
            use["depth_m"],
            s=6,
            alpha=0.28,
            color="#34495E",
            linewidths=0,
        )

    ax_sc.set_xlabel(xlabel)
    ax_sc.set_ylabel(ylabel)
    ax_sc.grid(True, alpha=style.grid_alpha, linewidth=0.4)
    if scatter_type is None and "position_type" in use.columns:
        ax_sc.legend(frameon=False, fontsize=style.legend_font_size - 1, loc="upper right")

    x_edges = np.histogram_bin_edges(use["top_height_agl_m"].to_numpy(float), bins=bins_x)
    y_edges = np.histogram_bin_edges(use["depth_m"].to_numpy(float), bins=bins_y)
    count_label = "Число" if style.language == "ru" else "Count"

    for kind in INVERSION_TYPES:
        g = use[use["position_type"] == kind] if "position_type" in use.columns else use.iloc[0:0]
        if g.empty:
            continue
        label = _type_title(kind, style)
        ax_hx.hist(
            g["top_height_agl_m"],
            bins=x_edges,
            histtype="stepfilled",
            alpha=0.38,
            color=TYPE_COLORS[kind],
            label=label,
        )
        ax_hy.hist(
            g["depth_m"],
            bins=y_edges,
            orientation="horizontal",
            histtype="stepfilled",
            alpha=0.38,
            color=TYPE_COLORS[kind],
        )

    ax_hx.set_ylabel(count_label)
    ax_hy.set_xlabel(count_label)
    plt.setp(ax_hx.get_xticklabels(), visible=False)
    plt.setp(ax_hy.get_yticklabels(), visible=False)
    ax_hx.legend(frameon=False, fontsize=style.legend_font_size - 1, loc="upper right", ncol=1)
    if title:
        ax_sc.set_title(title, fontsize=style.tick_font_size + 0.5)


def _joint_scatter_hist(
    fig: plt.Figure,
    gs: GridSpec,
    x: np.ndarray,
    y: np.ndarray,
    *,
    ref_heights: Mapping[int, float] | None,
    style: FigureStyle,
    color_values: np.ndarray | None = None,
    point_colors: np.ndarray | None = None,
    xlabel: str,
    ylabel: str,
    title: str | None = None,
    bins_x: int = 36,
    bins_y: int = 28,
    show_legend: bool = True,
    ylim: tuple[float, float] | None = None,
) -> None:
    ax_sc = fig.add_subplot(gs[1:4, 0:3])
    ax_hx = fig.add_subplot(gs[0, 0:3], sharex=ax_sc)
    ax_hy = fig.add_subplot(gs[1:4, 3], sharey=ax_sc)

    if point_colors is not None and len(point_colors) == len(x):
        ax_sc.scatter(x, y, c=point_colors, s=8, alpha=0.45, linewidths=0)
    elif color_values is not None and len(color_values) == len(x):
        sc = ax_sc.scatter(
            x, y, c=color_values, s=8, alpha=0.45, cmap="coolwarm",
            linewidths=0, vmin=np.nanpercentile(color_values, 5),
            vmax=np.nanpercentile(color_values, 95),
        )
        fig.colorbar(sc, ax=ax_sc, pad=0.02, fraction=0.035, label="γ, °C/100 м")
    else:
        ax_sc.scatter(x, y, s=6, alpha=0.28, color="#34495E", linewidths=0)

    if ref_heights:
        _draw_pressure_ref_lines(ax_sc, ref_heights, style=style, label_in_legend=show_legend)
    ax_sc.set_xlabel(xlabel)
    ax_sc.set_ylabel(ylabel)
    if ylim is not None:
        ax_sc.set_ylim(ylim)
    ax_sc.grid(True, alpha=style.grid_alpha, linewidth=0.4)
    if show_legend and ref_heights:
        ax_sc.legend(frameon=False, fontsize=style.legend_font_size - 1, loc="upper right")

    count_label = "Число" if style.language == "ru" else "Count"
    ax_hx.hist(x, bins=bins_x, color="#5D6D7E", alpha=0.75)
    ax_hy.hist(y, bins=bins_y, orientation="horizontal", color="#5D6D7E", alpha=0.75)
    ax_hx.set_ylabel(count_label)
    ax_hy.set_xlabel(count_label)
    plt.setp(ax_hx.get_xticklabels(), visible=False)
    plt.setp(ax_hy.get_yticklabels(), visible=False)
    if title:
        ax_sc.set_title(title, fontsize=style.tick_font_size + 0.5)


def plot_top_height_vs_depth_joint(
    layers: pd.DataFrame,
    style: FigureStyle,
    *,
    inversion_type: str | None = None,
    title: str | None = None,
):
    """Scatter + marginal histograms: X — высота верха AGL, Y — толщина слоя (полные профили).

    inversion_type: G / E / HE — только этот тип в облаке точек; боковые гистограммы всегда по всем трём.
    """
    use = _layer_height_depth_frame(layers)
    xlabel = "Высота верха над поверхностью, м" if style.language == "ru" else "Top height AGL, m"
    ylabel = "Толщина инверсионного слоя, м" if style.language == "ru" else "Inversion layer depth, m"
    if inversion_type is not None:
        type_title = _type_title(inversion_type, style)
        default_title = (
            f"Высота верха vs толщина — {type_title}"
            if style.language == "ru"
            else f"Top height vs depth — {type_title}"
        )
    else:
        default_title = (
            "Высота верха vs толщина инверсии (облако точек + гистограммы)"
            if style.language == "ru"
            else "Top height vs inversion depth (scatter + histograms)"
        )
    with article_rc(style):
        fig = plt.figure(figsize=(style.figure_width_in * 1.05, style.figure_height_in * 1.15))
        gs = GridSpec(4, 4, figure=fig, wspace=0.06, hspace=0.06)
        _joint_scatter_hist_layers(
            fig,
            gs,
            use,
            style,
            scatter_type=inversion_type,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title or default_title,
        )
        if style.show_title:
            suptitle = title or (
                f"Инверсии Алдана: {_type_title(inversion_type, style)}"
                if inversion_type is not None
                else "Инверсии Алдана: высота и толщина слоя"
            )
            fig.suptitle(suptitle, fontsize=style.title_font_size, y=1.02)
        fig.subplots_adjust(left=0.1, right=0.92, top=0.94, bottom=0.1)
        return fig


def plot_top_height_vs_depth_monthly_facets(
    layers: pd.DataFrame,
    style: FigureStyle,
    *,
    title: str | None = None,
):
    """12 панелей (месяцы): scatter высота vs толщина — только полные данные слоя."""
    use = _layer_height_depth_frame(layers)
    months = _months(style)
    with article_rc(style):
        fig, axes = plt.subplots(
            3, 4,
            figsize=(style.figure_width_in * 2.2, style.figure_height_in * 2.35),
            sharex=True,
            sharey=True,
        )
        xlabel = "H верха AGL, м" if style.language == "ru" else "Top H AGL, m"
        ylabel = "Толщина, м" if style.language == "ru" else "Depth, m"
        for ax, month in zip(axes.ravel(), range(1, 13)):
            g = use[use["month"] == month]
            if not g.empty:
                ax.scatter(
                    g["top_height_agl_m"],
                    g["depth_m"],
                    s=5,
                    alpha=0.35,
                    c=g["position_type"].map(TYPE_COLORS) if "position_type" in g.columns else "#34495E",
                    linewidths=0,
                )
            ax.set_title(f"{months[month - 1]} (n={len(g)})", fontsize=style.tick_font_size)
            ax.grid(True, alpha=style.grid_alpha, linewidth=0.35)
        for ax in axes[2, :]:
            ax.set_xlabel(xlabel)
        for ax in axes[:, 0]:
            ax.set_ylabel(ylabel)
        from .metrics import INVERSION_TYPES

        last = axes.ravel()[-1]
        type_handles = [
            mpl.lines.Line2D([0], [0], marker="o", color="w", markerfacecolor=TYPE_COLORS[k], markersize=6, label=_type_title(k, style))
            for k in INVERSION_TYPES
        ]
        last.legend(handles=type_handles, frameon=False, fontsize=style.legend_font_size - 1, loc="upper right")
        if style.show_title:
            fig.suptitle(
                title or (
                    "Высота верха vs толщина инверсии по месяцам (все годы)"
                    if style.language == "ru"
                    else "Top height vs depth by month (all years)"
                ),
                fontsize=style.title_font_size,
            )
        fig.tight_layout()
        return fig


def plot_top_height_vs_depth_boxplots(
    layers: pd.DataFrame,
    style: FigureStyle,
    *,
    title: str | None = None,
):
    """Boxplot: толщина и высота верха по месяцам."""
    use = _layer_height_depth_frame(layers)
    months = _months(style)
    depth_by_month = [use.loc[use["month"] == m, "depth_m"].dropna().to_numpy(float) for m in range(1, 13)]
    height_by_month = [use.loc[use["month"] == m, "top_height_agl_m"].dropna().to_numpy(float) for m in range(1, 13)]
    with article_rc(style):
        fig, axes = plt.subplots(1, 2, figsize=(style.figure_width_in * 1.65, style.figure_height_in * 1.05))
        for ax, data, ylabel in zip(
            axes,
            (depth_by_month, height_by_month),
            (
                "Толщина слоя, м" if style.language == "ru" else "Layer depth, m",
                "Высота верха AGL, м" if style.language == "ru" else "Top height AGL, m",
            ),
        ):
            bp = ax.boxplot(data, tick_labels=months, showfliers=False, patch_artist=True, widths=0.62)
            for patch in bp["boxes"]:
                patch.set_facecolor("#AED6F1")
                patch.set_alpha(0.75)
            for median in bp["medians"]:
                median.set_color("#1B4F72")
                median.set_linewidth(style.line_width)
            ax.set_xlabel("Месяц" if style.language == "ru" else "Month")
            ax.set_ylabel(ylabel)
            ax.grid(True, axis="y", alpha=style.grid_alpha, linewidth=0.5)
        if style.show_title:
            fig.suptitle(
                title or (
                    "Сезонное распределение толщины и высоты инверсий"
                    if style.language == "ru"
                    else "Seasonal distribution of depth and top height"
                ),
                fontsize=style.title_font_size,
            )
        fig.tight_layout()
        return fig


def plot_top_height_depth_gamma_3d(
    layers: pd.DataFrame,
    style: FigureStyle,
    *,
    inversion_type: str,
    title: str | None = None,
    elev: float = 24.0,
    azim: float = -58.0,
):
    """3D scatter: X — высота верха AGL, Y — толщина слоя, Z — γ слоя (по типу G/E/HE)."""
    from .metrics import INVERSION_TYPES

    if inversion_type not in INVERSION_TYPES:
        raise ValueError(f"inversion_type must be one of {INVERSION_TYPES}")
    preset = LAYER_3D_PRESETS[0]
    return plot_layers_scatter_3d(
        layers,
        style,
        x_col=preset["x"],
        y_col=preset["y"],
        z_col=preset["z"],
        xlabel=preset["xlabel_ru"] if style.language == "ru" else preset["xlabel_en"],
        ylabel=preset["ylabel_ru"] if style.language == "ru" else preset["ylabel_en"],
        zlabel=preset["zlabel_ru"] if style.language == "ru" else preset["zlabel_en"],
        filter_col="position_type",
        filter_value=inversion_type,
        color=TYPE_COLORS.get(inversion_type, "#34495E"),
        title=title or f"3D: {_type_title(inversion_type, style)}",
        note=(
            "Слои инверсии (eligible_article); оси — геометрия и γ слоя"
            if style.language == "ru"
            else "Inversion layers (eligible_article); geometry and layer γ"
        ),
        elev=elev,
        azim=azim,
    )


def _layer_3d_frame(layers: pd.DataFrame, x_col: str, y_col: str, z_col: str) -> pd.DataFrame:
    use = layers.dropna(subset=[x_col, y_col, z_col]).copy()
    if "depth_m" in (x_col, y_col, z_col):
        use = use[use["depth_m"] > 0]
    for col in (x_col, y_col):
        if col.endswith("_agl_m"):
            use = use[use[col] >= 0]
    return use


def _z_clip_limits(z: np.ndarray) -> tuple[float, float]:
    if z.size == 0:
        return 0.0, 1.0
    z_lo = float(np.nanpercentile(z, 1))
    z_hi = float(np.nanpercentile(z, 99))
    if z_hi <= z_lo:
        z_hi = z_lo + 1.0
    return z_lo, z_hi


def _axis_values_and_limits(col: str, values: np.ndarray) -> tuple[np.ndarray, float, float]:
    if col == "gamma_c_per_100m":
        lo = min(float(np.nanpercentile(values, 1)), 0.0) if values.size else 0.0
        hi = max(float(np.nanpercentile(values, 99)), 3.0) if values.size else 3.0
        return np.clip(values, lo, hi), lo, hi
    if col == "month":
        return values, 1.0, 12.0
    lo, hi = _z_clip_limits(values)
    return np.clip(values, lo, hi), lo, hi


def _axis_limit_hi(col: str, lo: float, hi: float) -> float:
    if col == "month":
        return hi
    return hi * 1.05


def _filter_layers_by_gamma_sign(
    layers: pd.DataFrame,
    gamma_sign: Literal["positive", "negative"] | None,
) -> pd.DataFrame:
    if gamma_sign is None or "gamma_c_per_100m" not in layers.columns:
        return layers
    if gamma_sign == "positive":
        return layers[layers["gamma_c_per_100m"] > 0]
    return layers[layers["gamma_c_per_100m"] < 0]


def _inversion_3d_axis_label(col: str, style: FigureStyle) -> str:
    ru, en = INVERSION_3D_AXIS_LABELS[col]
    return ru if style.language == "ru" else en


def plot_layers_scatter_3d(
    layers: pd.DataFrame,
    style: FigureStyle,
    *,
    x_col: str,
    y_col: str,
    z_col: str,
    xlabel: str,
    ylabel: str,
    zlabel: str,
    filter_col: str | None = None,
    filter_value: str | None = None,
    color: str = "#34495E",
    title: str | None = None,
    note: str | None = None,
    elev: float = 24.0,
    azim: float = -58.0,
    gamma_sign: Literal["positive", "negative"] | None = None,
):
    """Универсальный 3D scatter по слоям инверсии."""
    use = _filter_layers_by_gamma_sign(layers, gamma_sign)
    use = _layer_3d_frame(use, x_col, y_col, z_col)
    if filter_col is not None and filter_value is not None:
        use = use[use[filter_col].astype(str) == str(filter_value)]
    if use.empty:
        with article_rc(style):
            fig = plt.figure(figsize=(style.figure_width_in * 1.15, style.figure_height_in * 1.15))
            ax = fig.add_subplot(111, projection="3d")
            ax.text2D(0.5, 0.5, "Нет данных" if style.language == "ru" else "No data", transform=ax.transAxes, ha="center")
            return fig

    x = use[x_col].to_numpy(float)
    y = use[y_col].to_numpy(float)
    z = use[z_col].to_numpy(float)
    x_plot, x_lo, x_hi = _axis_values_and_limits(x_col, x)
    y_plot, y_lo, y_hi = _axis_values_and_limits(y_col, y)
    z_plot, z_lo, z_hi = _axis_values_and_limits(z_col, z)

    with article_rc(style):
        fig = plt.figure(figsize=(style.figure_width_in * 1.25, style.figure_height_in * 1.2))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x_plot, y_plot, z_plot, c=color, s=4, alpha=0.38, linewidths=0, depthshade=True)
        ax.set_xlabel(xlabel, labelpad=8)
        ax.set_ylabel(ylabel, labelpad=8)
        ax.set_zlabel(zlabel, labelpad=8)
        ax.set_xlim(x_lo, _axis_limit_hi(x_col, x_lo, x_hi))
        ax.set_ylim(y_lo, _axis_limit_hi(y_col, y_lo, y_hi))
        ax.set_zlim(z_lo, _axis_limit_hi(z_col, z_lo, z_hi))
        ax.view_init(elev=elev, azim=azim)
        ax.grid(True, alpha=style.grid_alpha)
        if style.show_title and title:
            fig.suptitle(title, fontsize=style.title_font_size, y=0.98)
        if note:
            fig.text(0.5, 0.02, note, ha="center", fontsize=style.tick_font_size - 1, color="#566573")
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.08, top=0.92)
        return fig


def plot_layers_scatter_3d_combined(
    layers: pd.DataFrame,
    style: FigureStyle,
    *,
    x_col: str,
    y_col: str,
    z_col: str,
    xlabel: str,
    ylabel: str,
    zlabel: str,
    group_col: str = "position_type",
    colors: Mapping[str, str] | None = None,
    title: str | None = None,
    note: str | None = None,
    elev: float = 24.0,
    azim: float = -58.0,
):
    """3D scatter: все группы на одном графике (разные цвета)."""
    from .metrics import SEASON_BY_MONTH

    palette = dict(colors or TYPE_COLORS)
    use = _layer_3d_frame(layers, x_col, y_col, z_col)
    if group_col == "season" and "season" not in use.columns:
        use = use.copy()
        use["season"] = use["month"].map(SEASON_BY_MONTH)

    with article_rc(style):
        fig = plt.figure(figsize=(style.figure_width_in * 1.25, style.figure_height_in * 1.2))
        ax = fig.add_subplot(111, projection="3d")
        z_lo, z_hi = 0.0, 1.0
        for key, g in use.groupby(group_col, sort=False):
            if g.empty:
                continue
            z = g[z_col].to_numpy(float)
            if z_col == "gamma_c_per_100m":
                z_lo = min(float(np.nanpercentile(z, 1)), 0.0)
                z_hi = max(float(np.nanpercentile(z, 99)), 3.0)
                z_plot = np.clip(z, z_lo, z_hi)
            else:
                z_lo, z_hi = _z_clip_limits(z)
                z_plot = np.clip(z, z_lo, z_hi)
            ax.scatter(
                g[x_col], g[y_col], z_plot,
                c=palette.get(str(key), "#34495E"),
                s=4, alpha=0.35, linewidths=0, depthshade=True, label=str(key),
            )
        ax.set_xlabel(xlabel, labelpad=8)
        ax.set_ylabel(ylabel, labelpad=8)
        ax.set_zlabel(zlabel, labelpad=8)
        ax.set_zlim(z_lo, z_hi * 1.05)
        ax.view_init(elev=elev, azim=azim)
        ax.grid(True, alpha=style.grid_alpha)
        ax.legend(frameon=False, fontsize=style.legend_font_size - 1, loc="upper left")
        if style.show_title and title:
            fig.suptitle(title, fontsize=style.title_font_size, y=0.98)
        if note:
            fig.text(0.5, 0.02, note, ha="center", fontsize=style.tick_font_size - 1, color="#566573")
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.08, top=0.92)
        return fig


def plot_reference_gamma_scatter_3d(
    gammas: pd.DataFrame,
    style: FigureStyle,
    *,
    pressure_hpa: float | None = None,
    title: str | None = None,
    elev: float = 24.0,
    azim: float = -58.0,
):
    """3D: высота AGL × γ × месяц (опорные изobar, не слои инверсии)."""
    use = gammas.dropna(subset=["height_agl_m", "gamma_c_per_100m", "month"]).copy()
    if pressure_hpa is not None:
        use = use[use["pressure_hpa"] == float(pressure_hpa)]
    xlabel = "Высота AGL, м" if style.language == "ru" else "Height AGL, m"
    ylabel = "γ, °C/100 м" if style.language == "ru" else "γ, °C/100 m"
    zlabel = "Месяц" if style.language == "ru" else "Month"
    note = (
        "γ между стандартными изobar; eligible_article, 00/12 UTC"
        if style.language == "ru"
        else "γ between standard isobars; eligible_article, 00/12 UTC"
    )

    if pressure_hpa is None:
        with article_rc(style):
            fig = plt.figure(figsize=(style.figure_width_in * 1.25, style.figure_height_in * 1.2))
            ax = fig.add_subplot(111, projection="3d")
            z = use["gamma_c_per_100m"].to_numpy(float)
            z_lo = min(float(np.nanpercentile(z, 1)), 0.0) if z.size else 0.0
            z_hi = max(float(np.nanpercentile(z, 99)), 3.0) if z.size else 3.0
            for p in sorted(use["pressure_hpa"].dropna().unique()):
                g = use[use["pressure_hpa"] == p]
                ax.scatter(
                    g["height_agl_m"],
                    np.clip(g["gamma_c_per_100m"], z_lo, z_hi),
                    g["month"],
                    c=REF_LINE_COLORS.get(int(round(p)), "#34495E"),
                    s=3, alpha=0.35, linewidths=0, label=f"{int(round(p))} гПа",
                )
            ax.set_xlabel(xlabel, labelpad=8)
            ax.set_ylabel(ylabel, labelpad=8)
            ax.set_zlabel(zlabel, labelpad=8)
            ax.set_zlim(1, 12)
            ax.view_init(elev=elev, azim=azim)
            ax.legend(frameon=False, fontsize=style.legend_font_size - 1)
            if style.show_title:
                fig.suptitle(
                    title or ("3D: γ на опорных изobar (все уровни)" if style.language == "ru" else "3D: γ at reference isobars"),
                    fontsize=style.title_font_size, y=0.98,
                )
            fig.text(0.5, 0.02, note, ha="center", fontsize=style.tick_font_size - 1, color="#566573")
            fig.subplots_adjust(left=0.02, right=0.98, bottom=0.08, top=0.92)
            return fig

    color = REF_LINE_COLORS.get(int(pressure_hpa), "#34495E")
    framed = use.rename(columns={"height_agl_m": "top_height_agl_m"}).copy()
    framed["month"] = framed["month"].astype(float)
    return plot_layers_scatter_3d(
        framed,
        style,
        x_col="top_height_agl_m",
        y_col="gamma_c_per_100m",
        z_col="month",
        xlabel=xlabel,
        ylabel=ylabel,
        zlabel=zlabel,
        color=color,
        title=title or f"3D: γ на {int(pressure_hpa)} гПа",
        note=note,
        elev=elev,
        azim=azim,
    )


def build_scatter_3d_figure_specs(
    layers: pd.DataFrame,
    reference_gammas: pd.DataFrame,
    style: FigureStyle,
) -> list[tuple[str, object]]:
    """Список (относительный путь, builder) для всех 3D-графиков библиотеки."""
    from .metrics import INVERSION_TYPES

    layers_work = layers.copy()
    layers_work["season"] = layers_work["month"].map(SEASON_BY_MONTH)
    specs: list[tuple[str, object]] = []

    def _labels(preset: dict[str, str]) -> tuple[str, str, str]:
        if style.language == "ru":
            return preset["xlabel_ru"], preset["ylabel_ru"], preset["zlabel_ru"]
        return preset["xlabel_en"], preset["ylabel_en"], preset["zlabel_en"]

    groupings: list[tuple[str, str, tuple, Mapping[str, str], object]] = [
        ("by_inversion_type", "position_type", INVERSION_TYPES, TYPE_COLORS, lambda v: _type_title(v, style)),
        (
            "by_cycle",
            "cycle",
            tuple(sorted(layers_work["cycle"].dropna().astype(str).unique())) or ("00", "12"),
            CYCLE_COLORS,
            lambda v: f"{v} UTC",
        ),
        (
            "by_season",
            "season",
            SEASON_ORDER,
            SEASON_COLORS_3D,
            lambda v: (SEASONS_RU if style.language == "ru" else SEASONS_EN)[v],
        ),
    ]

    for preset in LAYER_3D_PRESETS:
        pid = preset["id"]
        xlabel, ylabel, zlabel = _labels(preset)
        for folder, col, values, palette, label_fn in groupings:
            for val in values:
                label = label_fn(val)
                path = f"scatter_3d/{folder}/{pid}_{val}"

                def _make(p=preset, c=col, v=val, pal=palette, lbl=label, xl=xlabel, yl=ylabel, zl=zlabel):
                    return plot_layers_scatter_3d(
                        layers_work, style,
                        x_col=p["x"], y_col=p["y"], z_col=p["z"],
                        xlabel=xl, ylabel=yl, zlabel=zl,
                        filter_col=c, filter_value=str(v),
                        color=pal.get(str(v), "#34495E"),
                        title=f"3D: {p['id']} ({lbl})",
                    )

                specs.append((path, _make))
        specs.append(
            (
                f"scatter_3d/all_types_combined/{pid}_G_E_HE",
                lambda p=preset, xl=xlabel, yl=ylabel, zl=zlabel: plot_layers_scatter_3d_combined(
                    layers_work, style,
                    x_col=p["x"], y_col=p["y"], z_col=p["z"],
                    xlabel=xl, ylabel=yl, zlabel=zl,
                    group_col="position_type", colors=TYPE_COLORS,
                    title=f"3D: {p['id']} (G+E+HE)",
                ),
            )
        )

    ref_data = reference_gammas
    specs.append(
        (
            "scatter_3d/reference_gamma/height_gamma_month_all_pressures",
            lambda: plot_reference_gamma_scatter_3d(
                ref_data, style,
                title="3D: γ на опорных изobar (850/750/500 гПа)" if style.language == "ru" else "3D: γ at reference isobars",
            ),
        )
    )
    for pressure in REFERENCE_GAMMA_3D_PRESSURES:
        specs.append(
            (
                f"scatter_3d/reference_gamma/height_gamma_month_{int(pressure)}hPa",
                lambda p=pressure: plot_reference_gamma_scatter_3d(ref_data, style, pressure_hpa=p, title=f"3D: γ на {int(p)} гПа"),
            )
        )
    return specs


def build_inversion_scatter_3d_figure_specs(
    layers: pd.DataFrame,
    style: FigureStyle,
) -> list[tuple[str, object]]:
    """3D-графики инверсий (G/E/HE): знак γ и все перестановки осей."""
    from .metrics import INVERSION_TYPES

    specs: list[tuple[str, object]] = []
    sign_meta = (
        ("pos", "positive", "γ > 0", "γ > 0"),
        ("neg", "negative", "γ < 0", "γ < 0"),
    )

    for inv_type in INVERSION_TYPES:
        specs.append(
            (
                f"scatter_3d/top_height_depth_gamma_3d_{inv_type}",
                lambda t=inv_type: plot_top_height_depth_gamma_3d(layers, style, inversion_type=t),
            )
        )
        for sign_tag, sign_filter, sign_ru, sign_en in sign_meta:
            sign_label = sign_ru if style.language == "ru" else sign_en
            for x_col, y_col, z_col in permutations(INVERSION_3D_AXIS_COLS):
                axis_suffix = "_".join(INVERSION_3D_AXIS_SUFFIX[c] for c in (x_col, y_col, z_col))
                path = f"scatter_3d/top_height_depth_gamma_3d_{inv_type}_{sign_tag}_{axis_suffix}"
                xlabel = _inversion_3d_axis_label(x_col, style)
                ylabel = _inversion_3d_axis_label(y_col, style)
                zlabel = _inversion_3d_axis_label(z_col, style)
                axis_tag = " × ".join(
                    INVERSION_3D_AXIS_SUFFIX[c] for c in (x_col, y_col, z_col)
                )

                def _make(
                    t=inv_type,
                    sf=sign_filter,
                    sl=sign_label,
                    xc=x_col,
                    yc=y_col,
                    zc=z_col,
                    xl=xlabel,
                    yl=ylabel,
                    zl=zlabel,
                    at=axis_tag,
                ):
                    return plot_layers_scatter_3d(
                        layers,
                        style,
                        x_col=xc,
                        y_col=yc,
                        z_col=zc,
                        xlabel=xl,
                        ylabel=yl,
                        zlabel=zl,
                        filter_col="position_type",
                        filter_value=t,
                        color=TYPE_COLORS.get(t, "#34495E"),
                        gamma_sign=sf,
                        title=(
                            f"3D: {_type_title(t, style)} ({sl}) — {at}"
                            if style.language == "ru"
                            else f"3D: {_type_title(t, style)} ({sl}) — {at}"
                        ),
                        note=(
                            "Слои инверсии; оси — все перестановки геометрии и γ"
                            if style.language == "ru"
                            else "Inversion layers; all axis permutations of geometry and γ"
                        ),
                    )

                specs.append((path, _make))
    return specs


# --- Расширенные 3D-визуализации: bar3d / voxels / GIF / 2D-проекции ---

_EXTRA_3D_X = "top_height_agl_m"
_EXTRA_3D_Y = "depth_m"
_EXTRA_3D_Z = "gamma_c_per_100m"

GrowAxis = Literal["htop", "depth", "gamma", "month"]
_GROW_AXES: tuple[GrowAxis, ...] = ("htop", "depth", "gamma", "month")
_GROW_AXIS_COLS: dict[str, str] = {
    "htop": "top_height_agl_m",
    "depth": "depth_m",
    "gamma": "gamma_c_per_100m",
    "month": "month",
}
_GROW_AXIS_LABELS: dict[str, tuple[str, str]] = {
    "htop": ("Htop", "Htop"),
    "depth": ("толщина", "depth"),
    "gamma": ("γ", "γ"),
    "month": ("месяц", "month"),
}


def _percentile_edges(values: np.ndarray, n_bins: int, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.linspace(0.0, 1.0, n_bins + 1)
    lo = float(np.nanpercentile(vals, lo_pct))
    hi = float(np.nanpercentile(vals, hi_pct))
    if hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, n_bins + 1)


def _bin_edges_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    nx: int = 12,
    ny: int = 10,
    nz: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        _percentile_edges(x, nx),
        _percentile_edges(y, ny),
        _percentile_edges(z, nz),
    )


def _hist3d_counts(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    z_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """3D-гистограмма: counts shape (nx, ny, nz) и центры бинов."""
    counts, _ = np.histogramdd(
        np.column_stack([x, y, z]),
        bins=(x_edges, y_edges, z_edges),
    )
    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    yc = 0.5 * (y_edges[:-1] + y_edges[1:])
    zc = 0.5 * (z_edges[:-1] + z_edges[1:])
    return counts.astype(float), xc, yc, zc


def _prepare_extra_3d_frame(
    layers: pd.DataFrame,
    *,
    inversion_type: str | None = None,
) -> pd.DataFrame:
    use = _layer_3d_frame(layers, _EXTRA_3D_X, _EXTRA_3D_Y, _EXTRA_3D_Z)
    if inversion_type is not None:
        use = use[use["position_type"].astype(str) == str(inversion_type)]
    return use


def _empty_3d_fig(style: FigureStyle, title: str | None = None):
    with article_rc(style):
        fig = plt.figure(figsize=(style.figure_width_in * 1.25, style.figure_height_in * 1.2))
        ax = fig.add_subplot(111, projection="3d")
        ax.text2D(
            0.5, 0.5,
            "Нет данных" if style.language == "ru" else "No data",
            transform=ax.transAxes, ha="center",
        )
        if style.show_title and title:
            fig.suptitle(title, fontsize=style.title_font_size, y=0.98)
        return fig


def _hex_to_rgba(hex_color: str, alpha: float) -> tuple[float, float, float, float]:
    rgb = mpl.colors.to_rgb(hex_color)
    return (rgb[0], rgb[1], rgb[2], float(np.clip(alpha, 0.05, 1.0)))


def _grow_axis_col(grow_axis: str) -> str:
    if grow_axis not in _GROW_AXIS_COLS:
        raise ValueError(f"grow_axis must be one of {tuple(_GROW_AXIS_COLS)}")
    return _GROW_AXIS_COLS[grow_axis]


def _grow_axis_label(grow_axis: str, style: FigureStyle) -> str:
    ru, en = _GROW_AXIS_LABELS.get(grow_axis, (grow_axis, grow_axis))
    return ru if style.language == "ru" else en


def _grow_axis_n_frames(grow_axis: str, n_frames: int) -> int:
    return 12 if grow_axis == "month" else int(n_frames)


def _frame_thresholds(values: np.ndarray, n_frames: int, *, grow_axis: str) -> np.ndarray:
    if grow_axis == "month":
        return np.arange(1, 13, dtype=float)
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.linspace(0.0, 1.0, n_frames)
    lo = float(np.nanpercentile(vals, 1))
    hi = float(np.nanpercentile(vals, 99))
    if hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, n_frames)


def _fade_alphas(
    values: np.ndarray,
    threshold: float,
    *,
    alpha_max: float = 0.85,
    k: float = 3.0,
) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    revealed = np.isfinite(vals) & (vals <= threshold)
    alphas = np.zeros(vals.shape, dtype=float)
    if not np.any(revealed):
        return alphas
    vmin = float(np.nanmin(vals[revealed]))
    span = max(float(threshold) - vmin, 1e-9)
    age = (float(threshold) - vals[revealed]) / span
    alphas[revealed] = alpha_max * np.exp(-k * age)
    return alphas


def _subsample_frame(use: pd.DataFrame, max_n: int, seed: int = 0) -> pd.DataFrame:
    if max_n <= 0 or len(use) <= max_n:
        return use
    return use.sample(n=int(max_n), random_state=seed)


def _prepare_buildup_frame(
    layers: pd.DataFrame,
    *,
    inversion_type: str | None,
    grow_axis: str,
) -> pd.DataFrame:
    use = _prepare_extra_3d_frame(layers, inversion_type=inversion_type)
    if grow_axis == "month":
        if "month" not in use.columns:
            return use.iloc[0:0].copy()
        use = use.dropna(subset=["month"])
        month = use["month"].to_numpy(float)
        use = use[(month >= 1) & (month <= 12)]
    return use


def _extra_3d_cmap(inversion_type: str | None) -> str:
    return {"G": "Blues", "E": "Oranges", "HE": "Greens"}.get(str(inversion_type or ""), "viridis")


def _extra_3d_axis_labels(style: FigureStyle) -> tuple[str, str, str]:
    if style.language == "ru":
        return "Htop AGL, м", "Толщина, м", "γ, °C/100 м"
    return "Htop AGL, m", "Depth, m", "γ, °C/100 m"


def _style_extra_3d_scatter_ax(
    ax,
    style: FigureStyle,
    *,
    x_lim: tuple[float, float],
    y_lim: tuple[float, float],
    z_lim: tuple[float, float],
    elev: float,
    azim: float,
) -> None:
    xlabel, ylabel, zlabel = _extra_3d_axis_labels(style)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_zlim(*z_lim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.view_init(elev=elev, azim=azim)


def _style_voxel_ax(
    ax,
    style: FigureStyle,
    *,
    nx: int,
    ny: int,
    nz: int,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    z_edges: np.ndarray,
    elev: float,
    azim: float,
) -> None:
    xlabel, ylabel, zlabel = _extra_3d_axis_labels(style)
    ax.set_xticks(np.linspace(0, nx, 5))
    ax.set_yticks(np.linspace(0, ny, 5))
    ax.set_zticks(np.linspace(0, nz, 5))
    ax.set_xticklabels([f"{v:.0f}" for v in np.linspace(x_edges[0], x_edges[-1], 5)])
    ax.set_yticklabels([f"{v:.0f}" for v in np.linspace(y_edges[0], y_edges[-1], 5)])
    ax.set_zticklabels([f"{v:.1f}" for v in np.linspace(z_edges[0], z_edges[-1], 5)])
    ax.set_xlabel(xlabel, labelpad=8)
    ax.set_ylabel(ylabel, labelpad=8)
    ax.set_zlabel(zlabel, labelpad=8)
    ax.view_init(elev=elev, azim=azim)


def _as_gif_path(output_path: Path) -> Path:
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".gif":
        output_path = Path(str(output_path) + ".gif")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _htop_depth_mean_gamma_grid(
    use: pd.DataFrame,
    nx: int = 24,
    ny: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = use[_EXTRA_3D_X].to_numpy(float)
    y = use[_EXTRA_3D_Y].to_numpy(float)
    z = use[_EXTRA_3D_Z].to_numpy(float)
    x_edges = _percentile_edges(x, nx)
    y_edges = _percentile_edges(y, ny)
    sum_z, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=z)
    counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_z = np.where(counts > 0, sum_z / counts, np.nan)
    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    yc = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(xc, yc, indexing="ij")
    return xx, yy, mean_z


def _voxel_grow_field(
    use: pd.DataFrame,
    grow_axis: str,
    counts: np.ndarray,
    xc: np.ndarray,
    yc: np.ndarray,
    zc: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    z_edges: np.ndarray,
    min_count: int,
) -> np.ndarray:
    if grow_axis == "month":
        months = use["month"].to_numpy(float)
        x = use[_EXTRA_3D_X].to_numpy(float)
        y = use[_EXTRA_3D_Y].to_numpy(float)
        z = use[_EXTRA_3D_Z].to_numpy(float)
        first = np.full(counts.shape, np.inf)
        for month in range(1, 13):
            sel = months <= month
            if not np.any(sel):
                continue
            month_counts, _, _, _ = _hist3d_counts(x[sel], y[sel], z[sel], x_edges, y_edges, z_edges)
            first = np.where((month_counts >= min_count) & ~np.isfinite(first), float(month), first)
        return first
    if grow_axis == "htop":
        field = np.broadcast_to(xc[:, None, None], counts.shape).copy()
    elif grow_axis == "depth":
        field = np.broadcast_to(yc[None, :, None], counts.shape).copy()
    else:
        field = np.broadcast_to(zc[None, None, :], counts.shape).copy()
    return np.where(counts >= min_count, field, np.inf)


def plot_layers_bar3d(
    layers: pd.DataFrame,
    style: FigureStyle,
    *,
    inversion_type: str | None = None,
    nx: int = 12,
    ny: int = 10,
    title: str | None = None,
    elev: float = 24.0,
    azim: float = -58.0,
):
    """Столбики bar3d: X=Htop, Y=γ, высота = mean(depth) в ячейке."""
    use = _prepare_extra_3d_frame(layers, inversion_type=inversion_type)
    xlabel = "Высота верха AGL, м" if style.language == "ru" else "Top height AGL, m"
    ylabel = "γ, °C/100 м" if style.language == "ru" else "γ, °C/100 m"
    zlabel = "Средняя толщина, м" if style.language == "ru" else "Mean depth, m"
    default_title = (
        f"3D bar: {_type_title(inversion_type, style)}" if inversion_type
        else ("3D bar: G+E+HE" if style.language == "ru" else "3D bar: G+E+HE")
    )
    if use.empty:
        return _empty_3d_fig(style, title or default_title)

    x_edges = _percentile_edges(use[_EXTRA_3D_X].to_numpy(float), nx)
    g_edges = _percentile_edges(use[_EXTRA_3D_Z].to_numpy(float), ny)
    dx = float(np.diff(x_edges).mean()) * 0.85
    dy = float(np.diff(g_edges).mean()) * 0.85

    with article_rc(style):
        fig = plt.figure(figsize=(style.figure_width_in * 1.25, style.figure_height_in * 1.2))
        ax = fig.add_subplot(111, projection="3d")

        if inversion_type is None and "position_type" in use.columns:
            series = [
                (str(t), g, TYPE_COLORS.get(str(t), "#34495E"))
                for t, g in use.groupby("position_type", sort=False)
            ]
        else:
            color = TYPE_COLORS.get(str(inversion_type or "G"), "#34495E")
            series = [(str(inversion_type or "all"), use, color)]

        for label, gdf, color in series:
            gx = gdf[_EXTRA_3D_X].to_numpy(float)
            gd = gdf[_EXTRA_3D_Y].to_numpy(float)
            gg = gdf[_EXTRA_3D_Z].to_numpy(float)
            counts, _, _ = np.histogram2d(gx, gg, bins=[x_edges, g_edges])
            depth_sum, _, _ = np.histogram2d(gx, gg, bins=[x_edges, g_edges], weights=gd)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean_depth = np.where(counts > 0, depth_sum / counts, 0.0)
            ix, iy = np.nonzero(counts > 0)
            if ix.size == 0:
                continue
            xc = 0.5 * (x_edges[:-1] + x_edges[1:])
            yc = 0.5 * (g_edges[:-1] + g_edges[1:])
            xpos = xc[ix]
            ypos = yc[iy]
            zpos = np.zeros(ix.size)
            dz = mean_depth[ix, iy]
            n_cell = counts[ix, iy]
            alphas = 0.25 + 0.7 * (np.log1p(n_cell) / max(np.log1p(n_cell.max()), 1e-9))
            colors = [_hex_to_rgba(color, a) for a in alphas]
            ax.bar3d(
                xpos - dx / 2, ypos - dy / 2, zpos,
                dx, dy, dz,
                color=colors, shade=True, edgecolor="none",
                label=label if inversion_type is None else None,
            )

        ax.set_xlabel(xlabel, labelpad=8)
        ax.set_ylabel(ylabel, labelpad=8)
        ax.set_zlabel(zlabel, labelpad=8)
        ax.view_init(elev=elev, azim=azim)
        ax.grid(True, alpha=style.grid_alpha)
        if inversion_type is None:
            ax.legend(frameon=False, fontsize=style.legend_font_size - 1, loc="upper left")
        if style.show_title:
            fig.suptitle(title or default_title, fontsize=style.title_font_size, y=0.98)
        note = (
            "Бины Htop×γ; высота столбика = mean(depth)"
            if style.language == "ru"
            else "Htop×γ bins; bar height = mean(depth)"
        )
        fig.text(0.5, 0.02, note, ha="center", fontsize=style.tick_font_size - 1, color="#566573")
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.08, top=0.92)
        return fig


def plot_layers_voxels(
    layers: pd.DataFrame,
    style: FigureStyle,
    *,
    inversion_type: str | None = None,
    nx: int = 10,
    ny: int = 8,
    nz: int = 8,
    min_count: int = 1,
    title: str | None = None,
    elev: float = 24.0,
    azim: float = -58.0,
):
    """Воксельная 3D-гистограмма плотности N в (Htop, depth, γ)."""
    use = _prepare_extra_3d_frame(layers, inversion_type=inversion_type)
    default_title = (
        f"3D voxels: {_type_title(inversion_type, style)}" if inversion_type
        else "3D voxels: G+E+HE"
    )
    if use.empty:
        return _empty_3d_fig(style, title or default_title)

    x = use[_EXTRA_3D_X].to_numpy(float)
    y = use[_EXTRA_3D_Y].to_numpy(float)
    z = use[_EXTRA_3D_Z].to_numpy(float)
    x_edges, y_edges, z_edges = _bin_edges_3d(x, y, z, nx=nx, ny=ny, nz=nz)
    counts, _, _, _ = _hist3d_counts(x, y, z, x_edges, y_edges, z_edges)
    filled = counts >= float(min_count)
    if not np.any(filled):
        return _empty_3d_fig(style, title or default_title)

    base = TYPE_COLORS.get(str(inversion_type or "G"), "#2E86C1")
    logn = np.log1p(counts)
    vmax = float(logn[filled].max()) if np.any(filled) else 1.0
    norm = Normalize(vmin=0.0, vmax=max(vmax, 1e-9))
    cmap = mpl.colormaps["viridis"]
    facecolors = np.zeros(counts.shape + (4,))
    for i, j, k in zip(*np.nonzero(filled)):
        if inversion_type is None:
            rgba = list(cmap(norm(logn[i, j, k])))
        else:
            rgba = list(_hex_to_rgba(base, 0.25 + 0.7 * norm(logn[i, j, k])))
        facecolors[i, j, k] = rgba

    with article_rc(style):
        fig = plt.figure(figsize=(style.figure_width_in * 1.25, style.figure_height_in * 1.2))
        ax = fig.add_subplot(111, projection="3d")
        ax.voxels(filled, facecolors=facecolors, edgecolor="k", linewidth=0.15)
        # Масштаб осей в физические единицы через tick labels на индексах.
        ax.set_xticks(np.linspace(0, nx, 5))
        ax.set_yticks(np.linspace(0, ny, 5))
        ax.set_zticks(np.linspace(0, nz, 5))
        ax.set_xticklabels([f"{v:.0f}" for v in np.linspace(x_edges[0], x_edges[-1], 5)])
        ax.set_yticklabels([f"{v:.0f}" for v in np.linspace(y_edges[0], y_edges[-1], 5)])
        ax.set_zticklabels([f"{v:.1f}" for v in np.linspace(z_edges[0], z_edges[-1], 5)])
        ax.set_xlabel("Htop AGL, м" if style.language == "ru" else "Htop AGL, m", labelpad=8)
        ax.set_ylabel("Толщина, м" if style.language == "ru" else "Depth, m", labelpad=8)
        ax.set_zlabel("γ, °C/100 м" if style.language == "ru" else "γ, °C/100 m", labelpad=8)
        ax.view_init(elev=elev, azim=azim)
        if style.show_title:
            fig.suptitle(title or default_title, fontsize=style.title_font_size, y=0.98)
        note = (
            "Воксели: N в кубах (Htop × depth × γ); цвет ∝ log(N)"
            if style.language == "ru"
            else "Voxels: N in (Htop × depth × γ); color ∝ log(N)"
        )
        fig.text(0.5, 0.02, note, ha="center", fontsize=style.tick_font_size - 1, color="#566573")
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.08, top=0.92)
        return fig


def plot_layers_3d_projections(
    layers: pd.DataFrame,
    style: FigureStyle,
    *,
    inversion_type: str | None = None,
    title: str | None = None,
):
    """2D-проекции и гистограмма depth из тех же 3D-данных."""
    use = _prepare_extra_3d_frame(layers, inversion_type=inversion_type)
    default_title = (
        f"2D из 3D: {_type_title(inversion_type, style)}" if inversion_type
        else "2D из 3D: G+E+HE"
    )
    with article_rc(style):
        fig = plt.figure(figsize=(style.figure_width_in * 1.35, style.figure_height_in * 1.25))
        gs = GridSpec(2, 2, figure=fig, wspace=0.28, hspace=0.32)
        ax_xy = fig.add_subplot(gs[0, 0])
        ax_xz = fig.add_subplot(gs[0, 1])
        ax_yz = fig.add_subplot(gs[1, 0])
        ax_d = fig.add_subplot(gs[1, 1])

        if use.empty:
            for ax in (ax_xy, ax_xz, ax_yz, ax_d):
                ax.text(0.5, 0.5, "Нет данных" if style.language == "ru" else "No data", ha="center", va="center")
            if style.show_title:
                fig.suptitle(title or default_title, fontsize=style.title_font_size)
            return fig

        x = use[_EXTRA_3D_X].to_numpy(float)
        y = use[_EXTRA_3D_Y].to_numpy(float)
        z = use[_EXTRA_3D_Z].to_numpy(float)
        color = TYPE_COLORS.get(str(inversion_type or "G"), "#2E86C1")
        cmap = "Blues" if inversion_type == "G" else ("Oranges" if inversion_type == "E" else ("Greens" if inversion_type == "HE" else "viridis"))

        ax_xy.hexbin(x, y, gridsize=28, cmap=cmap, mincnt=1, linewidths=0)
        ax_xy.set_xlabel("Htop AGL, м" if style.language == "ru" else "Htop AGL, m")
        ax_xy.set_ylabel("Толщина, м" if style.language == "ru" else "Depth, m")
        ax_xy.set_title("Htop × depth")

        ax_xz.hexbin(x, z, gridsize=28, cmap=cmap, mincnt=1, linewidths=0)
        ax_xz.set_xlabel("Htop AGL, м" if style.language == "ru" else "Htop AGL, m")
        ax_xz.set_ylabel("γ, °C/100 м" if style.language == "ru" else "γ, °C/100 m")
        ax_xz.set_title("Htop × γ")

        ax_yz.hexbin(y, z, gridsize=28, cmap=cmap, mincnt=1, linewidths=0)
        ax_yz.set_xlabel("Толщина, м" if style.language == "ru" else "Depth, m")
        ax_yz.set_ylabel("γ, °C/100 м" if style.language == "ru" else "γ, °C/100 m")
        ax_yz.set_title("depth × γ")

        ax_d.hist(y, bins=30, color=color, alpha=0.85, edgecolor="white", linewidth=0.4)
        ax_d.set_xlabel("Толщина, м" if style.language == "ru" else "Depth, m")
        ax_d.set_ylabel("N")
        ax_d.set_title("Гистограмма depth" if style.language == "ru" else "Depth histogram")
        ax_d.grid(True, axis="y", alpha=style.grid_alpha, linewidth=0.5)

        if style.show_title:
            fig.suptitle(title or default_title, fontsize=style.title_font_size, y=0.98)
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.90)
        return fig


def _plot_layers_htop_depth_mesh(
    layers: pd.DataFrame,
    style: FigureStyle,
    *,
    inversion_type: str | None,
    kind: Literal["surface", "wireframe"],
    title: str | None = None,
    nx: int = 24,
    ny: int = 20,
    elev: float = 24.0,
    azim: float = -58.0,
):
    use = _prepare_extra_3d_frame(layers, inversion_type=inversion_type)
    kind_title = "3D surface" if kind == "surface" else "3D wireframe"
    default_title = (
        f"{kind_title}: {_type_title(inversion_type, style)}" if inversion_type
        else f"{kind_title}: G+E+HE"
    )
    if use.empty:
        return _empty_3d_fig(style, title or default_title)

    xx, yy, mean_z = _htop_depth_mean_gamma_grid(use, nx=nx, ny=ny)
    zz = np.ma.masked_invalid(mean_z)
    if zz.mask is True or (np.ma.is_masked(zz) and np.all(zz.mask)):
        return _empty_3d_fig(style, title or default_title)

    xlabel, ylabel, zlabel = _extra_3d_axis_labels(style)
    finite = mean_z[np.isfinite(mean_z)]
    z_lo = float(np.nanmin(finite)) if finite.size else 0.0
    z_hi = float(np.nanmax(finite)) if finite.size else 1.0
    if z_hi <= z_lo:
        z_hi = z_lo + 1.0
    color = TYPE_COLORS.get(str(inversion_type or "G"), "#2E86C1")
    cmap = _extra_3d_cmap(inversion_type)

    with article_rc(style):
        fig = plt.figure(figsize=(style.figure_width_in * 1.25, style.figure_height_in * 1.2))
        ax = fig.add_subplot(111, projection="3d")
        if kind == "surface":
            ax.plot_surface(
                xx, yy, zz,
                cmap=cmap,
                vmin=z_lo,
                vmax=z_hi,
                linewidth=0,
                antialiased=True,
                shade=True,
            )
        else:
            ax.plot_wireframe(
                xx, yy, zz,
                rstride=2,
                cstride=2,
                color=color,
                linewidth=0.7,
                alpha=0.85,
            )
        ax.set_xlabel(xlabel, labelpad=8)
        ax.set_ylabel(ylabel, labelpad=8)
        ax.set_zlabel(zlabel, labelpad=8)
        ax.view_init(elev=elev, azim=azim)
        if style.show_title:
            fig.suptitle(title or default_title, fontsize=style.title_font_size, y=0.98)
        note = (
            "Сетка Htop×depth; Z = mean(γ)"
            if style.language == "ru"
            else "Htop×depth grid; Z = mean(γ)"
        )
        fig.text(0.5, 0.02, note, ha="center", fontsize=style.tick_font_size - 1, color="#566573")
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.08, top=0.92)
        return fig


def plot_layers_surface(
    layers: pd.DataFrame,
    style: FigureStyle,
    *,
    inversion_type: str | None = None,
    title: str | None = None,
    nx: int = 24,
    ny: int = 20,
    elev: float = 24.0,
    azim: float = -58.0,
):
    """Поверхность mean(γ) на сетке Htop × depth."""
    return _plot_layers_htop_depth_mesh(
        layers, style, inversion_type=inversion_type, kind="surface",
        title=title, nx=nx, ny=ny, elev=elev, azim=azim,
    )


def plot_layers_wireframe(
    layers: pd.DataFrame,
    style: FigureStyle,
    *,
    inversion_type: str | None = None,
    title: str | None = None,
    nx: int = 24,
    ny: int = 20,
    elev: float = 24.0,
    azim: float = -58.0,
):
    """Каркас mean(γ) на сетке Htop × depth."""
    return _plot_layers_htop_depth_mesh(
        layers, style, inversion_type=inversion_type, kind="wireframe",
        title=title, nx=nx, ny=ny, elev=elev, azim=azim,
    )


def save_layers_scatter_3d_gif(
    layers: pd.DataFrame,
    style: FigureStyle,
    output_path: Path,
    *,
    inversion_type: str,
    n_frames: int = 36,
    dpi: int = 120,
    elev: float = 24.0,
) -> Path:
    """GIF вращения 3D scatter вокруг вертикали."""
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Для GIF нужен matplotlib.animation и Pillow") from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    use = _prepare_extra_3d_frame(layers, inversion_type=inversion_type)
    color = TYPE_COLORS.get(str(inversion_type), "#34495E")
    with article_rc(style):
        fig = plt.figure(figsize=(style.figure_width_in * 1.15, style.figure_height_in * 1.1))
        ax = fig.add_subplot(111, projection="3d")
        if use.empty:
            ax.text2D(0.5, 0.5, "Нет данных", transform=ax.transAxes, ha="center")
        else:
            x = use[_EXTRA_3D_X].to_numpy(float)
            y = use[_EXTRA_3D_Y].to_numpy(float)
            z = use[_EXTRA_3D_Z].to_numpy(float)
            x_plot, x_lo, x_hi = _axis_values_and_limits(_EXTRA_3D_X, x)
            y_plot, y_lo, y_hi = _axis_values_and_limits(_EXTRA_3D_Y, y)
            z_plot, z_lo, z_hi = _axis_values_and_limits(_EXTRA_3D_Z, z)
            ax.scatter(x_plot, y_plot, z_plot, c=color, s=4, alpha=0.38, linewidths=0, depthshade=True)
            ax.set_xlim(x_lo, _axis_limit_hi(_EXTRA_3D_X, x_lo, x_hi))
            ax.set_ylim(y_lo, _axis_limit_hi(_EXTRA_3D_Y, y_lo, y_hi))
            ax.set_zlim(z_lo, _axis_limit_hi(_EXTRA_3D_Z, z_lo, z_hi))
        ax.set_xlabel("Htop AGL, м" if style.language == "ru" else "Htop AGL, m")
        ax.set_ylabel("Толщина, м" if style.language == "ru" else "Depth, m")
        ax.set_zlabel("γ, °C/100 м" if style.language == "ru" else "γ, °C/100 m")
        if style.show_title:
            fig.suptitle(f"3D rotate: {_type_title(inversion_type, style)}", fontsize=style.title_font_size)

        def _update(frame: int):
            ax.view_init(elev=elev, azim=(360.0 * frame) / n_frames)
            return (ax,)

        anim = FuncAnimation(fig, _update, frames=n_frames, interval=80, blit=False)
        writer = PillowWriter(fps=12)
        anim.save(str(output_path), writer=writer, dpi=dpi)
        plt.close(fig)
    return output_path


def save_layers_scatter_3d_buildup_gif(
    layers: pd.DataFrame,
    style: FigureStyle,
    output_path: Path,
    *,
    inversion_type: str,
    grow_axis: GrowAxis,
    n_frames: int = 24,
    dpi: int = 100,
    elev: float = 24.0,
    azim: float = -58.0,
    max_points: int = 4000,
) -> Path:
    """GIF: точки нарастают от минимума по оси и затухают (шлейф объёма)."""
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Для GIF нужен matplotlib.animation и Pillow") from exc

    output_path = _as_gif_path(output_path)
    n_frames = _grow_axis_n_frames(grow_axis, n_frames)
    use = _subsample_frame(
        _prepare_buildup_frame(layers, inversion_type=inversion_type, grow_axis=grow_axis),
        max_points,
    )
    color = TYPE_COLORS.get(str(inversion_type), "#34495E")
    rgb = mpl.colors.to_rgb(color)
    axis_lbl = _grow_axis_label(grow_axis, style)
    title = f"3D buildup ({axis_lbl}): {_type_title(inversion_type, style)}"

    with article_rc(style):
        fig = plt.figure(figsize=(style.figure_width_in * 1.15, style.figure_height_in * 1.1))
        ax = fig.add_subplot(111, projection="3d")

        if use.empty:
            ax.text2D(0.5, 0.5, "Нет данных" if style.language == "ru" else "No data", transform=ax.transAxes, ha="center")
            if style.show_title:
                fig.suptitle(title, fontsize=style.title_font_size)

            def _empty_update(_frame: int):
                return (ax,)

            anim = FuncAnimation(fig, _empty_update, frames=max(n_frames, 2), interval=100, blit=False)
            anim.save(str(output_path), writer=PillowWriter(fps=10), dpi=dpi)
            plt.close(fig)
            return output_path

        x = use[_EXTRA_3D_X].to_numpy(float)
        y = use[_EXTRA_3D_Y].to_numpy(float)
        z = use[_EXTRA_3D_Z].to_numpy(float)
        grow_vals = use[_grow_axis_col(grow_axis)].to_numpy(float)
        x_plot, x_lo, x_hi = _axis_values_and_limits(_EXTRA_3D_X, x)
        y_plot, y_lo, y_hi = _axis_values_and_limits(_EXTRA_3D_Y, y)
        z_plot, z_lo, z_hi = _axis_values_and_limits(_EXTRA_3D_Z, z)
        x_lim = (x_lo, _axis_limit_hi(_EXTRA_3D_X, x_lo, x_hi))
        y_lim = (y_lo, _axis_limit_hi(_EXTRA_3D_Y, y_lo, y_hi))
        z_lim = (z_lo, _axis_limit_hi(_EXTRA_3D_Z, z_lo, z_hi))
        thresholds = _frame_thresholds(grow_vals, n_frames, grow_axis=grow_axis)

        def _update(frame: int):
            ax.cla()
            alphas = _fade_alphas(grow_vals, float(thresholds[frame]))
            mask = alphas > 0.02
            if np.any(mask):
                colors = np.zeros((int(mask.sum()), 4))
                colors[:, 0] = rgb[0]
                colors[:, 1] = rgb[1]
                colors[:, 2] = rgb[2]
                colors[:, 3] = alphas[mask]
                ax.scatter(
                    x_plot[mask], y_plot[mask], z_plot[mask],
                    c=colors, s=6, linewidths=0, depthshade=False,
                )
            _style_extra_3d_scatter_ax(ax, style, x_lim=x_lim, y_lim=y_lim, z_lim=z_lim, elev=elev, azim=azim)
            if style.show_title:
                fig.suptitle(title, fontsize=style.title_font_size)
            return (ax,)

        anim = FuncAnimation(fig, _update, frames=n_frames, interval=100, blit=False)
        anim.save(str(output_path), writer=PillowWriter(fps=10), dpi=dpi)
        plt.close(fig)
    return output_path


def save_layers_voxels_buildup_gif(
    layers: pd.DataFrame,
    style: FigureStyle,
    output_path: Path,
    *,
    inversion_type: str,
    grow_axis: GrowAxis,
    n_frames: int = 24,
    dpi: int = 100,
    elev: float = 24.0,
    azim: float = -58.0,
    nx: int = 10,
    ny: int = 8,
    nz: int = 8,
    min_count: int = 1,
) -> Path:
    """GIF: воксели плотности заполняются от минимума по оси с затуханием."""
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Для GIF нужен matplotlib.animation и Pillow") from exc

    output_path = _as_gif_path(output_path)
    n_frames = _grow_axis_n_frames(grow_axis, n_frames)
    use = _prepare_buildup_frame(layers, inversion_type=inversion_type, grow_axis=grow_axis)
    base = TYPE_COLORS.get(str(inversion_type), "#2E86C1")
    axis_lbl = _grow_axis_label(grow_axis, style)
    title = f"3D voxels fill ({axis_lbl}): {_type_title(inversion_type, style)}"

    with article_rc(style):
        fig = plt.figure(figsize=(style.figure_width_in * 1.15, style.figure_height_in * 1.1))
        ax = fig.add_subplot(111, projection="3d")

        if use.empty:
            ax.text2D(0.5, 0.5, "Нет данных" if style.language == "ru" else "No data", transform=ax.transAxes, ha="center")
            if style.show_title:
                fig.suptitle(title, fontsize=style.title_font_size)

            def _empty_update(_frame: int):
                return (ax,)

            anim = FuncAnimation(fig, _empty_update, frames=max(n_frames, 2), interval=100, blit=False)
            anim.save(str(output_path), writer=PillowWriter(fps=10), dpi=dpi)
            plt.close(fig)
            return output_path

        x = use[_EXTRA_3D_X].to_numpy(float)
        y = use[_EXTRA_3D_Y].to_numpy(float)
        z = use[_EXTRA_3D_Z].to_numpy(float)
        x_edges, y_edges, z_edges = _bin_edges_3d(x, y, z, nx=nx, ny=ny, nz=nz)
        counts, xc, yc, zc = _hist3d_counts(x, y, z, x_edges, y_edges, z_edges)
        grow_field = _voxel_grow_field(
            use, grow_axis, counts, xc, yc, zc, x_edges, y_edges, z_edges, min_count,
        )
        grow_vals = grow_field[np.isfinite(grow_field)]
        thresholds = _frame_thresholds(
            grow_vals if grow_vals.size else use[_grow_axis_col(grow_axis)].to_numpy(float),
            n_frames,
            grow_axis=grow_axis,
        )
        logn = np.log1p(counts)
        vmax = float(logn[np.isfinite(grow_field)].max()) if np.any(np.isfinite(grow_field)) else 1.0
        norm = Normalize(vmin=0.0, vmax=max(vmax, 1e-9))

        def _update(frame: int):
            ax.cla()
            thr = float(thresholds[frame])
            alphas = _fade_alphas(grow_field, thr, alpha_max=0.92, k=2.0)
            filled = alphas > 0.02
            facecolors = np.zeros(counts.shape + (4,))
            for i, j, k in zip(*np.nonzero(filled)):
                density = 0.35 + 0.65 * norm(logn[i, j, k])
                facecolors[i, j, k] = _hex_to_rgba(base, float(alphas[i, j, k] * density))
            if np.any(filled):
                ax.voxels(filled, facecolors=facecolors, edgecolor="k", linewidth=0.12)
            _style_voxel_ax(
                ax, style, nx=nx, ny=ny, nz=nz,
                x_edges=x_edges, y_edges=y_edges, z_edges=z_edges,
                elev=elev, azim=azim,
            )
            if style.show_title:
                fig.suptitle(title, fontsize=style.title_font_size)
            return (ax,)

        anim = FuncAnimation(fig, _update, frames=n_frames, interval=100, blit=False)
        anim.save(str(output_path), writer=PillowWriter(fps=10), dpi=dpi)
        plt.close(fig)
    return output_path


def build_scatter_3d_extra_figure_specs(
    layers: pd.DataFrame,
    style: FigureStyle,
) -> list[tuple[str, Callable[[], object]]]:
    """PNG-спеки: bars / voxels / projections_2d / surface / wireframe для G, E, HE и combined."""
    from .metrics import INVERSION_TYPES

    specs: list[tuple[str, Callable[[], object]]] = []
    kinds: list[tuple[str | None, str]] = [(t, t) for t in INVERSION_TYPES]
    kinds.append((None, "G_E_HE"))

    for inv_type, tag in kinds:
        label = _type_title(inv_type, style) if inv_type else "G+E+HE"
        specs.append(
            (
                f"scatter_3d/bars/top_height_gamma_mean_depth_{tag}",
                lambda t=inv_type, lbl=label: plot_layers_bar3d(
                    layers, style, inversion_type=t, title=f"3D bar: {lbl}",
                ),
            )
        )
        specs.append(
            (
                f"scatter_3d/voxels/top_height_depth_gamma_density_{tag}",
                lambda t=inv_type, lbl=label: plot_layers_voxels(
                    layers, style, inversion_type=t, title=f"3D voxels: {lbl}",
                ),
            )
        )
        specs.append(
            (
                f"scatter_3d/projections_2d/htop_depth_gamma_panels_{tag}",
                lambda t=inv_type, lbl=label: plot_layers_3d_projections(
                    layers, style, inversion_type=t, title=f"2D из 3D: {lbl}",
                ),
            )
        )
        specs.append(
            (
                f"scatter_3d/surface/htop_depth_mean_gamma_{tag}",
                lambda t=inv_type, lbl=label: plot_layers_surface(
                    layers, style, inversion_type=t, title=f"3D surface: {lbl}",
                ),
            )
        )
        specs.append(
            (
                f"scatter_3d/wireframe/htop_depth_mean_gamma_{tag}",
                lambda t=inv_type, lbl=label: plot_layers_wireframe(
                    layers, style, inversion_type=t, title=f"3D wireframe: {lbl}",
                ),
            )
        )
    return specs


def build_scatter_3d_animation_specs(
    layers: pd.DataFrame,
    style: FigureStyle,
) -> list[tuple[str, Callable[[Path], Path]]]:
    """GIF-спеки: вращение + нарастание scatter/voxels по 4 осям для G / E / HE."""
    from .metrics import INVERSION_TYPES

    specs: list[tuple[str, Callable[[Path], Path]]] = []
    for inv_type in INVERSION_TYPES:
        path = f"scatter_3d/animated/top_height_depth_gamma_rotate_{inv_type}"

        def _save_rotate(out: Path, t=inv_type) -> Path:
            return save_layers_scatter_3d_gif(layers, style, _as_gif_path(out), inversion_type=t)

        specs.append((path, _save_rotate))
        for axis in _GROW_AXES:
            fade_path = f"scatter_3d/animated/buildup_fade_{axis}_{inv_type}"
            voxels_path = f"scatter_3d/animated/voxels_fill_{axis}_{inv_type}"

            def _save_fade(out: Path, t=inv_type, a=axis) -> Path:
                return save_layers_scatter_3d_buildup_gif(
                    layers, style, _as_gif_path(out), inversion_type=t, grow_axis=a,
                )

            def _save_voxels(out: Path, t=inv_type, a=axis) -> Path:
                return save_layers_voxels_buildup_gif(
                    layers, style, _as_gif_path(out), inversion_type=t, grow_axis=a,
                )

            specs.append((fade_path, _save_fade))
            specs.append((voxels_path, _save_voxels))
    return specs


def plot_gamma_scatter_hist(
    gammas: pd.DataFrame,
    ref_heights: Mapping[int, float],
    style: FigureStyle,
    *,
    title: str | None = None,
):
    """Scatter γ vs высота AGL на опорных изobar (850/700/500 гПа) + гистограммы (линейная N)."""
    use = gammas.dropna(subset=["height_agl_m", "gamma_c_per_100m"]).copy()
    use = use[use["height_agl_m"] >= 0]
    xlabel = "Высота AGL, м" if style.language == "ru" else "Height AGL, m"
    ylabel = "γ, °C/100 м" if style.language == "ru" else "γ, °C/100 m"
    gamma_vals = use["gamma_c_per_100m"].to_numpy(float)
    y_hi = float(np.nanpercentile(gamma_vals, 99)) if gamma_vals.size else 20.0
    y_hi = max(y_hi, 5.0)
    point_colors = None
    if "pressure_hpa" in use.columns:
        point_colors = use["pressure_hpa"].map(
            lambda p: REF_LINE_COLORS.get(int(round(p)), "#34495E")
        ).to_numpy()
    with article_rc(style):
        fig = plt.figure(figsize=(style.figure_width_in * 1.05, style.figure_height_in * 1.15))
        gs = GridSpec(4, 4, figure=fig, wspace=0.06, hspace=0.06)
        _joint_scatter_hist(
            fig, gs,
            use["height_agl_m"].to_numpy(float),
            gamma_vals,
            ref_heights=ref_heights,
            style=style,
            point_colors=point_colors,
            xlabel=xlabel,
            ylabel=ylabel,
            title=None,
            show_legend=True,
            ylim=(0.0, y_hi * 1.05),
        )
        if style.show_title:
            fig.suptitle(
                title or (
                    "Температурный градиент γ на опорных изobar (850/700/500 гПа)"
                    if style.language == "ru"
                    else "Temperature gradient γ at reference isobars (850/700/500 hPa)"
                ),
                fontsize=style.title_font_size,
                y=1.02,
            )
        fig.subplots_adjust(left=0.1, right=0.92, top=0.94, bottom=0.1)
        return fig
