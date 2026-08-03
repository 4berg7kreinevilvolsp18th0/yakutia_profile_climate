"""Месячные графики пучков температурных профилей."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from gdex_bufr.profile_climate.metrics import PROFILE_STATUS_GOOD
from gdex_bufr.profile_climate.plot_filter import (
    describe_plot_filters,
    filter_plot_levels,
    is_profile_plot_eligible,
)

logger = logging.getLogger(__name__)

PROFILE_COLORS = plt.cm.tab20.colors


def _group_profiles(long_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in long_rows:
        grouped.setdefault(row["profile_id"], []).append(row)
    return grouped


def _metrics_by_id(metrics_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(r["profile_id"]): r for r in metrics_rows}


def _day_key(metric: dict[str, Any] | None, profile_id: str) -> str | None:
    if metric and metric.get("datetime_utc"):
        try:
            parsed = datetime.fromisoformat(str(metric["datetime_utc"]).replace("Z", "+00:00"))
            return parsed.date().isoformat()
        except ValueError:
            pass
    parts = profile_id.split("_")
    if len(parts) >= 2 and len(parts[1]) >= 8:
        token = parts[1]
        return f"{token[:4]}-{token[4:6]}-{token[6:8]}"
    return None


def _daily_label(day_key: str) -> str:
    try:
        parsed = datetime.strptime(day_key, "%Y-%m-%d")
        return f"{parsed:%d.%m}"
    except ValueError:
        return day_key


def _group_profiles_by_day(
    profiles: dict[str, list[dict[str, Any]]],
    profile_metrics: dict[str, dict[str, Any]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    by_day: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for profile_id, levels in profiles.items():
        day_key = _day_key(profile_metrics.get(profile_id), profile_id)
        if not day_key:
            continue
        by_day.setdefault(day_key, {})[profile_id] = levels
    return by_day


def _daily_mean_profiles(
    by_day: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    grid_points: int = 40,
) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    """Суточный усреднённый профиль: интерполяция по высоте, затем mean по срокам дня."""
    daily: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for day_key, day_profiles in by_day.items():
        if not day_profiles:
            continue
        min_h = min(min(float(lv["height_m"]) for lv in levels) for levels in day_profiles.values())
        max_h = max(max(float(lv["height_m"]) for lv in levels) for levels in day_profiles.values())
        grid = np.linspace(min_h, max_h, grid_points)
        mean_t = _mean_on_height_grid(day_profiles, grid)
        daily[day_key] = (grid, mean_t, len(day_profiles))
    return daily


def _mean_on_height_grid(
    profiles: dict[str, list[dict[str, Any]]],
    height_grid: np.ndarray,
) -> np.ndarray:
    temps: list[np.ndarray] = []
    for levels in profiles.values():
        heights = np.array([float(lv["height_m"]) for lv in levels], dtype=float)
        temperatures = np.array([float(lv["temperature_c"]) for lv in levels], dtype=float)
        if len(heights) < 2:
            continue
        order = np.argsort(heights)
        heights = heights[order]
        temperatures = temperatures[order]
        interp = np.interp(height_grid, heights, temperatures, left=np.nan, right=np.nan)
        temps.append(interp)
    if not temps:
        return np.full_like(height_grid, np.nan, dtype=float)
    return np.nanmean(np.vstack(temps), axis=0)


def _prepare_month_profiles(
    month_long: list[dict[str, Any]],
    month_metrics: list[dict[str, Any]],
    *,
    pressure_top_hpa: float,
    max_surface_pressure_hpa: float,
    plot_only_good: bool,
    plot_min_levels: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]], int]:
    """Возвращает отфильтрованные профили, метрики и число отброшенных."""
    metrics_map = _metrics_by_id(month_metrics)
    prepared: dict[str, list[dict[str, Any]]] = {}
    prepared_metrics: dict[str, dict[str, Any]] = {}
    rejected = 0

    for profile_id, raw_levels in _group_profiles(month_long).items():
        levels = filter_plot_levels(
            raw_levels,
            pressure_top_hpa=pressure_top_hpa,
            max_surface_pressure_hpa=max_surface_pressure_hpa,
            require_height=True,
        )
        metric = metrics_map.get(profile_id)
        if not is_profile_plot_eligible(
            metric,
            levels,
            plot_only_good=plot_only_good,
            min_levels=plot_min_levels,
        ):
            rejected += 1
            continue
        prepared[profile_id] = levels
        if metric:
            prepared_metrics[profile_id] = metric

    return prepared, prepared_metrics, rejected


def render_monthly_temperature_profiles(
    *,
    station_slug: str,
    station_name: str,
    year: int,
    month: int,
    long_rows: list[dict[str, Any]],
    metrics_rows: list[dict[str, Any]],
    output_path: Path,
    pressure_top_hpa: float = 500.0,
    max_surface_pressure_hpa: float = 1000.0,
    plot_only_good: bool = False,
    plot_min_levels: int = 3,
    show_mean: bool = True,
    min_profiles_per_month: int = 5,
) -> Path | None:
    month_long = [r for r in long_rows if int(r.get("year") or 0) == year and int(r.get("month") or 0) == month]
    month_metrics = [r for r in metrics_rows if int(r.get("year") or 0) == year and int(r.get("month") or 0) == month]

    if not month_long:
        logger.warning("Нет профилей для %s %04d-%02d", station_name or station_slug, year, month)
        return None

    profiles, profile_metrics, rejected = _prepare_month_profiles(
        month_long,
        month_metrics,
        pressure_top_hpa=pressure_top_hpa,
        max_surface_pressure_hpa=max_surface_pressure_hpa,
        plot_only_good=plot_only_good,
        plot_min_levels=plot_min_levels,
    )

    if not profiles:
        logger.warning("Нет пригодных профилей для %s %04d-%02d (отброшено %s)", station_name or station_slug, year, month, rejected)
        return None

    good_ids = {pid for pid, m in profile_metrics.items() if m.get("profile_status") == PROFILE_STATUS_GOOD}
    inversion_count = sum(1 for m in profile_metrics.values() if m.get("inversion_detected"))

    if len(profiles) < min_profiles_per_month:
        logger.warning(
            "Мало профилей для %s %04d-%02d: %s < %s",
            station_name or station_slug,
            year,
            month,
            len(profiles),
            min_profiles_per_month,
        )

    fig, ax = plt.subplots(figsize=(10, 10))
    by_day = _group_profiles_by_day(profiles, profile_metrics)
    daily_profiles = _daily_mean_profiles(by_day)
    day_keys = sorted(daily_profiles.keys())

    for index, day_key in enumerate(day_keys):
        grid, mean_t, _ = daily_profiles[day_key]
        color = PROFILE_COLORS[index % len(PROFILE_COLORS)]
        ax.plot(
            mean_t,
            grid,
            color=color,
            alpha=0.85,
            linewidth=1.6,
            label=_daily_label(day_key),
        )

    all_heights = [float(lv["height_m"]) for lv in month_long if lv.get("height_m") is not None]
    if show_mean and daily_profiles:
        min_h = min(grid[0] for grid, _, _ in daily_profiles.values())
        max_h = max(grid[-1] for grid, _, _ in daily_profiles.values())
        month_grid = np.linspace(min_h, max_h, 40)
        daily_for_mean = {
            day_key: [
                {"height_m": float(h), "temperature_c": float(t)}
                for h, t in zip(grid, mean_t, strict=True)
                if not np.isnan(t)
            ]
            for day_key, (grid, mean_t, _) in daily_profiles.items()
            if not np.all(np.isnan(mean_t))
        }
        mean_t = _mean_on_height_grid(daily_for_mean, month_grid)
        ax.plot(mean_t, month_grid, color="#d62728", linewidth=2.8, alpha=1.0, label="Средний за месяц", zorder=10)

    ax.set_xlabel("Температура, °C")
    ax.set_ylabel("Высота, м")
    if all_heights:
        plot_heights = [float(lv["height_m"]) for levels in profiles.values() for lv in levels]
        ax.set_ylim(min(plot_heights), max(plot_heights))
    ax.grid(True, alpha=0.3)
    profiles_in_month = sum(n for _, _, n in daily_profiles.values())
    raw_in_month = len(_group_profiles(month_long))
    ax.set_title(
        f"{station_name or station_slug} — {year}-{month:02d}\n"
        f"Суточных линий: {len(daily_profiles)} | на графике профилей: {profiles_in_month}/{raw_in_month} | "
        f"good: {len(good_ids)} | отброшено фильтром: {rejected} | инверсий: {inversion_count}"
    )
    ax.legend(loc="best", fontsize=8, ncol=2 if len(day_keys) > 8 else 1)

    filter_lines = describe_plot_filters(
        pressure_top_hpa=pressure_top_hpa,
        max_surface_pressure_hpa=max_surface_pressure_hpa,
        plot_only_good=plot_only_good,
        plot_min_levels=plot_min_levels,
        min_profiles_per_month=min_profiles_per_month,
    )
    filter_text = "Фильтры PNG:\n" + "\n".join(f"• {line}" for line in filter_lines)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(bottom=0.28)
    fig.text(
        0.02,
        0.02,
        filter_text,
        ha="left",
        va="bottom",
        fontsize=7,
        family="sans-serif",
        wrap=True,
        bbox={"boxstyle": "round", "facecolor": "#f7f7f7", "edgecolor": "#bbbbbb", "alpha": 0.95},
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def monthly_plot_path(
    output_root: Path,
    station_slug: str,
    year: int,
    month: int,
) -> Path:
    return (
        Path(output_root)
        / station_slug
        / str(year)
        / f"{station_slug}_{year}_{month:02d}_temperature_profiles_to_500hpa.png"
    )


def render_all_monthly_plots(
    *,
    station_slug: str,
    station_name: str,
    long_rows: list[dict[str, Any]],
    metrics_rows: list[dict[str, Any]],
    output_root: Path,
    start_year: int,
    end_year: int,
    start_month: int = 1,
    end_month: int = 12,
    pressure_top_hpa: float = 500.0,
    max_surface_pressure_hpa: float = 1000.0,
    plot_only_good: bool = False,
    plot_min_levels: int = 3,
    min_profiles_per_month: int = 5,
) -> list[str]:
    written: list[str] = []
    for year in range(start_year, end_year + 1):
        month_from = start_month if year == start_year else 1
        month_to = end_month if year == end_year else 12
        for month in range(month_from, month_to + 1):
            path = render_monthly_temperature_profiles(
                station_slug=station_slug,
                station_name=station_name,
                year=year,
                month=month,
                long_rows=long_rows,
                metrics_rows=metrics_rows,
                output_path=monthly_plot_path(output_root, station_slug, year, month),
                pressure_top_hpa=pressure_top_hpa,
                max_surface_pressure_hpa=max_surface_pressure_hpa,
                plot_only_good=plot_only_good,
                plot_min_levels=plot_min_levels,
                min_profiles_per_month=min_profiles_per_month,
            )
            if path:
                written.append(str(path))
    return written
