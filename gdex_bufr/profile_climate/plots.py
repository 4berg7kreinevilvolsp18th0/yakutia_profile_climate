"""Месячные графики пучков температурных профилей."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from gdex_bufr.profile_climate.metrics import PROFILE_STATUS_GOOD
from gdex_bufr.profile_climate.plot_filter import filter_plot_levels, is_profile_plot_eligible

logger = logging.getLogger(__name__)

PROFILE_COLORS = plt.cm.tab20.colors


def _group_profiles(long_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in long_rows:
        grouped.setdefault(row["profile_id"], []).append(row)
    return grouped


def _metrics_by_id(metrics_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(r["profile_id"]): r for r in metrics_rows}


def _profile_label(metric: dict[str, Any] | None, profile_id: str) -> str:
    if not metric:
        return profile_id[-8:]
    dt = metric.get("datetime_utc") or ""
    cycle = str(metric.get("cycle") or "").zfill(2)
    try:
        parsed = datetime.fromisoformat(str(dt).replace("Z", "+00:00"))
        return f"{parsed:%d.%m} {cycle}Z"
    except ValueError:
        return f"{dt} {cycle}Z" if dt else profile_id[-12:]


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
        interp = np.interp(height_grid, heights, temperatures)
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
    profile_ids = sorted(profiles.keys())

    for index, profile_id in enumerate(profile_ids):
        levels = profiles[profile_id]
        heights = [float(lv["height_m"]) for lv in levels]
        temps = [float(lv["temperature_c"]) for lv in levels]
        color = PROFILE_COLORS[index % len(PROFILE_COLORS)]
        label = _profile_label(profile_metrics.get(profile_id), profile_id)
        ax.plot(
            temps,
            heights,
            color=color,
            alpha=0.85,
            linewidth=1.6,
            label=label,
        )

    all_heights = [float(lv["height_m"]) for lv in month_long if lv.get("height_m") is not None]
    if show_mean and profiles:
        min_h = min(min(float(lv["height_m"]) for lv in levels) for levels in profiles.values())
        max_h = max(max(float(lv["height_m"]) for lv in levels) for levels in profiles.values())
        grid = np.linspace(min_h, max_h, 40)
        mean_t = _mean_on_height_grid(profiles, grid)
        ax.plot(mean_t, grid, color="#d62728", linewidth=2.8, alpha=1.0, label="Средний профиль", zorder=10)

    ax.set_xlabel("Температура, °C")
    ax.set_ylabel("Высота, м")
    if all_heights:
        plot_heights = [float(lv["height_m"]) for levels in profiles.values() for lv in levels]
        ax.set_ylim(min(plot_heights), max(plot_heights))
    ax.grid(True, alpha=0.3)
    ax.set_title(
        f"{station_name or station_slug} — {year}-{month:02d}\n"
        f"На графике: {len(profiles)} | good: {len(good_ids)} | отброшено: {rejected} | инверсий: {inversion_count}"
    )
    ax.legend(loc="best", fontsize=8, ncol=2 if len(profile_ids) > 8 else 1)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
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
