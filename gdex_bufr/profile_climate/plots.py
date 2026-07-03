"""Месячные графики пучков температурных профилей."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from gdex_bufr.profile_climate.metrics import PROFILE_STATUS_GOOD

logger = logging.getLogger(__name__)


def _group_profiles(long_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in long_rows:
        grouped.setdefault(row["profile_id"], []).append(row)
    for profile_id in grouped:
        grouped[profile_id].sort(key=lambda r: r.get("pressure_hpa", 0), reverse=True)
    return grouped


def _mean_profile(profiles: dict[str, list[dict[str, Any]]], pressure_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    temps: list[np.ndarray] = []
    for levels in profiles.values():
        pressures = np.array([lv["pressure_hpa"] for lv in levels], dtype=float)
        temperatures = np.array([lv["temperature_c"] for lv in levels], dtype=float)
        if len(pressures) < 2:
            continue
        order = np.argsort(pressures)
        pressures = pressures[order]
        temperatures = temperatures[order]
        interp = np.interp(pressure_grid, pressures[::-1], temperatures[::-1])
        temps.append(interp)
    if not temps:
        return pressure_grid, np.full_like(pressure_grid, np.nan, dtype=float)
    return pressure_grid, np.nanmean(np.vstack(temps), axis=0)


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
    show_mean: bool = True,
    min_profiles_per_month: int = 5,
) -> Path | None:
    month_long = [r for r in long_rows if int(r.get("year") or 0) == year and int(r.get("month") or 0) == month]
    month_metrics = [r for r in metrics_rows if int(r.get("year") or 0) == year and int(r.get("month") or 0) == month]

    if not month_long:
        logger.warning("Нет профилей для %s %04d-%02d", station_name or station_slug, year, month)
        return None

    profiles = _group_profiles(month_long)
    good_ids = {r["profile_id"] for r in month_metrics if r.get("profile_status") == PROFILE_STATUS_GOOD}
    inversion_count = sum(1 for r in month_metrics if r.get("inversion_detected"))

    if len(profiles) < min_profiles_per_month:
        logger.warning(
            "Мало профилей для %s %04d-%02d: %s < %s",
            station_name or station_slug,
            year,
            month,
            len(profiles),
            min_profiles_per_month,
        )

    fig, ax = plt.subplots(figsize=(8, 10))
    for profile_id, levels in profiles.items():
        pressures = [lv["pressure_hpa"] for lv in levels]
        temps = [lv["temperature_c"] for lv in levels]
        ax.plot(temps, pressures, color="#1f77b4", alpha=0.3, linewidth=1.0)

    if show_mean and profiles:
        min_p = min(min(lv["pressure_hpa"] for lv in levels) for levels in profiles.values())
        max_p = max(max(lv["pressure_hpa"] for lv in levels) for levels in profiles.values())
        grid = np.linspace(min(max_p, pressure_top_hpa), max(min_p, pressure_top_hpa), 40)
        _, mean_t = _mean_profile(profiles, grid)
        ax.plot(mean_t, grid, color="#d62728", linewidth=2.5, alpha=1.0, label="Средний профиль")

    ax.set_xlabel("Температура, °C")
    ax.set_ylabel("Давление, гПа")
    ax.set_ylim(pressure_top_hpa, max(lv["pressure_hpa"] for lv in month_long))
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.set_title(
        f"{station_name or station_slug} — {year}-{month:02d}\n"
        f"Профилей: {len(profiles)} | good: {len(good_ids)} | инверсий: {inversion_count}"
    )
    if show_mean:
        ax.legend(loc="best")

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
                min_profiles_per_month=min_profiles_per_month,
            )
            if path:
                written.append(str(path))
    return written
