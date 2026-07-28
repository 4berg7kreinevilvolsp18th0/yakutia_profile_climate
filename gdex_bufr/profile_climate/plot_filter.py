"""Фильтрация уровней и профилей для месячных графиков."""
from __future__ import annotations

from typing import Any

from gdex_bufr.profile_climate.metrics import (
    PROFILE_STATUS_BAD_PRESSURE,
    PROFILE_STATUS_DUPLICATE_LEVELS,
    PROFILE_STATUS_GOOD,
    PROFILE_STATUS_NO_SURFACE,
    PROFILE_STATUS_NO_TEMP,
    PROFILE_STATUS_SHORT,
)

# Статусы, которые не рисуем на графиках.
PLOT_EXCLUDE_STATUSES = frozenset({
    PROFILE_STATUS_NO_SURFACE,
    PROFILE_STATUS_NO_TEMP,
    PROFILE_STATUS_BAD_PRESSURE,
    PROFILE_STATUS_DUPLICATE_LEVELS,
})

# Допустимы для графика, если plot_only_good=False.
PLOT_ALLOWED_STATUSES = frozenset({
    PROFILE_STATUS_GOOD,
    PROFILE_STATUS_SHORT,
    "no_500",
})


def dedupe_levels_by_pressure(levels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Один уровень на давление; при дубле берём запись с температурой."""
    by_p: dict[float, dict[str, Any]] = {}
    for row in levels:
        p = row.get("pressure_hpa")
        if p is None:
            continue
        key = round(float(p), 1)
        prev = by_p.get(key)
        if prev is None or (row.get("temperature_c") is not None and prev.get("temperature_c") is None):
            by_p[key] = row
    return sorted(by_p.values(), key=lambda r: float(r["pressure_hpa"]), reverse=True)


def dedupe_levels_by_height(levels: list[dict[str, Any]], *, height_step_m: float = 10.0) -> list[dict[str, Any]]:
    """Один уровень на высоту (ADPUPA иногда даёт несколько T на одной H)."""
    by_h: dict[int, dict[str, Any]] = {}
    for row in levels:
        h = row.get("height_m")
        if h is None:
            continue
        key = int(round(float(h) / height_step_m))
        prev = by_h.get(key)
        if prev is None:
            by_h[key] = row
            continue
        # при дубле высоты оставляем уровень с давлением ближе к 1000 гПа (ниже по атмосфере)
        if float(row.get("pressure_hpa") or 0) > float(prev.get("pressure_hpa") or 0):
            by_h[key] = row
    return sorted(by_h.values(), key=lambda r: float(r["height_m"]))


def remove_temperature_spikes(
    levels: list[dict[str, Any]],
    *,
    max_delta_c: float = 10.0,
    max_height_step_m: float = 200.0,
) -> list[dict[str, Any]]:
    """Убирает уровни с нереалистичным скачком T на малой дистанции по высоте."""
    if len(levels) < 2:
        return levels
    sorted_levels = sorted(levels, key=lambda r: float(r["height_m"]))
    kept: list[dict[str, Any]] = [sorted_levels[0]]
    for row in sorted_levels[1:]:
        prev = kept[-1]
        dh = abs(float(row["height_m"]) - float(prev["height_m"]))
        dt = abs(float(row["temperature_c"]) - float(prev["temperature_c"]))
        if dh < max_height_step_m and dt > max_delta_c:
            continue
        kept.append(row)
    return kept


def filter_plot_levels(
    levels: list[dict[str, Any]],
    *,
    pressure_top_hpa: float = 500.0,
    max_surface_pressure_hpa: float = 1000.0,
    require_height: bool = True,
) -> list[dict[str, Any]]:
    """Оставляет уровни для графика: P в [top, surface], есть T и (опц.) высота."""
    cleaned = dedupe_levels_by_pressure(levels)
    cleaned = dedupe_levels_by_height(cleaned)
    result: list[dict[str, Any]] = []
    for row in cleaned:
        p = row.get("pressure_hpa")
        t = row.get("temperature_c")
        h = row.get("height_m")
        if p is None or t is None:
            continue
        p = float(p)
        if p > max_surface_pressure_hpa or p < pressure_top_hpa:
            continue
        if require_height and (h is None or float(h) < 0):
            continue
        result.append(row)
    result.sort(key=lambda r: float(r["height_m"] if require_height else r["pressure_hpa"]))
    result = remove_temperature_spikes(result)
    return result


def profile_passes_quality(
    levels: list[dict[str, Any]],
    *,
    max_temp_spread_c: float = 10.0,
    height_bin_m: float = 200.0,
    max_neighbor_delta_c: float = 10.0,
) -> bool:
    """Отбрасывает профили с горизонтальными зигзагами или большим соседним |ΔT|."""
    if len(levels) < 2:
        return True
    sorted_levels = sorted(levels, key=lambda r: float(r["height_m"]))
    temps = [float(lv["temperature_c"]) for lv in sorted_levels]
    if max(abs(temps[i + 1] - temps[i]) for i in range(len(temps) - 1)) >= max_neighbor_delta_c:
        return False
    bins: dict[int, list[float]] = {}
    for row in sorted_levels:
        key = int(float(row["height_m"]) // height_bin_m)
        bins.setdefault(key, []).append(float(row["temperature_c"]))
    for bin_temps in bins.values():
        if len(bin_temps) > 1 and max(bin_temps) - min(bin_temps) > max_temp_spread_c:
            return False
    return True


def is_profile_plot_eligible(
    metric: dict[str, Any] | None,
    levels: list[dict[str, Any]],
    *,
    plot_only_good: bool = False,
    min_levels: int = 3,
) -> bool:
    if len(levels) < min_levels:
        return False
    if not profile_passes_quality(levels):
        return False
    if metric is None:
        return True
    status = str(metric.get("profile_status") or "")
    if plot_only_good:
        return status == PROFILE_STATUS_GOOD
    if status in {PROFILE_STATUS_NO_SURFACE, PROFILE_STATUS_NO_TEMP, PROFILE_STATUS_BAD_PRESSURE}:
        return False
    if status == PROFILE_STATUS_DUPLICATE_LEVELS:
        pressures = [round(float(lv["pressure_hpa"]), 1) for lv in levels]
        return len(pressures) == len(set(pressures)) and len(levels) >= min_levels
    if status in PLOT_EXCLUDE_STATUSES:
        return False
    return status in PLOT_ALLOWED_STATUSES or status == PROFILE_STATUS_GOOD
