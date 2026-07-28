"""QC уровней и скоринг выбросов для наблюдений (зондов)."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from gdex_bufr.profile_climate.plot_filter import (
    dedupe_levels_by_height,
    dedupe_levels_by_pressure,
    remove_temperature_spikes,
)

# Физика уровней
TEMP_MIN_C = -90.0
TEMP_MAX_C = 45.0

# Spike по давлению: большой |ΔT| при малом |ΔP|
SPIKE_MAX_ABS_DT_C = 10.0
SPIKE_MAX_ABS_DP_HPA = 5.0

# Соседние скачки / градиент (дашборд)
OUTLIER_MAX_ABS_DT_C = 10.0
OUTLIER_MAX_DT_DP_SQ = 0.25
MIN_ABS_DP_HPA = 0.5

# MAD к медиане месяца
MAD_K = 5.0
MAD_OUTLIER_FRACTION = 0.25
MAD_GRID_POINTS = 40
MAD_MIN_LEVELS = 2

# Мало уровней
MIN_LEVELS_FLAG = 5


def clean_observation_levels(
    levels: list[dict[str, Any]],
    *,
    pressure_top_hpa: float = 500.0,
    max_surface_pressure_hpa: float = 1000.0,
    temp_min_c: float = TEMP_MIN_C,
    temp_max_c: float = TEMP_MAX_C,
) -> list[dict[str, Any]]:
    """Физика + dedupe + spike по высоте и давлению; P убывает (земля → верх)."""
    cleaned = dedupe_levels_by_pressure(levels)
    cleaned = dedupe_levels_by_height(cleaned)
    result: list[dict[str, Any]] = []
    for row in cleaned:
        p = row.get("pressure_hpa")
        t = row.get("temperature_c")
        h = row.get("height_m")
        if p is None or t is None or h is None:
            continue
        p_f = float(p)
        t_f = float(t)
        h_f = float(h)
        if p_f > max_surface_pressure_hpa or p_f < pressure_top_hpa:
            continue
        if h_f < 0:
            continue
        if t_f < temp_min_c or t_f > temp_max_c:
            continue
        result.append({
            **row,
            "pressure_hpa": p_f,
            "temperature_c": t_f,
            "height_m": h_f,
        })
    result.sort(key=lambda r: float(r["pressure_hpa"]), reverse=True)
    result = _enforce_decreasing_pressure(result)
    result = remove_temperature_spikes(result)
    result = remove_temperature_spikes_by_pressure(result)
    result.sort(key=lambda r: float(r["pressure_hpa"]), reverse=True)
    return result


def _enforce_decreasing_pressure(levels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Оставляет уровни со строго убывающим давлением (после сортировки high→low)."""
    if not levels:
        return levels
    kept: list[dict[str, Any]] = [levels[0]]
    last_p = float(levels[0]["pressure_hpa"])
    for row in levels[1:]:
        p = float(row["pressure_hpa"])
        if p < last_p:
            kept.append(row)
            last_p = p
    return kept


def remove_temperature_spikes_by_pressure(
    levels: list[dict[str, Any]],
    *,
    max_delta_c: float = SPIKE_MAX_ABS_DT_C,
    max_dp_hpa: float = SPIKE_MAX_ABS_DP_HPA,
) -> list[dict[str, Any]]:
    """Убирает уровни с |ΔT| > max_delta_c при |ΔP| < max_dp_hpa."""
    if len(levels) < 2:
        return levels
    sorted_levels = sorted(levels, key=lambda r: float(r["pressure_hpa"]), reverse=True)
    kept: list[dict[str, Any]] = [sorted_levels[0]]
    for row in sorted_levels[1:]:
        prev = kept[-1]
        dp = abs(float(row["pressure_hpa"]) - float(prev["pressure_hpa"]))
        dt = abs(float(row["temperature_c"]) - float(prev["temperature_c"]))
        if dp < max_dp_hpa and dt > max_delta_c:
            continue
        kept.append(row)
    return kept


def _as_float_array(values: Sequence[Any] | None) -> np.ndarray | None:
    if not values:
        return None
    return np.asarray([np.nan if v is None else float(v) for v in values], dtype=float)


def max_abs_dt(obs: dict[str, Any]) -> float:
    """max |ΔT| между соседними уровнями (по убыванию давления)."""
    t = _as_float_array(obs.get("temperature_c"))
    p = _as_float_array(obs.get("pressure_hpa"))
    if t is None or p is None:
        return float("inf")
    valid = ~np.isnan(t) & ~np.isnan(p)
    if valid.sum() < 2:
        return float("inf")
    t = t[valid]
    p = p[valid]
    order = np.argsort(-p)
    t = t[order]
    return float(np.max(np.abs(np.diff(t))))


def max_dt_dp_sq(obs: dict[str, Any], *, min_abs_dp_hpa: float = MIN_ABS_DP_HPA) -> float:
    """max (ΔT/ΔP)² между соседними уровнями по давлению."""
    t = _as_float_array(obs.get("temperature_c"))
    p = _as_float_array(obs.get("pressure_hpa"))
    if t is None or p is None:
        return float("inf")
    valid = ~np.isnan(t) & ~np.isnan(p)
    if valid.sum() < 2:
        return float("inf")
    t = t[valid]
    p = p[valid]
    order = np.argsort(-p)
    t = t[order]
    p = p[order]
    scores: list[float] = []
    for i in range(len(t) - 1):
        dp = abs(float(p[i + 1] - p[i]))
        if dp < min_abs_dp_hpa:
            continue
        dt = float(t[i + 1] - t[i])
        scores.append((dt / dp) ** 2)
    if not scores:
        return float("inf")
    return float(max(scores))


def is_few_levels(obs: dict[str, Any], *, min_levels: int = MIN_LEVELS_FLAG) -> bool:
    n = obs.get("n_levels")
    if n is not None:
        return int(n) < min_levels
    temps = obs.get("temperature_c") or []
    return len(temps) < min_levels


def interp_on_pressure_grid(
    pressures: np.ndarray,
    values: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """Интерполяция на сетку давления; xp должен возрастать для np.interp."""
    if len(pressures) < 2:
        return np.full_like(grid, np.nan, dtype=float)
    order = np.argsort(pressures)
    return np.interp(
        grid,
        pressures[order],
        values[order],
        left=np.nan,
        right=np.nan,
    )


def pressure_grid_for_obs(
    observations: Sequence[dict[str, Any]],
    *,
    grid_points: int = MAD_GRID_POINTS,
) -> np.ndarray | None:
    p_lo = None
    p_hi = None
    for obs in observations:
        p = _as_float_array(obs.get("pressure_hpa"))
        if p is None:
            continue
        valid = ~np.isnan(p)
        if valid.sum() < 1:
            continue
        lo = float(np.nanmin(p[valid]))
        hi = float(np.nanmax(p[valid]))
        p_lo = lo if p_lo is None else min(p_lo, lo)
        p_hi = hi if p_hi is None else max(p_hi, hi)
    if p_lo is None or p_hi is None or p_hi <= p_lo:
        return None
    return np.linspace(p_hi, p_lo, grid_points)


def mad_outlier_fraction(
    obs: dict[str, Any],
    median: np.ndarray,
    mad: np.ndarray,
    grid: np.ndarray,
    *,
    k: float = MAD_K,
) -> float:
    """Доля уровней на сетке P, где |T−med| > k·MAD (MAD=0 → только точное равенство)."""
    p = _as_float_array(obs.get("pressure_hpa"))
    t = _as_float_array(obs.get("temperature_c"))
    if p is None or t is None:
        return 1.0
    valid = ~np.isnan(t) & ~np.isnan(p)
    if valid.sum() < MAD_MIN_LEVELS:
        return 1.0
    t_grid = interp_on_pressure_grid(p[valid], t[valid], grid)
    comparable = ~np.isnan(t_grid) & ~np.isnan(median) & ~np.isnan(mad)
    if comparable.sum() < MAD_MIN_LEVELS:
        return 1.0
    residual = np.abs(t_grid[comparable] - median[comparable])
    threshold = k * mad[comparable]
    # где MAD≈0: флаг только при заметном отклонении
    flags = np.where(
        threshold < 1e-9,
        residual > 1e-6,
        residual > threshold,
    )
    return float(np.mean(flags))


def month_median_mad(
    observations: Sequence[dict[str, Any]],
    *,
    grid_points: int = MAD_GRID_POINTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Медиана и MAD температуры на общей сетке давления."""
    grid = pressure_grid_for_obs(observations, grid_points=grid_points)
    if grid is None:
        return None
    stack: list[np.ndarray] = []
    for obs in observations:
        p = _as_float_array(obs.get("pressure_hpa"))
        t = _as_float_array(obs.get("temperature_c"))
        if p is None or t is None:
            continue
        valid = ~np.isnan(t) & ~np.isnan(p)
        if valid.sum() < MAD_MIN_LEVELS:
            continue
        stack.append(interp_on_pressure_grid(p[valid], t[valid], grid))
    if len(stack) < 2:
        return None
    arr = np.vstack(stack)
    median = np.nanmedian(arr, axis=0)
    mad = np.nanmedian(np.abs(arr - median), axis=0)
    return grid, median, mad


def suggest_outliers_mad(
    observations: Sequence[dict[str, Any]],
    enabled_ids: set[str],
    *,
    k: float = MAD_K,
    fraction: float = MAD_OUTLIER_FRACTION,
) -> list[str]:
    """Кандидаты по MAD относительно медианы включённых наблюдений месяца."""
    pool = [o for o in observations if o["profile_id"] in enabled_ids]
    stats = month_median_mad(pool)
    if stats is None:
        return []
    grid, median, mad = stats
    out: list[tuple[float, str]] = []
    for obs in pool:
        frac = mad_outlier_fraction(obs, median, mad, grid, k=k)
        if frac >= fraction:
            out.append((frac, obs["profile_id"]))
    out.sort(reverse=True)
    return [pid for _, pid in out]


def suggest_outliers_abs_dt(
    observations: Sequence[dict[str, Any]],
    enabled_ids: set[str],
    *,
    threshold: float = OUTLIER_MAX_ABS_DT_C,
) -> list[str]:
    scored = []
    for obs in observations:
        pid = obs["profile_id"]
        if pid not in enabled_ids:
            continue
        scored.append((max_abs_dt(obs), pid))
    scored.sort(reverse=True)
    return [pid for score, pid in scored if score >= threshold]


def suggest_outliers_dt_dp_sq(
    observations: Sequence[dict[str, Any]],
    enabled_ids: set[str],
    *,
    threshold: float = OUTLIER_MAX_DT_DP_SQ,
) -> list[str]:
    scored = []
    for obs in observations:
        pid = obs["profile_id"]
        if pid not in enabled_ids:
            continue
        scored.append((max_dt_dp_sq(obs), pid))
    scored.sort(reverse=True)
    return [pid for score, pid in scored if score >= threshold]


def suggest_outliers_few_levels(
    observations: Sequence[dict[str, Any]],
    enabled_ids: set[str],
    *,
    min_levels: int = MIN_LEVELS_FLAG,
) -> list[str]:
    return [
        obs["profile_id"]
        for obs in observations
        if obs["profile_id"] in enabled_ids and is_few_levels(obs, min_levels=min_levels)
    ]
