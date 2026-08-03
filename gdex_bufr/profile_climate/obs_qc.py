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

# Hampel / residual spike
HAMPEL_WINDOW = 5  # нечётное: 2*h+1
HAMPEL_K = 3.0
SPIKE_ABS_C = 8.0
MAD_SCALE = 1.4826  # нормализация MAD ≈ σ

# Скор формы T − Ts
FORM_GRID_POINTS = 40
FORM_RMSE_MIN_C = 3.0
FORM_PERCENTILE = 95.0
FORM_MIN_LEVELS = 3

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
    """Физика + dedupe + spike по высоте/давлению + Hampel; P убывает (земля → верх)."""
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
    result = remove_hampel_spike_levels(result)
    result.sort(key=lambda r: float(r["pressure_hpa"]), reverse=True)
    result = _enforce_increasing_height_with_falling_pressure(result)
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


def _enforce_increasing_height_with_falling_pressure(
    levels: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """При убывающем P высота должна расти; иначе уровень — источник спиралей на оси H."""
    if not levels:
        return levels
    sorted_levels = sorted(levels, key=lambda r: float(r["pressure_hpa"]), reverse=True)
    kept: list[dict[str, Any]] = [sorted_levels[0]]
    last_h = float(sorted_levels[0]["height_m"])
    for row in sorted_levels[1:]:
        h = float(row["height_m"])
        if h > last_h:
            kept.append(row)
            last_h = h
    return kept


def prepare_plot_arrays(
    obs: dict[str, Any],
    y_axis: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """T и Y для графика: без спиралей (строго монотонный Y).

    Давление: сортировка по убыванию P, строгое убывание.
    Высота: сортировка по убыванию P, затем только растущая H (гидростатика).
    Возвращает (temperature_c, y_values) или None.
    """
    temps = obs.get("temperature_c")
    if not temps:
        return None
    t = np.asarray([np.nan if v is None else float(v) for v in temps], dtype=float)

    pressures = obs.get("pressure_hpa")
    heights = obs.get("heights_m")
    p = None
    if pressures:
        p = np.asarray([np.nan if v is None else float(v) for v in pressures], dtype=float)
    h = None
    if heights:
        h = np.asarray([np.nan if v is None else float(v) for v in heights], dtype=float)

    if y_axis == "pressure":
        if p is None:
            return None
        n = min(len(t), len(p))
        t = t[:n]
        p = p[:n]
        valid = ~np.isnan(t) & ~np.isnan(p)
        if valid.sum() < 2:
            return None
        t = t[valid]
        p = p[valid]
        order = np.argsort(-p)
        t = t[order]
        p = p[order]
        keep_t = [float(t[0])]
        keep_y = [float(p[0])]
        for i in range(1, len(p)):
            pi = float(p[i])
            if pi < keep_y[-1]:
                keep_y.append(pi)
                keep_t.append(float(t[i]))
    else:
        if h is None:
            return None
        n = min(len(t), len(h))
        t = t[:n]
        h = h[:n]
        if p is not None:
            p = p[: min(len(p), n)]
            if len(p) == n:
                valid = ~np.isnan(t) & ~np.isnan(h) & ~np.isnan(p)
                if valid.sum() < 2:
                    return None
                t = t[valid]
                h = h[valid]
                p = p[valid]
                order = np.argsort(-p)
                t = t[order]
                h = h[order]
            else:
                valid = ~np.isnan(t) & ~np.isnan(h)
                if valid.sum() < 2:
                    return None
                t = t[valid]
                h = h[valid]
                order = np.argsort(h)
                t = t[order]
                h = h[order]
        else:
            valid = ~np.isnan(t) & ~np.isnan(h)
            if valid.sum() < 2:
                return None
            t = t[valid]
            h = h[valid]
            order = np.argsort(h)
            t = t[order]
            h = h[order]
        keep_t = [float(t[0])]
        keep_y = [float(h[0])]
        for i in range(1, len(h)):
            hi = float(h[i])
            if hi > keep_y[-1]:
                keep_y.append(hi)
                keep_t.append(float(t[i]))

    if len(keep_t) < 2:
        return None
    return np.asarray(keep_t, dtype=float), np.asarray(keep_y, dtype=float)


def raw_plot_arrays(
    obs: dict[str, Any],
    y_axis: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """T и Y без сортировки, dedupe и QC — в исходном порядке наблюдения."""
    temps = _as_float_array(obs.get("temperature_c"))
    y_values = _as_float_array(
        obs.get("pressure_hpa") if y_axis == "pressure" else obs.get("heights_m")
    )
    if temps is None or y_values is None:
        return None
    n = min(len(temps), len(y_values))
    temps = temps[:n]
    y_values = y_values[:n]
    valid = ~np.isnan(temps) & ~np.isnan(y_values)
    if valid.sum() < 1:
        return None
    return temps[valid], y_values[valid]


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


def hampel_residuals(
    temps: np.ndarray,
    *,
    window: int = HAMPEL_WINDOW,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Классический Hampel: медиана соседей без самой точки → |r| и общий порог.

    Возвращает (local_median, |r|, threshold).
    """
    n = len(temps)
    if n < 3:
        zeros = np.zeros(n, dtype=float)
        return temps.astype(float), zeros, float("inf")
    w = window if window % 2 == 1 else window + 1
    w = max(3, w)
    half = w // 2
    x = temps.astype(float)
    local_med = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        neighbors = np.concatenate([x[lo:i], x[i + 1:hi]])
        if len(neighbors) == 0:
            local_med[i] = x[i]
        else:
            local_med[i] = float(np.median(neighbors))
    abs_r = np.abs(x - local_med)
    mad = float(np.median(abs_r))
    threshold = max(HAMPEL_K * MAD_SCALE * mad, SPIKE_ABS_C)
    return local_med, abs_r, threshold


def remove_hampel_spike_levels(
    levels: list[dict[str, Any]],
    *,
    window: int = HAMPEL_WINDOW,
) -> list[dict[str, Any]]:
    """Убирает уровни, помеченные Hampel как spike."""
    if len(levels) < 5:
        return levels
    sorted_levels = sorted(levels, key=lambda r: float(r["pressure_hpa"]), reverse=True)
    temps = np.asarray([float(r["temperature_c"]) for r in sorted_levels], dtype=float)
    _, abs_r, threshold = hampel_residuals(temps, window=window)
    return [lv for lv, r in zip(sorted_levels, abs_r) if r <= threshold]


def spike_scores(obs: dict[str, Any]) -> tuple[float, int]:
    """(max |r|, n_spike) по Hampel/residual внутри зонда."""
    t = _as_float_array(obs.get("temperature_c"))
    p = _as_float_array(obs.get("pressure_hpa"))
    if t is None or p is None:
        return float("inf"), 10**9
    valid = ~np.isnan(t) & ~np.isnan(p)
    if valid.sum() < 3:
        return float("inf"), 10**9
    t = t[valid]
    p = p[valid]
    order = np.argsort(-p)
    t = t[order]
    _, abs_r, threshold = hampel_residuals(t)
    n_spike = int(np.sum(abs_r > threshold))
    return float(np.max(abs_r)), n_spike


def is_spike_outlier(obs: dict[str, Any]) -> bool:
    _, n_spike = spike_scores(obs)
    return n_spike >= 1


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
    grid_points: int = FORM_GRID_POINTS,
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


def _surface_temp(obs: dict[str, Any]) -> float | None:
    ts = obs.get("t_surface_c")
    if ts is not None and not (isinstance(ts, float) and np.isnan(ts)):
        return float(ts)
    t = _as_float_array(obs.get("temperature_c"))
    p = _as_float_array(obs.get("pressure_hpa"))
    if t is None or p is None:
        return None
    valid = ~np.isnan(t) & ~np.isnan(p)
    if valid.sum() < 1:
        return None
    # земля = максимальное давление
    idx = int(np.nanargmax(np.where(valid, p, -np.inf)))
    return float(t[idx])


def shape_anomaly_on_grid(obs: dict[str, Any], grid: np.ndarray) -> np.ndarray | None:
    """T(p) − Ts на сетке давления."""
    p = _as_float_array(obs.get("pressure_hpa"))
    t = _as_float_array(obs.get("temperature_c"))
    ts = _surface_temp(obs)
    if p is None or t is None or ts is None:
        return None
    valid = ~np.isnan(t) & ~np.isnan(p)
    if valid.sum() < FORM_MIN_LEVELS:
        return None
    t_grid = interp_on_pressure_grid(p[valid], t[valid], grid)
    return t_grid - ts


def form_rmse(obs: dict[str, Any], median_anom: np.ndarray, grid: np.ndarray) -> float:
    """RMSE формы после вычитания среднего смещения профиля относительно медианы."""
    anom = shape_anomaly_on_grid(obs, grid)
    if anom is None:
        return float("inf")
    comparable = ~np.isnan(anom) & ~np.isnan(median_anom)
    if comparable.sum() < FORM_MIN_LEVELS:
        return float("inf")
    diff = anom[comparable] - median_anom[comparable]
    diff = diff - np.nanmean(diff)  # не бить за равномерный сдвиг
    return float(np.sqrt(np.nanmean(diff ** 2)))


def month_median_shape(
    observations: Sequence[dict[str, Any]],
    *,
    grid_points: int = FORM_GRID_POINTS,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Медиана формы T−Ts на общей сетке давления."""
    grid = pressure_grid_for_obs(observations, grid_points=grid_points)
    if grid is None:
        return None
    stack: list[np.ndarray] = []
    for obs in observations:
        anom = shape_anomaly_on_grid(obs, grid)
        if anom is None:
            continue
        stack.append(anom)
    if len(stack) < 2:
        return None
    return grid, np.nanmedian(np.vstack(stack), axis=0)


def form_rmse_threshold(
    observations: Sequence[dict[str, Any]],
    *,
    percentile: float = FORM_PERCENTILE,
    min_c: float = FORM_RMSE_MIN_C,
) -> float | None:
    """Порог = max(P95 скоров, FORM_RMSE_MIN_C)."""
    stats = month_median_shape(observations)
    if stats is None:
        return None
    grid, median_anom = stats
    scores = [form_rmse(obs, median_anom, grid) for obs in observations]
    finite = [s for s in scores if s != float("inf") and s == s]
    if len(finite) < 2:
        return None
    return max(float(np.percentile(finite, percentile)), min_c)


def suggest_outliers_spike(
    observations: Sequence[dict[str, Any]],
    enabled_ids: set[str],
) -> list[str]:
    scored: list[tuple[int, float, str]] = []
    for obs in observations:
        pid = obs["profile_id"]
        if pid not in enabled_ids:
            continue
        max_r, n_spike = spike_scores(obs)
        if n_spike >= 1:
            scored.append((n_spike, max_r, pid))
    scored.sort(reverse=True)
    return [pid for _, _, pid in scored]


def suggest_outliers_form(
    observations: Sequence[dict[str, Any]],
    enabled_ids: set[str],
    *,
    percentile: float = FORM_PERCENTILE,
    min_c: float = FORM_RMSE_MIN_C,
) -> list[str]:
    pool = [o for o in observations if o["profile_id"] in enabled_ids]
    stats = month_median_shape(pool)
    if stats is None:
        return []
    grid, median_anom = stats
    scores = {obs["profile_id"]: form_rmse(obs, median_anom, grid) for obs in pool}
    finite = [s for s in scores.values() if s != float("inf") and s == s]
    if len(finite) < 2:
        return []
    threshold = max(float(np.percentile(finite, percentile)), min_c)
    out = [(s, pid) for pid, s in scores.items() if s >= threshold]
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
