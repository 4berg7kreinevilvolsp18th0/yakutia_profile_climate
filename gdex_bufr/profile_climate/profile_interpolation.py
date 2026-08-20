"""Интерполяция температурных профилей на общую вертикальную сетку."""
from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

# Фиксированная сетка проекта (500–925 гПа, шаг 25) — как в article_figures_config.yaml
DEFAULT_PRESSURE_GRID_HPA: tuple[float, ...] = tuple(float(x) for x in range(500, 926, 25))

# Высота AGL: 0–5000 м, шаг 100 м
DEFAULT_HEIGHT_GRID_AGL_M: tuple[float, ...] = tuple(float(x) for x in range(0, 5001, 100))


def default_target_grid(*, coordinate: Literal["pressure", "height"] = "pressure") -> np.ndarray:
    if coordinate == "pressure":
        # для графика: от поверхности к верху (убывание P)
        return np.asarray(DEFAULT_PRESSURE_GRID_HPA[::-1], dtype=float)
    return np.asarray(DEFAULT_HEIGHT_GRID_AGL_M, dtype=float)


def _as_float_array(values: Sequence[float] | np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    return arr


def interpolate_temperature_profile(
    pressure: Sequence[float] | np.ndarray,
    height: Sequence[float] | np.ndarray | None,
    temperature: Sequence[float] | np.ndarray,
    target_grid: np.ndarray,
    *,
    coordinate: Literal["pressure", "height"] = "pressure",
) -> np.ndarray:
    """Линейная интерполяция T на target_grid; без экстраполяции (NaN вне диапазона).

    T_i(P*) или T_i(H*) между двумя реальными уровнями профиля.
    """
    p = _as_float_array(pressure)
    t = _as_float_array(temperature)
    if p is None or t is None or len(p) < 2 or len(t) < 2:
        return np.full(np.asarray(target_grid).shape, np.nan, dtype=float)

    if coordinate == "pressure":
        coord = p
    else:
        h = _as_float_array(height)
        if h is None or len(h) != len(p):
            return np.full(np.asarray(target_grid).shape, np.nan, dtype=float)
        coord = h

    valid = ~np.isnan(coord) & ~np.isnan(t)
    if valid.sum() < 2:
        return np.full(np.asarray(target_grid).shape, np.nan, dtype=float)

    x = coord[valid]
    y = t[valid]
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    # убрать дубликаты координат (оставить первое)
    uniq_mask = np.concatenate([[True], np.diff(x_sorted) > 0])
    x_sorted = x_sorted[uniq_mask]
    y_sorted = y_sorted[uniq_mask]
    if len(x_sorted) < 2:
        return np.full(np.asarray(target_grid).shape, np.nan, dtype=float)

    grid = np.asarray(target_grid, dtype=float)
    return np.interp(grid, x_sorted, y_sorted, left=np.nan, right=np.nan)


def interp_on_pressure_grid(
    pressures: np.ndarray,
    values: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """Совместимость с obs_qc / build_daily_profiles."""
    return interpolate_temperature_profile(
        pressures,
        None,
        values,
        grid,
        coordinate="pressure",
    )
