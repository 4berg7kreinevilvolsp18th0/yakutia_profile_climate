"""Заполнение высоты уровней: наблюдённая / Φ→z / линейная интерп / барометрия."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from gdex_bufr.meteo_parser_bridge import (
    estimate_geopotential_height_m,
    geopotential_to_height_m,
)

# Высота станции над уровнем моря (м), WMO / pogodaiklimat / GCOS
STATION_ELEVATION_M: dict[str, float] = {
    "31004": 679.0,  # Алдан ≈ 678–679 м
    "24959": 103.0,  # Якутск
}

# Типичное давление у поверхности при «стандартной» атмосфере не фиксировано:
# для Алдана (~679 м) по ISA ≈ 935 гПа; в данных обычно 920–945 гПа.
ALDAN_TYPICAL_SURFACE_HPA = 935.0


def station_elevation_m(station_id: str | None) -> float | None:
    if not station_id:
        return None
    key = str(station_id).zfill(5)[-5:]
    return STATION_ELEVATION_M.get(key)


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def observed_or_geopot_height_m(
    *,
    height_m: float | None,
    geopotential_height_m: float | None = None,
    geopotential_m2s2: float | None = None,
) -> float | None:
    """Приоритет: height_m → geopotential_height_m → Φ→z (MetPy)."""
    for candidate in (height_m, geopotential_height_m):
        h = _finite(candidate)
        if h is not None:
            return h
    phi = _finite(geopotential_m2s2)
    if phi is not None:
        return round(geopotential_to_height_m(phi), 1)
    return None


def barometric_height_m(
    pressure_hpa: float,
    *,
    surface_pressure_hpa: float,
    station_elevation_m: float | None = None,
) -> float:
    """Высота над уровнем моря: z_station + барометрический прирост от поверхности."""
    above_sfc = estimate_geopotential_height_m(
        pressure_hpa,
        surface_pressure_hpa=surface_pressure_hpa,
    )
    base = 0.0 if station_elevation_m is None else float(station_elevation_m)
    return round(base + above_sfc, 1)


def interpolate_heights_on_pressure(
    pressures: Sequence[float],
    heights: Sequence[float | None],
) -> list[float | None]:
    """Линейная интерполяция H по P (только внутри диапазона известных точек)."""
    n = len(pressures)
    if n == 0:
        return []
    p = np.asarray([float(x) for x in pressures], dtype=float)
    h = np.asarray(
        [np.nan if v is None else float(v) for v in heights],
        dtype=float,
    )
    known = ~np.isnan(h) & ~np.isnan(p)
    if known.sum() == 0:
        return [None] * n
    if known.sum() == 1:
        # одна точка — только она, без экстраполяции
        out: list[float | None] = [None] * n
        idx = int(np.flatnonzero(known)[0])
        out[idx] = round(float(h[idx]), 1)
        return out

    order = np.argsort(p[known])
    xp = p[known][order]
    fp = h[known][order]
    # уникальные P (при дублях — среднее H)
    uniq_p, inv = np.unique(xp, return_inverse=True)
    uniq_h = np.zeros_like(uniq_p)
    counts = np.zeros_like(uniq_p)
    for i, u in enumerate(inv):
        uniq_h[u] += fp[i]
        counts[u] += 1
    uniq_h /= counts

    interp = np.interp(p, uniq_p, uniq_h, left=np.nan, right=np.nan)
    return [None if np.isnan(v) else round(float(v), 1) for v in interp]


def fill_profile_level_heights(
    levels: list[dict[str, Any]],
    *,
    surface_pressure_hpa: float | None,
    station_id: str | None = None,
) -> list[dict[str, Any]]:
    """Добавляет height_obs_m, height_interp_m, height_baro_m и итоговый height_m."""
    if not levels:
        return levels

    elev = station_elevation_m(station_id)
    pressures = [_finite(lv.get("pressure_hpa")) for lv in levels]
    obs_heights: list[float | None] = []
    for lv in levels:
        obs_heights.append(
            observed_or_geopot_height_m(
                height_m=_finite(lv.get("height_m")),
                geopotential_height_m=_finite(lv.get("geopotential_height_m")),
                geopotential_m2s2=_finite(lv.get("geopotential_m2s2")),
            )
        )

    # для интерп. якоря: только «наблюдённые» (включая Φ→z)
    interp_heights = interpolate_heights_on_pressure(
        [p if p is not None else np.nan for p in pressures],
        obs_heights,
    )

    p_sfc = _finite(surface_pressure_hpa)
    if p_sfc is None:
        valid_p = [p for p in pressures if p is not None]
        p_sfc = max(valid_p) if valid_p else None

    out: list[dict[str, Any]] = []
    for lv, p, h_obs, h_interp in zip(levels, pressures, obs_heights, interp_heights):
        row = dict(lv)
        h_baro = None
        if p is not None and p_sfc is not None:
            h_baro = barometric_height_m(
                p,
                surface_pressure_hpa=p_sfc,
                station_elevation_m=elev,
            )
        row["height_obs_m"] = h_obs
        row["height_interp_m"] = h_interp
        row["height_baro_m"] = h_baro
        # итоговая рабочая высота: наблюдение → интерп → барометрия
        if h_obs is not None:
            row["height_m"] = h_obs
            row["height_source"] = "observed_or_geopot"
        elif h_interp is not None:
            row["height_m"] = h_interp
            row["height_source"] = "interp"
        elif h_baro is not None:
            row["height_m"] = h_baro
            row["height_source"] = "baro"
        else:
            row["height_m"] = None
            row["height_source"] = None
        out.append(row)
    return out


def fill_long_dataframe_heights(long_df, metrics_df=None, *, station_id_default: str | None = None):
    """Векторизова через groupby profile_id (pandas DataFrame)."""
    import pandas as pd

    if long_df is None or len(long_df) == 0:
        return long_df

    metrics_map: dict[str, Any] = {}
    if metrics_df is not None and len(metrics_df):
        for row in metrics_df.itertuples(index=False):
            metrics_map[str(row.profile_id)] = row

    pieces: list = []
    for profile_id, group in long_df.groupby("profile_id", sort=False):
        metric = metrics_map.get(str(profile_id))
        station_id = None
        p_sfc = None
        if metric is not None:
            station_id = str(getattr(metric, "station_id", "") or "")
            p_sfc = _finite(getattr(metric, "p_surface_hpa", None))
        if not station_id:
            station_id = str(group["station_id"].iloc[0]) if "station_id" in group.columns else station_id_default
        if p_sfc is None and "pressure_hpa" in group.columns:
            p_sfc = float(group["pressure_hpa"].max())

        levels = group.to_dict(orient="records")
        filled = fill_profile_level_heights(
            levels,
            surface_pressure_hpa=p_sfc,
            station_id=station_id,
        )
        pieces.append(pd.DataFrame(filled))

    if not pieces:
        return long_df
    return pd.concat(pieces, ignore_index=True)
