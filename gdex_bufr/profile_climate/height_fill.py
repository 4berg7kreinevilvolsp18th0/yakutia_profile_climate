"""Заполнение высоты уровней: наблюдённая / Φ→z / линейная интерп / барометрия."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from gdex_bufr.meteo_parser_bridge import (
    _EARTH_RADIUS_M,
    _G0_M_S2,
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


def _pick_observed_height(
    level: dict[str, Any],
    *,
    station_elevation_m: float | None,
) -> tuple[float | None, str | None]:
    """Берёт высоту уровня из наблюдений / Φ / высоты станции.

    Порядок: 010009 → 007007 → Φ→z → height_m/geopot → высота станции для SFC.
    """
    direct = _finite(level.get("height_010009_m"))
    if direct is None:
        direct = _finite(level.get("height_007007_m"))
    if direct is not None:
        return direct, "level"

    phi = _finite(level.get("geopotential_m2s2"))
    if phi is not None:
        return round(geopotential_to_height_m(phi), 1), "phi"

    height = observed_or_geopot_height_m(
        height_m=_finite(level.get("height_m")),
        geopotential_height_m=_finite(level.get("geopotential_height_m")),
    )
    if height is not None:
        return height, "observed_or_geopot"

    if str(level.get("VSIG") or "").upper() == "SFC" and station_elevation_m is not None:
        return station_elevation_m, "station_007001"
    return None, None


def _choose_final_height(
    h_obs: float | None,
    h_source: str | None,
    h_interp: float | None,
    h_baro: float | None,
) -> tuple[float | None, str | None]:
    """Итоговая высота: наблюдение → интерполяция → барометрия."""
    if h_obs is not None:
        return h_obs, h_source
    if h_interp is not None:
        return h_interp, "interp"
    if h_baro is not None:
        return h_baro, "baro"
    return None, None


def fill_profile_level_heights(
    levels: list[dict[str, Any]],
    *,
    surface_pressure_hpa: float | None,
    station_id: str | None = None,
    station_elevation_override_m: float | None = None,
) -> list[dict[str, Any]]:
    """Добавляет height_obs_m, height_interp_m, height_baro_m и итоговый height_m."""
    if not levels:
        return levels

    elev = _finite(station_elevation_override_m)
    if elev is None:
        elev = station_elevation_m(station_id)
    pressures = [_finite(lv.get("pressure_hpa")) for lv in levels]

    obs_heights: list[float | None] = []
    obs_sources: list[str | None] = []
    for lv in levels:
        height, source = _pick_observed_height(lv, station_elevation_m=elev)
        obs_heights.append(height)
        obs_sources.append(source)

    # Интерполяция только по «наблюдённым» точкам (включая Φ→z).
    interp_heights = interpolate_heights_on_pressure(
        [p if p is not None else np.nan for p in pressures],
        obs_heights,
    )

    p_sfc = _finite(surface_pressure_hpa)
    if p_sfc is None:
        valid_p = [p for p in pressures if p is not None]
        p_sfc = max(valid_p) if valid_p else None

    out: list[dict[str, Any]] = []
    for lv, p, h_obs, h_source, h_interp in zip(
        levels, pressures, obs_heights, obs_sources, interp_heights
    ):
        row = dict(lv)
        h_baro = None
        if p is not None and p_sfc is not None:
            h_baro = barometric_height_m(
                p,
                surface_pressure_hpa=p_sfc,
                station_elevation_m=elev,
            )
        final_h, final_source = _choose_final_height(h_obs, h_source, h_interp, h_baro)
        row["height_obs_m"] = h_obs
        row["height_interp_m"] = h_interp
        row["height_baro_m"] = h_baro
        row["height_m"] = final_h
        row["height_source"] = final_source
        row["height_msl_m"] = final_h
        row["height_agl_m"] = (
            None if final_h is None or elev is None else round(float(final_h) - float(elev), 1)
        )
        out.append(row)
    return out


def fill_long_dataframe_heights(long_df, metrics_df=None, *, station_id_default: str | None = None):
    """Заполняет height_* для DataFrame profiles_long (быстрый путь на numpy)."""
    import pandas as pd

    if long_df is None or len(long_df) == 0:
        return long_df

    df = long_df.copy()
    metrics_map: dict[str, Any] = {}
    if metrics_df is not None and len(metrics_df):
        for row in metrics_df.itertuples(index=False):
            metrics_map[str(row.profile_id)] = {
                "station_id": str(getattr(row, "station_id", "") or ""),
                "p_surface_hpa": _finite(getattr(row, "p_surface_hpa", None)),
                "station_elevation_m": _finite(
                    getattr(row, "station_elevation_m", None)
                ),
            }

    n = len(df)
    height_interp = np.full(n, np.nan)
    height_baro = np.full(n, np.nan)
    height_final = np.full(n, np.nan)
    elevation_by_row = np.full(n, np.nan)
    height_source = np.array([None] * n, dtype=object)

    # наблюдаемая / Φ→z (аналитика MetPy, векторно — без вызова MetPy на каждую строку)
    h_col = df["height_m"].to_numpy(dtype=float) if "height_m" in df.columns else np.full(n, np.nan)
    gh_col = (
        df["geopotential_height_m"].to_numpy(dtype=float)
        if "geopotential_height_m" in df.columns
        else np.full(n, np.nan)
    )
    phi_col = (
        df["geopotential_m2s2"].to_numpy(dtype=float)
        if "geopotential_m2s2" in df.columns
        else np.full(n, np.nan)
    )
    height_obs = np.where(~np.isnan(h_col), h_col, gh_col)
    need_phi = np.isnan(height_obs) & ~np.isnan(phi_col)
    if need_phi.any():
        phi = phi_col[need_phi]
        denom = _G0_M_S2 * _EARTH_RADIUS_M - phi
        z_phi = np.where(np.abs(denom) < 1e-9, np.nan, (phi * _EARTH_RADIUS_M) / denom)
        height_obs[need_phi] = z_phi
    # явный брак (напр. Φ<0 у поверхности) → дальше interp/baro
    height_obs = np.where(height_obs < -50.0, np.nan, height_obs)

    p_all = df["pressure_hpa"].to_numpy(dtype=float)
    pid_all = df["profile_id"].astype(str).to_numpy()
    sid_all = (
        df["station_id"].astype(str).to_numpy()
        if "station_id" in df.columns
        else np.array([station_id_default or ""] * n)
    )

    # индексы по profile_id
    order = np.argsort(pid_all, kind="mergesort")
    pid_sorted = pid_all[order]
    breaks = np.flatnonzero(pid_sorted[1:] != pid_sorted[:-1]) + 1
    starts = np.r_[0, breaks]
    ends = np.r_[breaks, len(pid_sorted)]

    for a, b in zip(starts, ends):
        idx = order[a:b]
        pid = pid_sorted[a]
        meta = metrics_map.get(pid, {})
        station_id = meta.get("station_id") or sid_all[idx[0]] or station_id_default
        elev = meta.get("station_elevation_m")
        if elev is None:
            elev = station_elevation_m(station_id)
        if elev is not None:
            elevation_by_row[idx] = float(elev)
        p = p_all[idx]
        h_obs = height_obs[idx]
        p_sfc = meta.get("p_surface_hpa")
        if p_sfc is None:
            finite_p = p[~np.isnan(p)]
            p_sfc = float(np.max(finite_p)) if len(finite_p) else None

        interp = interpolate_heights_on_pressure(
            [float(x) if not np.isnan(x) else np.nan for x in p],
            [None if np.isnan(x) else float(x) for x in h_obs],
        )
        for j, row_i in enumerate(idx):
            hi = interp[j]
            if hi is not None:
                height_interp[row_i] = hi
            hb = None
            if p_sfc is not None and not np.isnan(p[j]):
                hb = barometric_height_m(
                    float(p[j]),
                    surface_pressure_hpa=float(p_sfc),
                    station_elevation_m=elev,
                )
                height_baro[row_i] = hb
            if not np.isnan(h_obs[j]):
                height_final[row_i] = h_obs[j]
                height_source[row_i] = "observed_or_geopot"
            elif hi is not None:
                height_final[row_i] = hi
                height_source[row_i] = "interp"
            elif hb is not None:
                height_final[row_i] = hb
                height_source[row_i] = "baro"

    df["height_obs_m"] = height_obs
    df["height_interp_m"] = height_interp
    df["height_baro_m"] = height_baro
    df["height_m"] = height_final
    df["height_msl_m"] = height_final
    df["height_agl_m"] = height_final - elevation_by_row
    df["height_source"] = height_source
    return df
