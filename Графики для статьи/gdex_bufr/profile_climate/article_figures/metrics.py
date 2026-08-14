from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, theilslopes

from .config import AnalysisConfig, InversionConfig, LayerClassConfig


SEASON_BY_MONTH = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}
SEASON_ORDER = ("DJF", "MAM", "JJA", "SON")
INVERSION_TYPES = ("G", "E", "HE")
TYPE_LABELS_RU = {"G": "Приземная (G)", "E": "Приподнятая (E)", "HE": "Высокая приподнятая (HE)"}
TYPE_LABELS_EN = {"G": "Surface (G)", "E": "Elevated (E)", "HE": "High elevated (HE)"}


@dataclass
class InversionResult:
    detected: bool = False
    candidate: bool = False
    quality: str = "none"
    top_pressure_hpa: float | None = None
    top_height_m: float | None = None
    top_temp_c: float | None = None
    delta_t_c: float | None = None
    confirm_drop_c: float | None = None


def _detect_arrays(
    pressure: np.ndarray,
    temperature: np.ndarray,
    height: np.ndarray,
    config: InversionConfig,
) -> InversionResult:
    valid = np.isfinite(pressure) & np.isfinite(temperature)
    pressure = pressure[valid]
    temperature = temperature[valid]
    height = height[valid]
    if pressure.size < 2:
        return InversionResult()
    order = np.argsort(-pressure, kind="stable")
    pressure = pressure[order]
    temperature = temperature[order]
    height = height[order]

    surface_temp = float(temperature[0])
    top_index = 0
    growing = False
    for idx in range(1, pressure.size):
        delta = float(temperature[idx] - temperature[top_index])
        if delta > config.min_inversion_delta_c:
            top_index = idx
            growing = True
        else:
            break
    if not growing:
        return InversionResult()

    top_temp = float(temperature[top_index])
    top_pressure = float(pressure[top_index])
    top_height = float(height[top_index]) if np.isfinite(height[top_index]) else None
    above_start = top_index + 1
    above_count = pressure.size - above_start

    def candidate(quality: str, confirm_drop: float | None) -> InversionResult:
        return InversionResult(
            detected=quality == "confirmed",
            candidate=True,
            quality=quality,
            top_pressure_hpa=top_pressure,
            top_height_m=top_height,
            top_temp_c=top_temp,
            delta_t_c=top_temp - surface_temp,
            confirm_drop_c=confirm_drop,
        )

    if above_count < config.confirm_drop_levels:
        return candidate("rejected_no_lapse", None)

    prev_temp = top_temp
    for idx in range(above_start, above_start + config.confirm_drop_levels):
        current = float(temperature[idx])
        if current - prev_temp > -config.min_drop_delta_c:
            return candidate("rejected_no_lapse", None)
        prev_temp = current

    depth_indices = np.flatnonzero(top_pressure - pressure[above_start:] >= config.confirm_depth_hpa)
    if depth_indices.size == 0:
        return candidate("rejected_no_lapse", None)
    end_idx = above_start + int(depth_indices[0])
    confirm_drop = float(temperature[end_idx] - top_temp)
    if confirm_drop >= 0:
        return candidate("rejected_no_lapse", confirm_drop)
    return candidate("confirmed", confirm_drop)


def detect_surface_inversion_v2(levels: pd.DataFrame, config: InversionConfig) -> InversionResult:
    """Публичная DataFrame-обёртка для тестов и единичных профилей."""
    pressure = pd.to_numeric(levels["pressure_hpa"], errors="coerce").to_numpy(float)
    temperature = pd.to_numeric(levels["temperature_c"], errors="coerce").to_numpy(float)
    height = (
        pd.to_numeric(levels["height_m"], errors="coerce").to_numpy(float)
        if "height_m" in levels.columns
        else np.full(len(levels), np.nan)
    )
    return _detect_arrays(pressure, temperature, height, config)


def compute_inversion_metrics(
    df: pd.DataFrame,
    profile_qc: pd.DataFrame,
    config: AnalysisConfig,
) -> pd.DataFrame:
    """Рассчитать v2-инверсию для каждого профиля.

    Используются numpy-массивы и индексы групп, поэтому полный массив Алдана
    обрабатывается за секунды, а не за минуты.
    """
    profile_ids = df["profile_id"].astype(str).to_numpy()
    pressure_all = df["pressure_hpa"].to_numpy(float)
    temperature_all = df["temperature_c"].to_numpy(float)
    height_all = df["height_m"].to_numpy(float)
    group_indices = df.assign(_profile_id_str=profile_ids).groupby("_profile_id_str", sort=False).indices
    qc_map = profile_qc.assign(profile_id=profile_qc["profile_id"].astype(str)).set_index("profile_id").to_dict("index")

    rows: list[dict[str, Any]] = []
    top = config.pressure_top_hpa
    bottom = config.pressure_bottom_hpa
    for profile_id, idx in group_indices.items():
        q = qc_map[profile_id]
        idx = np.asarray(idx, dtype=int)
        p = pressure_all[idx]
        mask = np.isfinite(p) & (p >= top) & (p <= bottom)
        p = p[mask]
        t = temperature_all[idx][mask]
        h = height_all[idx][mask]
        result = _detect_arrays(p, t, h, config.inversion)
        if p.size:
            surface_idx = int(np.nanargmax(p))
            surface_temp = float(t[surface_idx]) if np.isfinite(t[surface_idx]) else np.nan
            surface_height = float(h[surface_idx]) if np.isfinite(h[surface_idx]) else np.nan
        else:
            surface_temp = surface_height = np.nan
        top_height_agl = (
            result.top_height_m - surface_height
            if result.top_height_m is not None and np.isfinite(surface_height)
            else np.nan
        )
        rows.append(
            {
                "profile_id": profile_id,
                "datetime_utc": q["datetime_utc"],
                "year": int(q["year"]),
                "month": int(q["month"]),
                "cycle": str(q["cycle"]),
                "profile_status": q["profile_status"],
                "strict_surface_ok": bool(q["strict_surface_ok"]),
                "eligible_article": bool(q["eligible_article"]),
                "surface_temperature_c": surface_temp,
                "inversion_detected": result.detected,
                "inversion_candidate": result.candidate,
                "inversion_quality": result.quality,
                "inversion_top_pressure_hpa": result.top_pressure_hpa,
                "inversion_top_height_m": result.top_height_m,
                "inversion_top_height_agl_m": top_height_agl,
                "inversion_top_temp_c": result.top_temp_c,
                "inversion_delta_t_c": result.delta_t_c,
                "inversion_confirm_drop_c": result.confirm_drop_c,
            }
        )
    return pd.DataFrame.from_records(rows).sort_values("datetime_utc").reset_index(drop=True)


def _interpolate_arrays(pressure: np.ndarray, temperature: np.ndarray, grid: np.ndarray) -> np.ndarray:
    valid = np.isfinite(pressure) & np.isfinite(temperature)
    p = pressure[valid]
    t = temperature[valid]
    if p.size < 2:
        return np.full(grid.shape, np.nan, dtype=float)
    order = np.argsort(p, kind="stable")
    p = p[order]
    t = t[order]
    # QC-пригодные профили дублей не имеют; unique оставляет функцию устойчивой.
    p, unique_idx = np.unique(p, return_index=True)
    t = t[unique_idx]
    if p.size < 2:
        return np.full(grid.shape, np.nan, dtype=float)
    values = np.interp(grid, p, t)
    values[(grid < p[0]) | (grid > p[-1])] = np.nan
    return values


def interpolate_eligible_profiles(
    df: pd.DataFrame,
    profile_qc: pd.DataFrame,
    pressure_grid_hpa: Iterable[float],
    *,
    cycles: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Одна таблица интерполяции для повторного использования разными графиками."""
    grid = np.asarray(tuple(float(x) for x in pressure_grid_hpa), dtype=float)
    eligible = profile_qc[profile_qc["eligible_article"]].copy()
    if cycles is not None:
        wanted = {str(x).zfill(2) for x in cycles}
        eligible = eligible[eligible["cycle"].isin(wanted)]
    meta = eligible.assign(profile_id=eligible["profile_id"].astype(str)).set_index("profile_id")[["year", "month", "cycle"]].to_dict("index")
    eligible_ids = set(meta)

    profile_ids = df["profile_id"].astype(str).to_numpy()
    pressure_all = df["pressure_hpa"].to_numpy(float)
    temperature_all = df["temperature_c"].to_numpy(float)
    group_indices = df.assign(_profile_id_str=profile_ids).groupby("_profile_id_str", sort=False).indices

    chunks = []
    for profile_id, idx in group_indices.items():
        if profile_id not in eligible_ids:
            continue
        idx = np.asarray(idx, dtype=int)
        values = _interpolate_arrays(pressure_all[idx], temperature_all[idx], grid)
        valid = np.isfinite(values)
        if not valid.any():
            continue
        m = meta[profile_id]
        chunks.append(
            pd.DataFrame(
                {
                    "profile_id": profile_id,
                    "year": int(m["year"]),
                    "month": int(m["month"]),
                    "cycle": str(m["cycle"]),
                    "pressure_hpa": grid[valid],
                    "temperature_c": values[valid],
                }
            )
        )
    if not chunks:
        return pd.DataFrame(columns=["profile_id", "year", "month", "cycle", "pressure_hpa", "temperature_c"])
    return pd.concat(chunks, ignore_index=True)


def compute_seasonal_climatology(
    df: pd.DataFrame,
    profile_qc: pd.DataFrame,
    config: AnalysisConfig,
    *,
    cycles: Iterable[str] | None = None,
    interpolated: pd.DataFrame | None = None,
) -> pd.DataFrame:
    interpolated = interpolated if interpolated is not None else interpolate_eligible_profiles(
        df, profile_qc, config.pressure_grid_hpa, cycles=cycles or config.cycles
    )
    if interpolated.empty:
        raise ValueError("Не удалось построить интерполированные профили")
    data = interpolated.copy()
    data["season"] = data["month"].map(SEASON_BY_MONTH)
    out = (
        data.groupby(["season", "pressure_hpa"])["temperature_c"]
        .agg(
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            mean="mean",
            count="count",
        )
        .reset_index()
    )
    out["season"] = pd.Categorical(out["season"], categories=SEASON_ORDER, ordered=True)
    return out.sort_values(["season", "pressure_hpa"]).reset_index(drop=True)


def monthly_inversion_frequency(metrics: pd.DataFrame) -> pd.DataFrame:
    use = metrics[metrics["eligible_article"]].copy()
    return (
        use.groupby(["month", "cycle"])["inversion_detected"]
        .agg(profiles="size", inversions="sum", frequency_percent="mean")
        .reset_index()
        .assign(frequency_percent=lambda x: x["frequency_percent"] * 100.0)
    )


def annual_inversion_frequency(
    metrics: pd.DataFrame,
    *,
    start_year: int,
    end_year: int,
    moving_window: int = 5,
) -> tuple[pd.DataFrame, dict[str, float]]:
    use = metrics[
        metrics["eligible_article"]
        & metrics["year"].between(start_year, end_year, inclusive="both")
    ].copy()
    annual = (
        use.groupby("year")["inversion_detected"]
        .agg(profiles="size", inversions="sum", frequency_percent="mean")
        .reset_index()
        .assign(frequency_percent=lambda x: x["frequency_percent"] * 100.0)
    )
    annual["moving_average_percent"] = annual["frequency_percent"].rolling(
        moving_window, center=True, min_periods=max(2, moving_window // 2)
    ).mean()
    if len(annual) >= 3:
        slope, intercept, low, high = theilslopes(annual["frequency_percent"], annual["year"], 0.95)
        tau, p_value = kendalltau(annual["year"], annual["frequency_percent"])
        annual["sen_trend_percent"] = intercept + slope * annual["year"]
        stats = {
            "sen_slope_pp_per_year": float(slope),
            "sen_slope_low": float(low),
            "sen_slope_high": float(high),
            "kendall_tau": float(tau),
            "p_value": float(p_value),
        }
    else:
        annual["sen_trend_percent"] = np.nan
        stats = {k: np.nan for k in ["sen_slope_pp_per_year", "sen_slope_low", "sen_slope_high", "kendall_tau", "p_value"]}
    return annual, stats


def pressure_level_annual_series(
    df: pd.DataFrame,
    profile_qc: pd.DataFrame,
    config: AnalysisConfig,
    *,
    levels_hpa: Iterable[float] | None = None,
    interpolated: pd.DataFrame | None = None,
) -> pd.DataFrame:
    levels = tuple(float(x) for x in (levels_hpa or config.standard_pressure_levels_hpa))
    if interpolated is None:
        data = interpolate_eligible_profiles(df, profile_qc, levels, cycles=config.cycles)
    else:
        data = interpolated[interpolated["pressure_hpa"].isin(levels)].copy()
    return (
        data.groupby(["year", "pressure_hpa"])["temperature_c"]
        .agg(median_temperature_c="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75), profiles="size")
        .reset_index()
    )


def _confirm_segment(
    pressure: np.ndarray,
    temperature: np.ndarray,
    top_index: int,
    config: InversionConfig,
) -> tuple[bool, float | None]:
    top_temp = float(temperature[top_index])
    top_pressure = float(pressure[top_index])
    above_start = top_index + 1
    above_count = pressure.size - above_start
    if above_count < config.confirm_drop_levels:
        return False, None
    prev_temp = top_temp
    for idx in range(above_start, above_start + config.confirm_drop_levels):
        current = float(temperature[idx])
        if current - prev_temp > -config.min_drop_delta_c:
            return False, None
        prev_temp = current
    depth_indices = np.flatnonzero(top_pressure - pressure[above_start:] >= config.confirm_depth_hpa)
    if depth_indices.size == 0:
        return False, None
    end_idx = above_start + int(depth_indices[0])
    confirm_drop = float(temperature[end_idx] - top_temp)
    if confirm_drop >= 0:
        return False, confirm_drop
    return True, confirm_drop


def _classify_position(
    base_height_agl_m: float,
    layers_cfg: LayerClassConfig,
) -> str:
    if base_height_agl_m <= layers_cfg.surface_tolerance_m:
        return "G"
    if base_height_agl_m <= layers_cfg.he_threshold_m:
        return "E"
    return "HE"


def _detect_confirmed_layers_arrays(
    pressure: np.ndarray,
    temperature: np.ndarray,
    height: np.ndarray,
    inversion_cfg: InversionConfig,
    layers_cfg: LayerClassConfig,
) -> list[dict[str, Any]]:
    valid = np.isfinite(pressure) & np.isfinite(temperature)
    pressure = pressure[valid]
    temperature = temperature[valid]
    height = height[valid]
    if pressure.size < 2:
        return []
    order = np.argsort(-pressure, kind="stable")
    pressure = pressure[order]
    temperature = temperature[order]
    height = height[order]
    surface_height = float(height[0]) if np.isfinite(height[0]) else np.nan

    layers: list[dict[str, Any]] = []
    i = 0
    n = pressure.size
    while i < n - 1:
        if float(temperature[i + 1]) - float(temperature[i]) > inversion_cfg.min_inversion_delta_c:
            base = i
            j = i
            while j < n - 1:
                if float(temperature[j + 1]) - float(temperature[j]) > inversion_cfg.min_inversion_delta_c:
                    j += 1
                else:
                    break
            if j > base:
                ok, confirm_drop = _confirm_segment(pressure, temperature, j, inversion_cfg)
                if ok:
                    base_h = float(height[base]) if np.isfinite(height[base]) else np.nan
                    top_h = float(height[j]) if np.isfinite(height[j]) else np.nan
                    base_agl = (
                        base_h - surface_height
                        if np.isfinite(base_h) and np.isfinite(surface_height)
                        else np.nan
                    )
                    top_agl = (
                        top_h - surface_height
                        if np.isfinite(top_h) and np.isfinite(surface_height)
                        else np.nan
                    )
                    depth = top_agl - base_agl if np.isfinite(top_agl) and np.isfinite(base_agl) else np.nan
                    delta_t = float(temperature[j] - temperature[base])
                    gamma = (
                        (delta_t / depth) * 100.0
                        if np.isfinite(depth) and depth > 1.0
                        else np.nan
                    )
                    pos = (
                        _classify_position(float(base_agl), layers_cfg)
                        if np.isfinite(base_agl)
                        else "E"
                    )
                    layers.append(
                        {
                            "position_type": pos,
                            "base_pressure_hpa": float(pressure[base]),
                            "top_pressure_hpa": float(pressure[j]),
                            "base_height_m": base_h,
                            "top_height_m": top_h,
                            "base_height_agl_m": base_agl,
                            "top_height_agl_m": top_agl,
                            "depth_m": depth,
                            "delta_t_c": delta_t,
                            "gamma_c_per_100m": gamma,
                            "confirm_drop_c": confirm_drop,
                        }
                    )
                i = j if j > base else i + 1
            else:
                i += 1
        else:
            i += 1
    return layers


def compute_inversion_layers(
    df: pd.DataFrame,
    profile_qc: pd.DataFrame,
    config: AnalysisConfig,
) -> pd.DataFrame:
    """Подтверждённые слои роста T с классификацией G/E/HE по высоте основания."""
    profile_ids = df["profile_id"].astype(str).to_numpy()
    pressure_all = df["pressure_hpa"].to_numpy(float)
    temperature_all = df["temperature_c"].to_numpy(float)
    height_all = df["height_m"].to_numpy(float)
    group_indices = df.assign(_profile_id_str=profile_ids).groupby("_profile_id_str", sort=False).indices
    qc_map = (
        profile_qc.assign(profile_id=profile_qc["profile_id"].astype(str))
        .set_index("profile_id")
        .to_dict("index")
    )

    rows: list[dict[str, Any]] = []
    top = config.pressure_top_hpa
    bottom = config.pressure_bottom_hpa
    for profile_id, idx in group_indices.items():
        q = qc_map[profile_id]
        if not bool(q.get("eligible_article", False)):
            continue
        idx = np.asarray(idx, dtype=int)
        p = pressure_all[idx]
        mask = np.isfinite(p) & (p >= top) & (p <= bottom)
        layers = _detect_confirmed_layers_arrays(
            p[mask],
            temperature_all[idx][mask],
            height_all[idx][mask],
            config.inversion,
            config.layers,
        )
        for layer_index, layer in enumerate(layers):
            rows.append(
                {
                    "profile_id": profile_id,
                    "datetime_utc": q["datetime_utc"],
                    "year": int(q["year"]),
                    "month": int(q["month"]),
                    "cycle": str(q["cycle"]),
                    "layer_index": layer_index,
                    **layer,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "profile_id", "datetime_utc", "year", "month", "cycle", "layer_index",
                "position_type", "base_pressure_hpa", "top_pressure_hpa",
                "base_height_m", "top_height_m", "base_height_agl_m", "top_height_agl_m",
                "depth_m", "delta_t_c", "gamma_c_per_100m", "confirm_drop_c",
            ]
        )
    return pd.DataFrame.from_records(rows).sort_values(
        ["datetime_utc", "layer_index"]
    ).reset_index(drop=True)


def profile_type_flags(layers: pd.DataFrame, profile_qc: pd.DataFrame) -> pd.DataFrame:
    """Профиль × флаги наличия G/E/HE (для матриц повторяемости)."""
    eligible = profile_qc[profile_qc["eligible_article"]].copy()
    eligible["profile_id"] = eligible["profile_id"].astype(str)
    out = eligible[["profile_id", "year", "month", "cycle"]].copy()
    for kind in INVERSION_TYPES:
        out[f"has_{kind}"] = False
    if layers.empty:
        return out
    present = (
        layers.groupby(["profile_id", "position_type"], sort=False)
        .size()
        .reset_index(name="n")
    )
    for kind in INVERSION_TYPES:
        ids = set(present.loc[present["position_type"] == kind, "profile_id"].astype(str))
        out[f"has_{kind}"] = out["profile_id"].isin(ids)
    return out


def frequency_matrix_by_type(
    flags: pd.DataFrame,
    *,
    inversion_type: str,
) -> pd.DataFrame:
    """Матрица год×месяц: % профилей с данным типом инверсии."""
    col = f"has_{inversion_type}"
    if col not in flags.columns:
        raise ValueError(f"Нет колонки {col}")
    pivot = (
        flags.groupby(["year", "month"])[col]
        .mean()
        .mul(100.0)
        .reset_index()
        .pivot(index="year", columns="month", values=col)
        .reindex(columns=range(1, 13))
        .sort_index()
    )
    return pivot


def _bin_edges_table(counts: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Добавляет bin_left/right/center из колонки Interval."""
    intervals = counts["bin"]
    left = []
    right = []
    for item in intervals:
        if pd.isna(item):
            left.append(np.nan)
            right.append(np.nan)
        else:
            left.append(float(item.left))
            right.append(float(item.right))
    out = counts.drop(columns=["bin"]).copy()
    out["bin_left"] = np.asarray(left, dtype=float)
    out["bin_right"] = np.asarray(right, dtype=float)
    out["bin_center"] = (out["bin_left"] + out["bin_right"]) / 2.0
    return out


def height_count_table(
    layers: pd.DataFrame,
    *,
    bin_edges: Sequence[float],
    by_month: bool = True,
) -> pd.DataFrame:
    """Число инверсий по бинам высоты верха AGL."""
    edges = np.asarray(tuple(bin_edges), dtype=float)
    use = layers.dropna(subset=["top_height_agl_m"]).copy()
    if use.empty:
        return pd.DataFrame(columns=["month", "bin_left", "bin_right", "bin_center", "count"])
    use["bin"] = pd.cut(use["top_height_agl_m"], bins=edges, right=False, include_lowest=True)
    group_cols = ["month", "bin"] if by_month else ["bin"]
    counts = use.groupby(group_cols, observed=True).size().reset_index(name="count")
    if not by_month:
        counts["month"] = 0
    out = _bin_edges_table(counts, "count")
    return out.sort_values(["month", "bin_left"]).reset_index(drop=True)


def gamma_count_table(
    layers: pd.DataFrame,
    *,
    bin_edges: Sequence[float],
    by_month: bool = False,
) -> pd.DataFrame:
    """Число дней/слоёв по бинам температурного градиента γ (°C/100 м)."""
    edges = np.asarray(tuple(bin_edges), dtype=float)
    use = layers.dropna(subset=["gamma_c_per_100m"]).copy()
    if use.empty:
        return pd.DataFrame(columns=["month", "bin_left", "bin_right", "bin_center", "days"])
    # один профиль — один день; берём максимальный γ среди слоёв профиля
    per_day = (
        use.groupby(["profile_id", "year", "month"], sort=False)["gamma_c_per_100m"]
        .max()
        .reset_index()
    )
    per_day["bin"] = pd.cut(per_day["gamma_c_per_100m"], bins=edges, right=False, include_lowest=True)
    group_cols = ["month", "bin"] if by_month else ["bin"]
    counts = per_day.groupby(group_cols, observed=True).size().reset_index(name="days")
    if not by_month:
        counts["month"] = 0
    out = _bin_edges_table(counts, "days")
    return out.sort_values(["month", "bin_left"]).reset_index(drop=True)
