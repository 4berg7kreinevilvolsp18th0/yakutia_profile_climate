from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, theilslopes

from .config import AnalysisConfig, InversionConfig, LayerClassConfig

_GAP_V3 = None


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


def _gap_v3_detect():
    """Gap-v3 из основного репозитория: сортировка по height_m, не по давлению."""
    global _GAP_V3
    if _GAP_V3 is not None:
        return _GAP_V3
    repo = Path(__file__).resolve().parents[4]
    path = repo / "gdex_bufr" / "profile_climate" / "inversion_layers.py"
    if path.exists():
        spec = importlib.util.spec_from_file_location("_project_inversion_layers", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules["_project_inversion_layers"] = mod
        spec.loader.exec_module(mod)
        _GAP_V3 = mod.detect_inversion_layers_gap_v3
        return _GAP_V3
    from gdex_bufr.profile_climate.inversion_layers import detect_inversion_layers_gap_v3
    _GAP_V3 = detect_inversion_layers_gap_v3
    return _GAP_V3


def _collapse_duplicate_heights(
    z: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Одинаковая height_m внутри профиля → среднее T и P, затем снизу вверх."""
    valid = np.isfinite(z) & np.isfinite(t)
    z = np.asarray(z, dtype=float)[valid]
    t = np.asarray(t, dtype=float)[valid]
    p = np.asarray(p, dtype=float)[valid]
    if z.size == 0:
        return z, t, p
    order = np.argsort(z, kind="mergesort")
    z = z[order]
    t = t[order]
    p = p[order]
    change = np.ones(z.size, dtype=bool)
    change[1:] = z[1:] != z[:-1]
    starts = np.flatnonzero(change)
    ones = np.ones_like(t)
    counts = np.add.reduceat(ones, starts)
    t_out = np.add.reduceat(t, starts) / counts
    p_finite = np.where(np.isfinite(p), p, 0.0)
    p_ok = np.where(np.isfinite(p), ones, 0.0)
    p_sum = np.add.reduceat(p_finite, starts)
    p_n = np.add.reduceat(p_ok, starts)
    p_out = np.divide(p_sum, p_n, out=np.full(starts.size, np.nan), where=p_n > 0)
    return z[starts], t_out, p_out


def layer_geometry_qc(layers: pd.DataFrame) -> dict[str, int]:
    """Счётчики физически невозможных границ слоя."""
    if layers.empty:
        return {"layers": 0, "negative_depth": 0, "top_below_base": 0, "negative_base_agl": 0}
    depth = pd.to_numeric(layers["depth_m"], errors="coerce")
    base = pd.to_numeric(layers["base_height_agl_m"], errors="coerce")
    top = pd.to_numeric(layers["top_height_agl_m"], errors="coerce")
    return {
        "layers": int(len(layers)),
        "negative_depth": int((depth <= 0).fillna(False).sum()),
        "top_below_base": int((top <= base).fillna(False).sum()),
        "negative_base_agl": int((base < -1e-6).fillna(False).sum()),
    }


def compute_inversion_layers_pressure_order(
    df: pd.DataFrame,
    profile_qc: pd.DataFrame,
    config: AnalysisConfig,
) -> pd.DataFrame:
    """Старый поиск: сегменты в порядке давления (для QC «до исправления»)."""
    return _compute_inversion_layers_with_detector(
        df, profile_qc, config, method="pressure-order v2-layers"
    )


def compute_inversion_layers(
    df: pd.DataFrame,
    profile_qc: pd.DataFrame,
    config: AnalysisConfig,
) -> pd.DataFrame:
    """Слои gap-v3: height_m — основная вертикальная координата."""
    return _compute_inversion_layers_with_detector(
        df, profile_qc, config, method="height-primary gap-v3"
    )


def _compute_inversion_layers_with_detector(
    df: pd.DataFrame,
    profile_qc: pd.DataFrame,
    config: AnalysisConfig,
    *,
    method: str,
) -> pd.DataFrame:
    """Слои роста T с классификацией G/E/HE по высоте основания."""
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
    detect_v3 = _gap_v3_detect() if method.startswith("height-primary") else None
    for profile_id, idx in group_indices.items():
        q = qc_map[profile_id]
        if not bool(q.get("eligible_article", False)):
            continue
        idx = np.asarray(idx, dtype=int)
        p = pressure_all[idx]
        mask = np.isfinite(p) & (p >= top) & (p <= bottom)
        p_use = p[mask]
        t_use = temperature_all[idx][mask]
        h_use = height_all[idx][mask]
        if detect_v3 is not None:
            z, t, p_col = _collapse_duplicate_heights(h_use, t_use, p_use)
            detected = detect_v3(
                z,
                t,
                p_col,
                max_embedded_gap_m=float(config.layers.max_embedded_gap_m),
                min_strength_c=float(config.layers.min_strength_c),
                he_threshold_m=float(config.layers.he_threshold_m),
                surface_tolerance_m=float(config.layers.surface_tolerance_m),
            )
            z0 = float(z[0]) if z.size else 0.0
            layers = []
            for ly in detected:
                layers.append(
                    {
                        "position_type": ly.position_type,
                        "base_pressure_hpa": ly.base_pressure_hpa,
                        "top_pressure_hpa": ly.top_pressure_hpa,
                        "base_height_m": ly.base_height_m,
                        "top_height_m": ly.top_height_m,
                        "base_height_agl_m": ly.base_height_m - z0,
                        "top_height_agl_m": ly.top_height_m - z0,
                        "depth_m": ly.depth_m,
                        "delta_t_c": ly.delta_t_c,
                        "gamma_c_per_100m": ly.mean_gradient_c_100m,
                        "confirm_drop_c": np.nan,
                        "embedded_gap_count": ly.embedded_gap_count,
                        "embedded_gap_depth_total_m": ly.embedded_gap_depth_total_m,
                        "method": method,
                    }
                )
        else:
            layers = _detect_confirmed_layers_arrays(
                p_use, t_use, h_use, config.inversion, config.layers,
            )
            for layer in layers:
                layer["method"] = method
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
                "embedded_gap_count", "embedded_gap_depth_total_m", "method",
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


def _edges_with_overflow(bin_edges: Sequence[float], *, both_sides: bool = False) -> np.ndarray:
    edges = np.asarray(tuple(bin_edges), dtype=float)
    if both_sides and np.isfinite(edges[0]):
        edges = np.insert(edges, 0, -np.inf)
    if np.isfinite(edges[-1]):
        edges = np.append(edges, np.inf)
    return edges


def _bin_frame(edges: np.ndarray) -> pd.DataFrame:
    left = edges[:-1]
    right = edges[1:]
    center = np.where(
        ~np.isfinite(left),
        right,
        np.where(np.isfinite(right), (left + right) / 2.0, left),
    )
    return pd.DataFrame({"bin_left": left, "bin_right": right, "bin_center": center})


def _assign_bins(values: pd.Series, edges: np.ndarray) -> pd.Series:
    return pd.cut(values, bins=edges, right=False, include_lowest=True)


def height_count_table(
    layers: pd.DataFrame,
    *,
    bin_edges: Sequence[float],
    by_month: bool = True,
    by_type: bool = False,
    value_col: str = "top_height_agl_m",
    count_name: str = "count",
) -> pd.DataFrame:
    """Число инверсий по бинам высоты. Все бины на месте, включая overflow ≥ last."""
    edges = _edges_with_overflow(bin_edges)
    bins = _bin_frame(edges)
    use = layers.dropna(subset=[value_col]).copy()
    months = list(range(1, 13)) if by_month else [0]
    types = list(INVERSION_TYPES) if by_type else [None]
    grid_parts = []
    for month in months:
        for kind in types:
            part = bins.copy()
            part["month"] = month
            if by_type:
                part["position_type"] = kind
            grid_parts.append(part)
    grid = pd.concat(grid_parts, ignore_index=True)
    if use.empty:
        grid[count_name] = 0
        return grid.sort_values(
            [c for c in ["month", "position_type", "bin_left"] if c in grid.columns]
        ).reset_index(drop=True)

    use["bin"] = _assign_bins(use[value_col], edges)
    use = use[use["bin"].notna()].copy()
    if not by_month:
        use["month"] = 0
    group_cols = ["month"]
    if by_type:
        group_cols.append("position_type")
    group_cols.append("bin")
    counts = use.groupby(group_cols, observed=False).size().reset_index(name=count_name)
    counted = _bin_edges_table(counts, count_name)
    keys = ["month", "position_type", "bin_left"] if by_type else ["month", "bin_left"]
    out = grid.merge(counted[keys + [count_name]], on=keys, how="left")
    out[count_name] = out[count_name].fillna(0).astype(int)
    return out.sort_values(keys).reset_index(drop=True)


def compute_interval_gammas(
    df: pd.DataFrame,
    profile_qc: pd.DataFrame,
    config: AnalysisConfig,
) -> pd.DataFrame:
    """γ = 100·dT/dz по всем соседним интервалам пригодного профиля (и + и −)."""
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
    top = config.pressure_top_hpa
    bottom = config.pressure_bottom_hpa
    rows: list[dict[str, Any]] = []
    for profile_id, idx in group_indices.items():
        q = qc_map.get(profile_id)
        if q is None or not bool(q.get("eligible_article", False)):
            continue
        idx = np.asarray(idx, dtype=int)
        p = pressure_all[idx]
        mask = np.isfinite(p) & (p >= top) & (p <= bottom)
        z, t, _p = _collapse_duplicate_heights(height_all[idx][mask], temperature_all[idx][mask], p[mask])
        if z.size < 2:
            continue
        dz = np.diff(z)
        dt = np.diff(t)
        ok = (dz > 0) & np.isfinite(dz) & np.isfinite(dt)
        if not ok.any():
            continue
        gamma = 100.0 * dt[ok] / dz[ok]
        month = int(q["month"])
        year = int(q["year"])
        for value in gamma:
            rows.append(
                {
                    "profile_id": profile_id,
                    "year": year,
                    "month": month,
                    "gamma_c_per_100m": float(value),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["profile_id", "year", "month", "gamma_c_per_100m"])
    return pd.DataFrame.from_records(rows)


def gamma_count_table(
    layers: pd.DataFrame,
    *,
    bin_edges: Sequence[float],
    by_month: bool = False,
) -> pd.DataFrame:
    """Число интервалов по бинам γ, включая отрицательные и overflow с обеих сторон."""
    edges = _edges_with_overflow(bin_edges, both_sides=True)
    bins = _bin_frame(edges)
    use = layers.dropna(subset=["gamma_c_per_100m"]).copy()
    months = list(range(1, 13)) if by_month else [0]
    grid = pd.concat([bins.assign(month=m) for m in months], ignore_index=True)
    if use.empty:
        grid["days"] = 0
        return grid.sort_values(["month", "bin_left"]).reset_index(drop=True)
    if not by_month:
        use = use.copy()
        use["month"] = 0
    use["bin"] = _assign_bins(use["gamma_c_per_100m"], edges)
    use = use[use["bin"].notna()]
    counts = use.groupby(["month", "bin"], observed=False).size().reset_index(name="days")
    counted = _bin_edges_table(counts, "days")
    out = grid.merge(counted[["month", "bin_left", "days"]], on=["month", "bin_left"], how="left")
    out["days"] = out["days"].fillna(0).astype(int)
    return out.sort_values(["month", "bin_left"]).reset_index(drop=True)


def recurrence_percent_table(
    layers: pd.DataFrame,
    profile_qc: pd.DataFrame,
    *,
    bin_edges: Sequence[float],
    value_col: str,
    by_month: bool = False,
) -> pd.DataFrame:
    """Повторяемость слоёв в бине, % от пригодных профилей."""
    eligible = int(profile_qc["eligible_article"].sum()) if not profile_qc.empty else 0
    if by_month:
        eligible_by_month = (
            profile_qc[profile_qc["eligible_article"]]
            .groupby("month")
            .size()
            .reindex(range(1, 13), fill_value=0)
        )
    counts = height_count_table(
        layers,
        bin_edges=bin_edges,
        by_month=by_month,
        by_type=True,
        value_col=value_col,
        count_name="profiles",
    )
    if by_month:
        counts["eligible_profiles"] = counts["month"].map(eligible_by_month).fillna(0).astype(int)
    else:
        counts["eligible_profiles"] = eligible
    denom = counts["eligible_profiles"].replace(0, np.nan)
    counts["recurrence_percent"] = counts["profiles"] / denom * 100.0
    counts["recurrence_percent"] = counts["recurrence_percent"].fillna(0.0)
    return counts


def monthly_median_iqr_table(layers: pd.DataFrame, value_col: str) -> pd.DataFrame:
    use = layers.dropna(subset=[value_col]).copy()
    if use.empty:
        return pd.DataFrame(columns=["month", "position_type", "median", "q25", "q75"])
    rows = []
    for month in range(1, 13):
        for kind in INVERSION_TYPES:
            vals = use.loc[
                (use["month"] == month) & (use["position_type"] == kind),
                value_col,
            ]
            rows.append(
                {
                    "month": month,
                    "position_type": kind,
                    "median": float(vals.median()) if len(vals) else np.nan,
                    "q25": float(vals.quantile(0.25)) if len(vals) else np.nan,
                    "q75": float(vals.quantile(0.75)) if len(vals) else np.nan,
                    "n": int(len(vals)),
                }
            )
    return pd.DataFrame(rows)


def annual_median_table(layers: pd.DataFrame, value_col: str) -> pd.DataFrame:
    use = layers.dropna(subset=[value_col]).copy()
    if use.empty:
        return pd.DataFrame(columns=["year", "position_type", "median", "n"])
    out = (
        use.groupby(["year", "position_type"], sort=False)[value_col]
        .agg(median="median", n="size")
        .reset_index()
    )
    return out.sort_values(["position_type", "year"]).reset_index(drop=True)


def year_month_median_matrix(layers: pd.DataFrame, *, inversion_type: str, value_col: str) -> pd.DataFrame:
    use = layers[layers["position_type"] == inversion_type].dropna(subset=[value_col])
    if use.empty:
        return pd.DataFrame(columns=range(1, 13))
    return (
        use.groupby(["year", "month"])[value_col]
        .median()
        .reset_index()
        .pivot(index="year", columns="month", values=value_col)
        .reindex(columns=range(1, 13))
        .sort_index()
    )
