"""Сборка профилей наблюдений (зондов) для интерактивного дашборда."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gdex_bufr.profile_climate.height_fill import (  # noqa: E402
    ALDAN_TYPICAL_SURFACE_HPA,
    STATION_ELEVATION_M,
    fill_long_dataframe_heights,
)
from gdex_bufr.profile_climate.paths import catalog_station_dir  # noqa: E402
from gdex_bufr.profile_climate.obs_qc import (  # noqa: E402
    clean_observation_levels,
    interp_on_pressure_grid,
)

GRID_POINTS = 40
MAX_SURFACE_PRESSURE_HPA = 1000.0
PRESSURE_TOP_HPA = 500.0
PLOT_MIN_LEVELS = 3
SCHEMA = "observations_v1"
LEVEL_MODES = ("raw", "clean")

# Необязательные возможности сборки: дашборд включает элементы UI только при их наличии,
# поэтому схема остаётся observations_v1 и старые JSON продолжают открываться.
FEATURES = (
    "inversion_quality",      # inversion_quality / inversion_candidate / confirm_drop
    "height_variants",        # heights_interp_m / heights_baro_m на каждом наблюдении
    "height_source_counts",   # состав источников высоты внутри зонда
    "surface_context",        # p_surface_hpa / station_elevation_m
    "inversion_v3",           # inversion_layers_v3 / pattern / n_layers (gap-merge)
)

DEFAULT_DIR = catalog_station_dir()
LEGACY_DIR = Path("gdex_outputs") / "актуальное"
DEFAULT_LONG_CSV = DEFAULT_DIR / "profiles_long.csv"
DEFAULT_METRICS_CSV = DEFAULT_DIR / "profile_metrics.csv"
DEFAULT_OUTPUT = DEFAULT_DIR / "daily_profiles.json"


def _day_key(dt: str) -> str:
    parsed = datetime.fromisoformat(str(dt).replace("Z", "+00:00"))
    return parsed.date().isoformat()


def _finite_metric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:
        return None
    return f


def _metric_flag(metric: Any, name: str) -> bool:
    value = getattr(metric, name, False)
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "да"}
    return bool(value)


def _metric_inversion_fields(metric: Any) -> dict[str, Any]:
    """Поля инверсии из метрик, включая семантику v2 (quality / candidate)."""
    h = _finite_metric(getattr(metric, "inversion_top_height_m", None))
    p = _finite_metric(getattr(metric, "inversion_top_pressure_hpa", None))
    t = _finite_metric(getattr(metric, "inversion_top_temp_c", None))
    d = _finite_metric(getattr(metric, "inversion_delta_t_c", None))
    drop = _finite_metric(getattr(metric, "inversion_confirm_drop_c", None))
    quality = getattr(metric, "inversion_quality", None)
    try:
        if quality is None or pd.isna(quality):
            quality = ""
    except (TypeError, ValueError):
        pass
    return {
        "inversion_top_height_m": None if h is None else round(h, 1),
        "inversion_top_pressure_hpa": None if p is None else round(p, 1),
        "inversion_top_temp_c": None if t is None else round(t, 2),
        "inversion_delta_t_c": None if d is None else round(d, 2),
        "inversion_confirm_drop_c": None if drop is None else round(drop, 2),
        "inversion_quality": str(quality or ""),
        "inversion_candidate": _metric_flag(metric, "inversion_candidate"),
    }


def _empty_v3_fields() -> dict[str, Any]:
    return {
        "inversion_layers_v3": [],
        "n_inversion_layers_v3": 0,
        "inversion_pattern_v3": "NONE",
        "has_G_v3": False,
        "has_E_v3": False,
        "has_HE_v3": False,
        "strongest_delta_t_c_v3": None,
    }


def _v3_fields_from_maps(
    profile_id: str,
    layers_by_profile: dict[str, list[dict[str, Any]]] | None,
    summary_by_profile: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    if not layers_by_profile and not summary_by_profile:
        return _empty_v3_fields()
    layers = (layers_by_profile or {}).get(profile_id, [])
    summary = (summary_by_profile or {}).get(profile_id) or {}
    return {
        "inversion_layers_v3": layers,
        "n_inversion_layers_v3": int(summary.get("n_inversion_layers", len(layers))),
        "inversion_pattern_v3": str(summary.get("pattern", "NONE" if not layers else "MULTI")),
        "has_G_v3": bool(summary.get("has_G", any(ly.get("position_type") == "G" for ly in layers))),
        "has_E_v3": bool(summary.get("has_E", any(ly.get("position_type") == "E" for ly in layers))),
        "has_HE_v3": bool(summary.get("has_HE", any(ly.get("position_type") == "HE" for ly in layers))),
        "strongest_delta_t_c_v3": summary.get("strongest_delta_t_c"),
    }


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "да"}
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(value)


def _load_v3_maps_from_csv(
    layers_csv: Path | None,
    summary_csv: Path | None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    layers_by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    summary_by: dict[str, dict[str, Any]] = {}
    if layers_csv is not None and layers_csv.exists():
        df = pd.read_csv(layers_csv)
        for row in df.itertuples(index=False):
            pid = str(row.profile_id)
            layers_by[pid].append({
                "layer_index": int(getattr(row, "layer_index", len(layers_by[pid]))),
                "position_type": str(getattr(row, "position_type", "")),
                "base_height_m": _finite_metric(getattr(row, "base_height_m", None)),
                "top_height_m": _finite_metric(getattr(row, "top_height_m", None)),
                "base_height_agl_m": _finite_metric(getattr(row, "base_height_agl_m", None)),
                "top_height_agl_m": _finite_metric(getattr(row, "top_height_agl_m", None)),
                "base_pressure_hpa": _finite_metric(getattr(row, "base_pressure_hpa", None)),
                "top_pressure_hpa": _finite_metric(getattr(row, "top_pressure_hpa", None)),
                "base_temperature_c": _finite_metric(getattr(row, "base_temperature_c", None)),
                "top_temperature_c": _finite_metric(getattr(row, "top_temperature_c", None)),
                "depth_m": _finite_metric(getattr(row, "depth_m", None)),
                "delta_t_c": _finite_metric(getattr(row, "delta_t_c", None)),
                "mean_gradient_c_100m": _finite_metric(getattr(row, "mean_gradient_c_100m", None)),
                "embedded_gap_count": int(getattr(row, "embedded_gap_count", 0) or 0),
                "method": str(getattr(row, "method", "gap_v3")),
            })
    if summary_csv is not None and summary_csv.exists():
        sdf = pd.read_csv(summary_csv)
        for row in sdf.itertuples(index=False):
            summary_by[str(row.profile_id)] = {
                "n_inversion_layers": int(getattr(row, "n_inversion_layers", 0) or 0),
                "has_G": _as_bool(getattr(row, "has_G", False)),
                "has_E": _as_bool(getattr(row, "has_E", False)),
                "has_HE": _as_bool(getattr(row, "has_HE", False)),
                "pattern": str(getattr(row, "pattern", "NONE") or "NONE"),
                "strongest_delta_t_c": _finite_metric(getattr(row, "strongest_delta_t_c", None)),
            }
    return dict(layers_by), summary_by


def resolve_xlsx(path: Path | None, search_dir: Path) -> Path | None:
    """Явный --xlsx или последний aldan_profile_climate_*.xlsx / profile_climate.xlsx."""
    if path is not None:
        return path if path.exists() else None
    if not search_dir.exists():
        return None
    actual = search_dir / "aldan_actual.xlsx"
    if actual.exists():
        return actual
    stamped = sorted(
        (
            p
            for p in search_dir.glob("*_profile_climate_*.xlsx")
            if "heights_fixed" not in p.name
        ),
        key=lambda p: p.stat().st_mtime,
    )
    if stamped:
        return stamped[-1]
    plain = search_dir / "profile_climate.xlsx"
    return plain if plain.exists() else None


def load_long_and_metrics(
    long_csv: Path,
    metrics_csv: Path,
    *,
    xlsx: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """CSV если есть, иначе листы profiles_long / profile_metrics из Excel."""
    if long_csv.exists() and metrics_csv.exists():
        return (
            pd.read_csv(long_csv, low_memory=False),
            pd.read_csv(metrics_csv, low_memory=False),
            f"csv:{long_csv}|{metrics_csv}",
        )

    if long_csv == DEFAULT_LONG_CSV and metrics_csv == DEFAULT_METRICS_CSV:
        legacy_long = LEGACY_DIR / "profiles_long.csv"
        legacy_metrics = LEGACY_DIR / "profile_metrics.csv"
        if legacy_long.exists() and legacy_metrics.exists():
            return (
                pd.read_csv(legacy_long, low_memory=False),
                pd.read_csv(legacy_metrics, low_memory=False),
                f"legacy_csv:{legacy_long}|{legacy_metrics}",
            )

    xlsx_path = resolve_xlsx(xlsx, long_csv.parent if long_csv.parent.exists() else DEFAULT_DIR)
    if xlsx_path is None and long_csv == DEFAULT_LONG_CSV:
        xlsx_path = resolve_xlsx(None, LEGACY_DIR)
    if xlsx_path is None:
        raise FileNotFoundError(
            "Нет CSV (profiles_long / profile_metrics) и нет Excel. "
            f"Искали CSV: {long_csv}, {metrics_csv}. "
            "Положите xlsx в ту же папку или укажите --xlsx PATH."
        )

    sheet_names = set(pd.ExcelFile(xlsx_path).sheet_names)
    long_sheet = "profiles_working" if "profiles_working" in sheet_names else "profiles_long"
    long_df = pd.read_excel(xlsx_path, sheet_name=long_sheet)
    metrics_df = pd.read_excel(xlsx_path, sheet_name="profile_metrics")
    return long_df, metrics_df, f"xlsx:{xlsx_path}"


def _series_to_levels(group: pd.DataFrame) -> list[dict[str, Any]]:
    levels: list[dict[str, Any]] = []
    for row in group.itertuples(index=False):
        height = None if pd.isna(getattr(row, "height_m", None)) else float(row.height_m)
        levels.append({
            "pressure_hpa": float(row.pressure_hpa),
            "temperature_c": float(row.temperature_c),
            "height_m": height,
            "height_obs_m": None if pd.isna(getattr(row, "height_obs_m", None)) else float(row.height_obs_m),
            "height_interp_m": None if pd.isna(getattr(row, "height_interp_m", None)) else float(row.height_interp_m),
            "height_baro_m": None if pd.isna(getattr(row, "height_baro_m", None)) else float(row.height_baro_m),
            "height_source": getattr(row, "height_source", None),
            "geopotential_m2s2": (
                None if pd.isna(getattr(row, "geopotential_m2s2", None))
                else float(row.geopotential_m2s2)
            ),
            "geopotential_height_m": (
                None if pd.isna(getattr(row, "geopotential_height_m", None))
                else float(row.geopotential_height_m)
            ),
        })
    return levels


def _obs_arrays(
    levels: list[dict[str, Any]],
) -> tuple[list[float | None], list[float], list[float], list[float | None], list[float | None]]:
    heights = [
        None if lv.get("height_m") is None else round(float(lv["height_m"]), 1)
        for lv in levels
    ]
    heights_interp = [
        None if lv.get("height_interp_m") is None else round(float(lv["height_interp_m"]), 1)
        for lv in levels
    ]
    heights_baro = [
        None if lv.get("height_baro_m") is None else round(float(lv["height_baro_m"]), 1)
        for lv in levels
    ]
    pressures = [round(float(lv["pressure_hpa"]), 2) for lv in levels]
    temps = [round(float(lv["temperature_c"]), 3) for lv in levels]
    return heights, pressures, temps, heights_interp, heights_baro


def _station_elevation(
    long_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    station_id: str,
) -> float | None:
    """Высота станции: из метрик, иначе из long-таблицы, иначе из справочника."""
    for frame in (metrics_df, long_df):
        if frame is None or "station_elevation_m" not in frame.columns:
            continue
        values = frame["station_elevation_m"].dropna()
        if len(values):
            elevation = _finite_metric(values.iloc[0])
            if elevation is not None:
                return round(elevation, 1)
    fallback = STATION_ELEVATION_M.get(str(station_id).zfill(5)[-5:])
    return None if fallback is None else round(float(fallback), 1)


def _height_source_counts(levels: list[dict[str, Any]]) -> dict[str, int]:
    """Сколько уровней получило высоту каждым методом (level/phi/interp/baro/…)."""
    counts: dict[str, int] = {}
    for level in levels:
        source = level.get("height_source")
        if source is None or (isinstance(source, float) and source != source):
            source = "none"
        key = str(source)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _make_observation(
    *,
    profile_id: str,
    datetime_utc: str,
    cycle: Any,
    heights_m: list[Any],
    heights_interp_m: list[Any],
    heights_baro_m: list[Any],
    pressure_hpa: list[Any],
    temperature_c: list[Any],
    n_levels: int,
    t_surface_c: float | None,
    inversion_detected: bool,
    inv_fields: dict[str, Any],
    profile_status: str,
    p_surface_hpa: float | None = None,
    station_elevation_m: float | None = None,
    height_source_counts: dict[str, int] | None = None,
    missing_levels: bool = False,
    v3_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Единая структура наблюдения — общая для профилей с уровнями и без них."""
    observation: dict[str, Any] = {
        "profile_id": profile_id,
        "datetime_utc": datetime_utc,
        "cycle": str(cycle).zfill(2)[-2:],
        "heights_m": heights_m,
        "heights_interp_m": heights_interp_m,
        "heights_baro_m": heights_baro_m,
        "pressure_hpa": pressure_hpa,
        "temperature_c": temperature_c,
        "n_levels": n_levels,
        "t_surface_c": t_surface_c,
        "inversion_detected": inversion_detected,
        **inv_fields,
        **(v3_fields if v3_fields is not None else _empty_v3_fields()),
        "profile_status": profile_status,
        "p_surface_hpa": None if p_surface_hpa is None else round(float(p_surface_hpa), 2),
        "station_elevation_m": (
            None if station_elevation_m is None else round(float(station_elevation_m), 1)
        ),
        "height_source_counts": height_source_counts or {},
    }
    if missing_levels:
        observation["missing_levels"] = True
    return observation


def _day_mean_on_pressure(
    observations: list[dict[str, Any]],
    *,
    grid_points: int,
) -> dict[str, Any] | None:
    series_p: list[np.ndarray] = []
    series_t: list[np.ndarray] = []
    series_h: list[np.ndarray] = []
    for obs in observations:
        p = np.asarray(obs["pressure_hpa"], dtype=float)
        t = np.asarray(
            [np.nan if v is None else v for v in obs["temperature_c"]],
            dtype=float,
        )
        h = np.asarray(obs["heights_m"], dtype=float)
        valid = ~np.isnan(t) & ~np.isnan(p) & ~np.isnan(h)
        if valid.sum() < 2:
            continue
        series_p.append(p[valid])
        series_t.append(t[valid])
        series_h.append(h[valid])
    if not series_p:
        return None

    p_lo = min(float(p.min()) for p in series_p)
    p_hi = max(float(p.max()) for p in series_p)
    if p_hi <= p_lo:
        return None
    grid = np.linspace(p_hi, p_lo, grid_points)
    stacked_t = np.vstack([interp_on_pressure_grid(p, t, grid) for p, t in zip(series_p, series_t)])
    stacked_h = np.vstack([interp_on_pressure_grid(p, h, grid) for p, h in zip(series_p, series_h)])
    mean_t = np.nanmean(stacked_t, axis=0)
    mean_h = np.nanmean(stacked_h, axis=0)
    return {
        "pressure_hpa": [round(float(x), 2) for x in grid],
        "heights_m": [None if np.isnan(x) else round(float(x), 1) for x in mean_h],
        "temperature_c": [None if np.isnan(x) else round(float(x), 3) for x in mean_t],
    }


REJECTED_CLEAN_STATUSES = frozenset({
    "no_temp", "bad_pressure", "duplicate_levels", "no_surface_level",
})


def _observation_from_group(
    profile_id: str,
    group: pd.DataFrame,
    metric: Any,
    *,
    level_mode: str,
    pressure_top_hpa: float,
    max_surface_pressure_hpa: float,
    min_levels: int,
    v3_fields: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """Собирает одно наблюдение из строк профиля. None — пропуск."""
    status = getattr(metric, "profile_status", None) if metric is not None else None
    if level_mode == "clean" and status in REJECTED_CLEAN_STATUSES:
        return None

    levels = _series_to_levels(group)
    if level_mode == "clean":
        levels = clean_observation_levels(
            levels,
            pressure_top_hpa=pressure_top_hpa,
            max_surface_pressure_hpa=max_surface_pressure_hpa,
        )
    if len(levels) < min_levels:
        return None

    dt = str(group["datetime_utc"].iloc[0])
    try:
        day = _day_key(dt)
    except ValueError:
        return None

    heights, pressures, temps, heights_interp, heights_baro = _obs_arrays(levels)
    t_surface = None
    inversion = False
    status_text = ""
    inv_fields = _metric_inversion_fields(None)
    p_surface = None
    station_z = None
    if metric is not None:
        t_s = getattr(metric, "t_surface_c", None)
        if t_s is not None and not (isinstance(t_s, float) and np.isnan(t_s)):
            t_surface = round(float(t_s), 3)
        inversion = _metric_flag(metric, "inversion_detected")
        status_text = str(getattr(metric, "profile_status", "") or "")
        inv_fields = _metric_inversion_fields(metric)
        p_surface = _finite_metric(getattr(metric, "p_surface_hpa", None))
        station_z = _finite_metric(getattr(metric, "station_elevation_m", None))
    if t_surface is None and temps:
        t_surface = temps[0]
    if p_surface is None and pressures:
        p_surface = max(pressures)

    return day, _make_observation(
        profile_id=profile_id,
        datetime_utc=dt,
        cycle=group["cycle"].iloc[0],
        heights_m=heights,
        heights_interp_m=heights_interp,
        heights_baro_m=heights_baro,
        pressure_hpa=pressures,
        temperature_c=temps,
        n_levels=len(levels),
        t_surface_c=t_surface,
        inversion_detected=inversion,
        inv_fields=inv_fields,
        profile_status=status_text,
        p_surface_hpa=p_surface,
        station_elevation_m=station_z,
        height_source_counts=_height_source_counts(levels),
        v3_fields=v3_fields,
    )


def _append_metrics_only_profiles(
    by_day: dict[str, list[dict[str, Any]]],
    metrics_df: pd.DataFrame,
    profiles_with_levels: set[str],
    *,
    layers_by_profile: dict[str, list[dict[str, Any]]] | None = None,
    summary_by_profile: dict[str, dict[str, Any]] | None = None,
) -> None:
    """В raw-режиме добавляет профили без уровней (только метрики)."""
    for metric in metrics_df.itertuples(index=False):
        profile_id = str(metric.profile_id)
        if profile_id in profiles_with_levels:
            continue
        dt = str(getattr(metric, "datetime_utc", ""))
        try:
            day = _day_key(dt)
        except ValueError:
            continue
        t_surface = _finite_metric(getattr(metric, "t_surface_c", None))
        by_day[day].append(_make_observation(
            profile_id=profile_id,
            datetime_utc=dt,
            cycle=getattr(metric, "cycle", ""),
            heights_m=[],
            heights_interp_m=[],
            heights_baro_m=[],
            pressure_hpa=[],
            temperature_c=[],
            n_levels=0,
            t_surface_c=None if t_surface is None else round(t_surface, 3),
            inversion_detected=_metric_flag(metric, "inversion_detected"),
            inv_fields=_metric_inversion_fields(metric),
            profile_status=str(getattr(metric, "profile_status", "") or ""),
            p_surface_hpa=_finite_metric(getattr(metric, "p_surface_hpa", None)),
            station_elevation_m=_finite_metric(getattr(metric, "station_elevation_m", None)),
            missing_levels=True,
            v3_fields=_v3_fields_from_maps(
                profile_id, layers_by_profile, summary_by_profile,
            ),
        ))


def _build_months_payload(
    by_day: dict[str, list[dict[str, Any]]],
    *,
    grid_points: int,
) -> tuple[dict[str, dict[str, Any]], int]:
    """Группирует дни по месяцам и считает суточные средние."""
    months: dict[str, dict[str, Any]] = {}
    n_observations = 0
    for day, observations in sorted(by_day.items()):
        observations.sort(key=lambda o: (o["datetime_utc"], o["cycle"], o["profile_id"]))
        month_key = day[:7]
        day_mean = _day_mean_on_pressure(observations, grid_points=grid_points)
        t_surfaces = [o["t_surface_c"] for o in observations if o.get("t_surface_c") is not None]
        months.setdefault(month_key, {"days": []})
        months[month_key]["days"].append({
            "date": day,
            "n_profiles": len(observations),
            "n_good": sum(1 for o in observations if o.get("profile_status") == "good"),
            "n_missing_levels": sum(1 for o in observations if o.get("missing_levels")),
            "inversion_detected": any(o.get("inversion_detected") for o in observations),
            "t_surface_c": (
                round(float(np.mean(t_surfaces)), 3) if t_surfaces else None
            ),
            "observations": observations,
            "day_mean": day_mean,
        })
        n_observations += len(observations)

    for month_key in months:
        months[month_key]["days"].sort(key=lambda d: d["date"])
    return months, n_observations


def build_daily_profiles(
    long_csv: Path,
    metrics_csv: Path,
    *,
    xlsx: Path | None = None,
    pressure_top_hpa: float = PRESSURE_TOP_HPA,
    max_surface_pressure_hpa: float = MAX_SURFACE_PRESSURE_HPA,
    plot_min_levels: int | None = None,
    grid_points: int = GRID_POINTS,
    level_mode: str = "raw",
    layers_v3_csv: Path | None = None,
    summary_v3_csv: Path | None = None,
    compute_v3: bool = False,
    v3_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if level_mode not in LEVEL_MODES:
        raise ValueError(f"level_mode должен быть одним из {LEVEL_MODES}: {level_mode}")
    min_levels = (1 if level_mode == "raw" else PLOT_MIN_LEVELS) if plot_min_levels is None else plot_min_levels

    # 1) Читаем таблицы и заполняем высоты.
    long_df, metrics_df, source = load_long_and_metrics(long_csv, metrics_csv, xlsx=xlsx)
    print(f"Источник таблиц: {source}")
    print(
        f"Станция Алдан: высота {STATION_ELEVATION_M.get('31004')} м н.у.м.; "
        f"типичное P у поверхности ~ {ALDAN_TYPICAL_SURFACE_HPA} гПа (не константа)"
    )

    long_df = long_df.dropna(subset=["temperature_c", "pressure_hpa"])
    long_df = long_df[
        (long_df["pressure_hpa"] <= max_surface_pressure_hpa)
        & (long_df["pressure_hpa"] >= pressure_top_hpa)
    ]
    long_df = fill_long_dataframe_heights(long_df, metrics_df)

    metrics_map = {
        str(row.profile_id): row
        for row in metrics_df.itertuples(index=False)
    }

    layers_by_profile: dict[str, list[dict[str, Any]]] = {}
    summary_by_profile: dict[str, dict[str, Any]] = {}
    if compute_v3:
        from gdex_bufr.profile_climate.inversion_layers import (
            detect_inversion_layers_gap_v3,
            layers_to_dashboard_payload,
            summarize_inversion_layers,
        )

        params = {
            "max_embedded_gap_m": 100.0,
            "min_strength_c": 0.3,
            "min_depth_m": None,
            "he_threshold_m": 250.0,
            "max_gap_drop_c": None,
            **(v3_params or {}),
        }
        for profile_id, group in long_df.groupby("profile_id", sort=False):
            pid = str(profile_id)
            z = pd.to_numeric(group["height_m"], errors="coerce").to_numpy(dtype=float)
            t = pd.to_numeric(group["temperature_c"], errors="coerce").to_numpy(dtype=float)
            p = pd.to_numeric(group["pressure_hpa"], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(z) & np.isfinite(t)
            z, t, p = z[mask], t[mask], p[mask]
            if z.size < 2:
                summary_by_profile[pid] = summarize_inversion_layers(pid, [], z0=0.0)
                layers_by_profile[pid] = []
                continue
            layers = detect_inversion_layers_gap_v3(
                z,
                t,
                p,
                max_embedded_gap_m=float(params["max_embedded_gap_m"]),
                min_strength_c=float(params["min_strength_c"]),
                min_depth_m=params.get("min_depth_m"),
                he_threshold_m=float(params["he_threshold_m"]),
                max_gap_drop_c=params.get("max_gap_drop_c"),
            )
            order = np.argsort(z, kind="mergesort")
            z0 = float(z[order][0])
            layers_by_profile[pid] = layers_to_dashboard_payload(layers, z0=z0)
            summary_by_profile[pid] = summarize_inversion_layers(pid, layers, z0=z0)
    elif layers_v3_csv is not None or summary_v3_csv is not None:
        layers_by_profile, summary_by_profile = _load_v3_maps_from_csv(
            layers_v3_csv, summary_v3_csv,
        )

    # 2) Собираем наблюдения по дням.
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    profiles_with_levels: set[str] = set()
    for profile_id, group in long_df.groupby("profile_id"):
        profile_id = str(profile_id)
        profiles_with_levels.add(profile_id)
        built = _observation_from_group(
            profile_id,
            group,
            metrics_map.get(profile_id),
            level_mode=level_mode,
            pressure_top_hpa=pressure_top_hpa,
            max_surface_pressure_hpa=max_surface_pressure_hpa,
            min_levels=min_levels,
            v3_fields=_v3_fields_from_maps(
                profile_id, layers_by_profile, summary_by_profile,
            ),
        )
        if built is None:
            continue
        day, obs = built
        by_day[day].append(obs)

    # 3) В raw добавляем профили без уровней, чтобы ничего не терялось.
    if level_mode == "raw":
        _append_metrics_only_profiles(
            by_day,
            metrics_df,
            profiles_with_levels,
            layers_by_profile=layers_by_profile,
            summary_by_profile=summary_by_profile,
        )

    # 4) Собираем месяцы и итоговый JSON.
    months, n_observations = _build_months_payload(by_day, grid_points=grid_points)
    station_name = str(long_df["station_name"].iloc[0]) if len(long_df) else ""
    station_id = str(long_df["station_id"].iloc[0]) if len(long_df) else ""
    station_z = _station_elevation(long_df, metrics_df, station_id)

    return {
        "schema": SCHEMA,
        "features": list(FEATURES),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "station_id": station_id,
        "station_name": station_name,
        "station_elevation_m": station_z,
        "source_tables": source,
        "pressure_top_hpa": pressure_top_hpa,
        "max_surface_pressure_hpa": max_surface_pressure_hpa,
        "level_mode": level_mode,
        "plot_min_levels": min_levels,
        "grid_points": grid_points,
        "n_days": sum(len(m["days"]) for m in months.values()),
        "n_observations": n_observations,
        "n_levels": sum(
            obs["n_levels"]
            for month in months.values()
            for day in month["days"]
            for obs in day["observations"]
        ),
        "months": months,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Собрать daily_profiles.json (observations_v1)")
    parser.add_argument("--long", default=str(DEFAULT_LONG_CSV))
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS_CSV))
    parser.add_argument(
        "--xlsx",
        help="Excel с листами profiles_long и profile_metrics "
             "(если CSV нет — берётся автоматически из папки --long)",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--level-mode",
        choices=LEVEL_MODES,
        default="raw",
        help="raw: сохранить все уровни без QC; clean: прежняя предварительная очистка",
    )
    parser.add_argument(
        "--min-levels",
        type=int,
        help="Минимум уровней (по умолчанию: raw=1, clean=3)",
    )
    parser.add_argument(
        "--compute-v3",
        action="store_true",
        help="Посчитать gap-v3 слои из profiles_long и вложить в JSON",
    )
    parser.add_argument("--layers-v3", help="Готовый inversion_layers_v3.csv")
    parser.add_argument("--summary-v3", help="Готовый profile_inversion_summary_v3.csv")
    args = parser.parse_args()

    payload = build_daily_profiles(
        Path(args.long),
        Path(args.metrics),
        xlsx=Path(args.xlsx) if args.xlsx else None,
        level_mode=args.level_mode,
        plot_min_levels=args.min_levels,
        compute_v3=bool(args.compute_v3),
        layers_v3_csv=Path(args.layers_v3) if args.layers_v3 else None,
        summary_v3_csv=Path(args.summary_v3) if args.summary_v3 else None,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output": str(out),
        "schema": payload["schema"],
        "features": payload.get("features"),
        "station_elevation_m": payload.get("station_elevation_m"),
        "source_tables": payload.get("source_tables"),
        "n_days": payload["n_days"],
        "n_observations": payload["n_observations"],
        "n_levels": payload["n_levels"],
        "n_months": len(payload["months"]),
        "station": payload["station_name"],
        "level_mode": payload["level_mode"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
