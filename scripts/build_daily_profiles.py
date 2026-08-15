"""Сборка профилей наблюдений (зондов) для интерактивного дашборда."""
from __future__ import annotations

import json
import argparse
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
    "inversion_from_top",     # inversion_from_top_tops / count (поиск сверху)
    "height_variants",        # heights_interp_m / heights_baro_m на каждом наблюдении
    "height_source_counts",   # состав источников высоты внутри зонда
    "surface_context",        # p_surface_hpa / station_elevation_m
    "inversion_v3",           # inversion_layers_v3 / pattern / n_layers (gap-merge)
)
#Станция каталога - это папка с данными о станции, которая содержит файлы с данными о наблюдениях.
DEFAULT_DIR = catalog_station_dir()
LEGACY_DIR = Path("gdex_outputs") / "актуальное"
DEFAULT_LONG_CSV = DEFAULT_DIR / "profiles_long.csv"
DEFAULT_METRICS_CSV = DEFAULT_DIR / "profile_metrics.csv"
DEFAULT_OUTPUT = DEFAULT_DIR / "daily_profiles.json"


def _day_key(dt: str) -> str:
    parsed = datetime.fromisoformat(str(dt).replace("Z", "+00:00"))
    return parsed.date().isoformat()

#Функция _finite_metric проверяет, является ли значение числом и не является NaN.
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

#Функция _metric_flag проверяет, является ли значение логическим и не является NaN. Если является, то возвращает True, иначе False.
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

#Функция _parse_from_top_tops преобразует строку в список словарей. Если строка не является JSON, то возвращает пустой список. Если строка является JSON, то преобразует её в список словарей.
def _parse_from_top_tops(raw: Any) -> list[dict[str, Any]]:
    """JSON-строка / list из метрик → список вершин. Если строка не является JSON, то возвращает пустой список. Если строка является JSON, то преобразует её в список словарей."""
    if raw is None:
        return []
    try:
        if isinstance(raw, float) and pd.isna(raw):
            return []
    except (TypeError, ValueError):
        pass
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return []
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
    return []

#Функция _from_top_fields_from_levels пересчитывает поля from_top по уровням наблюдения. Если в метриках поля ещё нет, то пересчитывает их. Вызывается в функции _metric_inversion_fields.
def _from_top_fields_from_levels(levels: list[dict[str, Any]]) -> dict[str, Any]:
    """Пересчёт from_top по уровням наблюдения (если в метриках поля ещё нет)."""
    from gdex_bufr.profile_climate.inversion import (
        detect_inversions_from_top,
        inversions_from_top_as_metrics,
    )

    prepared: list[dict[str, Any]] = []
    for lv in levels:
        p = lv.get("pressure_hpa")
        t = lv.get("temperature_c")
        if p is None or t is None:
            continue
        prepared.append({
            "pressure_hpa": float(p),
            "temperature_c": float(t),
            "height_m": lv.get("height_m"),
        })
    prepared.sort(key=lambda row: row["pressure_hpa"], reverse=True)
    meta = inversions_from_top_as_metrics(detect_inversions_from_top(prepared))
    return {
        "inversion_from_top_count": int(meta["inversion_from_top_count"]),
        "inversion_from_top_tops": meta["inversion_from_top_tops"],
    }

#Функция _metric_inversion_fields извлекает поля инверсии из метрик, включая семантику v2 (quality / candidate). Если в метриках поля ещё нет, то пересчитывает их. Вызывается в функции _build_months_payload.
def _metric_inversion_fields(
    metric: Any,
    *,
    levels: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Поля инверсии из метрик, включая семантику v2 (quality / candidate)."""
    h = _finite_metric(getattr(metric, "inversion_top_height_m", None)) if metric is not None else None
    p = _finite_metric(getattr(metric, "inversion_top_pressure_hpa", None)) if metric is not None else None
    t = _finite_metric(getattr(metric, "inversion_top_temp_c", None)) if metric is not None else None
    d = _finite_metric(getattr(metric, "inversion_delta_t_c", None)) if metric is not None else None
    drop = (
        _finite_metric(getattr(metric, "inversion_confirm_drop_c", None))
        if metric is not None
        else None
    )
    quality = getattr(metric, "inversion_quality", None) if metric is not None else None
    try:
        if quality is None or pd.isna(quality):
            quality = ""
    except (TypeError, ValueError):
        pass

    tops: list[dict[str, Any]] = []
    count = 0
    if metric is not None:
        tops = _parse_from_top_tops(getattr(metric, "inversion_from_top_tops", None))
        count_raw = getattr(metric, "inversion_from_top_count", None)
        count_val = _finite_metric(count_raw)
        if count_val is not None:
            count = int(count_val)
        elif tops:
            count = sum(1 for x in tops if str(x.get("quality") or "") == "confirmed")
    if not tops and levels:
        computed = _from_top_fields_from_levels(levels)
        tops = computed["inversion_from_top_tops"]
        count = int(computed["inversion_from_top_count"])

    return {
        "inversion_top_height_m": None if h is None else round(h, 1),
        "inversion_top_pressure_hpa": None if p is None else round(p, 1),
        "inversion_top_temp_c": None if t is None else round(t, 2),
        "inversion_delta_t_c": None if d is None else round(d, 2),
        "inversion_confirm_drop_c": None if drop is None else round(drop, 2),
        "inversion_quality": str(quality or ""),
        "inversion_candidate": (
            _metric_flag(metric, "inversion_candidate") if metric is not None else False
        ),
        "inversion_from_top_count": count,
        "inversion_from_top_tops": tops,
    }

#Функция _empty_v3_fields возвращает пустые поля для v3.
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

#Функция _v3_fields_from_maps извлекает поля v3 из словарей. Если словари пусты, то возвращает пустые поля. Если словари не пусты, то извлекает поля из словарей. Вызывается в функции _make_observation.
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

#Функция _as_bool преобразует строку в логическое значение. Если строка не является логическим значением, то возвращает False.
def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "да"}
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(value)

#Функция _load_v3_maps_from_csv загружает словари v3 из CSV файлов. Если CSV файлы не существуют, то возвращает пустые словари. Если CSV файлы существуют, то загружает словари из CSV файлов. Вызывается в функции _build_months_payload.
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

#Функция resolve_xlsx находит последний Excel файл в каталоге. Если файл не существует, то возвращает None. Если файл существует, то возвращает путь к файлу. Вызывается в функции load_long_and_metrics.
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

#Функция load_long_and_metrics загружает long-таблицу и метрики из CSV или Excel файлов. Если CSV файлы не существуют, то загружает из Excel файлов. Если Excel файлы не существуют, то возвращает None. Если Excel файлы существуют, то загружает из Excel файлов. Вызывается в функции build_daily_profiles.
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

#Функция _series_to_levels преобразует строки в список словарей. Если строка не является числом, то возвращает None. Если строка является числом, то преобразует её в число. Вызывается в функции _make_observation.
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

#Функция _obs_arrays извлекает массивы высот, давления, температуры и высот интерполяции из списка словарей. Вызывается в функции _make_observation. Возвращает кортеж из пяти списков: высот, давления, температуры, высот интерполяции и высот барометрических.
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

#Функция _station_elevation извлекает высоту станции из метрик или long-таблицы. Если высота станции не найдена, то возвращает None. Вызывается в функции _make_observation. Возвращает высоту станции в метрах.
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

#Функция _height_source_counts подсчитывает количество уровней, получивших высоту каждым методом (level/phi/interp/baro/…). Вызывается в функции _make_observation. Возвращает словарь с количеством уровней для каждого метода. Например, {'level': 10, 'phi': 5, 'interp': 3, 'baro': 2}.
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

#Функция _make_observation собирает одно наблюдение из списка словарей. Вызывается в функции _observation_from_group. Возвращает словарь с наблюдением. Например, {'profile_id': '1234567890', 'datetime_utc': '2026-01-01 12:00:00', 'cycle': '01', 'heights_m': [100.0, 101.0, 102.0], 'heights_interp_m': [100.5, 101.5, 102.5], 'heights_baro_m': [100.2, 101.2, 102.2], 'pressure_hpa': [1000.0, 999.0, 998.0], 'temperature_c': [20.0, 20.1, 20.2], 'n_levels': 3, 't_surface_c': 20.0, 'inversion_detected': True, 'inv_fields': {'inversion_from_top_tops': [100.0, 101.0, 102.0], 'inversion_from_bottom_tops': [100.0, 101.0, 102.0], 'inversion_from_top_bottoms': [100.0, 101.0, 102.0], 'inversion_from_bottom_bottoms': [100.0, 101.0, 102.0]}, 'profile_status': 'good', 'p_surface_hpa': 1000.0, 'station_elevation_m': 100.0, 'height_source_counts': {'level': 10, 'phi': 5, 'interp': 3, 'baro': 2}, 'missing_levels': False, 'v3_fields': {'inversion_layers_v3': [100.0, 101.0, 102.0], 'pattern': 'NONE', 'n_layers': 3, 'strongest_delta_t_c_v3': 1.0}}.
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

#Функция _day_mean_on_pressure вычисляет среднее значение температуры и высоты на давлении. Вызывается в функции _build_months_payload. Возвращает словарь с средним значением температуры и высоты на давлении. Например, {'pressure_hpa': [1000.0, 999.0, 998.0], 'heights_m': [100.0, 101.0, 102.0], 'temperature_c': [20.0, 20.1, 20.2]}. Если данные невалидны, то возвращает None.
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

#Функция REJECTED_CLEAN_STATUSES определяет статусы профилей, которые не будут включены в средние значения. Вызывается в функции _observation_from_group. Возвращает множество статусов. Например, {'no_temp', 'bad_pressure', 'duplicate_levels', 'no_surface_level'}.
REJECTED_CLEAN_STATUSES = frozenset({
    "no_temp", "bad_pressure", "duplicate_levels", "no_surface_level",
})

#Функция _observation_from_group собирает одно наблюдение из строк профиля. None — пропуск. Вызывается в функции build_daily_profiles. Возвращает кортеж из дня и словаря с наблюдением. Например, ('2026-01-01', {'profile_id': '1234567890', 'datetime_utc': '2026-01-01 12:00:00', 'cycle': '01', 'heights_m': [100.0, 101.0, 102.0], 'heights_interp_m': [100.5, 101.5, 102.5], 'heights_baro_m': [100.2, 101.2, 102.2], 'pressure_hpa': [1000.0, 999.0, 998.0], 'temperature_c': [20.0, 20.1, 20.2], 'n_levels': 3, 't_surface_c': 20.0, 'inversion_detected': True, 'inv_fields': {'inversion_from_top_tops': [100.0, 101.0, 102.0], 'inversion_from_bottom_tops': [100.0, 101.0, 102.0], 'inversion_from_top_bottoms': [100.0, 101.0, 102.0], 'inversion_from_bottom_bottoms': [100.0, 101.0, 102.0]}, 'profile_status': 'good', 'p_surface_hpa': 1000.0, 'station_elevation_m': 100.0, 'height_source_counts': {'level': 10, 'phi': 5, 'interp': 3, 'baro': 2}, 'missing_levels': False, 'v3_fields': {'inversion_layers_v3': [100.0, 101.0, 102.0], 'pattern': 'NONE', 'n_layers': 3, 'strongest_delta_t_c_v3': 1.0}}).
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
        inv_fields = _metric_inversion_fields(metric, levels=levels)
        p_surface = _finite_metric(getattr(metric, "p_surface_hpa", None))
        station_z = _finite_metric(getattr(metric, "station_elevation_m", None))
    if t_surface is None and temps:
        t_surface = temps[0]
    if p_surface is None and pressures:
        p_surface = max(pressures)
    if not inv_fields.get("inversion_from_top_tops") and levels:
        inv_fields = {
            **inv_fields,
            **_from_top_fields_from_levels(levels),
        }

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

#Функция _append_metrics_only_profiles добавляет профили без уровней (только метрики) в словарь по дням. Вызывается в функции build_daily_profiles. Возвращает None. Если профиль уже есть в словаре, то пропускает его. Если профиль не есть в словаре, то добавляет его в словарь.
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

#Функция _build_months_payload группирует дни по месяцам и считает суточные средние. Вызывается в функции build_daily_profiles. Возвращает словарь с средними значениями температуры и высоты на давлении. Например, {'2026-01': {'days': [{'date': '2026-01-01', 'n_profiles': 10, 'n_good': 8, 'n_missing_levels': 2, 'inversion_detected': True, 't_surface_c': 20.0, 'observations': [{'profile_id': '1234567890', 'datetime_utc': '2026-01-01 12:00:00', 'cycle': '01', 'heights_m': [100.0, 101.0, 102.0], 'heights_interp_m': [100.5, 101.5, 102.5], 'heights_baro_m': [100.2, 101.2, 102.2], 'pressure_hpa': [1000.0, 999.0, 998.0], 'temperature_c': [20.0, 20.1, 20.2], 'n_levels': 3, 't_surface_c': 20.0, 'inversion_detected': True, 'inv_fields': {'inversion_from_top_tops': [100.0, 101.0, 102.0], 'inversion_from_bottom_tops': [100.0, 101.0, 102.0], 'inversion_from_top_bottoms': [100.0, 101.0, 102.0], 'inversion_from_bottom_bottoms': [100.0, 101.0, 102.0]}, 'profile_status': 'good', 'p_surface_hpa': 1000.0, 'station_elevation_m': 100.0, 'height_source_counts': {'level': 10, 'phi': 5, 'interp': 3, 'baro': 2}, 'missing_levels': False, 'v3_fields': {'inversion_layers_v3': [100.0, 101.0, 102.0], 'pattern': 'NONE', 'n_layers': 3, 'strongest_delta_t_c_v3': 1.0}}, ...], 'day_mean': {'pressure_hpa': [1000.0, 999.0, 998.0], 'heights_m': [100.0, 101.0, 102.0], 'temperature_c': [20.0, 20.1, 20.2]}]}}}.
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

#Функция build_daily_profiles собирает ежедневные профили из long-таблицы и метрик. Вызывается в функции main. Возвращает словарь с ежедневными профилями. Например, {'2026-01-01': {'n_profiles': 10, 'n_good': 8, 'n_missing_levels': 2, 'inversion_detected': True, 't_surface_c': 20.0, 'observations': [{'profile_id': '1234567890', 'datetime_utc': '2026-01-01 12:00:00', 'cycle': '01', 'heights_m': [100.0, 101.0, 102.0], 'heights_interp_m': [100.5, 101.5, 102.5], 'heights_baro_m': [100.2, 101.2, 102.2], 'pressure_hpa': [1000.0, 999.0, 998.0], 'temperature_c': [20.0, 20.1, 20.2], 'n_levels': 3, 't_surface_c': 20.0, 'inversion_detected': True, 'inv_fields': {'inversion_from_top_tops': [100.0, 101.0, 102.0], 'inversion_from_bottom_tops': [100.0, 101.0, 102.0], 'inversion_from_top_bottoms': [100.0, 101.0, 102.0], 'inversion_from_bottom_bottoms': [100.0, 101.0, 102.0]}, 'profile_status': 'good', 'p_surface_hpa': 1000.0, 'station_elevation_m': 100.0, 'height_source_counts': {'level': 10, 'phi': 5, 'interp': 3, 'baro': 2}, 'missing_levels': False, 'v3_fields': {'inversion_layers_v3': [100.0, 101.0, 102.0], 'pattern': 'NONE', 'n_layers': 3, 'strongest_delta_t_c_v3': 1.0}}, ...]}}.
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

    # 1) Читаем таблицы и заполняем высоты. Читаем long-таблицу и метрики из CSV или Excel файлов. Если CSV файлы не существуют, то загружает из Excel файлов. Если Excel файлы не существуют, то возвращает None. Если Excel файлы существуют, то загружает из Excel файлов. Вызывается в функции build_daily_profiles. Возвращает кортеж из long-таблицы, метрик и источника данных.
    long_df, metrics_df, source = load_long_and_metrics(long_csv, metrics_csv, xlsx=xlsx) #long_df - long-таблица, metrics_df - метрики, source - источник данных.
    print(f"Источник таблиц: {source}")
    print(
        f"Станция Алдан: высота {STATION_ELEVATION_M.get('31004')} м н.у.м.; " #STATION_ELEVATION_M - высота станции Алдан в метрах.
        f"типичное P у поверхности ~ {ALDAN_TYPICAL_SURFACE_HPA} гПа (не константа)" #ALDAN_TYPICAL_SURFACE_HPA - типичное давление у поверхности в гПа.
    )

    long_df = long_df.dropna(subset=["temperature_c", "pressure_hpa"]) #long_df - long-таблица, temperature_c - температура, pressure_hpa - давление.
    long_df = long_df[
        (long_df["pressure_hpa"] <= max_surface_pressure_hpa) #max_surface_pressure_hpa - максимальное давление у поверхности в гПа.
        & (long_df["pressure_hpa"] >= pressure_top_hpa) #pressure_top_hpa - минимальное давление у поверхности в гПа.
    ]
    long_df = fill_long_dataframe_heights(long_df, metrics_df) #long_df - long-таблица, metrics_df - метрики.

    metrics_map = { #metrics_map - словарь с метриками.
        str(row.profile_id): row #row - строка метрик.
        for row in metrics_df.itertuples(index=False) #metrics_df - метрики.
    }

    layers_by_profile: dict[str, list[dict[str, Any]]] = {} #layers_by_profile - словарь с уровнями.
    summary_by_profile: dict[str, dict[str, Any]] = {} #summary_by_profile - словарь с суммарией уровней.
    if compute_v3:
        from gdex_bufr.profile_climate.inversion_layers import ( #gdex_bufr.profile_climate.inversion_layers - модуль с функциями для работы с уровнями инверсии.
            detect_inversion_layers_gap_v3, #detect_inversion_layers_gap_v3 - функция для детектирования уровней инверсии.
            layers_to_dashboard_payload, #layers_to_dashboard_payload - функция для преобразования уровней инверсии в формат для dashboard.
            summarize_inversion_layers, #summarize_inversion_layers - функция для суммаризации уровней инверсии.
        )

        from gdex_bufr.profile_climate.config import load_profile_climate_config #gdex_bufr.profile_climate.config - модуль с функциями для работы с конфигурацией.

        cfg_params = load_profile_climate_config( #cfg_params - конфигурация.
            ROOT / "profile_climate_config.yaml" #ROOT / "profile_climate_config.yaml" - путь к конфигурации.
        ).v3_detect_kwargs() #v3_detect_kwargs() - функция для получения параметров для детектирования уровней инверсии.
        params = {**cfg_params, **(v3_params or {})} #params - параметры для детектирования уровней инверсии.
        for profile_id, group in long_df.groupby("profile_id", sort=False): #profile_id - id профиля, group - группа профилей.
            pid = str(profile_id)
            z = pd.to_numeric(group["height_m"], errors="coerce").to_numpy(dtype=float) #z - высоты.
            t = pd.to_numeric(group["temperature_c"], errors="coerce").to_numpy(dtype=float) #t - температуры.
            p = pd.to_numeric(group["pressure_hpa"], errors="coerce").to_numpy(dtype=float) #p - давления.
            mask = np.isfinite(z) & np.isfinite(t) #mask - маска для высот и температур.
            z, t, p = z[mask], t[mask], p[mask] #z, t, p - высоты, температуры и давления.
            if z.size < 2:
                summary_by_profile[pid] = summarize_inversion_layers(pid, [], z0=0.0) #summary_by_profile[pid] - суммаризация уровней инверсии.
                layers_by_profile[pid] = [] #layers_by_profile[pid] - уровни инверсии. Пустой список.
                continue
            layers = detect_inversion_layers_gap_v3(z, t, p, max_embedded_gap_m=float(params["max_embedded_gap_m"]), min_strength_c=float(params["min_strength_c"]), min_depth_m=params.get("min_depth_m"), he_threshold_m=float(params["he_threshold_m"]), max_gap_drop_c=params.get("max_gap_drop_c"), surface_tolerance_m=float(params.get("surface_tolerance_m", 30.0)))
            order = np.argsort(z, kind="mergesort") #order - порядок высот.
            z0 = float(z[order][0]) #z0 - высота нуля.
            layers_by_profile[pid] = layers_to_dashboard_payload(layers, z0=z0) #layers_by_profile[pid] - уровни инверсии в формате для dashboard.
            summary_by_profile[pid] = summarize_inversion_layers(pid, layers, z0=z0) #summary_by_profile[pid] - суммаризация уровней инверсии.
    elif layers_v3_csv is not None or summary_v3_csv is not None:
        layers_by_profile, summary_by_profile = _load_v3_maps_from_csv(layers_v3_csv, summary_v3_csv) #layers_by_profile, summary_by_profile - словари с уровнями и суммарией уровней.

    # 2) Собираем наблюдения по дням. Собираем наблюдения по дням из long-таблицы. Вызывается в функции build_daily_profiles. Возвращает словарь с наблюдениями по дням. Например, {'2026-01-01': [{'profile_id': '1234567890', 'datetime_utc': '2026-01-01 12:00:00', 'cycle': '01', 'heights_m': [100.0, 101.0, 102.0], 'heights_interp_m': [100.5, 101.5, 102.5], 'heights_baro_m': [100.2, 101.2, 102.2], 'pressure_hpa': [1000.0, 999.0, 998.0], 'temperature_c': [20.0, 20.1, 20.2], 'n_levels': 3, 't_surface_c': 20.0, 'inversion_detected': True, 'inv_fields': {'inversion_from_top_tops': [100.0, 101.0, 102.0], 'inversion_from_bottom_tops': [100.0, 101.0, 102.0], 'inversion_from_top_bottoms': [100.0, 101.0, 102.0], 'inversion_from_bottom_bottoms': [100.0, 101.0, 102.0]}, 'profile_status': 'good', 'p_surface_hpa': 1000.0, 'station_elevation_m': 100.0, 'height_source_counts': {'level': 10, 'phi': 5, 'interp': 3, 'baro': 2}, 'missing_levels': False, 'v3_fields': {'inversion_layers_v3': [100.0, 101.0, 102.0], 'pattern': 'NONE', 'n_layers': 3, 'strongest_delta_t_c_v3': 1.0}}, ...]}}.
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list) #by_day - словарь с наблюдениями по дням.
    profiles_with_levels: set[str] = set() #profiles_with_levels - множество с id профилей.
    for profile_id, group in long_df.groupby("profile_id"): #profile_id - id профиля, group - группа профилей.
        profile_id = str(profile_id) #profile_id - id профиля.
        profiles_with_levels.add(profile_id) #profiles_with_levels.add(profile_id) - добавляет id профиля в множество.
        built = _observation_from_group( #built - наблюдение. Вызывается в функции build_daily_profiles. Возвращает кортеж из дня и словаря с наблюдением. Например, ('2026-01-01', {'profile_id': '1234567890', 'datetime_utc': '2026-01-01 12:00:00', 'cycle': '01', 'heights_m': [100.0, 101.0, 102.0], 'heights_interp_m': [100.5, 101.5, 102.5], 'heights_baro_m': [100.2, 101.2, 102.2], 'pressure_hpa': [1000.0, 999.0, 998.0], 'temperature_c': [20.0, 20.1, 20.2], 'n_levels': 3, 't_surface_c': 20.0, 'inversion_detected': True, 'inv_fields': {'inversion_from_top_tops': [100.0, 101.0, 102.0], 'inversion_from_bottom_tops': [100.0, 101.0, 102.0], 'inversion_from_top_bottoms': [100.0, 101.0, 102.0], 'inversion_from_bottom_bottoms': [100.0, 101.0, 102.0]}, 'profile_status': 'good', 'p_surface_hpa': 1000.0, 'station_elevation_m': 100.0, 'height_source_counts': {'level': 10, 'phi': 5, 'interp': 3, 'baro': 2}, 'missing_levels': False, 'v3_fields': {'inversion_layers_v3': [100.0, 101.0, 102.0], 'pattern': 'NONE', 'n_layers': 3, 'strongest_delta_t_c_v3': 1.0}}).
            profile_id,
            group,
            metrics_map.get(profile_id), #metrics_map.get(profile_id) - метрики профиля.
            level_mode=level_mode, #level_mode - режим уровней.
            pressure_top_hpa=pressure_top_hpa, #pressure_top_hpa - минимальное давление у поверхности в гПа.
            max_surface_pressure_hpa=max_surface_pressure_hpa, #max_surface_pressure_hpa - максимальное давление у поверхности в гПа.
            min_levels=min_levels, #min_levels - минимальное количество уровней.
            v3_fields=_v3_fields_from_maps(profile_id, layers_by_profile, summary_by_profile), #v3_fields - поля v3.
        )
        if built is None:
            continue
        day, obs = built
        by_day[day].append(obs)

    # 3) В raw добавляем профили без уровней, чтобы ничего не терялось. В raw добавляем профили без уровней, чтобы ничего не терялось. Вызывается в функции build_daily_profiles. Возвращает None. Если режим уровней raw, то добавляет профили без уровней в словарь по дням.
    if level_mode == "raw":
        _append_metrics_only_profiles(
            by_day,
            metrics_df,
            profiles_with_levels,
            layers_by_profile=layers_by_profile,
            summary_by_profile=summary_by_profile,
        )

    # 4) Собираем месяцы и итоговый JSON. Собираем месяцы и итоговый JSON. Вызывается в функции build_daily_profiles. Возвращает кортеж из словаря с месяцами и количества наблюдений. Например, ({'2026-01': {'days': [{'date': '2026-01-01', 'n_profiles': 10, 'n_good': 8, 'n_missing_levels': 2, 'inversion_detected': True, 't_surface_c': 20.0, 'observations': [{'profile_id': '1234567890', 'datetime_utc': '2026-01-01 12:00:00', 'cycle': '01', 'heights_m': [100.0, 101.0, 102.0], 'heights_interp_m': [100.5, 101.5, 102.5], 'heights_baro_m': [100.2, 101.2, 102.2], 'pressure_hpa': [1000.0, 999.0, 998.0], 'temperature_c': [20.0, 20.1, 20.2], 'n_levels': 3, 't_surface_c': 20.0, 'inversion_detected': True, 'inv_fields': {'inversion_from_top_tops': [100.0, 101.0, 102.0], 'inversion_from_bottom_tops': [100.0, 101.0, 102.0], 'inversion_from_top_bottoms': [100.0, 101.0, 102.0], 'inversion_from_bottom_bottoms': [100.0, 101.0, 102.0]}, 'profile_status': 'good', 'p_surface_hpa': 1000.0, 'station_elevation_m': 100.0, 'height_source_counts': {'level': 10, 'phi': 5, 'interp': 3, 'baro': 2}, 'missing_levels': False, 'v3_fields': {'inversion_layers_v3': [100.0, 101.0, 102.0], 'pattern': 'NONE', 'n_layers': 3, 'strongest_delta_t_c_v3': 1.0}}, ...], 'day_mean': {'pressure_hpa': [1000.0, 999.0, 998.0], 'heights_m': [100.0, 101.0, 102.0], 'temperature_c': [20.0, 20.1, 20.2]}]}}, 100).
    months, n_observations = _build_months_payload(by_day, grid_points=grid_points)
    station_name = str(long_df["station_name"].iloc[0]) if len(long_df) else "" #station_name - название станции.
    station_id = str(long_df["station_id"].iloc[0]) if len(long_df) else "" #station_id - id станции.
    station_z = _station_elevation(long_df, metrics_df, station_id) #station_z - высота станции.

    return { #return - возвращает словарь с ежедневными профилями.
        "schema": SCHEMA,
        "features": list(FEATURES), #FEATURES - список признаков.
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), #built_at - дата и время сборки.
        "station_id": station_id, #station_id - id станции.
        "station_name": station_name, #station_name - название станции.
        "station_elevation_m": station_z,
        "source_tables": source, #source - источник данных.
        "pressure_top_hpa": pressure_top_hpa, #pressure_top_hpa - минимальное давление у поверхности в гПа.
        "max_surface_pressure_hpa": max_surface_pressure_hpa, #max_surface_pressure_hpa - максимальное давление у поверхности в гПа.
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
#Функция main собирает ежедневные профили из long-таблицы и метрик. Вызывается в функции main. Возвращает 0.
def main() -> int:
    parser = argparse.ArgumentParser(description="Собрать daily_profiles.json (observations_v1)")
    parser.add_argument("--long", default=str(DEFAULT_LONG_CSV)) #long - путь к long-таблице.
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS_CSV)) #metrics - путь к метрикам.
    parser.add_argument(
        "--xlsx",
        default=None,
        help="Excel с листами profiles_long и profile_metrics "
             "(если CSV нет — берётся автоматически из папки --long)",
    ) #Excel с листами profiles_long и profile_metrics (если CSV нет — берётся автоматически из папки --long)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT)) #output - путь к выходному файлу.
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
    parser.add_argument("--layers-v3", default=None, help="Готовый inversion_layers_v3.csv")
    parser.add_argument("--summary-v3", default=None, help="Готовый profile_inversion_summary_v3.csv")
    args = parser.parse_args()
#Функция build_daily_profiles собирает ежедневные профили из long-таблицы и метрик. Вызывается в функции main. Возвращает словарь с ежедневными профилями. 
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
