"""Интерактивный дашборд температурных профилей (наблюдения / сроки).

Запуск:
  py -3 -m streamlit run scripts/profile_dashboard.py

Кнопки QC только предлагают отключение наблюдений — можно править вручную.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gdex_bufr.profile_climate.obs_qc import (  # noqa: E402
    FORM_PERCENTILE,
    FORM_RMSE_MIN_C,
    HAMPEL_K,
    MIN_ABS_DP_HPA,
    MIN_LEVELS_FLAG,
    OUTLIER_MAX_ABS_DT_C,
    OUTLIER_MAX_DT_DP_SQ,
    SPIKE_ABS_C,
    form_rmse,
    form_rmse_threshold,
    is_few_levels,
    is_spike_outlier,
    max_abs_dt,
    max_dt_dp_sq,
    month_median_shape,
    prepare_plot_arrays,
    raw_plot_arrays,
    spike_scores,
    suggest_outliers_abs_dt,
    suggest_outliers_dt_dp_sq,
    suggest_outliers_few_levels,
    suggest_outliers_form,
    suggest_outliers_spike,
)
from gdex_bufr.profile_climate.config import load_profile_climate_config  # noqa: E402
from gdex_bufr.profile_climate import inversion as _inversion_mod  # noqa: E402
from gdex_bufr.profile_climate.paths import catalog_station_dir  # noqa: E402

# Streamlit может держать старый inversion.py в sys.modules.
_inversion_mod = importlib.reload(_inversion_mod)


def _v3_cfg_dict() -> dict[str, Any]:
    """Параметры gap-v3: YAML, если доступен; иначе дефолты (Streamlit может держать старый config)."""
    defaults = {
        "max_embedded_gap_m": 100.0,
        "min_strength_c": 0.3,
        "min_depth_m": None,
        "he_threshold_m": 250.0,
        "max_gap_drop_c": None,
        "surface_tolerance_m": 30.0,
    }
    try:
        cfg = load_profile_climate_config(ROOT / "profile_climate_config.yaml")
    except Exception:  # noqa: BLE001
        return defaults
    getter = getattr(cfg, "v3_detect_kwargs", None)
    if callable(getter):
        return {**defaults, **getter()}
    return {
        "max_embedded_gap_m": float(getattr(cfg, "inversion_v3_max_embedded_gap_m", 100.0)),
        "min_strength_c": float(getattr(cfg, "inversion_v3_min_strength_c", 0.3)),
        "min_depth_m": getattr(cfg, "inversion_v3_min_depth_m", None),
        "he_threshold_m": float(getattr(cfg, "inversion_v3_he_threshold_m", 250.0)),
        "max_gap_drop_c": getattr(cfg, "inversion_v3_max_gap_drop_c", None),
        "surface_tolerance_m": float(getattr(cfg, "inversion_v3_surface_tolerance_m", 30.0)),
    }


_V3_CFG = _v3_cfg_dict()

FAR_EAST_DATA = ROOT / catalog_station_dir() / "daily_profiles.json"
LEGACY_ACTUAL = ROOT / "gdex_outputs" / "актуальное" / "daily_profiles.json"
LEGACY_ALDAN = ROOT / "gdex_outputs" / "результаты-алдан" / "daily_profiles.json"


def _data_path_from_cli() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data")
    args, _ = parser.parse_known_args()
    if args.data:
        return Path(args.data)
    if FAR_EAST_DATA.exists():
        return FAR_EAST_DATA
    if LEGACY_ACTUAL.exists():
        return LEGACY_ACTUAL
    return FAR_EAST_DATA


DEFAULT_DATA = _data_path_from_cli()
LEGACY_DATA = LEGACY_ACTUAL if LEGACY_ACTUAL.exists() else LEGACY_ALDAN
REQUIRED_SCHEMA = "observations_v1"
MANUAL_LABEL_COLUMNS = [
    "profile_id",
    "annotator",
    "layer_index",
    "base_height_m",
    "top_height_m",
    "position_type",
    "confidence",
    "comment",
]

OBS_PALETTE = [
    "#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E",
    "#E6AB02", "#A6761D", "#666666", "#1F78B4", "#B2DF8A",
    "#33A02C", "#FB9A99", "#E31A1C", "#FDBF6F", "#FF7F00",
    "#CAB2D6", "#6A3D9A", "#FFFF99", "#B15928", "#8DD3C7",
    "#80B1D3", "#FDB462", "#B3DE69", "#FCCDE5",
]
MEAN_COLOR = "#8B1E3F"
DAY_MEAN_COLOR = "#4A4A4A"
NO_CLASS_COLOR = "#888888"

QUALITY_ANY = "любое"
QUALITY_LABELS = {
    "confirmed": "confirmed — подтверждённая",
    "rejected_no_lapse": "rejected_no_lapse — кандидат без падения выше",
    "none": "none — роста T от земли нет",
}
V3_TYPE_COLORS = {
    "G": "#C0392B",
    "E": "#2980B9",
    "HE": "#8E44AD",
}
LAYER_COUNT_ANY = "любое"
LAYER_COUNT_OPTIONS = {
    LAYER_COUNT_ANY: "любое",
    "0": "0 слоёв",
    "1": "1 слой",
    "2+": "2 и больше",
    "MULTI": "pattern MULTI",
}
V2_V3_ANY = "любое"
V2_V3_OPTIONS = {
    V2_V3_ANY: "любое",
    "both": "есть и v2, и v3",
    "only_v2": "только v2",
    "only_v3": "только v3",
    "neither": "ни v2, ни v3",
}
STATUS_ANY = "любое"
SOURCE_ANY = "любой"


@st.cache_data(show_spinner="Загрузка профилей…")
def load_daily(path: str, mtime_ns: int) -> dict:
    """mtime_ns — ключ кэша: после пересборки JSON подхватывается новый файл."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def has_levels(obs: dict) -> bool:
    """Наблюдение с уровнями (метрики без профиля рисовать нечем)."""
    if obs.get("missing_levels"):
        return False
    return bool(obs.get("temperature_c")) and bool(obs.get("pressure_hpa"))


def _iter_observations(days: list[dict]) -> list[dict]:
    out: list[dict] = []
    for day in days:
        for obs in day.get("observations") or []:
            out.append({**obs, "date": day["date"]})
    return out


def _parse_day(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _obs_v3_layers(obs: dict) -> list[dict]:
    layers = obs.get("inversion_layers_v3")
    return list(layers) if isinstance(layers, list) else []


def _primary_type_from_layers(layers: list[dict]) -> str | None:
    for kind in ("G", "E", "HE"):
        if any(str(ly.get("position_type") or "") == kind for ly in layers):
            return kind
    return None


def _obs_has_type(obs: dict, kind: str) -> bool:
    flag = obs.get(f"has_{kind}_v3")
    if flag is not None:
        return bool(flag)
    return any(str(ly.get("position_type") or "") == kind for ly in _obs_v3_layers(obs))


def _obs_primary_type(obs: dict, layers: list[dict] | None = None) -> str | None:
    """Один класс для цвета профиля: G → E → HE."""
    if layers is not None:
        return _primary_type_from_layers(layers)
    for kind in ("G", "E", "HE"):
        if _obs_has_type(obs, kind):
            return kind
    return None


def _obs_profile_color(
    obs: dict,
    *,
    color_by_class: bool,
    fallback: str,
    layers: list[dict] | None = None,
) -> str:
    if not color_by_class:
        return fallback
    kind = _obs_primary_type(obs, layers=layers)
    if kind is None:
        return NO_CLASS_COLOR
    return V3_TYPE_COLORS.get(kind, NO_CLASS_COLOR)


def _finite(value: Any) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _obs_layer_metric_extrema(obs: dict) -> dict[str, float | None]:
    """Мин/макс по слоям v3: основание/верх AGL, толщина, ΔT, γ."""
    layers = _obs_v3_layers(obs)
    if not layers:
        return {
            "base_agl_min": None,
            "base_agl_max": None,
            "top_agl_min": None,
            "top_agl_max": None,
            "depth_min": None,
            "depth_max": None,
            "delta_t_min": None,
            "delta_t_max": None,
            "gamma_min": None,
            "gamma_max": None,
        }
    bases = [_finite(ly.get("base_height_agl_m")) for ly in layers]
    tops = [_finite(ly.get("top_height_agl_m")) for ly in layers]
    depths = [_finite(ly.get("depth_m")) for ly in layers]
    deltas = [_finite(ly.get("delta_t_c")) for ly in layers]
    gammas = [
        _finite(ly.get("mean_gradient_c_100m") if ly.get("mean_gradient_c_100m") is not None else ly.get("gamma_c_per_100m"))
        for ly in layers
    ]
    bases_f = [v for v in bases if v is not None]
    tops_f = [v for v in tops if v is not None]
    depths_f = [v for v in depths if v is not None]
    deltas_f = [v for v in deltas if v is not None]
    gammas_f = [v for v in gammas if v is not None]
    return {
        "base_agl_min": min(bases_f) if bases_f else None,
        "base_agl_max": max(bases_f) if bases_f else None,
        "top_agl_min": min(tops_f) if tops_f else None,
        "top_agl_max": max(tops_f) if tops_f else None,
        "depth_min": min(depths_f) if depths_f else None,
        "depth_max": max(depths_f) if depths_f else None,
        "delta_t_min": min(deltas_f) if deltas_f else None,
        "delta_t_max": max(deltas_f) if deltas_f else None,
        "gamma_min": min(gammas_f) if gammas_f else None,
        "gamma_max": max(gammas_f) if gammas_f else None,
    }


def _obs_height_sources(obs: dict) -> set[str]:
    counts = obs.get("height_source_counts")
    if isinstance(counts, dict):
        return {str(key) for key, value in counts.items() if value}
    sources = obs.get("height_sources")
    if isinstance(sources, dict):
        return {str(key) for key, value in sources.items() if value}
    if isinstance(sources, list):
        return {str(item) for item in sources}
    return set()


def _obs_v2_v3_relation(obs: dict) -> str:
    has_v2 = bool(obs.get("inversion_detected"))
    has_v3 = int(obs.get("n_inversion_layers_v3") or 0) > 0
    if has_v2 and has_v3:
        return "both"
    if has_v2:
        return "only_v2"
    if has_v3:
        return "only_v3"
    return "neither"


def _range_overlaps(
    value_min: float | None,
    value_max: float | None,
    lo: float,
    hi: float,
) -> bool:
    """Есть ли пересечение [value_min, value_max] с [lo, hi]."""
    if value_min is None and value_max is None:
        return False
    left = value_min if value_min is not None else value_max
    right = value_max if value_max is not None else value_min
    assert left is not None and right is not None
    return not (right < lo or left > hi)


def filter_observations(
    observations: list[dict],
    *,
    cycle_mode: str,
    day_from: date,
    day_to: date,
    inversion_only: bool,
    inversion_quality: str = QUALITY_ANY,
    hide_missing_levels: bool = False,
    inversion_v3_only: bool = False,
    types: tuple[str, ...] | list[str] | None = None,
    top_agl_range: tuple[float, float] | None = None,
    base_agl_range: tuple[float, float] | None = None,
    depth_range: tuple[float, float] | None = None,
    delta_t_range: tuple[float, float] | None = None,
    gamma_range: tuple[float, float] | None = None,
    layer_count_mode: str = LAYER_COUNT_ANY,
    height_sources: tuple[str, ...] | list[str] | None = None,
    profile_status: str = STATUS_ANY,
    v2_v3_mode: str = V2_V3_ANY,
    profile_id_query: str = "",
) -> list[dict]:
    """Месяц уже выбран снаружи; здесь даты / cycle / инверсия / расширенные фильтры."""
    wanted_types = {str(t) for t in (types or ())}
    wanted_sources = {str(s) for s in (height_sources or ())}
    query = str(profile_id_query or "").strip().lower()
    out: list[dict] = []
    for obs in observations:
        d = _parse_day(obs["date"])
        if d < day_from or d > day_to:
            continue
        cy = str(obs.get("cycle", "")).zfill(2)[-2:]
        if cycle_mode == "00+12" and cy not in {"00", "12"}:
            continue
        if cycle_mode == "00" and cy != "00":
            continue
        if cycle_mode == "12" and cy != "12":
            continue
        if inversion_only and not obs.get("inversion_detected"):
            continue
        if inversion_v3_only and int(obs.get("n_inversion_layers_v3") or 0) <= 0:
            continue
        if inversion_quality != QUALITY_ANY:
            if str(obs.get("inversion_quality") or "") != inversion_quality:
                continue
        if hide_missing_levels and not has_levels(obs):
            continue
        if query and query not in str(obs.get("profile_id") or "").lower():
            continue
        if profile_status != STATUS_ANY:
            if str(obs.get("profile_status") or "") != profile_status:
                continue
        if v2_v3_mode != V2_V3_ANY and _obs_v2_v3_relation(obs) != v2_v3_mode:
            continue
        n_layers = int(obs.get("n_inversion_layers_v3") or 0)
        pattern = str(obs.get("inversion_pattern_v3") or obs.get("pattern_v3") or "")
        if layer_count_mode == "0" and n_layers != 0:
            continue
        if layer_count_mode == "1" and n_layers != 1:
            continue
        if layer_count_mode == "2+" and n_layers < 2:
            continue
        if layer_count_mode == "MULTI" and pattern != "MULTI":
            continue
        if wanted_types:
            if not any(_obs_has_type(obs, kind) for kind in wanted_types):
                continue
        if wanted_sources:
            sources = _obs_height_sources(obs)
            if not sources.intersection(wanted_sources):
                continue
        extrema = _obs_layer_metric_extrema(obs)
        needs_layers = any(
            rng is not None
            for rng in (top_agl_range, base_agl_range, depth_range, delta_t_range, gamma_range)
        )
        if needs_layers and n_layers <= 0:
            continue
        if top_agl_range is not None and not _range_overlaps(
            extrema["top_agl_min"], extrema["top_agl_max"], *top_agl_range
        ):
            continue
        if base_agl_range is not None and not _range_overlaps(
            extrema["base_agl_min"], extrema["base_agl_max"], *base_agl_range
        ):
            continue
        if depth_range is not None and not _range_overlaps(
            extrema["depth_min"], extrema["depth_max"], *depth_range
        ):
            continue
        if delta_t_range is not None and not _range_overlaps(
            extrema["delta_t_min"], extrema["delta_t_max"], *delta_t_range
        ):
            continue
        if gamma_range is not None and not _range_overlaps(
            extrema["gamma_min"], extrema["gamma_max"], *gamma_range
        ):
            continue
        out.append(obs)
    return out


def _visible_export_rows(observations: list[dict]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for obs in observations:
        extrema = _obs_layer_metric_extrema(obs)
        rows.append(
            {
                "profile_id": obs.get("profile_id"),
                "date": obs.get("date"),
                "datetime_utc": obs.get("datetime_utc"),
                "cycle": obs.get("cycle"),
                "profile_status": obs.get("profile_status"),
                "inversion_detected_v2": bool(obs.get("inversion_detected")),
                "inversion_quality_v2": obs.get("inversion_quality"),
                "n_inversion_layers_v3": int(obs.get("n_inversion_layers_v3") or 0),
                "has_G_v3": _obs_has_type(obs, "G"),
                "has_E_v3": _obs_has_type(obs, "E"),
                "has_HE_v3": _obs_has_type(obs, "HE"),
                "primary_type": _obs_primary_type(obs),
                "v2_v3": _obs_v2_v3_relation(obs),
                "top_agl_min_m": extrema["top_agl_min"],
                "top_agl_max_m": extrema["top_agl_max"],
                "base_agl_min_m": extrema["base_agl_min"],
                "base_agl_max_m": extrema["base_agl_max"],
                "depth_min_m": extrema["depth_min"],
                "depth_max_m": extrema["depth_max"],
                "delta_t_min_c": extrema["delta_t_min"],
                "delta_t_max_c": extrema["delta_t_max"],
                "gamma_min": extrema["gamma_min"],
                "gamma_max": extrema["gamma_max"],
                "height_sources": ";".join(sorted(_obs_height_sources(obs))),
            }
        )
    return rows


def month_mean(
    observations: list[dict],
    enabled: set[str],
    *,
    y_axis: str = "height",
    apply_plot_qc: bool = True,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Устаревшая обёртка: Method A на фиксированной сетке (см. profile_averaging)."""
    from gdex_bufr.profile_climate.profile_averaging import (
        AveragingConfig,
        AveragingFilters,
        compute_profile_average,
    )

    pool = [o for o in observations if o["profile_id"] in enabled]
    if not pool:
        return None
    years = [int(o["date"][:4]) for o in pool]
    months = {int(o["date"][5:7]) for o in pool}
    filt = AveragingFilters(
        year_start=min(years),
        year_end=max(years),
        selected_months=frozenset(months),
        cycle_mode="all",
    )
    cfg = AveragingConfig(
        method="A",
        coordinate="pressure" if y_axis == "pressure" else "height",
        apply_plot_qc=apply_plot_qc,
        multi_month_mode="combined",
        min_samples_a=1,
        min_samples_b=1,
    )
    result = compute_profile_average(pool, filt, cfg)
    if not result.months:
        return None
    item = result.months[0]
    return item.grid, item.central


def observation_plot_arrays(
    obs: dict,
    y_axis: str,
    *,
    apply_plot_qc: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Массивы для видимой кривой: исходные по умолчанию, QC только по запросу."""
    if apply_plot_qc:
        return prepare_plot_arrays(obs, y_axis)
    return raw_plot_arrays(obs, y_axis)


def _manual_labels_path(data_file: Path) -> Path:
    return data_file.parent / "manual_inversion_labels.csv"


def _append_manual_label(path: Path, row: dict[str, Any]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANUAL_LABEL_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in MANUAL_LABEL_COLUMNS})


def _first_valid_temp(temps: np.ndarray) -> float | None:
    for value in temps:
        if not np.isnan(value):
            return float(value)
    return None


def _inversion_label_suffix(obs: dict) -> str:
    if not obs.get("inversion_detected"):
        # v2: кандидат без подтверждённого падения T выше верха
        if obs.get("inversion_candidate"):
            return " · inv? кандидат"
        return ""
    h = obs.get("inversion_top_height_m")
    p = obs.get("inversion_top_pressure_hpa")
    bits = [" · inv"]
    if h is not None:
        bits.append(f" {float(h):.0f} м")
    if p is not None:
        bits.append(f" / {float(p):.0f} гПа")
    return "".join(bits)


def _format_height_sources(obs: dict) -> str:
    """Состав источников высоты внутри зонда, например `level×12, interp×3`."""
    counts = obs.get("height_source_counts") or {}
    if not counts:
        return ""
    ordered = sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
    return ", ".join(f"{name}×{count}" for name, count in ordered)


def _finite_or_none(value: float) -> float | None:
    if value == float("inf") or value != value:  # noqa: PLR0124
        return None
    return round(value, 4)


def _cycle_dash(cycle: str) -> str:
    c = str(cycle).zfill(2)[-2:]
    if c == "00":
        return "solid"
    if c == "12":
        return "dash"
    return "dot"


def _obs_state_key(state_scope: str, profile_id: str) -> str:
    return f"obs::{state_scope}::{profile_id}"


def _set_enabled(state_scope: str, observations: list[dict], predicate) -> None:
    for obs in observations:
        st.session_state[_obs_state_key(state_scope, obs["profile_id"])] = bool(predicate(obs))


def _add_inversion_marker(
    fig: go.Figure,
    obs: dict,
    *,
    y_axis: str,
    color: str,
    day_key: str,
) -> None:
    """Ромб на верху приземной инверсии v2 (снизу вверх), если координаты есть."""
    inv_t = obs.get("inversion_top_temp_c")
    inv_y = (
        obs.get("inversion_top_pressure_hpa")
        if y_axis == "pressure"
        else obs.get("inversion_top_height_m")
    )
    if inv_t is None or inv_y is None:
        return
    h_m = obs.get("inversion_top_height_m")
    p_hpa = obs.get("inversion_top_pressure_hpa")
    d_t = obs.get("inversion_delta_t_c")
    inv_hover = (
        f"Верх инверсии v2 (снизу вверх) {obs.get('datetime_utc', day_key)}<br>"
        f"T=%{{x:.1f}} °C<br>"
        + (f"H={h_m:.0f} м<br>" if h_m is not None else "")
        + (f"P={p_hpa:.0f} гПа<br>" if p_hpa is not None else "")
        + (f"ΔT={d_t:.1f} °C<br>" if d_t is not None else "")
        + "<extra></extra>"
    )
    fig.add_trace(go.Scatter(
        x=[float(inv_t)],
        y=[float(inv_y)],
        mode="markers",
        name=f"{day_key[8:]}·{obs.get('cycle', '??')} inv-v2",
        marker=dict(
            size=11,
            symbol="diamond",
            color=color,
            line=dict(width=1.2, color="#222222"),
        ),
        showlegend=False,
        hovertemplate=inv_hover,
    ))


def _obs_levels_for_v2(obs: dict) -> list[dict]:
    temps = obs.get("temperature_c") or []
    press = obs.get("pressure_hpa") or []
    heights = obs.get("heights_m") or []
    n = min(len(temps), len(press))
    rows: list[dict] = []
    for i in range(n):
        if press[i] is None:
            continue
        height = None
        if i < len(heights) and heights[i] is not None:
            height = float(heights[i])
        rows.append({
            "temperature_c": None if temps[i] is None else float(temps[i]),
            "pressure_hpa": float(press[i]),
            "height_m": height,
        })
    rows.sort(key=lambda lv: lv["pressure_hpa"], reverse=True)
    return rows


def _from_top_tops_for_obs(obs: dict) -> list[dict]:
    """Вершины from_top из JSON или пересчёт по уровням наблюдения."""
    tops = obs.get("inversion_from_top_tops")
    if isinstance(tops, list) and tops:
        return [x for x in tops if isinstance(x, dict)]
    levels = _obs_levels_for_v2(obs)
    if len(levels) < 2:
        return []
    detect_inversions_from_top = _inversion_mod.detect_inversions_from_top
    return [h.as_dict() for h in detect_inversions_from_top(levels)]


def _add_from_top_inversion_markers(
    fig: go.Figure,
    obs: dict,
    *,
    y_axis: str,
    color: str,
    day_key: str,
) -> None:
    """Круги на confirmed-вершинах метода «сверху вниз»."""
    tops = _from_top_tops_for_obs(obs)
    xs: list[float] = []
    ys: list[float] = []
    hovers: list[str] = []
    for hit in tops:
        if str(hit.get("quality") or "") != "confirmed":
            continue
        inv_t = hit.get("temperature_c")
        inv_y = (
            hit.get("pressure_hpa") if y_axis == "pressure" else hit.get("height_m")
        )
        if inv_t is None or inv_y is None:
            continue
        xs.append(float(inv_t))
        ys.append(float(inv_y))
        h_m = hit.get("height_m")
        p_hpa = hit.get("pressure_hpa")
        d_t = hit.get("delta_t_c")
        hovers.append(
            f"Инверсия сверху вниз {obs.get('datetime_utc', day_key)}<br>"
            f"T={float(inv_t):.1f} °C<br>"
            + (f"H={float(h_m):.0f} м<br>" if h_m is not None else "")
            + (f"P={float(p_hpa):.0f} гПа<br>" if p_hpa is not None else "")
            + (f"ΔT={float(d_t):.1f} °C<br>" if d_t is not None else "")
        )
    if not xs:
        return
    fig.add_trace(go.Scatter(
        x=xs,
        y=ys,
        mode="markers",
        name=f"{day_key[8:]}·{obs.get('cycle', '??')} from_top",
        marker=dict(
            size=9,
            symbol="circle",
            color=color,
            line=dict(width=1.0, color="#444444"),
            opacity=0.9,
        ),
        showlegend=False,
        hovertext=hovers,
        hoverinfo="text",
    ))


def _add_v3_layer_overlays(
    fig: go.Figure,
    obs: dict,
    *,
    y_axis: str,
    day_key: str,
    layers: list[dict] | None = None,
    types: set[str] | None = None,
) -> None:
    """Отрезки base→top для слоёв gap-v3 (G/E/HE), цвет фиксирован по классу.

    На оси высоты Y берётся из профиля по давлению слоя (как у кривой T–H),
    а не только из полей base/top_height_m — иначе сегменты «ломаются» при
    провалах/дублях H(P).
    """
    use_layers = layers if layers is not None else (obs.get("inversion_layers_v3") or [])
    for layer in use_layers:
        pos = str(layer.get("position_type") or "?")
        if types and pos not in types:
            continue
        color = V3_TYPE_COLORS.get(pos, "#555555")
        t0 = layer.get("base_temperature_c")
        t1 = layer.get("top_temperature_c")
        y0 = _layer_endpoint_y(obs, layer, which="base", y_axis=y_axis)
        y1 = _layer_endpoint_y(obs, layer, which="top", y_axis=y_axis)
        if None in (t0, t1, y0, y1):
            continue
        depth = layer.get("depth_m")
        delta = layer.get("delta_t_c")
        hover = (
            f"gap-v3 {pos} {obs.get('datetime_utc', day_key)}<br>"
            f"base T={float(t0):.1f} → top T={float(t1):.1f}<br>"
            + (f"D={float(depth):.0f} м<br>" if depth is not None else "")
            + (f"ΔT={float(delta):.2f} °C<br>" if delta is not None else "")
            + "<extra></extra>"
        )
        fig.add_trace(go.Scatter(
            x=[float(t0), float(t1)],
            y=[float(y0), float(y1)],
            mode="lines+markers",
            name=f"v3-{pos}",
            legendgroup=f"v3-{pos}",
            line=dict(width=3.2, color=color),
            marker=dict(size=7, symbol="square", color=color),
            opacity=0.9,
            showlegend=False,
            hovertemplate=hover,
        ))


def _height_at_pressure(obs: dict, pressure_hpa: float) -> float | None:
    """Высота профиля на ближайшем уровне по давлению (как на кривой графика)."""
    pressures = obs.get("pressure_hpa") or []
    heights = obs.get("heights_m") or []
    best_h: float | None = None
    best_dp = float("inf")
    for p_raw, h_raw in zip(pressures, heights):
        p = _finite(p_raw)
        h = _finite(h_raw)
        if p is None or h is None:
            continue
        dp = abs(p - pressure_hpa)
        if dp < best_dp:
            best_dp = dp
            best_h = h
    return best_h


def _layer_endpoint_y(
    obs: dict,
    layer: dict,
    *,
    which: str,
    y_axis: str,
) -> float | None:
    """Y конца слоя: давление или высота, согласованная с кривой профиля."""
    if which == "base":
        p = _finite(layer.get("base_pressure_hpa"))
        h = _finite(layer.get("base_height_m"))
    else:
        p = _finite(layer.get("top_pressure_hpa"))
        h = _finite(layer.get("top_height_m"))
    if y_axis == "pressure":
        return p
    if p is not None:
        matched = _height_at_pressure(obs, p)
        if matched is not None:
            return matched
    return h


def _add_class_legend(fig: go.Figure) -> None:
    """Одна легенда на класс G/E/HE."""
    labels = {"G": "G приземная", "E": "E приподнятая", "HE": "HE высокая"}
    for kind in ("G", "E", "HE"):
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name=labels[kind],
            legendgroup=kind,
            line=dict(width=3.0, color=V3_TYPE_COLORS[kind]),
            showlegend=True,
        ))
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode="lines",
        name="без слоя v3",
        legendgroup="none",
        line=dict(width=2.0, color=NO_CLASS_COLOR),
        showlegend=True,
    ))


def _recompute_v3_layers_for_obs(
    obs: dict,
    *,
    max_embedded_gap_m: float,
    min_strength_c: float,
    min_depth_m: float | None,
    he_threshold_m: float,
    max_gap_drop_c: float | None = None,
    surface_tolerance_m: float = 30.0,
) -> list[dict]:
    from gdex_bufr.profile_climate.inversion_layers import (
        detect_inversion_layers_gap_v3,
        layers_to_dashboard_payload,
    )

    z = np.asarray(
        [np.nan if v is None else v for v in (obs.get("heights_m") or [])],
        dtype=float,
    )
    t = np.asarray(
        [np.nan if v is None else v for v in (obs.get("temperature_c") or [])],
        dtype=float,
    )
    p = np.asarray(
        [np.nan if v is None else v for v in (obs.get("pressure_hpa") or [])],
        dtype=float,
    )
    mask = np.isfinite(z) & np.isfinite(t)
    if mask.sum() < 2:
        return []
    layers = detect_inversion_layers_gap_v3(
        z[mask],
        t[mask],
        p[mask] if p.size == z.size else None,
        max_embedded_gap_m=max_embedded_gap_m,
        min_strength_c=min_strength_c,
        min_depth_m=min_depth_m,
        he_threshold_m=he_threshold_m,
        max_gap_drop_c=max_gap_drop_c,
        surface_tolerance_m=surface_tolerance_m,
    )
    z0 = float(np.min(z[mask]))
    return layers_to_dashboard_payload(layers, z0=z0)


def _build_figure(
    *,
    visible_by_day: dict[str, list[dict]],
    enabled: set[str],
    days: list[dict],
    y_axis: str,
    y_axis_label: str,
    apply_plot_qc: bool,
    show_day_means: bool,
    show_inv_top: bool,
    show_inv_from_top: bool,
    show_v3_layers: bool,
    mean: tuple[np.ndarray, np.ndarray] | None,
    station_name: str,
    month_key: str,
    v3_layers_override: dict[str, list[dict]] | None = None,
    color_by_class: bool = True,
    layer_types: set[str] | None = None,
) -> go.Figure:
    """Собирает Plotly-график включённых наблюдений."""
    fig = go.Figure()
    y_hover = "P=%{y:.1f} гПа" if y_axis == "pressure" else "h=%{y:.0f} м"
    color_idx = 0
    day_lookup = {d["date"]: d for d in days}
    if color_by_class:
        _add_class_legend(fig)

    for day_key in sorted(visible_by_day):
        day_has_enabled = False
        for obs in visible_by_day[day_key]:
            if obs["profile_id"] not in enabled:
                continue
            prepared = observation_plot_arrays(obs, y_axis, apply_plot_qc=apply_plot_qc)
            if prepared is None:
                continue
            t_vals, y_vals = prepared
            day_has_enabled = True
            fallback = OBS_PALETTE[color_idx % len(OBS_PALETTE)]
            color_idx += 1
            overlay_layers = None
            if show_v3_layers:
                if v3_layers_override is not None:
                    overlay_layers = list(
                        v3_layers_override.get(obs["profile_id"]) or []
                    )
                else:
                    overlay_layers = list(obs.get("inversion_layers_v3") or [])
            color = _obs_profile_color(
                obs,
                color_by_class=color_by_class,
                fallback=fallback,
                layers=overlay_layers if show_v3_layers else None,
            )
            primary = _obs_primary_type(
                obs, layers=overlay_layers if show_v3_layers else None,
            )
            name = f"{day_key[8:]}·{obs.get('cycle', '??')}"
            if color_by_class and primary:
                name = f"{name} · {primary}"
            fig.add_trace(go.Scatter(
                x=t_vals,
                y=y_vals,
                mode="lines+markers" if not apply_plot_qc else "lines",
                name=name,
                legendgroup=primary or "none",
                showlegend=not color_by_class,
                line=dict(
                    width=1.6,
                    color=color,
                    dash=_cycle_dash(str(obs.get("cycle", ""))),
                ),
                marker=dict(size=3, color=color),
                opacity=0.88,
                connectgaps=False,
                hovertemplate=(
                    f"{obs.get('datetime_utc', day_key)}<br>"
                    f"CY{obs.get('cycle', '??')}"
                    + (f" · {primary}" if primary else "")
                    + f"<br>T=%{{x:.1f}} °C<br>{y_hover}<extra></extra>"
                ),
            ))
            if show_inv_top and obs.get("inversion_detected"):
                _add_inversion_marker(
                    fig, obs, y_axis=y_axis, color=color, day_key=day_key,
                )
            if show_inv_from_top:
                _add_from_top_inversion_markers(
                    fig, obs, y_axis=y_axis, color=color, day_key=day_key,
                )
            if show_v3_layers:
                _add_v3_layer_overlays(
                    fig,
                    obs,
                    y_axis=y_axis,
                    day_key=day_key,
                    layers=overlay_layers,
                    types=layer_types,
                )

        day = day_lookup.get(day_key)
        if show_day_means and day_has_enabled and day and day.get("day_mean"):
            prepared = observation_plot_arrays(
                day["day_mean"], y_axis, apply_plot_qc=apply_plot_qc,
            )
            if prepared is not None:
                t_vals, y_vals = prepared
                fig.add_trace(go.Scatter(
                    x=t_vals,
                    y=y_vals,
                    mode="lines",
                    name=f"{day_key[8:]} mean",
                    line=dict(width=2.0, color=DAY_MEAN_COLOR, dash="dot"),
                    opacity=0.55,
                    connectgaps=False,
                    hovertemplate=(
                        f"{day_key} day mean<br>"
                        f"T=%{{x:.1f}} °C<br>{y_hover}<extra></extra>"
                    ),
                ))

    if mean is not None:
        fig.add_trace(go.Scatter(
            x=mean[1],
            y=mean[0],
            mode="lines",
            name="month mean",
            line=dict(width=3.2, color=MEAN_COLOR),
            opacity=0.95,
            hovertemplate=f"month mean<br>T=%{{x:.1f}} °C<br>{y_hover}<extra></extra>",
        ))

    yaxis_cfg: dict = dict(title=y_axis_label)
    if y_axis == "pressure":
        yaxis_cfg["autorange"] = "reversed"
    fig.update_layout(
        title=f"{station_name} · {month_key}",
        xaxis_title="Температура, °C",
        yaxis=yaxis_cfg,
        height=720,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        margin=dict(l=40, r=20, t=60, b=40),
        template="plotly_white",
    )
    return fig


def _qc_table_rows(enabled_obs: list[dict]) -> tuple[list[dict], dict[str, int], float | None]:
    """Строки таблицы QC и счётчики флагов."""
    shape_stats = month_median_shape(enabled_obs)
    form_thr = form_rmse_threshold(enabled_obs)
    rows = []
    flags = {"spike": 0, "form": 0, "dt": 0, "grad": 0, "few": 0}
    for obs in enabled_obs:
        max_dt = max_abs_dt(obs)
        grad_sq = max_dt_dp_sq(obs)
        max_r, n_spike = spike_scores(obs)
        few = is_few_levels(obs)
        frmse = None
        if shape_stats is not None:
            grid, median_anom = shape_stats
            frmse = form_rmse(obs, median_anom, grid)
            if frmse == float("inf"):
                frmse = None
        flag_dt = max_dt >= OUTLIER_MAX_ABS_DT_C
        flag_grad = grad_sq >= OUTLIER_MAX_DT_DP_SQ
        flag_spike = is_spike_outlier(obs)
        flag_form = frmse is not None and form_thr is not None and frmse >= form_thr
        if flag_dt:
            flags["dt"] += 1
        if flag_grad:
            flags["grad"] += 1
        if flag_spike:
            flags["spike"] += 1
        if flag_form:
            flags["form"] += 1
        if few:
            flags["few"] += 1
        rows.append({
            "Дата": obs["date"],
            "Cycle": obs.get("cycle"),
            "profile_id": obs["profile_id"],
            "max |r| spike, °C": _finite_or_none(max_r),
            "n_spike": n_spike if n_spike < 10**8 else None,
            "form RMSE, °C": None if frmse is None else round(frmse, 3),
            "max |ΔT|, °C": _finite_or_none(max_dt),
            "max (ΔT/ΔP)²": _finite_or_none(grad_sq),
            "Уровней": obs.get("n_levels"),
            "Выброс spike?": "да" if flag_spike else "",
            "Выброс форма?": "да" if flag_form else "",
            "Выброс |ΔT|?": "да" if flag_dt else "",
            "Выброс (ΔT/ΔP)²?": "да" if flag_grad else "",
            "Мало уровней?": "да" if few else "",
            "Ts, °C": obs.get("t_surface_c"),
            "Инверсия": "да" if obs.get("inversion_detected") else "нет",
            "Качество": obs.get("inversion_quality") or "",
            "Источники H": _format_height_sources(obs),
            "H_inv, м": obs.get("inversion_top_height_m"),
            "P_inv, гПа": obs.get("inversion_top_pressure_hpa"),
            "ΔT_inv, °C": obs.get("inversion_delta_t_c"),
        })
    rows.sort(
        key=lambda r: (
            r["form RMSE, °C"] if r["form RMSE, °C"] is not None else -1,
            r["n_spike"] if r["n_spike"] is not None else -1,
            r["max (ΔT/ΔP)²"] if r["max (ΔT/ΔP)²"] is not None else -1,
            r["max |ΔT|, °C"] if r["max |ΔT|, °C"] is not None else -1,
        ),
        reverse=True,
    )
    return rows, flags, form_thr


def main() -> None:
    st.set_page_config(page_title="Aldan profile dashboard", layout="wide")
    st.title("Алдан — профили наблюдений")
    st.caption(
        "Одна кривая = один зонд (срок). По умолчанию показаны все исходные уровни "
        "без предварительной QC-фильтрации. Фильтры и исключение выбросов применяются только вручную."
    )

    data_path = st.sidebar.text_input("daily_profiles.json", str(DEFAULT_DATA))
    data_file = Path(data_path)
    if LEGACY_DATA.exists():
        st.sidebar.caption(f"Эталон (legacy): {LEGACY_DATA}")
    if not data_file.exists() and LEGACY_DATA.exists():
        data_file = LEGACY_DATA
    if not data_file.exists():
        st.error(
            "Нет файла данных. Сначала выполните:\n\n"
            "`python scripts/build_daily_profiles.py`"
        )
        return

    data_mtime_ns = data_file.stat().st_mtime_ns
    try:
        data = load_daily(str(data_file), data_mtime_ns)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as error:
        load_daily.clear()
        st.error(
            f"Не удалось прочитать `{data_file}`: {error}\n\n"
            "Файл повреждён или запись не завершилась. Пересоберите:\n\n"
            "`python scripts/build_daily_profiles.py`"
        )
        return
    if not isinstance(data, dict):
        st.error(f"Ожидался JSON-объект, а в `{data_file}` лежит `{type(data).__name__}`.")
        return
    if data.get("schema") != REQUIRED_SCHEMA:
        st.error(
            f"Нужен JSON со схемой `{REQUIRED_SCHEMA}` (сейчас: `{data.get('schema')}`).\n\n"
            "Пересоберите:\n\n`python scripts/build_daily_profiles.py`\n\n"
            "Если уже пересобрали — сбросьте кэш Streamlit (кнопка ниже или **C** в браузере)."
        )
        if st.button("Сбросить кэш данных"):
            load_daily.clear()
            st.rerun()
        return

    months = sorted(data["months"].keys())
    if not months:
        st.error("В JSON нет месяцев.")
        return

    level_mode = data.get("level_mode", "legacy/clean")
    features = set(data.get("features") or [])
    st.sidebar.success(
        f"schema={REQUIRED_SCHEMA} · уровни={level_mode} · "
        f"n_obs={data.get('n_observations', '—')} · n_levels={data.get('n_levels', '—')}"
        + (
            f" · z_ст={data['station_elevation_m']:.0f} м"
            if data.get("station_elevation_m") is not None
            else ""
        )
    )
    if "inversion_quality" not in features:
        st.sidebar.caption(
            "JSON собран прежней версией: нет полей качества инверсии (v2). "
            "Пересоберите `python scripts/build_daily_profiles.py`, чтобы включить фильтр."
        )

    years = sorted({m[:4] for m in months})
    col_y, col_m = st.sidebar.columns(2)
    year = col_y.selectbox("Год", years, index=len(years) - 1)
    year_months = [m for m in months if m.startswith(year)]
    month_labels = {m: m[5:] for m in year_months}
    month_key = col_m.selectbox(
        "Месяц",
        year_months,
        format_func=lambda m: month_labels[m],
        index=0,
    )

    days = data["months"][month_key]["days"]
    observations = _iter_observations(days)
    if not observations:
        st.warning("В выбранном месяце нет наблюдений.")
        return

    day_dates = sorted({_parse_day(o["date"]) for o in observations})
    d_min, d_max = day_dates[0], day_dates[-1]

    st.sidebar.markdown("### Фильтр")
    cycle_mode = st.sidebar.radio(
        "Срок (UTC)",
        options=["00+12", "00", "12", "Все сроки"],
        index=0,
        horizontal=True,
        help="00+12 — только основные сроки. «Все сроки» включает 06/18, если они есть в JSON.",
    )
    if d_min == d_max:
        day_from = day_to = d_min
        st.sidebar.caption(f"День: {d_min.isoformat()}")
    else:
        day_from, day_to = st.sidebar.slider(
            "Диапазон дней",
            min_value=d_min,
            max_value=d_max,
            value=(d_min, d_max),
            format="DD.MM",
        )

    profile_id_query = st.sidebar.text_input(
        "Поиск profile_id",
        value="",
        help="Подстрока без учёта регистра.",
    )
    preset = st.sidebar.selectbox(
        "Пресет фильтров",
        options=[
            "без пресета",
            "только G",
            "только E",
            "только HE",
            "confirmed v2",
            "v3 MULTI",
            "v2 и v3 вместе",
            "только v3",
        ],
        index=0,
    )

    inversion_only = st.sidebar.checkbox(
        "Только с инверсией (v2)",
        value=(preset == "confirmed v2"),
        help="Фильтр по legacy v2: inversion_detected=True (confirmed).",
    )
    inversion_v3_only = False
    if "inversion_v3" in features:
        inversion_v3_only = st.sidebar.checkbox(
            "Только со слоями gap-v3",
            value=(preset in {"только G", "только E", "только HE", "v3 MULTI", "только v3"}),
            help="n_inversion_layers_v3 > 0 (не заменяет фильтр v2).",
        )
    inversion_quality = QUALITY_ANY
    if "inversion_quality" in features:
        present_qualities = sorted(
            {
                str(o.get("inversion_quality") or "")
                for o in observations
                if o.get("inversion_quality")
            }
        )
        if present_qualities:
            default_quality = "confirmed" if preset == "confirmed v2" and "confirmed" in present_qualities else QUALITY_ANY
            quality_options = [QUALITY_ANY, *present_qualities]
            quality_index = quality_options.index(default_quality) if default_quality in quality_options else 0
            inversion_quality = st.sidebar.selectbox(
                "Качество инверсии (v2)",
                options=quality_options,
                index=quality_index,
                format_func=lambda q: QUALITY_LABELS.get(q, q),
                help="confirmed = подтверждена падением T выше верха; остальные — для разбора.",
            )
    hide_missing_levels = st.sidebar.checkbox(
        "Скрыть наблюдения без уровней",
        value=False,
        help="Профили, у которых есть только метрики: в списке видны, но на графике их нет.",
    )

    selected_types: list[str] = []
    layer_count_mode = LAYER_COUNT_ANY
    v2_v3_mode = V2_V3_ANY
    top_agl_range = None
    base_agl_range = None
    depth_range = None
    delta_t_range = None
    gamma_range = None
    selected_sources: list[str] = []
    profile_status = STATUS_ANY

    if "inversion_v3" in features:
        st.sidebar.markdown("### Слои v3 (G/E/HE)")
        default_types: list[str] = []
        if preset == "только G":
            default_types = ["G"]
        elif preset == "только E":
            default_types = ["E"]
        elif preset == "только HE":
            default_types = ["HE"]
        selected_types = st.sidebar.multiselect(
            "Классы инверсии",
            options=["G", "E", "HE"],
            default=default_types,
            help="Пустой список = не фильтровать по классу. Цвет профиля = класс.",
        )
        layer_default = "MULTI" if preset == "v3 MULTI" else LAYER_COUNT_ANY
        layer_keys = list(LAYER_COUNT_OPTIONS.keys())
        layer_count_mode = st.sidebar.selectbox(
            "Число слоёв v3",
            options=layer_keys,
            index=layer_keys.index(layer_default),
            format_func=lambda key: LAYER_COUNT_OPTIONS[key],
        )
        v2v3_default = (
            "both" if preset == "v2 и v3 вместе"
            else "only_v3" if preset == "только v3"
            else V2_V3_ANY
        )
        v2v3_keys = list(V2_V3_OPTIONS.keys())
        v2_v3_mode = st.sidebar.selectbox(
            "Сравнение v2 / v3",
            options=v2v3_keys,
            index=v2v3_keys.index(v2v3_default),
            format_func=lambda key: V2_V3_OPTIONS[key],
        )

        use_geom = st.sidebar.checkbox("Фильтр по геометрии слоёв", value=False)
        if use_geom:
            top_agl_range = st.sidebar.slider("Высота верха AGL, м", 0.0, 5000.0, (0.0, 5000.0), 50.0)
            base_agl_range = st.sidebar.slider("Высота основания AGL, м", 0.0, 5000.0, (0.0, 5000.0), 50.0)
            depth_range = st.sidebar.slider("Толщина слоя, м", 0.0, 3000.0, (0.0, 3000.0), 25.0)
            delta_t_range = st.sidebar.slider("ΔT слоя, °C", 0.0, 20.0, (0.0, 20.0), 0.2)
            gamma_range = st.sidebar.slider("γ слоя, °C/100 м", -5.0, 20.0, (-5.0, 20.0), 0.2)

    present_statuses = sorted(
        {
            str(o.get("profile_status") or "")
            for o in observations
            if o.get("profile_status")
        }
    )
    if present_statuses:
        profile_status = st.sidebar.selectbox(
            "Статус QC профиля",
            options=[STATUS_ANY, *present_statuses],
        )

    all_sources = sorted({src for o in observations for src in _obs_height_sources(o)})
    if all_sources:
        selected_sources = st.sidebar.multiselect(
            "Источник высоты",
            options=all_sources,
            default=[],
            help="Пустой список = любой источник.",
        )

    st.sidebar.markdown("### Отображение")
    show_day_means = st.sidebar.checkbox(
        "Суточные средние",
        value=False,
        help="Серые линии day_mean для дней с ≥1 включённым наблюдением.",
    )
    color_by_class = st.sidebar.checkbox(
        "Цвет профиля по классу G/E/HE",
        value=True,
        help="G красный, E синий, HE фиолетовый. Без слоя v3 — серый.",
    )
    show_inv_top = st.sidebar.checkbox(
        "Верх инверсии v2 (снизу вверх, ромб)",
        value=True,
        help="Ромб на confirmed-верху приземной инверсии (метод от земли).",
    )
    show_inv_from_top = st.sidebar.checkbox(
        "Инверсии сверху вниз (круги)",
        value=True,
        help="Круги на confirmed-вершинах всех слоёв роста T (проход сверху вниз).",
    )
    has_v3_data = any(int(o.get("n_inversion_layers_v3") or 0) > 0 for o in observations)
    show_v3_layers = st.sidebar.checkbox(
        "Слои gap-v3 (G/E/HE)",
        value=True if has_v3_data else False,
        help=(
            "Отрезки base→top слоёв из daily_profiles.json. "
            "Если в файле слоёв нет — пересоберите JSON или включите пересчёт по параметрам ниже."
        ),
    )
    if show_v3_layers and not has_v3_data:
        st.sidebar.warning(
            "В JSON нет слоёв v3. Пересоберите: "
            "`py -3 scripts/build_daily_profiles.py` "
            "или включите «Пересчитать слои по параметрам ниже»."
        )

    st.sidebar.markdown("### Параметры детекции gap-v3")
    v3_gap = st.sidebar.slider(
        "max_embedded_gap_m", 60.0, 140.0, float(_V3_CFG["max_embedded_gap_m"]), 10.0,
    )
    v3_strength = st.sidebar.slider(
        "min_strength_c", 0.1, 1.0, float(_V3_CFG["min_strength_c"]), 0.1,
    )
    v3_he = st.sidebar.slider(
        "he_base_threshold_m", 100.0, 400.0, float(_V3_CFG["he_threshold_m"]), 25.0,
    )
    v3_min_depth_on = st.sidebar.checkbox(
        "Включить min_depth_m",
        value=_V3_CFG["min_depth_m"] is not None,
    )
    v3_min_depth = st.sidebar.number_input(
        "min_depth_m",
        min_value=0.0,
        value=float(_V3_CFG["min_depth_m"] or 50.0),
        step=10.0,
        disabled=not v3_min_depth_on,
    )
    v3_gap_drop_on = st.sidebar.checkbox(
        "Ограничить падение внутри gap",
        value=_V3_CFG["max_gap_drop_c"] is not None,
        help="Не склеивать сегменты, если T в разрыве падает сильнее порога.",
    )
    v3_gap_drop = st.sidebar.slider(
        "max_gap_drop_c",
        0.2,
        2.0,
        float(_V3_CFG["max_gap_drop_c"] or 0.5),
        0.1,
        disabled=not v3_gap_drop_on,
    )
    v3_use_slider_params = st.sidebar.checkbox(
        "Пересчитать слои по параметрам ниже",
        value=False,
        help=(
            "Выключено (рекомендуется): слои из daily_profiles.json. "
            "Включено: заново найти слои для профилей на экране по слайдерам "
            "(не перезаписывает JSON)."
        ),
    )

    y_axis_label = st.sidebar.radio(
        "Вертикальная ось",
        options=["Давление, гПа", "Высота, м"],
        index=0,
        help=(
            "Перевод осей 1:1: уровни всегда идут от земли вверх по давлению. "
            "Меняется только координата Y (гПа или м), набор температур тот же."
        ),
    )
    y_axis = "pressure" if y_axis_label.startswith("Давление") else "height"
    apply_plot_qc = st.sidebar.checkbox(
        "Подготовить кривые (убрать петли и дубли)",
        value=False,
        help=(
            "Выключено: те же уровни, что и по гПа; на оси метров видны провалы H(P). "
            "Включено: на оси метров отбрасываются уровни с падением высоты при подъёме."
        ),
    )

    visible = filter_observations(
        observations,
        cycle_mode=cycle_mode,
        day_from=day_from,
        day_to=day_to,
        inversion_only=inversion_only,
        inversion_quality=inversion_quality,
        hide_missing_levels=hide_missing_levels,
        inversion_v3_only=inversion_v3_only,
        types=selected_types or None,
        top_agl_range=tuple(top_agl_range) if top_agl_range is not None else None,
        base_agl_range=tuple(base_agl_range) if base_agl_range is not None else None,
        depth_range=tuple(depth_range) if depth_range is not None else None,
        delta_t_range=tuple(delta_t_range) if delta_t_range is not None else None,
        gamma_range=tuple(gamma_range) if gamma_range is not None else None,
        layer_count_mode=layer_count_mode,
        height_sources=selected_sources or None,
        profile_status=profile_status,
        v2_v3_mode=v2_v3_mode,
        profile_id_query=profile_id_query,
    )
    visible_ids = {o["profile_id"] for o in visible}
    visible_plottable = [o for o in visible if has_levels(o)]
    n_visible_missing = len(visible) - len(visible_plottable)
    state_scope = f"{data_file.resolve()}::{data_mtime_ns}::{month_key}"
    visible_by_day: dict[str, list[dict]] = {}
    for obs in visible:
        visible_by_day.setdefault(obs["date"], []).append(obs)

    st.sidebar.markdown("### Наблюдения")
    p1, p2 = st.sidebar.columns(2)
    if p1.button("Все видимые", width="stretch"):
        _set_enabled(state_scope, observations, lambda o: o["profile_id"] in visible_ids)
    if p2.button("Сброс видимых", width="stretch"):
        for obs in visible:
            st.session_state[_obs_state_key(state_scope, obs["profile_id"])] = False
    p3, p4 = st.sidebar.columns(2)
    if p3.button("Только 00", width="stretch"):
        _set_enabled(
            state_scope,
            observations,
            lambda o: o["profile_id"] in visible_ids and str(o.get("cycle", "")).zfill(2)[-2:] == "00",
        )
    if p4.button("Только 12", width="stretch"):
        _set_enabled(
            state_scope,
            observations,
            lambda o: o["profile_id"] in visible_ids and str(o.get("cycle", "")).zfill(2)[-2:] == "12",
        )

    st.sidebar.markdown("### Выбросы (кандидаты)")
    q1, q2 = st.sidebar.columns(2)
    q3, q4 = st.sidebar.columns(2)
    q5, _ = st.sidebar.columns(2)

    def _apply_outliers(outlier_ids: set[str]) -> None:
        for obs in visible:
            pid = obs["profile_id"]
            st.session_state[_obs_state_key(state_scope, pid)] = pid not in outlier_ids

    visible_id_list = [o["profile_id"] for o in visible]
    if q1.button(
        "по spike",
        width="stretch",
        help=f"Hampel: |r| > max({HAMPEL_K}·1.4826·MAD(r), {SPIKE_ABS_C}°C)",
    ):
        _apply_outliers(set(suggest_outliers_spike(visible, set(visible_id_list))))
    if q2.button(
        "по форме",
        width="stretch",
        help=f"RMSE(T−Ts) ≥ max(P{FORM_PERCENTILE:.0f}, {FORM_RMSE_MIN_C}°C)",
    ):
        _apply_outliers(set(suggest_outliers_form(visible, set(visible_id_list))))
    if q3.button(
        "по |ΔT|",
        width="stretch",
        help=f"max |ΔT| ≥ {OUTLIER_MAX_ABS_DT_C} °C (по давлению)",
    ):
        _apply_outliers(set(suggest_outliers_abs_dt(visible, set(visible_id_list))))
    if q4.button(
        "по (ΔT/ΔP)²",
        width="stretch",
        help=f"max (ΔT/ΔP)² ≥ {OUTLIER_MAX_DT_DP_SQ}",
    ):
        _apply_outliers(set(suggest_outliers_dt_dp_sq(visible, set(visible_id_list))))
    if q5.button(
        "мало уровней",
        width="stretch",
        help=f"n_levels < {MIN_LEVELS_FLAG}",
    ):
        _apply_outliers(set(suggest_outliers_few_levels(visible, set(visible_id_list))))

    st.sidebar.caption(
        f"spike k={HAMPEL_K} abs≥{SPIKE_ABS_C}°C · "
        f"form P{FORM_PERCENTILE:.0f}/min{FORM_RMSE_MIN_C}°C · "
        f"|ΔT|≥{OUTLIER_MAX_ABS_DT_C}°C · "
        f"(ΔT/ΔP)²≥{OUTLIER_MAX_DT_DP_SQ} · "
        f"n<{MIN_LEVELS_FLAG} · min|ΔP|={MIN_ABS_DP_HPA} гПа"
    )

    enabled: set[str] = set()
    with st.sidebar.expander("Список наблюдений", expanded=True):
        if not visible_by_day:
            st.caption("Нет наблюдений по текущему фильтру.")
        for day_key in sorted(visible_by_day):
            day_obs = visible_by_day[day_key]
            st.markdown(f"**{day_key[8:]}** · n={len(day_obs)}")
            for obs in day_obs:
                key = _obs_state_key(state_scope, obs["profile_id"])
                if key not in st.session_state:
                    st.session_state[key] = True
                label = (
                    f"CY{obs.get('cycle', '??')} · "
                    f"Ts={obs.get('t_surface_c')}°C · "
                    f"L={obs.get('n_levels', len(obs.get('temperature_c') or []))}"
                )
                if obs.get("missing_levels"):
                    label += " · нет уровней"
                label += _inversion_label_suffix(obs)
                if st.checkbox(label, key=key):
                    enabled.add(obs["profile_id"])

    st.sidebar.caption(f"Включено: {len(enabled)} / видимых {len(visible)}")

    enabled_plottable = [o for o in visible_plottable if o["profile_id"] in enabled]
    n_enabled_missing = sum(
        1 for o in visible if o["profile_id"] in enabled and not has_levels(o)
    )
    mean = None

    v3_override: dict[str, list[dict]] | None = None
    if show_v3_layers and v3_use_slider_params:
        v3_override = {}
        for obs in enabled_plottable:
            v3_override[obs["profile_id"]] = _recompute_v3_layers_for_obs(
                obs,
                max_embedded_gap_m=float(v3_gap),
                min_strength_c=float(v3_strength),
                min_depth_m=float(v3_min_depth) if v3_min_depth_on else None,
                he_threshold_m=float(v3_he),
                max_gap_drop_c=float(v3_gap_drop) if v3_gap_drop_on else None,
                surface_tolerance_m=float(_V3_CFG["surface_tolerance_m"]),
            )

    tab_month, tab_clim = st.tabs(["Профили месяца", "Климатологическое усреднение"])

    with tab_clim:
        from scripts.dashboard_climatology import render_climatology_tab

        render_climatology_tab(data)

    with tab_month:
        st.caption(
            "Линия «month mean» заменена вкладкой «Климатологическое усреднение» "
            "(методы A/B, фиксированная сетка 500–925 гПа)."
        )

        # На графике участвуют только наблюдения с уровнями — счётчики считаем по ним же.
        st.info(
            f"Уровни: **{'подготовленные' if apply_plot_qc else 'все исходные без QC'}** · "
            f"срок **{cycle_mode}** · дни "
            f"**{day_from.isoformat()}…{day_to.isoformat()}**"
            + (" · только инверсии v2" if inversion_only else "")
            + (" · только слои v3" if inversion_v3_only else "")
            + (
                f" · качество **{inversion_quality}**"
                if inversion_quality != QUALITY_ANY
                else ""
            )
            + f" · на графике **{len(enabled_plottable)}** из **{len(visible_plottable)}**"
        )

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("На графике", f"{len(enabled_plottable)} / {len(visible_plottable)}")
        m2.metric(
            "Без уровней",
            f"{n_enabled_missing} / {n_visible_missing}",
            help="Профили только с метриками: в счёт кривых не входят.",
        )
        m3.metric("Дней с данными", len({o["date"] for o in enabled_plottable}))
        m4.metric(
            "С инверсией v2",
            sum(1 for o in enabled_plottable if o.get("inversion_detected")),
        )
        n_v3 = sum(1 for o in enabled_plottable if int(o.get("n_inversion_layers_v3") or 0) > 0)
        inv_heights = [
            float(o["inversion_top_height_m"])
            for o in enabled_plottable
            if o.get("inversion_detected") and o.get("inversion_top_height_m") is not None
        ]
        if inv_heights:
            m5.metric("Ср. H_inv v2, м", f"{sum(inv_heights) / len(inv_heights):.0f}")
        else:
            m5.metric("Со слоями v3", n_v3)

        fig = _build_figure(
            visible_by_day=visible_by_day,
            enabled=enabled,
            days=days,
            y_axis=y_axis,
            y_axis_label=y_axis_label,
            apply_plot_qc=apply_plot_qc,
            show_day_means=show_day_means,
            show_inv_top=show_inv_top,
            show_inv_from_top=show_inv_from_top,
            show_v3_layers=show_v3_layers,
            mean=mean,
            station_name=str(data.get("station_name", "Aldan")),
            month_key=month_key,
            v3_layers_override=v3_override,
            color_by_class=color_by_class,
            layer_types=set(selected_types) if selected_types else None,
        )
        st.plotly_chart(fig, width="stretch")

        if enabled_plottable:
            import pandas as pd

            export_df = pd.DataFrame(_visible_export_rows(enabled_plottable))
            st.download_button(
                "Скачать видимую выборку (CSV)",
                data=export_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"dashboard_selection_{month_key}.csv",
                mime="text/csv",
            )

        if enabled_plottable:
            st.subheader("Ручная разметка слоя (gold set)")
            labels_path = _manual_labels_path(data_file)
            options = [o["profile_id"] for o in enabled_plottable]
            pick = st.selectbox("profile_id", options=options)
            chosen = next(o for o in enabled_plottable if o["profile_id"] == pick)
            c1, c2, c3 = st.columns(3)
            base_h = c1.number_input(
                "Base, м",
                value=float(chosen.get("heights_m")[0] or 0.0) if chosen.get("heights_m") else 0.0,
                step=10.0,
            )
            top_h = c2.number_input(
                "Top, м",
                value=float(chosen.get("heights_m")[-1] or 0.0) if chosen.get("heights_m") else 0.0,
                step=10.0,
            )
            pos_type = c3.selectbox("Type", options=["G", "E", "HE"])
            c4, c5 = st.columns(2)
            confidence = c4.selectbox("confidence", options=["high", "medium", "low"])
            annotator = c5.text_input("annotator", value="operator")
            comment = st.text_input("comment", value="")
            if st.button("Сохранить слой в manual_inversion_labels.csv"):
                existing_n = 0
                if labels_path.exists():
                    import csv as _csv

                    with labels_path.open(encoding="utf-8", newline="") as handle:
                        existing_n = sum(
                            1 for r in _csv.DictReader(handle) if r.get("profile_id") == pick
                        )
                _append_manual_label(labels_path, {
                    "profile_id": pick,
                    "annotator": annotator,
                    "layer_index": existing_n,
                    "base_height_m": round(float(base_h), 1),
                    "top_height_m": round(float(top_h), 1),
                    "position_type": pos_type,
                    "confidence": confidence,
                    "comment": comment,
                })
                st.success(f"Записано в {labels_path}")
            st.caption(f"Файл меток: {labels_path}")

        if show_v3_layers and enabled_plottable:
            export_rows = []
            for obs in enabled_plottable:
                layers = (
                    (v3_override or {}).get(obs["profile_id"])
                    if v3_override is not None
                    else (obs.get("inversion_layers_v3") or [])
                )
                for layer in layers or []:
                    export_rows.append({
                        "profile_id": obs["profile_id"],
                        "datetime_utc": obs.get("datetime_utc"),
                        "cycle": obs.get("cycle"),
                        "inversion_detected_v2": bool(obs.get("inversion_detected")),
                        "max_embedded_gap_m": float(v3_gap),
                        "min_strength_c": float(v3_strength),
                        "he_threshold_m": float(v3_he),
                        "min_depth_m": float(v3_min_depth) if v3_min_depth_on else None,
                        "max_gap_drop_c": float(v3_gap_drop) if v3_gap_drop_on else None,
                        **layer,
                    })
            if export_rows:
                import csv
                from io import StringIO

                buf = StringIO()
                fieldnames = list(export_rows[0].keys())
                writer = csv.DictWriter(buf, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(export_rows)
                st.download_button(
                    "Экспорт сравнения v2/v3 (видимые)",
                    data=buf.getvalue(),
                    file_name=f"inversion_compare_{month_key}.csv",
                    mime="text/csv",
                )

        if enabled_plottable:
            rows, flags, form_thr = _qc_table_rows(enabled_plottable)
            st.subheader("Сравнение критериев выбросов (наблюдения)")
            c_a, c_b, c_c, c_d, c_e = st.columns(5)
            c_a.metric("Флаг spike", flags["spike"])
            c_b.metric("Флаг форма", flags["form"])
            c_c.metric("Флаг |ΔT|", flags["dt"])
            c_d.metric("Флаг (ΔT/ΔP)²", flags["grad"])
            c_e.metric("Мало уровней", flags["few"])
            if form_thr is not None:
                st.caption(
                    f"Порог формы (P{FORM_PERCENTILE:.0f}, min {FORM_RMSE_MIN_C}°C): {form_thr:.2f} °C"
                )
            st.dataframe(rows, width="stretch", hide_index=True)
            st.caption(
                f"Источник: {Path(data_path).name} · schema={REQUIRED_SCHEMA} · "
                f"spike k={HAMPEL_K} abs≥{SPIKE_ABS_C}°C · "
                f"form P{FORM_PERCENTILE:.0f}/min{FORM_RMSE_MIN_C}°C · "
                f"|ΔT|≥{OUTLIER_MAX_ABS_DT_C}°C · "
                f"(ΔT/ΔP)²≥{OUTLIER_MAX_DT_DP_SQ} · "
                f"n<{MIN_LEVELS_FLAG}"
            )


if __name__ == "__main__":
    main()
