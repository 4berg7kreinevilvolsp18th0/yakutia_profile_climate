"""Многослойные температурные инверсии: gap-merge v3 (параллельно legacy v2)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

PositionType = Literal["G", "E", "HE"]
PATTERN_NONE = "NONE"
PATTERN_MULTI = "MULTI"


@dataclass
class InversionLayer:
    base_idx: int
    top_idx: int
    base_height_m: float
    top_height_m: float
    base_pressure_hpa: float | None
    top_pressure_hpa: float | None
    base_temperature_c: float
    top_temperature_c: float
    depth_m: float
    delta_t_c: float
    mean_gradient_c_100m: float
    position_type: PositionType
    embedded_gap_count: int
    embedded_gap_depth_total_m: float
    method: str = "gap_v3"

    def as_row(
        self,
        *,
        profile_id: str,
        layer_index: int,
        z0: float,
    ) -> dict[str, Any]:
        return {
            "profile_id": profile_id,
            "layer_index": layer_index,
            "base_idx": self.base_idx,
            "top_idx": self.top_idx,
            "base_height_m": self.base_height_m,
            "top_height_m": self.top_height_m,
            "base_height_agl_m": self.base_height_m - z0,
            "top_height_agl_m": self.top_height_m - z0,
            "base_pressure_hpa": self.base_pressure_hpa,
            "top_pressure_hpa": self.top_pressure_hpa,
            "base_temperature_c": self.base_temperature_c,
            "top_temperature_c": self.top_temperature_c,
            "depth_m": self.depth_m,
            "delta_t_c": self.delta_t_c,
            "mean_gradient_c_100m": self.mean_gradient_c_100m,
            "position_type": self.position_type,
            "embedded_gap_count": self.embedded_gap_count,
            "embedded_gap_depth_total_m": self.embedded_gap_depth_total_m,
            "method": self.method,
        }


@dataclass
class _MergedRun:
    base_idx: int
    top_idx: int
    embedded_gap_count: int = 0
    embedded_gap_depth_total_m: float = 0.0


def positive_runs(gradient: np.ndarray) -> list[tuple[int, int]]:
    """Смежные интервалы с g>0 → пары индексов точек (base_idx, top_idx)."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    n = len(gradient)
    for i in range(n):
        g = gradient[i]
        is_pos = bool(np.isfinite(g) and g > 0)
        if is_pos:
            if start is None:
                start = i
            if i == n - 1 and start is not None:
                runs.append((start, i + 1))
                start = None
        elif start is not None:
            runs.append((start, i))
            start = None
    return runs


def merge_short_gaps(
    runs: list[tuple[int, int]],
    z: np.ndarray,
    *,
    max_gap_depth_m: float = 100.0,
    max_gap_drop_c: float | None = None,
    t: np.ndarray | None = None,
    max_total_embedded_gap_m: float | None = None,
    max_gap_fraction: float | None = None,
) -> list[_MergedRun]:
    """Объединяет положительные сегменты через тонкий неинверсионный промежуток."""
    if not runs:
        return []

    merged: list[_MergedRun] = [
        _MergedRun(base_idx=runs[0][0], top_idx=runs[0][1]),
    ]

    for base, top in runs[1:]:
        prev = merged[-1]
        gap_depth = float(z[base] - z[prev.top_idx])
        if gap_depth < 0:
            merged.append(_MergedRun(base_idx=base, top_idx=top))
            continue

        allow = gap_depth < max_gap_depth_m
        if allow and max_gap_drop_c is not None and t is not None:
            # Экспериментально: слишком сильное падение T в gap не склеиваем.
            drop = float(t[base] - t[prev.top_idx])
            if drop < -abs(max_gap_drop_c):
                allow = False
        new_total = prev.embedded_gap_depth_total_m + gap_depth
        new_depth = float(z[top] - z[prev.base_idx])
        if allow and max_total_embedded_gap_m is not None and new_total > max_total_embedded_gap_m:
            allow = False
        if (
            allow
            and max_gap_fraction is not None
            and new_depth > 0
            and (new_total / new_depth) > max_gap_fraction
        ):
            allow = False

        if allow:
            prev.embedded_gap_count += 1
            prev.embedded_gap_depth_total_m += gap_depth
            prev.top_idx = top
        else:
            merged.append(_MergedRun(base_idx=base, top_idx=top))

    return merged


def position_type(
    base_idx: int,
    base_height_agl_m: float,
    *,
    he_threshold_m: float = 250.0,
    surface_tolerance_m: float = 30.0,
) -> PositionType:
    # G — слой у поверхности по AGL, не по индексу массива.
    _ = base_idx
    if base_height_agl_m <= surface_tolerance_m:
        return "G"
    if base_height_agl_m <= he_threshold_m:
        return "E"
    return "HE"


def detect_inversion_layers_gap_v3(
    z,
    t,
    p=None,
    *,
    max_embedded_gap_m: float = 100.0,
    min_strength_c: float = 0.3,
    min_depth_m: float | None = None,
    he_threshold_m: float = 250.0,
    max_gap_drop_c: float | None = None,
    surface_tolerance_m: float = 30.0,
    max_total_embedded_gap_m: float | None = None,
    max_gap_fraction: float | None = None,
) -> list[InversionLayer]:
    """Gap-merge v3: все инверсионные слои профиля (G / E / HE)."""
    z_arr = np.asarray(z, dtype=float)
    t_arr = np.asarray(t, dtype=float)
    if p is None:
        p_arr = np.full_like(z_arr, np.nan, dtype=float)
    else:
        p_arr = np.asarray(p, dtype=float)

    if z_arr.size < 2 or t_arr.size != z_arr.size:
        return []

    order = np.argsort(z_arr, kind="mergesort")
    z_arr = z_arr[order]
    t_arr = t_arr[order]
    p_arr = p_arr[order]

    dz = np.diff(z_arr)
    dtemp = np.diff(t_arr)
    grad = np.full_like(dtemp, np.nan, dtype=float)
    valid = (dz > 0) & np.isfinite(dtemp) & np.isfinite(dz)
    grad[valid] = 100.0 * dtemp[valid] / dz[valid]

    runs = positive_runs(grad)
    merged = merge_short_gaps(
        runs,
        z_arr,
        max_gap_depth_m=max_embedded_gap_m,
        max_gap_drop_c=max_gap_drop_c,
        t=t_arr,
        max_total_embedded_gap_m=max_total_embedded_gap_m,
        max_gap_fraction=max_gap_fraction,
    )

    z0 = float(z_arr[0])
    layers: list[InversionLayer] = []
    for run in merged:
        base_idx = int(run.base_idx)
        top_idx = int(run.top_idx)
        if top_idx <= base_idx:
            continue
        if not np.isfinite(t_arr[base_idx]) or not np.isfinite(t_arr[top_idx]):
            continue

        depth = float(z_arr[top_idx] - z_arr[base_idx])
        strength = float(t_arr[top_idx] - t_arr[base_idx])
        if strength < min_strength_c:
            continue
        if min_depth_m is not None and depth < min_depth_m:
            continue
        if depth <= 0:
            continue

        base_agl = float(z_arr[base_idx] - z0)
        pos = position_type(
            base_idx,
            base_agl,
            he_threshold_m=he_threshold_m,
            surface_tolerance_m=surface_tolerance_m,
        )
        base_p = float(p_arr[base_idx]) if np.isfinite(p_arr[base_idx]) else None
        top_p = float(p_arr[top_idx]) if np.isfinite(p_arr[top_idx]) else None

        layers.append(
            InversionLayer(
                base_idx=base_idx,
                top_idx=top_idx,
                base_height_m=float(z_arr[base_idx]),
                top_height_m=float(z_arr[top_idx]),
                base_pressure_hpa=base_p,
                top_pressure_hpa=top_p,
                base_temperature_c=float(t_arr[base_idx]),
                top_temperature_c=float(t_arr[top_idx]),
                depth_m=depth,
                delta_t_c=strength,
                mean_gradient_c_100m=100.0 * strength / depth,
                position_type=pos,
                embedded_gap_count=int(run.embedded_gap_count),
                embedded_gap_depth_total_m=float(run.embedded_gap_depth_total_m),
            )
        )
    return layers


def _pattern_from_flags(has_g: bool, has_e: bool, has_he: bool, n: int) -> str:
    if n <= 0:
        return PATTERN_NONE
    types = []
    if has_g:
        types.append("G")
    if has_e:
        types.append("E")
    if has_he:
        types.append("HE")
    if n >= 3 or len(types) >= 3:
        return PATTERN_MULTI
    if not types:
        return PATTERN_MULTI if n > 1 else PATTERN_NONE
    if len(types) == 1:
        # Один геометрический тип, но может быть несколько слоёв того же типа.
        return types[0] if n == 1 else PATTERN_MULTI
    return "+".join(types)


def summarize_inversion_layers(
    profile_id: str,
    layers: list[InversionLayer],
    *,
    z0: float | None = None,
) -> dict[str, Any]:
    """Сводка по профилю для климатологии и сравнения с v2."""
    if z0 is None:
        if layers:
            # AGL базы нижнего слоя: если G, base_agl≈0 → z0 ≈ base_height
            z0 = layers[0].base_height_m if layers[0].position_type == "G" else layers[0].base_height_m
        else:
            z0 = 0.0

    has_g = any(ly.position_type == "G" for ly in layers)
    has_e = any(ly.position_type == "E" for ly in layers)
    has_he = any(ly.position_type == "HE" for ly in layers)
    n = len(layers)

    lowest_idx = 0 if layers else None
    strongest_idx = None
    if layers:
        strongest_idx = max(range(n), key=lambda i: layers[i].delta_t_c)

    def _agl(ly: InversionLayer, which: str) -> float:
        h = ly.base_height_m if which == "base" else ly.top_height_m
        return float(h - float(z0))

    row: dict[str, Any] = {
        "profile_id": profile_id,
        "n_inversion_layers": n,
        "has_G": has_g,
        "has_E": has_e,
        "has_HE": has_he,
        "lowest_layer_index": lowest_idx,
        "strongest_layer_index": strongest_idx,
        "lowest_base_agl_m": None if not layers else _agl(layers[0], "base"),
        "lowest_top_agl_m": None if not layers else _agl(layers[0], "top"),
        "lowest_delta_t_c": None if not layers else layers[0].delta_t_c,
        "strongest_delta_t_c": None if strongest_idx is None else layers[strongest_idx].delta_t_c,
        "strongest_depth_m": None if strongest_idx is None else layers[strongest_idx].depth_m,
        "pattern": _pattern_from_flags(has_g, has_e, has_he, n),
    }
    return row


def layers_to_dashboard_payload(
    layers: list[InversionLayer],
    *,
    z0: float,
) -> list[dict[str, Any]]:
    """Компактные слои для JSON дашборда."""
    out: list[dict[str, Any]] = []
    for i, ly in enumerate(layers):
        out.append({
            "layer_index": i,
            "position_type": ly.position_type,
            "base_height_m": round(ly.base_height_m, 1),
            "top_height_m": round(ly.top_height_m, 1),
            "base_height_agl_m": round(ly.base_height_m - z0, 1),
            "top_height_agl_m": round(ly.top_height_m - z0, 1),
            "base_pressure_hpa": None if ly.base_pressure_hpa is None else round(ly.base_pressure_hpa, 1),
            "top_pressure_hpa": None if ly.top_pressure_hpa is None else round(ly.top_pressure_hpa, 1),
            "base_temperature_c": round(ly.base_temperature_c, 2),
            "top_temperature_c": round(ly.top_temperature_c, 2),
            "depth_m": round(ly.depth_m, 1),
            "delta_t_c": round(ly.delta_t_c, 2),
            "mean_gradient_c_100m": round(ly.mean_gradient_c_100m, 3),
            "embedded_gap_count": ly.embedded_gap_count,
            "method": ly.method,
        })
    return out


def detect_from_level_dicts(
    levels: list[dict[str, Any]],
    **kwargs: Any,
) -> list[InversionLayer]:
    """Удобная обёртка: список dict с height_m / temperature_c / pressure_hpa."""
    z: list[float] = []
    t: list[float] = []
    p: list[float] = []
    for lv in levels:
        h = lv.get("height_m")
        temp = lv.get("temperature_c")
        if h is None or temp is None:
            continue
        try:
            hf = float(h)
            tf = float(temp)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(hf) or not np.isfinite(tf):
            continue
        z.append(hf)
        t.append(tf)
        pr = lv.get("pressure_hpa")
        try:
            pf = float(pr) if pr is not None else float("nan")
        except (TypeError, ValueError):
            pf = float("nan")
        p.append(pf)
    return detect_inversion_layers_gap_v3(z, t, p, **kwargs)
