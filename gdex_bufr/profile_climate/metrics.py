"""Метрики температурного профиля до заданного давления."""
from __future__ import annotations

from typing import Any

from gdex_bufr.profile_climate.inversion import detect_surface_inversion


PROFILE_STATUS_GOOD = "good"
PROFILE_STATUS_SHORT = "short"
PROFILE_STATUS_NO_500 = "no_500"
PROFILE_STATUS_NO_TEMP = "no_temp"
PROFILE_STATUS_BAD_PRESSURE = "bad_pressure"
PROFILE_STATUS_DUPLICATE_LEVELS = "duplicate_levels"
PROFILE_STATUS_NO_SURFACE = "no_surface_level"


def _levels_to_500(levels: list[dict[str, Any]], pressure_top_hpa: float) -> list[dict[str, Any]]:
    return [lv for lv in levels if lv.get("pressure_hpa") is not None and lv["pressure_hpa"] >= pressure_top_hpa]


def _nearest_top_level(levels: list[dict[str, Any]], pressure_top_hpa: float) -> dict[str, Any] | None:
    candidates = [lv for lv in levels if lv.get("pressure_hpa") is not None and lv["pressure_hpa"] <= pressure_top_hpa]
    if not candidates:
        return levels[-1] if levels else None
    return min(candidates, key=lambda lv: abs(lv["pressure_hpa"] - pressure_top_hpa))


def compute_profile_metrics(
    levels: list[dict[str, Any]],
    *,
    pressure_top_hpa: float = 500.0,
    min_levels_to_500: int = 5,
    min_inversion_delta_c: float = 0.2,
    confirm_drop_levels: int = 2,
    confirm_depth_hpa: float = 30.0,
    min_drop_delta_c: float = 0.2,
    n_levels_total: int | None = None,
) -> dict[str, Any]:
    """Считает метрики профиля и статус пригодности."""
    total = n_levels_total if n_levels_total is not None else len(levels)

    if not levels:
        return _empty_metrics(total, PROFILE_STATUS_NO_SURFACE)

    pressures = [lv.get("pressure_hpa") for lv in levels]
    if any(p is None or p <= 0 for p in pressures):
        return _empty_metrics(total, PROFILE_STATUS_BAD_PRESSURE)

    if len(set(round(p, 2) for p in pressures if p is not None)) != len([p for p in pressures if p is not None]):
        return _empty_metrics(total, PROFILE_STATUS_DUPLICATE_LEVELS)

    if any(lv.get("temperature_c") is None for lv in levels):
        return _empty_metrics(total, PROFILE_STATUS_NO_TEMP)

    sorted_levels = sorted(levels, key=lambda lv: lv["pressure_hpa"], reverse=True)
    surface = sorted_levels[0]
    if surface.get("temperature_c") is None or surface.get("pressure_hpa") is None:
        return _empty_metrics(total, PROFILE_STATUS_NO_SURFACE)

    trimmed = _levels_to_500(sorted_levels, pressure_top_hpa)
    if not trimmed:
        return _empty_metrics(total, PROFILE_STATUS_NO_500)

    min_pressure = min(lv["pressure_hpa"] for lv in trimmed)
    if min_pressure > pressure_top_hpa:
        return _fill_partial_metrics(
            total=total,
            trimmed=trimmed,
            surface=surface,
            status=PROFILE_STATUS_NO_500,
            pressure_top_hpa=pressure_top_hpa,
            min_inversion_delta_c=min_inversion_delta_c,
            confirm_drop_levels=confirm_drop_levels,
            confirm_depth_hpa=confirm_depth_hpa,
            min_drop_delta_c=min_drop_delta_c,
        )

    top_level = _nearest_top_level(trimmed, pressure_top_hpa) or trimmed[-1]
    n_to_500 = len(trimmed)

    status = PROFILE_STATUS_GOOD
    if n_to_500 < min_levels_to_500:
        status = PROFILE_STATUS_SHORT

    inversion = detect_surface_inversion(
        trimmed,
        min_inversion_delta_c=min_inversion_delta_c,
        confirm_drop_levels=confirm_drop_levels,
        confirm_depth_hpa=confirm_depth_hpa,
        min_drop_delta_c=min_drop_delta_c,
    )

    return {
        "n_levels_total": total,
        "n_levels_to_500": n_to_500,
        "p_surface_hpa": surface["pressure_hpa"],
        "t_surface_c": surface["temperature_c"],
        "p_top_hpa": top_level.get("pressure_hpa"),
        "t_top_c": top_level.get("temperature_c"),
        "delta_t_top_surface_c": (
            (top_level["temperature_c"] - surface["temperature_c"])
            if top_level.get("temperature_c") is not None
            else None
        ),
        "profile_status": status,
        **inversion.as_dict(),
    }


def _empty_metrics(n_levels_total: int, status: str) -> dict[str, Any]:
    return {
        "n_levels_total": n_levels_total,
        "n_levels_to_500": 0,
        "p_surface_hpa": None,
        "t_surface_c": None,
        "p_top_hpa": None,
        "t_top_c": None,
        "delta_t_top_surface_c": None,
        "inversion_detected": False,
        "inversion_candidate": False,
        "inversion_quality": "none",
        "inversion_top_pressure_hpa": None,
        "inversion_top_height_m": None,
        "inversion_top_temp_c": None,
        "inversion_delta_t_c": None,
        "inversion_confirm_drop_c": None,
        "profile_status": status,
    }


def _fill_partial_metrics(
    *,
    total: int,
    trimmed: list[dict[str, Any]],
    surface: dict[str, Any],
    status: str,
    pressure_top_hpa: float,
    min_inversion_delta_c: float,
    confirm_drop_levels: int = 2,
    confirm_depth_hpa: float = 30.0,
    min_drop_delta_c: float = 0.2,
) -> dict[str, Any]:
    top_level = trimmed[-1]
    inversion = detect_surface_inversion(
        trimmed,
        min_inversion_delta_c=min_inversion_delta_c,
        confirm_drop_levels=confirm_drop_levels,
        confirm_depth_hpa=confirm_depth_hpa,
        min_drop_delta_c=min_drop_delta_c,
    )
    return {
        "n_levels_total": total,
        "n_levels_to_500": len(trimmed),
        "p_surface_hpa": surface.get("pressure_hpa"),
        "t_surface_c": surface.get("temperature_c"),
        "p_top_hpa": top_level.get("pressure_hpa"),
        "t_top_c": top_level.get("temperature_c"),
        "delta_t_top_surface_c": (
            (top_level["temperature_c"] - surface["temperature_c"])
            if top_level.get("temperature_c") is not None and surface.get("temperature_c") is not None
            else None
        ),
        "profile_status": status,
        **inversion.as_dict(),
    }
