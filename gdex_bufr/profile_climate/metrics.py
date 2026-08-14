"""Метрики температурного профиля до заданного давления."""
from __future__ import annotations

import json
from typing import Any

from gdex_bufr.profile_climate.inversion import (
    detect_inversions_from_top,
    detect_surface_inversion,
    inversions_from_top_as_metrics,
)


PROFILE_STATUS_GOOD = "good"
PROFILE_STATUS_SHORT = "short"
PROFILE_STATUS_NO_500 = "no_500"
PROFILE_STATUS_NO_TEMP = "no_temp"
PROFILE_STATUS_BAD_PRESSURE = "bad_pressure"
PROFILE_STATUS_DUPLICATE_LEVELS = "duplicate_levels"
PROFILE_STATUS_NO_SURFACE = "no_surface_level"

# Пустые поля инверсии — когда профиль бракованный или слишком короткий.
_EMPTY_INVERSION = {
    "inversion_detected": False,
    "inversion_candidate": False,
    "inversion_quality": "none",
    "inversion_top_pressure_hpa": None,
    "inversion_top_height_m": None,
    "inversion_top_temp_c": None,
    "inversion_delta_t_c": None,
    "inversion_confirm_drop_c": None,
    "inversion_from_top_count": 0,
    "inversion_from_top_tops": "[]",
}


def _levels_to_500(levels: list[dict[str, Any]], pressure_top_hpa: float) -> list[dict[str, Any]]:
    """Оставляет уровни от земли до верхней границы (включительно)."""
    return [
        lv for lv in levels
        if lv.get("pressure_hpa") is not None and lv["pressure_hpa"] >= pressure_top_hpa
    ]


def _nearest_top_level(levels: list[dict[str, Any]], pressure_top_hpa: float) -> dict[str, Any] | None:
    """Уровень ближе всего к верхней границе (например 500 гПа)."""
    candidates = [
        lv for lv in levels
        if lv.get("pressure_hpa") is not None and lv["pressure_hpa"] <= pressure_top_hpa
    ]
    if not candidates:
        return levels[-1] if levels else None
    return min(candidates, key=lambda lv: abs(lv["pressure_hpa"] - pressure_top_hpa))


def _delta_t(top: dict[str, Any], surface: dict[str, Any]) -> float | None:
    """Разница температур верх − поверхность, если обе есть."""
    t_top = top.get("temperature_c")
    t_sfc = surface.get("temperature_c")
    if t_top is None or t_sfc is None:
        return None
    return t_top - t_sfc


def _base_metrics(
    *,
    total: int,
    n_to_500: int,
    surface: dict[str, Any] | None,
    top: dict[str, Any] | None,
    status: str,
    inversion: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Собирает общий словарь метрик (и для хороших, и для частичных профилей)."""
    return {
        "n_levels_total": total,
        "n_levels_to_500": n_to_500,
        "p_surface_hpa": None if surface is None else surface.get("pressure_hpa"),
        "t_surface_c": None if surface is None else surface.get("temperature_c"),
        "p_top_hpa": None if top is None else top.get("pressure_hpa"),
        "t_top_c": None if top is None else top.get("temperature_c"),
        "delta_t_top_surface_c": (
            None if surface is None or top is None else _delta_t(top, surface)
        ),
        "profile_status": status,
        **(inversion if inversion is not None else _EMPTY_INVERSION),
    }


def _empty_metrics(n_levels_total: int, status: str) -> dict[str, Any]:
    return _base_metrics(
        total=n_levels_total,
        n_to_500=0,
        surface=None,
        top=None,
        status=status,
    )


def _find_qc_status(levels: list[dict[str, Any]]) -> str | None:
    """Быстрые проверки брака. None = профиль можно считать дальше."""
    pressures = [lv.get("pressure_hpa") for lv in levels]
    if any(p is None or p <= 0 for p in pressures):
        return PROFILE_STATUS_BAD_PRESSURE

    valid_p = [p for p in pressures if p is not None]
    if len(set(round(p, 2) for p in valid_p)) != len(valid_p):
        return PROFILE_STATUS_DUPLICATE_LEVELS

    if any(lv.get("temperature_c") is None for lv in levels):
        return PROFILE_STATUS_NO_TEMP

    return None


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

    bad = _find_qc_status(levels)
    if bad is not None:
        return _empty_metrics(total, bad)

    # От земли вверх: большее давление → меньшее.
    sorted_levels = sorted(levels, key=lambda lv: lv["pressure_hpa"], reverse=True)
    surface = sorted_levels[0]
    if surface.get("temperature_c") is None or surface.get("pressure_hpa") is None:
        return _empty_metrics(total, PROFILE_STATUS_NO_SURFACE)

    trimmed = _levels_to_500(sorted_levels, pressure_top_hpa)
    if not trimmed:
        return _empty_metrics(total, PROFILE_STATUS_NO_500)

    inversion = detect_surface_inversion(
        trimmed,
        min_inversion_delta_c=min_inversion_delta_c,
        confirm_drop_levels=confirm_drop_levels,
        confirm_depth_hpa=confirm_depth_hpa,
        min_drop_delta_c=min_drop_delta_c,
    ).as_dict()
    from_top = inversions_from_top_as_metrics(
        detect_inversions_from_top(
            trimmed,
            min_inversion_delta_c=min_inversion_delta_c,
            confirm_drop_levels=confirm_drop_levels,
            confirm_depth_hpa=confirm_depth_hpa,
            min_drop_delta_c=min_drop_delta_c,
        )
    )
    # CSV-friendly: список вершин как JSON-строка; в JSON daily — разберём обратно.
    inversion["inversion_from_top_count"] = from_top["inversion_from_top_count"]
    inversion["inversion_from_top_tops"] = json.dumps(
        from_top["inversion_from_top_tops"], ensure_ascii=False
    )

    # Профиль не дотягивает до верхней границы.
    min_pressure = min(lv["pressure_hpa"] for lv in trimmed)
    if min_pressure > pressure_top_hpa:
        return _base_metrics(
            total=total,
            n_to_500=len(trimmed),
            surface=surface,
            top=trimmed[-1],
            status=PROFILE_STATUS_NO_500,
            inversion=inversion,
        )

    top_level = _nearest_top_level(trimmed, pressure_top_hpa) or trimmed[-1]
    n_to_500 = len(trimmed)
    status = PROFILE_STATUS_SHORT if n_to_500 < min_levels_to_500 else PROFILE_STATUS_GOOD

    return _base_metrics(
        total=total,
        n_to_500=n_to_500,
        surface=surface,
        top=top_level,
        status=status,
        inversion=inversion,
    )
