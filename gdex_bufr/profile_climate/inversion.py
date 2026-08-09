"""Поиск приземной температурной инверсии в вертикальном профиле (v2)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

QUALITY_NONE = "none"
QUALITY_CONFIRMED = "confirmed"
QUALITY_REJECTED_NO_LAPSE = "rejected_no_lapse"


@dataclass
class InversionResult:
    inversion_detected: bool = False
    inversion_candidate: bool = False
    inversion_quality: str = QUALITY_NONE
    inversion_top_pressure_hpa: float | None = None
    inversion_top_height_m: float | None = None
    inversion_top_temp_c: float | None = None
    inversion_delta_t_c: float | None = None
    inversion_confirm_drop_c: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "inversion_detected": self.inversion_detected,
            "inversion_candidate": self.inversion_candidate,
            "inversion_quality": self.inversion_quality,
            "inversion_top_pressure_hpa": self.inversion_top_pressure_hpa,
            "inversion_top_height_m": self.inversion_top_height_m,
            "inversion_top_temp_c": self.inversion_top_temp_c,
            "inversion_delta_t_c": self.inversion_delta_t_c,
            "inversion_confirm_drop_c": self.inversion_confirm_drop_c,
        }


def _candidate_result(
    *,
    surface_temp: float,
    inversion_top: dict[str, Any],
    quality: str,
    confirm_drop_c: float | None,
) -> InversionResult:
    top_temp = inversion_top.get("temperature_c")
    if top_temp is None:
        return InversionResult()
    confirmed = quality == QUALITY_CONFIRMED
    return InversionResult(
        inversion_detected=confirmed,
        inversion_candidate=True,
        inversion_quality=quality,
        inversion_top_pressure_hpa=inversion_top.get("pressure_hpa"),
        inversion_top_height_m=inversion_top.get("height_m"),
        inversion_top_temp_c=top_temp,
        inversion_delta_t_c=top_temp - surface_temp,
        inversion_confirm_drop_c=confirm_drop_c,
    )


def _has_consecutive_cooling(
    levels_above: list[dict[str, Any]],
    *,
    top_temp: float,
    confirm_drop_levels: int,
    min_drop_delta_c: float,
) -> bool:
    """Есть ли подряд несколько шагов охлаждения выше верха инверсии."""
    if len(levels_above) < confirm_drop_levels:
        return False
    prev_temp = top_temp
    for level in levels_above[:confirm_drop_levels]:
        temp = level.get("temperature_c")
        if temp is None:
            return False
        # Падение должно быть не слабее порога.
        if (temp - prev_temp) > -min_drop_delta_c:
            return False
        prev_temp = temp
    return True


def _pressure_window_drop(
    levels_above: list[dict[str, Any]],
    *,
    top_temp: float,
    top_pressure_hpa: float,
    confirm_depth_hpa: float,
) -> float | None:
    """Суммарное изменение T в слое толщиной ≥ confirm_depth_hpa.

    Возвращает None, если окно слишком короткое.
    """
    window: list[dict[str, Any]] = []
    for level in levels_above:
        p = level.get("pressure_hpa")
        if p is None or level.get("temperature_c") is None:
            break
        window.append(level)
        if top_pressure_hpa - float(p) >= confirm_depth_hpa:
            break

    if not window:
        return None
    last_p = window[-1].get("pressure_hpa")
    if last_p is None or top_pressure_hpa - float(last_p) < confirm_depth_hpa:
        return None
    return float(window[-1]["temperature_c"]) - float(top_temp)


def _confirm_sustained_lapse(
    levels_above: list[dict[str, Any]],
    *,
    top_temp: float,
    top_pressure_hpa: float | None,
    confirm_drop_levels: int,
    confirm_depth_hpa: float,
    min_drop_delta_c: float,
) -> tuple[bool, float | None]:
    """Проверка устойчивого падения T выше верха инверсии.

    Возвращает (ok, суммарное падение T в окне подтверждения).
    """
    if not _has_consecutive_cooling(
        levels_above,
        top_temp=top_temp,
        confirm_drop_levels=confirm_drop_levels,
        min_drop_delta_c=min_drop_delta_c,
    ):
        return False, None

    if top_pressure_hpa is None:
        return False, None

    confirm_drop = _pressure_window_drop(
        levels_above,
        top_temp=top_temp,
        top_pressure_hpa=float(top_pressure_hpa),
        confirm_depth_hpa=confirm_depth_hpa,
    )
    if confirm_drop is None:
        return False, None
    if confirm_drop >= 0:
        return False, confirm_drop
    return True, confirm_drop


def _find_inversion_top(
    levels: list[dict[str, Any]],
    *,
    min_inversion_delta_c: float,
) -> tuple[dict[str, Any], int] | None:
    """Ищет верх инверсии: непрерывный рост T от поверхности вверх."""
    surface = levels[0]
    if surface.get("temperature_c") is None:
        return None

    inversion_top = surface
    top_index = 0
    growing = False

    for idx, level in enumerate(levels[1:], start=1):
        temp = level.get("temperature_c")
        prev_temp = inversion_top.get("temperature_c")
        if temp is None or prev_temp is None:
            break
        if temp - prev_temp > min_inversion_delta_c:
            growing = True
            inversion_top = level
            top_index = idx
        else:
            # Рост закончился (или так и не начался).
            break

    if not growing:
        return None
    if inversion_top.get("temperature_c") is None:
        return None
    return inversion_top, top_index


def detect_surface_inversion(
    levels: list[dict[str, Any]],
    *,
    min_inversion_delta_c: float = 0.2,
    confirm_drop_levels: int = 2,
    confirm_depth_hpa: float = 30.0,
    min_drop_delta_c: float = 0.2,
) -> InversionResult:
    """Приземная инверсия: рост T от земли + устойчивое падение выше верха.

    Этап 1 — кандидат: непрерывный рост T с порогом min_inversion_delta_c.
    Этап 2 — подтверждение: confirm_drop_levels шагов падения и слой
    толщиной ≥ confirm_depth_hpa с суммарным падением T.
    inversion_detected=True только при quality=confirmed.
    """
    if len(levels) < 2:
        return InversionResult()

    surface_temp = levels[0].get("temperature_c")
    if surface_temp is None:
        return InversionResult()

    found = _find_inversion_top(levels, min_inversion_delta_c=min_inversion_delta_c)
    if found is None:
        return InversionResult()

    inversion_top, top_index = found
    ok, confirm_drop = _confirm_sustained_lapse(
        levels[top_index + 1 :],
        top_temp=float(inversion_top["temperature_c"]),
        top_pressure_hpa=inversion_top.get("pressure_hpa"),
        confirm_drop_levels=confirm_drop_levels,
        confirm_depth_hpa=confirm_depth_hpa,
        min_drop_delta_c=min_drop_delta_c,
    )
    quality = QUALITY_CONFIRMED if ok else QUALITY_REJECTED_NO_LAPSE
    return _candidate_result(
        surface_temp=float(surface_temp),
        inversion_top=inversion_top,
        quality=quality,
        confirm_drop_c=confirm_drop,
    )
