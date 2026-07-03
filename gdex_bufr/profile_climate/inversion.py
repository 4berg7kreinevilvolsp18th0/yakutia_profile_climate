"""Поиск приземной температурной инверсии в вертикальном профиле."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class InversionResult:
    inversion_detected: bool = False
    inversion_top_pressure_hpa: float | None = None
    inversion_top_height_m: float | None = None
    inversion_top_temp_c: float | None = None
    inversion_delta_t_c: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "inversion_detected": self.inversion_detected,
            "inversion_top_pressure_hpa": self.inversion_top_pressure_hpa,
            "inversion_top_height_m": self.inversion_top_height_m,
            "inversion_top_temp_c": self.inversion_top_temp_c,
            "inversion_delta_t_c": self.inversion_delta_t_c,
        }


def detect_surface_inversion(
    levels: list[dict[str, Any]],
    *,
    min_inversion_delta_c: float = 0.2,
) -> InversionResult:
    """Находит верхнюю границу приземной инверсии по росту T при движении вверх."""
    if len(levels) < 2:
        return InversionResult()

    surface = levels[0]
    surface_temp = surface.get("temperature_c")
    if surface_temp is None:
        return InversionResult()

    inversion_top = surface
    growing = False

    for level in levels[1:]:
        temp = level.get("temperature_c")
        if temp is None:
            break
        prev_temp = inversion_top.get("temperature_c")
        if prev_temp is None:
            break
        delta = temp - prev_temp
        if delta > min_inversion_delta_c:
            growing = True
            inversion_top = level
        elif growing:
            break
        else:
            break

    if not growing:
        return InversionResult()

    top_temp = inversion_top.get("temperature_c")
    if top_temp is None:
        return InversionResult()

    return InversionResult(
        inversion_detected=True,
        inversion_top_pressure_hpa=inversion_top.get("pressure_hpa"),
        inversion_top_height_m=inversion_top.get("height_m"),
        inversion_top_temp_c=top_temp,
        inversion_delta_t_c=top_temp - surface_temp,
    )
