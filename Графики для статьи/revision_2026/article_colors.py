"""Единая палитра для всех графиков статьи (ревизия 2026)."""

from __future__ import annotations

ARTICLE_COLORS: dict[str, str] = {
    "00": "#2471A3",
    "12": "#C0392B",
    "G": "#B03A2E",
    "E": "#2471A3",
    "HE": "#6C3483",
}

CYCLE_MARKERS: dict[str, str] = {
    "00": "o",
    "12": "x",
}

LEVEL_COLORS: dict[float, str] = {
    850.0: "#1A5276",
    700.0: "#B9770E",
    500.0: "#196F3D",
}

REFERENCE_GAMMA_STANDARD = -0.6  # °C/100 m, не сухоадиабат
REFERENCE_GAMMA_DRY_ADIABATIC = -0.98  # °C/100 m
