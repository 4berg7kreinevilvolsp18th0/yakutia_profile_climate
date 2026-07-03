"""Расширяемый модуль метеодиаграмм для radiosonde профилей."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gdex_bufr.meteo_parser_bridge import (
    DATA_STATUS_EMPTY,
    DATA_STATUS_NO_THERMO,
    DATA_STATUS_PARTIAL,
    RadiosondeProfile,
    VerticalLevel,
    profile_plot_types,
)
from gdex_bufr.plot_config import PlotStyle, apply_matplotlib_theme

PLOT_REGISTRY = {
    "skewt",
    "profile",
    "wind",
    "hodograph",
    "rh",
    "thermo",
    "height",
    "theta_e",
    "ttd",
    "wind_shear",
    "composite",
    "map",
}


def _ref_color(style: PlotStyle) -> str:
    return style.theme.colors.axis_reference


def _apply_ax_grid(ax, style: PlotStyle, *, show: bool = True, alpha: float = 0.45) -> None:
    if show:
        ax.grid(True, alpha=alpha, color=style.theme.grid_color, linestyle="-", linewidth=0.6)


def _line_plot_kwargs(
    cfg,
    *,
    color: str | None = None,
    linewidth: float | None = None,
) -> dict[str, object]:
    """Маркер на каждой точке профиля (markevery=1)."""
    kwargs: dict[str, object] = {
        "marker": getattr(cfg, "marker", "o"),
        "markevery": 1,
        "linewidth": linewidth if linewidth is not None else getattr(cfg, "linewidth", 1.5),
    }
    markersize = getattr(cfg, "markersize", None)
    if markersize is not None:
        kwargs["markersize"] = markersize
    if color is not None:
        kwargs["color"] = color
    return kwargs


def _profile_footer(profile: RadiosondeProfile, style: PlotStyle) -> str:
    parts: list[str] = []
    if profile.station_id:
        parts.append(f"WMO {profile.station_id}")
    if profile.latitude_deg is not None and profile.longitude_deg is not None:
        parts.append(f"{profile.latitude_deg:.2f}°N, {profile.longitude_deg:.2f}°E")
    if profile.report_datetime_utc:
        parts.append(f"UTC {profile.report_datetime_utc}")
    table = profile.metadata.get("table_edition")
    if table is not None:
        parts.append(f"BUFR table {table}")
    return "  ·  ".join(parts)


def _add_profile_footer(fig, profile: RadiosondeProfile, style: PlotStyle) -> None:
    footer = _profile_footer(profile, style)
    if not footer:
        return
    fig.text(
        0.5,
        0.01,
        footer,
        ha="center",
        va="bottom",
        fontsize=9,
        color=style.theme.text_color,
        alpha=0.85,
    )


def _finalize_profile_figure(fig, profile: RadiosondeProfile, style: PlotStyle) -> None:
    fig.subplots_adjust(bottom=0.08)
    _add_profile_footer(fig, profile, style)


def _valid_levels(levels: Iterable[VerticalLevel]) -> list[VerticalLevel]:
    return [
        lv
        for lv in levels
        if lv.pressure_hpa is not None
        and (lv.air_temperature_c is not None or lv.dew_point_temperature_c is not None)
    ]


def _wind_components(levels: Iterable[VerticalLevel]) -> tuple[np.ndarray, np.ndarray]:
    u_vals: list[float] = []
    v_vals: list[float] = []
    for lv in levels:
        if lv.wind_speed is None or lv.wind_direction_deg is None:
            u_vals.append(np.nan)
            v_vals.append(np.nan)
            continue
        rad = np.deg2rad(270 - lv.wind_direction_deg)
        u_vals.append(lv.wind_speed * np.sin(rad))
        v_vals.append(lv.wind_speed * np.cos(rad))
    return np.array(u_vals), np.array(v_vals)


def _status_label(profile: RadiosondeProfile) -> str:
    labels = {
        DATA_STATUS_EMPTY: "Нет данных",
        DATA_STATUS_NO_THERMO: "Только ветер (без T/Td)",
        DATA_STATUS_PARTIAL: "Неполный профиль",
    }
    base = labels.get(profile.data_status, profile.data_status)
    reason = profile.data_status_reason
    if reason:
        return f"{base}\n({reason})"
    return base


def plot_unavailable(
    profile: RadiosondeProfile,
    output_path: Path,
    style: PlotStyle,
    *,
    plot_type: str,
    reason: str | None = None,
) -> Path:
    apply_matplotlib_theme(style.theme)
    fig, ax = plt.subplots(figsize=style.profile.figsize)
    ax.axis("off")
    title = style.format_title("{station_id} {datetime}", profile)
    font_size = style.thermo.font_size
    ax.set_title(title, fontsize=font_size + 2)
    message = reason or _status_label(profile)
    ax.text(
        0.5,
        0.5,
        f"{plot_type}\n{message}",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=font_size,
        color=style.theme.text_color,
        bbox={"boxstyle": "round", "facecolor": style.theme.axes_facecolor, "alpha": 0.9},
    )
    _add_profile_footer(fig, profile, style)
    return _save_figure(fig, output_path, style)


def _save_figure(fig, output_path: Path, style: PlotStyle) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=style.dpi, bbox_inches="tight", format=style.export_format)
    plt.close(fig)
    return output_path


def _profile_stem(profile: RadiosondeProfile) -> str:
    station = profile.station_id or f"subset{profile.subset_index}"
    dt = (profile.report_datetime_utc or "unknown").replace(":", "").replace("T", "_")
    return f"{station}_{dt}"


def _pressure_tick_values(pressures: list[float], step: float = 50.0) -> np.ndarray:
    if not pressures:
        return np.arange(1000.0, 99.0, -step)
    p_min = min(pressures)
    p_max = max(pressures)
    top = float(np.ceil(p_max / step) * step)
    bottom = float(np.floor(p_min / step) * step)
    return np.arange(top, bottom - step * 0.01, -step)


def _apply_pressure_axis_hpa(
    ax,
    pressures: list[float],
    style: PlotStyle,
    *,
    show_grid: bool = True,
    grid_alpha: float | None = None,
) -> None:
    """Линейная ось давления в гПа с горизонтальными линиями через step гПа."""
    step = style.pressure_tick_step_hpa
    alpha = style.profile.grid_alpha if grid_alpha is None else grid_alpha
    ticks = _pressure_tick_values(pressures, step)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{int(t)}" for t in ticks])
    pad = step * 0.15
    ax.set_ylim(ticks[-1] - pad, ticks[0] + pad)
    ax.invert_yaxis()
    ax.set_ylabel("Давление, гПа")
    ax.set_axisbelow(True)
    grid_color = style.theme.grid_color
    if show_grid:
        ax.xaxis.grid(True, linestyle="--", alpha=alpha, color=grid_color)
        ax.yaxis.grid(True, linestyle="-", alpha=alpha, color=grid_color)


def plot_skewt(profile: RadiosondeProfile, output_path: Path, style: PlotStyle) -> Path:
    import metpy.calc as mpcalc
    from metpy.plots import SkewT
    from metpy.units import units

    apply_matplotlib_theme(style.theme)
    cfg = style.skewt
    levels = _valid_levels(profile.levels)
    if len(levels) < 2:
        raise ValueError("Not enough levels for Skew-T plot")

    p = np.array([lv.pressure_hpa for lv in levels]) * units.hPa
    t = np.array([lv.air_temperature_c for lv in levels]) * units.degC
    td = np.array([
        lv.dew_point_temperature_c if lv.dew_point_temperature_c is not None else np.nan
        for lv in levels
    ]) * units.degC

    fig = plt.figure(figsize=cfg.figsize)
    skew = SkewT(fig, rotation=cfg.rotation)
    skew.plot(
        p,
        t,
        cfg.color_temperature,
        label="Температура",
        markevery=1,
        **_line_plot_kwargs(cfg, linewidth=cfg.linewidth),
    )
    skew.plot(
        p,
        td,
        cfg.color_dewpoint,
        label="Точка росы",
        markevery=1,
        **_line_plot_kwargs(cfg, linewidth=cfg.linewidth),
    )
    if cfg.show_dry_adiabats:
        skew.plot_dry_adiabats()
    if cfg.show_moist_adiabats:
        skew.plot_moist_adiabats()
    if cfg.show_mixing_lines:
        skew.plot_mixing_lines()

    if cfg.show_barbs:
        u, v = _wind_components(levels)
        u = u * units("m/s")
        v = v * units("m/s")
        if np.isfinite(u.m).any() and np.isfinite(v.m).any():
            skip = max(1, cfg.barb_skip)
            skew.plot_barbs(
                p[::skip],
                u[::skip],
                v[::skip],
                barbcolor=cfg.color_barbs,
                flagcolor=cfg.color_barbs,
            )

    skew.ax.set_ylim(cfg.pressure_max_hpa, cfg.pressure_min_hpa)
    skew.ax.set_xlim(cfg.temp_min_c, cfg.temp_max_c)
    skew.ax.set_title(style.format_title(cfg.title_template, profile))
    skew.ax.legend(loc="upper right", framealpha=0.92)
    fig.subplots_adjust(bottom=0.08)
    _add_profile_footer(fig, profile, style)
    return _save_figure(fig, output_path, style)


def plot_vertical_profile(profile: RadiosondeProfile, output_path: Path, style: PlotStyle) -> Path:
    apply_matplotlib_theme(style.theme)
    cfg = style.profile
    levels = _valid_levels(profile.levels)
    pressures = [lv.pressure_hpa for lv in levels]
    temps = [lv.air_temperature_c for lv in levels]
    dewpoints = [
        lv.dew_point_temperature_c if lv.dew_point_temperature_c is not None else np.nan
        for lv in levels
    ]

    fig, ax = plt.subplots(figsize=cfg.figsize)
    ax.plot(
        temps,
        pressures,
        label="Температура",
        **_line_plot_kwargs(cfg, color=cfg.color_temperature),
    )
    ax.plot(
        dewpoints,
        pressures,
        label="Точка росы",
        **_line_plot_kwargs(cfg, color=cfg.color_dewpoint),
    )
    ax.set_xlabel("Температура, °C")
    _apply_pressure_axis_hpa(ax, pressures, style, show_grid=cfg.show_grid, grid_alpha=cfg.grid_alpha)
    ax.set_title(style.format_title(cfg.title_template, profile))
    ax.legend()
    fig.subplots_adjust(bottom=0.08)
    _add_profile_footer(fig, profile, style)
    return _save_figure(fig, output_path, style)


def plot_wind_profile(profile: RadiosondeProfile, output_path: Path, style: PlotStyle) -> Path:
    apply_matplotlib_theme(style.theme)
    cfg = style.wind
    levels = [lv for lv in profile.levels if lv.pressure_hpa is not None and lv.wind_speed is not None]
    if not levels:
        raise ValueError("No wind levels available")

    pressures = [lv.pressure_hpa for lv in levels]
    speeds = [lv.wind_speed for lv in levels]
    directions = [lv.wind_direction_deg if lv.wind_direction_deg is not None else np.nan for lv in levels]

    fig, axes = plt.subplots(1, 2, figsize=cfg.figsize, sharey=True)
    axes[0].plot(speeds, pressures, **_line_plot_kwargs(cfg, color=cfg.color_speed))
    axes[0].set_xlabel("Скорость ветра, м/с")
    _apply_pressure_axis_hpa(axes[0], pressures, style, show_grid=cfg.show_grid, grid_alpha=cfg.grid_alpha)

    axes[1].plot(directions, pressures, **_line_plot_kwargs(cfg, color=cfg.color_direction))
    axes[1].set_xlabel("Направление, °")
    if cfg.show_grid:
        axes[1].xaxis.grid(True, linestyle="--", alpha=cfg.grid_alpha, color=style.theme.grid_color)
    fig.suptitle(style.format_title(cfg.title_template, profile))
    fig.subplots_adjust(bottom=0.08)
    _add_profile_footer(fig, profile, style)
    return _save_figure(fig, output_path, style)


def plot_hodograph(profile: RadiosondeProfile, output_path: Path, style: PlotStyle) -> Path:
    apply_matplotlib_theme(style.theme)
    cfg = style.hodograph
    u_vals, v_vals = _wind_components(profile.levels)
    mask = np.isfinite(u_vals) & np.isfinite(v_vals)
    u_vals = u_vals[mask]
    v_vals = v_vals[mask]
    if len(u_vals) < 2:
        raise ValueError("Not enough wind vectors for hodograph")

    fig, ax = plt.subplots(figsize=cfg.figsize)
    ax.plot(
        u_vals,
        v_vals,
        color=cfg.color,
        marker=cfg.marker,
        markersize=cfg.markersize,
        markevery=1,
        linewidth=cfg.linewidth,
    )
    ref = _ref_color(style)
    ax.axhline(0, color=ref, linewidth=0.8)
    ax.axvline(0, color=ref, linewidth=0.8)
    ax.set_xlabel("U, м/с")
    ax.set_ylabel("V, м/с")
    ax.set_title(style.format_title(cfg.title_template, profile))
    _apply_ax_grid(ax, style, show=cfg.show_grid, alpha=cfg.grid_alpha)
    _finalize_profile_figure(fig, profile, style)
    return _save_figure(fig, output_path, style)


def plot_rh_profile(profile: RadiosondeProfile, output_path: Path, style: PlotStyle) -> Path:
    import metpy.calc as mpcalc
    from metpy.units import units

    apply_matplotlib_theme(style.theme)
    cfg = style.rh
    levels = _valid_levels(profile.levels)
    pressures: list[float] = []
    rh_vals: list[float] = []
    for lv in levels:
        if lv.air_temperature_c is None or lv.dew_point_temperature_c is None:
            continue
        t = lv.air_temperature_c * units.degC
        td = lv.dew_point_temperature_c * units.degC
        rh = mpcalc.relative_humidity_from_dewpoint(t, td).to("percent").m
        pressures.append(lv.pressure_hpa)
        rh_vals.append(float(rh))

    if len(pressures) < 2:
        raise ValueError("Not enough levels for RH plot")

    fig, ax = plt.subplots(figsize=cfg.figsize)
    ax.plot(rh_vals, pressures, **_line_plot_kwargs(cfg, color=cfg.color))
    ax.set_xlim(cfg.rh_min, cfg.rh_max)
    ax.set_xlabel("Относительная влажность, %")
    _apply_pressure_axis_hpa(ax, pressures, style, show_grid=cfg.show_grid, grid_alpha=cfg.grid_alpha)
    ax.set_title(style.format_title(cfg.title_template, profile))
    _finalize_profile_figure(fig, profile, style)
    return _save_figure(fig, output_path, style)


def plot_height_profile(profile: RadiosondeProfile, output_path: Path, style: PlotStyle) -> Path:
    apply_matplotlib_theme(style.theme)
    cfg = style.height
    levels = [lv for lv in profile.levels if lv.pressure_hpa is not None and lv.geopotential_height_m is not None]
    if len(levels) < 2:
        raise ValueError("Not enough geopotential height levels")

    pressures = [lv.pressure_hpa for lv in levels]
    heights = [lv.geopotential_height_m for lv in levels]
    fig, ax = plt.subplots(figsize=cfg.figsize)
    ax.plot(heights, pressures, **_line_plot_kwargs(cfg, color=cfg.color))
    ax.set_xlabel("Геопотенциальная высота, м")
    _apply_pressure_axis_hpa(ax, pressures, style, show_grid=cfg.show_grid, grid_alpha=cfg.grid_alpha)
    ax.set_title(style.format_title(cfg.title_template, profile))
    _finalize_profile_figure(fig, profile, style)
    return _save_figure(fig, output_path, style)


def plot_theta_e_profile(profile: RadiosondeProfile, output_path: Path, style: PlotStyle) -> Path:
    import metpy.calc as mpcalc
    from metpy.units import units

    apply_matplotlib_theme(style.theme)
    cfg = style.theta_e
    levels = _valid_levels(profile.levels)
    if len(levels) < 2:
        raise ValueError("Not enough levels for theta-e plot")

    pressures: list[float] = []
    theta_e_vals: list[float] = []
    for lv in levels:
        if lv.air_temperature_c is None or lv.dew_point_temperature_c is None:
            continue
        p = lv.pressure_hpa * units.hPa
        t = lv.air_temperature_c * units.degC
        td = lv.dew_point_temperature_c * units.degC
        theta_e = mpcalc.equivalent_potential_temperature(p, t, td).to("degC").m
        pressures.append(lv.pressure_hpa)
        theta_e_vals.append(float(theta_e))

    if len(pressures) < 2:
        raise ValueError("Not enough levels for theta-e plot")

    fig, ax = plt.subplots(figsize=cfg.figsize)
    ax.plot(theta_e_vals, pressures, **_line_plot_kwargs(cfg, color=cfg.color))
    ax.set_xlabel("Экв. потенциальная температура θe, °C")
    _apply_pressure_axis_hpa(ax, pressures, style, show_grid=cfg.show_grid, grid_alpha=cfg.grid_alpha)
    ax.set_title(style.format_title(cfg.title_template, profile))
    _finalize_profile_figure(fig, profile, style)
    return _save_figure(fig, output_path, style)


def plot_ttd_profile(profile: RadiosondeProfile, output_path: Path, style: PlotStyle) -> Path:
    apply_matplotlib_theme(style.theme)
    cfg = style.ttd
    levels = _valid_levels(profile.levels)
    pressures: list[float] = []
    spreads: list[float] = []
    for lv in levels:
        if lv.air_temperature_c is None or lv.dew_point_temperature_c is None:
            continue
        pressures.append(lv.pressure_hpa)
        spreads.append(lv.air_temperature_c - lv.dew_point_temperature_c)

    if len(pressures) < 2:
        raise ValueError("Not enough levels for T-Td plot")

    fig, ax = plt.subplots(figsize=cfg.figsize)
    ax.plot(spreads, pressures, **_line_plot_kwargs(cfg, color=cfg.color))
    ax.set_xlabel("T − Td, °C")
    _apply_pressure_axis_hpa(ax, pressures, style, show_grid=cfg.show_grid, grid_alpha=cfg.grid_alpha)
    ax.set_title(style.format_title(cfg.title_template, profile))
    _finalize_profile_figure(fig, profile, style)
    return _save_figure(fig, output_path, style)


def plot_wind_shear_profile(profile: RadiosondeProfile, output_path: Path, style: PlotStyle) -> Path:
    apply_matplotlib_theme(style.theme)
    cfg = style.wind_shear
    levels = [
        lv for lv in profile.levels
        if lv.pressure_hpa is not None and lv.wind_speed is not None and lv.wind_direction_deg is not None
    ]
    if len(levels) < 2:
        raise ValueError("Not enough wind levels for shear plot")

    u_vals, v_vals = _wind_components(levels)
    pressures = np.array([lv.pressure_hpa for lv in levels])
    shear_p: list[float] = []
    shear_vals: list[float] = []
    layer = cfg.layer_hpa
    for idx in range(len(levels) - 1):
        dp = abs(pressures[idx] - pressures[idx + 1])
        if dp < layer * 0.5:
            continue
        du = u_vals[idx] - u_vals[idx + 1]
        dv = v_vals[idx] - v_vals[idx + 1]
        shear = float(np.hypot(du, dv) / (dp / layer))
        shear_p.append(float(pressures[idx]))
        shear_vals.append(shear)

    if len(shear_p) < 2:
        raise ValueError("Could not compute wind shear layers")

    fig, ax = plt.subplots(figsize=cfg.figsize)
    ax.plot(shear_vals, shear_p, **_line_plot_kwargs(cfg, color=cfg.color))
    ax.set_xlabel(f"Сдвиг ветра, м/с на {layer:.0f} гПа")
    _apply_pressure_axis_hpa(ax, shear_p, style, show_grid=cfg.show_grid, grid_alpha=cfg.grid_alpha)
    ax.set_title(style.format_title(cfg.title_template, profile))
    _finalize_profile_figure(fig, profile, style)
    return _save_figure(fig, output_path, style)


def plot_thermo_indices(profile: RadiosondeProfile, output_path: Path, style: PlotStyle) -> Path:
    import metpy.calc as mpcalc
    from metpy.units import units

    apply_matplotlib_theme(style.theme)
    cfg = style.thermo
    levels = _valid_levels(profile.levels)
    if len(levels) < 5:
        raise ValueError("Not enough levels for thermodynamic indices")

    p = np.array([lv.pressure_hpa for lv in levels]) * units.hPa
    t = np.array([lv.air_temperature_c for lv in levels]) * units.degC
    td = np.array([
        lv.dew_point_temperature_c if lv.dew_point_temperature_c is not None else np.nan
        for lv in levels
    ]) * units.degC

    lines: list[str] = []
    if cfg.show_lcl:
        try:
            lcl_p, lcl_t = mpcalc.lcl(p[0], t[0], td[0])
            lines.append(f"Уровень конденсации (LCL): {lcl_p.to('hPa').m:.0f} гПа, {lcl_t.to('degC').m:.1f} °C")
        except Exception:
            lines.append("LCL: н/д")

    if cfg.show_cape_cin:
        try:
            cape, cin = mpcalc.cape_cin(p, t, td)
            lines.append(f"CAPE: {cape.m:.0f} Дж/кг")
            lines.append(f"CIN: {cin.m:.0f} Дж/кг")
        except Exception:
            lines.append("CAPE/CIN: н/д")

    if cfg.show_lfc_el:
        try:
            lfc_p, lfc_t = mpcalc.lfc(p, t, td)
            el_p, el_t = mpcalc.el(p, t, td)
            if lfc_p is not None:
                lines.append(f"Уровень свободной конвекции (LFC): {lfc_p.to('hPa').m:.0f} гПа")
            if el_p is not None:
                lines.append(f"Equilibrium level (EL): {el_p.to('hPa').m:.0f} гПа")
        except Exception:
            lines.append("LFC/EL: н/д")

    fig, ax = plt.subplots(figsize=cfg.figsize)
    ax.axis("off")
    ax.set_title(style.format_title(cfg.title_template, profile), fontsize=cfg.font_size + 2)
    text = "\n".join(lines) if lines else "Индексы не рассчитаны"
    ax.text(
        0.05,
        0.85,
        text,
        transform=ax.transAxes,
        fontsize=cfg.font_size,
        verticalalignment="top",
        family="monospace",
        color=cfg.text_color,
        bbox={"boxstyle": "round", "facecolor": cfg.box_facecolor, "alpha": 0.92, "edgecolor": style.theme.grid_color},
    )
    _finalize_profile_figure(fig, profile, style)
    return _save_figure(fig, output_path, style)


def plot_composite(profile: RadiosondeProfile, output_path: Path, style: PlotStyle) -> Path:
    apply_matplotlib_theme(style.theme)
    cfg = style.composite
    panels: list[tuple[str, callable]] = []
    if cfg.include_skewt:
        panels.append(("skewt", _draw_skewt_axes))
    if cfg.include_profile:
        panels.append(("profile", _draw_profile_axes))
    if cfg.include_wind:
        panels.append(("wind", _draw_wind_axes))
    if cfg.include_hodograph:
        panels.append(("hodograph", _draw_hodograph_axes))
    if cfg.include_rh:
        panels.append(("rh", _draw_rh_axes))
    if not panels:
        raise ValueError("Composite has no panels enabled")

    n = len(panels)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=cfg.figsize)
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, (_, draw_fn) in zip(axes_flat, panels, strict=False):
        draw_fn(profile, ax, style)
    for ax in axes_flat[len(panels) :]:
        ax.axis("off")
    fig.suptitle(style.format_title("Сводка {station_id} {datetime}", profile))
    fig.tight_layout()
    return _save_figure(fig, output_path, style)


def _draw_rh_axes(profile: RadiosondeProfile, ax, style: PlotStyle) -> None:
    import metpy.calc as mpcalc
    from metpy.units import units

    cfg = style.rh
    levels = _valid_levels(profile.levels)
    pressures: list[float] = []
    rh_vals: list[float] = []
    for lv in levels:
        if lv.air_temperature_c is None or lv.dew_point_temperature_c is None:
            continue
        rh = mpcalc.relative_humidity_from_dewpoint(
            lv.air_temperature_c * units.degC,
            lv.dew_point_temperature_c * units.degC,
        ).to("percent").m
        pressures.append(lv.pressure_hpa)
        rh_vals.append(float(rh))
    if len(pressures) < 2:
        ax.text(0.5, 0.5, "No RH", ha="center", va="center", transform=ax.transAxes)
        return
    ax.plot(rh_vals, pressures, **_line_plot_kwargs(cfg, color=cfg.color))
    ax.set_xlim(cfg.rh_min, cfg.rh_max)
    ax.set_xlabel("RH (%)")
    _apply_pressure_axis_hpa(ax, pressures, style, show_grid=cfg.show_grid, grid_alpha=cfg.grid_alpha)
    ax.set_title("Влажность")


def _draw_skewt_axes(profile: RadiosondeProfile, ax, style: PlotStyle) -> None:
    _draw_profile_axes(profile, ax, style, title="Skew-T T/Td")


def _draw_profile_axes(profile: RadiosondeProfile, ax, style: PlotStyle, *, title: str = "Профиль T/Td") -> None:
    cfg = style.profile
    levels = _valid_levels(profile.levels)
    if len(levels) < 2:
        ax.text(0.5, 0.5, "No profile", ha="center", va="center", transform=ax.transAxes)
        return
    pressures = [lv.pressure_hpa for lv in levels]
    temps = [lv.air_temperature_c for lv in levels]
    dewpoints = [
        lv.dew_point_temperature_c if lv.dew_point_temperature_c is not None else np.nan
        for lv in levels
    ]
    ax.plot(
        temps,
        pressures,
        label="T",
        **_line_plot_kwargs(cfg, color=cfg.color_temperature),
    )
    ax.plot(
        dewpoints,
        pressures,
        label="Td",
        **_line_plot_kwargs(cfg, color=cfg.color_dewpoint),
    )
    ax.set_xlabel("T (C)")
    _apply_pressure_axis_hpa(ax, pressures, style, show_grid=cfg.show_grid, grid_alpha=cfg.grid_alpha)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)


def _draw_wind_axes(profile: RadiosondeProfile, ax, style: PlotStyle) -> None:
    cfg = style.wind
    levels = [lv for lv in profile.levels if lv.pressure_hpa is not None and lv.wind_speed is not None]
    if not levels:
        ax.text(0.5, 0.5, "No wind", ha="center", va="center", transform=ax.transAxes)
        return
    pressures = [lv.pressure_hpa for lv in levels]
    speeds = [lv.wind_speed for lv in levels]
    ax.plot(speeds, pressures, **_line_plot_kwargs(cfg, color=cfg.color_speed))
    ax.set_xlabel("Скорость")
    _apply_pressure_axis_hpa(ax, pressures, style, show_grid=cfg.show_grid, grid_alpha=cfg.grid_alpha)
    ax.set_title("Ветер")


def _draw_hodograph_axes(profile: RadiosondeProfile, ax, style: PlotStyle) -> None:
    cfg = style.hodograph
    u_vals, v_vals = _wind_components(profile.levels)
    mask = np.isfinite(u_vals) & np.isfinite(v_vals)
    if mask.sum() < 2:
        ax.text(0.5, 0.5, "No wind", ha="center", va="center", transform=ax.transAxes)
        return
    ax.plot(
        u_vals[mask],
        v_vals[mask],
        color=cfg.color,
        marker=cfg.marker,
        markersize=cfg.markersize,
        markevery=1,
    )
    ref = _ref_color(style)
    ax.axhline(0, color=ref, linewidth=0.5)
    ax.axvline(0, color=ref, linewidth=0.5)
    ax.set_title("Годограф")
    _apply_ax_grid(ax, style, show=cfg.show_grid, alpha=cfg.grid_alpha)


def plot_station_map(profiles: list[RadiosondeProfile], output_path: Path, style: PlotStyle) -> Path:
    apply_matplotlib_theme(style.theme)
    cfg = style.map
    lats = [p.latitude_deg for p in profiles if p.latitude_deg is not None]
    lons = [p.longitude_deg for p in profiles if p.longitude_deg is not None]
    if not lats:
        raise ValueError("No station coordinates for map plot")

    fig, ax = plt.subplots(figsize=cfg.figsize)
    ax.scatter(lons, lats, s=cfg.marker_size, alpha=cfg.marker_alpha, c=cfg.color)
    ax.set_xlabel("Долгота, °")
    ax.set_ylabel("Широта, °")
    ax.set_title(cfg.title_template.format(count=len(lats), station_id="", datetime=""))
    _apply_ax_grid(ax, style, show=cfg.show_grid, alpha=cfg.grid_alpha)
    return _save_figure(fig, output_path, style)


_PLOT_HANDLERS = {
    "skewt": plot_skewt,
    "profile": plot_vertical_profile,
    "wind": plot_wind_profile,
    "hodograph": plot_hodograph,
    "rh": plot_rh_profile,
    "thermo": plot_thermo_indices,
    "height": plot_height_profile,
    "theta_e": plot_theta_e_profile,
    "ttd": plot_ttd_profile,
    "wind_shear": plot_wind_shear_profile,
    "composite": plot_composite,
}


def output_dir_for_file(
    base_dir: Path,
    bufr_path: Path,
    style: PlotStyle,
    *,
    cycle: str | None = None,
) -> Path:
    """Строю каталог вывода по дате/циклу из имени BUFR."""
    out = base_dir
    name = bufr_path.stem
    parts = name.split(".")
    obs_date = parts[-1] if parts else ""
    if style.output_subdir_by_date and len(obs_date) == 8:
        out = out / obs_date[:4] / obs_date
    if style.output_subdir_by_cycle and cycle:
        out = out / f"t{cycle}z"
    return out


def render_plots(
    profile: RadiosondeProfile,
    outputs_dir: Path,
    plot_types: Iterable[str],
    *,
    all_profiles: list[RadiosondeProfile] | None = None,
    style: PlotStyle | None = None,
    min_levels: int = 2,
    write_unavailable: bool = True,
) -> list[Path]:
    style = style or PlotStyle()
    apply_matplotlib_theme(style.theme)
    outputs: list[Path] = []
    stem = _profile_stem(profile)
    suffix = style.output_suffix()
    requested = list(plot_types)
    allowed = set(profile_plot_types(profile, requested, min_levels=min_levels))

    for plot_type in requested:
        if plot_type == "map":
            if all_profiles:
                map_profiles = [p for p in all_profiles if p.latitude_deg is not None and p.longitude_deg is not None]
                if map_profiles:
                    try:
                        outputs.append(
                            plot_station_map(map_profiles, outputs_dir / f"stations_map{suffix}", style)
                        )
                    except ValueError as exc:
                        logger.warning("Skip map plot: %s", exc)
            continue

        output_path = outputs_dir / f"{stem}_{plot_type}{suffix}"
        if plot_type not in allowed:
            if write_unavailable and profile.data_status in {DATA_STATUS_EMPTY, DATA_STATUS_NO_THERMO, DATA_STATUS_PARTIAL}:
                try:
                    outputs.append(
                        plot_unavailable(profile, output_path, style, plot_type=plot_type)
                    )
                except Exception as exc:
                    logger.warning("Skip unavailable marker %s for %s: %s", plot_type, stem, exc)
            continue

        handler = _PLOT_HANDLERS.get(plot_type)
        if handler is None:
            continue
        try:
            outputs.append(handler(profile, output_path, style))
        except ValueError as exc:
            logger.warning("Skip plot %s for %s: %s", plot_type, stem, exc)
            if write_unavailable:
                try:
                    outputs.append(
                        plot_unavailable(profile, output_path, style, plot_type=plot_type, reason=str(exc))
                    )
                except Exception as marker_exc:
                    logger.warning("Skip unavailable marker %s for %s: %s", plot_type, stem, marker_exc)
        except Exception as exc:
            logger.warning("Plot failed %s for %s: %s", plot_type, stem, exc)
    return outputs
