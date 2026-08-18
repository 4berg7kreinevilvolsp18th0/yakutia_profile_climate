"""Рисунки ревизии: joint scatter, γ, heatmap, multilayer, QC."""
from __future__ import annotations

from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

from gdex_bufr.profile_climate.article_figures.config import FigureStyle
from gdex_bufr.profile_climate.article_figures.metrics import INVERSION_TYPES

from .metrics import heatmap_matrix, shared_abs_limit
from .style import (
    LEVEL_COLORS,
    LEVEL_LABELS,
    MONTHS_RU,
    SEASON_ORDER,
    SEASONS_RU,
    TYPE_COLORS,
    TYPE_LABELS,
    finish,
    revision_rc,
)

STANDARD_LEVELS = (850.0, 700.0, 500.0)


def _density_colors(x: np.ndarray, y: np.ndarray, bins: int = 60) -> tuple[np.ndarray, float]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 4:
        return np.ones(x.size), 1.0
    h, xe, ye = np.histogram2d(x, y, bins=bins)
    ix = np.clip(np.searchsorted(xe, x, side="right") - 1, 0, h.shape[0] - 1)
    iy = np.clip(np.searchsorted(ye, y, side="right") - 1, 0, h.shape[1] - 1)
    counts = h[ix, iy]
    z = np.log10(np.maximum(counts, 1.0))
    return z, float(z.max() if z.size else 1.0)


def _valid_depth_base_layers(layers: pd.DataFrame) -> pd.DataFrame:
    use = layers.copy()
    x = pd.to_numeric(use["base_height_agl_m"], errors="coerce")
    y = pd.to_numeric(use["depth_m"], errors="coerce")
    mask = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y > 0)
    return use.loc[mask]


def _typed_marginal_histograms(
    ax_histx: plt.Axes,
    ax_histy: plt.Axes,
    layers: pd.DataFrame,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    n_bins: int = 40,
    alpha: float = 0.55,
) -> None:
    for kind in INVERSION_TYPES:
        g = layers[layers["position_type"] == kind]
        if g.empty:
            continue
        x = g["base_height_agl_m"].to_numpy(float)
        y = g["depth_m"].to_numpy(float)
        color = TYPE_COLORS[kind]
        ax_histx.hist(
            x,
            bins=n_bins,
            range=xlim,
            color=color,
            alpha=alpha,
            log=True,
            label=TYPE_LABELS[kind],
        )
        ax_histy.hist(
            y,
            bins=n_bins,
            range=ylim,
            orientation="horizontal",
            color=color,
            alpha=alpha,
            log=True,
        )


def plot_joint_depth_vs_base(
    layers: pd.DataFrame,
    style: FigureStyle,
    *,
    caption: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    vmax: float | None = None,
) -> plt.Figure:
    use = _valid_depth_base_layers(layers)
    x = use["base_height_agl_m"].to_numpy(float)
    y = use["depth_m"].to_numpy(float)
    if xlim is None:
        xlim = (0.0, float(np.nanpercentile(x, 99.5)) if x.size else 1.0)
    if ylim is None:
        hi = float(np.nanpercentile(y, 99.5)) if y.size else 10.0
        ylim = (0.0, max(hi, 1.0))
    med_x = float(np.median(x)) if x.size else np.nan
    med_y = float(np.median(y)) if y.size else np.nan
    q25x, q75x = (np.nanpercentile(x, [25, 75]) if x.size else (np.nan, np.nan))
    q25y, q75y = (np.nanpercentile(y, [25, 75]) if y.size else (np.nan, np.nan))
    type_lines = []
    for kind in INVERSION_TYPES:
        n_kind = int((use["position_type"] == kind).sum())
        if n_kind:
            type_lines.append(f"{kind}: {n_kind}")

    with revision_rc(style):
        fig = plt.figure(figsize=(style.figure_width_in, style.figure_height_in * 1.25))
        gs = fig.add_gridspec(2, 2, width_ratios=[4, 1.15], height_ratios=[1.15, 4], hspace=0.04, wspace=0.04)
        ax_histx = fig.add_subplot(gs[0, 0])
        ax = fig.add_subplot(gs[1, 0], sharex=ax_histx)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
        for kind in INVERSION_TYPES:
            g = use[use["position_type"] == kind]
            if g.empty:
                continue
            ax.scatter(
                g["base_height_agl_m"],
                g["depth_m"],
                c=TYPE_COLORS[kind],
                s=6,
                alpha=0.42,
                linewidths=0,
                rasterized=True,
                label=TYPE_LABELS[kind],
            )
        ax.axvline(med_x, color="#1C2833", linewidth=0.9, linestyle="--")
        ax.axhline(med_y, color="#1C2833", linewidth=0.9, linestyle="--")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Высота основания AGL, м")
        ax.set_ylabel("Толщина слоя depth_m, м")
        ax.grid(True, alpha=style.grid_alpha, linewidth=0.4)
        stats = (
            f"N = {x.size}\n"
            f"med base = {med_x:.0f} м (Q25–Q75 {q25x:.0f}–{q75x:.0f})\n"
            f"med depth = {med_y:.0f} м (Q25–Q75 {q25y:.0f}–{q75y:.0f})"
        )
        if type_lines:
            stats += "\n" + ", ".join(type_lines)
        ax.text(
            0.98,
            0.97,
            stats,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#D5D8DC", alpha=0.9),
        )
        if x.size:
            _typed_marginal_histograms(ax_histx, ax_histy, use, xlim=xlim, ylim=ylim)
        ax_histx.tick_params(labelbottom=False)
        ax_histy.tick_params(labelleft=False)
        ax_histx.set_ylabel("N")
        ax_histy.set_xlabel("N")
        ax.legend(loc="upper left", framealpha=0.92, borderpad=0.35, handletextpad=0.35)
        return finish(fig, style, caption)


def plot_monthly_depth_vs_base_12(
    layers: pd.DataFrame,
    style: FigureStyle,
    *,
    caption: str | None = None,
) -> tuple[plt.Figure, tuple[float, float], tuple[float, float], float]:
    use = _valid_depth_base_layers(layers)
    x_all = use["base_height_agl_m"].to_numpy(float)
    y_all = use["depth_m"].to_numpy(float)
    xlim = (0.0, float(np.nanpercentile(x_all, 99.5)) if x_all.size else 1000.0)
    ylim = (0.0, float(np.nanpercentile(y_all, 99.5)) if y_all.size else 1000.0)
    _, vmax = _density_colors(x_all, y_all)
    with revision_rc(style):
        fig, axes = plt.subplots(4, 3, figsize=(style.figure_width_in * 1.45, style.figure_height_in * 2.05), sharex=True, sharey=True)
        for ax, month in zip(axes.ravel(), range(1, 13)):
            g = use[use["month"] == month]
            for kind in INVERSION_TYPES:
                gt = g[g["position_type"] == kind]
                if gt.empty:
                    continue
                ax.scatter(
                    gt["base_height_agl_m"],
                    gt["depth_m"],
                    c=TYPE_COLORS[kind],
                    s=4,
                    alpha=0.42,
                    linewidths=0,
                    rasterized=True,
                    label=TYPE_LABELS[kind] if month == 1 else None,
                )
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_title(MONTHS_RU[month - 1], fontsize=style.tick_font_size)
            ax.grid(True, alpha=0.2, linewidth=0.3)
            ax.text(0.97, 0.95, f"N={len(g)}", transform=ax.transAxes, ha="right", va="top", fontsize=6.5)
        axes[3, 1].set_xlabel("Основание AGL, м")
        axes[1, 0].set_ylabel("Толщина depth_m, м")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=3, framealpha=0.92, bbox_to_anchor=(0.5, 0.995))
        fig.subplots_adjust(bottom=0.08, top=0.94, right=0.96)
        if caption:
            fig.text(0.5, 0.01, caption, ha="center", fontsize=7.5, color="#4D4D4D")
        return fig, xlim, ylim, vmax


def plot_depth_boxplots(layers: pd.DataFrame, style: FigureStyle, *, caption: str | None = None) -> plt.Figure:
    with revision_rc(style):
        fig, axes = plt.subplots(2, 2, figsize=(style.figure_width_in * 1.35, style.figure_height_in * 1.7))
        month_vals = [layers.loc[layers["month"] == m, "depth_m"].dropna().to_numpy() for m in range(1, 13)]
        _box(axes[0, 0], month_vals, MONTHS_RU, "Месяц", log=True)
        season_vals = [layers.loc[layers["season"] == s, "depth_m"].dropna().to_numpy() for s in SEASON_ORDER]
        _box(axes[0, 1], season_vals, [SEASONS_RU[s] for s in SEASON_ORDER], "Сезон", log=True)
        type_vals = [layers.loc[layers["position_type"] == k, "depth_m"].dropna().to_numpy() for k in INVERSION_TYPES]
        _box(axes[1, 0], type_vals, [TYPE_LABELS[k] for k in INVERSION_TYPES], "Тип", log=True, colors=[TYPE_COLORS[k] for k in INVERSION_TYPES])
        grouped = []
        labels = []
        colors = []
        for m in range(1, 13):
            for k in INVERSION_TYPES:
                grouped.append(layers.loc[(layers["month"] == m) & (layers["position_type"] == k), "depth_m"].dropna().to_numpy())
                labels.append(f"{MONTHS_RU[m - 1]} {k}" if k == "G" else k)
                colors.append(TYPE_COLORS[k])
        _box(axes[1, 1], grouped, labels, "Месяц × тип", log=True, colors=colors, rotate=True, small=True)
        axes[0, 0].set_ylabel("depth_m, м")
        axes[1, 0].set_ylabel("depth_m, м")
        fig.tight_layout()
        if caption:
            fig.subplots_adjust(bottom=0.08)
            fig.text(0.5, 0.01, caption, ha="center", fontsize=7.5, color="#4D4D4D")
        return fig


def _box(ax, groups, labels, xlabel, *, log=False, colors=None, rotate=False, small=False):
    data = [np.asarray(g, dtype=float) for g in groups]
    data = [g[np.isfinite(g) & (g > 0)] if log else g[np.isfinite(g)] for g in data]
    bp = ax.boxplot(
        data,
        tick_labels=labels,
        showfliers=False,
        whis=(5, 95),
        patch_artist=True,
        widths=0.65 if not small else 0.55,
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor((colors[i] if colors else "#AED6F1"))
        patch.set_alpha(0.75)
    if log:
        ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.4)
    ax.set_xlabel(xlabel)
    ax.tick_params(axis="x", rotation=90 if rotate else 0, labelsize=6 if small else 8)
    ymin = ax.get_ylim()[0]
    for i, g in enumerate(data, start=1):
        ax.text(i, ymin, str(len(g)), ha="center", va="bottom", fontsize=5.5, color="#566573")


def plot_gamma_sfc_annual_cycle(monthly: pd.DataFrame, style: FigureStyle, *, caption: str | None = None) -> plt.Figure:
    x = monthly["month"].to_numpy(int)
    with revision_rc(style):
        fig, ax = plt.subplots(figsize=(style.figure_width_in, style.figure_height_in))
        ymins, ymaxs = [], []
        for level in STANDARD_LEVELS:
            k = int(level)
            med = monthly[f"median_{k}"]
            q25 = monthly[f"q25_{k}"]
            q75 = monthly[f"q75_{k}"]
            ax.plot(x, med, marker="o", color=LEVEL_COLORS[level], label=LEVEL_LABELS[level], linewidth=1.8)
            ax.fill_between(x, q25, q75, color=LEVEL_COLORS[level], alpha=0.16, linewidth=0)
            ymins.extend(q25.dropna().tolist())
            ymaxs.extend(q75.dropna().tolist())
        ax.axhline(0, color="#7F8C8D", linewidth=0.9, linestyle="--")
        ax.set_xticks(range(1, 13), labels=MONTHS_RU)
        ax.set_xlabel("Месяц")
        ax.set_ylabel("γ_sfc-P, °C/100 м")
        ax.legend(frameon=False)
        ax.grid(True, axis="y", alpha=0.25)
        if ymins and ymaxs:
            lim = max(abs(min(ymins)), abs(max(ymaxs)), 0.2)
            ax.set_ylim(-lim, lim)
        return finish(fig, style, caption)


def plot_gamma_sfc_monthly_panels(year_month: pd.DataFrame, style: FigureStyle, *, caption: str | None = None) -> plt.Figure:
    finite = []
    for level in STANDARD_LEVELS:
        finite.append(year_month[f"median_{int(level)}"].to_numpy(float))
        finite.append(year_month[f"q25_{int(level)}"].to_numpy(float))
        finite.append(year_month[f"q75_{int(level)}"].to_numpy(float))
    lim = shared_abs_limit(*finite)
    with revision_rc(style):
        fig, axes = plt.subplots(4, 3, figsize=(style.figure_width_in * 1.4, style.figure_height_in * 2.0), sharey=True)
        for ax, month in zip(axes.ravel(), range(1, 13)):
            g = year_month[year_month["month"] == month].sort_values("year")
            for level in STANDARD_LEVELS:
                k = int(level)
                ax.plot(g["year"], g[f"median_{k}"], color=LEVEL_COLORS[level], linewidth=1.3, label=LEVEL_LABELS[level] if month == 1 else None)
                ax.fill_between(g["year"], g[f"q25_{k}"], g[f"q75_{k}"], color=LEVEL_COLORS[level], alpha=0.12, linewidth=0)
            ax.axhline(0, color="#7F8C8D", linewidth=0.8, linestyle="--")
            ax.set_ylim(-lim, lim)
            ax.set_title(MONTHS_RU[month - 1], fontsize=8)
            ax.grid(True, axis="y", alpha=0.2, linewidth=0.3)
        axes[0, 0].legend(frameon=False, fontsize=7)
        axes[3, 1].set_xlabel("Год")
        axes[1, 0].set_ylabel("γ_sfc-P, °C/100 м")
        fig.tight_layout()
        if caption:
            fig.subplots_adjust(bottom=0.06)
            fig.text(0.5, 0.01, caption, ha="center", fontsize=7.5, color="#4D4D4D")
        return fig


def plot_local_gamma_hist(local: pd.DataFrame, style: FigureStyle, *, caption: str | None = None) -> plt.Figure:
    vals = local["gamma_local_c_100m"].dropna().to_numpy(float)
    with revision_rc(style):
        fig, ax = plt.subplots(figsize=(style.figure_width_in, style.figure_height_in))
        ax.hist(vals, bins=np.linspace(-15, 15, 61), color="#5B2C6F", alpha=0.75)
        ax.axvline(0, color="#7F8C8D", linestyle="--", linewidth=0.9)
        ax.set_xlabel("γ_local, °C/100 м")
        ax.set_ylabel("Число интервалов")
        ax.text(0.98, 0.95, f"N = {vals.size}", transform=ax.transAxes, ha="right", va="top", fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)
        return finish(fig, style, caption)


def plot_local_gamma_hist_seasons(local: pd.DataFrame, style: FigureStyle, *, caption: str | None = None) -> plt.Figure:
    with revision_rc(style):
        fig, axes = plt.subplots(2, 2, figsize=(style.figure_width_in, style.figure_height_in * 1.35), sharex=True, sharey=True)
        for ax, season in zip(axes.ravel(), SEASON_ORDER):
            vals = local.loc[local["season"] == season, "gamma_local_c_100m"].dropna().to_numpy(float)
            ax.hist(vals, bins=np.linspace(-15, 15, 49), color="#1F618D", alpha=0.8)
            ax.axvline(0, color="#7F8C8D", linestyle="--", linewidth=0.8)
            ax.set_title(SEASONS_RU[season])
            ax.grid(True, axis="y", alpha=0.2)
        axes[1, 0].set_xlabel("γ_local, °C/100 м")
        axes[0, 0].set_ylabel("Число интервалов")
        fig.tight_layout()
        if caption:
            fig.subplots_adjust(bottom=0.08)
            fig.text(0.5, 0.01, caption, ha="center", fontsize=7.5, color="#4D4D4D")
        return fig


def plot_month_height_heatmap(
    table: pd.DataFrame,
    style: FigureStyle,
    *,
    label: str,
    cmap: str,
    vmin: float,
    vmax: float,
    center0: bool,
    caption: str | None = None,
) -> plt.Figure:
    mat = heatmap_matrix(table)
    values = mat.to_numpy(float)
    with revision_rc(style):
        fig, ax = plt.subplots(figsize=(style.figure_width_in, style.figure_height_in * 1.15))
        cmap_obj = mpl.colormaps[cmap].copy()
        if center0:
            image = ax.imshow(values, aspect="auto", cmap=cmap_obj, vmin=vmin, vmax=vmax, interpolation="nearest")
        else:
            image = ax.imshow(values, aspect="auto", cmap=cmap_obj, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xticks(range(12), labels=MONTHS_RU)
        yticks = np.arange(len(mat.index))
        ax.set_yticks(yticks, labels=[f"{v:.0f}" for v in mat.index])
        ax.set_xlabel("Месяц")
        ax.set_ylabel("Середина интервала AGL, м")
        fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label=label)
        return finish(fig, style, caption)


def plot_local_gamma_box_month(local: pd.DataFrame, style: FigureStyle, *, caption: str | None = None) -> plt.Figure:
    vals = [local.loc[local["month"] == m, "gamma_local_c_100m"].dropna().to_numpy() for m in range(1, 13)]
    with revision_rc(style):
        fig, ax = plt.subplots(figsize=(style.figure_width_in, style.figure_height_in))
        bp = ax.boxplot(vals, tick_labels=MONTHS_RU, showfliers=False, patch_artist=True, widths=0.62)
        for patch in bp["boxes"]:
            patch.set_facecolor("#AED6F1")
        ax.axhline(0, color="#7F8C8D", linestyle="--", linewidth=0.9)
        ax.set_xlabel("Месяц")
        ax.set_ylabel("γ_local, °C/100 м")
        ax.grid(True, axis="y", alpha=0.25)
        return finish(fig, style, caption)


def plot_local_gamma_00_12(local: pd.DataFrame, style: FigureStyle, *, caption: str | None = None) -> plt.Figure:
    with revision_rc(style):
        fig, axes = plt.subplots(1, 2, figsize=(style.figure_width_in * 1.2, style.figure_height_in), sharex=True, sharey=True)
        for ax, cycle, title in zip(axes, ("00", "12"), ("00 UTC", "12 UTC")):
            vals = local.loc[local["cycle"].str.zfill(2).str[-2:] == cycle, "gamma_local_c_100m"].dropna().to_numpy()
            ax.hist(vals, bins=np.linspace(-15, 15, 49), color="#117A65", alpha=0.8)
            ax.axvline(0, color="#7F8C8D", linestyle="--")
            ax.set_title(title)
            ax.set_xlabel("γ_local, °C/100 м")
            ax.text(0.97, 0.95, f"N={vals.size}", transform=ax.transAxes, ha="right", va="top", fontsize=8)
        axes[0].set_ylabel("Число интервалов")
        fig.tight_layout()
        if caption:
            fig.subplots_adjust(bottom=0.16)
            fig.text(0.5, 0.01, caption, ha="center", fontsize=7.5, color="#4D4D4D")
        return fig


def plot_seasonal_gamma_z(local: pd.DataFrame, style: FigureStyle, *, caption: str | None = None) -> plt.Figure:
    edges = np.arange(0.0, 4001.0, 200.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    with revision_rc(style):
        fig, ax = plt.subplots(figsize=(style.figure_width_in * 0.85, style.figure_height_in * 1.2))
        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for i, season in enumerate(SEASON_ORDER):
            g = local[local["season"] == season]
            med = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                chunk = g[(g["z_mid_agl_m"] >= lo) & (g["z_mid_agl_m"] < hi)]["gamma_local_c_100m"]
                med.append(float(chunk.median()) if len(chunk) else np.nan)
            ax.plot(med, centers, label=SEASONS_RU[season], color=cycle[i], linewidth=1.8)
        ax.axvline(0, color="#7F8C8D", linestyle="--", linewidth=0.9)
        ax.set_xlabel("медиана γ_local, °C/100 м")
        ax.set_ylabel("Высота AGL, м")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.25)
        return finish(fig, style, caption)


def plot_hexbin(x, y, style, xlabel, ylabel, caption=None, gridsize=35) -> plt.Figure:
    xf = np.asarray(x, float)
    yf = np.asarray(y, float)
    ok = np.isfinite(xf) & np.isfinite(yf)
    with revision_rc(style):
        fig, ax = plt.subplots(figsize=(style.figure_width_in, style.figure_height_in))
        hb = ax.hexbin(xf[ok], yf[ok], gridsize=gridsize, bins="log", cmap="magma", mincnt=1, rasterized=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.colorbar(hb, ax=ax, fraction=0.035, pad=0.02, label="log10(N)")
        return finish(fig, style, caption)


def plot_depth_delta_density(layers: pd.DataFrame, style: FigureStyle, *, caption: str | None = None) -> plt.Figure:
    return plot_hexbin(
        layers["depth_m"], layers["delta_t_c"], style,
        "Толщина depth_m, м", "ΔT слоя, °C", caption,
    )


def plot_multilayer_hist(counts: pd.DataFrame, style: FigureStyle, *, caption: str | None = None) -> plt.Figure:
    with revision_rc(style):
        fig, ax = plt.subplots(figsize=(style.figure_width_in * 0.85, style.figure_height_in))
        vals = counts["n_inversion_layers"].to_numpy(int)
        bins = np.arange(-0.5, min(8, vals.max() + 1.5), 1)
        ax.hist(vals, bins=bins, color="#1A5276", rwidth=0.85)
        ax.set_xlabel("Число слоёв v3 на профиль")
        ax.set_ylabel("Число профилей")
        ax.grid(True, axis="y", alpha=0.25)
        return finish(fig, style, caption)


def plot_multilayer_monthly_stack(counts: pd.DataFrame, style: FigureStyle, *, caption: str | None = None) -> plt.Figure:
    classes = ["0", "1", "2", "3+"]
    colors = ["#BDC3C7", "#2471A3", "#B9770E", "#6C3483"]
    with revision_rc(style):
        fig, ax = plt.subplots(figsize=(style.figure_width_in, style.figure_height_in))
        bottom = np.zeros(12)
        for cls, color in zip(classes, colors):
            fracs = []
            for m in range(1, 13):
                g = counts[counts["month"] == m]
                fracs.append(100.0 * (g["n_class"].astype(str) == cls).mean() if len(g) else 0.0)
            ax.bar(range(1, 13), fracs, bottom=bottom, color=color, label=f"{cls} слоёв", width=0.82)
            bottom += np.asarray(fracs)
        ax.set_xticks(range(1, 13), labels=MONTHS_RU)
        ax.set_ylabel("% пригодных профилей")
        ax.set_ylim(0, 100)
        ax.legend(frameon=False, ncol=4)
        ax.grid(True, axis="y", alpha=0.25)
        return finish(fig, style, caption)


def plot_multilayer_season_stack(counts: pd.DataFrame, style: FigureStyle, *, caption: str | None = None) -> plt.Figure:
    classes = ["0", "1", "2", "3+"]
    colors = ["#BDC3C7", "#2471A3", "#B9770E", "#6C3483"]
    with revision_rc(style):
        fig, ax = plt.subplots(figsize=(style.figure_width_in * 0.8, style.figure_height_in))
        x = np.arange(4)
        bottom = np.zeros(4)
        for cls, color in zip(classes, colors):
            fracs = []
            for s in SEASON_ORDER:
                g = counts[counts["season"] == s]
                fracs.append(100.0 * (g["n_class"].astype(str) == cls).mean() if len(g) else 0.0)
            ax.bar(x, fracs, bottom=bottom, color=color, label=f"{cls} слоёв", width=0.7)
            bottom += np.asarray(fracs)
        ax.set_xticks(x, labels=[SEASONS_RU[s] for s in SEASON_ORDER])
        ax.set_ylabel("% пригодных профилей")
        ax.set_ylim(0, 100)
        ax.legend(frameon=False)
        return finish(fig, style, caption)


def plot_year_month_heatmap(matrix: pd.DataFrame, style: FigureStyle, *, label: str, vmin: float, vmax: float, caption: str | None = None, cmap="YlOrRd") -> plt.Figure:
    values = matrix.to_numpy(float)
    with revision_rc(style):
        fig, ax = plt.subplots(figsize=(style.figure_width_in, max(style.figure_height_in, 5.0)))
        cmap_obj = mpl.colormaps[cmap].copy()
        image = ax.imshow(np.ma.masked_invalid(values), aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap_obj, interpolation="nearest")
        ax.set_xticks(range(12), labels=MONTHS_RU)
        ax.set_yticks(range(len(matrix.index)), labels=[str(y) for y in matrix.index])
        ax.set_xlabel("Месяц")
        ax.set_ylabel("Год")
        fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label=label)
        return finish(fig, style, caption)


def plot_multilayer_00_12(counts: pd.DataFrame, style: FigureStyle, *, caption: str | None = None) -> plt.Figure:
    with revision_rc(style):
        fig, ax = plt.subplots(figsize=(style.figure_width_in, style.figure_height_in))
        for cycle, color in (("00", "#1A5276"), ("12", "#B9770E")):
            g = counts[counts["cycle"].astype(str).str.zfill(2).str[-2:] == cycle]
            fracs = [100.0 * g.loc[g["month"] == m, "multilayer"].mean() if (g["month"] == m).any() else np.nan for m in range(1, 13)]
            ax.plot(range(1, 13), fracs, marker="o", color=color, label=f"{cycle} UTC")
        ax.set_xticks(range(1, 13), labels=MONTHS_RU)
        ax.set_ylabel("P(n_layers ≥ 2), %")
        ax.legend(frameon=False)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_ylim(bottom=0)
        return finish(fig, style, caption)


def plot_gehe_summary(layers: pd.DataFrame, monthly_freq: pd.DataFrame, style: FigureStyle, *, caption: str | None = None) -> plt.Figure:
    with revision_rc(style):
        fig, axes = plt.subplots(4, 3, figsize=(style.figure_width_in * 1.45, style.figure_height_in * 2.35), sharex="col")
        metrics = [
            ("base_height_agl_m", "Основание AGL, м"),
            ("depth_m", "Толщина, м"),
            ("delta_t_c", "ΔT, °C"),
        ]
        ymax = {col: float(np.nanpercentile(layers[col], 95)) for col, _ in metrics}
        for col_i, kind in enumerate(INVERSION_TYPES):
            g = layers[layers["position_type"] == kind]
            axes[0, col_i].plot(monthly_freq["month"], monthly_freq[f"F_{kind}"], color=TYPE_COLORS[kind], marker="o")
            axes[0, col_i].set_title(TYPE_LABELS[kind])
            axes[0, col_i].set_ylim(0, float(np.nanmax(monthly_freq[[f"F_{k}" for k in INVERSION_TYPES]].to_numpy())) * 1.08)
            axes[0, col_i].grid(True, axis="y", alpha=0.25)
            for row_i, (col, ylab) in enumerate(metrics, start=1):
                med = [g.loc[g["month"] == m, col].median() for m in range(1, 13)]
                q25 = [g.loc[g["month"] == m, col].quantile(0.25) for m in range(1, 13)]
                q75 = [g.loc[g["month"] == m, col].quantile(0.75) for m in range(1, 13)]
                ax = axes[row_i, col_i]
                ax.plot(range(1, 13), med, color=TYPE_COLORS[kind], marker="o", linewidth=1.6)
                ax.fill_between(range(1, 13), q25, q75, color=TYPE_COLORS[kind], alpha=0.18)
                ax.set_ylim(0, ymax[col] * 1.05 if ymax[col] > 0 else 1)
                ax.grid(True, axis="y", alpha=0.25)
                if col_i == 0:
                    ax.set_ylabel(ylab)
            axes[0, 0].set_ylabel("F, % профилей")
            axes[3, col_i].set_xticks(range(1, 13), labels=MONTHS_RU, rotation=45)
        fig.tight_layout()
        if caption:
            fig.subplots_adjust(bottom=0.08)
            fig.text(0.5, 0.01, caption, ha="center", fontsize=7.5, color="#4D4D4D")
        return fig


def plot_type01_shared(matrices: dict[str, pd.DataFrame], style: FigureStyle, *, caption: str | None = None) -> plt.Figure:
    vmax = 0.0
    for mat in matrices.values():
        finite = mat.to_numpy(float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            vmax = max(vmax, float(np.nanmax(finite)))
    vmax = max(vmax, 1.0)
    with revision_rc(style):
        fig, axes = plt.subplots(1, 3, figsize=(style.figure_width_in * 1.7, style.figure_height_in * 1.35), sharey=True)
        image = None
        for ax, kind in zip(axes, INVERSION_TYPES):
            values = matrices[kind].to_numpy(float)
            image = ax.imshow(np.ma.masked_invalid(values), aspect="auto", vmin=0, vmax=vmax, cmap="YlOrRd", interpolation="nearest")
            ax.set_title(TYPE_LABELS[kind])
            ax.set_xticks(range(12), labels=MONTHS_RU, rotation=45)
            ax.set_yticks(range(len(matrices[kind].index)), labels=[str(y) for y in matrices[kind].index])
            ax.set_xlabel("Месяц")
        axes[0].set_ylabel("Год")
        fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label="Повторяемость профилей, %")
        fig.subplots_adjust(bottom=0.14, right=0.9)
        if caption:
            fig.text(0.5, 0.01, caption, ha="center", fontsize=7.5, color="#4D4D4D")
        return fig


def plot_scatter_typed(layers: pd.DataFrame, style: FigureStyle, xcol, ycol, xlabel, ylabel, caption=None) -> plt.Figure:
    with revision_rc(style):
        fig, ax = plt.subplots(figsize=(style.figure_width_in, style.figure_height_in))
        for kind in INVERSION_TYPES:
            g = layers[layers["position_type"] == kind]
            ax.scatter(g[xcol], g[ycol], s=8, alpha=0.35, color=TYPE_COLORS[kind], label=TYPE_LABELS[kind], linewidths=0, rasterized=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.25)
        return finish(fig, style, caption)


def plot_violin(groups, labels, ylabel, style, caption=None, colors=None) -> plt.Figure:
    with revision_rc(style):
        fig, ax = plt.subplots(figsize=(style.figure_width_in, style.figure_height_in))
        data = [np.asarray(g, float) for g in groups]
        data = [g[np.isfinite(g)] for g in data]
        parts = ax.violinplot(data, showmedians=True, showextrema=False)
        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(colors[i] if colors else "#5DADE2")
            body.set_alpha(0.7)
        ax.set_xticks(range(1, len(labels) + 1), labels=labels)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
        return finish(fig, style, caption)


def plot_ridgeline(series_by_group: list[np.ndarray], labels: list[str], xlabel: str, style: FigureStyle, caption=None) -> plt.Figure:
    with revision_rc(style):
        fig, ax = plt.subplots(figsize=(style.figure_width_in, style.figure_height_in * 1.35))
        ymax = 0.0
        for i, (vals, lab) in enumerate(zip(series_by_group, labels)):
            v = np.asarray(vals, float)
            v = v[np.isfinite(v)]
            if v.size < 5:
                continue
            bins = np.linspace(np.nanpercentile(v, 1), np.nanpercentile(v, 99), 40)
            hist, edges = np.histogram(v, bins=bins, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            offset = i * (np.nanmax(hist) * 0.85 if hist.size else 1)
            ax.fill_between(centers, offset, hist + offset, alpha=0.55, color="#2471A3")
            ax.plot(centers, hist + offset, color="#1B4F72", linewidth=0.8)
            ax.text(centers[0], offset, lab, fontsize=7.5, va="bottom")
            ymax = max(ymax, offset + (np.nanmax(hist) if hist.size else 0))
        ax.set_xlabel(xlabel)
        ax.set_yticks([])
        ax.set_ylim(0, ymax * 1.05 if ymax else 1)
        return finish(fig, style, caption)


def plot_ecdf(layers: pd.DataFrame, col: str, xlabel: str, style: FigureStyle, caption=None) -> plt.Figure:
    with revision_rc(style):
        fig, ax = plt.subplots(figsize=(style.figure_width_in, style.figure_height_in))
        for kind in INVERSION_TYPES:
            v = np.sort(layers.loc[layers["position_type"] == kind, col].dropna().to_numpy(float))
            if v.size == 0:
                continue
            y = np.arange(1, v.size + 1) / v.size
            ax.plot(v, y, color=TYPE_COLORS[kind], label=TYPE_LABELS[kind], linewidth=1.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("ECDF")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(0, 1)
        return finish(fig, style, caption)


def plot_month_type_heatmap(monthly_freq: pd.DataFrame, style: FigureStyle, caption=None) -> plt.Figure:
    mat = monthly_freq.set_index("month")[[f"F_{k}" for k in INVERSION_TYPES]].T
    mat.index = [TYPE_LABELS[k] for k in INVERSION_TYPES]
    with revision_rc(style):
        fig, ax = plt.subplots(figsize=(style.figure_width_in, 2.6))
        image = ax.imshow(mat.to_numpy(float), aspect="auto", vmin=0, vmax=float(np.nanmax(mat.to_numpy())), cmap="YlOrRd")
        ax.set_xticks(range(12), labels=MONTHS_RU)
        ax.set_yticks(range(3), labels=list(mat.index))
        fig.colorbar(image, ax=ax, fraction=0.04, label="% профилей")
        return finish(fig, style, caption)


def plot_bubble_year_month(flags: pd.DataFrame, layers: pd.DataFrame, style: FigureStyle, caption=None) -> plt.Figure:
    work = flags.copy()
    work["_any"] = work[[f"has_{k}" for k in INVERSION_TYPES]].any(axis=1)
    freq = work.groupby(["year", "month"])["_any"].mean().mul(100.0).reset_index(name="freq")
    med = layers.groupby(["year", "month"])["delta_t_c"].median().reset_index()
    tab = freq.merge(med, on=["year", "month"], how="left")
    with revision_rc(style):
        fig, ax = plt.subplots(figsize=(style.figure_width_in, style.figure_height_in * 1.2))
        sc = ax.scatter(
            tab["month"], tab["year"], s=np.clip(tab["freq"], 0, None) * 3.5,
            c=tab["delta_t_c"], cmap="plasma", alpha=0.85, edgecolors="none",
        )
        ax.set_xticks(range(1, 13), labels=MONTHS_RU)
        ax.set_xlabel("Месяц")
        ax.set_ylabel("Год")
        fig.colorbar(sc, ax=ax, fraction=0.035, label="медиана ΔT слоя, °C")
        return finish(fig, style, caption)


def plot_seasonal_phase(layers: pd.DataFrame, counts: pd.DataFrame, style: FigureStyle, caption=None) -> plt.Figure:
    with revision_rc(style):
        fig, ax = plt.subplots(figsize=(style.figure_width_in * 0.9, style.figure_height_in))
        for season in SEASON_ORDER:
            ly = layers[layers["season"] == season]
            pr = counts[counts["season"] == season]
            freq = 100.0 * pr["has_any"].mean() if len(pr) else 0
            ax.scatter(
                ly["depth_m"].median(), ly["delta_t_c"].median(),
                s=max(freq, 1) * 8, color={"DJF": "#2874A3", "MAM": "#1E8449", "JJA": "#B7950B", "SON": "#A04000"}[season],
                label=f"{SEASONS_RU[season]} ({freq:.0f}%)", alpha=0.85,
            )
        ax.set_xlabel("медиана depth_m, м")
        ax.set_ylabel("медиана ΔT, °C")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.25)
        return finish(fig, style, caption)


def plot_density_by_type(layers: pd.DataFrame, style: FigureStyle, caption=None) -> plt.Figure:
    with revision_rc(style):
        fig, axes = plt.subplots(1, 3, figsize=(style.figure_width_in * 1.5, style.figure_height_in), sharex=True, sharey=True)
        xlim = (0, float(np.nanpercentile(layers["depth_m"], 99)))
        ylim = (0, float(np.nanpercentile(layers["delta_t_c"], 99)))
        last = None
        for ax, kind in zip(axes, INVERSION_TYPES):
            g = layers[layers["position_type"] == kind]
            last = ax.hexbin(g["depth_m"], g["delta_t_c"], gridsize=28, bins="log", cmap="magma", mincnt=1, rasterized=True, extent=(*xlim, *ylim))
            ax.set_title(TYPE_LABELS[kind])
            ax.set_xlabel("depth_m, м")
        axes[0].set_ylabel("ΔT, °C")
        fig.colorbar(last, ax=axes.ravel().tolist(), fraction=0.02, label="log10(N)")
        fig.tight_layout()
        if caption:
            fig.subplots_adjust(bottom=0.16)
            fig.text(0.5, 0.01, caption, ha="center", fontsize=7.5, color="#4D4D4D")
        return fig


def plot_mean_layers_month(counts: pd.DataFrame, style: FigureStyle, caption=None) -> plt.Figure:
    with revision_rc(style):
        fig, ax = plt.subplots(figsize=(style.figure_width_in, style.figure_height_in))
        med = [counts.loc[counts["month"] == m, "n_inversion_layers"].median() for m in range(1, 13)]
        mean = [counts.loc[counts["month"] == m, "n_inversion_layers"].mean() for m in range(1, 13)]
        ax.plot(range(1, 13), mean, marker="o", label="среднее", color="#1A5276")
        ax.plot(range(1, 13), med, marker="s", label="медиана", color="#B9770E")
        ax.set_xticks(range(1, 13), labels=MONTHS_RU)
        ax.set_ylabel("слоёв на профиль")
        ax.legend(frameon=False)
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", alpha=0.25)
        return finish(fig, style, caption)


def plot_qc_base_top(layers: pd.DataFrame, style: FigureStyle, caption=None) -> plt.Figure:
    with revision_rc(style):
        fig, ax = plt.subplots(figsize=(style.figure_width_in * 0.85, style.figure_height_in))
        ax.scatter(layers["base_height_agl_m"], layers["top_height_agl_m"], s=6, alpha=0.2, rasterized=True, color="#1A5276", linewidths=0)
        lim = float(np.nanmax([layers["base_height_agl_m"].max(), layers["top_height_agl_m"].max()]))
        ax.plot([0, lim], [0, lim], "--", color="#7F8C8D", linewidth=0.9)
        ax.set_xlabel("base AGL, м")
        ax.set_ylabel("top AGL, м")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        n_bad = int((layers["top_height_agl_m"] <= layers["base_height_agl_m"]).sum())
        ax.text(0.05, 0.95, f"top ≤ base: {n_bad}", transform=ax.transAxes, va="top")
        return finish(fig, style, caption)


def plot_simple_hist(values, xlabel, style, caption=None, logy=False) -> plt.Figure:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    with revision_rc(style):
        fig, ax = plt.subplots(figsize=(style.figure_width_in, style.figure_height_in))
        ax.hist(v, bins=40, color="#5D6D7E")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("N")
        ax.text(0.97, 0.95, f"N={v.size}", transform=ax.transAxes, ha="right", va="top")
        return finish(fig, style, caption)


def plot_counts_year_month(qc: pd.DataFrame, style: FigureStyle, caption=None) -> plt.Figure:
    use = qc[qc["eligible_article"]]
    mat = use.groupby(["year", "month"]).size().reset_index(name="n").pivot(index="year", columns="month", values="n").reindex(columns=range(1, 13))
    return plot_year_month_heatmap(mat, style, label="Число пригодных профилей", vmin=0, vmax=float(np.nanmax(mat.to_numpy())), caption=caption, cmap="cividis")
