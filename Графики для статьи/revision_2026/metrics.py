"""Расчёты ревизии: две γ, знаменатель профилей, геометрия слоёв."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from gdex_bufr.profile_climate.article_figures.config import AnalysisConfig
from gdex_bufr.profile_climate.article_figures.metrics import (
    INVERSION_TYPES,
    SEASON_BY_MONTH,
    SEASON_ORDER,
    _collapse_duplicate_heights,
    _edges_with_overflow,
    compute_inversion_layers,
    profile_type_flags,
)

STANDARD_LEVELS_HPA = (850.0, 700.0, 500.0)
HEIGHT_AGL_EDGES_M = (
    0.0, 100.0, 200.0, 400.0, 600.0, 800.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0,
)
DEPTH_EDGES_M = (0.0, 50.0, 100.0, 150.0, 250.0, 400.0, 600.0, 800.0, 1200.0, 2000.0)
GAMMA_EXTREME_ABS_THRESHOLD_C_PER_100M = 15.0


def _profile_datetime_fields(q: dict) -> dict[str, object]:
    dt = q.get("datetime_utc")
    if dt is None or (isinstance(dt, float) and np.isnan(dt)):
        return {"datetime_utc": pd.NaT, "date": pd.NaT}
    ts = pd.Timestamp(dt)
    return {"datetime_utc": ts, "date": ts.normalize()}


def filter_extreme_gamma_local(
    local: pd.DataFrame,
    *,
    threshold: float = GAMMA_EXTREME_ABS_THRESHOLD_C_PER_100M,
) -> pd.DataFrame:
    """Интервалы с |γ_local| ≥ порога; дата профиля для QC экстремумов."""
    if local.empty:
        return pd.DataFrame(
            columns=[
                "profile_id",
                "datetime_utc",
                "date",
                "year",
                "month",
                "cycle",
                "season",
                "gamma_local_c_100m",
                "z_mid_agl_m",
                "dz_m",
                "dt_c",
            ]
        )
    use = local.copy()
    gamma = pd.to_numeric(use["gamma_local_c_100m"], errors="coerce")
    out = use.loc[gamma.abs() >= threshold].copy()
    cols = [
        "profile_id",
        "datetime_utc",
        "date",
        "year",
        "month",
        "cycle",
        "season",
        "gamma_local_c_100m",
        "z_mid_agl_m",
        "dz_m",
        "dt_c",
    ]
    out = out[[c for c in cols if c in out.columns]]
    sort_cols = [c for c in ("date", "datetime_utc", "profile_id") if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True)


def filter_extreme_gamma_sfc(
    sfc: pd.DataFrame,
    *,
    threshold: float = GAMMA_EXTREME_ABS_THRESHOLD_C_PER_100M,
    levels_hpa: Sequence[float] = STANDARD_LEVELS_HPA,
) -> pd.DataFrame:
    """Профиль × изобара, где |γ_sfc-P| ≥ порога."""
    rows: list[pd.DataFrame] = []
    base_cols = ["profile_id", "datetime_utc", "date", "year", "month", "cycle", "season"]
    for level in levels_hpa:
        col = f"gamma_sfc_{int(level)}"
        if col not in sfc.columns:
            continue
        vals = pd.to_numeric(sfc[col], errors="coerce")
        chunk = sfc.loc[vals.abs() >= threshold, [c for c in base_cols if c in sfc.columns] + [col]].copy()
        if chunk.empty:
            continue
        chunk = chunk.rename(columns={col: "gamma_sfc_c_100m"})
        chunk["pressure_hpa"] = float(level)
        rows.append(chunk)
    if not rows:
        return pd.DataFrame(
            columns=base_cols + ["pressure_hpa", "gamma_sfc_c_100m"]
        )
    out = pd.concat(rows, ignore_index=True)
    sort_cols = [c for c in ("date", "datetime_utc", "pressure_hpa", "profile_id") if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True)


def valid_layers(layers: pd.DataFrame) -> pd.DataFrame:
    """Физически допустимые слои: base≥0, top>base, depth>0."""
    if layers.empty:
        return layers.copy()
    use = layers.copy()
    base = pd.to_numeric(use["base_height_agl_m"], errors="coerce")
    top = pd.to_numeric(use["top_height_agl_m"], errors="coerce")
    depth = pd.to_numeric(use["depth_m"], errors="coerce")
    recomputed = top - base
    mask = (
        base.ge(-1e-6)
        & top.gt(base)
        & depth.gt(0)
        & np.isfinite(recomputed)
        & (np.abs(depth - recomputed) < 1e-3)
    )
    out = use.loc[mask].copy()
    out["depth_m"] = recomputed.loc[mask].to_numpy(float)
    out["season"] = out["month"].map(SEASON_BY_MONTH)
    return out.reset_index(drop=True)


def layer_geometry_ok(row: dict) -> bool:
    base = float(row["base_height_agl_m"])
    top = float(row["top_height_agl_m"])
    depth = float(row["depth_m"])
    return base >= -1e-6 and top > base and depth > 0 and abs(depth - (top - base)) < 1e-6


def gamma_local_interval(t0: float, t1: float, z0: float, z1: float) -> float:
    dz = z1 - z0
    if dz <= 0 or not np.isfinite(dz):
        return float("nan")
    return 100.0 * (t1 - t0) / dz


def gamma_sfc_to_level(t_sfc: float, t_p: float, h_sfc: float, h_p: float) -> float:
    dh = h_p - h_sfc
    if dh <= 0 or not np.isfinite(dh):
        return float("nan")
    return 100.0 * (t_p - t_sfc) / dh


def interpolate_inside(p: np.ndarray, values: np.ndarray, p_target: float) -> float:
    """Линейная интерполяция по давлению без экстраполяции."""
    valid = np.isfinite(p) & np.isfinite(values)
    p_use = np.asarray(p, dtype=float)[valid]
    v_use = np.asarray(values, dtype=float)[valid]
    if p_use.size < 2:
        return float("nan")
    order = np.argsort(p_use, kind="mergesort")
    p_use = p_use[order]
    v_use = v_use[order]
    p_uniq, idx = np.unique(p_use, return_index=True)
    v_use = v_use[idx]
    if p_uniq.size < 2:
        return float("nan")
    if p_target < p_uniq[0] or p_target > p_uniq[-1]:
        return float("nan")
    return float(np.interp(p_target, p_uniq, v_use))


def compute_sfc_level_gamma(
    df: pd.DataFrame,
    profile_qc: pd.DataFrame,
    config: AnalysisConfig,
    *,
    levels_hpa: Sequence[float] = STANDARD_LEVELS_HPA,
) -> pd.DataFrame:
    """Средний градиент от нижнего уровня профиля до изобары P (без экстраполяции)."""
    eligible = set(
        profile_qc.loc[profile_qc["eligible_article"], "profile_id"].astype(str)
    )
    qc_map = (
        profile_qc.assign(profile_id=profile_qc["profile_id"].astype(str))
        .set_index("profile_id")
        .to_dict("index")
    )
    profile_ids = df["profile_id"].astype(str).to_numpy()
    p_all = df["pressure_hpa"].to_numpy(float)
    t_all = df["temperature_c"].to_numpy(float)
    z_all = df["height_m"].to_numpy(float)
    groups = df.assign(_pid=profile_ids).groupby("_pid", sort=False).indices
    rows: list[dict] = []
    top = config.pressure_top_hpa
    bottom = config.pressure_bottom_hpa
    for pid, idx in groups.items():
        if pid not in eligible:
            continue
        q = qc_map[pid]
        idx = np.asarray(idx, dtype=int)
        p = p_all[idx]
        mask = np.isfinite(p) & (p >= top) & (p <= bottom)
        p = p[mask]
        t = t_all[idx][mask]
        z = z_all[idx][mask]
        ok = np.isfinite(t) & np.isfinite(z)
        p, t, z = p[ok], t[ok], z[ok]
        if p.size < 2:
            continue
        sfc = int(np.nanargmax(p))
        t_sfc = float(t[sfc])
        z_sfc = float(z[sfc])
        p_min = float(np.nanmin(p))
        p_max = float(np.nanmax(p))
        rec: dict = {
            "profile_id": pid,
            **_profile_datetime_fields(q),
            "year": int(q["year"]),
            "month": int(q["month"]),
            "cycle": str(q["cycle"]),
            "season": SEASON_BY_MONTH.get(int(q["month"])),
            "t_surface_c": t_sfc,
            "h_surface_m": z_sfc,
            "p_surface_hpa": float(p[sfc]),
        }
        for level in levels_hpa:
            key = int(level)
            if level < p_min or level > p_max:
                rec[f"t_{key}_c"] = np.nan
                rec[f"h_{key}_m"] = np.nan
                rec[f"delta_t_surface_{key}"] = np.nan
                rec[f"delta_h_surface_{key}"] = np.nan
                rec[f"gamma_sfc_{key}"] = np.nan
                continue
            t_p = interpolate_inside(p, t, float(level))
            h_p = interpolate_inside(p, z, float(level))
            rec[f"t_{key}_c"] = t_p
            rec[f"h_{key}_m"] = h_p
            rec[f"delta_t_surface_{key}"] = t_p - t_sfc if np.isfinite(t_p) else np.nan
            rec[f"delta_h_surface_{key}"] = h_p - z_sfc if np.isfinite(h_p) else np.nan
            rec[f"gamma_sfc_{key}"] = gamma_sfc_to_level(t_sfc, t_p, z_sfc, h_p)
        rows.append(rec)
    return pd.DataFrame.from_records(rows)


def sfc_gamma_monthly(table: pd.DataFrame, levels_hpa: Sequence[float] = STANDARD_LEVELS_HPA) -> pd.DataFrame:
    rows = []
    for month in range(1, 13):
        g = table[table["month"] == month]
        rec = {"month": month, "n_profiles": int(len(g))}
        for level in levels_hpa:
            col = f"gamma_sfc_{int(level)}"
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            rec[f"median_{int(level)}"] = float(vals.median()) if len(vals) else np.nan
            rec[f"q25_{int(level)}"] = float(vals.quantile(0.25)) if len(vals) else np.nan
            rec[f"q75_{int(level)}"] = float(vals.quantile(0.75)) if len(vals) else np.nan
            rec[f"n_{int(level)}"] = int(len(vals))
        rows.append(rec)
    return pd.DataFrame(rows)


def _normalize_cycle(series: pd.Series) -> pd.Series:
    return series.astype(str).str.zfill(2).str[-2:]


def sfc_gamma_monthly_by_cycle(
    table: pd.DataFrame,
    levels_hpa: Sequence[float] = STANDARD_LEVELS_HPA,
) -> pd.DataFrame:
    rows = []
    cy = _normalize_cycle(table["cycle"])
    for month in range(1, 13):
        for cycle in ("00", "12"):
            g = table[(table["month"] == month) & (cy == cycle)]
            rec = {"month": month, "cycle": cycle, "n_profiles": int(len(g))}
            for level in levels_hpa:
                col = f"gamma_sfc_{int(level)}"
                vals = pd.to_numeric(g[col], errors="coerce").dropna()
                rec[f"median_{int(level)}"] = float(vals.median()) if len(vals) else np.nan
                rec[f"q25_{int(level)}"] = float(vals.quantile(0.25)) if len(vals) else np.nan
                rec[f"q75_{int(level)}"] = float(vals.quantile(0.75)) if len(vals) else np.nan
                rec[f"n_{int(level)}"] = int(len(vals))
            rows.append(rec)
    return pd.DataFrame(rows)


def sfc_gamma_year_month(table: pd.DataFrame, levels_hpa: Sequence[float] = STANDARD_LEVELS_HPA) -> pd.DataFrame:
    rows = []
    for (year, month), g in table.groupby(["year", "month"], sort=True):
        rec = {"year": int(year), "month": int(month), "n_profiles": int(len(g))}
        for level in levels_hpa:
            col = f"gamma_sfc_{int(level)}"
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            rec[f"median_{int(level)}"] = float(vals.median()) if len(vals) else np.nan
            rec[f"q25_{int(level)}"] = float(vals.quantile(0.25)) if len(vals) else np.nan
            rec[f"q75_{int(level)}"] = float(vals.quantile(0.75)) if len(vals) else np.nan
            rec[f"n_{int(level)}"] = int(len(vals))
        rows.append(rec)
    return pd.DataFrame(rows)


def compute_local_gammas(
    df: pd.DataFrame,
    profile_qc: pd.DataFrame,
    config: AnalysisConfig,
) -> pd.DataFrame:
    """γ_i = 100·ΔT/Δz по всем соседним интервалам пригодного профиля."""
    eligible = set(
        profile_qc.loc[profile_qc["eligible_article"], "profile_id"].astype(str)
    )
    qc_map = (
        profile_qc.assign(profile_id=profile_qc["profile_id"].astype(str))
        .set_index("profile_id")
        .to_dict("index")
    )
    profile_ids = df["profile_id"].astype(str).to_numpy()
    p_all = df["pressure_hpa"].to_numpy(float)
    t_all = df["temperature_c"].to_numpy(float)
    z_all = df["height_m"].to_numpy(float)
    groups = df.assign(_pid=profile_ids).groupby("_pid", sort=False).indices
    rows: list[dict] = []
    top = config.pressure_top_hpa
    bottom = config.pressure_bottom_hpa
    for pid, idx in groups.items():
        if pid not in eligible:
            continue
        q = qc_map[pid]
        idx = np.asarray(idx, dtype=int)
        p = p_all[idx]
        mask = np.isfinite(p) & (p >= top) & (p <= bottom)
        z, t, _p = _collapse_duplicate_heights(z_all[idx][mask], t_all[idx][mask], p[mask])
        if z.size < 2:
            continue
        z0 = float(z[0])
        dz = np.diff(z)
        dt = np.diff(t)
        ok = (dz > 0) & np.isfinite(dz) & np.isfinite(dt)
        month = int(q["month"])
        year = int(q["year"])
        cycle = str(q["cycle"])
        season = SEASON_BY_MONTH.get(month)
        for i in np.flatnonzero(ok):
            gamma = 100.0 * float(dt[i]) / float(dz[i])
            z_mid = 0.5 * (float(z[i]) + float(z[i + 1]))
            rows.append(
                {
                    "profile_id": pid,
                    **_profile_datetime_fields(q),
                    "year": year,
                    "month": month,
                    "cycle": cycle,
                    "season": season,
                    "gamma_local_c_100m": gamma,
                    "z_mid_agl_m": z_mid - z0,
                    "dz_m": float(dz[i]),
                    "dt_c": float(dt[i]),
                }
            )
    return pd.DataFrame.from_records(rows)


def month_height_gamma_heatmap(
    local: pd.DataFrame,
    *,
    height_edges: Sequence[float] = HEIGHT_AGL_EDGES_M,
    stat: str = "median",
) -> pd.DataFrame:
    """Месяц × бин высоты AGL: медиана γ_local или P(γ>0)/P(γ<0)."""
    edges = np.asarray(tuple(height_edges) + (np.inf,), dtype=float)
    left = edges[:-1]
    use = local.dropna(subset=["gamma_local_c_100m", "z_mid_agl_m"]).copy()
    use["bin"] = pd.cut(use["z_mid_agl_m"], bins=edges, right=False, include_lowest=True)
    rows = []
    for month in range(1, 13):
        gm = use[use["month"] == month]
        for i, lo in enumerate(left):
            hi = edges[i + 1]
            chunk = gm[gm["z_mid_agl_m"].ge(lo) & gm["z_mid_agl_m"].lt(hi if np.isfinite(hi) else 1e12)]
            vals = chunk["gamma_local_c_100m"]
            n = int(len(vals))
            if stat == "median":
                value = float(vals.median()) if n else np.nan
            elif stat == "p_positive":
                value = float((vals > 0).mean() * 100.0) if n else np.nan
            elif stat == "p_negative":
                value = float((vals < 0).mean() * 100.0) if n else np.nan
            else:
                raise ValueError(stat)
            rows.append(
                {
                    "month": month,
                    "bin_left": float(lo),
                    "bin_right": float(hi) if np.isfinite(hi) else np.nan,
                    "n_intervals": n,
                    "value": value,
                }
            )
    return pd.DataFrame(rows)


def heatmap_matrix(table: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    bins = sorted(table["bin_left"].unique())
    mat = (
        table.pivot(index="bin_left", columns="month", values=value_col)
        .reindex(index=bins, columns=range(1, 13))
        .sort_index(ascending=False)
    )
    return mat


def profile_layer_counts(layers: pd.DataFrame, profile_qc: pd.DataFrame) -> pd.DataFrame:
    eligible = profile_qc[profile_qc["eligible_article"]].copy()
    eligible["profile_id"] = eligible["profile_id"].astype(str)
    counts = (
        layers.groupby("profile_id", sort=False).size().rename("n_inversion_layers")
        if not layers.empty
        else pd.Series(dtype=int)
    )
    out = eligible[["profile_id", "year", "month", "cycle"]].copy()
    out["n_inversion_layers"] = out["profile_id"].map(counts).fillna(0).astype(int)
    out["season"] = out["month"].map(SEASON_BY_MONTH)
    out["has_any"] = out["n_inversion_layers"] > 0
    out["multilayer"] = out["n_inversion_layers"] >= 2
    out["n_class"] = pd.cut(
        out["n_inversion_layers"],
        bins=[-0.5, 0.5, 1.5, 2.5, 99],
        labels=["0", "1", "2", "3+"],
    )
    return out


def frequency_percent(
    flags: pd.DataFrame,
    *,
    condition: pd.Series,
    group_cols: Sequence[str],
) -> pd.DataFrame:
    work = flags.copy()
    work["_hit"] = condition.astype(bool)
    grouped = work.groupby(list(group_cols), sort=False)["_hit"]
    out = grouped.agg(n_eligible="size", n_hit="sum").reset_index()
    out["frequency_percent"] = np.where(
        out["n_eligible"] > 0,
        100.0 * out["n_hit"] / out["n_eligible"],
        np.nan,
    )
    return out


def monthly_type_frequency(flags: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for month in range(1, 13):
        g = flags[flags["month"] == month]
        n = int(len(g))
        rec = {"month": month, "n_eligible": n}
        for kind in INVERSION_TYPES:
            n_hit = int(g[f"has_{kind}"].sum()) if n else 0
            rec[f"n_{kind}"] = n_hit
            rec[f"F_{kind}"] = 100.0 * n_hit / n if n else np.nan
        rec["n_any"] = int(g[[f"has_{k}" for k in INVERSION_TYPES]].any(axis=1).sum()) if n else 0
        rec["F_any"] = 100.0 * rec["n_any"] / n if n else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def year_month_frequency(flags: pd.DataFrame, col: str) -> pd.DataFrame:
    work = flags.copy()
    if col == "any":
        work["_hit"] = work[[f"has_{k}" for k in INVERSION_TYPES]].any(axis=1)
    else:
        work["_hit"] = work[col]
    pivot = (
        work.groupby(["year", "month"])["_hit"]
        .mean()
        .mul(100.0)
        .reset_index()
        .pivot(index="year", columns="month", values="_hit")
        .reindex(columns=range(1, 13))
        .sort_index()
    )
    return pivot


def unique_profile_bin_percent(
    layers: pd.DataFrame,
    profile_qc: pd.DataFrame,
    *,
    value_col: str,
    bin_edges: Sequence[float],
    by_month: bool = True,
) -> pd.DataFrame:
    """Повторяемость: доля eligible-профилей с ≥1 слоем в бине (не число слоёв)."""
    edges = _edges_with_overflow(bin_edges)
    eligible = profile_qc[profile_qc["eligible_article"]].copy()
    months = list(range(1, 13)) if by_month else [0]
    rows = []
    use = layers.dropna(subset=[value_col]).copy()
    lefts = [float(x) for x in edges[:-1]]
    rights = [float(x) for x in edges[1:]]
    for month in months:
        if by_month:
            elig_n = int((eligible["month"] == month).sum())
            chunk = use[use["month"] == month]
        else:
            elig_n = int(len(eligible))
            chunk = use
        vals = pd.to_numeric(chunk[value_col], errors="coerce") if not chunk.empty else pd.Series(dtype=float)
        for lo, hi in zip(lefts, rights):
            hi_cmp = hi if np.isfinite(hi) else 1e18
            in_bin = chunk.loc[(vals >= lo) & (vals < hi_cmp)] if not chunk.empty else chunk
            n_prof = int(in_bin["profile_id"].nunique()) if not in_bin.empty else 0
            n_layers = int(len(in_bin))
            rows.append(
                {
                    "month": month,
                    "bin_left": lo,
                    "bin_right": hi if np.isfinite(hi) else np.nan,
                    "n_layers": n_layers,
                    "n_profiles": n_prof,
                    "n_eligible": elig_n,
                    "frequency_percent": 100.0 * n_prof / elig_n if elig_n else np.nan,
                }
            )
    return pd.DataFrame(rows)


def type_summary(layers: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n_all = int(len(flags))
    for kind in INVERSION_TYPES:
        g = layers[layers["position_type"] == kind]
        n_prof = int(flags[f"has_{kind}"].sum()) if n_all else 0
        rows.append(
            {
                "position_type": kind,
                "n_layers": int(len(g)),
                "n_profiles": n_prof,
                "n_eligible": n_all,
                "frequency_percent": 100.0 * n_prof / n_all if n_all else np.nan,
                "median_base_agl_m": float(g["base_height_agl_m"].median()) if len(g) else np.nan,
                "median_top_agl_m": float(g["top_height_agl_m"].median()) if len(g) else np.nan,
                "median_depth_m": float(g["depth_m"].median()) if len(g) else np.nan,
                "median_delta_t_c": float(g["delta_t_c"].median()) if len(g) else np.nan,
                "median_gamma_c_100m": float(g["gamma_c_per_100m"].median()) if len(g) else np.nan,
                "q25_depth_m": float(g["depth_m"].quantile(0.25)) if len(g) else np.nan,
                "q75_depth_m": float(g["depth_m"].quantile(0.75)) if len(g) else np.nan,
                "q25_delta_t_c": float(g["delta_t_c"].quantile(0.25)) if len(g) else np.nan,
                "q75_delta_t_c": float(g["delta_t_c"].quantile(0.75)) if len(g) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def shared_abs_limit(*arrays: np.ndarray, q: float = 97.5) -> float:
    finite = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    return float(max(np.nanpercentile(np.abs(finite), q), 0.1))


def prepare_revision_tables(
    df: pd.DataFrame,
    profile_qc: pd.DataFrame,
    config: AnalysisConfig,
    layers: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    layers_raw = layers if layers is not None else compute_inversion_layers(df, profile_qc, config)
    layers_ok = valid_layers(layers_raw)
    flags = profile_type_flags(layers_ok, profile_qc)
    sfc = compute_sfc_level_gamma(df, profile_qc, config)
    local = compute_local_gammas(df, profile_qc, config)
    counts = profile_layer_counts(layers_ok, profile_qc)
    return {
        "layers": layers_ok,
        "layers_raw": layers_raw,
        "flags": flags,
        "sfc_gamma": sfc,
        "sfc_gamma_monthly": sfc_gamma_monthly(sfc),
        "sfc_gamma_monthly_by_cycle": sfc_gamma_monthly_by_cycle(sfc),
        "sfc_gamma_year_month": sfc_gamma_year_month(sfc),
        "local_gamma": local,
        "local_gamma_extreme_ge_15": filter_extreme_gamma_local(local),
        "sfc_gamma_extreme_ge_15": filter_extreme_gamma_sfc(sfc),
        "local_median_heatmap": month_height_gamma_heatmap(local, stat="median"),
        "local_ppos_heatmap": month_height_gamma_heatmap(local, stat="p_positive"),
        "local_pneg_heatmap": month_height_gamma_heatmap(local, stat="p_negative"),
        "layer_counts": counts,
        "monthly_type_frequency": monthly_type_frequency(flags),
        "type_summary": type_summary(layers_ok, flags),
        "bin_base_profile_percent": unique_profile_bin_percent(
            layers_ok, profile_qc, value_col="base_height_agl_m", bin_edges=config.layers.height_bin_edges_m,
        ),
        "bin_depth_profile_percent": unique_profile_bin_percent(
            layers_ok, profile_qc, value_col="depth_m", bin_edges=DEPTH_EDGES_M,
        ),
        "bin_top_profile_percent": unique_profile_bin_percent(
            layers_ok, profile_qc, value_col="top_height_agl_m", bin_edges=config.layers.height_bin_edges_m,
            by_month=False,
        ),
    }
