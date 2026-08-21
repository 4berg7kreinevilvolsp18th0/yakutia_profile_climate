#!/usr/bin/env python3
"""Аудит данных и математики графиков статьи (ревизия 2026)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gdex_bufr.profile_climate.article_figures.config import load_yaml_config  # noqa: E402
from gdex_bufr.profile_climate.article_figures.data import build_profile_qc, load_profiles  # noqa: E402
from gdex_bufr.profile_climate.article_figures.metrics import profile_type_flags  # noqa: E402
from revision_2026.master_table import build_article_graphics_master_table  # noqa: E402
from revision_2026.metrics import (  # noqa: E402
    STANDARD_LEVELS_HPA,
    compute_local_gammas,
    compute_sfc_level_gamma,
    month_height_gamma_heatmap,
    prepare_revision_tables,
    valid_layers,
)


def _stats(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    return {
        "N": int(len(valid)),
        "NaN": int(s.isna().sum()),
        "mean": float(valid.mean()) if len(valid) else np.nan,
        "median": float(valid.median()) if len(valid) else np.nan,
        "min": float(valid.min()) if len(valid) else np.nan,
        "max": float(valid.max()) if len(valid) else np.nan,
        "q25": float(valid.quantile(0.25)) if len(valid) else np.nan,
        "q75": float(valid.quantile(0.75)) if len(valid) else np.nan,
    }


def _cycle_counts(profiles: pd.DataFrame) -> dict[str, int]:
    cy = profiles["cycle"].astype(str).str.zfill(2).str[-2:]
    return {
        "N_00": int((cy == "00").sum()),
        "N_12": int((cy == "12").sum()),
        "N_other": int((~cy.isin(["00", "12"])).sum()),
    }


def build_figure_audit_rows(
    layers: pd.DataFrame,
    profiles: pd.DataFrame,
    local: pd.DataFrame,
    sfc: pd.DataFrame,
    flags: pd.DataFrame,
) -> list[dict]:
    rows: list[dict] = []

    def add(folder: str, figure: str, metric: str, series: pd.Series, *, formula: str, unit: str, filt: str = "eligible"):
        st = _stats(series)
        row = {
            "folder": folder,
            "figure": figure,
            "metric": metric,
            "formula": formula,
            "unit": unit,
            "filter": filt,
            "period": f"{profiles['year'].min()}–{profiles['year'].max()}",
            **st,
        }
        row.update(_cycle_counts(profiles))
        row["N_profiles"] = int(profiles["profile_id"].nunique())
        row["N_layers"] = int(len(layers))
        rows.append(row)

    add(
        "01_thickness",
        "inversion_depth_vs_base_joint",
        "base_height_agl_m",
        layers["base_height_agl_m"],
        formula="AGL от нижнего уровня профиля (z0)",
        unit="m",
    )
    add(
        "01_thickness",
        "inversion_depth_vs_base_joint",
        "depth_m",
        layers["depth_m"],
        formula="top_height_agl_m - base_height_agl_m",
        unit="m",
    )

    for level in STANDARD_LEVELS_HPA:
        key = int(level)
        col = f"gamma_sfc_{key}"
        add(
            "02_gamma_sfc_P",
            "type03_gamma_annual_cycle_850_700_500",
            f"gamma_sfc_{key}",
            sfc[col],
            formula=f"100*(T_{key}-T_sfc)/(H_{key}-H_sfc), no extrapolation",
            unit="°C/100 m",
        )

    med = month_height_gamma_heatmap(local, stat="median")
    add(
        "03_gamma_local",
        "gamma_local_month_height_median",
        "gamma_local_median",
        med["value"],
        formula="median over intervals: 100*ΔT/Δz",
        unit="°C/100 m",
    )

    add(
        "03_gamma_local",
        "gamma_local_month_height_median",
        "gamma_local_intervals",
        local["gamma_local_c_100m"],
        formula="100*(T_{i+1}-T_i)/(z_{i+1}-z_i)",
        unit="°C/100 m",
    )

    add(
        "04_multilayer",
        "n_layers_histogram",
        "n_inversion_layers",
        profiles["n_layers"],
        formula="count v3 layers per eligible profile",
        unit="count",
    )

    add(
        "06_extra",
        "violin_depth_by_month",
        "depth_m",
        layers["depth_m"],
        formula="layer depth_m",
        unit="m",
    )

    add(
        "diagnostic",
        "qc_depth_histogram",
        "depth_m",
        layers["depth_m"],
        formula="same as 01/06 depth_m",
        unit="m",
    )

    # crosscheck gamma_850 diagnostic vs 02
    for folder, figure in (("02_gamma_sfc_P", "type03_gamma_annual_cycle_850_700_500"), ("diagnostic", "qc_eligible_counts")):
        add(folder, figure, "gamma_sfc_850", sfc["gamma_sfc_850"], formula="shared compute_sfc_level_gamma", unit="°C/100 m")

    return rows


def build_crosscheck(audit_rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(audit_rows)
    keys = df.groupby("metric")
    cross: list[dict] = []
    for metric, g in keys:
        if len(g) < 2:
            continue
        ref = g.iloc[0]
        for _, row in g.iloc[1:].iterrows():
            diff_n = row["N"] - ref["N"]
            cross.append(
                {
                    "metric": metric,
                    "folder_a": ref["folder"],
                    "figure_a": ref["figure"],
                    "N_a": ref["N"],
                    "folder_b": row["folder"],
                    "figure_b": row["figure"],
                    "N_b": row["N"],
                    "delta_N": diff_n,
                    "median_a": ref["median"],
                    "median_b": row["median"],
                    "reason": "identical filter expected" if diff_n == 0 else "CHECK filters/formula",
                }
            )
    return pd.DataFrame(cross)


def write_audit_md(rows: list[dict], path: Path) -> None:
    lines = ["# GRAPHICS DATA AUDIT\n", f"Строк метрик: {len(rows)}\n"]
    for r in rows:
        lines.append(
            f"## {r['folder']} / {r['figure']} / {r['metric']}\n"
            f"- formula: {r['formula']}\n"
            f"- unit: {r['unit']}\n"
            f"- N={r['N']}, median={r['median']}, q25={r['q25']}, q75={r['q75']}\n"
            f"- N00={r.get('N_00')}, N12={r.get('N_12')}\n"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit article graphics data")
    parser.add_argument("--input", required=True, help="profiles_long.csv")
    parser.add_argument("--config", default="article_figures_config.yaml")
    parser.add_argument("--output", default="revision_2026/output/audit")
    args = parser.parse_args()

    analysis, _style = load_yaml_config(args.config)
    df = load_profiles(args.input, station_id=analysis.station_id, cycles=analysis.cycles)
    qc = build_profile_qc(df, analysis)
    tables = prepare_revision_tables(df, qc, analysis)
    layers = tables["layers"]
    local = tables["local_gamma"]
    sfc = tables["sfc_gamma"]
    flags = tables["flags"]
    layers_mt, profiles_mt = build_article_graphics_master_table(df, qc, analysis, layers=layers)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    export_dir = out / "tables"
    export_dir.mkdir(exist_ok=True)
    layers_mt.to_csv(export_dir / "article_graphics_master_layers.csv", index=False, encoding="utf-8-sig")
    profiles_mt.to_csv(export_dir / "article_graphics_master_profiles.csv", index=False, encoding="utf-8-sig")

    rows = build_figure_audit_rows(layers, profiles_mt, local, sfc, flags)
    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(out / "GRAPHICS_DATA_AUDIT.csv", index=False, encoding="utf-8-sig")
    write_audit_md(rows, out / "GRAPHICS_DATA_AUDIT.md")

    cross = build_crosscheck(rows)
    cross.to_csv(out / "graphics_crosscheck.csv", index=False, encoding="utf-8-sig")

    # 03_05 source for gamma_local_month_height_median
    med = month_height_gamma_heatmap(local, stat="median")
    med.to_csv(out / "03_05_source_data.csv", index=False, encoding="utf-8-sig")

    print(f"Audit written to {out}")
    print(f"Layers: {len(layers)}, profiles: {len(profiles_mt)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
