"""Офлайн-расчёт gap-v3 инверсий из profiles_long.csv (без повторного decode)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gdex_bufr.profile_climate.export import (  # noqa: E402
    write_comparison_v2_v3_csv,
    write_inversion_layers_v3_csv,
    write_profile_inversion_summary_v3_csv,
)
from gdex_bufr.profile_climate.inversion_layers import (  # noqa: E402
    detect_inversion_layers_gap_v3,
    layers_to_dashboard_payload,
    summarize_inversion_layers,
)


def _finite(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values


def compute_v3_for_long(
    long_df: pd.DataFrame,
    *,
    max_embedded_gap_m: float,
    min_strength_c: float,
    min_depth_m: float | None,
    he_threshold_m: float,
    max_gap_drop_c: float | None,
    pressure_top_hpa: float | None = 500.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Возвращает (layer_rows, summary_rows, layers_by_profile для dashboard)."""
    layer_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    by_profile: dict[str, list[dict[str, Any]]] = {}

    required = {"profile_id", "height_m", "temperature_c"}
    missing = required - set(long_df.columns)
    if missing:
        raise ValueError(f"profiles_long не хватает колонок: {sorted(missing)}")

    for profile_id, group in long_df.groupby("profile_id", sort=False):
        pid = str(profile_id)
        g = group
        if pressure_top_hpa is not None and "pressure_hpa" in g.columns:
            p_all = pd.to_numeric(g["pressure_hpa"], errors="coerce")
            g = g.loc[p_all.isna() | (p_all >= float(pressure_top_hpa))]

        z = _finite(g["height_m"])
        t = _finite(g["temperature_c"])
        if "pressure_hpa" in g.columns:
            p = _finite(g["pressure_hpa"])
        else:
            p = np.full_like(z, np.nan)

        mask = np.isfinite(z) & np.isfinite(t)
        z, t, p = z[mask], t[mask], p[mask]
        if z.size < 2:
            summary_rows.append(summarize_inversion_layers(pid, [], z0=0.0))
            by_profile[pid] = []
            continue

        layers = detect_inversion_layers_gap_v3(
            z,
            t,
            p,
            max_embedded_gap_m=max_embedded_gap_m,
            min_strength_c=min_strength_c,
            min_depth_m=min_depth_m,
            he_threshold_m=he_threshold_m,
            max_gap_drop_c=max_gap_drop_c,
        )
        order = np.argsort(z, kind="mergesort")
        z0 = float(z[order][0])
        for i, ly in enumerate(layers):
            layer_rows.append(ly.as_row(profile_id=pid, layer_index=i, z0=z0))
        summary_rows.append(summarize_inversion_layers(pid, layers, z0=z0))
        by_profile[pid] = layers_to_dashboard_payload(layers, z0=z0)

    return layer_rows, summary_rows, by_profile


def build_comparison_rows(
    summary_rows: list[dict[str, Any]],
    metrics_df: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    if metrics_df is None or metrics_df.empty:
        return []
    metrics_map = {
        str(row.profile_id): row
        for row in metrics_df.itertuples(index=False)
    }
    out: list[dict[str, Any]] = []
    for s in summary_rows:
        pid = str(s["profile_id"])
        m = metrics_map.get(pid)
        out.append({
            "profile_id": pid,
            "inversion_detected_v2": bool(getattr(m, "inversion_detected", False)) if m else False,
            "inversion_candidate_v2": bool(getattr(m, "inversion_candidate", False)) if m else False,
            "inversion_quality_v2": str(getattr(m, "inversion_quality", "") or "") if m else "",
            "inversion_delta_t_c_v2": getattr(m, "inversion_delta_t_c", None) if m else None,
            "n_inversion_layers_v3": s["n_inversion_layers"],
            "has_G_v3": s["has_G"],
            "has_E_v3": s["has_E"],
            "has_HE_v3": s["has_HE"],
            "pattern_v3": s["pattern"],
            "strongest_delta_t_c_v3": s["strongest_delta_t_c"],
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gap-v3 инверсии из profiles_long (параллельно legacy v2)",
    )
    parser.add_argument(
        "--station-dir",
        default="",
        help="Каталог станции (profiles_long.csv + profile_metrics.csv); v3 пишется в <dir>/v3",
    )
    parser.add_argument("--profiles-long", default="", help="Путь к profiles_long.csv")
    parser.add_argument("--output-dir", default="", help="Каталог для CSV v3")
    parser.add_argument("--metrics", default="", help="profile_metrics.csv для comparison_v2_v3")
    parser.add_argument("--max-embedded-gap-m", type=float, default=100.0)
    parser.add_argument("--min-strength-c", type=float, default=0.3)
    parser.add_argument("--min-depth-m", type=float, default=None)
    parser.add_argument("--he-threshold-m", type=float, default=250.0)
    parser.add_argument(
        "--max-gap-drop-c",
        type=float,
        default=None,
        help="Экспериментальный порог падения T в gap (по умолчанию выкл.)",
    )
    parser.add_argument("--pressure-top-hpa", type=float, default=500.0)
    parser.add_argument(
        "--params-json",
        default="",
        help="Дополнительно записать параметры прогона в JSON",
    )
    args = parser.parse_args()

    if not args.station_dir and not (args.profiles_long and args.output_dir):
        from gdex_bufr.profile_climate.paths import catalog_station_dir

        args.station_dir = str(catalog_station_dir())

    if args.station_dir:
        station_dir = Path(args.station_dir)
        long_path = Path(args.profiles_long) if args.profiles_long else station_dir / "profiles_long.csv"
        out_dir = Path(args.output_dir) if args.output_dir else station_dir / "v3"
        if not args.metrics:
            metrics_candidate = station_dir / "profile_metrics.csv"
            if metrics_candidate.exists():
                args.metrics = str(metrics_candidate)
    elif args.profiles_long and args.output_dir:
        long_path = Path(args.profiles_long)
        out_dir = Path(args.output_dir)
    else:
        parser.error("Укажите --station-dir или пару --profiles-long и --output-dir")
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    long_df = pd.read_csv(long_path)
    layer_rows, summary_rows, _ = compute_v3_for_long(
        long_df,
        max_embedded_gap_m=args.max_embedded_gap_m,
        min_strength_c=args.min_strength_c,
        min_depth_m=args.min_depth_m,
        he_threshold_m=args.he_threshold_m,
        max_gap_drop_c=args.max_gap_drop_c,
        pressure_top_hpa=args.pressure_top_hpa,
    )

    layers_path = write_inversion_layers_v3_csv(layer_rows, out_dir)
    summary_path = write_profile_inversion_summary_v3_csv(summary_rows, out_dir)

    metrics_df = None
    comparison_path = None
    if args.metrics:
        metrics_path = Path(args.metrics)
        if metrics_path.exists():
            metrics_df = pd.read_csv(metrics_path)
            cmp_rows = build_comparison_rows(summary_rows, metrics_df)
            comparison_path = write_comparison_v2_v3_csv(cmp_rows, out_dir)

    params = {
        "method": "gap_v3",
        "max_embedded_gap_m": args.max_embedded_gap_m,
        "min_strength_c": args.min_strength_c,
        "min_depth_m": args.min_depth_m,
        "he_threshold_m": args.he_threshold_m,
        "max_gap_drop_c": args.max_gap_drop_c,
        "pressure_top_hpa": args.pressure_top_hpa,
        "profiles_long": str(long_path.resolve()),
        "n_profiles": len(summary_rows),
        "n_layers": len(layer_rows),
        "n_profiles_with_layers": sum(1 for s in summary_rows if s["n_inversion_layers"] > 0),
    }
    params_path = Path(args.params_json) if args.params_json else out_dir / "inversion_v3_params.json"
    params_path.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"layers: {layers_path} ({len(layer_rows)} rows)")
    print(f"summary: {summary_path} ({len(summary_rows)} rows)")
    if comparison_path is not None:
        print(f"comparison: {comparison_path}")
    print(f"params: {params_path}")
    print(
        f"profiles_with_layers={params['n_profiles_with_layers']} / {params['n_profiles']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
