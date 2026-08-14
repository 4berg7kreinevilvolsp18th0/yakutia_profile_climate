"""Batch sensitivity gap-v3: сетка max_embedded_gap_m и опциональные лимиты склейки."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gdex_bufr.profile_climate.config import load_profile_climate_config  # noqa: E402

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "compute_inversion_v3", ROOT / "scripts" / "compute_inversion_v3.py",
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_mod)
compute_v3_for_long = _mod.compute_v3_for_long


def main() -> int:
    parser = argparse.ArgumentParser(description="Sensitivity runner gap-v3")
    parser.add_argument("--config", default="profile_climate_config.yaml")
    parser.add_argument(
        "--profiles-long",
        default="gdex_outputs/far_east/stations/aldan/profiles_long.csv",
    )
    parser.add_argument(
        "--output",
        default="gdex_outputs/far_east/stations/aldan/v3/sensitivity_gap.csv",
    )
    parser.add_argument("--gaps", default="60,80,100,120,140")
    parser.add_argument("--max-total-embedded-gap-m", type=float, default=None)
    parser.add_argument("--max-gap-fraction", type=float, default=None)
    parser.add_argument("--limit-profiles", type=int, default=None)
    args = parser.parse_args()

    cfg = load_profile_climate_config(args.config)
    v3 = cfg.v3_detect_kwargs()
    long_df = pd.read_csv(args.profiles_long)
    if args.limit_profiles:
        keep = long_df["profile_id"].drop_duplicates().head(args.limit_profiles)
        long_df = long_df[long_df["profile_id"].isin(keep)]

    gaps = [float(x) for x in args.gaps.split(",") if x.strip()]
    rows: list[dict] = []
    for gap in gaps:
        _, summary_rows, _ = compute_v3_for_long(
            long_df,
            max_embedded_gap_m=gap,
            min_strength_c=v3["min_strength_c"],
            min_depth_m=v3["min_depth_m"],
            he_threshold_m=v3["he_threshold_m"],
            max_gap_drop_c=v3["max_gap_drop_c"],
            pressure_top_hpa=cfg.pressure_top_hpa,
            surface_tolerance_m=v3["surface_tolerance_m"],
            max_total_embedded_gap_m=args.max_total_embedded_gap_m,
            max_gap_fraction=args.max_gap_fraction,
        )
        n = len(summary_rows)
        with_layers = sum(1 for s in summary_rows if s["n_inversion_layers"] > 0)
        rows.append({
            "max_embedded_gap_m": gap,
            "min_strength_c": v3["min_strength_c"],
            "he_threshold_m": v3["he_threshold_m"],
            "max_total_embedded_gap_m": args.max_total_embedded_gap_m,
            "max_gap_fraction": args.max_gap_fraction,
            "n_profiles": n,
            "n_profiles_with_layers": with_layers,
            "n_G": sum(1 for s in summary_rows if s["has_G"]),
            "n_E": sum(1 for s in summary_rows if s["has_E"]),
            "n_HE": sum(1 for s in summary_rows if s["has_HE"]),
            "n_layers": sum(int(s["n_inversion_layers"]) for s in summary_rows),
        })
        print(f"gap={gap}: layers={rows[-1]['n_layers']} profiles={with_layers}/{n}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
