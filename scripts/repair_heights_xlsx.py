"""Починка высот в Excel/CSV: interp + baro + Φ→z."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gdex_bufr.profile_climate.export import PROFILES_LONG_COLUMNS  # noqa: E402
from gdex_bufr.profile_climate.height_fill import (  # noqa: E402
    ALDAN_TYPICAL_SURFACE_HPA,
    STATION_ELEVATION_M,
    fill_long_dataframe_heights,
)


def _qc_report(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    def _stats(df: pd.DataFrame) -> dict:
        h = df["height_m"] if "height_m" in df.columns else pd.Series(dtype=float)
        return {
            "rows": int(len(df)),
            "height_null": int(h.isna().sum()) if len(h) else None,
            "height_min": None if h.dropna().empty else float(h.min()),
            "height_max": None if h.dropna().empty else float(h.max()),
            "height_lt0": int((h.dropna() < 0).sum()) if len(h) else None,
            "height_gt20000": int((h.dropna() > 20000).sum()) if len(h) else None,
        }

    sources = {}
    if "height_source" in after.columns:
        sources = {str(k): int(v) for k, v in after["height_source"].value_counts(dropna=False).items()}
    return {
        "station_aldan_elevation_m": STATION_ELEVATION_M.get("31004"),
        "aldan_typical_surface_hpa": ALDAN_TYPICAL_SURFACE_HPA,
        "before": _stats(before),
        "after": _stats(after),
        "height_source_counts": sources,
    }


def repair_xlsx(path: Path, *, output: Path | None = None) -> dict:
    xlsx = pd.ExcelFile(path)
    long_df = pd.read_excel(xlsx, sheet_name="profiles_long")
    metrics_df = pd.read_excel(xlsx, sheet_name="profile_metrics")
    before = long_df.copy()
    filled = fill_long_dataframe_heights(long_df, metrics_df)

    # сохранить все исходные колонки + новые height_*
    cols = list(dict.fromkeys([*PROFILES_LONG_COLUMNS, *filled.columns]))
    for col in cols:
        if col not in filled.columns:
            filled[col] = None
    filled = filled[cols]

    out = output or path.with_name(path.stem + "_heights_fixed.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        filled.to_excel(writer, sheet_name="profiles_long", index=False)
        metrics_df.to_excel(writer, sheet_name="profile_metrics", index=False)
        for sheet in xlsx.sheet_names:
            if sheet in {"profiles_long", "profile_metrics"}:
                continue
            pd.read_excel(xlsx, sheet_name=sheet).to_excel(writer, sheet_name=sheet, index=False)

    report = _qc_report(before, filled)
    report["input"] = str(path)
    report["output"] = str(out)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Починить высоты в profile_climate XLSX")
    parser.add_argument(
        "--xlsx",
        default="gdex_outputs/результаты-алдан",
        help="Путь к .xlsx или к каталогу (берётся последний *_profile_climate_*.xlsx)",
    )
    parser.add_argument("--output", help="Куда сохранить исправленный xlsx")
    args = parser.parse_args()

    target = Path(args.xlsx)
    if target.is_dir():
        stamped = sorted(target.glob("*_profile_climate_*.xlsx"), key=lambda p: p.stat().st_mtime)
        if not stamped:
            raise SystemExit(f"В каталоге нет xlsx: {target}")
        target = stamped[-1]

    report = repair_xlsx(target, output=Path(args.output) if args.output else None)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
