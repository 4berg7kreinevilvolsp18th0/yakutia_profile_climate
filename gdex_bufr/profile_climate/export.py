"""Экспорт profile_climate в CSV, XLSX и JSON."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

PROFILES_LONG_COLUMNS = [
    "station_id",
    "station_name",
    "datetime_utc",
    "year",
    "month",
    "cycle",
    "profile_id",
    "level_index",
    "pressure_hpa",
    "temperature_c",
    "height_m",
    "source_file",
    "qc_flag",
]

PROFILE_METRICS_COLUMNS = [
    "profile_id",
    "station_id",
    "station_name",
    "datetime_utc",
    "year",
    "month",
    "cycle",
    "n_levels_total",
    "n_levels_to_500",
    "p_surface_hpa",
    "t_surface_c",
    "p_top_hpa",
    "t_top_c",
    "delta_t_top_surface_c",
    "inversion_detected",
    "inversion_top_pressure_hpa",
    "inversion_top_height_m",
    "inversion_top_temp_c",
    "inversion_delta_t_c",
    "profile_status",
    "source_file",
]


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})
    return path


def write_profiles_long_csv(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    return _write_csv(output_dir / "profiles_long.csv", rows, PROFILES_LONG_COLUMNS)


def write_profile_metrics_csv(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    return _write_csv(output_dir / "profile_metrics.csv", rows, PROFILE_METRICS_COLUMNS)


def write_monthly_summary(metrics_rows: list[dict[str, Any]], output_dir: Path) -> Path:
    grouped: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in metrics_rows:
        key = (row.get("station_id", ""), row.get("station_name", ""), int(row.get("year") or 0), int(row.get("month") or 0))
        grouped[key].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (station_id, station_name, year, month), rows in sorted(grouped.items()):
        good = [r for r in rows if r.get("profile_status") == "good"]
        inversions = [r for r in rows if r.get("inversion_detected")]
        summary_rows.append({
            "station_id": station_id,
            "station_name": station_name,
            "year": year,
            "month": month,
            "profiles_total": len(rows),
            "profiles_good": len(good),
            "profiles_with_inversion": len(inversions),
            "mean_delta_t_top_surface_c": _mean([r.get("delta_t_top_surface_c") for r in good]),
            "mean_inversion_delta_t_c": _mean([r.get("inversion_delta_t_c") for r in inversions]),
        })

    columns = [
        "station_id",
        "station_name",
        "year",
        "month",
        "profiles_total",
        "profiles_good",
        "profiles_with_inversion",
        "mean_delta_t_top_surface_c",
        "mean_inversion_delta_t_c",
    ]
    return _write_csv(output_dir / "monthly_summary.csv", summary_rows, columns)


def write_station_summary(metrics_rows: list[dict[str, Any]], output_dir: Path) -> Path:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in metrics_rows:
        key = (row.get("station_id", ""), row.get("station_name", ""))
        grouped[key].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (station_id, station_name), rows in sorted(grouped.items()):
        good = [r for r in rows if r.get("profile_status") == "good"]
        inversions = [r for r in rows if r.get("inversion_detected")]
        summary_rows.append({
            "station_id": station_id,
            "station_name": station_name,
            "profiles_total": len(rows),
            "profiles_good": len(good),
            "profiles_with_inversion": len(inversions),
            "years_covered": len({r.get("year") for r in rows if r.get("year")}),
            "months_covered": len({(r.get("year"), r.get("month")) for r in rows if r.get("year") and r.get("month")}),
        })

    columns = [
        "station_id",
        "station_name",
        "profiles_total",
        "profiles_good",
        "profiles_with_inversion",
        "years_covered",
        "months_covered",
    ]
    return _write_csv(output_dir / "station_summary.csv", summary_rows, columns)


def write_summary_json(
    metrics_rows: list[dict[str, Any]],
    long_rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    config_info: dict[str, Any] | None = None,
) -> Path:
    payload = {
        "config": config_info or {},
        "profiles_total": len(metrics_rows),
        "levels_total": len(long_rows),
        "profiles_good": sum(1 for r in metrics_rows if r.get("profile_status") == "good"),
        "profiles_with_inversion": sum(1 for r in metrics_rows if r.get("inversion_detected")),
        "stations": _station_counts(metrics_rows),
    }
    path = output_dir / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_xlsx_exports(long_rows: list[dict[str, Any]], metrics_rows: list[dict[str, Any]], output_dir: Path) -> Path | None:
    if pd is None:
        return None
    path = output_dir / "profile_climate.xlsx"
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(long_rows, columns=PROFILES_LONG_COLUMNS).to_excel(writer, sheet_name="profiles_long", index=False)
        pd.DataFrame(metrics_rows, columns=PROFILE_METRICS_COLUMNS).to_excel(writer, sheet_name="profile_metrics", index=False)
    return path


def export_all(
    long_rows: list[dict[str, Any]],
    metrics_rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    config_info: dict[str, Any] | None = None,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    paths = {
        "profiles_long": str(write_profiles_long_csv(long_rows, output_dir)),
        "profile_metrics": str(write_profile_metrics_csv(metrics_rows, output_dir)),
        "monthly_summary": str(write_monthly_summary(metrics_rows, output_dir)),
        "station_summary": str(write_station_summary(metrics_rows, output_dir)),
        "summary_json": str(write_summary_json(metrics_rows, long_rows, output_dir, config_info=config_info)),
    }
    xlsx_path = write_xlsx_exports(long_rows, metrics_rows, output_dir)
    if xlsx_path:
        paths["xlsx"] = str(xlsx_path)
    return paths


def _mean(values: list[Any]) -> float | None:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return round(sum(nums) / len(nums), 3)


def _station_counts(metrics_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in metrics_rows:
        key = f"{row.get('station_id')}:{row.get('station_name', '')}"
        counts[key] += 1
    return dict(counts)
