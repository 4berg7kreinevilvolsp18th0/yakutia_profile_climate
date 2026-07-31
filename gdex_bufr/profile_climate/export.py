"""Экспорт profile_climate в CSV, XLSX и JSON."""
from __future__ import annotations

import csv
import json
import logging
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

logger = logging.getLogger(__name__)

PROFILES_LONG_COLUMNS = [
    "station_id",
    "station_name",
    "datetime_utc",
    "year",
    "month",
    "cycle",
    "profile_id",
    "subset_index",
    "latitude_deg",
    "longitude_deg",
    "data_status",
    "data_status_reason",
    "level_index",
    "SEQ",
    "VSIG",
    "vertical_significance_code",
    "replication_index",
    "PRES",
    "pressure_hpa",
    "GEOPOT",
    "geopotential_m2s2",
    "FLVL",
    "geopotential_height_m",
    "height_m",
    "AIR",
    "air_temperature_c",
    "temperature_c",
    "DEW-",
    "dew_point_temperature_c",
    "REL",
    "relative_humidity_percent",
    "WIND",
    "wind_direction_deg",
    "WIND.1",
    "wind_speed",
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
    "subset_index",
    "latitude_deg",
    "longitude_deg",
    "data_status",
    "data_status_reason",
    "table_edition",
    "n_pressure_raw",
    "n_temp_raw",
    "n_wind_raw",
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

# Как в gdex_bufr.xlsx_export.LEVEL_COLUMNS + ключи стыковки с климатическим слоем
DECODED_LEVEL_BASE_COLUMNS = [
    "profile_id",
    "station_name",
    "source_file",
    "station_id",
    "subset_index",
    "report_datetime_utc",
    "data_status",
    "REC",
    "OBS",
    "REPORT TIME",
    "WMO/STATION/SATELLITE ID",
    "LATI-",
    "LONGI-",
    "STN",
    "SEQ",
    "VSIG",
    "PRES",
    "GEOPOT",
    "FLVL",
    "AIR",
    "DEW-",
    "REL",
    "WIND",
    "WIND.1",
    "replication_index",
    "pressure_hpa",
    "geopotential_height_m",
    "geopotential_m2s2",
    "air_temperature_c",
    "dew_point_temperature_c",
    "wind_direction_deg",
    "wind_speed",
    "relative_humidity_percent",
    "vertical_significance_code",
]

# Type-суффиксы добавляются лениво через field_types.type_suffix_columns()
def _decoded_level_columns() -> list[str]:
    from gdex_bufr.profile_climate.field_types import type_suffix_columns

    return list(DECODED_LEVEL_BASE_COLUMNS) + type_suffix_columns()


DECODED_LEVEL_COLUMNS = _decoded_level_columns()

DEBUFR_ELEMENT_COLUMNS = [
    "profile_id",
    "station_name",
    "source_file",
    "station_id",
    "subset_index",
    "report_datetime_utc",
    "seq",
    "fxy",
    "name",
    "value",
    "value_text",
    "unit",
    "kind",
    "scale",
    "reference",
    "nbits",
]


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> Path:
    """Атомарная запись CSV с retry — защита от Windows Errno 22 / блокировок."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})

    last_error: Exception | None = None
    try:
        for attempt in range(1, 6):
            try:
                tmp_path.replace(path)
                return path
            except OSError as exc:
                last_error = exc
                logger.warning(
                    "Не удалось заменить %s (попытка %s/5): %s",
                    path.name,
                    attempt,
                    exc,
                )
                time.sleep(0.5 * attempt)

        assert last_error is not None
        raise last_error
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def append_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> Path:
    """Дописывает строки в CSV; пишет header, если файла ещё нет."""
    if not rows:
        return Path(path)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})
    return path


def write_profiles_long_csv(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    return _write_csv(output_dir / "profiles_long.csv", rows, PROFILES_LONG_COLUMNS)


def write_profile_metrics_csv(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    return _write_csv(output_dir / "profile_metrics.csv", rows, PROFILE_METRICS_COLUMNS)


def write_decoded_levels_csv(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    return _write_csv(output_dir / "decoded_levels.csv", rows, DECODED_LEVEL_COLUMNS)


def write_debufr_elements_csv(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    return _write_csv(output_dir / "debufr_elements.csv", rows, DEBUFR_ELEMENT_COLUMNS)


def write_field_types_csv(output_dir: Path, rows: list[dict[str, Any]] | None = None) -> Path:
    from gdex_bufr.profile_climate.field_types import FIELD_TYPE_COLUMNS, build_field_types_rows

    return _write_csv(output_dir / "field_types.csv", rows if rows is not None else build_field_types_rows(), FIELD_TYPE_COLUMNS)


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


def write_xlsx_exports(
    long_rows: list[dict[str, Any]],
    metrics_rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    decoded_rows: list[dict[str, Any]] | None = None,
    element_rows: list[dict[str, Any]] | None = None,
    field_type_rows: list[dict[str, Any]] | None = None,
) -> Path | None:
    if pd is None:
        return None
    from gdex_bufr.profile_climate.field_types import FIELD_TYPE_COLUMNS, build_field_types_rows

    path = output_dir / "profile_climate.xlsx"
    path.parent.mkdir(parents=True, exist_ok=True)
    types_rows = field_type_rows if field_type_rows is not None else build_field_types_rows()
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(long_rows, columns=PROFILES_LONG_COLUMNS).to_excel(writer, sheet_name="profiles_long", index=False)
        pd.DataFrame(metrics_rows, columns=PROFILE_METRICS_COLUMNS).to_excel(writer, sheet_name="profile_metrics", index=False)
        if decoded_rows is not None:
            pd.DataFrame(decoded_rows, columns=DECODED_LEVEL_COLUMNS).to_excel(
                writer, sheet_name="decoded_levels", index=False
            )
        if element_rows is not None:
            pd.DataFrame(element_rows, columns=DEBUFR_ELEMENT_COLUMNS).to_excel(
                writer, sheet_name="debufr_elements", index=False
            )
        pd.DataFrame(types_rows, columns=FIELD_TYPE_COLUMNS).to_excel(
            writer, sheet_name="field_types", index=False
        )
    return path


def export_checkpoint(
    long_rows: list[dict[str, Any]],
    metrics_rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    config_info: dict[str, Any] | None = None,
    decoded_rows: list[dict[str, Any]] | None = None,
    element_rows: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Быстрое промежуточное сохранение без XLSX."""
    output_dir = Path(output_dir)
    paths = {
        "profiles_long": str(write_profiles_long_csv(long_rows, output_dir)),
        "profile_metrics": str(write_profile_metrics_csv(metrics_rows, output_dir)),
        "monthly_summary": str(write_monthly_summary(metrics_rows, output_dir)),
        "station_summary": str(write_station_summary(metrics_rows, output_dir)),
        "summary_json": str(write_summary_json(metrics_rows, long_rows, output_dir, config_info=config_info)),
        "field_types": str(write_field_types_csv(output_dir)),
    }
    if decoded_rows is not None:
        paths["decoded_levels"] = str(write_decoded_levels_csv(decoded_rows, output_dir))
    if element_rows is not None:
        paths["debufr_elements"] = str(write_debufr_elements_csv(element_rows, output_dir))
    return paths


def export_all(
    long_rows: list[dict[str, Any]],
    metrics_rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    config_info: dict[str, Any] | None = None,
    decoded_rows: list[dict[str, Any]] | None = None,
    element_rows: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    paths = {
        "profiles_long": str(write_profiles_long_csv(long_rows, output_dir)),
        "profile_metrics": str(write_profile_metrics_csv(metrics_rows, output_dir)),
        "monthly_summary": str(write_monthly_summary(metrics_rows, output_dir)),
        "station_summary": str(write_station_summary(metrics_rows, output_dir)),
        "summary_json": str(write_summary_json(metrics_rows, long_rows, output_dir, config_info=config_info)),
        "field_types": str(write_field_types_csv(output_dir)),
    }
    if decoded_rows is not None:
        paths["decoded_levels"] = str(write_decoded_levels_csv(decoded_rows, output_dir))
    if element_rows is not None:
        paths["debufr_elements"] = str(write_debufr_elements_csv(element_rows, output_dir))
    xlsx_path = write_xlsx_exports(
        long_rows,
        metrics_rows,
        output_dir,
        decoded_rows=decoded_rows,
        element_rows=element_rows,
    )
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
