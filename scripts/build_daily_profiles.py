"""Сборка профилей наблюдений (зондов) для интерактивного дашборда."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gdex_bufr.meteo_parser_bridge import geopotential_to_height_m  # noqa: E402
from gdex_bufr.profile_climate.obs_qc import (  # noqa: E402
    clean_observation_levels,
    interp_on_pressure_grid,
)

GRID_POINTS = 40
MAX_SURFACE_PRESSURE_HPA = 1000.0
PRESSURE_TOP_HPA = 500.0
PLOT_MIN_LEVELS = 3
SCHEMA = "observations_v1"
LEVEL_MODES = ("raw", "clean")

DEFAULT_DIR = Path("gdex_outputs") / "результаты-алдан"
DEFAULT_LONG_CSV = DEFAULT_DIR / "profiles_long.csv"
DEFAULT_METRICS_CSV = DEFAULT_DIR / "profile_metrics.csv"
DEFAULT_OUTPUT = DEFAULT_DIR / "daily_profiles.json"


def _day_key(dt: str) -> str:
    parsed = datetime.fromisoformat(str(dt).replace("Z", "+00:00"))
    return parsed.date().isoformat()


def resolve_xlsx(path: Path | None, search_dir: Path) -> Path | None:
    """Явный --xlsx или последний aldan_profile_climate_*.xlsx / profile_climate.xlsx."""
    if path is not None:
        return path if path.exists() else None
    if not search_dir.exists():
        return None
    stamped = sorted(search_dir.glob("*_profile_climate_*.xlsx"), key=lambda p: p.stat().st_mtime)
    if stamped:
        return stamped[-1]
    plain = search_dir / "profile_climate.xlsx"
    return plain if plain.exists() else None


def load_long_and_metrics(
    long_csv: Path,
    metrics_csv: Path,
    *,
    xlsx: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """CSV если есть, иначе листы profiles_long / profile_metrics из Excel."""
    if long_csv.exists() and metrics_csv.exists():
        return (
            pd.read_csv(long_csv, low_memory=False),
            pd.read_csv(metrics_csv, low_memory=False),
            f"csv:{long_csv}|{metrics_csv}",
        )

    xlsx_path = resolve_xlsx(xlsx, long_csv.parent if long_csv.parent.exists() else DEFAULT_DIR)
    if xlsx_path is None:
        raise FileNotFoundError(
            "Нет CSV (profiles_long / profile_metrics) и нет Excel. "
            f"Искали CSV: {long_csv}, {metrics_csv}. "
            "Положите xlsx в ту же папку или укажите --xlsx PATH."
        )

    long_df = pd.read_excel(xlsx_path, sheet_name="profiles_long")
    metrics_df = pd.read_excel(xlsx_path, sheet_name="profile_metrics")
    return long_df, metrics_df, f"xlsx:{xlsx_path}"


def _series_to_levels(group: pd.DataFrame) -> list[dict[str, Any]]:
    levels: list[dict[str, Any]] = []
    has_geopot = "geopotential_m2s2" in group.columns
    for row in group.itertuples(index=False):
        height = None if pd.isna(row.height_m) else float(row.height_m)
        if height is None and has_geopot:
            geo = getattr(row, "geopotential_m2s2", None)
            if geo is not None and not pd.isna(geo):
                height = round(geopotential_to_height_m(float(geo)), 1)
        levels.append({
            "pressure_hpa": float(row.pressure_hpa),
            "temperature_c": float(row.temperature_c),
            "height_m": height,
        })
    return levels


def _obs_arrays(
    levels: list[dict[str, Any]],
) -> tuple[list[float | None], list[float], list[float]]:
    heights = [
        None if lv.get("height_m") is None else round(float(lv["height_m"]), 1)
        for lv in levels
    ]
    pressures = [round(float(lv["pressure_hpa"]), 2) for lv in levels]
    temps = [round(float(lv["temperature_c"]), 3) for lv in levels]
    return heights, pressures, temps


def _day_mean_on_pressure(
    observations: list[dict[str, Any]],
    *,
    grid_points: int,
) -> dict[str, Any] | None:
    series_p: list[np.ndarray] = []
    series_t: list[np.ndarray] = []
    series_h: list[np.ndarray] = []
    for obs in observations:
        p = np.asarray(obs["pressure_hpa"], dtype=float)
        t = np.asarray(
            [np.nan if v is None else v for v in obs["temperature_c"]],
            dtype=float,
        )
        h = np.asarray(obs["heights_m"], dtype=float)
        valid = ~np.isnan(t) & ~np.isnan(p) & ~np.isnan(h)
        if valid.sum() < 2:
            continue
        series_p.append(p[valid])
        series_t.append(t[valid])
        series_h.append(h[valid])
    if not series_p:
        return None

    p_lo = min(float(p.min()) for p in series_p)
    p_hi = max(float(p.max()) for p in series_p)
    if p_hi <= p_lo:
        return None
    grid = np.linspace(p_hi, p_lo, grid_points)
    stacked_t = np.vstack([interp_on_pressure_grid(p, t, grid) for p, t in zip(series_p, series_t)])
    stacked_h = np.vstack([interp_on_pressure_grid(p, h, grid) for p, h in zip(series_p, series_h)])
    mean_t = np.nanmean(stacked_t, axis=0)
    mean_h = np.nanmean(stacked_h, axis=0)
    return {
        "pressure_hpa": [round(float(x), 2) for x in grid],
        "heights_m": [None if np.isnan(x) else round(float(x), 1) for x in mean_h],
        "temperature_c": [None if np.isnan(x) else round(float(x), 3) for x in mean_t],
    }


def build_daily_profiles(
    long_csv: Path,
    metrics_csv: Path,
    *,
    xlsx: Path | None = None,
    pressure_top_hpa: float = PRESSURE_TOP_HPA,
    max_surface_pressure_hpa: float = MAX_SURFACE_PRESSURE_HPA,
    plot_min_levels: int | None = None,
    grid_points: int = GRID_POINTS,
    level_mode: str = "raw",
) -> dict[str, Any]:
    if level_mode not in LEVEL_MODES:
        raise ValueError(f"level_mode должен быть одним из {LEVEL_MODES}: {level_mode}")
    min_levels = (1 if level_mode == "raw" else PLOT_MIN_LEVELS) if plot_min_levels is None else plot_min_levels

    long_df, metrics_df, source = load_long_and_metrics(long_csv, metrics_csv, xlsx=xlsx)
    print(f"Источник таблиц: {source}")

    long_df = long_df.dropna(subset=["temperature_c", "pressure_hpa"])
    long_df = long_df[
        (long_df["pressure_hpa"] <= max_surface_pressure_hpa)
        & (long_df["pressure_hpa"] >= pressure_top_hpa)
    ]

    metrics_map = {
        str(row.profile_id): row
        for row in metrics_df.itertuples(index=False)
    }

    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    profiles_with_levels: set[str] = set()

    for profile_id, group in long_df.groupby("profile_id"):
        profile_id = str(profile_id)
        profiles_with_levels.add(profile_id)
        metric = metrics_map.get(profile_id)
        status = getattr(metric, "profile_status", None) if metric is not None else None
        if level_mode == "clean" and status in {
            "no_temp", "bad_pressure", "duplicate_levels", "no_surface_level",
        }:
            continue

        levels = _series_to_levels(group)
        if level_mode == "clean":
            levels = clean_observation_levels(
                levels,
                pressure_top_hpa=pressure_top_hpa,
                max_surface_pressure_hpa=max_surface_pressure_hpa,
            )
        if len(levels) < min_levels:
            continue

        dt = str(group["datetime_utc"].iloc[0])
        try:
            day = _day_key(dt)
        except ValueError:
            continue

        cycle = str(group["cycle"].iloc[0]).zfill(2)[-2:]
        heights, pressures, temps = _obs_arrays(levels)

        t_surface = None
        inversion = False
        if metric is not None:
            t_s = getattr(metric, "t_surface_c", None)
            if t_s is not None and not (isinstance(t_s, float) and np.isnan(t_s)):
                t_surface = round(float(t_s), 3)
            inversion = bool(getattr(metric, "inversion_detected", False))
            status = str(getattr(metric, "profile_status", "") or "")
        if t_surface is None and temps:
            t_surface = temps[0]

        by_day[day].append({
            "profile_id": profile_id,
            "datetime_utc": dt,
            "cycle": cycle,
            "heights_m": heights,
            "pressure_hpa": pressures,
            "temperature_c": temps,
            "n_levels": len(levels),
            "t_surface_c": t_surface,
            "inversion_detected": inversion,
            "profile_status": status or "",
        })

    if level_mode == "raw":
        for metric in metrics_df.itertuples(index=False):
            profile_id = str(metric.profile_id)
            if profile_id in profiles_with_levels:
                continue
            dt = str(getattr(metric, "datetime_utc", ""))
            try:
                day = _day_key(dt)
            except ValueError:
                continue
            t_surface = getattr(metric, "t_surface_c", None)
            if t_surface is not None and pd.isna(t_surface):
                t_surface = None
            inversion = getattr(metric, "inversion_detected", False)
            by_day[day].append({
                "profile_id": profile_id,
                "datetime_utc": dt,
                "cycle": str(getattr(metric, "cycle", "")).zfill(2)[-2:],
                "heights_m": [],
                "pressure_hpa": [],
                "temperature_c": [],
                "n_levels": 0,
                "t_surface_c": None if t_surface is None else round(float(t_surface), 3),
                "inversion_detected": False if pd.isna(inversion) else bool(inversion),
                "profile_status": str(getattr(metric, "profile_status", "") or ""),
                "missing_levels": True,
            })

    months: dict[str, dict[str, Any]] = {}
    n_observations = 0
    for day, observations in sorted(by_day.items()):
        observations.sort(key=lambda o: (o["datetime_utc"], o["cycle"], o["profile_id"]))
        month_key = day[:7]
        day_mean = _day_mean_on_pressure(observations, grid_points=grid_points)
        t_surfaces = [o["t_surface_c"] for o in observations if o.get("t_surface_c") is not None]
        months.setdefault(month_key, {"days": []})
        months[month_key]["days"].append({
            "date": day,
            "n_profiles": len(observations),
            "n_good": sum(1 for o in observations if o.get("profile_status") == "good"),
            "inversion_detected": any(o.get("inversion_detected") for o in observations),
            "t_surface_c": (
                round(float(np.mean(t_surfaces)), 3) if t_surfaces else None
            ),
            "observations": observations,
            "day_mean": day_mean,
        })
        n_observations += len(observations)

    for month_key in months:
        months[month_key]["days"].sort(key=lambda d: d["date"])

    station_name = str(long_df["station_name"].iloc[0]) if len(long_df) else ""
    station_id = str(long_df["station_id"].iloc[0]) if len(long_df) else ""

    return {
        "schema": SCHEMA,
        "station_id": station_id,
        "station_name": station_name,
        "source_tables": source,
        "pressure_top_hpa": pressure_top_hpa,
        "max_surface_pressure_hpa": max_surface_pressure_hpa,
        "level_mode": level_mode,
        "plot_min_levels": min_levels,
        "grid_points": grid_points,
        "n_days": sum(len(m["days"]) for m in months.values()),
        "n_observations": n_observations,
        "n_levels": sum(
            obs["n_levels"]
            for month in months.values()
            for day in month["days"]
            for obs in day["observations"]
        ),
        "months": months,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Собрать daily_profiles.json (observations_v1)")
    parser.add_argument("--long", default=str(DEFAULT_LONG_CSV))
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS_CSV))
    parser.add_argument(
        "--xlsx",
        help="Excel с листами profiles_long и profile_metrics "
             "(если CSV нет — берётся автоматически из папки --long)",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--level-mode",
        choices=LEVEL_MODES,
        default="raw",
        help="raw: сохранить все уровни без QC; clean: прежняя предварительная очистка",
    )
    parser.add_argument(
        "--min-levels",
        type=int,
        help="Минимум уровней (по умолчанию: raw=1, clean=3)",
    )
    args = parser.parse_args()

    payload = build_daily_profiles(
        Path(args.long),
        Path(args.metrics),
        xlsx=Path(args.xlsx) if args.xlsx else None,
        level_mode=args.level_mode,
        plot_min_levels=args.min_levels,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output": str(out),
        "schema": payload["schema"],
        "source_tables": payload.get("source_tables"),
        "n_days": payload["n_days"],
        "n_observations": payload["n_observations"],
        "n_levels": payload["n_levels"],
        "n_months": len(payload["months"]),
        "station": payload["station_name"],
        "level_mode": payload["level_mode"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
