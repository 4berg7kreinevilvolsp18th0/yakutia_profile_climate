"""Сборка компактных суточных профилей для интерактивного дашборда."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

GRID_POINTS = 40
MAX_SURFACE_PRESSURE_HPA = 1000.0
PRESSURE_TOP_HPA = 500.0
PLOT_MIN_LEVELS = 3


def _day_key(dt: str) -> str:
    parsed = datetime.fromisoformat(str(dt).replace("Z", "+00:00"))
    return parsed.date().isoformat()


def _mean_on_grid(heights: np.ndarray, temps: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if len(heights) < 2:
        return np.full_like(grid, np.nan, dtype=float)
    order = np.argsort(heights)
    return np.interp(
        grid,
        heights[order],
        temps[order],
        left=np.nan,
        right=np.nan,
    )


def build_daily_profiles(
    long_csv: Path,
    metrics_csv: Path,
    *,
    pressure_top_hpa: float = PRESSURE_TOP_HPA,
    max_surface_pressure_hpa: float = MAX_SURFACE_PRESSURE_HPA,
    plot_min_levels: int = PLOT_MIN_LEVELS,
    grid_points: int = GRID_POINTS,
) -> dict[str, Any]:
    long_df = pd.read_csv(long_csv)
    metrics_df = pd.read_csv(metrics_csv)

    # Фильтр уровней как на PNG-графиках
    long_df = long_df.dropna(subset=["height_m", "temperature_c", "pressure_hpa"])
    long_df = long_df[
        (long_df["pressure_hpa"] <= max_surface_pressure_hpa)
        & (long_df["pressure_hpa"] >= pressure_top_hpa)
    ]

    metrics_map = {
        str(row.profile_id): row
        for row in metrics_df.itertuples(index=False)
    }

    # Группируем профили → дни
    by_day_profiles: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    day_meta: dict[str, dict[str, Any]] = {}

    for profile_id, group in long_df.groupby("profile_id"):
        metric = metrics_map.get(str(profile_id))
        status = getattr(metric, "profile_status", None) if metric is not None else None
        if status in {"no_temp", "bad_pressure", "duplicate_levels", "no_surface_level"}:
            continue
        if len(group) < plot_min_levels:
            continue

        dt = str(group["datetime_utc"].iloc[0])
        try:
            day = _day_key(dt)
        except ValueError:
            continue

        heights = group["height_m"].to_numpy(dtype=float)
        temps = group["temperature_c"].to_numpy(dtype=float)
        by_day_profiles[day].append((heights, temps))

        if day not in day_meta:
            day_meta[day] = {
                "n_profiles": 0,
                "inversion_detected": False,
                "statuses": [],
                "t_surface_vals": [],
            }
        day_meta[day]["n_profiles"] += 1
        if metric is not None:
            day_meta[day]["statuses"].append(str(getattr(metric, "profile_status", "")))
            if bool(getattr(metric, "inversion_detected", False)):
                day_meta[day]["inversion_detected"] = True
            t_s = getattr(metric, "t_surface_c", None)
            if t_s is not None and not (isinstance(t_s, float) and np.isnan(t_s)):
                day_meta[day]["t_surface_vals"].append(float(t_s))

    months: dict[str, dict[str, Any]] = {}
    for day, profiles in sorted(by_day_profiles.items()):
        month_key = day[:7]
        min_h = min(float(h.min()) for h, _ in profiles)
        max_h = max(float(h.max()) for h, _ in profiles)
        grid = np.linspace(min_h, max_h, grid_points)
        stacked = np.vstack([_mean_on_grid(h, t, grid) for h, t in profiles])
        mean_t = np.nanmean(stacked, axis=0)

        meta = day_meta[day]
        t_surface = (
            float(np.nanmean(meta["t_surface_vals"]))
            if meta["t_surface_vals"]
            else float(mean_t[0]) if not np.isnan(mean_t[0]) else None
        )
        good_count = sum(1 for s in meta["statuses"] if s == "good")

        months.setdefault(month_key, {"days": []})
        months[month_key]["days"].append({
            "date": day,
            "heights_m": [round(float(x), 1) for x in grid],
            "temperature_c": [None if np.isnan(x) else round(float(x), 3) for x in mean_t],
            "n_profiles": meta["n_profiles"],
            "n_good": good_count,
            "inversion_detected": meta["inversion_detected"],
            "t_surface_c": None if t_surface is None else round(t_surface, 3),
        })

    # Сортировка дней внутри месяца
    for month_key in months:
        months[month_key]["days"].sort(key=lambda d: d["date"])

    station_name = str(long_df["station_name"].iloc[0]) if len(long_df) else ""
    station_id = str(long_df["station_id"].iloc[0]) if len(long_df) else ""

    return {
        "station_id": station_id,
        "station_name": station_name,
        "pressure_top_hpa": pressure_top_hpa,
        "max_surface_pressure_hpa": max_surface_pressure_hpa,
        "grid_points": grid_points,
        "n_days": sum(len(m["days"]) for m in months.values()),
        "months": months,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Собрать daily_profiles.json для дашборда")
    parser.add_argument("--long", default="gdex_outputs/profile_climate/aldan/profiles_long.csv")
    parser.add_argument("--metrics", default="gdex_outputs/profile_climate/aldan/profile_metrics.csv")
    parser.add_argument("--output", default="gdex_outputs/profile_climate/aldan/daily_profiles.json")
    args = parser.parse_args()

    payload = build_daily_profiles(Path(args.long), Path(args.metrics))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output": str(out),
        "n_days": payload["n_days"],
        "n_months": len(payload["months"]),
        "station": payload["station_name"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
