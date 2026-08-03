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

from gdex_bufr.profile_climate.obs_qc import (  # noqa: E402
    clean_observation_levels,
    interp_on_pressure_grid,
)

GRID_POINTS = 40
MAX_SURFACE_PRESSURE_HPA = 1000.0
PRESSURE_TOP_HPA = 500.0
PLOT_MIN_LEVELS = 3
SCHEMA = "observations_v1"


def _day_key(dt: str) -> str:
    parsed = datetime.fromisoformat(str(dt).replace("Z", "+00:00"))
    return parsed.date().isoformat()


def _series_to_levels(group: pd.DataFrame) -> list[dict[str, Any]]:
    levels: list[dict[str, Any]] = []
    for row in group.itertuples(index=False):
        levels.append({
            "pressure_hpa": float(row.pressure_hpa),
            "temperature_c": float(row.temperature_c),
            "height_m": float(row.height_m),
        })
    return levels


def _obs_arrays(levels: list[dict[str, Any]]) -> tuple[list[float], list[float | None], list[float | None]]:
    heights = [round(float(lv["height_m"]), 1) for lv in levels]
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
    pressure_top_hpa: float = PRESSURE_TOP_HPA,
    max_surface_pressure_hpa: float = MAX_SURFACE_PRESSURE_HPA,
    plot_min_levels: int = PLOT_MIN_LEVELS,
    grid_points: int = GRID_POINTS,
) -> dict[str, Any]:
    long_df = pd.read_csv(long_csv)
    metrics_df = pd.read_csv(metrics_csv)

    long_df = long_df.dropna(subset=["height_m", "temperature_c", "pressure_hpa"])
    long_df = long_df[
        (long_df["pressure_hpa"] <= max_surface_pressure_hpa)
        & (long_df["pressure_hpa"] >= pressure_top_hpa)
    ]

    metrics_map = {
        str(row.profile_id): row
        for row in metrics_df.itertuples(index=False)
    }

    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for profile_id, group in long_df.groupby("profile_id"):
        metric = metrics_map.get(str(profile_id))
        status = getattr(metric, "profile_status", None) if metric is not None else None
        if status in {"no_temp", "bad_pressure", "duplicate_levels", "no_surface_level"}:
            continue

        levels = clean_observation_levels(
            _series_to_levels(group),
            pressure_top_hpa=pressure_top_hpa,
            max_surface_pressure_hpa=max_surface_pressure_hpa,
        )
        if len(levels) < plot_min_levels:
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
            "profile_id": str(profile_id),
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
        "pressure_top_hpa": pressure_top_hpa,
        "max_surface_pressure_hpa": max_surface_pressure_hpa,
        "grid_points": grid_points,
        "n_days": sum(len(m["days"]) for m in months.values()),
        "n_observations": n_observations,
        "months": months,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Собрать daily_profiles.json (observations_v1)")
    parser.add_argument("--long", default="gdex_outputs/результаты-алдан/profiles_long.csv")
    parser.add_argument("--metrics", default="gdex_outputs/результаты-алдан/profile_metrics.csv")
    parser.add_argument("--output", default="gdex_outputs/результаты-алдан/daily_profiles.json")
    args = parser.parse_args()

    payload = build_daily_profiles(Path(args.long), Path(args.metrics))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output": str(out),
        "schema": payload["schema"],
        "n_days": payload["n_days"],
        "n_observations": payload["n_observations"],
        "n_months": len(payload["months"]),
        "station": payload["station_name"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
