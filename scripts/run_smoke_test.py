"""Smoke-тест пайплайна profile_climate на синтетических профилях Алдана."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from gdex_bufr.meteo_parser_bridge import RadiosondeProfile, VerticalLevel
from gdex_bufr.profile_climate.export import export_all
from gdex_bufr.profile_climate.extract import process_profile
from gdex_bufr.profile_climate.plots import render_all_monthly_plots


def _synthetic_profile(day: int, cycle: str) -> RadiosondeProfile:
    levels = []
    pressures = [1000, 925, 850, 700, 600, 500]
    base_t = -28 + day * 0.3
    temps = [base_t, base_t + 1, base_t + 3, base_t + 1, base_t - 2, base_t - 6]
    for index, (pressure, temp) in enumerate(zip(pressures, temps)):
        levels.append(VerticalLevel(
            pressure_hpa=float(pressure),
            air_temperature_c=float(temp),
            geopotential_height_m=float(index * 500),
        ))
    return RadiosondeProfile(
        source_file=f"gdas.adpupa.t{cycle}z.199901{day:02d}.bufr",
        subset_index=0,
        station_id="31004",
        latitude_deg=58.37,
        longitude_deg=125.22,
        report_datetime_utc=datetime(1999, 1, day, int(cycle), 0).isoformat() + "Z",
        levels=levels,
    )


def main() -> int:
    output_dir = Path("gdex_outputs/profile_climate")
    plots_dir = Path("gdex_outputs/monthly_temperature_profiles")

    long_rows: list[dict] = []
    metrics_rows: list[dict] = []

    for day in range(1, 32):
        for cycle in ("00", "12"):
            profile = _synthetic_profile(day, cycle)
            rows, metric = process_profile(profile, station_name="Aldan")
            long_rows.extend(rows)
            metrics_rows.append(metric)

    paths = export_all(long_rows, metrics_rows, output_dir, config_info={"mode": "smoke_synthetic", "station": "31004"})
    written = render_all_monthly_plots(
        station_slug="aldan",
        station_name="Aldan",
        long_rows=long_rows,
        metrics_rows=metrics_rows,
        output_root=plots_dir,
        start_year=1999,
        end_year=1999,
        start_month=1,
        end_month=1,
        min_profiles_per_month=5,
    )

    print("exports:", paths)
    print("plots:", written[0] if written else "none")
    print("profiles:", len(metrics_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
