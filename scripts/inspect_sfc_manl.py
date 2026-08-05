"""Инспекция SFC / MANL уровней и методов высоты для одного зонда.

Пример:
  python scripts/inspect_sfc_manl.py --date 2000-09-14 --cycle 12
  python scripts/inspect_sfc_manl.py --bufr gdex_data/raw/2000/gdas.adpupa.t12z.20000914.bufr
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gdex_bufr.bufr_adapter import decode_bufr_file  # noqa: E402
from gdex_bufr.bufr_tables import get_registry  # noqa: E402
from gdex_bufr.meteo_parser_bridge import (  # noqa: E402
    estimate_geopotential_height_m,
    geopotential_to_height_m,
)
from gdex_bufr.profile_climate.extract import (  # noqa: E402
    _pick_station_surface,
    extract_temperature_levels,
)
from gdex_bufr.profile_climate.height_fill import STATION_ELEVATION_M  # noqa: E402

STATION_ID = "31004"
STATION_H_M = STATION_ELEVATION_M.get(STATION_ID, 679.0)


def _resolve_bufr(date_s: str | None, cycle: str, bufr: Path | None) -> Path:
    if bufr is not None:
        return bufr
    if not date_s:
        raise SystemExit("Укажите --bufr или --date YYYY-MM-DD")
    ymd = date_s.replace("-", "")
    year = ymd[:4]
    cycle = str(cycle).zfill(2)[-2:]
    candidates = [
        ROOT / "gdex_data" / "raw" / year / f"gdas.adpupa.t{cycle}z.{ymd}.bufr",
        ROOT / "gdex_data" / "bufr_алдан" / year / f"gdas.adpupa.t{cycle}z.{ymd}.bufr",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise SystemExit(
        "BUFR не найден. Искали:\n  " + "\n  ".join(str(p) for p in candidates)
    )


def _is_sfc_or_manl(vsig: str | None, code: int | None) -> bool:
    if vsig:
        u = vsig.upper()
        if "SFC" in u or "MANL" in u or "SURFACE" in u or "STANDARD" in u:
            return True
    if code is None:
        return False
    # NCEP-коды и битовые комбинации с Surface (1) / Standard (2)
    if code in {1, 2, 32, 64}:
        return True
    return bool(code & 0b11)  # биты 0/1 ≈ surface/standard при LSB-схеме


def _height_methods(level, *, p_sfc: float | None, station_z: float) -> dict:
    h_obs = level.geopotential_height_m
    phi = level.geopotential_m2s2
    h_phi = None if phi is None else round(geopotential_to_height_m(phi), 1)
    h_baro = None
    if level.pressure_hpa is not None and p_sfc is not None:
        h_baro = round(
            float(station_z)
            + estimate_geopotential_height_m(
                level.pressure_hpa,
                surface_pressure_hpa=p_sfc,
            ),
            1,
        )
    return {
        "height_obs_or_flvl_m": h_obs,
        "geopotential_m2s2": phi,
        "height_phi_to_z_m": h_phi,
        "height_baro_station_m": h_baro,
        "station_elev_m": station_z,
        "delta_obs_vs_station_m": None if h_obs is None else round(h_obs - float(station_z), 1),
        "delta_phi_vs_station_m": None if h_phi is None else round(h_phi - float(station_z), 1),
        "delta_baro_vs_station_m": None if h_baro is None else round(h_baro - float(station_z), 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="SFC/MANL + сравнение высот")
    parser.add_argument("--date", help="YYYY-MM-DD")
    parser.add_argument("--cycle", default="12", help="00 или 12")
    parser.add_argument("--bufr", type=Path, help="Путь к .bufr")
    parser.add_argument("--station", default=STATION_ID)
    parser.add_argument("--all-levels", action="store_true", help="Печатать все уровни, не только SFC/MANL")
    parser.add_argument("--json-out", type=Path, help="Сохранить JSON-отчёт")
    args = parser.parse_args()

    path = _resolve_bufr(args.date, args.cycle, args.bufr)
    registry = get_registry()
    profiles = decode_bufr_file(path, station_id=str(args.station).zfill(5)[-5:], registry=registry)
    if not profiles:
        raise SystemExit(f"Нет профилей станции {args.station} в {path}")

    reports = []
    for profile in profiles:
        levels = list(profile.levels)
        preferred_lv = _pick_station_surface(
            levels,
            station_id=profile.station_id,
            bufr_station_elevation_m=profile.station_elevation_m,
        )
        p_sfc = preferred_lv.pressure_hpa if preferred_lv is not None else None
        if p_sfc is None:
            nums = [lv.pressure_hpa for lv in levels if lv.pressure_hpa is not None]
            p_sfc = max(nums) if nums else None

        z_st = profile.station_elevation_m
        if z_st is None:
            z_st = STATION_H_M

        selected = []
        for lv in levels:
            keep = args.all_levels or _is_sfc_or_manl(
                lv.vertical_significance,
                lv.vertical_significance_code,
            )
            if not keep:
                continue
            flag_text = []
            if lv.vertical_significance_code is not None:
                flag_text = registry.decode_flag_bits("008001", lv.vertical_significance_code)
                if not flag_text:
                    code_one = registry.decode_code_value("008001", lv.vertical_significance_code)
                    if code_one:
                        flag_text = [code_one]
            row = {
                "seq": lv.seq,
                "VSIG": lv.vertical_significance,
                "VSIG_code": lv.vertical_significance_code,
                "VSIG_flag_text": flag_text,
                "pressure_hpa": lv.pressure_hpa,
                "temperature_c": lv.air_temperature_c,
                "dewpoint_c": lv.dew_point_temperature_c,
                "rh_percent": lv.relative_humidity_percent,
                "wind_dir_deg": lv.wind_direction_deg,
                "wind_speed": lv.wind_speed,
                **_height_methods(lv, p_sfc=p_sfc, station_z=z_st),
            }
            selected.append(row)

        preferred = next(
            (r for r in selected if preferred_lv is not None and r["seq"] == preferred_lv.seq),
            None,
        )
        climate_levels = extract_temperature_levels(profile, pressure_top_hpa=500.0)

        report = {
            "bufr": str(path),
            "station_id": profile.station_id,
            "datetime_utc": profile.report_datetime_utc,
            "n_levels_total": len(levels),
            "n_sfc_manl": len(selected),
            "p_sfc_hpa": p_sfc,
            "station_elevation_m": z_st,
            "station_elevation_from_bufr": profile.station_elevation_m is not None,
            "station_height_fxy": "007001" if profile.station_elevation_m is not None else None,
            "preferred_sfc": preferred,
            "climate_levels_to_500": climate_levels,
            "climate_surface": climate_levels[0] if climate_levels else None,
            "levels": selected,
        }
        reports.append(report)

        print("=" * 72)
        print(f"BUFR: {path.name}")
        print(f"station={profile.station_id}  datetime={profile.report_datetime_utc}")
        print(
            f"Height of station (007001)="
            f"{profile.station_elevation_m if profile.station_elevation_m is not None else '—'} м  "
            f"(fallback config={STATION_H_M} м)  used_z_st={z_st} м"
        )
        print(f"levels_total={len(levels)}  SFC/MANL={len(selected)}  P_sfc={p_sfc}")
        if preferred:
            print(
                f"PREFERRED SFC: seq={preferred['seq']} P={preferred['pressure_hpa']} "
                f"H={preferred['height_obs_or_flvl_m']} "
                f"(Δ vs z_st={preferred['delta_obs_vs_station_m']} м)"
            )
        if climate_levels:
            cs = climate_levels[0]
            print(
                f"CLIMATE START: P={cs.get('pressure_hpa')} T={cs.get('temperature_c')} "
                f"H={cs.get('height_m')} VSIG={cs.get('VSIG')}  (n_to_500={len(climate_levels)})"
            )
        print("-" * 72)
        for row in selected:
            print(
                f"seq={row['seq']:>3}  VSIG={row['VSIG']!s:<8} code={row['VSIG_code']!s:<4} "
                f"flags={row['VSIG_flag_text']}  "
                f"P={row['pressure_hpa']}  T={row['temperature_c']}  "
                f"H_obs={row['height_obs_or_flvl_m']}  "
                f"H_Φ→z={row['height_phi_to_z_m']}  "
                f"H_baro={row['height_baro_station_m']}  "
                f"Δobs={row['delta_obs_vs_station_m']}"
            )
            print(
                f"         Td={row['dewpoint_c']}  RH={row['rh_percent']}  "
                f"wind={row['wind_dir_deg']}/{row['wind_speed']}  Φ={row['geopotential_m2s2']}"
            )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
