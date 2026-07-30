"""Контрольная сверка декодирования pybufrkit CLI vs Python API."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from gdex_bufr.bufr_adapter import decode_bufr_file, pybufrkit_decode_json
from gdex_bufr.meteo_parser_bridge import RadiosondeProfile


def profile_decode_qc(profile: RadiosondeProfile) -> dict[str, Any]:
    """Structural and physical checks that catch level/unit aliasing."""
    invalid_pressure = [
        level.seq
        for level in profile.levels
        if level.pressure_hpa is None or not (0 < level.pressure_hpa <= 1100)
    ]
    invalid_temperature = [
        level.seq
        for level in profile.levels
        if level.air_temperature_c is not None
        and not (-110 <= level.air_temperature_c <= 70)
    ]

    # Use one temperature per pressure and look for a near-vertical "tooth".
    # The historical 1000 Pa -> 1000 hPa bug creates 30-50 C jumps within
    # only a few hPa of otherwise valid lower-tropospheric observations.
    thermo_by_pressure: dict[float, float] = {}
    for level in profile.levels:
        if level.pressure_hpa is None or level.air_temperature_c is None:
            continue
        if level.pressure_hpa < 500:
            continue
        thermo_by_pressure.setdefault(
            round(level.pressure_hpa, 1),
            level.air_temperature_c,
        )
    thermo = sorted(thermo_by_pressure.items(), reverse=True)
    suspicious_jumps: list[dict[str, float]] = []
    for (pressure_a, temp_a), (pressure_b, temp_b) in zip(thermo, thermo[1:]):
        pressure_delta = abs(pressure_a - pressure_b)
        temp_delta = abs(temp_a - temp_b)
        if pressure_delta <= 100 and temp_delta >= 25:
            suspicious_jumps.append({
                "pressure_a_hpa": pressure_a,
                "temperature_a_c": temp_a,
                "pressure_b_hpa": pressure_b,
                "temperature_b_c": temp_b,
                "delta_pressure_hpa": pressure_delta,
                "delta_temperature_c": temp_delta,
            })

    pressures = [
        level.pressure_hpa
        for level in profile.levels
        if level.pressure_hpa is not None
    ]
    return {
        "ok": not invalid_pressure
        and not invalid_temperature
        and not suspicious_jumps,
        "pressure_min_hpa": min(pressures, default=None),
        "pressure_max_hpa": max(pressures, default=None),
        "invalid_pressure_level_sequences": invalid_pressure,
        "invalid_temperature_level_sequences": invalid_temperature,
        "suspicious_tropospheric_jumps": suspicious_jumps,
        "levels_with_temperature": sum(
            level.air_temperature_c is not None for level in profile.levels
        ),
        "levels_with_geopotential": sum(
            level.geopotential_m2s2 is not None for level in profile.levels
        ),
        "levels_with_wind": sum(
            level.wind_speed is not None for level in profile.levels
        ),
    }


def compare_decode_outputs(
    path: Path,
    *,
    station_id: str | None = None,
) -> dict[str, Any]:
    """Compare API/CLI availability and validate the adapted profile."""
    profiles = decode_bufr_file(
        path,
        max_profiles=1 if station_id is None else None,
        station_id=station_id,
    )
    cli_json = pybufrkit_decode_json(path)
    profile = profiles[0] if profiles else None
    qc = profile_decode_qc(profile) if profile else None
    return {
        "file": str(path),
        "api_profiles": len(profiles),
        "cli_messages": len(cli_json),
        "profile_station_id": profile.station_id if profile else None,
        "profile_levels": len(profile.levels) if profile else 0,
        "profile_lat_lon": (
            (profile.latitude_deg, profile.longitude_deg) if profile else None
        ),
        "cli_has_data": bool(cli_json),
        "profile_qc": qc,
        "ok": bool(profiles)
        and bool(cli_json)
        and profile is not None
        and len(profile.levels) >= 2
        and bool(qc and qc["ok"]),
    }
