"""Контрольная сверка декодирования pybufrkit CLI vs Python API."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from gdex_bufr.bufr_adapter import decode_bufr_file, pybufrkit_decode_json


def compare_decode_outputs(path: Path) -> dict[str, Any]:
    """Сравниваю API-декодирование и JSON CLI pybufrkit."""
    profiles = decode_bufr_file(path, max_profiles=1)
    cli_json = pybufrkit_decode_json(path)
    profile = profiles[0] if profiles else None
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
        "ok": bool(profiles) and bool(cli_json) and (profile is not None and len(profile.levels) >= 2),
    }
