"""Воркеры ProcessPool для быстрой расшифровки (отдельный модуль — стабильный spawn на Windows)."""
from __future__ import annotations

import contextlib
import io
import logging
from pathlib import Path
from typing import Any

_WORKER: dict[str, Any] = {}


def worker_init(
    station_id: str,
    station_name: str,
    pressure_top: float,
    min_levels: int,
    min_inv: float,
    decode_mode: str,
    tables_dir: str,
    export_dir: str,
    project_root: str,
) -> None:
    import sys

    root = str(Path(project_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)

    logging.getLogger().setLevel(logging.ERROR)
    for name in list(logging.root.manager.loggerDict):
        if "bufr" in name.lower() or "pybufrkit" in name.lower():
            logging.getLogger(name).setLevel(logging.CRITICAL)
    logging.getLogger("PyBufrKit").setLevel(logging.CRITICAL)

    from gdex_bufr.bufr_adapter import _make_decoder, init_decoder_tables
    from gdex_bufr.meteo_parser_bridge import ensure_meteo_parser_import

    ensure_meteo_parser_import((Path(project_root) / "../meteo_parser").resolve())

    registry = init_decoder_tables({
        "directory": tables_dir,
        "wmo_version": "latest",
        "master_table_version": 43,
        "export_dir": export_dir,
        "export_on_update": False,
    })
    _WORKER["station_id"] = station_id
    _WORKER["station_name"] = station_name
    _WORKER["pressure_top"] = pressure_top
    _WORKER["min_levels"] = min_levels
    _WORKER["min_inv"] = min_inv
    _WORKER["decode_mode"] = decode_mode
    _WORKER["registry"] = registry
    _WORKER["decoder"] = _make_decoder(registry)


def decode_one(bufr_path: str) -> tuple[str, list[dict], list[dict], list[dict], list[dict], str | None]:
    """Возвращает (path, long_rows, metrics_rows, decoded_rows, element_rows, error)."""
    from gdex_bufr.bufr_adapter import decode_bufr_file
    from gdex_bufr.profile_climate.extract import process_profile

    path = Path(bufr_path)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            profiles = decode_bufr_file(
                path,
                station_id=_WORKER["station_id"],
                max_profiles=1,
                registry=_WORKER["registry"],
                decode_mode=_WORKER["decode_mode"],
                decoder=_WORKER["decoder"],
            )
        long_rows: list[dict] = []
        metrics_rows: list[dict] = []
        decoded_rows: list[dict] = []
        element_rows: list[dict] = []
        for profile in profiles:
            rows, metric, decoded, elements = process_profile(
                profile,
                station_name=_WORKER["station_name"],
                pressure_top_hpa=_WORKER["pressure_top"],
                min_levels_to_500=_WORKER["min_levels"],
                min_inversion_delta_c=_WORKER["min_inv"],
            )
            long_rows.extend(rows)
            metrics_rows.append(metric)
            decoded_rows.extend(decoded)
            element_rows.extend(elements)
        return bufr_path, long_rows, metrics_rows, decoded_rows, element_rows, None
    except Exception as exc:  # noqa: BLE001
        return bufr_path, [], [], [], [], str(exc)
