"""Один процесс декодирования: один справочник, один декодер, много BUFR-файлов.

Между процессами передаются путь порции и счётчики. Миллионы словарей уровней
и диагностических элементов остаются внутри воркера и его файла результата.
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path

from gdex_bufr.profile_climate.run_store import source_key, write_part

_context: dict = {}


def initialize(options: dict) -> None:
    from gdex_bufr.bufr_adapter import _make_decoder, init_decoder_tables
    from gdex_bufr.profile_climate import height_fill

    registry = init_decoder_tables(options["tables"])
    # Передаём каталог явно: альтернативный YAML должен действовать и в воркерах.
    height_fill._CATALOG_ELEVATION = options["elevations"]
    _context.clear()
    _context.update(options, registry=registry, decoder=_make_decoder(registry))


def decode_file(filename: str) -> dict:
    from gdex_bufr.bufr_adapter import decode_bufr_file
    from gdex_bufr.profile_climate.extract import process_profile

    started = time.perf_counter()
    try:
        path = Path(filename)
        mode = _context["output_mode"]
        profiles = decode_bufr_file(
            path, station_id=set(_context["names"]), registry=_context["registry"],
            decoder=_context["decoder"], decode_mode=_context["decode_mode"],
            collect_elements=mode == "audit", strict=_context["strict"],
            fast_station_filter=_context["fast_station_filter"],
        )
        data = {"long": [], "metrics": [], "decoded": [], "elements": []}
        for profile in profiles:
            rows, metric, decoded, elements = process_profile(
                profile, station_name=_context["names"].get(profile.station_id, ""),
                registry=_context["registry"], output_mode=mode, **_context["science"],
            )
            data["long"].extend(rows)
            data["metrics"].append(metric)
            data["decoded"].extend(decoded)
            data["elements"].extend(elements)
        part = write_part(Path(_context["parts_dir"]), source_key(path), data)
        return {"part": str(part), "profiles": len(profiles), "levels": len(data["long"]),
                "seconds": time.perf_counter() - started, "error": None}
    except Exception:
        # Ошибка содержит traceback; родитель покажет её и сохранит для повторения.
        return {"error": traceback.format_exc(), "seconds": time.perf_counter() - started}
