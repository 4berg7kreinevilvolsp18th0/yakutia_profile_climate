"""Дополнительные декодеры. Основной pybufrkit остаётся вариантом по умолчанию.

Нативные варианты экспериментальные: доступность библиотеки не означает, что
все шаблоны конкретного архива уже прошли независимую проверку.
"""
from pathlib import Path


BACKENDS = ("pybufrkit", "eccodes", "ncepbufr")


def decode(path: Path, *, backend="pybufrkit", station_id=None, registry=None,
           collect_elements=False, **kwargs):
    if backend == "pybufrkit":
        from gdex_bufr.bufr_adapter import decode_bufr_file
        return decode_bufr_file(path, station_id=station_id, registry=registry,
                                collect_elements=collect_elements, **kwargs)
    if backend == "eccodes":
        from .eccodes_backend import decode_file
    elif backend == "ncepbufr":
        from .ncep_backend import decode_file
    else:
        raise ValueError(f"Неизвестный декодер: {backend}")
    if kwargs.get("decode_mode", "adpupa") != "adpupa":
        raise ValueError("Экспериментальные декодеры пока поддерживают только ADPUPA")
    return decode_file(Path(path), station_id=station_id, registry=registry,
                       collect_elements=collect_elements)
