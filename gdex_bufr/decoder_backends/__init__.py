"""Дополнительные декодеры. Основной pybufrkit остаётся вариантом по умолчанию.

Нативные варианты экспериментальные: доступность библиотеки не означает, что
все шаблоны конкретного архива уже прошли независимую проверку.
"""
from pathlib import Path


BACKENDS = ("pybufrkit", "eccodes", "ncepbufr")


def backend_version(backend):
    """Проверка зависимости до начала прогона и версия для его журнала."""
    if backend == "ncepbufr":
        import ncepbufr
        value = ncepbufr.__bufrlib_version__
        return value.decode("ascii") if isinstance(value, bytes) else str(value)
    if backend == "eccodes":
        import eccodes
        return eccodes.codes_get_api_version()
    if backend == "pybufrkit":
        from importlib.metadata import version
        return version("pybufrkit")
    raise ValueError(f"Неизвестный декодер: {backend}")


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
    if collect_elements:
        raise ValueError("Нативные адаптеры поддерживают climate/decoded; полный audit пока доступен только в pybufrkit")
    if kwargs.get("strict", True) is not True:
        raise ValueError("Нативные адаптеры требуют strict=True")
    unsupported = set(kwargs) - {"decode_mode", "strict", "decoder", "fast_station_filter"}
    if unsupported:
        raise ValueError(f"Неподдержанные параметры нативного адаптера: {sorted(unsupported)}")
    return decode_file(Path(path), station_id=station_id, registry=registry,
                       collect_elements=collect_elements)
