"""ecCodes распаковывает наблюдения; встроенные справочники подготовлены отдельно.

Один процесс — один исходный файл. ecCodes кэширует таблицы глобально, поэтому
повторный файл в этом процессе намеренно запрещён: архив меняет локальные
таблицы, не всегда меняя их номер. Внешний конвейер запускает отдельный процесс.
"""
import math
import os
import tempfile
from pathlib import Path

from .common import profile_from_pairs
from .embedded_tables import write_eccodes_definitions

_used = False


def decode_file(path, *, station_id=None, registry=None, collect_elements=False):
    global _used
    if _used:
        raise RuntimeError("ecCodes для следующего файла требует новый процесс (изоляция таблиц)")
    import eccodes as ec
    from gdex_bufr.bufr_adapter import _station_filter_set
    from gdex_bufr.bufr_tables import get_registry

    _used = True
    registry = registry or get_registry()
    wanted = _station_filter_set(station_id)
    profiles = []
    with tempfile.TemporaryDirectory(prefix="bufr_eccodes_") as temporary:
        definition_root = Path(temporary)
        digest = write_eccodes_definitions(path.read_bytes(), registry, definition_root)
        ec.codes_set_definitions_path(str(definition_root) + os.pathsep + ec.codes_definition_path())
        with path.open("rb") as handle:
            while (message := ec.codes_bufr_new_from_file(handle)) is not None:
                try:
                    category = ec.codes_get(message, "dataCategory")
                    subsets = ec.codes_get(message, "numberOfSubsets")
                    if category != 2 or not subsets:
                        continue
                    header = {name: ec.codes_get(message, key) for name, key in {
                        "master_table_number": "masterTableNumber", "master_table_version": "masterTablesVersionNumber",
                        "local_table_version": "localTablesVersionNumber", "bufr_header_edition": "edition",
                        "data_category": "dataCategory", "data_subcategory": "dataSubCategory",
                        "originating_centre": "bufrHeaderCentre", "n_subsets": "numberOfSubsets",
                    }.items()}
                    if ec.codes_get(message, "compressedData"):
                        raise ValueError("Сжатый ADPUPA пока не проверен ecCodes-адаптером")
                    ec.codes_set(message, "unpack", 1)
                    blocks = ec.codes_get_array(message, "d001001")
                    stations = ec.codes_get_array(message, "d001002")
                    selected = {i for i, (b, s) in enumerate(zip(blocks, stations))
                                if wanted is None or f"{int(b):02d}{int(s):03d}" in wanted}
                    if not selected:
                        continue
                    iterator = ec.codes_bufr_keys_iterator_new(message)
                    subset = -1
                    pairs = []
                    def append_profile():
                        if subset in selected:
                            profile = profile_from_pairs(path, pairs, subset, header, registry, "eccodes", collect_elements)
                            profile.metadata["embedded_tables_sha256"] = digest
                            profiles.append(profile)
                    try:
                        while ec.codes_bufr_keys_iterator_next(iterator):
                            key = ec.codes_bufr_keys_iterator_get_name(iterator)
                            if key == "subsetNumber":
                                append_profile()
                                subset += 1
                                pairs = []
                            elif subset in selected and key.rsplit("#", 1)[-1].startswith("d"):
                                abbreviation = key.rsplit("#", 1)[-1]
                                if not abbreviation[1:].isdigit():
                                    continue
                                value = ec.codes_get(message, key)
                                if isinstance(value, (int, float)) and (not math.isfinite(value) or abs(value) >= 1e99 or value == ec.CODES_MISSING_LONG):
                                    value = None
                                pairs.append((int(abbreviation[1:]), value))
                        append_profile()
                    finally:
                        ec.codes_bufr_keys_iterator_delete(iterator)
                finally:
                    ec.codes_release(message)
    return profiles
