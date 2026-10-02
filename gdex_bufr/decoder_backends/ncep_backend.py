"""BUFRLIB: встроенные таблицы и наблюдения читает библиотека NOAA.

UFBREP возвращает согласованные строки уровней, включая пропуски. Нельзя
вызывать чтение каждого параметра отдельно и затем склеивать по номеру строки:
у температуры и ветра часто разное число уровней.
"""
import re
import tempfile
from pathlib import Path

from .common import framed_messages, profile_from_pairs


def parse_mnemonics(text):
    """Отображение FXY → имя поля из таблицы самого файла, не из догадки по году."""
    mapping = {}
    for line in text.splitlines():
        match = re.match(r"\|\s*(\w+)\s*\|\s*(0\d{5})\s*\|", line)
        if match:
            name, code = match.groups()
            mapping.setdefault(int(code), name)
            if name in {"WDIR", "WSPD", "TMDB", "TMDP", "PRLC", "VSIG"}:
                mapping[int(code)] = name
    return mapping


def decode_file(path, *, station_id=None, registry=None, collect_elements=False):
    import ncepbufr
    import numpy as np
    from gdex_bufr.bufr_adapter import (
        _make_decoder, _message_header_metadata, _station_filter_set,
        ADPUPA_LEVEL_FIELD_IDS, PROFILE_CODED_DESCRIPTORS,
    )
    from gdex_bufr.bufr_tables import get_registry

    if collect_elements:
        raise ValueError("BUFRLIB-адаптер не экспортирует полный дамп элементов; используйте climate/decoded")
    registry = registry or get_registry()
    wanted = _station_filter_set(station_id)
    # Только шапки, без распаковки наблюдений: сохраняем исходные номера таблиц.
    reader = _make_decoder(registry)
    headers = []
    observed = False
    for raw in framed_messages(path.read_bytes()):
        message = reader.process(raw, info_only=True)
        if observed and message.data_category.value == 11 and message.n_subsets.value:
            raise ValueError("Смена таблиц посреди файла пока не поддержана BUFRLIB-адаптером")
        if message.data_category.value != 11:
            headers.append(_message_header_metadata(message))
            observed = observed or bool(message.n_subsets.value)
    profiles = []
    bufr = ncepbufr.open(str(path))
    try:
        with tempfile.TemporaryDirectory(prefix="bufr_ncep_") as temporary:
            table = Path(temporary) / "table.txt"
            bufr.dump_table(str(table))
            mapping = parse_mnemonics(table.read_text(encoding="ascii"))
        required = {1001, 1002, 8001, 7004}
        if not required <= mapping.keys():
            raise ValueError("Встроенная таблица не содержит основных полей ADPUPA")
        header_codes = [1001, 1002, 4001, 4002, 4003, 4004, 4005, 5002, 6002, 7001]
        header_codes += [int(x) for x in PROFILE_CODED_DESCRIPTORS]
        header_codes = list(dict.fromkeys(code for code in header_codes if code in mapping))
        level_codes = [8001, 7004] + sorted(ADPUPA_LEVEL_FIELD_IDS - {8001, 7004})
        level_codes = [code for code in level_codes if code in mapping]
        message_index = 0
        while bufr.advance() == 0:
            if message_index >= len(headers):
                raise ValueError("BUFRLIB вернул больше сообщений, чем содержит файл")
            header = headers[message_index]
            message_index += 1
            if header["n_subsets"] != bufr.subsets:
                raise ValueError("Расхождение числа subset между шапкой и BUFRLIB")
            subset = -1
            while bufr.load_subset() == 0:
                subset += 1
                if header["data_category"] != 2:
                    continue
                identity = bufr.read_subset("WMOB WMOS")
                if not identity.shape[1] or np.ma.getmaskarray(identity[:, 0]).any():
                    continue
                sid = f"{int(identity[0, 0]):02d}{int(identity[1, 0]):03d}"
                if wanted is not None and sid not in wanted:
                    continue
                values = bufr.read_subset(" ".join(mapping[c] for c in header_codes))
                if values.shape[1] != 1:
                    raise ValueError("Неоднозначная или отсутствующая шапка профиля BUFRLIB")
                pairs = [(code, None if np.ma.is_masked(values[i, 0]) else float(values[i, 0]))
                         for i, code in enumerate(header_codes)]
                levels = bufr.read_subset(" ".join(mapping[c] for c in level_codes), rep=True)
                if levels.shape[1] >= ncepbufr._maxdim:
                    raise ValueError("Достигнут предел числа уровней BUFRLIB: усечение не допускается")
                for column in range(levels.shape[1]):
                    for row, code in enumerate(level_codes):
                        value = levels[row, column]
                        # Отсутствующий альтернативный дескриптор не должен
                        # заслонять реально закодированную температуру другого FXY.
                        if np.ma.is_masked(value) and code not in (8001, 7004):
                            continue
                        pairs.append((code, None if np.ma.is_masked(value) else float(value)))
                profile = profile_from_pairs(path, pairs, subset, header, registry, "ncepbufr")
                profiles.append(profile)
            if subset + 1 != header["n_subsets"]:
                raise ValueError("BUFRLIB не прочитал все subset сообщения")
        if message_index != len(headers):
            raise ValueError("BUFRLIB не прочитал все сообщения файла")
    finally:
        bufr.close()
    return profiles
