"""Перенос встроенных таблиц NCEP в формат ecCodes.

Здесь pybufrkit читает ТОЛЬКО служебные сообщения категории 11, не наблюдения.
Эту общую зависимость важно учитывать при независимой валидации. Стоимость
подготовки входит в время ecCodes в нашем бенчмарке.
"""
import hashlib
import json
import os
import re
from pathlib import Path

from .common import framed_messages


def normalize_ncep_sequences(table_d):
    """В NCEP отдельная последовательность иногда содержит лишь оператор повтора.

    Например 360001 = [101000, 031002], а повторяемый элемент находится уже
    в родителе. Встраиваем только такие префиксы, не меняя значения/битовый поток.
    Обычные последовательности сохраняются, чтобы число повторяемых элементов
    не изменилось при раскрытии вложенности.
    """
    prefixes = {}
    for code, entry in table_d.items():
        members = [int(x) for x in entry[1]]
        if (len(members) == 2 and members[0] == 101000
                and members[1] in (31000, 31001, 31002)):
            prefixes[int(code)] = members
    result = {}
    for code, entry in table_d.items():
        members = []
        for item in entry[1]:
            members.extend(prefixes.get(int(item), [int(item)]))
        result[code] = [entry[0], members]
    return result


def read_embedded_tables(raw, registry):
    from gdex_bufr.bufr_adapter import _make_decoder
    from pybufrkit.dataprocessor import BufrTableDefinitionProcessor

    decoder = _make_decoder(registry)
    table_b, table_d = {}, {}
    versions = set()
    observed = False
    for chunk in framed_messages(raw):
        header = decoder.process(chunk, info_only=True)
        if header.data_category.value == 11 and header.n_subsets.value:
            if observed:
                raise ValueError("Смена встроенных таблиц посреди файла пока не поддержана ecCodes-адаптером")
            message = decoder.process(chunk)
            _, b, d = BufrTableDefinitionProcessor().process(message)
            table_b.update(b)
            table_d.update(d)
        elif header.data_category.value == 2 and header.n_subsets.value:
            observed = True
            versions.add((header.master_table_number.value, header.master_table_version.value))
    if not table_b or not table_d:
        raise ValueError("Для этого ADPUPA-файла не найдены встроенные таблицы NCEP")
    return table_b, table_d, versions


def load_wmo_tables(registry, master, version):
    """Версия из проекта либо штатный путь этой версии в установленном ecCodes.

    Дистрибутив ecCodes сам задаёт совместимые версии символическими ссылками
    (например 12 → 13). Не выбираем произвольную более новую таблицу.
    """
    original = registry.tables_root / str(master) / "0_0" / str(version)
    if (original / "TableB.json").is_file() and (original / "TableD.json").is_file():
        return (json.loads((original / "TableB.json").read_text(encoding="utf-8")),
                json.loads((original / "TableD.json").read_text(encoding="utf-8")))
    import eccodes
    for root in eccodes.codes_definition_path().split(os.pathsep):
        source = Path(root) / "bufr" / "tables" / str(master) / "wmo" / str(version)
        if not (source / "element.table").is_file() or not (source / "sequence.def").is_file():
            continue
        b = {}
        for line in (source / "element.table").read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split("|")
            b[fields[0]] = [fields[3], fields[4], *map(int, fields[5:8])]
        sequences = (source / "sequence.def").read_text(encoding="utf-8")
        d = {code: ["", [int(x.strip()) for x in members.split(",") if x.strip()]]
             for code, members in re.findall(r'"(\d{6})"\s*=\s*\[([^\]]*)\]', sequences)}
        if not b or not d:
            raise ValueError(f"Пустые штатные таблицы ecCodes: {source}")
        return b, d
    raise ValueError(f"Нет доступной WMO-таблицы {master}/{version}; используйте окружение decoder-environment.yml")


def write_eccodes_definitions(raw, registry, directory: Path):
    b_extra, d_extra, versions = read_embedded_tables(raw, registry)
    digest = hashlib.sha256(json.dumps([b_extra, d_extra], sort_keys=True).encode()).hexdigest()
    for master, version in versions:
        # Берём именно заявленную WMO-версию, не заменяем её молча на «latest».
        b, d = load_wmo_tables(registry, master, version)
        b.update(b_extra)
        d.update(d_extra)
        d = normalize_ncep_sequences(d)
        dest = directory / "bufr" / "tables" / str(master) / "wmo" / str(version)
        dest.mkdir(parents=True, exist_ok=True)
        lines = ["#code|abbreviation|type|name|unit|scale|reference|width|crex_unit|crex_scale|crex_width"]
        for code, fields in sorted(b.items()):
            name, unit, scale, reference, width = fields[:5]
            kind = ("string" if "CCITT" in unit.upper() else "table" if "CODE" in unit.upper()
                    else "flag" if "FLAG" in unit.upper() else "double")
            clean_name = name.replace("|", " ").replace("\n", " ")
            lines.append(f"{int(code):06d}|d{int(code):06d}|{kind}|{clean_name}|{unit}|{scale}|{reference}|{width}|{unit}|0|0")
        (dest / "element.table").write_text("\n".join(lines) + "\n", encoding="utf-8")
        sequences = [f'"{int(code):06d}" = [ ' + ", ".join(f"{int(x):06d}" for x in entry[1]) + " ]"
                     for code, entry in sorted(d.items()) if entry[1]]
        # Пустые записи — зарезервированные дескрипторы, а не последовательности.
        # ecCodes отвергает синтаксис []; если файл ссылается на такую запись,
        # отсутствие определения вызовет явную ошибку вместо пропуска данных.
        (dest / "sequence.def").write_text("\n".join(sequences) + "\n", encoding="utf-8")
    return digest
