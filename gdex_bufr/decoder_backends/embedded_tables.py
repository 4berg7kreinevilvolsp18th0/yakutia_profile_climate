"""Перенос встроенных таблиц NCEP в формат ecCodes.

Здесь pybufrkit читает ТОЛЬКО служебные сообщения категории 11, не наблюдения.
Эту общую зависимость важно учитывать при независимой валидации. Стоимость
подготовки входит в время ecCodes в нашем бенчмарке.
"""
import hashlib
import json
from pathlib import Path

from .common import framed_messages


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


def write_eccodes_definitions(raw, registry, directory: Path):
    b_extra, d_extra, versions = read_embedded_tables(raw, registry)
    digest = hashlib.sha256(json.dumps([b_extra, d_extra], sort_keys=True).encode()).hexdigest()
    for master, version in versions:
        # Берём именно заявленную WMO-версию, не заменяем её молча на «latest».
        original = registry.tables_root / str(master) / "0_0" / str(version)
        if not (original / "TableB.json").is_file():
            raise ValueError(f"Нет WMO-таблицы {master}/{version} для ecCodes")
        b = json.loads((original / "TableB.json").read_text(encoding="utf-8"))
        d = json.loads((original / "TableD.json").read_text(encoding="utf-8"))
        b.update(b_extra)
        d.update(d_extra)
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
                     for code, entry in sorted(d.items())]
        (dest / "sequence.def").write_text("\n".join(sequences) + "\n", encoding="utf-8")
    return digest
