"""Типы данных полей уровня в стиле NCEPLIBS-bufr debufr/ufdump."""
from __future__ import annotations

from typing import Any

from gdex_bufr.bufr_tables import BufrTablesRegistry, get_registry

# Колонка экспорта → основной FXY (WMO/NCEP ADPUPA).
# Для температуры/точки росы первичный — стандартный; NCEP non-standard — альтернатива.
LEVEL_FIELD_FXY: dict[str, str] = {
    "VSIG": "008001",
    "PRES": "007004",
    "GEOPOT": "010008",
    "FLVL": "010009",
    "AIR": "012101",
    "DEW-": "012103",
    "REL": "013003",
    "WIND": "011001",
    "WIND.1": "011002",
}

LEVEL_FIELD_FXY_ALT: dict[str, tuple[str, ...]] = {
    "AIR": ("012225", "012023"),
    "DEW-": ("012227", "012024"),
    "FLVL": ("007007",),  # height coordinate sometimes used as height proxy
}

# Ключевые колонки, для которых в decoded_levels пишем *_fxy/*_unit/*_kind
TYPED_LEVEL_COLUMNS: tuple[str, ...] = (
    "VSIG",
    "PRES",
    "GEOPOT",
    "FLVL",
    "AIR",
    "DEW-",
    "REL",
    "WIND",
    "WIND.1",
)

FIELD_TYPE_COLUMNS = [
    "column",
    "fxy",
    "alt_fxy",
    "name",
    "name_ru",
    "unit",
    "kind",
    "scale",
    "reference",
    "nbits",
]


def build_field_types_rows(registry: BufrTablesRegistry | None = None) -> list[dict[str, Any]]:
    """Однократный справочник типов колонок (как unit/desc в ufdump)."""
    reg = registry or get_registry()
    rows: list[dict[str, Any]] = []
    for column in TYPED_LEVEL_COLUMNS:
        fxy = LEVEL_FIELD_FXY[column]
        info = reg.lookup_descriptor(fxy)
        alt = LEVEL_FIELD_FXY_ALT.get(column, ())
        rows.append({
            "column": column,
            "fxy": info.fxy,
            "alt_fxy": ",".join(alt) if alt else "",
            "name": info.name,
            "name_ru": info.name_ru or "",
            "unit": info.unit,
            "kind": info.kind,
            "scale": info.scale,
            "reference": info.reference,
            "nbits": info.nbits,
        })
    return rows


def level_type_annotations(registry: BufrTablesRegistry | None = None) -> dict[str, Any]:
    """Словарь PRES_fxy / PRES_unit / PRES_kind … для строк decoded_levels."""
    reg = registry or get_registry()
    out: dict[str, Any] = {}
    for column in TYPED_LEVEL_COLUMNS:
        fxy = LEVEL_FIELD_FXY[column]
        info = reg.lookup_descriptor(fxy)
        out[f"{column}_fxy"] = info.fxy
        out[f"{column}_unit"] = info.unit
        out[f"{column}_kind"] = info.kind
    return out


def type_suffix_columns() -> list[str]:
    cols: list[str] = []
    for column in TYPED_LEVEL_COLUMNS:
        cols.extend([f"{column}_fxy", f"{column}_unit", f"{column}_kind"])
    return cols
