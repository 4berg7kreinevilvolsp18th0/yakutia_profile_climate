"""Справочник WMO BUFR: дескрипторы, code/flag tables, экспорт."""
from __future__ import annotations

import csv
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gdex_bufr.tables_manager import (
    DEFAULT_WMO_VERSION,
    resolve_wmo_version,
    tables_are_installed,
    wmo_tables_sn_dir,
)

logger = logging.getLogger(__name__)

UNITS_CODE = frozenset({"CODE TABLE", "COMMON CODE TABLE C1", "COMMON CODE TABLE C2"})
UNITS_FLAG = frozenset({"FLAG TABLE"})

# NCEP ADPUPA / GDEX: локальные дескрипторы, которых нет в WMO Table B.
ADPUPA_LOCAL_DESCRIPTORS: dict[str, tuple[str, str, int, int, int, str]] = {
    "012225": ("Non-standard temperature", "K", 1, 0, 16, "K"),
    "012227": ("Non-standard dewpoint temperature", "K", 1, 0, 16, "K"),
}

ADPUPA_GLOSSARY_RU: dict[str, str] = {
    "005002": "Широта (грубая точность)",
    "006002": "Долгота (грубая точность)",
    "001001": "Блок станции WMO",
    "001002": "Номер станции WMO",
    "004001": "Год",
    "004002": "Месяц",
    "004003": "День",
    "004004": "Час",
    "004005": "Минута",
    "007004": "Давление",
    "010009": "Геопотенциальная высота",
    "011001": "Направление ветра",
    "011002": "Скорость ветра",
    "012225": "Температура (NCEP non-standard)",
    "012227": "Точка росы (NCEP non-standard)",
    "013003": "Относительная влажность",
    "002001": "Тип станции",
    "008010": "Тип зонда",
}


@dataclass(frozen=True)
class DescriptorInfo:
    fxy: str
    name: str
    unit: str
    scale: int
    reference: int
    nbits: int
    kind: str  # numeric | code | flag | operator | sequence | unknown
    name_ru: str | None = None
    code_table_id: str | None = None


@dataclass
class BufrTablesConfig:
    directory: Path = Path("./gdex_data/bufr_tables")
    wmo_version: int = DEFAULT_WMO_VERSION
    master_table_version: int | None = None
    export_dir: Path = Path("./gdex_data/bufr_tables_export")
    export_on_update: bool = True

    @property
    def resolved_version(self) -> int:
        return self.master_table_version or self.wmo_version


class BufrTablesRegistry:
    """Кэш TableGroup pybufrkit и JSON code/flag для проекта."""

    def __init__(self, config: BufrTablesConfig | None = None) -> None:
        self.config = config or BufrTablesConfig()
        self._table_group = None
        self._code_flag_raw: dict[str, list[list[str]]] | None = None
        self._table_b_raw: dict[str, list[Any]] | None = None

    @property
    def tables_root(self) -> Path:
        return Path(self.config.directory).resolve()

    @property
    def version_dir(self) -> Path:
        return wmo_tables_sn_dir(self.tables_root, self.config.resolved_version)

    def is_ready(self) -> bool:
        return tables_are_installed(self.tables_root, self.config.resolved_version)

    def _load_json_caches(self) -> None:
        if self._code_flag_raw is not None:
            return
        version_dir = self.version_dir
        if not self.is_ready():
            self._code_flag_raw = {}
            self._table_b_raw = {}
            return
        self._table_b_raw = json.loads((version_dir / "TableB.json").read_text(encoding="utf-8"))
        self._code_flag_raw = json.loads((version_dir / "code_and_flag.json").read_text(encoding="utf-8"))

    def get_table_group(self):
        if self._table_group is not None:
            return self._table_group
        from pybufrkit.tables import TableGroupCacheManager

        root = str(self.tables_root) if self.is_ready() else None
        self._table_group = TableGroupCacheManager.get_table_group(
            root,
            None,
            0,
            0,
            0,
            self.config.resolved_version,
            0,
            normalize=True,
        )
        self._table_group.B.load_code_and_flag()
        return self._table_group

    def _descriptor_kind(self, unit: str) -> str:
        if unit in UNITS_CODE:
            return "code"
        if unit in UNITS_FLAG:
            return "flag"
        return "numeric"

    def lookup_descriptor(self, fxy: str) -> DescriptorInfo:
        fxy = normalize_fxy(fxy)
        local = ADPUPA_LOCAL_DESCRIPTORS.get(fxy)
        if local:
            name, unit, scale, ref, nbits, _crex = local
            return DescriptorInfo(
                fxy=fxy,
                name=name,
                unit=unit,
                scale=scale,
                reference=ref,
                nbits=nbits,
                kind=self._descriptor_kind(unit),
                name_ru=ADPUPA_GLOSSARY_RU.get(fxy),
                code_table_id=fxy if unit in UNITS_CODE | UNITS_FLAG else None,
            )

        self._load_json_caches()
        raw = (self._table_b_raw or {}).get(fxy)
        if raw:
            name, unit, scale, ref, nbits = raw[0], raw[1], raw[2], raw[3], raw[4]
            return DescriptorInfo(
                fxy=fxy,
                name=name,
                unit=unit,
                scale=scale,
                reference=ref,
                nbits=nbits,
                kind=self._descriptor_kind(unit),
                name_ru=ADPUPA_GLOSSARY_RU.get(fxy),
                code_table_id=fxy if unit in UNITS_CODE | UNITS_FLAG else None,
            )

        try:
            tg = self.get_table_group()
            descriptor = tg.lookup(int(fxy))
            if hasattr(descriptor, "name"):
                unit = descriptor.unit
                return DescriptorInfo(
                    fxy=fxy,
                    name=descriptor.name,
                    unit=unit,
                    scale=descriptor.scale,
                    reference=descriptor.refval,
                    nbits=descriptor.nbits,
                    kind=self._descriptor_kind(unit),
                    name_ru=ADPUPA_GLOSSARY_RU.get(fxy),
                    code_table_id=fxy if unit in UNITS_CODE | UNITS_FLAG else None,
                )
        except Exception as exc:
            logger.debug("pybufrkit lookup failed for %s: %s", fxy, exc)

        return DescriptorInfo(
            fxy=fxy,
            name="Unknown descriptor",
            unit="",
            scale=0,
            reference=0,
            nbits=0,
            kind="unknown",
            name_ru=ADPUPA_GLOSSARY_RU.get(fxy),
        )

    def code_table_entries(self, fxy: str) -> list[tuple[int, str]]:
        self._load_json_caches()
        key = normalize_fxy(fxy)
        entries = (self._code_flag_raw or {}).get(key, [])
        result: list[tuple[int, str]] = []
        for code_str, description in entries:
            try:
                result.append((int(code_str), description))
            except ValueError:
                continue
        return result

    def decode_code_value(self, fxy: str, raw_value: Any) -> str | None:
        if raw_value is None:
            return None
        try:
            code = int(raw_value)
        except (TypeError, ValueError):
            return None
        for value, text in self.code_table_entries(fxy):
            if value == code:
                return text
        return None

    def decode_flag_bits(self, fxy: str, raw_value: Any) -> list[str]:
        if raw_value is None:
            return []
        try:
            bits = int(raw_value)
        except (TypeError, ValueError):
            return []
        active: list[str] = []
        for value, text in self.code_table_entries(fxy):
            try:
                flag_bit = int(value)
            except ValueError:
                continue
            if flag_bit >= 0 and (bits & (1 << flag_bit)):
                active.append(text)
        return active

    def apply_scaled_value(self, fxy: str, raw_value: Any) -> float | None:
        if raw_value is None:
            return None
        info = self.lookup_descriptor(fxy)
        if info.kind in {"code", "flag", "unknown"} and info.kind != "numeric":
            if info.kind != "unknown":
                return float(raw_value)
            return None
        try:
            number = float(raw_value)
        except (TypeError, ValueError):
            return None
        if info.scale:
            return (number + info.reference) / (10 ** info.scale)
        return number + info.reference

    def export_reference_files(self, export_dir: Path | None = None) -> dict[str, str]:
        export_dir = Path(export_dir or self.config.export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        self._load_json_caches()
        paths: dict[str, str] = {}

        desc_path = export_dir / "descriptors.csv"
        with desc_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["fxy", "name", "name_ru", "unit", "scale", "reference", "nbits", "kind", "code_table_id"]
            )
            rows = sorted((self._table_b_raw or {}).items())
            for fxy, fields in rows:
                info = self.lookup_descriptor(fxy)
                writer.writerow(
                    [
                        fxy,
                        info.name,
                        info.name_ru or "",
                        info.unit,
                        info.scale,
                        info.reference,
                        info.nbits,
                        info.kind,
                        info.code_table_id or "",
                    ]
                )
            for fxy in sorted(ADPUPA_LOCAL_DESCRIPTORS):
                if fxy not in (self._table_b_raw or {}):
                    info = self.lookup_descriptor(fxy)
                    writer.writerow(
                        [
                            fxy,
                            info.name,
                            info.name_ru or "",
                            info.unit,
                            info.scale,
                            info.reference,
                            info.nbits,
                            info.kind,
                            info.code_table_id or "",
                        ]
                    )
        paths["descriptors_csv"] = str(desc_path)

        code_path = export_dir / "code_tables.json"
        code_path.write_text(
            json.dumps(self._code_flag_raw or {}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        paths["code_tables_json"] = str(code_path)

        glossary_path = export_dir / "adpupa_glossary_ru.json"
        glossary_path.write_text(
            json.dumps(ADPUPA_GLOSSARY_RU, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        paths["glossary_ru_json"] = str(glossary_path)
        return paths

    def lookup_via_cli(self, fxy: str) -> str:
        cmd = [
            sys.executable,
            "-m",
            "pybufrkit",
            "lookup",
            "-l",
            normalize_fxy(fxy),
            "--code-and-flag",
        ]
        if self.is_ready():
            cmd.extend(["-t", str(self.tables_root)])
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return (proc.stdout or proc.stderr or "").strip()


_registry: BufrTablesRegistry | None = None


def normalize_fxy(fxy: str | int) -> str:
    digits = "".join(ch for ch in str(fxy).strip() if ch.isdigit())
    return digits.zfill(6)


def get_registry(config: BufrTablesConfig | None = None) -> BufrTablesRegistry:
    global _registry
    if config is not None:
        _registry = BufrTablesRegistry(config)
    elif _registry is None:
        _registry = BufrTablesRegistry()
    return _registry


def configure_from_app(raw: dict[str, Any] | None) -> BufrTablesRegistry:
    raw = raw or {}
    version = resolve_wmo_version(raw.get("wmo_version"))
    master = raw.get("master_table_version")
    config = BufrTablesConfig(
        directory=Path(raw.get("directory", "./gdex_data/bufr_tables")),
        wmo_version=version,
        master_table_version=int(master) if master is not None else None,
        export_dir=Path(raw.get("export_dir", "./gdex_data/bufr_tables_export")),
        export_on_update=bool(raw.get("export_on_update", True)),
    )
    return get_registry(config)
