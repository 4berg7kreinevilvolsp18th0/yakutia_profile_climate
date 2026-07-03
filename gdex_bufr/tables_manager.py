"""Скачивание и установка WMO BUFR-таблиц для pybufrkit."""
from __future__ import annotations

import csv
import io
import json
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Any
from urllib.request import urlopen

logger = logging.getLogger(__name__)

DEFAULT_WMO_VERSION = 43
LATEST_WMO_VERSION = 43
# Редакции, которые встречаются в заголовках BUFR GDEX/NCEP при отсутствии отдельного архива WMO.
TABLE_VERSION_ALIASES = (4, 33, 36)


def resolve_wmo_version(version: str | int | None) -> int:
    """Преобразую 'latest' или None в числовую редакцию таблиц WMO."""
    if version is None:
        return DEFAULT_WMO_VERSION
    if isinstance(version, int):
        return version
    text = str(version).strip().lower()
    if text in {"", "latest", "default"}:
        return LATEST_WMO_VERSION
    return int(text)


def wmo_tables_sn_dir(tables_root: Path, master_table_version: int) -> Path:
    """Каталог pybufrkit: tables_root/0/0_0/{version}/."""
    return Path(tables_root) / "0" / "0_0" / str(master_table_version)


def tables_are_installed(tables_root: Path, version: int) -> bool:
    """Проверяю, что TableB.json не пустой."""
    table_b = wmo_tables_sn_dir(tables_root, version) / "TableB.json"
    if not table_b.exists():
        return False
    try:
        data = json.loads(table_b.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(data)


def download_wmo_release(version: int, *, tag: str | None = None) -> bytes:
    tag = tag or str(version)
    url = f"https://github.com/wmo-im/BUFR4/archive/refs/tags/v{tag}.zip"
    logger.info("Downloading WMO BUFR tables v%s from %s", version, url)
    return urlopen(url, timeout=120).read()


def _process_table_b(content: str) -> dict[str, list[Any]]:
    lines = csv.reader(io.StringIO(content), quoting=csv.QUOTE_MINIMAL)
    next(lines, None)
    offset = 0
    result: dict[str, list[Any]] = {}
    for line in lines:
        if len(line) < 11:
            continue
        crex_scale = 0 if line[9 + offset] == "" else int(line[9 + offset])
        crex_data_width = 0 if line[10 + offset] == "" else int(line[10 + offset])
        result[line[2]] = [
            line[3],
            line[4 + offset],
            int(line[5 + offset]),
            int(line[6 + offset]),
            int(line[7 + offset]),
            line[8 + offset],
            crex_scale,
            crex_data_width,
        ]
    return result


def _process_table_d(content: str) -> dict[str, list[Any]]:
    lines = csv.reader(io.StringIO(content), quoting=csv.QUOTE_MINIMAL)
    next(lines, None)
    result: dict[str, list[Any]] = {}
    for line in lines:
        if len(line) < 6:
            continue
        entry = result.setdefault(line[2], [line[3], []])
        entry[1].append(line[5])
    return result


def _process_code_flag(content: str) -> dict[str, list[list[str]]]:
    lines = csv.reader(io.StringIO(content), quoting=csv.QUOTE_MINIMAL)
    next(lines, None)
    result: dict[str, list[list[str]]] = {}
    for line in lines:
        if len(line) < 4:
            continue
        result.setdefault(line[0], []).append([line[2], line[3]])
    return result


def convert_tables_from_zip(version: int, data: bytes) -> dict[str, dict[str, Any]]:
    """Конвертирую ZIP WMO BUFR4; пути в архиве всегда с '/'."""
    prefix = f"BUFR4-{version}/"
    zf = zipfile.ZipFile(io.BytesIO(data), "r")
    table_b: dict[str, Any] = {}
    table_d: dict[str, Any] = {}
    code_flag: dict[str, Any] = {}
    for fileinfo in zf.infolist():
        name = fileinfo.filename.replace("\\", "/")
        if not name.startswith(prefix):
            continue
        payload = zf.read(fileinfo).decode("utf-8")
        if "BUFRCREX_TableB_en_" in name:
            table_b.update(_process_table_b(payload))
        elif "BUFR_TableD_en_" in name:
            table_d.update(_process_table_d(payload))
        elif "BUFRCREX_CodeFlag_en_" in name:
            code_flag.update(_process_code_flag(payload))
    return {"b": table_b, "d": table_d, "code_and_flag": code_flag}


def write_pybufrkit_tables(version: int, tables: dict[str, dict[str, Any]], output_dir: Path) -> Path:
    base_dir = Path(output_dir) / str(version)
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "TableB.json").write_text(
        json.dumps(tables["b"], ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    (base_dir / "TableD.json").write_text(
        json.dumps(tables["d"], ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    (base_dir / "code_and_flag.json").write_text(
        json.dumps(tables["code_and_flag"], ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return base_dir


def sync_table_version_aliases(
    tables_root: Path,
    source_version: int,
    aliases: tuple[int, ...] = TABLE_VERSION_ALIASES,
) -> list[str]:
    """Копирую установленную редакцию в каталоги старых master_table_version."""
    src = wmo_tables_sn_dir(tables_root, source_version)
    if not tables_are_installed(tables_root, source_version):
        return []
    synced: list[str] = []
    for alias in aliases:
        if alias == source_version:
            continue
        dst = wmo_tables_sn_dir(tables_root, alias)
        if dst.exists():
            continue
        shutil.copytree(src, dst)
        synced.append(str(dst))
    return synced


def update_wmo_tables(
    tables_root: Path,
    version: str | int | None = None,
    *,
    overwrite: bool = False,
    tag: str | None = None,
) -> dict[str, Any]:
    """Скачиваю и сохраняю таблицы WMO в формате pybufrkit."""
    resolved = resolve_wmo_version(version)
    sn_dir = wmo_tables_sn_dir(tables_root, resolved)
    if tables_are_installed(tables_root, resolved) and not overwrite:
        return {
            "status": "exists",
            "version": resolved,
            "path": str(sn_dir),
            "descriptor_count": len(json.loads((sn_dir / "TableB.json").read_text(encoding="utf-8"))),
        }

    raw = download_wmo_release(resolved, tag=tag or str(resolved))
    converted = convert_tables_from_zip(resolved, raw)
    if not converted["b"]:
        raise RuntimeError(f"Table B is empty after conversion for version {resolved}")

    output_parent = tables_root / "0" / "0_0"
    output_parent.mkdir(parents=True, exist_ok=True)
    base_dir = write_pybufrkit_tables(resolved, converted, output_parent)
    aliases = sync_table_version_aliases(tables_root, resolved)
    try:
        from pybufrkit.tables import TableGroupCacheManager

        TableGroupCacheManager.invalidate()
    except Exception:
        pass
    return {
        "status": "updated",
        "version": resolved,
        "path": str(base_dir),
        "descriptor_count": len(converted["b"]),
        "sequence_count": len(converted["d"]),
        "code_flag_tables": len(converted["code_and_flag"]),
        "aliases_synced": aliases,
    }
