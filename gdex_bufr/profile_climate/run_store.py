"""Возобновляемая расшифровка: готовые порции данных и небольшой журнал SQLite.

Воркер сначала сохраняет файл порции через атомарное переименование. Только
после этого родитель отмечает источник готовым. Прерывание между этими шагами
означает повторную обработку, а не потерю профилей. CSV — производный экспорт;
его всегда можно собрать заново из порций без повторного чтения BUFR.
"""
from __future__ import annotations

import gzip
import base64
import hashlib
import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def source_key(path: Path) -> str:
    """Полный путь различает одноимённые файлы из разных архивов."""
    return os.path.normcase(str(path.resolve()))


def fingerprint(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _encode_diagnostic_value(value):
    """Некоторые BUFR-поля — байты, не текст: сохраняем их без потерь."""
    if isinstance(value, bytes):
        return {"__bufr_bytes_base64__": base64.b64encode(value).decode("ascii")}
    raise TypeError(f"Неподдерживаемый тип данных порции: {type(value).__name__}")


def _restore_diagnostic_value(value):
    if set(value) == {"__bufr_bytes_base64__"}:
        return base64.b64decode(value["__bufr_bytes_base64__"], validate=True)
    return value


def write_part(directory: Path, source: str, payload: dict) -> Path:
    """Одна сжатая порция на BUFR; повторный запуск заменяет её целиком."""
    directory.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(source.encode("utf-8")).hexdigest() + ".json.gz"
    target = directory / name
    fd, temporary = tempfile.mkstemp(dir=directory, suffix=".tmp")
    os.close(fd)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=1) as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"), default=_encode_diagnostic_value)
        Path(temporary).replace(target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def read_part(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle, object_hook=_restore_diagnostic_value)


class RunStore:
    """Журнал принадлежит одному процессу; параллельные воркеры пишут лишь порции."""

    def __init__(self, output: Path, config: dict):
        self.output = output
        output.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(output / "run.sqlite")
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS sources (
                source TEXT PRIMARY KEY, size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL,
                status TEXT NOT NULL, part TEXT, profiles INTEGER NOT NULL DEFAULT 0,
                error TEXT, seconds REAL, updated_at TEXT NOT NULL
            );
        """)
        signature = json.dumps(config, sort_keys=True, ensure_ascii=False)
        previous = self.connection.execute("SELECT value FROM settings WHERE key='config'").fetchone()
        if previous and previous[0] != signature:
            self.close()
            raise ValueError("Параметры отличаются от сохранённого прогона. Укажите другую папку --output.")
        with self.connection:
            self.connection.execute("INSERT OR IGNORE INTO settings VALUES ('config', ?)", (signature,))

    def completed(self, path: Path) -> bool:
        row = self.connection.execute("SELECT * FROM sources WHERE source=?", (source_key(path),)).fetchone()
        return bool(row and row["status"] in ("completed", "empty")
                    and (row["size"], row["mtime_ns"]) == fingerprint(path)
                    and row["part"] and (self.output / row["part"]).is_file())

    def record(self, path: Path, before: tuple[int, int], result: dict) -> None:
        """Смена файла во время чтения делает результат ошибкой, требующей повтора."""
        error = result.get("error")
        if not error and fingerprint(path) != before:
            error = "Исходный BUFR изменился во время обработки"
        part = result.get("part")
        if not error and (not part or not Path(part).is_file()):
            error = "Воркер не сохранил порцию данных"
        with self.connection:
            self.connection.execute("""
                INSERT OR REPLACE INTO sources VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (source_key(path), *before,
                  "failed" if error else ("completed" if result.get("profiles") else "empty"),
                  str(Path(part).relative_to(self.output)) if part else None,
                  result.get("profiles", 0), error, result.get("seconds", 0),
                  datetime.now(timezone.utc).isoformat()))

    def results(self):
        """Читаем по одному источнику: размер всего архива не определяет расход RAM."""
        rows = self.connection.execute("SELECT part FROM sources WHERE status != 'failed' ORDER BY source")
        for row in rows:
            yield read_part(self.output / row[0])

    def failures(self) -> list[dict]:
        return [dict(row) for row in self.connection.execute("SELECT source,error FROM sources WHERE status='failed'")]

    def close(self) -> None:
        self.connection.close()
