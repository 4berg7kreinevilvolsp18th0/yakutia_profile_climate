"""SQLite-состояние очереди загрузки (потокобезопасное)."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

_UNSET = object()


@dataclass
class DownloadRecord:
    url: str
    local_path: str
    obs_date: str
    obs_type: str
    cycle: str
    expected_size: int | None
    status: str
    attempts: int
    bytes_downloaded: int
    etag: str | None
    last_modified: str | None
    error_message: str | None
    retry_after: str | None
    updated_at: str


class DownloadState:
    """Храню статус каждого файла manifest в SQLite."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=60.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=60000")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS downloads (
                    url TEXT PRIMARY KEY,
                    local_path TEXT NOT NULL,
                    obs_date TEXT,
                    obs_type TEXT,
                    cycle TEXT,
                    expected_size INTEGER,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    bytes_downloaded INTEGER NOT NULL DEFAULT 0,
                    etag TEXT,
                    last_modified TEXT,
                    error_message TEXT,
                    retry_after TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_downloads_status ON downloads(status);
                CREATE TABLE IF NOT EXISTS daily_stats (
                    day TEXT PRIMARY KEY,
                    bytes_downloaded INTEGER NOT NULL DEFAULT 0
                );
                """
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _today_key() -> str:
        return datetime.now(timezone.utc).date().isoformat()

    def upsert_pending(self, rows: list[dict]) -> None:
        with self._conn() as conn:
            for row in rows:
                conn.execute(
                    """
                    INSERT INTO downloads (
                        url, local_path, obs_date, obs_type, cycle, expected_size,
                        status, attempts, bytes_downloaded, etag, last_modified,
                        error_message, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, 0, NULL, NULL, NULL, ?)
                    ON CONFLICT(url) DO NOTHING
                    """,
                    (
                        row["url"],
                        row["local_path"],
                        row.get("obs_date"),
                        row.get("obs_type"),
                        row.get("cycle"),
                        row.get("expected_size"),
                        self._now(),
                    ),
                )

    def reset_stale_downloading(self) -> int:
        """После падения процесса возвращаю зависшие downloading в очередь."""
        with self._conn() as conn:
            cur = conn.execute(
                """
                UPDATE downloads
                SET status = 'partial', updated_at = ?
                WHERE status = 'downloading'
                """,
                (self._now(),),
            )
            return cur.rowcount

    def _ensure_retry_after_column(self, conn: sqlite3.Connection) -> None:
        cur = conn.execute("PRAGMA table_info(downloads)")
        columns = {row["name"] for row in cur.fetchall()}
        if "retry_after" not in columns:
            conn.execute("ALTER TABLE downloads ADD COLUMN retry_after TEXT")

    def claim_next(self, limit: int = 1) -> list[DownloadRecord]:
        """Атомарно забираю следующие файлы для параллельных воркеров."""
        now = self._now()
        with self._conn() as conn:
            self._ensure_retry_after_column(conn)
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.execute(
                """
                SELECT * FROM downloads
                WHERE status IN ('pending', 'partial')
                   OR (status = 'failed' AND (retry_after IS NULL OR retry_after <= ?))
                ORDER BY
                    CASE status WHEN 'pending' THEN 0 WHEN 'partial' THEN 1 ELSE 2 END,
                    obs_date, cycle, obs_type
                LIMIT ?
                """,
                (now, limit),
            )
            rows = [dict(row) for row in cur.fetchall()]
            for row in rows:
                conn.execute(
                    """
                    UPDATE downloads
                    SET status = 'downloading', updated_at = ?
                    WHERE url = ?
                    """,
                    (self._now(), row["url"]),
                )
            conn.commit()
            return [DownloadRecord(**row) for row in rows]

    def fetch_next(self, limit: int = 1) -> list[DownloadRecord]:
        return self.claim_next(limit)

    def mark_status(
        self,
        url: str,
        status: str,
        *,
        bytes_downloaded: int | None = None,
        expected_size: int | None = None,
        etag: str | None = None,
        last_modified: str | None = None,
        error_message: str | None = None,
        increment_attempt: bool = False,
        retry_after: str | None | object = _UNSET,
    ) -> None:
        with self._conn() as conn:
            self._ensure_retry_after_column(conn)
            params: list[object] = [
                status,
                bytes_downloaded,
                expected_size,
                etag,
                last_modified,
                error_message,
                1 if increment_attempt else 0,
            ]
            retry_sql = ""
            if retry_after is not _UNSET:
                retry_sql = ", retry_after = ?"
                params.append(retry_after)
            params.append(self._now())
            params.append(url)
            conn.execute(
                f"""
                UPDATE downloads
                SET status = ?,
                    bytes_downloaded = COALESCE(?, bytes_downloaded),
                    expected_size = COALESCE(?, expected_size),
                    etag = COALESCE(?, etag),
                    last_modified = COALESCE(?, last_modified),
                    error_message = ?,
                    attempts = attempts + ?{retry_sql},
                    updated_at = ?
                WHERE url = ?
                """,
                tuple(params),
            )

    def add_daily_bytes(self, nbytes: int) -> int:
        day = self._today_key()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO daily_stats(day, bytes_downloaded) VALUES (?, ?)
                ON CONFLICT(day) DO UPDATE SET bytes_downloaded = bytes_downloaded + excluded.bytes_downloaded
                """,
                (day, nbytes),
            )
            cur = conn.execute("SELECT bytes_downloaded FROM daily_stats WHERE day = ?", (day,))
            row = cur.fetchone()
            return int(row["bytes_downloaded"]) if row else nbytes

    def get_daily_bytes(self) -> int:
        day = self._today_key()
        with self._conn() as conn:
            cur = conn.execute("SELECT bytes_downloaded FROM daily_stats WHERE day = ?", (day,))
            row = cur.fetchone()
            return int(row["bytes_downloaded"]) if row else 0

    def would_exceed_daily_budget(self, budget_bytes: int, extra_bytes: int = 0) -> bool:
        if budget_bytes <= 0:
            return False
        return self.get_daily_bytes() + extra_bytes > budget_bytes

    def summary(self) -> dict[str, int]:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT status, COUNT(*) AS cnt FROM downloads GROUP BY status"
            )
            return {row["status"]: row["cnt"] for row in cur.fetchall()}
