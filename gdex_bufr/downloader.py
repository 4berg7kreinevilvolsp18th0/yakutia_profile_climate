"""Параллельный downloader с resume, SQLite-состоянием и daemon-режимом."""
from __future__ import annotations

import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from gdex_bufr.config import AppConfig
from gdex_bufr.manifest import ManifestEntry, load_manifest
from gdex_bufr.state import DownloadRecord, DownloadState

logger = logging.getLogger(__name__)

DOWNLOAD_CHUNK_BYTES = 512 * 1024
HTTP_POOL_SIZE = 10
BYTES_PER_GB = 1024**3


def resolve_ssl_verify(cfg: AppConfig) -> bool | str:
    """Возвращаю параметр verify для requests (False, путь к CA-bundle или certifi)."""
    if not cfg.ssl_verify:
        return False
    if cfg.ssl_cert_bundle is not None:
        return str(cfg.ssl_cert_bundle)
    try:
        import certifi

        return certifi.where()
    except ImportError:
        return True


class RateLimiter:
    """Token-bucket с блокировкой для потоков. requests_per_minute=0 отключает лимит."""

    def __init__(self, requests_per_minute: float, min_delay_seconds: float, jitter_seconds: float) -> None:
        self.enabled = requests_per_minute > 0 or min_delay_seconds > 0
        if requests_per_minute > 0:
            self.interval = max(min_delay_seconds, 60.0 / requests_per_minute)
        else:
            self.interval = max(0.0, min_delay_seconds)
        self.jitter_seconds = jitter_seconds
        self._lock = threading.Lock()
        self._last_request = 0.0

    def wait(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request
            target = self.interval + random.uniform(0, self.jitter_seconds)
            if elapsed < target:
                time.sleep(target - elapsed)
            self._last_request = time.monotonic()


class PoliteDownloader:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.state = DownloadState(cfg.state_db)
        self.limiter = RateLimiter(cfg.requests_per_minute, cfg.min_delay_seconds, cfg.delay_jitter_seconds)
        self.daily_budget_bytes = int(cfg.daily_byte_budget_gb * BYTES_PER_GB) if cfg.daily_byte_budget_gb > 0 else 0
        self._budget_lock = threading.Lock()
        self._budget_exhausted = threading.Event()
        self._stats_lock = threading.Lock()
        self._processed = 0
        self._completed = 0
        self._failed = 0
        self._partial = 0
        self._limit: int | None = None
        self._ssl_verify = resolve_ssl_verify(cfg)

    def seed_queue(self, entries: list[ManifestEntry]) -> None:
        rows = [asdict(entry) for entry in entries]
        self.state.upsert_pending(rows)

    def _backoff_seconds(self, attempt: int) -> float:
        delay = min(
            self.cfg.backoff_max_seconds,
            self.cfg.backoff_base_seconds * (2 ** max(attempt - 1, 0)),
        )
        return delay + random.uniform(0, 1.0)

    def _validate_file(self, path: Path, expected_size: int | None) -> bool:
        if not path.exists() or path.stat().st_size == 0:
            return False
        if expected_size is not None and path.stat().st_size != expected_size:
            return False
        return True

    def _make_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update({"User-Agent": self.cfg.user_agent})
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=HTTP_POOL_SIZE,
            pool_maxsize=HTTP_POOL_SIZE,
            max_retries=0,
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _download_status_from_response(self, response: requests.Response, resume_from: int) -> tuple[str, int]:
        """Статус одной попытки загрузки по HTTP-коду (без записи тела ответа)."""
        code = response.status_code
        if code == 416:
            return "completed", resume_from
        if code == 404:
            return "not_found", resume_from
        if code == 403:
            return "forbidden", resume_from
        if code == 429:
            logger.debug("Rate limited (429) for %s", response.url)
            time.sleep(self.cfg.cooldown_on_429_seconds)
            return "failed", resume_from
        if code >= 500 or code not in (200, 206):
            return "failed", resume_from
        return "ok", resume_from

    def _download_once(self, record: DownloadRecord, session: requests.Session) -> tuple[str, int]:
        if self._budget_exhausted.is_set():
            return "partial", 0

        local_path = Path(record.local_path)
        part_path = local_path.with_suffix(local_path.suffix + ".part")
        local_path.parent.mkdir(parents=True, exist_ok=True)

        headers: dict[str, str] = {}
        resume_from = 0
        if part_path.exists():
            resume_from = part_path.stat().st_size
            headers["Range"] = f"bytes={resume_from}-"

        self.limiter.wait()
        response = session.get(
            record.url,
            headers=headers,
            stream=True,
            timeout=self.cfg.download_timeout_seconds,
            verify=self._ssl_verify,
        )

        status, resume_from = self._download_status_from_response(response, resume_from)
        if status != "ok":
            if status == "completed" and response.status_code == 416:
                part_path.replace(local_path)
                return "completed", local_path.stat().st_size
            return status, resume_from

        expected_size = record.expected_size
        content_length = response.headers.get("Content-Length")
        if expected_size is None and content_length and content_length.isdigit():
            expected_size = resume_from + int(content_length) if response.status_code == 206 else int(content_length)

        mode = "ab" if response.status_code == 206 else "wb"
        if response.status_code == 200 and part_path.exists():
            part_path.unlink(missing_ok=True)
            mode = "wb"
            resume_from = 0

        downloaded = resume_from
        with part_path.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                if not chunk:
                    continue
                if self.daily_budget_bytes > 0:
                    with self._budget_lock:
                        if self.state.would_exceed_daily_budget(self.daily_budget_bytes, len(chunk)):
                            logger.warning("Daily byte budget reached; pausing queue.")
                            self._budget_exhausted.set()
                            return "partial", downloaded
                handle.write(chunk)
                downloaded += len(chunk)

        etag = response.headers.get("ETag")
        last_modified = response.headers.get("Last-Modified")
        if expected_size is not None and downloaded != expected_size:
            self.state.mark_status(
                record.url,
                "partial",
                bytes_downloaded=downloaded,
                expected_size=expected_size,
                etag=etag,
                last_modified=last_modified,
                error_message=f"size mismatch: got {downloaded}, expected {expected_size}",
                increment_attempt=True,
            )
            return "partial", downloaded

        part_path.replace(local_path)
        if downloaded > resume_from:
            self.state.add_daily_bytes(downloaded - resume_from)
        self.state.mark_status(
            record.url,
            "completed",
            bytes_downloaded=downloaded,
            expected_size=expected_size,
            etag=etag,
            last_modified=last_modified,
            error_message=None,
        )
        return "completed", downloaded

    def _retry_after_iso(self, delay_seconds: float) -> str:
        when = datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)
        return when.isoformat()

    def _mark_terminal_http_error(
        self,
        record: DownloadRecord,
        *,
        status: str,
        message: str,
    ) -> str:
        self.state.mark_status(record.url, status, error_message=message, retry_after=None)
        logger.warning("%s, skipped: %s", message, record.url)
        return status

    def _process_record(self, record: DownloadRecord) -> str:
        local_path = Path(record.local_path)
        if self._validate_file(local_path, record.expected_size):
            self.state.mark_status(
                record.url,
                "completed",
                bytes_downloaded=local_path.stat().st_size,
                retry_after=None,
            )
            return "completed"

        session = self._make_session()
        attempt = 0
        last_error = ""
        final_status = "failed"
        try:
            while attempt < self.cfg.max_retries:
                if self._budget_exhausted.is_set():
                    self.state.mark_status(record.url, "partial", error_message="daily budget reached")
                    return "partial"
                attempt += 1
                try:
                    status, _ = self._download_once(record, session)
                    if status == "completed":
                        return "completed"
                    if status == "partial":
                        final_status = "partial"
                        break
                    if status == "not_found":
                        return self._mark_terminal_http_error(
                            record, status="not_found", message="HTTP 404 Not Found"
                        )
                    if status == "forbidden":
                        return self._mark_terminal_http_error(
                            record, status="forbidden", message="HTTP 403 Forbidden"
                        )
                    last_error = f"HTTP error ({status})"
                    logger.debug(
                        "Download retry (%s/%s): %s — %s",
                        attempt,
                        self.cfg.max_retries,
                        record.url,
                        last_error,
                    )
                except requests.RequestException as exc:
                    last_error = str(exc)
                    logger.debug(
                        "Download retry (%s/%s): %s — %s",
                        attempt,
                        self.cfg.max_retries,
                        record.url,
                        exc,
                    )
                if attempt < self.cfg.max_retries:
                    time.sleep(self._backoff_seconds(attempt))
        finally:
            session.close()

        if final_status == "partial":
            return "partial"

        cooldown = self._retry_after_iso(self.cfg.failed_retry_cooldown_seconds)
        self.state.mark_status(
            record.url,
            "failed",
            error_message=last_error or "max retries exceeded",
            increment_attempt=True,
            retry_after=cooldown,
        )
        logger.warning(
            "Download failed after %s attempts, cooldown until %s: %s — %s",
            self.cfg.max_retries,
            cooldown,
            record.url,
            last_error,
        )
        return "failed"

    def _worker_loop(self) -> None:
        while not self._budget_exhausted.is_set():
            with self._stats_lock:
                if self._limit is not None and self._processed >= self._limit:
                    return
            records = self.state.claim_next(limit=1)
            if not records:
                return
            status = self._process_record(records[0])
            with self._stats_lock:
                self._processed += 1
                if status == "completed":
                    self._completed += 1
                elif status == "partial":
                    self._partial += 1
                elif status in {"not_found", "forbidden"}:
                    pass
                else:
                    self._failed += 1
                if self.cfg.progress_every_n > 0 and self._processed % self.cfg.progress_every_n == 0:
                    logger.info(
                        "Progress: processed=%s completed=%s failed=%s partial=%s queue=%s",
                        self._processed,
                        self._completed,
                        self._failed,
                        self._partial,
                        self.state.summary(),
                    )

    def download_batch(self, *, limit: int | None = None) -> dict:
        self._processed = 0
        self._completed = 0
        self._failed = 0
        self._partial = 0
        self._budget_exhausted.clear()
        self._limit = limit
        self.state.reset_stale_downloading()

        workers = max(1, self.cfg.max_concurrency)
        if limit is not None and limit < workers:
            workers = max(1, limit)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(self._worker_loop) for _ in range(workers)]
            for future in futures:
                future.result()

        return {
            "processed": self._processed,
            "completed": self._completed,
            "failed": self._failed,
            "partial": self._partial,
            "state_summary": self.state.summary(),
            "budget_exhausted": self._budget_exhausted.is_set(),
        }

    def run_from_manifest(self, manifest_path: Path, *, limit: int | None = None) -> dict:
        entries = load_manifest(manifest_path)
        self.seed_queue(entries)
        return self.download_batch(limit=limit)

    def run_daemon(self, manifest_path: Path) -> None:
        """Бесконечный цикл: докачивает очередь, перезапускается после ошибок и пауз."""
        logger.info(
            "Daemon started: workers=%s rpm=%s delay=%ss budget_gb=%s",
            self.cfg.max_concurrency,
            self.cfg.requests_per_minute,
            self.cfg.min_delay_seconds,
            self.cfg.daily_byte_budget_gb,
        )
        while True:
            try:
                if manifest_path.exists():
                    self.seed_queue(load_manifest(manifest_path))
                result = self.download_batch()
                logger.info("Batch finished: %s", result)
                summary = result.get("state_summary", {})
                pending = self._pending_download_count(summary)
                if pending == 0 and summary.get("downloading", 0) == 0:
                    logger.info("All files downloaded.")
                    break
                if result.get("budget_exhausted"):
                    sleep_s = self.cfg.budget_reset_sleep_seconds
                    logger.info("Sleeping %ss until daily budget reset.", sleep_s)
                    time.sleep(sleep_s)
                    continue
                if result.get("processed", 0) == 0:
                    logger.info("Queue idle, sleeping %ss.", self.cfg.daemon_idle_seconds)
                    time.sleep(self.cfg.daemon_idle_seconds)
            except Exception:
                logger.exception("Daemon batch failed, retry in %ss", self.cfg.daemon_error_sleep_seconds)
                time.sleep(self.cfg.daemon_error_sleep_seconds)

    def _pending_download_count(self, summary: dict) -> int:
        return (
            summary.get("pending", 0)
            + summary.get("failed", 0)
            + summary.get("partial", 0)
            + summary.get("downloading", 0)
        )
