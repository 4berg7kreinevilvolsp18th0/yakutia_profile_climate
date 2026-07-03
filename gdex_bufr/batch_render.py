"""Пакетное декодирование BUFR и построение метеодиаграмм."""
from __future__ import annotations

import csv
import logging
import re
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path

from gdex_bufr.bufr_adapter import decode_bufr_file, export_decoded_fields_csv, init_decoder_tables
from gdex_bufr.bufr_tables import ADPUPA_GLOSSARY_RU, get_registry
from gdex_bufr.config import AppConfig
from gdex_bufr.meteo_parser_bridge import RadiosondeProfile
from gdex_bufr.plot_config import PlotStyle
from gdex_bufr.plots import output_dir_for_file, render_plots
from gdex_bufr.xlsx_export import append_profiles_xlsx

logger = logging.getLogger(__name__)

FILENAME_RE = re.compile(
    r"gdas\.(?P<obs_type>[a-z]+)\.t(?P<cycle>\d{2})z\.(?P<obs_date>\d{8})\.bufr$"
)


def _parse_bufr_meta(path: Path) -> dict[str, str]:
    match = FILENAME_RE.search(path.name)
    if not match:
        return {}
    return match.groupdict()


class BatchRenderState:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=60.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS render_log (
                    bufr_path TEXT PRIMARY KEY,
                    obs_date TEXT,
                    cycle TEXT,
                    profiles_decoded INTEGER NOT NULL DEFAULT 0,
                    plots_written INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_render_log_status ON render_log(status);
                """
            )

    @staticmethod
    def _now() -> str:
        return datetime.utcnow().isoformat() + "Z"

    def is_done(self, bufr_path: str) -> bool:
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT status FROM render_log WHERE bufr_path = ?",
                (bufr_path,),
            ).fetchone()
            return row is not None and row["status"] == "completed"

    def mark(self, bufr_path: str, *, status: str, obs_date: str = "", cycle: str = "",
             profiles_decoded: int = 0, plots_written: int = 0, error_message: str | None = None) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                """
                INSERT INTO render_log (
                    bufr_path, obs_date, cycle, profiles_decoded, plots_written,
                    status, error_message, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(bufr_path) DO UPDATE SET
                    profiles_decoded = excluded.profiles_decoded,
                    plots_written = excluded.plots_written,
                    status = excluded.status,
                    error_message = excluded.error_message,
                    updated_at = excluded.updated_at
                """,
                (bufr_path, obs_date, cycle, profiles_decoded, plots_written, status, error_message, self._now()),
            )

    def summary(self) -> dict[str, int]:
        with self._conn() as conn:
            cur = conn.execute("SELECT status, COUNT(*) AS cnt FROM render_log GROUP BY status")
            return {row["status"]: row["cnt"] for row in cur.fetchall()}


def list_bufr_files(
    cfg: AppConfig,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int | None = None,
    only_completed_downloads: bool = True,
) -> list[Path]:
    """Список локальных BUFR для обработки."""
    files: list[Path] = []

    if only_completed_downloads and cfg.state_db.exists():
        conn = sqlite3.connect(cfg.state_db)
        conn.row_factory = sqlite3.Row
        query = "SELECT local_path, obs_date FROM downloads WHERE status = 'completed'"
        params: list[object] = []
        if start_date:
            query += " AND obs_date >= ?"
            params.append(start_date.isoformat())
        if end_date:
            query += " AND obs_date <= ?"
            params.append(end_date.isoformat())
        query += " ORDER BY obs_date, cycle"
        if limit:
            query += f" LIMIT {int(limit)}"
        for row in conn.execute(query, params):
            path = Path(row["local_path"])
            if path.exists():
                files.append(path)
        conn.close()
        return files

    pattern = "**/*.bufr"
    for path in sorted(cfg.data_dir.glob(pattern)):
        meta = _parse_bufr_meta(path)
        obs_date_str = meta.get("obs_date", "")
        if len(obs_date_str) == 8:
            obs_d = date(int(obs_date_str[:4]), int(obs_date_str[4:6]), int(obs_date_str[6:8]))
            if start_date and obs_d < start_date:
                continue
            if end_date and obs_d > end_date:
                continue
        files.append(path)
        if limit and len(files) >= limit:
            break
    return files


class BatchRenderer:
    def __init__(self, cfg: AppConfig, plot_style: PlotStyle) -> None:
        self.cfg = cfg
        self.plot_style = plot_style
        self.registry = init_decoder_tables({
            "directory": str(cfg.bufr_tables.directory),
            "wmo_version": cfg.bufr_tables.wmo_version,
            "master_table_version": cfg.bufr_tables.master_table_version,
            "export_dir": str(cfg.bufr_tables.export_dir),
            "export_on_update": cfg.bufr_tables.export_on_update,
        })
        render_db = cfg.outputs_dir / "render_state.sqlite"
        self.state = BatchRenderState(render_db)
        self._csv_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._processed = 0
        self._profiles = 0
        self._plots = 0
        self._skipped = 0
        self._errors = 0

    def _xlsx_path(self) -> Path:
        return self.cfg.outputs_dir / "decoded_profiles.xlsx"

    def _fields_dict_path(self) -> Path:
        return self.cfg.outputs_dir / "bufr_fields_dictionary.csv"

    def _append_xlsx(self, profiles: list[RadiosondeProfile]) -> None:
        if not self.plot_style.export_xlsx or not profiles:
            return
        append_profiles_xlsx(self._xlsx_path(), profiles)

    def _append_fields_dictionary(self, profiles: list[RadiosondeProfile]) -> None:
        if not profiles:
            return
        path = self._fields_dict_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        seen: set[str] = set()
        if path.exists():
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    seen.add(row.get("descriptor", ""))

        registry = get_registry()
        rows: list[dict[str, str]] = []
        for profile in profiles:
            coded = profile.metadata.get("coded_metadata") or {}
            for fxy, payload in coded.items():
                if fxy in seen:
                    continue
                seen.add(fxy)
                info = registry.lookup_descriptor(fxy)
                rows.append({
                    "descriptor": fxy,
                    "name": info.name,
                    "name_ru": info.name_ru or ADPUPA_GLOSSARY_RU.get(fxy, ""),
                    "unit": info.unit,
                    "kind": info.kind,
                    "example_value": str(payload.get("value", "")),
                    "example_text": str(payload.get("value_text", "")),
                })

        if not rows:
            return
        write_header = not path.exists()
        with self._csv_lock, path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["descriptor", "name", "name_ru", "unit", "kind", "example_value", "example_text"],
            )
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    def _plots_exist(self, out_dir: Path, profile: RadiosondeProfile) -> bool:
        if not self.plot_style.skip_existing:
            return False
        suffix = self.plot_style.output_suffix()
        from gdex_bufr.plots import _profile_stem

        stem = _profile_stem(profile)
        for plot_type in self.plot_style.plot_types:
            if plot_type == "map":
                continue
            candidate = out_dir / f"{stem}_{plot_type}{suffix}"
            if not candidate.exists():
                return False
        return True

    def _process_file(self, bufr_path: Path) -> dict:
        bufr_key = str(bufr_path.resolve())
        meta = _parse_bufr_meta(bufr_path)
        obs_date = meta.get("obs_date", "")
        cycle = meta.get("cycle", "")

        if self.plot_style.skip_existing and self.state.is_done(bufr_key):
            return {"status": "skipped", "reason": "already rendered"}

        try:
            profiles = decode_bufr_file(
                bufr_path,
                max_profiles=self.plot_style.max_profiles_per_file,
                registry=self.registry,
                decode_mode=self.cfg.decode_mode,
            )
        except Exception as exc:
            self.state.mark(bufr_key, status="error", obs_date=obs_date, cycle=cycle, error_message=str(exc))
            return {"status": "error", "error": str(exc)}

        if not profiles:
            self.state.mark(bufr_key, status="empty", obs_date=obs_date, cycle=cycle)
            return {"status": "empty"}

        out_dir = output_dir_for_file(self.cfg.outputs_dir, bufr_path, self.plot_style, cycle=cycle)
        plot_types = [p for p in self.plot_style.plot_types if p != "map"]
        plots_written = 0
        for profile in profiles:
            if self._plots_exist(out_dir, profile):
                continue
            try:
                outputs = render_plots(
                    profile,
                    out_dir,
                    plot_types,
                    all_profiles=profiles,
                    style=self.plot_style,
                    min_levels=2,
                )
                plots_written += len(outputs)
            except Exception as exc:
                logger.warning("Plot failed for %s subset %s: %s", bufr_path.name, profile.subset_index, exc)

        if "map" in self.plot_style.plot_types:
            map_path = out_dir / f"stations_map{self.plot_style.output_suffix()}"
            if not (self.plot_style.skip_existing and map_path.exists()):
                try:
                    from gdex_bufr.plots import plot_station_map

                    plot_station_map(profiles, map_path, self.plot_style)
                    plots_written += 1
                except Exception as exc:
                    logger.debug("Map plot skipped for %s: %s", bufr_path.name, exc)

        self._append_xlsx(profiles)
        self._append_fields_dictionary(profiles)
        if self.cfg.decode_mode == "full":
            for profile in profiles:
                station = profile.station_id or f"subset{profile.subset_index}"
                export_decoded_fields_csv(
                    profile,
                    out_dir / f"decoded_fields_{station}.csv",
                )
        self.state.mark(
            bufr_key,
            status="completed",
            obs_date=obs_date,
            cycle=cycle,
            profiles_decoded=len(profiles),
            plots_written=plots_written,
        )
        return {"status": "completed", "profiles": len(profiles), "plots": plots_written}

    def run(
        self,
        files: list[Path],
        *,
        progress_every: int = 25,
    ) -> dict:
        self._processed = 0
        self._profiles = 0
        self._plots = 0
        self._skipped = 0
        self._errors = 0

        workers = max(1, self.plot_style.workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self._process_file, path): path for path in files}
            for future in as_completed(futures):
                bufr_path = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"status": "error", "error": str(exc)}
                    logger.exception("Unhandled error for %s", bufr_path)

                with self._stats_lock:
                    self._processed += 1
                    status = result.get("status")
                    if status == "completed":
                        self._profiles += result.get("profiles", 0)
                        self._plots += result.get("plots", 0)
                    elif status == "skipped":
                        self._skipped += 1
                    elif status in {"error", "empty"}:
                        self._errors += 1

                    if progress_every > 0 and self._processed % progress_every == 0:
                        logger.info(
                            "Batch progress: files=%s profiles=%s plots=%s skipped=%s errors=%s render_state=%s",
                            self._processed,
                            self._profiles,
                            self._plots,
                            self._skipped,
                            self._errors,
                            self.state.summary(),
                        )

        return {
            "files_processed": self._processed,
            "profiles_decoded": self._profiles,
            "plots_written": self._plots,
            "skipped": self._skipped,
            "errors": self._errors,
            "render_state": self.state.summary(),
            "outputs_dir": str(self.cfg.outputs_dir),
            "xlsx": str(self._xlsx_path()) if self.plot_style.export_xlsx else None,
        }
