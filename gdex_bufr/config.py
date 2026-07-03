"""Загрузка конфигурации проекта."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from gdex_bufr.bufr_tables import BufrTablesConfig
from gdex_bufr.plot_config import PlotStyle, load_plot_style
from gdex_bufr.tables_manager import resolve_wmo_version


@dataclass
class AppConfig:
    dataset_id: str = "d351000"
    base_url: str = "https://data.rda.ucar.edu/d351000/bufr"
    obs_types: list[str] = field(default_factory=lambda: ["adpupa"])
    synoptic_hours: list[str] = field(default_factory=lambda: ["00", "06", "12", "18"])
    start_date: date = date(1999, 10, 1)
    end_date: date | None = None
    data_dir: Path = Path("./data/raw")
    state_db: Path = Path("./data/download_state.sqlite")
    manifest_path: Path = Path("./data/manifest.jsonl")
    max_concurrency: int = 1
    requests_per_minute: float = 12.0
    min_delay_seconds: float = 2.0
    delay_jitter_seconds: float = 1.5
    daily_byte_budget_gb: float = 20.0
    max_retries: int = 8
    failed_retry_cooldown_seconds: float = 3600.0
    backoff_base_seconds: float = 5.0
    backoff_max_seconds: float = 600.0
    cooldown_on_429_seconds: float = 900.0
    download_timeout_seconds: int = 180
    progress_every_n: int = 50
    daemon_idle_seconds: float = 30.0
    daemon_error_sleep_seconds: float = 60.0
    budget_reset_sleep_seconds: float = 3600.0
    user_agent: str = "gdex-bufr-downloader/1.0 (research)"
    meteo_parser_path: Path | None = None
    outputs_dir: Path = Path("./outputs")
    plots: PlotStyle = field(default_factory=PlotStyle)
    bufr_tables: BufrTablesConfig = field(default_factory=BufrTablesConfig)
    decode_mode: str = "adpupa"
    ssl_verify: bool = True
    ssl_cert_bundle: Path | None = None


def _parse_date(value: str | date | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def load_config(path: str | Path) -> AppConfig:
    """Читаю YAML-конфиг и возвращаю dataclass."""
    cfg_path = Path(path).resolve()
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

    end_date = _parse_date(raw.get("end_date"))
    if end_date is None:
        end_date = date.today()

    meteo_path = raw.get("meteo_parser_path")
    return AppConfig(
        dataset_id=str(raw.get("dataset_id", "d351000")),
        base_url=str(raw.get("base_url", "https://data.rda.ucar.edu/d351000/bufr")).rstrip("/"),
        obs_types=list(raw.get("obs_types") or ["adpupa"]),
        synoptic_hours=[str(h).zfill(2) for h in (raw.get("synoptic_hours") or ["00", "06", "12", "18"])],
        start_date=_parse_date(raw.get("start_date")) or date(1999, 10, 1),
        end_date=end_date,
        data_dir=Path(raw.get("data_dir", "./data/raw")),
        state_db=Path(raw.get("state_db", "./data/download_state.sqlite")),
        manifest_path=Path(raw.get("manifest_path", "./data/manifest.jsonl")),
        max_concurrency=int(raw.get("max_concurrency", 1)),
        requests_per_minute=float(raw.get("requests_per_minute", 12)),
        min_delay_seconds=float(raw.get("min_delay_seconds", 2.0)),
        delay_jitter_seconds=float(raw.get("delay_jitter_seconds", 1.5)),
        daily_byte_budget_gb=float(raw.get("daily_byte_budget_gb", 20.0)),
        max_retries=int(raw.get("max_retries", 8)),
        failed_retry_cooldown_seconds=float(raw.get("failed_retry_cooldown_seconds", 3600.0)),
        backoff_base_seconds=float(raw.get("backoff_base_seconds", 5.0)),
        backoff_max_seconds=float(raw.get("backoff_max_seconds", 600.0)),
        cooldown_on_429_seconds=float(raw.get("cooldown_on_429_seconds", 900.0)),
        download_timeout_seconds=int(raw.get("download_timeout_seconds", 180)),
        progress_every_n=int(raw.get("progress_every_n", 50)),
        daemon_idle_seconds=float(raw.get("daemon_idle_seconds", 30.0)),
        daemon_error_sleep_seconds=float(raw.get("daemon_error_sleep_seconds", 60.0)),
        budget_reset_sleep_seconds=float(raw.get("budget_reset_sleep_seconds", 3600.0)),
        user_agent=str(raw.get("user_agent", "gdex-bufr-downloader/1.0 (research)")),
        meteo_parser_path=Path(meteo_path).resolve() if meteo_path else None,
        outputs_dir=Path(raw.get("outputs_dir", "./outputs")),
        plots=load_plot_style(raw.get("plots")),
        bufr_tables=_load_bufr_tables(raw.get("bufr_tables")),
        decode_mode=str(raw.get("decode_mode", "adpupa")).strip().lower(),
        ssl_verify=bool(raw.get("ssl_verify", True)),
        ssl_cert_bundle=Path(raw["ssl_cert_bundle"]).resolve() if raw.get("ssl_cert_bundle") else None,
    )


def _load_bufr_tables(raw: dict[str, Any] | None) -> BufrTablesConfig:
    raw = raw or {}
    version = resolve_wmo_version(raw.get("wmo_version"))
    master = raw.get("master_table_version")
    return BufrTablesConfig(
        directory=Path(raw.get("directory", "./gdex_data/bufr_tables")),
        wmo_version=version,
        master_table_version=int(master) if master is not None else None,
        export_dir=Path(raw.get("export_dir", "./gdex_data/bufr_tables_export")),
        export_on_update=bool(raw.get("export_on_update", True)),
    )
