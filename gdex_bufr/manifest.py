"""Построение manifest для synoptic BUFR d351000."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import requests

from gdex_bufr.config import AppConfig

FILENAME_RE = re.compile(
    r"^gdas\.(?P<obs_type>[a-z]+)\.t(?P<cycle>\d{2})z\.(?P<obs_date>\d{8})\.bufr$"
)


@dataclass
class ManifestEntry:
    url: str
    local_path: str
    filename: str
    obs_date: str
    obs_type: str
    cycle: str
    expected_size: int | None = None
    source: str = "generated"


def _daterange(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def build_filename(obs_type: str, cycle: str, obs_date: date) -> str:
    return f"gdas.{obs_type}.t{cycle}z.{obs_date.strftime('%Y%m%d')}.bufr"


def build_url(base_url: str, obs_date: date, filename: str) -> str:
    return f"{base_url}/{obs_date.year}/{filename}"


def build_local_path(data_dir: Path, obs_date: date, filename: str) -> Path:
    return data_dir / str(obs_date.year) / filename


def generate_manifest(cfg: AppConfig) -> list[ManifestEntry]:
    """Генерирую manifest по официальной схеме имён NCAR ds351.0."""
    end = cfg.end_date or date.today()
    entries: list[ManifestEntry] = []
    for obs_date in _daterange(cfg.start_date, end):
        for obs_type in cfg.obs_types:
            for cycle in cfg.synoptic_hours:
                filename = build_filename(obs_type, cycle, obs_date)
                local_path = build_local_path(cfg.data_dir, obs_date, filename)
                entries.append(
                    ManifestEntry(
                        url=build_url(cfg.base_url, obs_date, filename),
                        local_path=str(local_path),
                        filename=filename,
                        obs_date=obs_date.isoformat(),
                        obs_type=obs_type,
                        cycle=cycle,
                    )
                )
    return entries


def crawl_year_directory(
    cfg: AppConfig,
    year: int,
    session: requests.Session | None = None,
) -> list[ManifestEntry]:
    """Опционально: парсю HTML-листинг года с HTTPS/OSDF."""
    session = session or requests.Session()
    url = f"{cfg.base_url}/{year}/"
    response = session.get(url, timeout=60, headers={"User-Agent": cfg.user_agent})
    response.raise_for_status()
    hrefs = re.findall(r'href="([^"]+\.bufr)"', response.text, flags=re.IGNORECASE)
    entries: list[ManifestEntry] = []
    for href in hrefs:
        filename = Path(href).name
        match = FILENAME_RE.match(filename)
        if not match:
            continue
        obs_type = match.group("obs_type")
        cycle = match.group("cycle")
        obs_date = datetime_from_yyyymmdd(match.group("obs_date"))
        if obs_type not in cfg.obs_types:
            continue
        if cycle not in cfg.synoptic_hours:
            continue
        if obs_date < cfg.start_date or obs_date > (cfg.end_date or date.today()):
            continue
        local_path = build_local_path(cfg.data_dir, obs_date, filename)
        entries.append(
            ManifestEntry(
                url=f"{cfg.base_url}/{year}/{filename}",
                local_path=str(local_path),
                filename=filename,
                obs_date=obs_date.isoformat(),
                obs_type=obs_type,
                cycle=cycle,
                source="crawled",
            )
        )
    return entries


def datetime_from_yyyymmdd(value: str) -> date:
    return date(int(value[0:4]), int(value[4:6]), int(value[6:8]))


def save_manifest(entries: list[ManifestEntry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")


def load_manifest(path: Path) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entries.append(ManifestEntry(**json.loads(line)))
    return entries


def manifest_stats(entries: list[ManifestEntry]) -> dict:
    total = len(entries)
    by_type: dict[str, int] = {}
    by_year: dict[str, int] = {}
    for entry in entries:
        by_type[entry.obs_type] = by_type.get(entry.obs_type, 0) + 1
        year = entry.obs_date[:4]
        by_year[year] = by_year.get(year, 0) + 1
    return {
        "files_total": total,
        "by_obs_type": by_type,
        "by_year": by_year,
        "date_min": min((e.obs_date for e in entries), default=None),
        "date_max": max((e.obs_date for e in entries), default=None),
    }


def build_manifest_for_config(cfg: AppConfig, *, crawl: bool = False) -> list[ManifestEntry]:
    if crawl:
        session = requests.Session()
        entries: list[ManifestEntry] = []
        for year in range(cfg.start_date.year, (cfg.end_date or date.today()).year + 1):
            entries.extend(crawl_year_directory(cfg, year, session=session))
        entries.sort(key=lambda e: (e.obs_date, e.cycle, e.obs_type))
        return entries
    return generate_manifest(cfg)
