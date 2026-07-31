"""Скачивание BUFR и извлечение профилей станции Алдан (WMO 31004).

BUFR-файлы GDEX содержат все станции мира — отдельно «скачать только Алдан»
с сервера нельзя (сервер отдаёт глобальный файл целиком).

Режим по умолчанию — потоковый (--stream): для каждого файла
  скачать глобальный BUFR → извлечь Алдан → удалить BUFR с диска.
Так на диске остаются только CSV/XLSX Алдана, а не сотни ГБ глобальных данных.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from datetime import date, datetime
from pathlib import Path

import requests
import urllib3

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from gdex_bufr.bufr_adapter import decode_bufr_file
from gdex_bufr.bufr_tables import get_registry
from gdex_bufr.config import load_config
from gdex_bufr.downloader import PoliteDownloader, resolve_ssl_verify
from gdex_bufr.manifest import build_manifest_for_config, generate_manifest, manifest_stats, save_manifest
from gdex_bufr.profile_climate.cli import cmd_station_profiles
from gdex_bufr.profile_climate.config import load_profile_climate_config
from gdex_bufr.profile_climate.export import export_all
from gdex_bufr.profile_climate.extract import process_profile

STATION_ID = "31004"
STATION_SLUG = "aldan"
STATION_NAME = "Aldan"
DATASET_START = date(1999, 10, 1)


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _clamp_start(start: date) -> date:
    if start < DATASET_START:
        logging.warning(
            "Датасет d351000 начинается с %s; start-date скорректирован.",
            DATASET_START.isoformat(),
        )
        return DATASET_START
    return start


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Скачать BUFR и извлечь температурные профили Алдана (31004)",
    )
    parser.add_argument("--config", default="gdex_config.yaml", help="gdex_config.yaml")
    parser.add_argument(
        "--profile-config",
        default="profile_climate_config.yaml",
        help="profile_climate_config.yaml",
    )
    parser.add_argument("--start-date", help="YYYY-MM-DD (не раньше 1999-10-01)")
    parser.add_argument("--end-date", help="YYYY-MM-DD")
    parser.add_argument(
        "--cycles",
        default="00,12",
        help="Синоптические сроки через запятую (по умолчанию 00,12)",
    )
    parser.add_argument("--limit-files", type=int, help="Ограничить число BUFR-файлов")
    parser.add_argument(
        "--output",
        default="gdex_outputs/profile_climate/aldan",
        help="Каталог CSV/XLSX для Алдана",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--manifest-only",
        action="store_true",
        help="Только построить manifest",
    )
    mode.add_argument(
        "--download-only",
        action="store_true",
        help="Только скачать по существующему manifest",
    )
    mode.add_argument(
        "--extract-only",
        action="store_true",
        help="Только извлечь Алдан из уже скачанных BUFR",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Фоновая докачка (как gdex_bufr download --daemon); только для --keep-bufr",
    )
    parser.add_argument(
        "--keep-bufr",
        action="store_true",
        help="Классический режим: скачать все глобальные BUFR и хранить на диске",
    )
    return parser


def _resolve_dates(args: argparse.Namespace, pc_cfg) -> tuple[date, date]:
    start = _parse_date(args.start_date) if args.start_date else pc_cfg.start_date
    end = _parse_date(args.end_date) if args.end_date else pc_cfg.end_date
    return _clamp_start(start), end


def _run_manifest(app_cfg, start: date, end: date, cycles: str) -> dict:
    app_cfg.start_date = start
    app_cfg.end_date = end
    app_cfg.synoptic_hours = [
        cycle.strip().zfill(2)[-2:]
        for cycle in cycles.split(",")
        if cycle.strip()
    ]
    entries = build_manifest_for_config(app_cfg)
    stats = manifest_stats(entries)
    save_manifest(entries, app_cfg.manifest_path)
    logging.info("Manifest: %s (%s файлов)", app_cfg.manifest_path, stats["files_total"])
    return stats


def _run_download(app_cfg, args: argparse.Namespace) -> dict:
    if not app_cfg.manifest_path.exists():
        raise SystemExit(f"Manifest не найден: {app_cfg.manifest_path}. Сначала запустите без --download-only.")
    downloader = PoliteDownloader(app_cfg)
    if args.daemon:
        downloader.run_daemon(app_cfg.manifest_path)
        return downloader.state.summary()
    return downloader.run_from_manifest(app_cfg.manifest_path, limit=args.limit_files)


def _stream_download_extract(app_cfg, pc_cfg, args: argparse.Namespace, start: date, end: date) -> dict:
    """Потоково: скачать глобальный BUFR → извлечь Алдан → удалить файл."""
    app_cfg.start_date = start
    app_cfg.end_date = end
    entries = generate_manifest(app_cfg)
    cycles = [c.strip().zfill(2)[-2:] for c in args.cycles.split(",") if c.strip()]
    entries = [e for e in entries if str(e.cycle).zfill(2)[-2:] in cycles]
    if args.limit_files:
        entries = entries[: args.limit_files]

    registry = get_registry()
    ssl_verify = resolve_ssl_verify(app_cfg)
    pressure_top = float(pc_cfg.pressure_top_hpa)
    output_dir = Path(args.output)
    tmp_dir = Path(tempfile.mkdtemp(prefix="aldan_bufr_"))

    session = requests.Session()
    session.headers.update({"User-Agent": app_cfg.user_agent})

    long_rows: list[dict] = []
    metrics_rows: list[dict] = []
    decoded_rows: list[dict] = []
    element_rows: list[dict] = []
    downloaded = 0
    skipped_404 = 0
    failed = 0
    profiles_found = 0

    total = len(entries)
    logging.info("Потоковый режим: %s файлов к обработке (BUFR удаляются после извлечения)", total)

    max_retries = max(1, app_cfg.max_retries)

    for idx, entry in enumerate(entries, start=1):
        tmp_path = tmp_dir / entry.filename
        status = "failed"
        for attempt in range(1, max_retries + 1):
            try:
                resp = session.get(entry.url, timeout=app_cfg.download_timeout_seconds, verify=ssl_verify)
                if resp.status_code == 404:
                    status = "not_found"
                    break
                if resp.status_code == 200 and resp.content:
                    tmp_path.write_bytes(resp.content)
                    status = "ok"
                    break
                logging.debug("HTTP %s (%s/%s): %s", resp.status_code, attempt, max_retries, entry.url)
            except requests.RequestException as exc:
                logging.debug("Ошибка загрузки (%s/%s) %s: %s", attempt, max_retries, entry.url, exc)
            if attempt < max_retries:
                time.sleep(app_cfg.backoff_base_seconds * attempt)

        if status == "not_found":
            skipped_404 += 1
            continue
        if status != "ok":
            failed += 1
            logging.warning("Не удалось скачать после %s попыток: %s", max_retries, entry.url)
            continue
        downloaded += 1

        try:
            profiles = decode_bufr_file(
                tmp_path,
                station_id=STATION_ID,
                registry=registry,
                decode_mode=app_cfg.decode_mode,
            )
            for profile in profiles:
                rows, metric, decoded, elements = process_profile(
                    profile,
                    station_name=STATION_NAME,
                    pressure_top_hpa=pressure_top,
                    min_levels_to_500=pc_cfg.min_levels_to_500,
                    min_inversion_delta_c=pc_cfg.min_inversion_delta_c,
                )
                long_rows.extend(rows)
                metrics_rows.append(metric)
                decoded_rows.extend(decoded)
                element_rows.extend(elements)
                profiles_found += 1
        except Exception as exc:  # noqa: BLE001 — декодер может падать на отдельных файлах
            logging.warning("Ошибка декодирования %s: %s", tmp_path.name, exc)
        finally:
            tmp_path.unlink(missing_ok=True)

        if idx % 50 == 0 or idx == total:
            logging.info(
                "[%s/%s] downloaded=%s 404=%s failed=%s profiles=%s",
                idx, total, downloaded, skipped_404, failed, profiles_found,
            )

    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    paths = export_all(
        long_rows,
        metrics_rows,
        output_dir,
        decoded_rows=decoded_rows,
        element_rows=element_rows,
        config_info={
            "station_id": STATION_ID,
            "station_slug": STATION_SLUG,
            "station_name": STATION_NAME,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "pressure_top_hpa": pressure_top,
            "cycles": cycles,
            "mode": "stream_delete_after",
        },
    )
    return {
        "files_downloaded": downloaded,
        "skipped_404": skipped_404,
        "failed": failed,
        "profiles": profiles_found,
        "levels": len(long_rows),
        "outputs": paths,
    }


def _run_extract(app_cfg, pc_cfg, args: argparse.Namespace, start: date, end: date) -> int:
    ns = argparse.Namespace(
        station=STATION_SLUG,
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        pressure_top=pc_cfg.pressure_top_hpa,
        cycles=args.cycles,
        output=args.output,
        limit_files=args.limit_files,
        include_all_files=False,
    )
    return cmd_station_profiles(app_cfg, pc_cfg, ns)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    app_cfg = load_config(args.config)
    pc_cfg = load_profile_climate_config(args.profile_config)
    start, end = _resolve_dates(args, pc_cfg)

    logging.info(
        "Алдан %s (%s): %s — %s, cycles=%s",
        STATION_ID,
        STATION_NAME,
        start.isoformat(),
        end.isoformat(),
        args.cycles,
    )

    result: dict = {
        "station_id": STATION_ID,
        "station_slug": STATION_SLUG,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "cycles": args.cycles,
    }

    stream_mode = not (args.keep_bufr or args.manifest_only or args.download_only or args.extract_only)

    if stream_mode:
        logging.info("Режим: потоковый (скачать → извлечь Алдан → удалить BUFR)")
        result["stream"] = _stream_download_extract(app_cfg, pc_cfg, args, start, end)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["stream"]["profiles"] else 1

    if not args.download_only and not args.extract_only:
        result["manifest"] = _run_manifest(app_cfg, start, end, args.cycles)
        if args.manifest_only:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0

    if not args.manifest_only and not args.extract_only:
        result["download"] = _run_download(app_cfg, args)
        if args.download_only:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0

    if not args.manifest_only and not args.download_only:
        extract_code = _run_extract(app_cfg, pc_cfg, args, start, end)
        if extract_code != 0:
            return extract_code

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
