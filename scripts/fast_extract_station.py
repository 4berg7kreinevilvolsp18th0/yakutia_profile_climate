"""Быстрая расшифровка локальных BUFR для одной станции.

Отличия от обычного station-profiles:
- ProcessPool (настоящий параллелизм CPU, не потоки)
- докачка: пропускает уже извлечённые source_file
- тихий режим (глушит print/шум pybufrkit)
- checkpoint каждые N файлов без полного XLSX до конца
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gdex_bufr.batch_render import list_bufr_files
from gdex_bufr.config import load_config
from gdex_bufr.profile_climate.config import load_profile_climate_config
from gdex_bufr.profile_climate.export import export_all, export_checkpoint
from gdex_bufr.profile_climate.extract import normalize_station_id

logger = logging.getLogger(__name__)

# Глобалы воркера (инициализируются один раз на процесс)
_WORKER: dict = {}


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _worker_init(
    station_id: str,
    station_name: str,
    pressure_top: float,
    min_levels: int,
    min_inv: float,
    decode_mode: str,
    tables_dir: str,
    export_dir: str,
) -> None:
    # Глушим шум pybufrkit в дочернем процессе
    logging.getLogger().setLevel(logging.ERROR)
    for name in list(logging.root.manager.loggerDict):
        if "bufr" in name.lower() or "pybufrkit" in name.lower():
            logging.getLogger(name).setLevel(logging.CRITICAL)
    logging.getLogger("PyBufrKit").setLevel(logging.CRITICAL)

    from gdex_bufr.bufr_adapter import _make_decoder, init_decoder_tables

    registry = init_decoder_tables({
        "directory": tables_dir,
        "wmo_version": "latest",
        "master_table_version": 43,
        "export_dir": export_dir,
        "export_on_update": False,
    })
    _WORKER["station_id"] = station_id
    _WORKER["station_name"] = station_name
    _WORKER["pressure_top"] = pressure_top
    _WORKER["min_levels"] = min_levels
    _WORKER["min_inv"] = min_inv
    _WORKER["decode_mode"] = decode_mode
    _WORKER["registry"] = registry
    _WORKER["decoder"] = _make_decoder(registry)


def _decode_one(bufr_path: str) -> tuple[str, list[dict], list[dict], str | None]:
    """Возвращает (path, long_rows, metrics_rows, error)."""
    from gdex_bufr.bufr_adapter import decode_bufr_file
    from gdex_bufr.profile_climate.extract import process_profile

    path = Path(bufr_path)
    try:
        # pybufrkit пишет «Continuing on next message...» в stdout
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            profiles = decode_bufr_file(
                path,
                station_id=_WORKER["station_id"],
                max_profiles=1,
                registry=_WORKER["registry"],
                decode_mode=_WORKER["decode_mode"],
                decoder=_WORKER["decoder"],
            )
        long_rows: list[dict] = []
        metrics_rows: list[dict] = []
        for profile in profiles:
            rows, metric = process_profile(
                profile,
                station_name=_WORKER["station_name"],
                pressure_top_hpa=_WORKER["pressure_top"],
                min_levels_to_500=_WORKER["min_levels"],
                min_inversion_delta_c=_WORKER["min_inv"],
            )
            long_rows.extend(rows)
            metrics_rows.append(metric)
        return bufr_path, long_rows, metrics_rows, None
    except Exception as exc:  # noqa: BLE001
        return bufr_path, [], [], str(exc)


def _load_done_sources(metrics_path: Path) -> set[str]:
    if not metrics_path.exists():
        return set()
    done: set[str] = set()
    with metrics_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            src = row.get("source_file") or ""
            if src:
                done.add(Path(src).name)
                done.add(src)
    return done


def _resolve_station(pc_cfg, station_arg: str) -> tuple[str, str, str]:
    if station_arg.isdigit():
        station = pc_cfg.station_by_id(station_arg)
        if station:
            return station.station_id, station.slug, station.name
        return normalize_station_id(station_arg), station_arg, station_arg
    station = pc_cfg.station_by_slug(station_arg)
    if station:
        return station.station_id, station.slug, station.name
    return station_arg, station_arg, station_arg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Быстрая расшифровка станции из локальных BUFR")
    p.add_argument("--config", default="gdex_config.yaml")
    p.add_argument("--profile-config", default="profile_climate_config.yaml")
    p.add_argument("--station", default="aldan", help="WMO ID или slug")
    p.add_argument("--start-date", default="1999-10-01")
    p.add_argument("--end-date", default="2026-07-08")
    p.add_argument("--cycles", default="00,12")
    p.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    p.add_argument("--checkpoint-every", type=int, default=200)
    p.add_argument("--fresh", action="store_true", help="Игнорировать уже извлечённые файлы")
    p.add_argument("--limit-files", type=int)
    p.add_argument("--output", default="")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    app_cfg = load_config(args.config)
    pc_cfg = load_profile_climate_config(args.profile_config)
    station_id, station_slug, station_name = _resolve_station(pc_cfg, args.station)
    start = _parse_date(args.start_date)
    end = _parse_date(args.end_date)
    cycles = [c.strip().zfill(2)[-2:] for c in args.cycles.split(",") if c.strip()]
    output_dir = Path(args.output or f"gdex_outputs/profile_climate/{station_slug}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "profile_metrics.csv"

    files = list_bufr_files(
        app_cfg,
        start_date=start,
        end_date=end,
        cycles=cycles,
        limit=args.limit_files,
        only_completed_downloads=False,
    )

    done = set() if args.fresh else _load_done_sources(metrics_path)
    pending = []
    for path in files:
        if path.name in done or str(path) in done:
            continue
        pending.append(path)

    logger.info(
        "Быстрая расшифровка %s (%s): всего=%s, уже готово=%s, осталось=%s, workers=%s",
        station_name,
        station_id,
        len(files),
        len(files) - len(pending),
        len(pending),
        args.workers,
    )

    # Загружаем уже накопленные строки (для resume)
    long_rows: list[dict] = []
    metrics_rows: list[dict] = []
    if not args.fresh and metrics_path.exists():
        from gdex_bufr.profile_climate.cli import _load_csv_rows

        long_path = output_dir / "profiles_long.csv"
        metrics_rows = _load_csv_rows(metrics_path)
        long_rows = _load_csv_rows(long_path) if long_path.exists() else []
        logger.info("Загружено из checkpoint: profiles=%s levels=%s", len(metrics_rows), len(long_rows))

    config_info = {
        "station_id": station_id,
        "station_slug": station_slug,
        "station_name": station_name,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "pressure_top_hpa": pc_cfg.pressure_top_hpa,
        "cycles": cycles,
        "mode": "fast_process_pool",
    }

    if not pending:
        paths = export_all(long_rows, metrics_rows, output_dir, config_info=config_info)
        print(json.dumps({"profiles": len(metrics_rows), "levels": len(long_rows), "outputs": paths}, ensure_ascii=False, indent=2))
        return 0 if metrics_rows else 1

    processed = 0
    errors = 0
    found = 0

    tables_dir = Path(app_cfg.bufr_tables.directory)
    if not tables_dir.is_absolute():
        tables_dir = (ROOT / tables_dir).resolve()
    export_dir = Path(app_cfg.bufr_tables.export_dir)
    if not export_dir.is_absolute():
        export_dir = (ROOT / export_dir).resolve()

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_init,
        initargs=(
            station_id,
            station_name,
            float(pc_cfg.pressure_top_hpa),
            int(pc_cfg.min_levels_to_500),
            float(pc_cfg.min_inversion_delta_c),
            app_cfg.decode_mode,
            str(tables_dir),
            str(export_dir),
        ),
    ) as pool:
        futures = {pool.submit(_decode_one, str(path)): path for path in pending}
        for future in as_completed(futures):
            path = futures[future]
            try:
                _, file_long, file_metrics, err = future.result()
            except Exception as exc:  # noqa: BLE001
                errors += 1
                logger.warning("Сбой %s: %s", path.name, exc)
                processed += 1
                continue

            if err:
                errors += 1
                logger.debug("Ошибка %s: %s", path.name, err)
            if file_metrics:
                found += len(file_metrics)
                long_rows.extend(file_long)
                metrics_rows.extend(file_metrics)
            processed += 1

            if processed % args.checkpoint_every == 0 or processed == len(pending):
                export_checkpoint(long_rows, metrics_rows, output_dir, config_info=config_info)
                logger.info(
                    "[%s/%s] profiles=%s levels=%s found_this_run=%s errors=%s",
                    processed,
                    len(pending),
                    len(metrics_rows),
                    len(long_rows),
                    found,
                    errors,
                )

    paths = export_all(long_rows, metrics_rows, output_dir, config_info=config_info)
    print(json.dumps({
        "station_id": station_id,
        "files_total": len(files),
        "files_pending": len(pending),
        "files_processed": processed,
        "profiles": len(metrics_rows),
        "levels": len(long_rows),
        "found_this_run": found,
        "errors": errors,
        "outputs": paths,
    }, ensure_ascii=False, indent=2))
    return 0 if metrics_rows else 1


if __name__ == "__main__":
    # Нужно для Windows ProcessPool
    raise SystemExit(main())
