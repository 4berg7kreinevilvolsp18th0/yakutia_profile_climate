"""Быстрая расшифровка локальных BUFR для одной станции.

Отличия от обычного station-profiles:
- ProcessPool (настоящий параллелизм CPU, не потоки)
- докачка: пропускает уже извлечённые source_file
- тихий режим (глушит print/шум pybufrkit)
- checkpoint каждые N файлов без полного XLSX до конца
- decoded_levels / debufr_elements пишутся потоково (без OOM)

Важно для Windows spawn: на уровне модуля только stdlib + sys.path.
Импорты gdex_bufr — только внутри main().
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


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

    os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from gdex_bufr.batch_render import list_bufr_files
    from gdex_bufr.config import load_config
    from gdex_bufr.meteo_parser_bridge import ensure_meteo_parser_import
    from gdex_bufr.profile_climate.config import load_profile_climate_config
    from gdex_bufr.profile_climate.export import (
        DECODED_LEVEL_COLUMNS,
        DEBUFR_ELEMENT_COLUMNS,
        append_csv,
        export_all,
        export_checkpoint,
        write_field_types_csv,
    )
    from gdex_bufr.profile_climate.extract import normalize_station_id
    from gdex_bufr.profile_climate.fast_worker import decode_one, worker_init

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

    app_cfg = load_config(args.config)
    ensure_meteo_parser_import(app_cfg.meteo_parser_path)
    pc_cfg = load_profile_climate_config(args.profile_config)
    station_id, station_slug, station_name = _resolve_station(pc_cfg, args.station)
    start = _parse_date(args.start_date)
    end = _parse_date(args.end_date)
    cycles = [c.strip().zfill(2)[-2:] for c in args.cycles.split(",") if c.strip()]
    output_dir = Path(args.output or f"gdex_outputs/результаты-{station_slug}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "profile_metrics.csv"
    decoded_path = output_dir / "decoded_levels.csv"
    elements_path = output_dir / "debufr_elements.csv"

    if args.fresh:
        for name in (
            "profiles_long.csv",
            "profile_metrics.csv",
            "decoded_levels.csv",
            "debufr_elements.csv",
            "monthly_summary.csv",
            "station_summary.csv",
            "summary.json",
            "field_types.csv",
            "profile_climate.xlsx",
        ):
            (output_dir / name).unlink(missing_ok=True)
        for old_xlsx in output_dir.glob("*_profile_climate_*.xlsx"):
            old_xlsx.unlink(missing_ok=True)

    files = list_bufr_files(
        app_cfg,
        start_date=start,
        end_date=end,
        cycles=cycles,
        limit=args.limit_files,
        only_completed_downloads=False,
    )

    done = set() if args.fresh else _load_done_sources(metrics_path)
    pending = [path for path in files if path.name not in done and str(path) not in done]

    logger.info(
        "Быстрая расшифровка %s (%s): всего=%s, уже готово=%s, осталось=%s, workers=%s",
        station_name,
        station_id,
        len(files),
        len(files) - len(pending),
        len(pending),
        args.workers,
    )

    long_rows: list[dict] = []
    metrics_rows: list[dict] = []
    decoded_streamed = 0
    elements_streamed = 0
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
        "mode": "fast_process_pool_stream",
    }

    write_field_types_csv(output_dir)

    if not pending:
        paths = export_all(long_rows, metrics_rows, output_dir, config_info=config_info)
        print(json.dumps({"profiles": len(metrics_rows), "levels": len(long_rows), "outputs": paths}, ensure_ascii=False, indent=2))
        return 0 if metrics_rows else 1

    processed = 0
    errors = 0
    found = 0
    batch_size = max(args.workers * 4, 40)

    tables_dir = Path(app_cfg.bufr_tables.directory)
    if not tables_dir.is_absolute():
        tables_dir = (ROOT / tables_dir).resolve()
    export_dir = Path(app_cfg.bufr_tables.export_dir)
    if not export_dir.is_absolute():
        export_dir = (ROOT / export_dir).resolve()

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=worker_init,
        initargs=(
            station_id,
            station_name,
            float(pc_cfg.pressure_top_hpa),
            int(pc_cfg.min_levels_to_500),
            float(pc_cfg.min_inversion_delta_c),
            app_cfg.decode_mode,
            str(tables_dir),
            str(export_dir),
            str(ROOT),
        ),
    ) as pool:
        for batch_start in range(0, len(pending), batch_size):
            batch = pending[batch_start: batch_start + batch_size]
            futures = {pool.submit(decode_one, str(path)): path for path in batch}
            for future in as_completed(futures):
                path = futures[future]
                try:
                    _, file_long, file_metrics, file_decoded, file_elements, err = future.result()
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
                    if file_decoded:
                        append_csv(decoded_path, file_decoded, DECODED_LEVEL_COLUMNS)
                        decoded_streamed += len(file_decoded)
                    if file_elements:
                        append_csv(elements_path, file_elements, DEBUFR_ELEMENT_COLUMNS)
                        elements_streamed += len(file_elements)
                processed += 1

                if processed % args.checkpoint_every == 0 or processed == len(pending):
                    export_checkpoint(long_rows, metrics_rows, output_dir, config_info=config_info)
                    write_field_types_csv(output_dir)
                    logger.info(
                        "[%s/%s] profiles=%s levels=%s decoded=%s elements=%s found=%s errors=%s",
                        processed,
                        len(pending),
                        len(metrics_rows),
                        len(long_rows),
                        decoded_streamed,
                        elements_streamed,
                        found,
                        errors,
                    )

    paths = export_all(long_rows, metrics_rows, output_dir, config_info=config_info)
    paths["decoded_levels"] = str(decoded_path) if decoded_path.exists() else ""
    paths["debufr_elements"] = str(elements_path) if elements_path.exists() else ""
    paths["field_types"] = str(output_dir / "field_types.csv")
    print(json.dumps({
        "station_id": station_id,
        "files_total": len(files),
        "files_pending": len(pending),
        "files_processed": processed,
        "profiles": len(metrics_rows),
        "levels": len(long_rows),
        "decoded_streamed": decoded_streamed,
        "elements_streamed": elements_streamed,
        "found_this_run": found,
        "errors": errors,
        "outputs": paths,
    }, ensure_ascii=False, indent=2))
    return 0 if metrics_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
