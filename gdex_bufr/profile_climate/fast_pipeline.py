"""Понятный порядок массовой обработки: выбрать → декодировать → сохранить → экспортировать.

Порции сохраняются после каждого файла. Смена формул или параметров требует
новой папки, чтобы новый результат не смешивался со старым при продолжении.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


def positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Нужно положительное целое число")
    return number


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Возобновляемая расшифровка BUFR по станциям и регионам")
    p.add_argument("--config", default=str(ROOT / "gdex_config.yaml"))
    p.add_argument("--profile-config", default=str(ROOT / "profile_climate_config.yaml"))
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--station", default="", help="Slug или ВМО; несколько через запятую")
    selection.add_argument("--region", default="", help="По умолчанию default_region из каталога")
    p.add_argument("--start-date", help="YYYY-MM-DD; по умолчанию из YAML")
    p.add_argument("--end-date", help="YYYY-MM-DD; по умолчанию из YAML")
    p.add_argument("--cycles", help="Сроки через запятую; по умолчанию из YAML")
    p.add_argument("--workers", type=positive_integer, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    p.add_argument("--max-inflight", type=positive_integer, help="Размер очереди, обычно 2 × workers")
    p.add_argument("--retries", type=int, default=1, help="Число повторов неудачного файла")
    p.add_argument("--checkpoint-every", type=positive_integer, default=200,
                   help="Интервал вывода прогресса; данные сохраняются после каждого файла")
    p.add_argument("--output-mode", choices=("climate", "decoded", "audit"), default="climate")
    p.add_argument("--decoder", choices=("pybufrkit", "eccodes", "ncepbufr"), default="pybufrkit",
                   help="Движок BUFR; нативные варианты экспериментальные, нужны отдельные зависимости")
    p.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True,
                   help="Считать ошибку BUFR-сообщения ошибкой всего файла")
    p.add_argument("--xlsx", action="store_true", help="Дополнительно собрать Excel по станциям")
    p.add_argument("--fast-station-filter", action=argparse.BooleanOptionalAction, default=True,
                   help="Быстрый поиск индексов ВМО; отключение для контрольного сравнения")
    p.add_argument("--working-copy", action="store_true", help="Сохранить совместимую копию profiles_working.csv")
    p.add_argument("--export-only", action="store_true", help="Собрать CSV из готовых порций без декодирования")
    p.add_argument("--fresh", action="store_true", help="Новый прогон в отдельной подпапке; прежние данные сохраняются")
    p.add_argument("--limit-files", type=positive_integer)
    p.add_argument("--output", default="")
    p.add_argument("--actual", action="store_true", help="Совместимая папка актуальное и profiles_working.csv")
    p.add_argument("--input-dir", default="")
    p.add_argument("--pressure-top-hpa", type=float)
    p.add_argument("--min-levels", type=positive_integer)
    p.add_argument("--min-inversion-delta-c", type=float)
    p.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p


def select_stations(config, station_arg: str, region: str):
    """Slug включает все исторические ВМО-индексы; точный индекс — только себя."""
    if not station_arg:
        stations = config.stations_in_region(region or config.default_region)
    else:
        stations = []
        for token in station_arg.split(","):
            token = token.strip()
            station = config.station_by_id(token) if token.isdigit() else config.station_by_slug(token)
            if station is None:
                raise ValueError(f"Станция {token!r} отсутствует в каталоге YAML")
            stations.extend([station] if token.isdigit() else [s for s in config.stations if s.slug == token])
    if not stations:
        raise ValueError("Нет включённых станций в выбранном регионе")
    return list({s.station_id: s for s in stations}.values())


def process_pending(paths, store, options: dict, args) -> None:
    """Скользящая очередь: медленный файл не блокирует следующую пачку работ."""
    from .decode_worker import initialize, decode_file
    from .run_store import fingerprint

    started = time.perf_counter()
    processed = 0
    iterator = iter(paths)
    # ecCodes кэширует локальные таблицы глобально. Новый процесс на файл
    # исключает использование таблицы предыдущего файла с тем же номером.
    lifecycle = {"max_tasks_per_child": 1} if args.decoder == "eccodes" else {}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(options,), **lifecycle) as pool:
        futures = {}
        def submit(path, attempt=0):
            futures[pool.submit(decode_file, str(path))] = (path, fingerprint(path), attempt)
        for _ in range(args.max_inflight or args.workers * 2):
            path = next(iterator, None)
            if path is not None:
                submit(path)
        while futures:
            ready, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in ready:
                path, before, attempt = futures.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"error": repr(exc)}
                store.record(path, before, result)
                if not store.completed(path):
                    logger.warning("Ошибка %s: %s", path.name, result.get("error") or "источник изменился")
                    if attempt < args.retries:
                        submit(path, attempt + 1)
                        continue
                processed += 1
                if processed % args.checkpoint_every == 0 or processed == len(paths):
                    logger.info("[%s/%s] %.2f файла/с; ошибки=%s", processed, len(paths),
                                processed / max(time.perf_counter() - started, .001), len(store.failures()))
                path = next(iterator, None)
                if path is not None:
                    submit(path)


def run(args) -> int:
    from gdex_bufr.batch_render import list_bufr_files
    from gdex_bufr.config import load_config
    from .config import load_profile_climate_config
    from .run_store import RunStore
    from .stream_export import export_store
    from run_fast_extract import _acquire_output_lock, _release_output_lock

    cfg = load_config(args.config)
    from gdex_bufr.decoder_backends import backend_version
    try:
        selected_version = backend_version(args.decoder)
    except ImportError as exc:
        raise ValueError(f"Декодер {args.decoder} не установлен в этом Python; см. decoder-environment.yml") from exc
    if args.decoder != "pybufrkit" and args.output_mode == "audit":
        raise ValueError("Нативные адаптеры поддерживают climate/decoded, но не полный дамп audit")
    if args.decoder != "pybufrkit" and (cfg.decode_mode != "adpupa" or not args.strict):
        raise ValueError("Нативные адаптеры пока требуют decode_mode=adpupa и --strict")
    pc = load_profile_climate_config(args.profile_config)
    if args.retries < 0:
        raise ValueError("--retries не может быть отрицательным")
    stations = select_stations(pc, args.station, args.region)
    names = {s.station_id: s.name for s in stations}
    slugs = {s.station_id: s.slug for s in stations}
    start = datetime.strptime(args.start_date, "%Y-%m-%d").date() if args.start_date else pc.start_date
    end = datetime.strptime(args.end_date, "%Y-%m-%d").date() if args.end_date else pc.end_date
    if start > end:
        raise ValueError("Дата начала позже даты окончания")
    cycles = [s.strip().zfill(2) for s in args.cycles.split(",")] if args.cycles else pc.cycles
    if not cycles or not set(cycles) <= {"00", "06", "12", "18"}:
        raise ValueError("Допустимые сроки: 00,06,12,18")
    cfg.data_dir = (ROOT / (args.input_dir or cfg.data_dir)).resolve()
    tables = asdict(cfg.bufr_tables)
    for key in ("directory", "export_dir"):
        tables[key] = str((ROOT / tables[key]).resolve())
    tables["export_on_update"] = False
    science = {
        "pressure_top_hpa": args.pressure_top_hpa if args.pressure_top_hpa is not None else pc.pressure_top_hpa,
        "min_levels_to_500": args.min_levels or pc.min_levels_to_500,
        "min_inversion_delta_c": args.min_inversion_delta_c if args.min_inversion_delta_c is not None else pc.min_inversion_delta_c,
        "confirm_drop_levels": pc.confirm_drop_levels, "confirm_depth_hpa": pc.confirm_depth_hpa,
        "min_drop_delta_c": pc.min_drop_delta_c,
    }
    import math
    if (not all(math.isfinite(v) for v in science.values())
            or not 0 < science["pressure_top_hpa"] <= 1100
            or science["min_inversion_delta_c"] < 0
            or science["min_levels_to_500"] < 1
            or science["confirm_drop_levels"] < 1
            or science["confirm_depth_hpa"] < 0
            or science["min_drop_delta_c"] < 0):
        raise ValueError("Проверьте числовые параметры давления и инверсии")
    digest = hashlib.sha256()
    for module in sorted((ROOT / "gdex_bufr").rglob("*.py")):
        digest.update(module.relative_to(ROOT).as_posix().encode())
        digest.update(module.read_bytes())
    # Одинаковый путь к справочнику ещё не означает одинаковые таблицы BUFR.
    # Их обновление также требует отдельного прогона.
    table_digest = hashlib.sha256()
    table_root = Path(tables["directory"])
    for table in sorted(p for p in table_root.rglob("*") if p.is_file()):
        table_digest.update(table.relative_to(table_root).as_posix().encode())
        table_digest.update(table.read_bytes())
    signature = {"stations": [asdict(s) for s in stations], "science": science,
                 "input": str(cfg.data_dir), "start": str(start), "end": str(end),
                 "cycles": cycles, "output_mode": args.output_mode, "strict": args.strict,
                 "fast_station_filter": args.fast_station_filter,
                 "decoder": args.decoder, "decoder_version": selected_version,
                 "tables": tables, "tables_sha256": table_digest.hexdigest(), "decode_mode": cfg.decode_mode,
                 "code_sha256": digest.hexdigest(), "pybufrkit": version("pybufrkit")}
    name = next(iter(slugs.values())) if len(set(slugs.values())) == 1 else (args.region or pc.default_region)
    output = Path(args.output) if args.output else ROOT / "gdex_outputs" / "runs" / name
    if args.actual and not args.output:
        output = ROOT / "gdex_outputs" / "актуальное"
    output = output.resolve()
    if args.fresh:
        output = output / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    if not (output / "run.sqlite").exists() and (output / "profiles_long.csv").exists():
        raise ValueError("Здесь старый формат результатов. Выберите --fresh или новую --output; старые данные сохранятся.")
    if args.export_only and not (output / "run.sqlite").exists():
        raise ValueError("В папке ещё нет журнала run.sqlite")
    output.mkdir(parents=True, exist_ok=True)
    _acquire_output_lock(output)
    store = None
    try:
        store = RunStore(output, signature)
        files = list_bufr_files(cfg, start_date=start, end_date=end, cycles=cycles,
                               limit=args.limit_files, only_completed_downloads=False)
        pending = [p for p in files if not store.completed(p)]
        logger.info("Станций=%s, файлов=%s, осталось=%s. Результаты: %s", len(names), len(files), len(pending), output)
        if not files and not args.export_only:
            raise ValueError("Не найдено BUFR-файлов в выбранном диапазоне")
        if pending and not args.export_only:
            options = {"names": names, "elevations": {s.station_id: s.elevation_m for s in pc.stations if s.elevation_m is not None},
                       "tables": tables, "science": science, "decode_mode": cfg.decode_mode,
                       "output_mode": args.output_mode, "strict": args.strict,
                       "backend": args.decoder,
                       "fast_station_filter": args.fast_station_filter, "parts_dir": str(output / "parts")}
            process_pending(pending, store, options, args)
        summary = export_store(store, output, config=signature, station_slugs=slugs,
                               split=len(set(slugs.values())) > 1, output_mode=args.output_mode,
                               working_copy=args.working_copy or args.actual, xlsx=args.xlsx)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 2 if summary["errors"] else (0 if summary["profiles"] else 1)
    finally:
        if store is not None:
            store.close()
        _release_output_lock()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s: %(message)s")
    try:
        return run(args)
    except (ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 2
