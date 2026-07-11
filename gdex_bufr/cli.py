"""CLI для manifest, download, decode и plots."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

from gdex_bufr.batch_render import BatchRenderer, list_bufr_files
from gdex_bufr.bufr_adapter import decode_bufr_file, export_decoded_fields_csv, init_decoder_tables
from gdex_bufr.bufr_tables import get_registry, normalize_fxy
from gdex_bufr.config import AppConfig, load_config
from gdex_bufr.tables_manager import update_wmo_tables
from gdex_bufr.downloader import PoliteDownloader
from gdex_bufr.manifest import build_manifest_for_config, manifest_stats, save_manifest
from gdex_bufr.meteo_parser_bridge import load_meteo_parser_field_names
from gdex_bufr.plot_config import load_plot_style
from gdex_bufr.plots import PLOT_REGISTRY, render_plots
from gdex_bufr.station_plots import render_station_plots


def _parse_optional_date(value: str | None) -> date | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GDEX d351000 BUFR downloader and plotter")
    parser.add_argument("--config", default="gdex_config.yaml", help="Path to YAML config")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    sub = parser.add_subparsers(dest="command", required=True)

    manifest_cmd = sub.add_parser("manifest", help="Build manifest")
    manifest_cmd.add_argument("--crawl", action="store_true", help="Crawl year directories instead of generating names")
    manifest_cmd.add_argument("--dry-run", action="store_true", help="Print manifest stats only")
    manifest_cmd.add_argument("--start-date", help="Override start date YYYY-MM-DD")
    manifest_cmd.add_argument("--end-date", help="Override end date YYYY-MM-DD")

    download_cmd = sub.add_parser("download", help="Download files from manifest")
    download_cmd.add_argument("--limit-files", type=int, help="Download only N files")
    download_cmd.add_argument("--dry-run", action="store_true", help="Seed queue and print summary only")
    download_cmd.add_argument(
        "--daemon",
        action="store_true",
        help="Run continuously with auto-retry until queue is empty",
    )

    probe_cmd = sub.add_parser("probe-plots", help="Decode one BUFR and render plots")
    probe_cmd.add_argument("--file", help="Local BUFR file path")
    probe_cmd.add_argument("--download-one", action="store_true", help="Download one manifest file first")
    probe_cmd.add_argument(
        "--plots",
        default=None,
        help=f"Comma-separated plot types (default from config): {','.join(sorted(PLOT_REGISTRY))}",
    )

    batch_cmd = sub.add_parser("batch-plots", help="Decode BUFR files and render plots in batch")
    batch_cmd.add_argument("--start-date", help="Filter from YYYY-MM-DD")
    batch_cmd.add_argument("--end-date", help="Filter to YYYY-MM-DD")
    batch_cmd.add_argument("--limit-files", type=int, help="Process only N BUFR files")
    batch_cmd.add_argument(
        "--plots",
        default=None,
        help=f"Override plot types: {','.join(sorted(PLOT_REGISTRY))}",
    )
    batch_cmd.add_argument("--workers", type=int, help="Parallel workers (overrides config)")
    batch_cmd.add_argument("--max-profiles-per-file", type=int, help="Max radiosonde profiles per BUFR file")
    batch_cmd.add_argument("--no-skip-existing", action="store_true", help="Re-render even if already done")
    batch_cmd.add_argument("--dry-run", action="store_true", help="List files only, no decode/plot")

    station_cmd = sub.add_parser(
        "station-plots",
        help="Графики по станции WMO за одну дату (4 синоптических срока BUFR)",
    )
    station_cmd.add_argument("--station-id", required=True, help="WMO station id, e.g. 31977")
    station_cmd.add_argument("--date", required=True, help="Дата наблюдений YYYY-MM-DD")
    station_cmd.add_argument("--label", default="", help="Подпись станции для заголовков графиков")
    station_cmd.add_argument("--output-dir", help="Каталог вывода (по умолчанию outputs/stations/ID/DATE)")
    station_cmd.add_argument("--bufr-dir", help="Локальный кэш BUFR (не скачивать, если файлы уже есть)")
    station_cmd.add_argument("--tae", action="append", default=[], help="Файл TAE-3 (можно указать несколько раз)")
    station_cmd.add_argument("--lat", type=float, help="Широта для TAE")
    station_cmd.add_argument("--lon", type=float, help="Долгота для TAE")
    station_cmd.add_argument(
        "--plots",
        default=None,
        help=f"Типы графиков через запятую (по умолчанию из config): {','.join(sorted(PLOT_REGISTRY))}",
    )
    station_cmd.add_argument("--no-download", action="store_true", help="Не скачивать BUFR, только локальный кэш")

    check_cmd = sub.add_parser("check-decode", help="Compare pybufrkit API vs CLI decode")
    check_cmd.add_argument("--file", required=True, help="Local BUFR file path")

    tables_update_cmd = sub.add_parser("tables-update", help="Download/update WMO BUFR code tables")
    tables_update_cmd.add_argument("--version", default=None, help="WMO table version (default from config)")
    tables_update_cmd.add_argument("--overwrite", action="store_true", help="Force re-download")

    tables_export_cmd = sub.add_parser("tables-export", help="Export descriptors/code tables to CSV/JSON")
    tables_export_cmd.add_argument("--output-dir", help="Override export directory")

    lookup_cmd = sub.add_parser("lookup", help="Look up BUFR descriptor and code table")
    lookup_cmd.add_argument("--descriptor", required=True, help="Descriptor FXY, e.g. 012225")
    lookup_cmd.add_argument("--code-value", type=int, help="Decode coded value")

    station_profiles_cmd = sub.add_parser(
        "station-profiles",
        help="Извлечь температурные профили станции и экспортировать profile_climate CSV",
    )
    station_profiles_cmd.add_argument("--profile-config", default="profile_climate_config.yaml")
    station_profiles_cmd.add_argument("--station", help="WMO ID или slug станции (aldan, yakutsk)")
    station_profiles_cmd.add_argument("--start-date", help="YYYY-MM-DD")
    station_profiles_cmd.add_argument("--end-date", help="YYYY-MM-DD")
    station_profiles_cmd.add_argument("--pressure-top", type=float, help="Верхний уровень анализа, гПа (default 500)")
    station_profiles_cmd.add_argument("--cycles", help="Синоптические сроки через запятую, напр. 00,12")
    station_profiles_cmd.add_argument("--output", help="Каталог вывода profile_climate")
    station_profiles_cmd.add_argument("--limit-files", type=int, help="Ограничить число BUFR-файлов")
    station_profiles_cmd.add_argument("--workers", type=int, default=4, help="Параллельных потоков расшифровки")
    station_profiles_cmd.add_argument(
        "--include-all-files",
        action="store_true",
        help="Брать все локальные BUFR, не только completed в state_db",
    )

    monthly_plots_cmd = sub.add_parser(
        "monthly-profile-plots",
        help="Построить месячные PNG пучков температурных профилей",
    )
    monthly_plots_cmd.add_argument("--profile-config", default="profile_climate_config.yaml")
    monthly_plots_cmd.add_argument("--station", help="WMO ID или slug станции")
    monthly_plots_cmd.add_argument("--start-date", help="YYYY-MM-DD")
    monthly_plots_cmd.add_argument("--end-date", help="YYYY-MM-DD")
    monthly_plots_cmd.add_argument("--pressure-top", type=float, help="Верхний уровень анализа, гПа")
    monthly_plots_cmd.add_argument("--input", help="profiles_long.csv")
    monthly_plots_cmd.add_argument("--metrics", help="profile_metrics.csv")
    monthly_plots_cmd.add_argument("--output", help="Каталог PNG")

    discover_cmd = sub.add_parser(
        "discover-stations",
        help="Найти station_id и координаты в локальных BUFR-файлах",
    )
    discover_cmd.add_argument("--start-date", help="YYYY-MM-DD")
    discover_cmd.add_argument("--end-date", help="YYYY-MM-DD")
    discover_cmd.add_argument("--limit-files", type=int, help="Ограничить число BUFR-файлов")
    discover_cmd.add_argument("--max-profiles-per-file", type=int, default=50)
    discover_cmd.add_argument(
        "--include-all-files",
        action="store_true",
        help="Брать все локальные BUFR, не только completed в state_db",
    )

    return parser


def _apply_date_overrides(cfg: AppConfig, start_date: str | None, end_date: str | None) -> AppConfig:
    if start_date:
        cfg.start_date = _parse_optional_date(start_date) or cfg.start_date
    if end_date:
        cfg.end_date = _parse_optional_date(end_date) or cfg.end_date
    return cfg


def cmd_manifest(cfg: AppConfig, args: argparse.Namespace) -> int:
    cfg = _apply_date_overrides(cfg, args.start_date, args.end_date)
    entries = build_manifest_for_config(cfg, crawl=args.crawl)
    stats = manifest_stats(entries)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    save_manifest(entries, cfg.manifest_path)
    print(f"Manifest saved: {cfg.manifest_path} ({stats['files_total']} files)")
    return 0


def cmd_download(cfg: AppConfig, args: argparse.Namespace) -> int:
    if not cfg.manifest_path.exists():
        print("Manifest not found. Run: python -m gdex_bufr manifest", file=sys.stderr)
        return 2
    downloader = PoliteDownloader(cfg)
    if args.dry_run:
        from gdex_bufr.manifest import load_manifest

        entries = load_manifest(cfg.manifest_path)
        downloader.seed_queue(entries)
        print(json.dumps(downloader.state.summary(), ensure_ascii=False, indent=2))
        return 0
    if args.daemon:
        downloader.run_daemon(cfg.manifest_path)
        print(json.dumps(downloader.state.summary(), ensure_ascii=False, indent=2))
        return 0
    result = downloader.run_from_manifest(cfg.manifest_path, limit=args.limit_files)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _resolve_probe_file(cfg: AppConfig, args: argparse.Namespace) -> Path:
    if args.file:
        return Path(args.file)
    if args.download_one:
        if not cfg.manifest_path.exists():
            build_entries = build_manifest_for_config(cfg)
            save_manifest(build_entries[:1], cfg.manifest_path)
        downloader = PoliteDownloader(cfg)
        result = downloader.run_from_manifest(cfg.manifest_path, limit=1)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        from gdex_bufr.manifest import load_manifest

        entry = load_manifest(cfg.manifest_path)[0]
        return Path(entry.local_path)
    raise SystemExit("Provide --file or --download-one")


def _tables_config_dict(cfg: AppConfig) -> dict:
    return {
        "directory": str(cfg.bufr_tables.directory),
        "wmo_version": cfg.bufr_tables.wmo_version,
        "master_table_version": cfg.bufr_tables.master_table_version,
        "export_dir": str(cfg.bufr_tables.export_dir),
        "export_on_update": cfg.bufr_tables.export_on_update,
    }


def cmd_tables_update(cfg: AppConfig, args: argparse.Namespace) -> int:
    version = args.version or cfg.bufr_tables.wmo_version
    result = update_wmo_tables(
        cfg.bufr_tables.directory,
        version,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if cfg.bufr_tables.export_on_update:
        registry = init_decoder_tables(_tables_config_dict(cfg))
        paths = registry.export_reference_files(cfg.bufr_tables.export_dir)
        print(json.dumps({"exported": paths}, ensure_ascii=False, indent=2))
    return 0


def cmd_tables_export(cfg: AppConfig, args: argparse.Namespace) -> int:
    registry = init_decoder_tables(_tables_config_dict(cfg))
    export_dir = Path(args.output_dir) if args.output_dir else cfg.bufr_tables.export_dir
    paths = registry.export_reference_files(export_dir)
    print(json.dumps(paths, ensure_ascii=False, indent=2))
    return 0


def cmd_lookup(cfg: AppConfig, args: argparse.Namespace) -> int:
    registry = init_decoder_tables(_tables_config_dict(cfg))
    fxy = normalize_fxy(args.descriptor)
    info = registry.lookup_descriptor(fxy)
    payload: dict = {
        "descriptor": fxy,
        "name": info.name,
        "name_ru": info.name_ru,
        "unit": info.unit,
        "scale": info.scale,
        "reference": info.reference,
        "nbits": info.nbits,
        "kind": info.kind,
    }
    if args.code_value is not None:
        payload["code_value"] = args.code_value
        payload["code_text"] = registry.decode_code_value(fxy, args.code_value)
        payload["flag_bits"] = registry.decode_flag_bits(fxy, args.code_value)
    if info.kind in {"code", "flag"}:
        payload["code_table_sample"] = registry.code_table_entries(fxy)[:12]
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_probe_plots(cfg: AppConfig, args: argparse.Namespace) -> int:
    bufr_path = _resolve_probe_file(cfg, args)
    if not bufr_path.exists():
        print(f"File not found: {bufr_path}", file=sys.stderr)
        return 2
    registry = init_decoder_tables(_tables_config_dict(cfg))
    profiles = decode_bufr_file(
        bufr_path,
        max_profiles=5,
        registry=registry,
        decode_mode=cfg.decode_mode,
    )
    if not profiles:
        print("No profiles decoded", file=sys.stderr)
        return 1
    plot_types = [p.strip() for p in args.plots.split(",") if p.strip()] if args.plots else cfg.plots.plot_types
    outputs = render_plots(
        profiles[0],
        cfg.outputs_dir,
        plot_types,
        all_profiles=profiles,
        style=cfg.plots,
    )
    extra_exports = []
    if cfg.decode_mode == "full" and profiles:
        for profile in profiles:
            station = profile.station_id or f"subset{profile.subset_index}"
            path = export_decoded_fields_csv(
                profile,
                cfg.outputs_dir / f"decoded_fields_{station}.csv",
            )
            if path:
                extra_exports.append(str(path))

    print(json.dumps({
        "profiles": len(profiles),
        "outputs": [str(p) for p in outputs],
        "field_exports": extra_exports,
        "tables_ready": get_registry().is_ready(),
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_check_decode(args: argparse.Namespace) -> int:
    result = compare_decode_outputs(Path(args.file))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 1


def cmd_batch_plots(cfg: AppConfig, args: argparse.Namespace) -> int:
    plot_style = cfg.plots
    if args.plots:
        plot_style.plot_types = [p.strip() for p in args.plots.split(",") if p.strip()]
    if args.workers:
        plot_style.workers = args.workers
    if args.max_profiles_per_file is not None:
        plot_style.max_profiles_per_file = args.max_profiles_per_file
    if args.no_skip_existing:
        plot_style.skip_existing = False

    start = _parse_optional_date(args.start_date)
    end = _parse_optional_date(args.end_date)
    files = list_bufr_files(
        cfg,
        start_date=start,
        end_date=end,
        limit=args.limit_files,
        only_completed_downloads=True,
    )
    print(json.dumps({
        "files_to_process": len(files),
        "plot_types": plot_style.plot_types,
        "outputs_dir": str(cfg.outputs_dir),
        "workers": plot_style.workers,
    }, ensure_ascii=False, indent=2))

    if args.dry_run:
        for path in files[:20]:
            print(path)
        if len(files) > 20:
            print(f"... and {len(files) - 20} more")
        return 0

    if not files:
        print("No BUFR files found for the given filters.", file=sys.stderr)
        return 1

    renderer = BatchRenderer(cfg, plot_style)
    result = renderer.run(files, progress_every=cfg.progress_every_n)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_station_plots(cfg: AppConfig, args: argparse.Namespace) -> int:
    obs_date = _parse_optional_date(args.date)
    if obs_date is None:
        print("Invalid --date, use YYYY-MM-DD", file=sys.stderr)
        return 2

    plot_types = [p.strip() for p in args.plots.split(",") if p.strip()] if args.plots else None
    tae_files = [Path(p) for p in args.tae]
    bufr_dir = Path(args.bufr_dir) if args.bufr_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    try:
        summary = render_station_plots(
            cfg,
            station_id=str(args.station_id).zfill(5)[-5:],
            obs_date=obs_date,
            station_label=args.label,
            output_dir=output_dir,
            bufr_dir=bufr_dir,
            tae_files=tae_files,
            plot_types=plot_types,
            download=not args.no_download,
            latitude_deg=args.lat,
            longitude_deg=args.lon,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("bufr_profiles") or summary.get("tae_profiles") else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    cfg = load_config(args.config)
    init_decoder_tables(_tables_config_dict(cfg))
    field_names = load_meteo_parser_field_names(cfg.meteo_parser_path)
    if field_names:
        logging.info("Loaded %s meteo_parser decoded field names", len(field_names))

    if args.command == "manifest":
        return cmd_manifest(cfg, args)
    if args.command == "download":
        return cmd_download(cfg, args)
    if args.command == "probe-plots":
        return cmd_probe_plots(cfg, args)
    if args.command == "batch-plots":
        return cmd_batch_plots(cfg, args)
    if args.command == "station-plots":
        return cmd_station_plots(cfg, args)
    if args.command == "check-decode":
        return cmd_check_decode(args)
    if args.command == "tables-update":
        return cmd_tables_update(cfg, args)
    if args.command == "tables-export":
        return cmd_tables_export(cfg, args)
    if args.command == "lookup":
        return cmd_lookup(cfg, args)
    if args.command == "station-profiles":
        from gdex_bufr.profile_climate.cli import cmd_station_profiles
        from gdex_bufr.profile_climate.config import load_profile_climate_config

        pc_cfg = load_profile_climate_config(args.profile_config)
        return cmd_station_profiles(cfg, pc_cfg, args)
    if args.command == "monthly-profile-plots":
        from gdex_bufr.profile_climate.cli import cmd_monthly_profile_plots
        from gdex_bufr.profile_climate.config import load_profile_climate_config

        pc_cfg = load_profile_climate_config(args.profile_config)
        return cmd_monthly_profile_plots(pc_cfg, args)
    if args.command == "discover-stations":
        from gdex_bufr.profile_climate.cli import cmd_discover_stations

        return cmd_discover_stations(cfg, args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
