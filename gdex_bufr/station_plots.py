"""Графики по одной станции за выбранную дату (BUFR GDEX + опционально TAE)."""
from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

import requests
import urllib3

from gdex_bufr.bufr_adapter import decode_bufr_file, init_decoder_tables
from gdex_bufr.config import AppConfig
from gdex_bufr.meteo_parser_bridge import RadiosondeProfile
from gdex_bufr.plots import _apply_pressure_axis_hpa, _line_plot_kwargs, render_plots
from gdex_bufr.tae03_parser import parse_tae03
from gdex_bufr.xlsx_export import write_profiles_xlsx

logger = logging.getLogger(__name__)

SYNOPTIC_CYCLES = ("00", "06", "12", "18")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def bufr_name_for(cycle: str, obs_date: date) -> str:
    return f"gdas.adpupa.t{cycle}z.{obs_date.strftime('%Y%m%d')}.bufr"


def bufr_url_for(cycle: str, obs_date: date, base_url: str = "https://data.gdex.ucar.edu/d351000/bufr") -> str:
    year = obs_date.year
    return f"{base_url}/{year}/{bufr_name_for(cycle, obs_date)}"


def supplemental_bufr_names(obs_date: date) -> list[str]:
    """t00z следующего дня — 00 UTC после вечернего зондирования (23:30–01:00 местного)."""
    next_day = obs_date + timedelta(days=1)
    return [bufr_name_for("00", next_day)]


def _bufr_download_targets(
    obs_date: date,
    *,
    cycles: tuple[str, ...],
    include_next_day_t00z: bool,
    base_url: str,
) -> list[tuple[str, str]]:
    """Пары (имя файла, URL) для загрузки."""
    names = [bufr_name_for(cycle, obs_date) for cycle in cycles]
    if include_next_day_t00z:
        for extra in supplemental_bufr_names(obs_date):
            if extra not in names:
                names.append(extra)
    day_names = {bufr_name_for(cycle, obs_date) for cycle in cycles}
    root = base_url.rstrip("/")
    targets: list[tuple[str, str]] = []
    for name in names:
        year = obs_date.year if name in day_names else (obs_date + timedelta(days=1)).year
        targets.append((name, f"{root}/{year}/{name}"))
    return targets


def download_bufr_files(
    obs_date: date,
    bufr_dir: Path,
    *,
    cycles: tuple[str, ...] = SYNOPTIC_CYCLES,
    base_url: str = "https://data.gdex.ucar.edu/d351000/bufr",
    ssl_verify: bool = False,
    timeout_seconds: int = 300,
    max_retries: int = 3,
    include_next_day_t00z: bool = True,
) -> list[Path]:
    bufr_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    targets = _bufr_download_targets(
        obs_date,
        cycles=cycles,
        include_next_day_t00z=include_next_day_t00z,
        base_url=base_url,
    )
    for name, url in targets:
        dst = bufr_dir / name
        if dst.exists() and dst.stat().st_size > 0:
            downloaded.append(dst)
            continue
        for attempt in range(1, max_retries + 1):
            try:
                logger.info("Загрузка %s (попытка %s/%s)", url, attempt, max_retries)
                response = requests.get(url, timeout=timeout_seconds, verify=ssl_verify)
                if response.status_code == 200 and response.content:
                    dst.write_bytes(response.content)
                    downloaded.append(dst)
                    break
                logger.warning("Не удалось загрузить %s (HTTP %s)", name, response.status_code)
            except requests.RequestException as exc:
                logger.warning("Ошибка загрузки %s: %s", name, exc)
        else:
            logger.warning("Пропуск %s после %s попыток", name, max_retries)
    return downloaded


def decode_station_from_files(
    files: list[Path],
    station_id: str,
    *,
    registry,
    decode_mode: str = "adpupa",
) -> list[RadiosondeProfile]:
    profiles: list[RadiosondeProfile] = []
    for path in files:
        if not path.exists():
            continue
        found = decode_bufr_file(
            path,
            max_profiles=1,
            station_id=station_id,
            registry=registry,
            decode_mode=decode_mode,
        )
        if found:
            profiles.append(found[0])
            logger.info(
                "%s: %s, уровней %s, статус %s",
                path.name,
                found[0].report_datetime_utc,
                len(found[0].levels),
                found[0].data_status,
            )
        else:
            logger.warning("%s: станция %s не найдена", path.name, station_id)
    profiles.sort(key=lambda p: p.report_datetime_utc or "")
    return profiles


def _cycle_label(path: Path) -> str:
    name = path.name
    if name.startswith("gdas.adpupa."):
        return name.replace("gdas.adpupa.", "").replace(".bufr", "")
    for cycle in SYNOPTIC_CYCLES:
        if f".t{cycle}z." in name:
            return f"t{cycle}z"
    return path.stem


def render_station_comparison(
    tae_profiles: list[RadiosondeProfile],
    bufr_profiles: list[RadiosondeProfile],
    output_path: Path,
    *,
    plot_style,
    title: str,
) -> Path | None:
    import matplotlib.pyplot as plt

    if not bufr_profiles:
        return None

    fig, ax = plt.subplots(figsize=(8, 10))
    ref_pressures: list[float] = []
    point_style = _line_plot_kwargs(plot_style.profile)
    for profile in bufr_profiles:
        pressures, temps, dewpoints = [], [], []
        for lv in profile.levels:
            if lv.pressure_hpa is None or lv.air_temperature_c is None:
                continue
            pressures.append(lv.pressure_hpa)
            temps.append(lv.air_temperature_c)
            dewpoints.append(
                lv.dew_point_temperature_c if lv.dew_point_temperature_c is not None else float("nan")
            )
        if len(pressures) < 2:
            continue
        if not ref_pressures:
            ref_pressures = pressures
        label_dt = profile.report_datetime_utc or _cycle_label(Path(profile.source_file))
        ax.plot(temps, pressures, label=f"BUFR T ({label_dt})", **point_style)
        ax.plot(
            dewpoints,
            pressures,
            linestyle="--",
            label=f"BUFR Td ({label_dt})",
            **_line_plot_kwargs(plot_style.profile, linewidth=1.5),
        )

    for idx, profile in enumerate(tae_profiles, start=1):
        pressures, temps, _ = [], [], []
        for lv in profile.levels:
            if lv.pressure_hpa is None or lv.air_temperature_c is None:
                continue
            pressures.append(lv.pressure_hpa)
            temps.append(lv.air_temperature_c)
        if len(pressures) < 2:
            continue
        ax.plot(
            temps,
            pressures,
            linestyle=":",
            label=f"TAE #{idx} ({profile.report_datetime_utc})",
            **_line_plot_kwargs(plot_style.profile, linewidth=1.8),
        )

    if not ref_pressures:
        plt.close(fig)
        return None

    ax.set_xlabel("Температура, °C")
    _apply_pressure_axis_hpa(ax, ref_pressures, plot_style)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=plot_style.dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_station_plots(
    cfg: AppConfig,
    *,
    station_id: str,
    obs_date: date,
    station_label: str = "",
    output_dir: Path | None = None,
    bufr_dir: Path | None = None,
    tae_files: list[Path] | None = None,
    plot_types: list[str] | None = None,
    download: bool = True,
    latitude_deg: float | None = None,
    longitude_deg: float | None = None,
) -> dict:
    """Декодирует станцию за день и строит полный набор графиков."""
    tables_dir = Path(cfg.bufr_tables.directory)
    if not tables_dir.is_absolute():
        tables_dir = Path.cwd() / tables_dir
    export_dir = Path(cfg.bufr_tables.export_dir)
    if not export_dir.is_absolute():
        export_dir = Path.cwd() / export_dir

    registry = init_decoder_tables({
        "directory": str(tables_dir),
        "wmo_version": cfg.bufr_tables.wmo_version,
        "master_table_version": cfg.bufr_tables.master_table_version,
        "export_dir": str(export_dir),
        "export_on_update": cfg.bufr_tables.export_on_update,
    })
    if not registry.is_ready():
        raise RuntimeError(
            "BUFR-таблицы не установлены. Выполните: python -m gdex_bufr tables-update"
        )

    label_slug = station_label or station_id
    out_root = output_dir or (Path(cfg.outputs_dir) / "stations" / station_id / obs_date.isoformat())
    out_root.mkdir(parents=True, exist_ok=True)

    cache_dir = bufr_dir or (out_root / "bufr_raw")
    bufr_files = download_bufr_files(
        obs_date,
        cache_dir,
        base_url=f"{cfg.base_url.rstrip('/')}",
        ssl_verify=cfg.ssl_verify,
        timeout_seconds=cfg.download_timeout_seconds,
        max_retries=cfg.max_retries,
        include_next_day_t00z=True,
    ) if download else [
        cache_dir / bufr_name_for(c, obs_date) for c in SYNOPTIC_CYCLES
    ] + [cache_dir / name for name in supplemental_bufr_names(obs_date)]

    bufr_files = [p for p in bufr_files if p.exists()]
    bufr_profiles = decode_station_from_files(bufr_files, station_id, registry=registry)

    tae_profiles: list[RadiosondeProfile] = []
    for tae_path in tae_files or []:
        if tae_path.exists():
            tae_profiles.append(
                parse_tae03(
                    tae_path,
                    latitude_deg=latitude_deg,
                    longitude_deg=longitude_deg,
                )
            )

    plot_types = plot_types or [p for p in cfg.plots.plot_types if p != "map"]
    plot_style = cfg.plots
    written: list[str] = []

    for profile in bufr_profiles:
        cycle = _cycle_label(Path(profile.source_file))
        bundle_dir = out_root / "bufr" / cycle
        outputs = render_plots(
            profile,
            bundle_dir,
            plot_types,
            all_profiles=bufr_profiles,
            style=plot_style,
        )
        written.extend(str(p) for p in outputs)

    for idx, profile in enumerate(tae_profiles, start=1):
        bundle_dir = out_root / "tae" / f"release_{idx}"
        outputs = render_plots(
            profile,
            bundle_dir,
            plot_types,
            all_profiles=tae_profiles,
            style=plot_style,
        )
        written.extend(str(p) for p in outputs)

    comparison_title = f"{label_slug} — {obs_date.strftime('%d.%m.%Y')}"
    comparison_path = out_root / "comparison" / "temperature_all_cycles.png"
    cmp_path = render_station_comparison(
        tae_profiles,
        bufr_profiles,
        comparison_path,
        plot_style=plot_style,
        title=comparison_title,
    )
    if cmp_path:
        written.append(str(cmp_path))

    all_profiles = tae_profiles + bufr_profiles
    xlsx_path = out_root / f"{station_id}_{obs_date.strftime('%Y%m%d')}.xlsx"
    if all_profiles:
        write_profiles_xlsx(xlsx_path, all_profiles)
        written.append(str(xlsx_path))

    summary = {
        "station_id": station_id,
        "station_label": station_label,
        "obs_date": obs_date.isoformat(),
        "output_dir": str(out_root),
        "bufr_files": [p.name for p in bufr_files],
        "bufr_profiles": len(bufr_profiles),
        "tae_profiles": len(tae_profiles),
        "plots_written": len(written),
        "plot_paths": written,
    }
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
