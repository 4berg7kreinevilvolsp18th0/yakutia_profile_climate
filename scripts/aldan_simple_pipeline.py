"""Автономный контур Алдана: BUFR -> SFC/MANL/TXPR -> высоты -> инверсия.

Основной результат пишется только в ``gdex_outputs/алдан-simple``.

Примеры:
  python scripts/aldan_simple_pipeline.py --date 2000-09-14 --cycle 12
  python scripts/aldan_simple_pipeline.py --all --fresh
  python scripts/aldan_simple_pipeline.py --plots
  python scripts/aldan_simple_pipeline.py --dashboard
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gdex_bufr.bufr_adapter import decode_bufr_file  # noqa: E402
from gdex_bufr.meteo_parser_bridge import (  # noqa: E402
    RadiosondeProfile,
    VerticalLevel,
    estimate_geopotential_height_m,
    geopotential_to_height_m,
)

# --- Пути и параметры: весь отдельный контур настраивается здесь. ---
STATION_ID = "31004"
STATION_FALLBACK_M = 679.0
PRESSURE_TOP_HPA = 500.0
PRESSURE_MAX_HPA = 1000.0
BUFR_ROOT = ROOT / "gdex_data" / "raw"
OUT_DIR = ROOT / "gdex_outputs" / "алдан-simple"
PLOTS_DIR = OUT_DIR / "plots"
DASHBOARD_CMD = (
    f'"{sys.executable}" -m streamlit run "{Path(__file__).resolve()}" '
    f'-- --dashboard-app --output "{OUT_DIR}"'
)

ALLOWED_LEVEL_TYPES = frozenset({"SFC", "MANL", "TXPR"})
LONG_FIELDS = [
    "profile_id",
    "station_id",
    "datetime_utc",
    "cycle",
    "source_file",
    "subset_index",
    "level_index",
    "seq",
    "VSIG",
    "VSIG_raw",
    "vertical_significance_code",
    "pressure_hpa",
    "temperature_c",
    "dewpoint_c",
    "rh_percent",
    "wind_dir",
    "wind_speed",
    "height_bufr_m",
    "geopotential_m2s2",
    "height_phi_m",
    "station_elevation_m",
    "station_elevation_source",
    "height_obs_m",
    "height_interp_m",
    "height_baro_m",
    "height_m",
    "height_source",
    "qc_flag",
]
METRIC_FIELDS = [
    "profile_id",
    "station_id",
    "datetime_utc",
    "cycle",
    "source_file",
    "subset_index",
    "station_elevation_m",
    "station_elevation_source",
    "p_surface_hpa",
    "t_surface_c",
    "height_surface_bufr_m",
    "height_surface_m",
    "n_levels_total",
    "n_levels_to_500",
    "p_top_hpa",
    "t_top_c",
    "profile_status",
    "inversion_detected",
    "inversion_candidate",
    "inversion_quality",
    "inversion_top_pressure_hpa",
    "inversion_top_height_m",
    "inversion_top_temp_c",
    "inversion_delta_t_c",
    "inversion_confirm_drop_c",
]
SFC_FIELDS = [
    "profile_id",
    "station_id",
    "datetime_utc",
    "source_file",
    "subset_index",
    "seq",
    "pressure_hpa",
    "temperature_c",
    "height_bufr_m",
    "geopotential_m2s2",
    "station_elevation_m",
    "is_preferred",
]


# --- Общие функции. ---
def finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalize_station_id(value: str | None) -> str:
    return str(value or "").zfill(5)[-5:]


def normalized_level_type(value: str | None) -> str | None:
    """Приводит текущие SIGT/TROP к требуемой пользователем метке TXPR."""
    raw = str(value or "").upper()
    if raw in {"SIGT", "TROP", "TXPR"}:
        return "TXPR"
    if raw in {"SFC", "MANL"}:
        return raw
    return None


def cycle_from_profile(profile: RadiosondeProfile) -> str:
    name = Path(profile.source_file).name.lower()
    for cycle in ("00", "12"):
        if f".t{cycle}z." in name:
            return cycle
    if profile.report_datetime_utc:
        try:
            dt = datetime.fromisoformat(profile.report_datetime_utc.replace("Z", "+00:00"))
            return f"{dt.hour:02d}"
        except ValueError:
            pass
    return "00"


def profile_id(profile: RadiosondeProfile) -> str:
    sid = normalize_station_id(profile.station_id)
    stamp = (profile.report_datetime_utc or "unknown").replace("-", "").replace(":", "")
    stamp = stamp.replace("T", "_").replace("Z", "")[:13]
    return f"{sid}_{stamp}_{cycle_from_profile(profile)}_{profile.subset_index}"


def station_height(profile: RadiosondeProfile) -> tuple[float, str]:
    bufr_height = finite(profile.station_elevation_m)
    if bufr_height is not None:
        return bufr_height, "bufr_007001"
    return STATION_FALLBACK_M, "station_fallback"


# --- Surface: отделяем настоящую станцию от повторных SFC секций ADPUPA. ---
def pick_preferred_surface(
    levels: Iterable[VerticalLevel],
    station_elevation_m: float,
) -> VerticalLevel | None:
    all_levels = list(levels)
    surfaces = [
        level
        for level in all_levels
        if normalized_level_type(level.vertical_significance) == "SFC"
        and finite(level.pressure_hpa) is not None
    ]
    with_height = [level for level in surfaces if finite(level.geopotential_height_m) is not None]
    if with_height:
        return min(
            with_height,
            key=lambda level: abs(float(level.geopotential_height_m) - station_elevation_m),
        )
    if surfaces:
        return min(surfaces, key=lambda level: level.seq if level.seq is not None else 10**9)
    with_pressure = [level for level in all_levels if finite(level.pressure_hpa) is not None]
    return max(with_pressure, key=lambda level: float(level.pressure_hpa)) if with_pressure else None


def raw_sfc_rows(
    profile: RadiosondeProfile,
    preferred: VerticalLevel | None,
    station_z: float,
) -> list[dict[str, Any]]:
    pid = profile_id(profile)
    rows: list[dict[str, Any]] = []
    for level in profile.levels:
        if normalized_level_type(level.vertical_significance) != "SFC":
            continue
        rows.append(
            {
                "profile_id": pid,
                "station_id": normalize_station_id(profile.station_id),
                "datetime_utc": profile.report_datetime_utc or "",
                "source_file": profile.source_file,
                "subset_index": profile.subset_index,
                "seq": level.seq,
                "pressure_hpa": level.pressure_hpa,
                "temperature_c": level.air_temperature_c,
                "height_bufr_m": level.geopotential_height_m,
                "geopotential_m2s2": level.geopotential_m2s2,
                "station_elevation_m": station_z,
                "is_preferred": level is preferred,
            }
        )
    return rows


# --- Уровни SFC/MANL/TXPR и сырые значения BUFR. ---
def level_score(level: VerticalLevel) -> tuple[int, int]:
    """При дубле P выбирается наиболее полный уровень; SFC получает небольшой приоритет."""
    present = sum(
        value is not None
        for value in (
            level.air_temperature_c,
            level.dew_point_temperature_c,
            level.relative_humidity_percent,
            level.wind_direction_deg,
            level.wind_speed,
            level.geopotential_height_m,
            level.geopotential_m2s2,
        )
    )
    type_bonus = {"SFC": 3, "MANL": 2, "TXPR": 1}.get(
        normalized_level_type(level.vertical_significance) or "", 0
    )
    return present, type_bonus


def select_levels(
    profile: RadiosondeProfile,
    preferred: VerticalLevel | None,
    station_z: float,
    station_z_source: str,
) -> tuple[list[dict[str, Any]], float | None]:
    p_surface = finite(preferred.pressure_hpa) if preferred is not None else None
    candidates: list[VerticalLevel] = []
    for level in profile.levels:
        level_type = normalized_level_type(level.vertical_significance)
        pressure = finite(level.pressure_hpa)
        temperature = finite(level.air_temperature_c)
        if level_type not in ALLOWED_LEVEL_TYPES or pressure is None or temperature is None:
            continue
        if not (PRESSURE_TOP_HPA <= pressure <= PRESSURE_MAX_HPA):
            continue
        if p_surface is not None and pressure > p_surface + 2.0:
            continue
        candidates.append(level)

    # Один физический уровень на давление, как в текущем климатическом результате.
    by_pressure: dict[float, VerticalLevel] = {}
    for level in candidates:
        key = round(float(level.pressure_hpa), 1)
        previous = by_pressure.get(key)
        if previous is None or level_score(level) > level_score(previous):
            by_pressure[key] = level

    ordered = sorted(by_pressure.values(), key=lambda level: float(level.pressure_hpa), reverse=True)
    rows: list[dict[str, Any]] = []
    for index, level in enumerate(ordered):
        phi = finite(level.geopotential_m2s2)
        # Адаптер кладёт в geopotential_height_m также уже рассчитанное Φ→z.
        # Поэтому прямой BUFR height (010009/007007) отделяем по отсутствию Φ.
        raw_height = finite(level.geopotential_height_m) if phi is None else None
        phi_height = None
        if phi is not None:
            converted = geopotential_to_height_m(phi)
            if math.isfinite(converted):
                phi_height = round(converted, 1)
        is_preferred = level is preferred or (
            preferred is not None
            and normalized_level_type(level.vertical_significance) == "SFC"
            and abs(float(level.pressure_hpa) - float(preferred.pressure_hpa)) < 0.15
        )
        rows.append(
            {
                "level_index": index,
                "seq": level.seq,
                "VSIG": normalized_level_type(level.vertical_significance),
                "VSIG_raw": level.vertical_significance,
                "vertical_significance_code": level.vertical_significance_code,
                "pressure_hpa": finite(level.pressure_hpa),
                "temperature_c": finite(level.air_temperature_c),
                "dewpoint_c": finite(level.dew_point_temperature_c),
                "rh_percent": finite(level.relative_humidity_percent),
                "wind_dir": finite(level.wind_direction_deg),
                "wind_speed": finite(level.wind_speed),
                # Никогда не затирается: это значение непосредственно из BUFR.
                "height_bufr_m": raw_height,
                "geopotential_m2s2": phi,
                "height_phi_m": phi_height,
                "station_elevation_m": station_z,
                "station_elevation_source": station_z_source,
                "_is_preferred_sfc": is_preferred,
            }
        )
    return rows, p_surface


# --- Высоты: BUFR -> Phi -> station -> interpolation -> barometry. ---
def interpolate_internal(
    pressures: list[float],
    anchors: list[float | None],
) -> list[float | None]:
    known = sorted(
        ((pressure, height) for pressure, height in zip(pressures, anchors) if height is not None),
        key=lambda pair: pair[0],
    )
    if len(known) < 2:
        return [height for height in anchors]
    out: list[float | None] = []
    for pressure, original in zip(pressures, anchors):
        if original is not None:
            out.append(round(original, 1))
            continue
        lower = next((pair for pair in reversed(known) if pair[0] < pressure), None)
        upper = next((pair for pair in known if pair[0] > pressure), None)
        if lower is None or upper is None or upper[0] == lower[0]:
            out.append(None)
            continue
        fraction = (pressure - lower[0]) / (upper[0] - lower[0])
        out.append(round(lower[1] + fraction * (upper[1] - lower[1]), 1))
    return out


def fill_heights(
    rows: list[dict[str, Any]],
    *,
    p_surface: float | None,
    station_z: float,
) -> list[dict[str, Any]]:
    if not rows:
        return rows
    pressures = [float(row["pressure_hpa"]) for row in rows]
    observed: list[float | None] = []
    sources: list[str | None] = []
    for row in rows:
        raw_height = finite(row["height_bufr_m"])
        phi_height = finite(row["height_phi_m"])
        if raw_height is not None and raw_height >= -50.0:
            observed.append(raw_height)
            sources.append("level")
        elif phi_height is not None and phi_height >= -50.0:
            observed.append(phi_height)
            sources.append("phi")
        elif row["_is_preferred_sfc"]:
            observed.append(station_z)
            sources.append("station_007001")
        else:
            observed.append(None)
            sources.append(None)

    interpolated = interpolate_internal(pressures, observed)
    effective_surface = p_surface or max(pressures)
    out: list[dict[str, Any]] = []
    for row, pressure, obs, source, interp in zip(
        rows, pressures, observed, sources, interpolated
    ):
        baro = round(
            station_z
            + estimate_geopotential_height_m(
                pressure,
                surface_pressure_hpa=effective_surface,
            ),
            1,
        )
        final_height = obs
        final_source = source
        if final_height is None and interp is not None:
            final_height = interp
            final_source = "interp"
        if final_height is None:
            final_height = baro
            final_source = "baro"
        result = {key: value for key, value in row.items() if not key.startswith("_")}
        result.update(
            {
                "height_obs_m": obs,
                "height_interp_m": interp,
                "height_baro_m": baro,
                "height_m": round(float(final_height), 1),
                "height_source": final_source,
            }
        )
        flags: list[str] = []
        if result["height_m"] < -50.0:
            flags.append("height_negative")
        if row["_is_preferred_sfc"] and abs(result["height_m"] - station_z) > 250.0:
            flags.append("surface_height_far_from_station")
        result["qc_flag"] = "|".join(flags)
        out.append(result)
    return out


# --- Инверсия v2: рост от поверхности + устойчивое падение выше. ---
@dataclass
class InversionResult:
    inversion_detected: bool = False
    inversion_candidate: bool = False
    inversion_quality: str = "none"
    inversion_top_pressure_hpa: float | None = None
    inversion_top_height_m: float | None = None
    inversion_top_temp_c: float | None = None
    inversion_delta_t_c: float | None = None
    inversion_confirm_drop_c: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return dict(vars(self))


def detect_inversion(
    levels: list[dict[str, Any]],
    *,
    min_growth_c: float = 0.2,
    confirm_drop_levels: int = 2,
    confirm_depth_hpa: float = 30.0,
    min_drop_c: float = 0.2,
) -> InversionResult:
    if len(levels) < 2:
        return InversionResult()
    ordered = sorted(levels, key=lambda row: float(row["pressure_hpa"]), reverse=True)
    surface_temp = finite(ordered[0]["temperature_c"])
    if surface_temp is None:
        return InversionResult()
    top = ordered[0]
    top_index = 0
    growing = False
    for index, level in enumerate(ordered[1:], start=1):
        temp = finite(level["temperature_c"])
        previous = finite(top["temperature_c"])
        if temp is None or previous is None:
            break
        if temp - previous > min_growth_c:
            growing = True
            top = level
            top_index = index
        else:
            break
    if not growing:
        return InversionResult()

    above = ordered[top_index + 1 :]
    top_temp = float(top["temperature_c"])
    top_pressure = float(top["pressure_hpa"])
    consecutive_ok = len(above) >= confirm_drop_levels
    previous_temp = top_temp
    if consecutive_ok:
        for level in above[:confirm_drop_levels]:
            temp = finite(level["temperature_c"])
            if temp is None or temp - previous_temp > -min_drop_c:
                consecutive_ok = False
                break
            previous_temp = temp

    window_last: dict[str, Any] | None = None
    for level in above:
        if finite(level["temperature_c"]) is None:
            break
        window_last = level
        if top_pressure - float(level["pressure_hpa"]) >= confirm_depth_hpa:
            break
    depth_ok = (
        window_last is not None
        and top_pressure - float(window_last["pressure_hpa"]) >= confirm_depth_hpa
    )
    confirm_drop = (
        float(window_last["temperature_c"]) - top_temp if window_last is not None else None
    )
    confirmed = consecutive_ok and depth_ok and confirm_drop is not None and confirm_drop < 0
    return InversionResult(
        inversion_detected=confirmed,
        inversion_candidate=True,
        inversion_quality="confirmed" if confirmed else "rejected_no_lapse",
        inversion_top_pressure_hpa=top_pressure,
        inversion_top_height_m=finite(top["height_m"]),
        inversion_top_temp_c=top_temp,
        inversion_delta_t_c=top_temp - surface_temp,
        inversion_confirm_drop_c=confirm_drop,
    )


# --- Обработка одного профиля. ---
def process_profile(
    profile: RadiosondeProfile,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    station_z, station_z_source = station_height(profile)
    preferred = pick_preferred_surface(profile.levels, station_z)
    selected, p_surface = select_levels(
        profile, preferred, station_z, station_z_source
    )
    levels = fill_heights(selected, p_surface=p_surface, station_z=station_z)
    inversion = detect_inversion(levels)
    pid = profile_id(profile)
    base = {
        "profile_id": pid,
        "station_id": normalize_station_id(profile.station_id),
        "datetime_utc": profile.report_datetime_utc or "",
        "cycle": cycle_from_profile(profile),
        "source_file": profile.source_file,
        "subset_index": profile.subset_index,
    }
    long_rows = [{**base, **row} for row in levels]
    surface = levels[0] if levels else {}
    top = levels[-1] if levels else {}
    has_500 = bool(levels) and min(float(row["pressure_hpa"]) for row in levels) <= PRESSURE_TOP_HPA
    status = "good"
    if not levels:
        status = "no_surface_level"
    elif not has_500:
        status = "no_500"
    elif len(levels) < 5:
        status = "short"
    metric = {
        **base,
        "station_elevation_m": station_z,
        "station_elevation_source": station_z_source,
        "p_surface_hpa": surface.get("pressure_hpa"),
        "t_surface_c": surface.get("temperature_c"),
        "height_surface_bufr_m": surface.get("height_bufr_m"),
        "height_surface_m": surface.get("height_m"),
        "n_levels_total": len(profile.levels),
        "n_levels_to_500": len(levels),
        "p_top_hpa": top.get("pressure_hpa"),
        "t_top_c": top.get("temperature_c"),
        "profile_status": status,
        **inversion.as_dict(),
    }
    return long_rows, metric, raw_sfc_rows(profile, preferred, station_z)


# --- Export и пакетный decode. ---
def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def clear_own_output(output: Path) -> None:
    """Удаляет только известные продукты отдельного контура."""
    output.mkdir(parents=True, exist_ok=True)
    for name in ("profiles_long.csv", "profile_metrics.csv", "sfc_raw.csv", "summary.json"):
        path = output / name
        if path.exists():
            path.unlink()
    plots = output / "plots"
    if plots.exists():
        shutil.rmtree(plots)


def resolve_date_file(bufr_root: Path, date: str, cycle: str) -> Path:
    ymd = date.replace("-", "")
    filename = f"gdas.adpupa.t{cycle.zfill(2)}z.{ymd}.bufr"
    candidates = [bufr_root / ymd[:4] / filename, bufr_root / filename]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("BUFR не найден: " + ", ".join(str(path) for path in candidates))


def discover_bufr(bufr_root: Path) -> list[Path]:
    return sorted(bufr_root.rglob("gdas.adpupa.t*z.*.bufr"))


def run_decode(
    files: list[Path],
    *,
    station_id: str,
    output: Path,
    fresh: bool,
    bufr_root: Path,
) -> dict[str, Any]:
    if fresh:
        clear_own_output(output)
    output.mkdir(parents=True, exist_ok=True)
    long_rows: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    sfc_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for index, path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {path}")
        try:
            profiles = decode_bufr_file(path, station_id=normalize_station_id(station_id))
            for profile in profiles:
                rows, metric, raw_sfc = process_profile(profile)
                long_rows.extend(rows)
                metrics.append(metric)
                sfc_rows.extend(raw_sfc)
        except Exception as exc:
            errors.append({"file": str(path), "error": f"{type(exc).__name__}: {exc}"})
            print(f"  ERROR: {errors[-1]['error']}", file=sys.stderr)

    write_csv(output / "profiles_long.csv", long_rows, LONG_FIELDS)
    write_csv(output / "profile_metrics.csv", metrics, METRIC_FIELDS)
    write_csv(output / "sfc_raw.csv", sfc_rows, SFC_FIELDS)
    sfc_counts = Counter(str(row["profile_id"]) for row in sfc_rows)
    summary = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "station_id": normalize_station_id(station_id),
        "bufr_root": str(bufr_root),
        "output_dir": str(output),
        "files_total": len(files),
        "files_failed": len(errors),
        "profiles_total": len(metrics),
        "profiles_good": sum(row["profile_status"] == "good" for row in metrics),
        "profiles_with_inversion": sum(bool(row["inversion_detected"]) for row in metrics),
        "levels_total": len(long_rows),
        "levels_with_bufr_height": sum(row["height_bufr_m"] is not None for row in long_rows),
        "levels_with_phi": sum(row["geopotential_m2s2"] is not None for row in long_rows),
        "multi_sfc_profiles": sum(count > 1 for count in sfc_counts.values()),
        "errors": errors,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


# --- Графики отдельного контура. ---
def make_plots(output: Path) -> int:
    import matplotlib.pyplot as plt
    import pandas as pd

    source = output / "profiles_long.csv"
    if not source.exists():
        raise FileNotFoundError(f"Сначала создайте {source}")
    frame = pd.read_csv(source)
    if frame.empty:
        return 0
    frame["datetime"] = pd.to_datetime(frame["datetime_utc"], errors="coerce")
    frame = frame.dropna(subset=["datetime", "temperature_c", "height_m"])
    frame["month"] = frame["datetime"].dt.to_period("M").astype(str)
    plots_dir = output / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for month, month_frame in frame.groupby("month", sort=True):
        figure, axis = plt.subplots(figsize=(7, 9))
        for _, levels in month_frame.groupby("profile_id", sort=False):
            levels = levels.sort_values("height_m")
            axis.plot(levels["temperature_c"], levels["height_m"], alpha=0.18, linewidth=0.8)
        axis.set(title=f"Алдан: {month}", xlabel="Температура, °C", ylabel="Высота, м")
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(plots_dir / f"profiles_{month}.png", dpi=150)
        plt.close(figure)
        count += 1
    print(f"Графики: {count}; каталог: {plots_dir}")
    return count


# --- Дашборд отдельного контура (этот же файл запускается через Streamlit). ---
def run_dashboard(output: Path) -> None:
    import pandas as pd
    import plotly.graph_objects as go
    import streamlit as st

    st.set_page_config(page_title="Алдан simple", layout="wide")
    st.title("Алдан · SFC / MANL / TXPR")
    long_path = output / "profiles_long.csv"
    metrics_path = output / "profile_metrics.csv"
    st.caption(f"Источник: {output}")
    if not long_path.exists() or not metrics_path.exists():
        st.error("Нет profiles_long.csv/profile_metrics.csv. Сначала выполните decode.")
        return
    levels = pd.read_csv(long_path)
    metrics = pd.read_csv(metrics_path)
    if levels.empty:
        st.warning("Профили отсутствуют.")
        return
    metrics["datetime"] = pd.to_datetime(metrics["datetime_utc"], errors="coerce")
    dates = metrics["datetime"].dropna()
    date_min = dates.min().date()
    date_max = dates.max().date()
    left, middle, right = st.columns(3)
    with left:
        selected_dates = st.date_input(
            "Диапазон дат", value=(date_min, date_max), min_value=date_min, max_value=date_max
        )
    with middle:
        cycles = st.multiselect("Срок", ["00", "12"], default=["00", "12"])
    with right:
        inversion_only = st.checkbox("Только инверсии")
    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start, finish = selected_dates
    else:
        start = finish = selected_dates
    chosen = metrics[
        metrics["datetime"].dt.date.between(start, finish)
        & metrics["cycle"].astype(str).str.zfill(2).isin(cycles)
    ]
    if inversion_only:
        chosen = chosen[chosen["inversion_detected"].astype(str).str.lower().isin(["true", "1"])]
    selected_ids = set(chosen["profile_id"].astype(str))
    shown = levels[levels["profile_id"].astype(str).isin(selected_ids)]
    axis_mode = st.radio("Ось Y", ["Высота", "Давление"], horizontal=True)
    figure = go.Figure()
    for pid, group in shown.groupby("profile_id", sort=False):
        group = group.sort_values("height_m" if axis_mode == "Высота" else "pressure_hpa")
        metric = chosen[chosen["profile_id"].astype(str) == str(pid)].iloc[0]
        figure.add_trace(
            go.Scatter(
                x=group["temperature_c"],
                y=group["height_m"] if axis_mode == "Высота" else group["pressure_hpa"],
                mode="lines+markers",
                marker={"size": 4},
                name=str(metric["datetime_utc"]),
                customdata=group[["VSIG", "height_bufr_m", "height_source"]],
                hovertemplate=(
                    "T=%{x:.1f}<br>Y=%{y:.1f}<br>VSIG=%{customdata[0]}"
                    "<br>H BUFR=%{customdata[1]}<br>source=%{customdata[2]}<extra></extra>"
                ),
            )
        )
        inversion_detected = str(metric.get("inversion_detected", "")).lower() in {
            "true",
            "1",
        }
        if inversion_detected:
            figure.add_trace(
                go.Scatter(
                    x=[metric["inversion_top_temp_c"]],
                    y=[
                        metric["inversion_top_height_m"]
                        if axis_mode == "Высота"
                        else metric["inversion_top_pressure_hpa"]
                    ],
                    mode="markers",
                    marker={"symbol": "diamond", "size": 10, "color": "black"},
                    name=f"Инверсия {metric['datetime_utc']}",
                )
            )
    figure.update_layout(
        xaxis_title="Температура, °C",
        yaxis_title="Высота, м" if axis_mode == "Высота" else "Давление, гПа",
        height=760,
    )
    if axis_mode == "Давление":
        figure.update_yaxes(autorange="reversed")
    st.plotly_chart(figure, use_container_width=True)
    st.dataframe(chosen.drop(columns=["datetime"]), use_container_width=True)


def launch_dashboard(output: Path) -> int:
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(Path(__file__).resolve()),
        "--",
        "--dashboard-app",
        "--output",
        str(output),
    ]
    print("Дашборд:", subprocess.list2cmdline(command))
    return subprocess.call(command, cwd=ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--date", help="Дата YYYY-MM-DD")
    source.add_argument("--bufr", type=Path, help="Один BUFR-файл")
    source.add_argument("--all", action="store_true", help="Все BUFR под --bufr-root")
    parser.add_argument("--cycle", default="12", choices=["00", "12"])
    parser.add_argument("--station", default=STATION_ID)
    parser.add_argument("--bufr-root", type=Path, default=BUFR_ROOT)
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    parser.add_argument("--fresh", action="store_true", help="Очистить только продукты алдан-simple")
    parser.add_argument("--plots", action="store_true", help="Построить месячные PNG")
    parser.add_argument("--dashboard", action="store_true", help="Запустить отдельный Streamlit")
    parser.add_argument("--dashboard-app", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.resolve()
    if args.dashboard_app:
        run_dashboard(output)
        return 0
    files: list[Path] = []
    if args.date:
        files = [resolve_date_file(args.bufr_root, args.date, args.cycle)]
    elif args.bufr:
        files = [args.bufr]
    elif args.all:
        files = discover_bufr(args.bufr_root)
        if not files:
            raise SystemExit(f"BUFR-файлы не найдены в {args.bufr_root}")
    if files:
        run_decode(
            files,
            station_id=args.station,
            output=output,
            fresh=args.fresh,
            bufr_root=args.bufr_root,
        )
    elif not args.plots and not args.dashboard:
        build_parser().error("Укажите --date, --bufr, --all, --plots или --dashboard")
    if args.plots:
        make_plots(output)
    if args.dashboard:
        return launch_dashboard(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
