from __future__ import annotations

import calendar
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import AnalysisConfig


REQUIRED_COLUMNS = {
    "station_id",
    "datetime_utc",
    "profile_id",
    "cycle",
    "pressure_hpa",
    "temperature_c",
}

OPTIONAL_COLUMNS = {
    "station_name",
    "year",
    "month",
    "height_m",
    "data_status",
    "data_status_reason",
    "qc_flag",
    "VSIG",
    "vertical_significance_code",
}


def _normalise_cycle(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.map(lambda x: f"{int(x):02d}" if pd.notna(x) else "")


def load_profiles(
    path_or_buffer,
    *,
    station_id: str | None = None,
    cycles: Iterable[str] | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Загрузить long-format CSV и привести ключевые поля к единому типу."""
    header = pd.read_csv(path_or_buffer, nrows=0)
    missing = REQUIRED_COLUMNS.difference(header.columns)
    if missing:
        raise ValueError(f"В profiles_long отсутствуют обязательные столбцы: {sorted(missing)}")

    usecols = [c for c in header.columns if c in REQUIRED_COLUMNS or c in OPTIONAL_COLUMNS]
    df = pd.read_csv(
        path_or_buffer,
        usecols=usecols,
        low_memory=False,
        dtype={"station_id": "string", "profile_id": "string"},
    )
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], errors="coerce", utc=False)
    df = df[df["datetime_utc"].notna()].copy()
    df["station_id"] = df["station_id"].astype("string").str.replace(r"\.0$", "", regex=True)
    df["profile_id"] = df["profile_id"].astype("string")
    df["cycle"] = _normalise_cycle(df["cycle"])
    df["pressure_hpa"] = pd.to_numeric(df["pressure_hpa"], errors="coerce")
    df["temperature_c"] = pd.to_numeric(df["temperature_c"], errors="coerce")
    if "height_m" not in df.columns:
        df["height_m"] = np.nan
    else:
        df["height_m"] = pd.to_numeric(df["height_m"], errors="coerce")

    df["year"] = df["datetime_utc"].dt.year.astype(int)
    df["month"] = df["datetime_utc"].dt.month.astype(int)
    df["date"] = df["datetime_utc"].dt.normalize()

    if station_id is not None:
        df = df[df["station_id"] == str(station_id)].copy()
    if cycles is not None:
        wanted = {str(x).zfill(2) for x in cycles}
        df = df[df["cycle"].isin(wanted)].copy()
    if start_date is not None:
        df = df[df["datetime_utc"] >= pd.Timestamp(start_date)].copy()
    if end_date is not None:
        end_ts = pd.Timestamp(end_date)
        if end_ts.hour == 0 and end_ts.minute == 0 and len(str(end_date)) <= 10:
            end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        df = df[df["datetime_utc"] <= end_ts].copy()

    df = df.sort_values(["datetime_utc", "profile_id", "pressure_hpa"], ascending=[True, True, False])
    return df.reset_index(drop=True)


def build_profile_qc(df: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    """Сформировать одну строку на профиль со статусом пригодности.

    Реализация векторизована: это существенно быстрее покомпонентного обхода
    десятков тысяч профилей в интерфейсе Streamlit.
    """
    keys = ["profile_id"]
    meta_cols = ["station_id", "datetime_utc", "year", "month", "cycle"]
    meta = (
        df.sort_values(["profile_id", "datetime_utc"])
        .groupby("profile_id", sort=False)[meta_cols]
        .first()
    )
    total = df.groupby("profile_id", sort=False).size().rename("n_levels_total")

    work = df[df["pressure_hpa"].between(
        config.pressure_top_hpa,
        config.pressure_bottom_hpa,
        inclusive="both",
    )].copy()
    grouped = work.groupby("profile_id", sort=False)
    agg = grouped["pressure_hpa"].agg(
        n_levels_to_500="size",
        p_surface_hpa="max",
        p_top_hpa="min",
    )
    missing_temp = grouped["temperature_c"].apply(lambda x: bool(x.isna().any())).rename("has_missing_temp")

    rounded = work["pressure_hpa"].round(2)
    dup_flag = pd.DataFrame({"profile_id": work["profile_id"].astype(str), "p": rounded}).duplicated(["profile_id", "p"])
    duplicate_count = dup_flag.groupby(work["profile_id"].astype(str)).sum().rename("duplicate_pressure_count")

    surface_idx = grouped["pressure_hpa"].idxmax()
    surface_heights = work.loc[surface_idx, ["profile_id", "height_m"]].set_index("profile_id")["height_m"].rename("surface_height_m")

    out = meta.join(total, how="outer").join(agg, how="left").join(missing_temp, how="left")
    out = out.join(duplicate_count, how="left").join(surface_heights, how="left")
    out["n_levels_to_500"] = out["n_levels_to_500"].fillna(0).astype(int)
    out["duplicate_pressure_count"] = out["duplicate_pressure_count"].fillna(0).astype(int)
    out["has_missing_temp"] = out["has_missing_temp"].fillna(False).astype(bool)

    status = np.full(len(out), "good", dtype=object)
    no_levels = out["n_levels_to_500"].eq(0)
    status[no_levels.to_numpy()] = "no_500"
    bad_pressure = out["p_surface_hpa"].isna()
    status[bad_pressure.to_numpy()] = "bad_pressure"
    no_temp = out["has_missing_temp"] & ~no_levels
    status[no_temp.to_numpy()] = "no_temp"
    duplicates = out["duplicate_pressure_count"].gt(0) & ~no_levels & ~no_temp
    status[duplicates.to_numpy()] = "duplicate_levels"
    no_top = (
        out["p_top_hpa"].gt(config.pressure_top_hpa + config.exact_top_tolerance_hpa)
        & ~no_levels & ~no_temp & ~duplicates
    )
    status[no_top.to_numpy()] = "no_500"
    short = (
        out["n_levels_to_500"].lt(config.min_levels_to_500)
        & ~no_levels & ~no_temp & ~duplicates & ~no_top
    )
    status[short.to_numpy()] = "short"
    out["profile_status"] = status

    pressure_ok = out["p_surface_hpa"].isna() | out["p_surface_hpa"].le(config.max_surface_pressure_hpa)
    height_ok = out["surface_height_m"].isna() | (
        out["surface_height_m"].sub(config.station_elevation_m).abs().le(config.max_surface_height_deviation_m)
    )
    out["strict_surface_ok"] = pressure_ok & (height_ok if config.use_surface_height_qc else True)
    out["eligible_article"] = out["profile_status"].eq("good") & (
        out["strict_surface_ok"] | (not config.strict_surface_qc)
    )

    return out.reset_index().sort_values("datetime_utc").reset_index(drop=True)

def _month_bounds(year: int, month: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    first = pd.Timestamp(year=year, month=month, day=1)
    last_day = calendar.monthrange(year, month)[1]
    last = pd.Timestamp(year=year, month=month, day=last_day, hour=23, minute=59, second=59)
    return first, last


def compute_completeness(
    df: pd.DataFrame,
    *,
    cycles: Iterable[str] = ("00", "12"),
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    edge_months_within_observed_window: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Рассчитать месячную полноту как долю доступных сроков от ожидаемых."""
    cycles = tuple(str(x).zfill(2) for x in cycles)
    profiles = (
        df[df["cycle"].isin(cycles)][["datetime_utc", "date", "year", "month", "cycle", "profile_id"]]
        .drop_duplicates("profile_id")
        .copy()
    )
    if profiles.empty:
        raise ValueError("После фильтра по срокам не осталось профилей")

    observed_start = profiles["datetime_utc"].min()
    observed_end = profiles["datetime_utc"].max()
    window_start = max(observed_start, pd.Timestamp(start_date)) if start_date is not None else observed_start
    window_end = min(observed_end, pd.Timestamp(end_date)) if end_date is not None else observed_end

    periods = pd.period_range(window_start.to_period("M"), window_end.to_period("M"), freq="M")
    rows = []
    for period in periods:
        month_start, month_end = _month_bounds(period.year, period.month)
        if edge_months_within_observed_window:
            active_start = max(month_start, window_start.normalize())
            active_end = min(month_end, window_end)
        else:
            active_start, active_end = month_start, month_end
        expected_days = max(0, (active_end.normalize() - active_start.normalize()).days + 1)
        expected = expected_days * len(cycles)
        mask = (profiles["year"] == period.year) & (profiles["month"] == period.month)
        observed = profiles.loc[mask, ["date", "cycle"]].drop_duplicates().shape[0]
        completeness = min(100.0, observed / expected * 100.0) if expected else np.nan
        rows.append(
            {
                "year": period.year,
                "month": period.month,
                "observed_profiles": observed,
                "expected_profiles": expected,
                "completeness_percent": completeness,
            }
        )
    long = pd.DataFrame(rows)
    matrix = long.pivot(index="year", columns="month", values="completeness_percent").reindex(columns=range(1, 13))
    return long, matrix
