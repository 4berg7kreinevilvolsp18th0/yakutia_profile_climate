"""Усреднение температурных профилей: метод A и метод B."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Literal, Sequence

import numpy as np

from gdex_bufr.profile_climate.profile_interpolation import (
    default_target_grid,
    interpolate_temperature_profile,
)

Coordinate = Literal["pressure", "height"]
Method = Literal["A", "B"]
Statistic = Literal["mean", "median"]
RangeMode = Literal["q25_q75", "std1"]
MultiMonthMode = Literal["separate", "combined"]
CycleMode = Literal["00+12", "00", "12", "all"]

MONTH_NAMES_RU = (
    "Январь", "Февраль", "Март", "Апрель", "Май", "Июнь",
    "Июл", "Август", "Сентябрь", "Октябрь", "Ноябрь", "Декабрь",
)
SEASON_MONTHS = {
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}


@dataclass(frozen=True)
class YearMonth:
    year: int
    month: int

    def as_tuple(self) -> tuple[int, int]:
        return self.year, self.month


@dataclass
class AveragingFilters:
    year_start: int
    year_end: int
    selected_years: frozenset[int] | None = None
    selected_months: frozenset[int] = field(default_factory=lambda: frozenset(range(1, 13)))
    excluded_year_months: frozenset[tuple[int, int]] = field(default_factory=frozenset)
    cycle_mode: CycleMode = "00+12"
    profile_ids: frozenset[str] | None = None


@dataclass
class AveragingConfig:
    method: Method = "A"
    coordinate: Coordinate = "pressure"
    statistic: Statistic = "mean"
    range_mode: RangeMode | None = "q25_q75"
    multi_month_mode: MultiMonthMode = "separate"
    min_samples_a: int = 10
    min_samples_b: int = 5
    target_grid: np.ndarray | None = None
    apply_plot_qc: bool = False


@dataclass
class SingleMonthAverageResult:
    month: int
    grid: np.ndarray
    central: np.ndarray
    median: np.ndarray
    q25: np.ndarray
    q75: np.ndarray
    std: np.ndarray
    n_profiles: np.ndarray
    n_year_months: np.ndarray
    individual_profiles: list[np.ndarray] = field(default_factory=list)
    year_month_profiles: list[tuple[int, int, np.ndarray]] = field(default_factory=list)
    n_original_profiles: int = 0
    n_year_month_groups: int = 0
    included_year_months: list[tuple[int, int]] = field(default_factory=list)
    excluded_year_months: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class AveragingResult:
    coordinate: Coordinate
    method: Method
    statistic: Statistic
    range_mode: RangeMode | None
    months: list[SingleMonthAverageResult]
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_observation_year_month(obs: dict[str, Any]) -> tuple[int, int]:
    date_str = str(obs.get("date") or "")
    dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
    return dt.year, dt.month


def observation_cycle(obs: dict[str, Any]) -> str:
    return str(obs.get("cycle", "")).zfill(2)[-2:]


def cycle_matches(cycle: str, mode: CycleMode) -> bool:
    if mode == "all":
        return True
    if mode == "00+12":
        return cycle in {"00", "12"}
    return cycle == mode


def has_levels(obs: dict[str, Any]) -> bool:
    if obs.get("missing_levels"):
        return False
    return bool(obs.get("temperature_c")) and bool(obs.get("pressure_hpa"))


def extract_level_arrays(
    obs: dict[str, Any],
    *,
    apply_plot_qc: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """pressure, height, temperature — сырые уровни наблюдения."""
    if not has_levels(obs):
        return None
    if apply_plot_qc:
        from gdex_bufr.profile_climate.obs_qc import prepare_plot_arrays

        t_p = prepare_plot_arrays(obs, "pressure")
        t_h = prepare_plot_arrays(obs, "height")
        if t_p is None or t_h is None:
            return None
        t = t_p[0]
        p = t_p[1]
        h = t_h[1]
        return p, h, t

    p = np.asarray(obs["pressure_hpa"], dtype=float)
    t = np.asarray(obs["temperature_c"], dtype=float)
    h = np.asarray(obs.get("heights_m") or obs.get("height_m") or [], dtype=float)
    if len(h) != len(p):
        h = np.full_like(p, np.nan)
    return p, h, t


def filter_observations_for_averaging(
    observations: Sequence[dict[str, Any]],
    filters: AveragingFilters,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for obs in observations:
        if not has_levels(obs):
            continue
        year, month = parse_observation_year_month(obs)
        if year < filters.year_start or year > filters.year_end:
            continue
        if filters.selected_years is not None and year not in filters.selected_years:
            continue
        if month not in filters.selected_months:
            continue
        if (year, month) in filters.excluded_year_months:
            continue
        if not cycle_matches(observation_cycle(obs), filters.cycle_mode):
            continue
        pid = str(obs.get("profile_id") or "")
        if filters.profile_ids is not None and pid not in filters.profile_ids:
            continue
        out.append(obs)
    return out


def interpolate_observation(
    obs: dict[str, Any],
    grid: np.ndarray,
    *,
    coordinate: Coordinate,
    apply_plot_qc: bool = False,
) -> np.ndarray | None:
    levels = extract_level_arrays(obs, apply_plot_qc=apply_plot_qc)
    if levels is None:
        return None
    p, h, t = levels
    return interpolate_temperature_profile(p, h, t, grid, coordinate=coordinate)


def _nan_stat(values: np.ndarray, axis: int, stat: Statistic) -> np.ndarray:
    if stat == "median":
        return np.nanmedian(values, axis=axis)
    return np.nanmean(values, axis=axis)


def _apply_min_count(
    values: np.ndarray,
    counts: np.ndarray,
    min_count: int,
) -> np.ndarray:
    out = values.copy()
    out[counts < min_count] = np.nan
    return out


def compute_method_a_on_stack(
    stack: np.ndarray,
    *,
    statistic: Statistic = "mean",
    min_samples: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """stack: (n_profiles, n_levels)."""
    if stack.size == 0:
        n = stack.shape[1] if stack.ndim == 2 else 0
        empty = np.full(n, np.nan)
        return empty, empty, empty, empty, empty, np.zeros(n, dtype=int)

    counted = np.sum(~np.isnan(stack), axis=0)
    central = _nan_stat(stack, axis=0, stat=statistic)
    median = np.nanmedian(stack, axis=0)
    q25 = np.nanpercentile(stack, 25, axis=0)
    q75 = np.nanpercentile(stack, 75, axis=0)
    std = np.nanstd(stack, axis=0)
    central = _apply_min_count(central, counted, min_samples)
    return central, median, q25, q75, std, counted.astype(int)


def compute_method_b_on_year_month_profiles(
    ym_profiles: list[np.ndarray],
    *,
    statistic: Statistic = "mean",
    min_year_months: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not ym_profiles:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )
    stack = np.vstack(ym_profiles)
    counted = np.full(stack.shape[1], stack.shape[0], dtype=int)
    central = _nan_stat(stack, axis=0, stat=statistic)
    median = np.nanmedian(stack, axis=0)
    q25 = np.nanpercentile(stack, 25, axis=0)
    q75 = np.nanpercentile(stack, 75, axis=0)
    std = np.nanstd(stack, axis=0)
    valid_levels = np.sum(~np.isnan(stack), axis=0)
    central = _apply_min_count(central, valid_levels, min_year_months)
    return central, median, q25, q75, std, counted


def _year_month_mean_profiles(
    observations: list[dict[str, Any]],
    grid: np.ndarray,
    *,
    coordinate: Coordinate,
    apply_plot_qc: bool,
) -> dict[tuple[int, int], np.ndarray]:
    groups: dict[tuple[int, int], list[np.ndarray]] = {}
    for obs in observations:
        ym = parse_observation_year_month(obs)
        interp = interpolate_observation(
            obs, grid, coordinate=coordinate, apply_plot_qc=apply_plot_qc,
        )
        if interp is None:
            continue
        groups.setdefault(ym, []).append(interp)
    return {
        ym: np.nanmean(np.vstack(rows), axis=0)
        for ym, rows in groups.items()
        if rows
    }


def compute_single_month_average(
    observations: list[dict[str, Any]],
    month: int,
    config: AveragingConfig,
    filters: AveragingFilters,
) -> SingleMonthAverageResult | None:
    month_obs = [o for o in observations if parse_observation_year_month(o)[1] == month]
    if not month_obs:
        return None

    grid = config.target_grid
    if grid is None:
        grid = default_target_grid(coordinate=config.coordinate)

    stacks_a: list[np.ndarray] = []
    for obs in month_obs:
        row = interpolate_observation(
            obs, grid, coordinate=config.coordinate, apply_plot_qc=config.apply_plot_qc,
        )
        if row is not None:
            stacks_a.append(row)

    if not stacks_a:
        return None

    stack_a = np.vstack(stacks_a)
    central_a, median_a, q25_a, q75_a, std_a, n_profiles = compute_method_a_on_stack(
        stack_a,
        statistic=config.statistic,
        min_samples=config.min_samples_a,
    )

    ym_means = _year_month_mean_profiles(
        month_obs, grid, coordinate=config.coordinate, apply_plot_qc=config.apply_plot_qc,
    )
    ym_list = sorted(ym_means.items())
    ym_profiles = [v for _, v in ym_list]
    central_b, median_b, q25_b, q75_b, std_b, n_ym = compute_method_b_on_year_month_profiles(
        ym_profiles,
        statistic=config.statistic,
        min_year_months=config.min_samples_b,
    )

    if config.method == "B":
        central, median, q25, q75, std = central_b, median_b, q25_b, q75_b, std_b
    else:
        central, median, q25, q75, std = central_a, median_a, q25_a, q75_a, std_a

    all_ym_in_range = {
        (y, m)
        for y in range(filters.year_start, filters.year_end + 1)
        for m in [month]
        if m in filters.selected_months
        and (filters.selected_years is None or y in filters.selected_years)
    }
    included = [ym for ym in all_ym_in_range if ym in ym_means]
    excluded = sorted(all_ym_in_range - set(included) | filters.excluded_year_months)

    return SingleMonthAverageResult(
        month=month,
        grid=grid,
        central=central,
        median=median,
        q25=q25 if config.method == "B" else q25_a,
        q75=q75 if config.method == "B" else q75_a,
        std=std,
        n_profiles=n_profiles,
        n_year_months=n_ym if config.method == "B" else np.full_like(n_profiles, len(ym_profiles)),
        individual_profiles=stacks_a,
        year_month_profiles=[(y, m, arr) for (y, m), arr in ym_list],
        n_original_profiles=len(stacks_a),
        n_year_month_groups=len(ym_profiles),
        included_year_months=sorted(included),
        excluded_year_months=sorted(excluded),
    )


def compute_profile_average(
    observations: Sequence[dict[str, Any]],
    filters: AveragingFilters,
    config: AveragingConfig,
) -> AveragingResult:
    pool = filter_observations_for_averaging(observations, filters)
    months_to_process = sorted(filters.selected_months)

    if config.multi_month_mode == "combined":
        # один результат на объединённую выборку — month=0 маркер
        combined = compute_single_month_average(
            pool,
            month=months_to_process[0] if len(months_to_process) == 1 else months_to_process[0],
            config=config,
            filters=filters,
        )
        if combined is None:
            months_results: list[SingleMonthAverageResult] = []
        else:
            # пересчёт на всей выборке без фильтра по одному month
            grid = config.target_grid or default_target_grid(coordinate=config.coordinate)
            stacks: list[np.ndarray] = []
            for obs in pool:
                row = interpolate_observation(
                    obs, grid, coordinate=config.coordinate, apply_plot_qc=config.apply_plot_qc,
                )
                if row is not None:
                    stacks.append(row)
            stack_a = np.vstack(stacks) if stacks else np.empty((0, len(grid)))
            central_a, median_a, q25_a, q75_a, std_a, n_profiles = compute_method_a_on_stack(
                stack_a, statistic=config.statistic, min_samples=config.min_samples_a,
            )
            ym_means = _year_month_mean_profiles(
                pool, grid, coordinate=config.coordinate, apply_plot_qc=config.apply_plot_qc,
            )
            ym_list = sorted(ym_means.items())
            ym_profiles = [v for _, v in ym_list]
            central_b, median_b, q25_b, q75_b, std_b, n_ym = compute_method_b_on_year_month_profiles(
                ym_profiles, statistic=config.statistic, min_year_months=config.min_samples_b,
            )
            if config.method == "B":
                central, median, q25, q75, std = central_b, median_b, q25_b, q75_b, std_b
            else:
                central, median, q25, q75, std = central_a, median_a, q25_a, q75_a, std_a
            combined = SingleMonthAverageResult(
                month=0,
                grid=grid,
                central=central,
                median=median,
                q25=q25 if config.method == "B" else q25_a,
                q75=q75 if config.method == "B" else q75_a,
                std=std,
                n_profiles=n_profiles,
                n_year_months=n_ym,
                individual_profiles=stacks,
                year_month_profiles=[(y, m, arr) for (y, m), arr in ym_list],
                n_original_profiles=len(stacks),
                n_year_month_groups=len(ym_profiles),
                included_year_months=sorted(ym_means.keys()),
                excluded_year_months=sorted(filters.excluded_year_months),
            )
            months_results = [combined]
    else:
        months_results = []
        for m in months_to_process:
            res = compute_single_month_average(pool, m, config, filters)
            if res is not None:
                months_results.append(res)

    meta = {
        "method": config.method,
        "coordinate": config.coordinate,
        "statistic": config.statistic,
        "range_mode": config.range_mode,
        "year_start": filters.year_start,
        "year_end": filters.year_end,
        "selected_months": sorted(filters.selected_months),
        "cycle_mode": filters.cycle_mode,
        "n_pool_profiles": len(pool),
        "interpolation": "linear, no extrapolation",
        "grid_step": "25 hPa" if config.coordinate == "pressure" else "100 m AGL",
    }
    return AveragingResult(
        coordinate=config.coordinate,
        method=config.method,
        statistic=config.statistic,
        range_mode=config.range_mode,
        months=months_results,
        metadata=meta,
    )


def compare_methods_delta(
    observations: Sequence[dict[str, Any]],
    filters: AveragingFilters,
    config: AveragingConfig,
    month: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Возвращает grid, T_A, T_B − T_A для одного месяца."""
    cfg_a = AveragingConfig(**{**config.__dict__, "method": "A"})
    cfg_b = AveragingConfig(**{**config.__dict__, "method": "B"})
    filt = AveragingFilters(
        **{**filters.__dict__, "selected_months": frozenset([month])},
    )
    res_a = compute_profile_average(observations, filt, cfg_a)
    res_b = compute_profile_average(observations, filt, cfg_b)
    if not res_a.months or not res_b.months:
        return None
    a, b = res_a.months[0], res_b.months[0]
    return a.grid, a.central, a.central - b.central


def average_result_to_csv_rows(result: SingleMonthAverageResult, meta: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for i, coord in enumerate(result.grid):
        rows.append({
            "coordinate": float(coord),
            "mean_temperature_c": float(result.central[i]) if not np.isnan(result.central[i]) else "",
            "median_temperature_c": float(result.median[i]) if not np.isnan(result.median[i]) else "",
            "q25_temperature_c": float(result.q25[i]) if not np.isnan(result.q25[i]) else "",
            "q75_temperature_c": float(result.q75[i]) if not np.isnan(result.q75[i]) else "",
            "std_temperature_c": float(result.std[i]) if not np.isnan(result.std[i]) else "",
            "n_samples": int(result.n_profiles[i]),
            "n_year_months": int(result.n_year_months[i]) if i < len(result.n_year_months) else "",
            "method": meta.get("method"),
            "month": result.month,
            "cycle_filter": meta.get("cycle_mode"),
            "year_start": meta.get("year_start"),
            "year_end": meta.get("year_end"),
        })
    return rows
