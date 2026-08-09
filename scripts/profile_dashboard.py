"""Интерактивный дашборд температурных профилей (наблюдения / сроки).

Запуск:
  py -3 -m streamlit run scripts/profile_dashboard.py

Кнопки QC только предлагают отключение наблюдений — можно править вручную.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gdex_bufr.profile_climate.obs_qc import (  # noqa: E402
    FORM_PERCENTILE,
    FORM_RMSE_MIN_C,
    HAMPEL_K,
    MIN_ABS_DP_HPA,
    MIN_LEVELS_FLAG,
    OUTLIER_MAX_ABS_DT_C,
    OUTLIER_MAX_DT_DP_SQ,
    SPIKE_ABS_C,
    form_rmse,
    form_rmse_threshold,
    is_few_levels,
    is_spike_outlier,
    max_abs_dt,
    max_dt_dp_sq,
    month_median_shape,
    prepare_plot_arrays,
    raw_plot_arrays,
    spike_scores,
    suggest_outliers_abs_dt,
    suggest_outliers_dt_dp_sq,
    suggest_outliers_few_levels,
    suggest_outliers_form,
    suggest_outliers_spike,
)

ACTUAL_DATA = ROOT / "gdex_outputs" / "актуальное" / "daily_profiles.json"


def _data_path_from_cli() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data")
    args, _ = parser.parse_known_args()
    return (
        Path(args.data)
        if args.data
        else ACTUAL_DATA
    )


DEFAULT_DATA = _data_path_from_cli()
LEGACY_DATA = ROOT / "gdex_outputs" / "результаты-алдан" / "daily_profiles.json"
REQUIRED_SCHEMA = "observations_v1"

OBS_PALETTE = [
    "#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E",
    "#E6AB02", "#A6761D", "#666666", "#1F78B4", "#B2DF8A",
    "#33A02C", "#FB9A99", "#E31A1C", "#FDBF6F", "#FF7F00",
    "#CAB2D6", "#6A3D9A", "#FFFF99", "#B15928", "#8DD3C7",
    "#80B1D3", "#FDB462", "#B3DE69", "#FCCDE5",
]
MEAN_COLOR = "#8B1E3F"
DAY_MEAN_COLOR = "#4A4A4A"

QUALITY_ANY = "любое"
QUALITY_LABELS = {
    "confirmed": "confirmed — подтверждённая",
    "rejected_no_lapse": "rejected_no_lapse — кандидат без падения выше",
    "none": "none — роста T от земли нет",
}


@st.cache_data(show_spinner="Загрузка профилей…")
def load_daily(path: str, mtime_ns: int) -> dict:
    """mtime_ns — ключ кэша: после пересборки JSON подхватывается новый файл."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def has_levels(obs: dict) -> bool:
    """Наблюдение с уровнями (метрики без профиля рисовать нечем)."""
    if obs.get("missing_levels"):
        return False
    return bool(obs.get("temperature_c")) and bool(obs.get("pressure_hpa"))


def _iter_observations(days: list[dict]) -> list[dict]:
    out: list[dict] = []
    for day in days:
        for obs in day.get("observations") or []:
            out.append({**obs, "date": day["date"]})
    return out


def _parse_day(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def filter_observations(
    observations: list[dict],
    *,
    cycle_mode: str,
    day_from: date,
    day_to: date,
    inversion_only: bool,
    inversion_quality: str = QUALITY_ANY,
    hide_missing_levels: bool = False,
) -> list[dict]:
    """Месяц уже выбран снаружи; здесь даты / cycle / инверсия / наличие уровней."""
    out: list[dict] = []
    for obs in observations:
        d = _parse_day(obs["date"])
        if d < day_from or d > day_to:
            continue
        cy = str(obs.get("cycle", "")).zfill(2)[-2:]
        if cycle_mode == "00" and cy != "00":
            continue
        if cycle_mode == "12" and cy != "12":
            continue
        if inversion_only and not obs.get("inversion_detected"):
            continue
        if inversion_quality != QUALITY_ANY:
            if str(obs.get("inversion_quality") or "") != inversion_quality:
                continue
        if hide_missing_levels and not has_levels(obs):
            continue
        out.append(obs)
    return out


def month_mean(
    observations: list[dict],
    enabled: set[str],
    *,
    y_axis: str = "height",
    apply_plot_qc: bool = True,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Средний профиль включённых наблюдений — в том же режиме, что и сами кривые."""
    series: list[tuple[np.ndarray, np.ndarray]] = []
    for obs in observations:
        if obs["profile_id"] not in enabled:
            continue
        prepared = observation_plot_arrays(obs, y_axis, apply_plot_qc=apply_plot_qc)
        if prepared is None:
            continue
        t, y = prepared
        if len(t) < 2:
            continue
        series.append((y, t))
    if not series:
        return None

    y_lo = min(float(y.min()) for y, _ in series)
    y_hi = max(float(y.max()) for y, _ in series)
    if y_axis == "pressure":
        grid = np.linspace(y_hi, y_lo, 40)
    else:
        grid = np.linspace(y_lo, y_hi, 40)

    stack = []
    for y, t in series:
        order = np.argsort(y)
        stack.append(np.interp(grid, y[order], t[order], left=np.nan, right=np.nan))
    stacked = np.vstack(stack)
    # np.nanmean по колонке без данных выдаёт предупреждение — считаем только заполненные
    counted = np.count_nonzero(~np.isnan(stacked), axis=0)
    mean_values = np.full(grid.shape, np.nan, dtype=float)
    filled = counted > 0
    if filled.any():
        mean_values[filled] = np.nanmean(stacked[:, filled], axis=0)
    return grid, mean_values


def observation_plot_arrays(
    obs: dict,
    y_axis: str,
    *,
    apply_plot_qc: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Массивы для видимой кривой: исходные по умолчанию, QC только по запросу."""
    if apply_plot_qc:
        return prepare_plot_arrays(obs, y_axis)
    return raw_plot_arrays(obs, y_axis)


def _first_valid_temp(temps: np.ndarray) -> float | None:
    for value in temps:
        if not np.isnan(value):
            return float(value)
    return None


def _inversion_label_suffix(obs: dict) -> str:
    if not obs.get("inversion_detected"):
        # v2: кандидат без подтверждённого падения T выше верха
        if obs.get("inversion_candidate"):
            return " · inv? кандидат"
        return ""
    h = obs.get("inversion_top_height_m")
    p = obs.get("inversion_top_pressure_hpa")
    bits = [" · inv"]
    if h is not None:
        bits.append(f" {float(h):.0f} м")
    if p is not None:
        bits.append(f" / {float(p):.0f} гПа")
    return "".join(bits)


def _format_height_sources(obs: dict) -> str:
    """Состав источников высоты внутри зонда, например `level×12, interp×3`."""
    counts = obs.get("height_source_counts") or {}
    if not counts:
        return ""
    ordered = sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
    return ", ".join(f"{name}×{count}" for name, count in ordered)


def _finite_or_none(value: float) -> float | None:
    if value == float("inf") or value != value:  # noqa: PLR0124
        return None
    return round(value, 4)


def _cycle_dash(cycle: str) -> str:
    c = str(cycle).zfill(2)[-2:]
    if c == "00":
        return "solid"
    if c == "12":
        return "dash"
    return "dot"


def _obs_state_key(state_scope: str, profile_id: str) -> str:
    return f"obs::{state_scope}::{profile_id}"


def _set_enabled(state_scope: str, observations: list[dict], predicate) -> None:
    for obs in observations:
        st.session_state[_obs_state_key(state_scope, obs["profile_id"])] = bool(predicate(obs))


def _add_inversion_marker(
    fig: go.Figure,
    obs: dict,
    *,
    y_axis: str,
    color: str,
    day_key: str,
) -> None:
    """Ромб на верху инверсии, если координаты есть."""
    inv_t = obs.get("inversion_top_temp_c")
    inv_y = (
        obs.get("inversion_top_pressure_hpa")
        if y_axis == "pressure"
        else obs.get("inversion_top_height_m")
    )
    if inv_t is None or inv_y is None:
        return
    h_m = obs.get("inversion_top_height_m")
    p_hpa = obs.get("inversion_top_pressure_hpa")
    d_t = obs.get("inversion_delta_t_c")
    inv_hover = (
        f"Верх инверсии {obs.get('datetime_utc', day_key)}<br>"
        f"T=%{{x:.1f}} °C<br>"
        + (f"H={h_m:.0f} м<br>" if h_m is not None else "")
        + (f"P={p_hpa:.0f} гПа<br>" if p_hpa is not None else "")
        + (f"ΔT={d_t:.1f} °C<br>" if d_t is not None else "")
        + "<extra></extra>"
    )
    fig.add_trace(go.Scatter(
        x=[float(inv_t)],
        y=[float(inv_y)],
        mode="markers",
        name=f"{day_key[8:]}·{obs.get('cycle', '??')} inv",
        marker=dict(
            size=11,
            symbol="diamond",
            color=color,
            line=dict(width=1.2, color="#222222"),
        ),
        showlegend=False,
        hovertemplate=inv_hover,
    ))


def _build_figure(
    *,
    visible_by_day: dict[str, list[dict]],
    enabled: set[str],
    days: list[dict],
    y_axis: str,
    y_axis_label: str,
    apply_plot_qc: bool,
    show_day_means: bool,
    show_inv_top: bool,
    mean: tuple[np.ndarray, np.ndarray] | None,
    station_name: str,
    month_key: str,
) -> go.Figure:
    """Собирает Plotly-график включённых наблюдений."""
    fig = go.Figure()
    y_hover = "P=%{y:.1f} гПа" if y_axis == "pressure" else "h=%{y:.0f} м"
    color_idx = 0
    day_lookup = {d["date"]: d for d in days}

    for day_key in sorted(visible_by_day):
        day_has_enabled = False
        for obs in visible_by_day[day_key]:
            if obs["profile_id"] not in enabled:
                continue
            prepared = observation_plot_arrays(obs, y_axis, apply_plot_qc=apply_plot_qc)
            if prepared is None:
                continue
            t_vals, y_vals = prepared
            day_has_enabled = True
            color = OBS_PALETTE[color_idx % len(OBS_PALETTE)]
            color_idx += 1
            name = f"{day_key[8:]}·{obs.get('cycle', '??')}"
            fig.add_trace(go.Scatter(
                x=t_vals,
                y=y_vals,
                mode="lines+markers" if not apply_plot_qc else "lines",
                name=name,
                line=dict(
                    width=1.6,
                    color=color,
                    dash=_cycle_dash(str(obs.get("cycle", ""))),
                ),
                marker=dict(size=3),
                opacity=0.88,
                connectgaps=False,
                hovertemplate=(
                    f"{obs.get('datetime_utc', day_key)}<br>"
                    f"CY{obs.get('cycle', '??')}<br>"
                    f"T=%{{x:.1f}} °C<br>"
                    f"{y_hover}<extra></extra>"
                ),
            ))
            if show_inv_top and obs.get("inversion_detected"):
                _add_inversion_marker(
                    fig, obs, y_axis=y_axis, color=color, day_key=day_key,
                )

        day = day_lookup.get(day_key)
        if show_day_means and day_has_enabled and day and day.get("day_mean"):
            prepared = observation_plot_arrays(
                day["day_mean"], y_axis, apply_plot_qc=apply_plot_qc,
            )
            if prepared is not None:
                t_vals, y_vals = prepared
                fig.add_trace(go.Scatter(
                    x=t_vals,
                    y=y_vals,
                    mode="lines",
                    name=f"{day_key[8:]} mean",
                    line=dict(width=2.0, color=DAY_MEAN_COLOR, dash="dot"),
                    opacity=0.55,
                    connectgaps=False,
                    hovertemplate=(
                        f"Суточное среднее {day_key}<br>"
                        f"T=%{{x:.1f}} °C<br>"
                        f"{y_hover}<extra></extra>"
                    ),
                    showlegend=False,
                ))

    if mean is not None:
        fig.add_trace(go.Scatter(
            x=mean[1],
            y=mean[0],
            mode="lines",
            name="Среднее (включённые)",
            line=dict(width=3.5, color=MEAN_COLOR),
            connectgaps=False,
            hovertemplate=(
                "Среднее<br>"
                f"T=%{{x:.1f}} °C<br>"
                f"{y_hover}<extra></extra>"
            ),
        ))

    yaxis_cfg: dict = {"title": y_axis_label}
    if y_axis == "pressure":
        yaxis_cfg["autorange"] = "reversed"
    fig.update_layout(
        title=f"{station_name} — {month_key} (наблюдения)",
        xaxis_title="Температура, °C",
        yaxis=yaxis_cfg,
        height=720,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        margin=dict(l=40, r=20, t=60, b=40),
        template="plotly_white",
    )
    return fig


def _qc_table_rows(enabled_obs: list[dict]) -> tuple[list[dict], dict[str, int], float | None]:
    """Строки таблицы QC и счётчики флагов."""
    shape_stats = month_median_shape(enabled_obs)
    form_thr = form_rmse_threshold(enabled_obs)
    rows = []
    flags = {"spike": 0, "form": 0, "dt": 0, "grad": 0, "few": 0}
    for obs in enabled_obs:
        max_dt = max_abs_dt(obs)
        grad_sq = max_dt_dp_sq(obs)
        max_r, n_spike = spike_scores(obs)
        few = is_few_levels(obs)
        frmse = None
        if shape_stats is not None:
            grid, median_anom = shape_stats
            frmse = form_rmse(obs, median_anom, grid)
            if frmse == float("inf"):
                frmse = None
        flag_dt = max_dt >= OUTLIER_MAX_ABS_DT_C
        flag_grad = grad_sq >= OUTLIER_MAX_DT_DP_SQ
        flag_spike = is_spike_outlier(obs)
        flag_form = frmse is not None and form_thr is not None and frmse >= form_thr
        if flag_dt:
            flags["dt"] += 1
        if flag_grad:
            flags["grad"] += 1
        if flag_spike:
            flags["spike"] += 1
        if flag_form:
            flags["form"] += 1
        if few:
            flags["few"] += 1
        rows.append({
            "Дата": obs["date"],
            "Cycle": obs.get("cycle"),
            "profile_id": obs["profile_id"],
            "max |r| spike, °C": _finite_or_none(max_r),
            "n_spike": n_spike if n_spike < 10**8 else None,
            "form RMSE, °C": None if frmse is None else round(frmse, 3),
            "max |ΔT|, °C": _finite_or_none(max_dt),
            "max (ΔT/ΔP)²": _finite_or_none(grad_sq),
            "Уровней": obs.get("n_levels"),
            "Выброс spike?": "да" if flag_spike else "",
            "Выброс форма?": "да" if flag_form else "",
            "Выброс |ΔT|?": "да" if flag_dt else "",
            "Выброс (ΔT/ΔP)²?": "да" if flag_grad else "",
            "Мало уровней?": "да" if few else "",
            "Ts, °C": obs.get("t_surface_c"),
            "Инверсия": "да" if obs.get("inversion_detected") else "нет",
            "Качество": obs.get("inversion_quality") or "",
            "Источники H": _format_height_sources(obs),
            "H_inv, м": obs.get("inversion_top_height_m"),
            "P_inv, гПа": obs.get("inversion_top_pressure_hpa"),
            "ΔT_inv, °C": obs.get("inversion_delta_t_c"),
        })
    rows.sort(
        key=lambda r: (
            r["form RMSE, °C"] if r["form RMSE, °C"] is not None else -1,
            r["n_spike"] if r["n_spike"] is not None else -1,
            r["max (ΔT/ΔP)²"] if r["max (ΔT/ΔP)²"] is not None else -1,
            r["max |ΔT|, °C"] if r["max |ΔT|, °C"] is not None else -1,
        ),
        reverse=True,
    )
    return rows, flags, form_thr


def main() -> None:
    st.set_page_config(page_title="Aldan profile dashboard", layout="wide")
    st.title("Алдан — профили наблюдений")
    st.caption(
        "Одна кривая = один зонд (срок). По умолчанию показаны все исходные уровни "
        "без предварительной QC-фильтрации. Фильтры и исключение выбросов применяются только вручную."
    )

    data_path = st.sidebar.text_input("daily_profiles.json", str(DEFAULT_DATA))
    data_file = Path(data_path)
    if not data_file.exists() and data_file == ACTUAL_DATA and LEGACY_DATA.exists():
        data_file = LEGACY_DATA
        st.sidebar.caption(f"Fallback: {LEGACY_DATA}")
    if not data_file.exists():
        st.error(
            "Нет файла данных. Сначала выполните:\n\n"
            "`python scripts/build_daily_profiles.py`"
        )
        return

    data_mtime_ns = data_file.stat().st_mtime_ns
    try:
        data = load_daily(str(data_file), data_mtime_ns)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as error:
        load_daily.clear()
        st.error(
            f"Не удалось прочитать `{data_file}`: {error}\n\n"
            "Файл повреждён или запись не завершилась. Пересоберите:\n\n"
            "`python scripts/build_daily_profiles.py`"
        )
        return
    if not isinstance(data, dict):
        st.error(f"Ожидался JSON-объект, а в `{data_file}` лежит `{type(data).__name__}`.")
        return
    if data.get("schema") != REQUIRED_SCHEMA:
        st.error(
            f"Нужен JSON со схемой `{REQUIRED_SCHEMA}` (сейчас: `{data.get('schema')}`).\n\n"
            "Пересоберите:\n\n`python scripts/build_daily_profiles.py`\n\n"
            "Если уже пересобрали — сбросьте кэш Streamlit (кнопка ниже или **C** в браузере)."
        )
        if st.button("Сбросить кэш данных"):
            load_daily.clear()
            st.rerun()
        return

    months = sorted(data["months"].keys())
    if not months:
        st.error("В JSON нет месяцев.")
        return

    level_mode = data.get("level_mode", "legacy/clean")
    features = set(data.get("features") or [])
    st.sidebar.success(
        f"schema={REQUIRED_SCHEMA} · уровни={level_mode} · "
        f"n_obs={data.get('n_observations', '—')} · n_levels={data.get('n_levels', '—')}"
        + (
            f" · z_ст={data['station_elevation_m']:.0f} м"
            if data.get("station_elevation_m") is not None
            else ""
        )
    )
    if "inversion_quality" not in features:
        st.sidebar.caption(
            "JSON собран прежней версией: нет полей качества инверсии (v2). "
            "Пересоберите `python scripts/build_daily_profiles.py`, чтобы включить фильтр."
        )

    years = sorted({m[:4] for m in months})
    col_y, col_m = st.sidebar.columns(2)
    year = col_y.selectbox("Год", years, index=len(years) - 1)
    year_months = [m for m in months if m.startswith(year)]
    month_labels = {m: m[5:] for m in year_months}
    month_key = col_m.selectbox(
        "Месяц",
        year_months,
        format_func=lambda m: month_labels[m],
        index=0,
    )

    days = data["months"][month_key]["days"]
    observations = _iter_observations(days)
    if not observations:
        st.warning("В выбранном месяце нет наблюдений.")
        return

    day_dates = sorted({_parse_day(o["date"]) for o in observations})
    d_min, d_max = day_dates[0], day_dates[-1]

    st.sidebar.markdown("### Фильтр")
    cycle_mode = st.sidebar.radio(
        "Срок (UTC)",
        options=["00+12", "00", "12"],
        index=0,
        horizontal=True,
        help="Ограничивает видимый пул наблюдений.",
    )
    if d_min == d_max:
        day_from = day_to = d_min
        st.sidebar.caption(f"День: {d_min.isoformat()}")
    else:
        day_from, day_to = st.sidebar.slider(
            "Диапазон дней",
            min_value=d_min,
            max_value=d_max,
            value=(d_min, d_max),
            format="DD.MM",
        )
    inversion_only = st.sidebar.checkbox("Только с инверсией", value=False)
    inversion_quality = QUALITY_ANY
    if "inversion_quality" in features:
        present_qualities = sorted(
            {
                str(o.get("inversion_quality") or "")
                for o in observations
                if o.get("inversion_quality")
            }
        )
        if present_qualities:
            inversion_quality = st.sidebar.selectbox(
                "Качество инверсии (v2)",
                options=[QUALITY_ANY, *present_qualities],
                format_func=lambda q: QUALITY_LABELS.get(q, q),
                help="confirmed = подтверждена падением T выше верха; остальные — для разбора.",
            )
    hide_missing_levels = st.sidebar.checkbox(
        "Скрыть наблюдения без уровней",
        value=False,
        help="Профили, у которых есть только метрики: в списке видны, но на графике их нет.",
    )

    show_day_means = st.sidebar.checkbox(
        "Показать суточные средние",
        value=False,
        help="Серые линии day_mean для дней с ≥1 включённым наблюдением.",
    )
    show_inv_top = st.sidebar.checkbox(
        "Отметить верх инверсии",
        value=True,
        help="Маркер на графике и высота/давление верха в списке и таблице.",
    )

    y_axis_label = st.sidebar.radio(
        "Вертикальная ось",
        options=["Давление, гПа", "Высота, м"],
        index=0,
        help=(
            "Перевод осей 1:1: уровни всегда идут от земли вверх по давлению. "
            "Меняется только координата Y (гПа или м), набор температур тот же."
        ),
    )
    y_axis = "pressure" if y_axis_label.startswith("Давление") else "height"
    apply_plot_qc = st.sidebar.checkbox(
        "Подготовить кривые (убрать петли и дубли)",
        value=False,
        help=(
            "Выключено: те же уровни, что и по гПа; на оси метров видны провалы H(P). "
            "Включено: на оси метров отбрасываются уровни с падением высоты при подъёме."
        ),
    )

    visible = filter_observations(
        observations,
        cycle_mode=cycle_mode,
        day_from=day_from,
        day_to=day_to,
        inversion_only=inversion_only,
        inversion_quality=inversion_quality,
        hide_missing_levels=hide_missing_levels,
    )
    visible_ids = {o["profile_id"] for o in visible}
    visible_plottable = [o for o in visible if has_levels(o)]
    n_visible_missing = len(visible) - len(visible_plottable)
    state_scope = f"{data_file.resolve()}::{data_mtime_ns}::{month_key}"
    visible_by_day: dict[str, list[dict]] = {}
    for obs in visible:
        visible_by_day.setdefault(obs["date"], []).append(obs)

    st.sidebar.markdown("### Наблюдения")
    p1, p2 = st.sidebar.columns(2)
    if p1.button("Все видимые", width="stretch"):
        _set_enabled(state_scope, observations, lambda o: o["profile_id"] in visible_ids)
    if p2.button("Сброс видимых", width="stretch"):
        for obs in visible:
            st.session_state[_obs_state_key(state_scope, obs["profile_id"])] = False
    p3, p4 = st.sidebar.columns(2)
    if p3.button("Только 00", width="stretch"):
        _set_enabled(
            state_scope,
            observations,
            lambda o: o["profile_id"] in visible_ids and str(o.get("cycle", "")).zfill(2)[-2:] == "00",
        )
    if p4.button("Только 12", width="stretch"):
        _set_enabled(
            state_scope,
            observations,
            lambda o: o["profile_id"] in visible_ids and str(o.get("cycle", "")).zfill(2)[-2:] == "12",
        )

    st.sidebar.markdown("### Выбросы (кандидаты)")
    q1, q2 = st.sidebar.columns(2)
    q3, q4 = st.sidebar.columns(2)
    q5, _ = st.sidebar.columns(2)

    def _apply_outliers(outlier_ids: set[str]) -> None:
        for obs in visible:
            pid = obs["profile_id"]
            st.session_state[_obs_state_key(state_scope, pid)] = pid not in outlier_ids

    visible_id_list = [o["profile_id"] for o in visible]
    if q1.button(
        "по spike",
        width="stretch",
        help=f"Hampel: |r| > max({HAMPEL_K}·1.4826·MAD(r), {SPIKE_ABS_C}°C)",
    ):
        _apply_outliers(set(suggest_outliers_spike(visible, set(visible_id_list))))
    if q2.button(
        "по форме",
        width="stretch",
        help=f"RMSE(T−Ts) ≥ max(P{FORM_PERCENTILE:.0f}, {FORM_RMSE_MIN_C}°C)",
    ):
        _apply_outliers(set(suggest_outliers_form(visible, set(visible_id_list))))
    if q3.button(
        "по |ΔT|",
        width="stretch",
        help=f"max |ΔT| ≥ {OUTLIER_MAX_ABS_DT_C} °C (по давлению)",
    ):
        _apply_outliers(set(suggest_outliers_abs_dt(visible, set(visible_id_list))))
    if q4.button(
        "по (ΔT/ΔP)²",
        width="stretch",
        help=f"max (ΔT/ΔP)² ≥ {OUTLIER_MAX_DT_DP_SQ}",
    ):
        _apply_outliers(set(suggest_outliers_dt_dp_sq(visible, set(visible_id_list))))
    if q5.button(
        "мало уровней",
        width="stretch",
        help=f"n_levels < {MIN_LEVELS_FLAG}",
    ):
        _apply_outliers(set(suggest_outliers_few_levels(visible, set(visible_id_list))))

    st.sidebar.caption(
        f"spike k={HAMPEL_K} abs≥{SPIKE_ABS_C}°C · "
        f"form P{FORM_PERCENTILE:.0f}/min{FORM_RMSE_MIN_C}°C · "
        f"|ΔT|≥{OUTLIER_MAX_ABS_DT_C}°C · "
        f"(ΔT/ΔP)²≥{OUTLIER_MAX_DT_DP_SQ} · "
        f"n<{MIN_LEVELS_FLAG} · min|ΔP|={MIN_ABS_DP_HPA} гПа"
    )

    enabled: set[str] = set()
    with st.sidebar.expander("Список наблюдений", expanded=True):
        if not visible_by_day:
            st.caption("Нет наблюдений по текущему фильтру.")
        for day_key in sorted(visible_by_day):
            day_obs = visible_by_day[day_key]
            st.markdown(f"**{day_key[8:]}** · n={len(day_obs)}")
            for obs in day_obs:
                key = _obs_state_key(state_scope, obs["profile_id"])
                if key not in st.session_state:
                    st.session_state[key] = True
                label = (
                    f"CY{obs.get('cycle', '??')} · "
                    f"Ts={obs.get('t_surface_c')}°C · "
                    f"L={obs.get('n_levels', len(obs.get('temperature_c') or []))}"
                )
                if obs.get("missing_levels"):
                    label += " · нет уровней"
                label += _inversion_label_suffix(obs)
                if st.checkbox(label, key=key):
                    enabled.add(obs["profile_id"])

    st.sidebar.caption(f"Включено: {len(enabled)} / видимых {len(visible)}")
    mean = month_mean(visible, enabled, y_axis=y_axis, apply_plot_qc=apply_plot_qc)

    # На графике участвуют только наблюдения с уровнями — счётчики считаем по ним же.
    enabled_plottable = [o for o in visible_plottable if o["profile_id"] in enabled]
    n_enabled_missing = sum(
        1 for o in visible if o["profile_id"] in enabled and not has_levels(o)
    )

    st.info(
        f"Уровни: **{'подготовленные' if apply_plot_qc else 'все исходные без QC'}** · "
        f"срок **{cycle_mode}** · дни "
        f"**{day_from.isoformat()}…{day_to.isoformat()}**"
        + (" · только инверсии" if inversion_only else "")
        + (
            f" · качество **{inversion_quality}**"
            if inversion_quality != QUALITY_ANY
            else ""
        )
        + f" · на графике **{len(enabled_plottable)}** из **{len(visible_plottable)}**"
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("На графике", f"{len(enabled_plottable)} / {len(visible_plottable)}")
    m2.metric(
        "Без уровней",
        f"{n_enabled_missing} / {n_visible_missing}",
        help="Профили только с метриками: в счёт кривых не входят.",
    )
    m3.metric("Дней с данными", len({o["date"] for o in enabled_plottable}))
    m4.metric(
        "С инверсией",
        sum(1 for o in enabled_plottable if o.get("inversion_detected")),
    )
    inv_heights = [
        float(o["inversion_top_height_m"])
        for o in enabled_plottable
        if o.get("inversion_detected") and o.get("inversion_top_height_m") is not None
    ]
    if inv_heights:
        m5.metric("Ср. H_inv, м", f"{sum(inv_heights) / len(inv_heights):.0f}")
    elif mean is not None:
        ts = _first_valid_temp(mean[1])
        m5.metric("Ts среднего, °C", f"{ts:.1f}" if ts is not None else "—")
    else:
        m5.metric("Ts среднего, °C", "—")

    # График и таблица QC — отдельные шаги, чтобы main оставался последовательностью экрана
    fig = _build_figure(
        visible_by_day=visible_by_day,
        enabled=enabled,
        days=days,
        y_axis=y_axis,
        y_axis_label=y_axis_label,
        apply_plot_qc=apply_plot_qc,
        show_day_means=show_day_means,
        show_inv_top=show_inv_top,
        mean=mean,
        station_name=str(data.get("station_name", "Aldan")),
        month_key=month_key,
    )
    st.plotly_chart(fig, width="stretch")

    if enabled_plottable:
        rows, flags, form_thr = _qc_table_rows(enabled_plottable)
        st.subheader("Сравнение критериев выбросов (наблюдения)")
        c_a, c_b, c_c, c_d, c_e = st.columns(5)
        c_a.metric("Флаг spike", flags["spike"])
        c_b.metric("Флаг форма", flags["form"])
        c_c.metric("Флаг |ΔT|", flags["dt"])
        c_d.metric("Флаг (ΔT/ΔP)²", flags["grad"])
        c_e.metric("Мало уровней", flags["few"])
        if form_thr is not None:
            st.caption(
                f"Порог формы (P{FORM_PERCENTILE:.0f}, min {FORM_RMSE_MIN_C}°C): {form_thr:.2f} °C"
            )
        st.dataframe(rows, width="stretch", hide_index=True)
        st.caption(
            f"Источник: {Path(data_path).name} · schema={REQUIRED_SCHEMA} · "
            f"spike k={HAMPEL_K} abs≥{SPIKE_ABS_C}°C · "
            f"form P{FORM_PERCENTILE:.0f}/min{FORM_RMSE_MIN_C}°C · "
            f"|ΔT|≥{OUTLIER_MAX_ABS_DT_C}°C · "
            f"(ΔT/ΔP)²≥{OUTLIER_MAX_DT_DP_SQ} · "
            f"n<{MIN_LEVELS_FLAG}"
        )


if __name__ == "__main__":
    main()
