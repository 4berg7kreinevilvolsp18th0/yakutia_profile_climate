"""Интерактивный дашборд температурных профилей (наблюдения / сроки).

Запуск:
  py -3 -m streamlit run scripts/profile_dashboard.py

Кнопки QC только предлагают отключение наблюдений — можно править вручную.
"""
from __future__ import annotations

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
    spike_scores,
    suggest_outliers_abs_dt,
    suggest_outliers_dt_dp_sq,
    suggest_outliers_few_levels,
    suggest_outliers_form,
    suggest_outliers_spike,
)

DEFAULT_DATA = ROOT / "gdex_outputs" / "profile_climate" / "aldan" / "daily_profiles.json"
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


@st.cache_data(show_spinner="Загрузка профилей…")
def load_daily(path: str, mtime_ns: int) -> dict:
    """mtime_ns — ключ кэша: после пересборки JSON подхватывается новый файл."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
) -> list[dict]:
    """Месяц уже выбран снаружи; здесь даты / cycle / инверсия."""
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
        out.append(obs)
    return out


def month_mean(
    observations: list[dict],
    enabled: set[str],
    *,
    y_axis: str = "height",
) -> tuple[np.ndarray, np.ndarray] | None:
    """Средний профиль включённых наблюдений (после anti-spiral prepare)."""
    series: list[tuple[np.ndarray, np.ndarray]] = []
    for obs in observations:
        if obs["profile_id"] not in enabled:
            continue
        prepared = prepare_plot_arrays(obs, y_axis)
        if prepared is None:
            continue
        t, y = prepared
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
    return grid, np.nanmean(np.vstack(stack), axis=0)


def _first_valid_temp(temps: np.ndarray) -> float | None:
    for value in temps:
        if not np.isnan(value):
            return float(value)
    return None


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


def _obs_state_key(month_key: str, profile_id: str) -> str:
    return f"obs::{month_key}::{profile_id}"


def _set_enabled(month_key: str, observations: list[dict], predicate) -> None:
    for obs in observations:
        st.session_state[_obs_state_key(month_key, obs["profile_id"])] = bool(predicate(obs))


def main() -> None:
    st.set_page_config(page_title="Aldan profile dashboard", layout="wide")
    st.title("Алдан — профили наблюдений")
    st.caption(
        "Одна кривая = один зонд (срок). Фильтры: 00/12, диапазон дней, инверсия. "
        "QC: spike / форма / |ΔT| / (ΔT/ΔP)². Линии без спиралей (монотонный Y)."
    )

    data_path = st.sidebar.text_input("daily_profiles.json", str(DEFAULT_DATA))
    data_file = Path(data_path)
    if not data_file.exists():
        st.error(
            "Нет файла данных. Сначала выполните:\n\n"
            "`python scripts/build_daily_profiles.py`"
        )
        return

    data = load_daily(str(data_file), data_file.stat().st_mtime_ns)
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

    st.sidebar.success(f"schema={REQUIRED_SCHEMA} · n_obs={data.get('n_observations', '—')}")

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

    show_day_means = st.sidebar.checkbox(
        "Показать суточные средние",
        value=False,
        help="Серые линии day_mean для дней с ≥1 включённым наблюдением.",
    )

    y_axis_label = st.sidebar.radio(
        "Вертикальная ось",
        options=["Давление, гПа", "Высота, м"],
        index=0,
        help="После анти-спираль фильтра высота тоже без петель; давление надёжнее физически.",
    )
    y_axis = "pressure" if y_axis_label.startswith("Давление") else "height"

    visible = filter_observations(
        observations,
        cycle_mode=cycle_mode,
        day_from=day_from,
        day_to=day_to,
        inversion_only=inversion_only,
    )
    visible_ids = {o["profile_id"] for o in visible}
    visible_by_day: dict[str, list[dict]] = {}
    for obs in visible:
        visible_by_day.setdefault(obs["date"], []).append(obs)

    st.sidebar.markdown("### Наблюдения")
    p1, p2 = st.sidebar.columns(2)
    if p1.button("Все видимые", use_container_width=True):
        _set_enabled(month_key, observations, lambda o: o["profile_id"] in visible_ids)
    if p2.button("Сброс видимых", use_container_width=True):
        for obs in visible:
            st.session_state[_obs_state_key(month_key, obs["profile_id"])] = False
    p3, p4 = st.sidebar.columns(2)
    if p3.button("Только 00", use_container_width=True):
        _set_enabled(
            month_key,
            observations,
            lambda o: o["profile_id"] in visible_ids and str(o.get("cycle", "")).zfill(2)[-2:] == "00",
        )
    if p4.button("Только 12", use_container_width=True):
        _set_enabled(
            month_key,
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
            st.session_state[_obs_state_key(month_key, pid)] = pid not in outlier_ids

    visible_id_list = [o["profile_id"] for o in visible]
    if q1.button(
        "по spike",
        use_container_width=True,
        help=f"Hampel: |r| > max({HAMPEL_K}·1.4826·MAD(r), {SPIKE_ABS_C}°C)",
    ):
        _apply_outliers(set(suggest_outliers_spike(visible, set(visible_id_list))))
    if q2.button(
        "по форме",
        use_container_width=True,
        help=f"RMSE(T−Ts) ≥ max(P{FORM_PERCENTILE:.0f}, {FORM_RMSE_MIN_C}°C)",
    ):
        _apply_outliers(set(suggest_outliers_form(visible, set(visible_id_list))))
    if q3.button(
        "по |ΔT|",
        use_container_width=True,
        help=f"max |ΔT| ≥ {OUTLIER_MAX_ABS_DT_C} °C (по давлению)",
    ):
        _apply_outliers(set(suggest_outliers_abs_dt(visible, set(visible_id_list))))
    if q4.button(
        "по (ΔT/ΔP)²",
        use_container_width=True,
        help=f"max (ΔT/ΔP)² ≥ {OUTLIER_MAX_DT_DP_SQ}",
    ):
        _apply_outliers(set(suggest_outliers_dt_dp_sq(visible, set(visible_id_list))))
    if q5.button(
        "мало уровней",
        use_container_width=True,
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
                key = _obs_state_key(month_key, obs["profile_id"])
                if key not in st.session_state:
                    st.session_state[key] = True
                label = (
                    f"CY{obs.get('cycle', '??')} · "
                    f"Ts={obs.get('t_surface_c')}°C · "
                    f"L={obs.get('n_levels', len(obs.get('temperature_c') or []))}"
                )
                if obs.get("inversion_detected"):
                    label += " · inv"
                if st.checkbox(label, key=key):
                    enabled.add(obs["profile_id"])

    st.sidebar.caption(f"Включено: {len(enabled)} / видимых {len(visible)}")
    mean = month_mean(visible, enabled, y_axis=y_axis)

    st.info(
        f"Фильтр: срок **{cycle_mode}** · дни "
        f"**{day_from.isoformat()}…{day_to.isoformat()}**"
        + (" · только инверсии" if inversion_only else "")
        + f" · видимо **{len(visible)}**, включено **{len(enabled)}**"
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Наблюдений", f"{len(enabled)} / {len(visible)}")
    m2.metric(
        "Дней с данными",
        len({o["date"] for o in visible if o["profile_id"] in enabled}),
    )
    m3.metric(
        "С инверсией",
        sum(1 for o in visible if o["profile_id"] in enabled and o.get("inversion_detected")),
    )
    if mean is not None:
        ts = _first_valid_temp(mean[1])
        m4.metric("Ts среднего, °C", f"{ts:.1f}" if ts is not None else "—")
    else:
        m4.metric("Ts среднего, °C", "—")

    fig = go.Figure()
    y_hover = "P=%{y:.1f} гПа" if y_axis == "pressure" else "h=%{y:.0f} м"
    color_idx = 0
    day_lookup = {d["date"]: d for d in days}

    for day_key in sorted(visible_by_day):
        day_has_enabled = False
        for obs in visible_by_day[day_key]:
            if obs["profile_id"] not in enabled:
                continue
            prepared = prepare_plot_arrays(obs, y_axis)
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
                mode="lines",
                name=name,
                line=dict(
                    width=1.6,
                    color=color,
                    dash=_cycle_dash(str(obs.get("cycle", ""))),
                ),
                opacity=0.88,
                connectgaps=False,
                hovertemplate=(
                    f"{obs.get('datetime_utc', day_key)}<br>"
                    f"CY{obs.get('cycle', '??')}<br>"
                    f"T=%{{x:.1f}} °C<br>"
                    f"{y_hover}<extra></extra>"
                ),
            ))

        day = day_lookup.get(day_key)
        if show_day_means and day_has_enabled and day and day.get("day_mean"):
            prepared = prepare_plot_arrays(day["day_mean"], y_axis)
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
        title=f"{data.get('station_name', 'Aldan')} — {month_key} (наблюдения)",
        xaxis_title="Температура, °C",
        yaxis=yaxis_cfg,
        height=720,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        margin=dict(l=40, r=20, t=60, b=40),
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    if enabled:
        enabled_obs = [o for o in visible if o["profile_id"] in enabled]
        shape_stats = month_median_shape(enabled_obs)
        form_thr = form_rmse_threshold(enabled_obs)
        rows = []
        n_flag_dt = n_flag_grad = n_flag_spike = n_flag_form = n_flag_few = 0
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
            flag_form = (
                frmse is not None
                and form_thr is not None
                and frmse >= form_thr
            )
            if flag_dt:
                n_flag_dt += 1
            if flag_grad:
                n_flag_grad += 1
            if flag_spike:
                n_flag_spike += 1
            if flag_form:
                n_flag_form += 1
            if few:
                n_flag_few += 1
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
        st.subheader("Сравнение критериев выбросов (наблюдения)")
        c_a, c_b, c_c, c_d, c_e = st.columns(5)
        c_a.metric("Флаг spike", n_flag_spike)
        c_b.metric("Флаг форма", n_flag_form)
        c_c.metric("Флаг |ΔT|", n_flag_dt)
        c_d.metric("Флаг (ΔT/ΔP)²", n_flag_grad)
        c_e.metric("Мало уровней", n_flag_few)
        if form_thr is not None:
            st.caption(f"Порог формы (P{FORM_PERCENTILE:.0f}, min {FORM_RMSE_MIN_C}°C): {form_thr:.2f} °C")
        st.dataframe(rows, use_container_width=True, hide_index=True)
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
