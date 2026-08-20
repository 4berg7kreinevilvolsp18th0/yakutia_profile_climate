"""UI и графики климатологического усреднения для Streamlit-дашборда."""
from __future__ import annotations

import csv
import io
from datetime import datetime
from typing import Any, Iterable

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from gdex_bufr.profile_climate.profile_averaging import (
    MONTH_NAMES_RU,
    SEASON_MONTHS,
    AveragingConfig,
    AveragingFilters,
    AveragingResult,
    average_result_to_csv_rows,
    compare_methods_delta,
    compute_profile_average,
    parse_observation_year_month,
)

MEAN_COLOR = "#C0392B"
METHOD_B_COLOR = "#2471A3"
BUNDLE_COLOR = "rgba(120,120,120,0.35)"
YM_BUNDLE_COLOR = "rgba(100,100,160,0.45)"


def iter_all_observations(data: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict] = []
    for month_key, block in (data.get("months") or {}).items():
        for day in block.get("days") or []:
            for obs in day.get("observations") or []:
                out.append({**obs, "date": day["date"], "_month_key": month_key})
    return out


def _month_label(m: int) -> str:
    return MONTH_NAMES_RU[m - 1] if 1 <= m <= 12 else str(m)


def _parse_excluded_ym(text: str) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for line in text.replace(",", "\n").splitlines():
        chunk = line.strip()
        if not chunk:
            continue
        for fmt in ("%Y-%m", "%m.%Y", "%Y/%m"):
            try:
                dt = datetime.strptime(chunk, fmt)
                out.add((dt.year, dt.month))
                break
            except ValueError:
                continue
    return out


def _cycle_mode_from_ui(label: str) -> str:
    mapping = {
        "00 UTC": "00",
        "12 UTC": "12",
        "00 + 12 UTC": "00+12",
        "Все имеющиеся сроки": "all",
    }
    return mapping.get(label, "00+12")


def render_averaging_sidebar(data: dict[str, Any]) -> tuple[AveragingFilters, AveragingConfig, dict[str, Any]]:
    """Сайдбар блока «Усреднение температурных профилей»; возвращает фильтры, конфиг и флаги UI."""
    all_obs = iter_all_observations(data)
    years = sorted({parse_observation_year_month(o)[0] for o in all_obs}) or [2000]
    year_start, year_end = years[0], years[-1]

    st.sidebar.markdown("### Усреднение температурных профилей")
    show_average = st.sidebar.checkbox("Показывать средний профиль", value=True)

    method_label = st.sidebar.radio(
        "Метод",
        options=[
            "Метод A: среднее по всем отдельным профилям",
            "Метод B: сначала средний профиль каждого месяца каждого года",
        ],
        index=0,
    )
    method = "B" if method_label.startswith("Метод B") else "A"

    with st.sidebar.expander("Справка по методам", expanded=False):
        st.markdown(
            "**Метод A** — «Как выглядит средний радиозондовый профиль среди всех фактически "
            "имеющихся наблюдений?» Каждый профиль равноправен; год с большим числом запусков "
            "имеет больший вес.\n\n"
            "**Метод B** — «Как выглядит средний климатический профиль выбранного календарного "
            "месяца, если каждому году дать одинаковый вес?» Сначала средний профиль каждого "
            "(год, месяц), затем среднее по годам."
        )

    coord_label = st.sidebar.radio(
        "Вертикальная координата",
        options=["Давление", "Высота AGL"],
        index=0,
        horizontal=True,
    )
    coordinate = "pressure" if coord_label.startswith("Давление") else "height"

    stat_label = st.sidebar.radio("Статистика", options=["Среднее", "Медиана"], index=0, horizontal=True)
    statistic = "median" if stat_label == "Медиана" else "mean"

    range_mode = None
    if st.sidebar.checkbox("Показывать диапазон 25–75 %", value=True):
        range_mode = "q25_q75"
    elif st.sidebar.checkbox("Показывать ±1 σ", value=False):
        range_mode = "std1"

    st.sidebar.markdown("#### Месяцы")
    mc1, mc2, mc3 = st.sidebar.columns(3)
    selected_months: set[int] = set()
    if mc1.button("Все месяцы"):
        st.session_state["avg_months"] = list(range(1, 13))
    if mc2.button("DJF"):
        st.session_state["avg_months"] = list(SEASON_MONTHS["DJF"])
    if mc3.button("MAM"):
        st.session_state["avg_months"] = list(SEASON_MONTHS["MAM"])
    mc4, mc5, mc6 = st.sidebar.columns(3)
    if mc4.button("JJA"):
        st.session_state["avg_months"] = list(SEASON_MONTHS["JJA"])
    if mc5.button("SON"):
        st.session_state["avg_months"] = list(SEASON_MONTHS["SON"])
    if mc6.button("Очистить"):
        st.session_state["avg_months"] = []

    default_months = st.session_state.get("avg_months", [3])
    selected_months_list = st.sidebar.multiselect(
        "Календарные месяцы",
        options=list(range(1, 13)),
        default=default_months,
        format_func=_month_label,
    )
    st.session_state["avg_months"] = selected_months_list
    selected_months = set(selected_months_list) or {3}

    yr_range: tuple[int, int]
    if year_start >= year_end:
        yr_range = (year_start, year_end)
        st.sidebar.caption(f"Год: {year_start}")
    else:
        yr_range = st.sidebar.slider(
            "Диапазон лет",
            min_value=year_start,
            max_value=year_end,
            value=(year_start, year_end),
        )
    year_options = [y for y in years if yr_range[0] <= y <= yr_range[1]]
    selected_years_list = st.sidebar.multiselect(
        "Конкретные годы (пусто = все в диапазоне)",
        options=year_options,
        default=[],
    )
    selected_years = frozenset(selected_years_list) if selected_years_list else None

    st.sidebar.markdown("#### Исключения (year, month)")
    excl_text = st.sidebar.text_area(
        "Исключить пары год-месяц (по одной на строку, YYYY-MM)",
        value=st.session_state.get("avg_excl_text", ""),
        height=80,
    )
    st.session_state["avg_excl_text"] = excl_text
    excluded = _parse_excluded_ym(excl_text)

    # Матрица year × month (упрощённая)
    with st.sidebar.expander("Матрица включения year × month", expanded=False):
        matrix_month = st.selectbox("Месяц для матрицы", options=sorted(selected_months), format_func=_month_label)
        matrix_flags: dict[int, bool] = {}
        for y in year_options:
            key = f"ym_{y}_{matrix_month}"
            default_on = (y, matrix_month) not in excluded
            matrix_flags[y] = st.checkbox(f"{y}", value=default_on, key=key)
        for y, on in matrix_flags.items():
            if not on:
                excluded.add((y, matrix_month))
            elif (y, matrix_month) in excluded:
                excluded.discard((y, matrix_month))

    cycle_label = st.sidebar.radio(
        "Срок",
        options=["00 + 12 UTC", "00 UTC", "12 UTC", "Все имеющиеся сроки"],
        index=0,
    )
    cycle_mode = _cycle_mode_from_ui(cycle_label)

    multi_mode_label = st.sidebar.radio(
        "Несколько месяцев",
        options=["Отдельная линия для каждого месяца", "Объединить выбранные месяцы"],
        index=0,
    )
    multi_month_mode = "combined" if "Объединить" in multi_mode_label else "separate"

    compare_methods = st.sidebar.checkbox("Сравнить Method A vs Method B", value=False)
    show_bundle_a = st.sidebar.checkbox("Пучок: отдельные профили (Bundle A)", value=True)
    show_bundle_b = st.sidebar.checkbox("Пучок: year-month mean (Bundle B)", value=True)
    show_12_panel = st.sidebar.checkbox("Показать все 12 месяцев (4×3)", value=False)
    show_n_panel = st.sidebar.checkbox("Панель N на уровне", value=False)
    min_a = st.sidebar.number_input("min profiles (Method A)", min_value=1, value=10)
    min_b = st.sidebar.number_input("min year-months (Method B)", min_value=1, value=5)

    filters = AveragingFilters(
        year_start=yr_range[0],
        year_end=yr_range[1],
        selected_years=selected_years,
        selected_months=frozenset(selected_months),
        excluded_year_months=frozenset(excluded),
        cycle_mode=cycle_mode,  # type: ignore[arg-type]
    )
    config = AveragingConfig(
        method=method,  # type: ignore[arg-type]
        coordinate=coordinate,  # type: ignore[arg-type]
        statistic=statistic,  # type: ignore[arg-type]
        range_mode=range_mode,  # type: ignore[arg-type]
        multi_month_mode=multi_month_mode,  # type: ignore[arg-type]
        min_samples_a=int(min_a),
        min_samples_b=int(min_b),
    )
    ui_flags = {
        "show_average": show_average,
        "compare_methods": compare_methods,
        "show_bundle_a": show_bundle_a,
        "show_bundle_b": show_bundle_b,
        "show_12_panel": show_12_panel,
        "show_n_panel": show_n_panel,
        "cycle_label": cycle_label,
    }
    return filters, config, ui_flags


def _add_range_band(
    fig: go.Figure,
    grid: np.ndarray,
    central: np.ndarray,
    q25: np.ndarray,
    q75: np.ndarray,
    std: np.ndarray,
    range_mode: str | None,
    name: str,
) -> None:
    if range_mode == "q25_q75":
        fig.add_trace(go.Scatter(
            x=np.concatenate([q25, q75[::-1]]),
            y=np.concatenate([grid, grid[::-1]]),
            fill="toself",
            fillcolor="rgba(231,76,60,0.15)",
            line=dict(width=0),
            name=f"{name} Q25–Q75",
            showlegend=True,
            hoverinfo="skip",
        ))
    elif range_mode == "std1":
        lo = central - std
        hi = central + std
        fig.add_trace(go.Scatter(
            x=np.concatenate([lo, hi[::-1]]),
            y=np.concatenate([grid, grid[::-1]]),
            fill="toself",
            fillcolor="rgba(52,152,219,0.12)",
            line=dict(width=0),
            name=f"{name} ±1σ",
            showlegend=True,
            hoverinfo="skip",
        ))


def build_climatology_figure(
    result: AveragingResult,
    *,
    method: str,
    show_bundle_a: bool,
    show_bundle_b: bool,
    range_mode: str | None,
    station_name: str = "Aldan",
) -> go.Figure:
    fig = go.Figure()
    y_title = "Давление, гПа" if result.coordinate == "pressure" else "Высота AGL, м"
    x_title = "Температура, °C"

    for item in result.months:
        label = _month_label(item.month) if item.month else "Объединённая выборка"
        if show_bundle_a and method == "A":
            for prof in item.individual_profiles:
                fig.add_trace(go.Scatter(
                    x=prof, y=item.grid, mode="lines",
                    line=dict(width=0.8, color=BUNDLE_COLOR),
                    opacity=0.25, showlegend=False, hoverinfo="skip",
                ))
        if show_bundle_b:
            for y, m, prof in item.year_month_profiles:
                fig.add_trace(go.Scatter(
                    x=prof, y=item.grid, mode="lines",
                    line=dict(width=1.0, color=YM_BUNDLE_COLOR),
                    opacity=0.35, showlegend=False,
                    hovertemplate=f"{y}-{_month_label(m):.3s}<extra></extra>",
                ))
        _add_range_band(fig, item.grid, item.central, item.q25, item.q75, item.std, range_mode, label)
        fig.add_trace(go.Scatter(
            x=item.central,
            y=item.grid,
            mode="lines",
            name=f"{label} ({method})",
            line=dict(width=3.2, color=MEAN_COLOR if method == "A" else METHOD_B_COLOR),
        ))

    yaxis = dict(title=y_title)
    if result.coordinate == "pressure":
        yaxis["autorange"] = "reversed"
    fig.update_layout(
        title=f"{station_name} · Monthly climatological profiles",
        xaxis_title=x_title,
        yaxis=yaxis,
        height=720,
        template="plotly_white",
        legend=dict(orientation="v", yanchor="top", y=1, x=1.02),
    )
    return fig


def build_compare_figure(
    observations: list[dict],
    filters: AveragingFilters,
    config: AveragingConfig,
    month: int,
    station_name: str = "Aldan",
) -> go.Figure | None:
    cfg_a = AveragingConfig(**{**config.__dict__, "method": "A"})
    cfg_b = AveragingConfig(**{**config.__dict__, "method": "B"})
    filt = AveragingFilters(**{**filters.__dict__, "selected_months": frozenset([month])})
    ra = compute_profile_average(observations, filt, cfg_a)
    rb = compute_profile_average(observations, filt, cfg_b)
    if not ra.months or not rb.months:
        return None
    a, b = ra.months[0], rb.months[0]
    delta = a.central - b.central

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Method A vs B", "ΔT = A − B"))
    y_title = "P, гПа" if config.coordinate == "pressure" else "H, м"
    for item, color, name in ((a, MEAN_COLOR, "A"), (b, METHOD_B_COLOR, "B")):
        fig.add_trace(go.Scatter(x=item.central, y=item.grid, mode="lines", name=f"Method {name}",
                                 line=dict(width=2.5, color=color)), row=1, col=1)
    fig.add_trace(go.Scatter(x=delta, y=a.grid, mode="lines", name="ΔT",
                             line=dict(width=2, color="#16A085")), row=1, col=2)
    if config.coordinate == "pressure":
        fig.update_yaxes(autorange="reversed", title_text=y_title, row=1, col=1)
        fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_xaxes(title_text="T, °C", row=1, col=1)
    fig.update_xaxes(title_text="ΔT, °C", row=1, col=2)
    fig.update_layout(title=f"{station_name} · {_month_label(month)}", height=520, template="plotly_white")
    return fig


def build_12month_panel(
    observations: list[dict],
    filters: AveragingFilters,
    config: AveragingConfig,
) -> go.Figure:
    fig = make_subplots(rows=4, cols=3, subplot_titles=[_month_label(m) for m in range(1, 13)])
    filt_all = AveragingFilters(**{**filters.__dict__, "selected_months": frozenset(range(1, 13))})
    result = compute_profile_average(observations, filt_all, config)
    by_month = {r.month: r for r in result.months}
    t_vals: list[float] = []
    for m in range(1, 13):
        row, col = (m - 1) // 3 + 1, (m - 1) % 3 + 1
        item = by_month.get(m)
        if item is None:
            continue
        t_vals.extend(float(x) for x in item.central if not np.isnan(x))
        fig.add_trace(
            go.Scatter(x=item.central, y=item.grid, mode="lines", line=dict(width=2, color=MEAN_COLOR),
                       showlegend=False),
            row=row, col=col,
        )
    if config.coordinate == "pressure":
        fig.update_yaxes(autorange="reversed")
    if t_vals:
        t_lo, t_hi = np.percentile(t_vals, 2), np.percentile(t_vals, 98)
        fig.update_xaxes(range=[float(t_lo), float(t_hi)])
    fig.update_layout(height=900, title="12 monthly climatological profiles", template="plotly_white")
    return fig


def build_n_samples_panel(item, coordinate: str, method: str) -> go.Figure:
    n = item.n_profiles if method == "A" else item.n_year_months
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=n, y=item.grid, mode="lines+markers", name="N"))
    fig.update_layout(
        title="N на уровне",
        xaxis_title="N profiles" if method == "A" else "N year-months",
        yaxis_title="P, гПа" if coordinate == "pressure" else "H, m",
        height=400,
        template="plotly_white",
    )
    if coordinate == "pressure":
        fig.update_yaxes(autorange="reversed")
    return fig


def render_metadata_block(result: AveragingResult, item, filters: AveragingFilters, ui_flags: dict) -> None:
    st.markdown("#### Метаданные расчёта")
    excl_years = sorted({y for y, m in filters.excluded_year_months})
    st.markdown(
        f"- **Метод:** {result.method}\n"
        f"- **Месяцы:** {', '.join(_month_label(m) for m in sorted(filters.selected_months))}\n"
        f"- **Years included:** {filters.year_start}–{filters.year_end}\n"
        f"- **Years excluded (pairs):** {', '.join(f'{y}-{m:02d}' for y, m in sorted(filters.excluded_year_months)) or '—'}\n"
        f"- **Cycles:** {ui_flags.get('cycle_label', '')}\n"
        f"- **Individual profiles:** N = {item.n_original_profiles}\n"
        f"- **Year-month means:** N = {item.n_year_month_groups}\n"
        f"- **Vertical grid:** {result.metadata.get('grid_step')}\n"
        f"- **Interpolation:** {result.metadata.get('interpolation')}\n"
        f"- **Statistic:** {result.statistic}\n"
        f"- **Range:** {result.range_mode or '—'}"
    )


def render_climatology_tab(data: dict[str, Any]) -> None:
    filters, config, ui_flags = render_averaging_sidebar(data)
    if not ui_flags["show_average"]:
        st.info("Включите «Показывать средний профиль» в блоке усреднения.")
        return

    observations = iter_all_observations(data)
    station = str(data.get("station_name", "Aldan"))

    if ui_flags["show_12_panel"]:
        fig12 = build_12month_panel(observations, filters, config)
        st.plotly_chart(fig12, use_container_width=True)
        return

    result = compute_profile_average(observations, filters, config)
    if not result.months:
        st.warning("Нет данных для выбранных фильтров.")
        return

    if ui_flags["compare_methods"] and len(filters.selected_months) == 1:
        month = next(iter(filters.selected_months))
        fig_cmp = build_compare_figure(observations, filters, config, month, station)
        if fig_cmp is not None:
            st.plotly_chart(fig_cmp, use_container_width=True)

    fig = build_climatology_figure(
        result,
        method=config.method,
        show_bundle_a=ui_flags["show_bundle_a"],
        show_bundle_b=ui_flags["show_bundle_b"],
        range_mode=config.range_mode,
        station_name=station,
    )
    st.plotly_chart(fig, use_container_width=True)

    item = result.months[0]
    render_metadata_block(result, item, filters, ui_flags)

    if ui_flags["show_n_panel"]:
        st.plotly_chart(build_n_samples_panel(item, config.coordinate, config.method), use_container_width=True)

    # Export
    csv_buf = io.StringIO()
    writer = csv.DictWriter(
        csv_buf,
        fieldnames=[
            "coordinate", "mean_temperature_c", "median_temperature_c",
            "q25_temperature_c", "q75_temperature_c", "std_temperature_c",
            "n_samples", "n_year_months", "method", "month",
            "cycle_filter", "year_start", "year_end",
        ],
    )
    writer.writeheader()
    for row in average_result_to_csv_rows(item, result.metadata):
        writer.writerow(row)
    st.download_button(
        "Скачать данные среднего профиля",
        csv_buf.getvalue().encode("utf-8-sig"),
        file_name="mean_profile.csv",
        mime="text/csv",
    )

    ym_buf = io.StringIO()
    ym_writer = csv.writer(ym_buf)
    ym_writer.writerow(["year", "month"])
    for y, m in sorted(item.included_year_months):
        ym_writer.writerow([y, m])
    st.download_button(
        "Скачать selected_year_months.csv",
        ym_buf.getvalue().encode("utf-8-sig"),
        file_name="selected_year_months.csv",
        mime="text/csv",
    )
