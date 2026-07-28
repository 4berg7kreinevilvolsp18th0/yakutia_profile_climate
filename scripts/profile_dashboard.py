"""Интерактивный дашборд суточных температурных профилей.

Запуск:
  py -3 -m streamlit run scripts/profile_dashboard.py

PNG-графики не заменяет: позволяет выбрать месяц, отключить дни-выбросы
и сразу увидеть пересчитанное среднее.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "gdex_outputs" / "profile_climate" / "aldan" / "daily_profiles.json"

# Критерий A: max |ΔT| между соседними уровнями по высоте
OUTLIER_MAX_ABS_DT_C = 10.0  # °C

# Критерий B: max (ΔT/ΔP)² между соседними уровнями по давлению
OUTLIER_MAX_DT_DP_SQ = 0.25  # (°C/гПа)² ≈ |ΔT/ΔP| ≥ 0.5 °C/гПа
MIN_ABS_DP_HPA = 0.5  # защита от деления на почти нулевой ΔP


@st.cache_data(show_spinner="Загрузка суточных профилей…")
def load_daily(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _temps_as_float(day: dict) -> np.ndarray:
    return np.asarray([np.nan if v is None else v for v in day["temperature_c"]], dtype=float)


def _pressures_as_float(day: dict) -> np.ndarray | None:
    pressures = day.get("pressure_hpa")
    if not pressures:
        return None
    return np.asarray([np.nan if v is None else v for v in pressures], dtype=float)


def month_mean(days: list[dict], enabled: set[str]) -> tuple[np.ndarray, np.ndarray] | None:
    active = [d for d in days if d["date"] in enabled]
    if not active:
        return None
    min_h = min(d["heights_m"][0] for d in active)
    max_h = max(d["heights_m"][-1] for d in active)
    grid = np.linspace(min_h, max_h, 40)
    stack = []
    for day in active:
        h = np.asarray(day["heights_m"], dtype=float)
        t = _temps_as_float(day)
        valid = ~np.isnan(t)
        if valid.sum() < 2:
            continue
        stack.append(np.interp(grid, h[valid], t[valid], left=np.nan, right=np.nan))
    if not stack:
        return None
    return grid, np.nanmean(np.vstack(stack), axis=0)


def day_max_abs_dt(day: dict) -> float:
    """Максимальный |ΔT| между соседними валидными уровнями по высоте."""
    h = np.asarray(day["heights_m"], dtype=float)
    t = _temps_as_float(day)
    order = np.argsort(h)
    h = h[order]
    t = t[order]
    valid = ~np.isnan(t)
    if valid.sum() < 2:
        return float("inf")
    t = t[valid]
    return float(np.max(np.abs(np.diff(t))))


def day_max_dt_dp_sq(day: dict) -> float:
    """Максимум (ΔT/ΔP)² между соседними уровнями по давлению.

    Уровни сортируются по убыванию давления (земля → верх).
    Пары с |ΔP| < MIN_ABS_DP_HPA пропускаются.
    """
    t = _temps_as_float(day)
    p = _pressures_as_float(day)
    if p is None:
        return float("inf")
    valid = ~np.isnan(t) & ~np.isnan(p)
    if valid.sum() < 2:
        return float("inf")
    t = t[valid]
    p = p[valid]
    order = np.argsort(-p)  # высокое давление сначала
    t = t[order]
    p = p[order]

    scores: list[float] = []
    for i in range(len(t) - 1):
        dp = abs(float(p[i + 1] - p[i]))
        if dp < MIN_ABS_DP_HPA:
            continue
        dt = float(t[i + 1] - t[i])
        scores.append((dt / dp) ** 2)
    if not scores:
        return float("inf")
    return float(max(scores))


def suggest_outliers_abs_dt(days: list[dict], enabled: set[str]) -> list[str]:
    scored = []
    for day in days:
        if day["date"] not in enabled:
            continue
        max_dt = day_max_abs_dt(day)
        scored.append((max_dt, day["date"]))
    scored.sort(reverse=True)
    return [date for max_dt, date in scored if max_dt >= OUTLIER_MAX_ABS_DT_C]


def suggest_outliers_dt_dp_sq(days: list[dict], enabled: set[str]) -> list[str]:
    scored = []
    for day in days:
        if day["date"] not in enabled:
            continue
        score = day_max_dt_dp_sq(day)
        scored.append((score, day["date"]))
    scored.sort(reverse=True)
    return [date for score, date in scored if score >= OUTLIER_MAX_DT_DP_SQ]


def _first_valid_temp(temps: np.ndarray) -> float | None:
    for value in temps:
        if not np.isnan(value):
            return float(value)
    return None


def _finite_or_none(value: float) -> float | None:
    if value == float("inf") or value != value:  # noqa: PLR0124 — NaN check
        return None
    return round(value, 4)


def main() -> None:
    st.set_page_config(page_title="Aldan profile dashboard", layout="wide")
    st.title("Алдан — суточные профили")
    st.caption(
        "Два критерия выбросов для сравнения: max |ΔT| и max (ΔT/ΔP)². "
        "Кнопки только предлагают отключение дней — можно править вручную."
    )

    data_path = st.sidebar.text_input("daily_profiles.json", str(DEFAULT_DATA))
    if not Path(data_path).exists():
        st.error(
            "Нет файла данных. Сначала выполните:\n\n"
            "`py -3 scripts/build_daily_profiles.py`"
        )
        return

    data = load_daily(data_path)
    months = sorted(data["months"].keys())
    if not months:
        st.error("В JSON нет месяцев.")
        return

    has_pressure = any(
        bool(day.get("pressure_hpa"))
        for month in data["months"].values()
        for day in month.get("days", [])
    )
    if not has_pressure:
        st.warning(
            "В JSON нет pressure_hpa. Пересоберите данные:\n\n"
            "`python scripts/build_daily_profiles.py`\n\n"
            "Пока кнопка (ΔT/ΔP)² будет недоступна."
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
    all_dates = [d["date"] for d in days]

    st.sidebar.markdown("### Дни")
    c1, c2 = st.sidebar.columns(2)
    if c1.button("Все", use_container_width=True):
        for day in days:
            st.session_state[f"day::{month_key}::{day['date']}"] = True
    if c2.button("Сброс", use_container_width=True):
        for day in days:
            st.session_state[f"day::{month_key}::{day['date']}"] = False

    st.sidebar.markdown("### Выбросы (сравнить)")
    o1, o2 = st.sidebar.columns(2)
    if o1.button(
        "по |ΔT|",
        use_container_width=True,
        help=f"Выключить дни с max |ΔT| ≥ {OUTLIER_MAX_ABS_DT_C} °C",
    ):
        outliers = set(suggest_outliers_abs_dt(days, set(all_dates)))
        for day in days:
            st.session_state[f"day::{month_key}::{day['date']}"] = day["date"] not in outliers
    if o2.button(
        "по (ΔT/ΔP)²",
        use_container_width=True,
        disabled=not has_pressure,
        help=f"Выключить дни с max (ΔT/ΔP)² ≥ {OUTLIER_MAX_DT_DP_SQ} (°C/гПа)²",
    ):
        outliers = set(suggest_outliers_dt_dp_sq(days, set(all_dates)))
        for day in days:
            st.session_state[f"day::{month_key}::{day['date']}"] = day["date"] not in outliers

    st.sidebar.caption(
        f"|ΔT| ≥ {OUTLIER_MAX_ABS_DT_C} °C · "
        f"(ΔT/ΔP)² ≥ {OUTLIER_MAX_DT_DP_SQ} · min |ΔP|={MIN_ABS_DP_HPA} гПа"
    )

    enabled: set[str] = set()
    with st.sidebar.expander("Список дней", expanded=True):
        for day in days:
            key = f"day::{month_key}::{day['date']}"
            if key not in st.session_state:
                st.session_state[key] = True
            label = f"{day['date'][8:]} · Ts={day.get('t_surface_c')}°C · n={day['n_profiles']}"
            if day.get("inversion_detected"):
                label += " · inv"
            if st.checkbox(label, key=key):
                enabled.add(day["date"])

    st.sidebar.caption(f"Включено: {len(enabled)} / {len(all_dates)}")
    mean = month_mean(days, enabled)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Дней включено", f"{len(enabled)} / {len(all_dates)}")
    m2.metric("Профилей (сроков)", sum(d["n_profiles"] for d in days if d["date"] in enabled))
    m3.metric("С инверсией", sum(1 for d in days if d["date"] in enabled and d.get("inversion_detected")))
    if mean is not None:
        ts = _first_valid_temp(mean[1])
        m4.metric("Ts среднего, °C", f"{ts:.1f}" if ts is not None else "—")
    else:
        m4.metric("Ts среднего, °C", "—")

    fig = go.Figure()
    palette = [
        "#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B",
        "#EECA3B", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC",
    ]
    for idx, day in enumerate(days):
        if day["date"] not in enabled:
            continue
        fig.add_trace(go.Scatter(
            x=day["temperature_c"],
            y=day["heights_m"],
            mode="lines",
            name=day["date"][8:],
            line=dict(width=1.5, color=palette[idx % len(palette)]),
            opacity=0.85,
            connectgaps=False,
            hovertemplate=(
                f"{day['date']}<br>"
                "T=%{x:.1f} °C<br>"
                "h=%{y:.0f} м<extra></extra>"
            ),
        ))

    if mean is not None:
        fig.add_trace(go.Scatter(
            x=mean[1],
            y=mean[0],
            mode="lines",
            name="Среднее (включённые)",
            line=dict(width=3.5, color="#C44E52"),
            connectgaps=False,
            hovertemplate="Среднее<br>T=%{x:.1f} °C<br>h=%{y:.0f} м<extra></extra>",
        ))

    fig.update_layout(
        title=f"{data.get('station_name', 'Aldan')} — {month_key} (суточные средние)",
        xaxis_title="Температура, °C",
        yaxis_title="Высота, м",
        height=720,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        margin=dict(l=40, r=20, t=60, b=40),
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    if enabled:
        rows = []
        n_flag_dt = 0
        n_flag_grad = 0
        n_both = 0
        for day in days:
            if day["date"] not in enabled:
                continue
            max_dt = day_max_abs_dt(day)
            grad_sq = day_max_dt_dp_sq(day) if has_pressure else float("inf")
            flag_dt = max_dt >= OUTLIER_MAX_ABS_DT_C
            flag_grad = has_pressure and grad_sq >= OUTLIER_MAX_DT_DP_SQ
            if flag_dt:
                n_flag_dt += 1
            if flag_grad:
                n_flag_grad += 1
            if flag_dt and flag_grad:
                n_both += 1
            rows.append({
                "Дата": day["date"],
                "max |ΔT|, °C": _finite_or_none(max_dt),
                "max (ΔT/ΔP)²": None if not has_pressure else _finite_or_none(grad_sq),
                "Выброс |ΔT|?": "да" if flag_dt else "",
                "Выброс (ΔT/ΔP)²?": "да" if flag_grad else "",
                "Ts, °C": day.get("t_surface_c"),
                "Профилей": day["n_profiles"],
                "Инверсия": "да" if day.get("inversion_detected") else "нет",
            })
        rows.sort(
            key=lambda r: (
                r["max (ΔT/ΔP)²"] if r["max (ΔT/ΔP)²"] is not None else -1,
                r["max |ΔT|, °C"] if r["max |ΔT|, °C"] is not None else -1,
            ),
            reverse=True,
        )
        st.subheader("Сравнение двух критериев выбросов")
        c_a, c_b, c_c = st.columns(3)
        c_a.metric("Флаг |ΔT|", n_flag_dt)
        c_b.metric("Флаг (ΔT/ΔP)²", n_flag_grad if has_pressure else "—")
        c_c.metric("Оба сразу", n_both if has_pressure else "—")
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption(
            f"Источник: {Path(data_path).name} · "
            f"|ΔT| ≥ {OUTLIER_MAX_ABS_DT_C} °C · "
            f"(ΔT/ΔP)² ≥ {OUTLIER_MAX_DT_DP_SQ} (°C/гПа)² · "
            f"min |ΔP| = {MIN_ABS_DP_HPA} гПа"
        )


if __name__ == "__main__":
    main()
