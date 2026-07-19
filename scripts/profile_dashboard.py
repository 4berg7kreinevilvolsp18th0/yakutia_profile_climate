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
OUTLIER_RMS_THRESHOLD = 8.0  # °C — эвристика «выбивается из картины»


@st.cache_data(show_spinner="Загрузка суточных профилей…")
def load_daily(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
        t = np.asarray([np.nan if v is None else v for v in day["temperature_c"]], dtype=float)
        if len(h) < 2:
            continue
        stack.append(np.interp(grid, h, t))
    if not stack:
        return None
    return grid, np.nanmean(np.vstack(stack), axis=0)


def day_rms_to_mean(day: dict, mean_h: np.ndarray, mean_t: np.ndarray) -> float:
    h = np.asarray(day["heights_m"], dtype=float)
    t = np.asarray([np.nan if v is None else v for v in day["temperature_c"]], dtype=float)
    if len(h) < 2:
        return float("inf")
    interp = np.interp(mean_h, h, t)
    mask = ~np.isnan(interp) & ~np.isnan(mean_t)
    if mask.sum() < 3:
        return float("inf")
    return float(np.sqrt(np.nanmean((interp[mask] - mean_t[mask]) ** 2)))


def suggest_outliers(days: list[dict], enabled: set[str]) -> list[str]:
    mean = month_mean(days, enabled)
    if mean is None:
        return []
    mean_h, mean_t = mean
    scored = []
    for day in days:
        if day["date"] not in enabled:
            continue
        rms = day_rms_to_mean(day, mean_h, mean_t)
        scored.append((rms, day["date"]))
    scored.sort(reverse=True)
    return [date for rms, date in scored if rms >= OUTLIER_RMS_THRESHOLD]


def main() -> None:
    st.set_page_config(page_title="Aldan profile dashboard", layout="wide")
    st.title("Алдан — суточные профили")
    st.caption("Дополнение к PNG: выбор месяца и отключение дней, которые выбиваются из общей картины.")

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
    c1, c2, c3 = st.sidebar.columns(3)
    if c1.button("Все", use_container_width=True):
        for day in days:
            st.session_state[f"day::{month_key}::{day['date']}"] = True
    if c2.button("Сброс", use_container_width=True):
        for day in days:
            st.session_state[f"day::{month_key}::{day['date']}"] = False
    if c3.button("Выбросы", use_container_width=True, help=f"Выключить дни с RMS ≥ {OUTLIER_RMS_THRESHOLD} °C к среднему"):
        outliers = set(suggest_outliers(days, set(all_dates)))
        for day in days:
            st.session_state[f"day::{month_key}::{day['date']}"] = day["date"] not in outliers

    enabled: set[str] = set()
    with st.sidebar.expander(f"Список дней", expanded=True):
        for day in days:
            key = f"day::{month_key}::{day['date']}"
            if key not in st.session_state:
                st.session_state[key] = True
            label = f"{day['date'][8:]} · Ts={day.get('t_surface_c')}°C · n={day['n_profiles']}"
            if day.get("inversion_detected"):
                label += " · inv"
            if st.checkbox(label, key=key):
                enabled.add(day["date"])

    # обновим заголовок expander через caption
    st.sidebar.caption(f"Включено: {len(enabled)} / {len(all_dates)}")
    mean = month_mean(days, enabled)

    # Метрики
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Дней включено", f"{len(enabled)} / {len(all_dates)}")
    m2.metric("Профилей (сроков)", sum(d["n_profiles"] for d in days if d["date"] in enabled))
    m3.metric("С инверсией", sum(1 for d in days if d["date"] in enabled and d.get("inversion_detected")))
    if mean is not None:
        m4.metric("Ts среднего, °C", f"{mean[1][0]:.1f}" if not np.isnan(mean[1][0]) else "—")
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

    # Таблица выбросов относительно текущего среднего
    if mean is not None and enabled:
        rows = []
        for day in days:
            if day["date"] not in enabled:
                continue
            rms = day_rms_to_mean(day, mean[0], mean[1])
            rows.append({
                "Дата": day["date"],
                "RMS к среднему, °C": round(rms, 2),
                "Ts, °C": day.get("t_surface_c"),
                "Профилей": day["n_profiles"],
                "Инверсия": "да" if day.get("inversion_detected") else "нет",
                "Выброс?": "да" if rms >= OUTLIER_RMS_THRESHOLD else "",
            })
        rows.sort(key=lambda r: r["RMS к среднему, °C"], reverse=True)
        st.subheader("Отклонение дней от среднего")
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption(
            f"Источник: {Path(data_path).name} · порог выброса RMS ≥ {OUTLIER_RMS_THRESHOLD} °C · "
            "PNG остаются в gdex_outputs/monthly_temperature_profiles/"
        )


if __name__ == "__main__":
    main()
