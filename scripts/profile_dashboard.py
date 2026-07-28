"""Интерактивный дашборд температурных профилей (наблюдения / сроки).

Запуск:
  py -3 -m streamlit run scripts/profile_dashboard.py

Кнопки QC только предлагают отключение наблюдений — можно править вручную.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gdex_bufr.profile_climate.obs_qc import (  # noqa: E402
    MAD_K,
    MAD_OUTLIER_FRACTION,
    MIN_ABS_DP_HPA,
    MIN_LEVELS_FLAG,
    OUTLIER_MAX_ABS_DT_C,
    OUTLIER_MAX_DT_DP_SQ,
    is_few_levels,
    mad_outlier_fraction,
    max_abs_dt,
    max_dt_dp_sq,
    month_median_mad,
    suggest_outliers_abs_dt,
    suggest_outliers_dt_dp_sq,
    suggest_outliers_few_levels,
    suggest_outliers_mad,
)

DEFAULT_DATA = ROOT / "gdex_outputs" / "profile_climate" / "aldan" / "daily_profiles.json"
REQUIRED_SCHEMA = "observations_v1"

# Качественная палитра ~24 цвета (без доминирующего фиолетового)
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


def _temps_as_float(obs: dict) -> np.ndarray:
    return np.asarray(
        [np.nan if v is None else v for v in obs["temperature_c"]],
        dtype=float,
    )


def _vertical_as_float(obs: dict, axis: str) -> np.ndarray | None:
    if axis == "pressure":
        pressures = obs.get("pressure_hpa")
        if not pressures:
            return None
        return np.asarray([np.nan if v is None else v for v in pressures], dtype=float)
    return np.asarray(obs["heights_m"], dtype=float)


def month_mean(
    observations: list[dict],
    enabled: set[str],
    *,
    y_axis: str = "height",
) -> tuple[np.ndarray, np.ndarray] | None:
    """Средний профиль включённых наблюдений на сетке по высоте или давлению."""
    series: list[tuple[np.ndarray, np.ndarray]] = []
    for obs in observations:
        if obs["profile_id"] not in enabled:
            continue
        y = _vertical_as_float(obs, y_axis)
        if y is None:
            continue
        t = _temps_as_float(obs)
        valid = ~np.isnan(t) & ~np.isnan(y)
        if valid.sum() < 2:
            continue
        series.append((y[valid], t[valid]))
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


def main() -> None:
    st.set_page_config(page_title="Aldan profile dashboard", layout="wide")
    st.title("Алдан — профили наблюдений")
    st.caption(
        "Одна кривая = один зонд (срок). QC-кнопки предлагают кандидатов на отключение; "
        "среднее пересчитывается по включённым наблюдениям."
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

    show_day_means = st.sidebar.checkbox(
        "Показать суточные средние",
        value=False,
        help="Серые линии day_mean для дней с ≥1 включённым наблюдением.",
    )

    y_axis_label = st.sidebar.radio(
        "Вертикальная ось",
        options=["Давление, гПа", "Высота, м"],
        index=0,
        help="Давление надёжнее, если геопотенциальная высота сомнительна.",
    )
    y_axis = "pressure" if y_axis_label.startswith("Давление") else "height"

    days = data["months"][month_key]["days"]
    observations = _iter_observations(days)
    all_ids = [o["profile_id"] for o in observations]

    st.sidebar.markdown("### Наблюдения")
    c1, c2 = st.sidebar.columns(2)
    if c1.button("Все", use_container_width=True):
        for obs in observations:
            st.session_state[_obs_state_key(month_key, obs["profile_id"])] = True
    if c2.button("Сброс", use_container_width=True):
        for obs in observations:
            st.session_state[_obs_state_key(month_key, obs["profile_id"])] = False

    st.sidebar.markdown("### Выбросы (кандидаты)")
    q1, q2 = st.sidebar.columns(2)
    q3, q4 = st.sidebar.columns(2)

    def _apply_outliers(outlier_ids: set[str]) -> None:
        for obs in observations:
            pid = obs["profile_id"]
            st.session_state[_obs_state_key(month_key, pid)] = pid not in outlier_ids

    if q1.button(
        "по MAD",
        use_container_width=True,
        help=f"Доля уровней |T−med| > {MAD_K}·MAD ≥ {MAD_OUTLIER_FRACTION}",
    ):
        _apply_outliers(set(suggest_outliers_mad(observations, set(all_ids))))
    if q2.button(
        "по |ΔT|",
        use_container_width=True,
        help=f"max |ΔT| ≥ {OUTLIER_MAX_ABS_DT_C} °C (по давлению)",
    ):
        _apply_outliers(set(suggest_outliers_abs_dt(observations, set(all_ids))))
    if q3.button(
        "по (ΔT/ΔP)²",
        use_container_width=True,
        help=f"max (ΔT/ΔP)² ≥ {OUTLIER_MAX_DT_DP_SQ}",
    ):
        _apply_outliers(set(suggest_outliers_dt_dp_sq(observations, set(all_ids))))
    if q4.button(
        "мало уровней",
        use_container_width=True,
        help=f"n_levels < {MIN_LEVELS_FLAG}",
    ):
        _apply_outliers(set(suggest_outliers_few_levels(observations, set(all_ids))))

    st.sidebar.caption(
        f"MAD k={MAD_K} frac≥{MAD_OUTLIER_FRACTION} · "
        f"|ΔT|≥{OUTLIER_MAX_ABS_DT_C}°C · "
        f"(ΔT/ΔP)²≥{OUTLIER_MAX_DT_DP_SQ} · "
        f"n<{MIN_LEVELS_FLAG} · min|ΔP|={MIN_ABS_DP_HPA} гПа"
    )

    enabled: set[str] = set()
    with st.sidebar.expander("Список наблюдений", expanded=True):
        for day in days:
            st.markdown(f"**{day['date'][8:]}** · n={day.get('n_profiles', 0)}")
            for obs in day.get("observations") or []:
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

    st.sidebar.caption(f"Включено: {len(enabled)} / {len(all_ids)}")
    mean = month_mean(observations, enabled, y_axis=y_axis)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Наблюдений", f"{len(enabled)} / {len(all_ids)}")
    m2.metric(
        "Дней с данными",
        len({o["date"] for o in observations if o["profile_id"] in enabled}),
    )
    m3.metric(
        "С инверсией",
        sum(1 for o in observations if o["profile_id"] in enabled and o.get("inversion_detected")),
    )
    if mean is not None:
        ts = _first_valid_temp(mean[1])
        m4.metric("Ts среднего, °C", f"{ts:.1f}" if ts is not None else "—")
    else:
        m4.metric("Ts среднего, °C", "—")

    fig = go.Figure()
    y_hover = "P=%{y:.1f} гПа" if y_axis == "pressure" else "h=%{y:.0f} м"
    color_idx = 0

    for day in days:
        day_has_enabled = False
        for obs in day.get("observations") or []:
            if obs["profile_id"] not in enabled:
                continue
            day_has_enabled = True
            y_vals = obs.get("pressure_hpa") if y_axis == "pressure" else obs["heights_m"]
            if y_vals is None:
                continue
            color = OBS_PALETTE[color_idx % len(OBS_PALETTE)]
            color_idx += 1
            name = f"{day['date'][8:]}·{obs.get('cycle', '??')}"
            fig.add_trace(go.Scatter(
                x=obs["temperature_c"],
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
                    f"{obs.get('datetime_utc', day['date'])}<br>"
                    f"CY{obs.get('cycle', '??')}<br>"
                    f"T=%{{x:.1f}} °C<br>"
                    f"{y_hover}<extra></extra>"
                ),
            ))

        if show_day_means and day_has_enabled and day.get("day_mean"):
            dm = day["day_mean"]
            y_vals = dm.get("pressure_hpa") if y_axis == "pressure" else dm.get("heights_m")
            if y_vals:
                fig.add_trace(go.Scatter(
                    x=dm["temperature_c"],
                    y=y_vals,
                    mode="lines",
                    name=f"{day['date'][8:]} mean",
                    line=dict(width=2.0, color=DAY_MEAN_COLOR, dash="dot"),
                    opacity=0.55,
                    connectgaps=False,
                    hovertemplate=(
                        f"Суточное среднее {day['date']}<br>"
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
        mad_stats = month_median_mad([o for o in observations if o["profile_id"] in enabled])
        rows = []
        n_flag_dt = n_flag_grad = n_flag_mad = n_flag_few = 0
        for obs in observations:
            if obs["profile_id"] not in enabled:
                continue
            max_dt = max_abs_dt(obs)
            grad_sq = max_dt_dp_sq(obs)
            few = is_few_levels(obs)
            mad_frac = None
            if mad_stats is not None:
                grid, median, mad = mad_stats
                mad_frac = mad_outlier_fraction(obs, median, mad, grid)
            flag_dt = max_dt >= OUTLIER_MAX_ABS_DT_C
            flag_grad = grad_sq >= OUTLIER_MAX_DT_DP_SQ
            flag_mad = mad_frac is not None and mad_frac >= MAD_OUTLIER_FRACTION
            if flag_dt:
                n_flag_dt += 1
            if flag_grad:
                n_flag_grad += 1
            if flag_mad:
                n_flag_mad += 1
            if few:
                n_flag_few += 1
            rows.append({
                "Дата": obs["date"],
                "Cycle": obs.get("cycle"),
                "profile_id": obs["profile_id"],
                "max |ΔT|, °C": _finite_or_none(max_dt),
                "max (ΔT/ΔP)²": _finite_or_none(grad_sq),
                "MAD доля": None if mad_frac is None else round(mad_frac, 3),
                "Уровней": obs.get("n_levels"),
                "Выброс |ΔT|?": "да" if flag_dt else "",
                "Выброс (ΔT/ΔP)²?": "да" if flag_grad else "",
                "Выброс MAD?": "да" if flag_mad else "",
                "Мало уровней?": "да" if few else "",
                "Ts, °C": obs.get("t_surface_c"),
                "Инверсия": "да" if obs.get("inversion_detected") else "нет",
            })
        rows.sort(
            key=lambda r: (
                r["MAD доля"] if r["MAD доля"] is not None else -1,
                r["max (ΔT/ΔP)²"] if r["max (ΔT/ΔP)²"] is not None else -1,
                r["max |ΔT|, °C"] if r["max |ΔT|, °C"] is not None else -1,
            ),
            reverse=True,
        )
        st.subheader("Сравнение критериев выбросов (наблюдения)")
        c_a, c_b, c_c, c_d = st.columns(4)
        c_a.metric("Флаг MAD", n_flag_mad)
        c_b.metric("Флаг |ΔT|", n_flag_dt)
        c_c.metric("Флаг (ΔT/ΔP)²", n_flag_grad)
        c_d.metric("Мало уровней", n_flag_few)
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption(
            f"Источник: {Path(data_path).name} · schema={REQUIRED_SCHEMA} · "
            f"MAD k={MAD_K} frac≥{MAD_OUTLIER_FRACTION} · "
            f"|ΔT|≥{OUTLIER_MAX_ABS_DT_C}°C · "
            f"(ΔT/ΔP)²≥{OUTLIER_MAX_DT_DP_SQ} · "
            f"n<{MIN_LEVELS_FLAG}"
        )


if __name__ == "__main__":
    main()
