from __future__ import annotations

import io
import sys
import zipfile
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gdex_bufr.profile_climate.article_figures.config import AnalysisConfig, FigureStyle, InversionConfig
from gdex_bufr.profile_climate.article_figures.data import build_profile_qc, compute_completeness, load_profiles
from gdex_bufr.profile_climate.article_figures.metrics import (
    annual_inversion_frequency,
    compute_inversion_metrics,
    compute_seasonal_climatology,
    monthly_inversion_frequency,
    pressure_level_annual_series,
)
from gdex_bufr.profile_climate.article_figures.plots import (
    plot_annual_inversion_variability,
    plot_completeness_heatmap,
    plot_monthly_inversion_frequency,
    plot_monthly_inversion_intensity,
    plot_monthly_inversion_top_height,
    plot_monthly_profile_bundle,
    plot_pressure_level_time_series,
    plot_profile_qc_summary,
    plot_seasonal_temperature_profiles,
)

st.set_page_config(page_title="Графики статьи — Алдан", layout="wide")
st.title("Графики статьи по температурным профилям Алдана")
st.caption("Единый интерфейс для QC, климатологии, инверсий и экспорта публикационных рисунков.")


@st.cache_data(show_spinner="Загрузка profiles_long.csv…")
def cached_load(raw: bytes, name: str) -> pd.DataFrame:
    return load_profiles(io.BytesIO(raw))


@st.cache_data(show_spinner="Расчёт QC профилей…")
def cached_qc(df: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    return build_profile_qc(df, config)


@st.cache_data(show_spinner="Расчёт инверсий v2…")
def cached_inversions(df: pd.DataFrame, qc: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    return compute_inversion_metrics(df, qc, config)


@st.cache_data(show_spinner="Интерполяция сезонных профилей…")
def cached_seasonal(df: pd.DataFrame, qc: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    return compute_seasonal_climatology(df, qc, config)


def fig_bytes(fig, fmt: str, dpi: int) -> bytes:
    buffer = io.BytesIO()
    kwargs = {"format": fmt, "bbox_inches": "tight"}
    if fmt == "png":
        kwargs["dpi"] = dpi
    fig.savefig(buffer, **kwargs)
    return buffer.getvalue()


uploaded = st.sidebar.file_uploader("profiles_long.csv", type=["csv"])
path_value = st.sidebar.text_input("Или путь к CSV", value="gdex_outputs/результаты-алдан/profiles_long.csv")

raw = None
source_name = ""
if uploaded is not None:
    raw = uploaded.getvalue()
    source_name = uploaded.name
elif path_value and Path(path_value).exists():
    raw = Path(path_value).read_bytes()
    source_name = Path(path_value).name

if raw is None:
    st.info("Загрузите profiles_long.csv или укажите существующий путь в боковой панели.")
    st.stop()

df_all = cached_load(raw, source_name)
stations = sorted(df_all["station_id"].dropna().astype(str).unique())
station_id = st.sidebar.selectbox("WMO станции", stations, index=stations.index("31004") if "31004" in stations else 0)
cycles_available = sorted(x for x in df_all["cycle"].unique() if x)
cycles = tuple(st.sidebar.multiselect("Сроки UTC", cycles_available, default=[x for x in ["00", "12"] if x in cycles_available]))
if not cycles:
    st.warning("Выберите хотя бы один срок UTC")
    st.stop()

strict_qc = st.sidebar.checkbox("Строгий QC нижнего уровня", value=True)
max_surface_pressure = st.sidebar.number_input("Макс. давление нижнего уровня, гПа", 930.0, 1000.0, 960.0, 1.0)
use_height_qc = st.sidebar.checkbox("Контролировать высоту нижнего уровня", value=False)
height_tolerance = st.sidebar.number_input("Допуск высоты нижнего уровня, м", 50.0, 2000.0, 250.0, 25.0, disabled=not use_height_qc)
min_delta = st.sidebar.number_input("Минимальный рост T, °C", 0.0, 3.0, 0.2, 0.1)
confirm_levels = st.sidebar.number_input("Шагов падения для подтверждения", 1, 10, 2, 1)
confirm_depth = st.sidebar.number_input("Глубина подтверждения, гПа", 5.0, 150.0, 30.0, 5.0)
min_drop = st.sidebar.number_input("Минимальное падение за шаг, °C", 0.0, 3.0, 0.2, 0.1)

analysis = AnalysisConfig(
    station_id=station_id,
    cycles=cycles,
    strict_surface_qc=strict_qc,
    max_surface_pressure_hpa=float(max_surface_pressure),
    use_surface_height_qc=bool(use_height_qc),
    max_surface_height_deviation_m=float(height_tolerance),
    inversion=InversionConfig(
        min_inversion_delta_c=float(min_delta),
        confirm_drop_levels=int(confirm_levels),
        confirm_depth_hpa=float(confirm_depth),
        min_drop_delta_c=float(min_drop),
    ),
)

df = df_all[(df_all["station_id"] == station_id) & df_all["cycle"].isin(cycles)].copy()
qc = cached_qc(df, analysis)

st.sidebar.divider()
journal_mode = st.sidebar.checkbox("Журнальный режим", value=True)
show_title = st.sidebar.checkbox("Заголовок внутри рисунка", value=False)
language = st.sidebar.selectbox("Язык рисунка", ["ru", "en"], index=0)
cmap = st.sidebar.selectbox("Палитра полноты", ["cividis", "viridis", "magma", "plasma"], index=0)
dpi = st.sidebar.select_slider("DPI PNG", [150, 300, 600], value=600)
style = FigureStyle(language=language, journal_mode=journal_mode, show_title=show_title, completeness_cmap=cmap, dpi=dpi)

summary_tab, plot_tab, batch_tab, tables_tab = st.tabs(["Обзор", "Конструктор", "Пакет статьи", "Таблицы"])

with summary_tab:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Строк уровней", f"{len(df):,}".replace(",", " "))
    c2.metric("Профилей", f"{len(qc):,}".replace(",", " "))
    c3.metric("Пригодно для статьи", f"{int(qc['eligible_article'].sum()):,}".replace(",", " "))
    c4.metric("Период", f"{df['datetime_utc'].min():%Y-%m-%d} — {df['datetime_utc'].max():%Y-%m-%d}")
    st.dataframe(qc["profile_status"].value_counts().rename_axis("status").reset_index(name="profiles"), use_container_width=True)

with plot_tab:
    plot_name = st.selectbox(
        "Тип графика",
        [
            "1. Полнота наблюдений",
            "2. Сезонные температурные профили",
            "3. Годовой ход инверсий по срокам",
            "4. Межгодовая изменчивость инверсий",
            "5. Интенсивность инверсий по месяцам",
            "6. Высота верха инверсий по месяцам",
            "7. Статусы QC",
            "8. Пучок профилей выбранного месяца",
            "9. Температура на изобарических уровнях по годам",
        ],
    )

    fig = None
    table = None
    filename = "figure"
    if plot_name.startswith("1"):
        table, matrix = compute_completeness(df, cycles=cycles)
        annotate = st.checkbox("Подписывать проблемные ячейки (<80%)", value=True)
        fig = plot_completeness_heatmap(matrix, style, annotate_below=80 if annotate else None)
        filename = "completeness_heatmap"
    elif plot_name.startswith("2"):
        table = cached_seasonal(df, qc, analysis)
        fig = plot_seasonal_temperature_profiles(table, style)
        filename = "seasonal_temperature_profiles"
    elif plot_name.startswith("3"):
        inv = cached_inversions(df, qc, analysis)
        table = monthly_inversion_frequency(inv)
        fig = plot_monthly_inversion_frequency(table, style)
        filename = "monthly_inversion_frequency"
    elif plot_name.startswith("4"):
        inv = cached_inversions(df, qc, analysis)
        year_min, year_max = int(inv["year"].min()), int(inv["year"].max())
        start, end = st.slider("Период тренда", year_min, year_max, (max(2005, year_min), min(2025, year_max)))
        window = st.slider("Окно скользящего среднего", 3, 11, 5, 2)
        table, stats = annual_inversion_frequency(inv, start_year=start, end_year=end, moving_window=window)
        fig = plot_annual_inversion_variability(table, stats, style)
        st.json(stats)
        filename = "annual_inversion_variability"
    elif plot_name.startswith("5"):
        inv = cached_inversions(df, qc, analysis)
        table = inv[inv["inversion_detected"]]
        fig = plot_monthly_inversion_intensity(inv, style)
        filename = "monthly_inversion_intensity"
    elif plot_name.startswith("6"):
        inv = cached_inversions(df, qc, analysis)
        table = inv[inv["inversion_detected"]]
        fig = plot_monthly_inversion_top_height(inv, style)
        filename = "monthly_inversion_top_height"
    elif plot_name.startswith("7"):
        table = qc
        fig = plot_profile_qc_summary(qc, style)
        filename = "profile_qc_summary"
    elif plot_name.startswith("8"):
        years = sorted(df["year"].unique())
        year = st.selectbox("Год", years, index=len(years) - 1)
        months = sorted(df.loc[df["year"] == year, "month"].unique())
        month = st.selectbox("Месяц", months)
        table = df[(df["year"] == year) & (df["month"] == month)]
        fig = plot_monthly_profile_bundle(df, style, year=int(year), month=int(month), cycles=cycles)
        filename = f"profiles_{year}_{month:02d}"
    else:
        table = pressure_level_annual_series(df, qc, analysis)
        fig = plot_pressure_level_time_series(table, style)
        filename = "pressure_level_time_series"

    st.pyplot(fig, use_container_width=False)
    png = fig_bytes(fig, "png", dpi)
    svg = fig_bytes(fig, "svg", dpi)
    c1, c2 = st.columns(2)
    c1.download_button("Скачать PNG", png, file_name=f"{filename}.png", mime="image/png")
    c2.download_button("Скачать SVG", svg, file_name=f"{filename}.svg", mime="image/svg+xml")
    if table is not None:
        st.dataframe(table.head(1000), use_container_width=True)
        st.download_button("Скачать таблицу CSV", table.to_csv(index=False).encode("utf-8-sig"), file_name=f"{filename}.csv", mime="text/csv")
    plt.close(fig)

with batch_tab:
    st.write("Собирает четыре основных рисунка статьи и вспомогательные таблицы в один ZIP.")
    if st.button("Сформировать пакет", type="primary"):
        with st.spinner("Расчёт и экспорт…"):
            inv = cached_inversions(df, qc, analysis)
            seasonal = cached_seasonal(df, qc, analysis)
            completeness_long, completeness_matrix = compute_completeness(df, cycles=cycles)
            monthly = monthly_inversion_frequency(inv)
            annual, stats = annual_inversion_frequency(inv, start_year=analysis.trend_start_year, end_year=analysis.trend_end_year, moving_window=analysis.moving_average_window)
            figures = {
                "fig01_completeness": plot_completeness_heatmap(completeness_matrix, style),
                "fig02_seasonal_profiles": plot_seasonal_temperature_profiles(seasonal, style),
                "fig03_monthly_inversions": plot_monthly_inversion_frequency(monthly, style),
                "fig04_annual_inversions": plot_annual_inversion_variability(annual, stats, style),
            }
            tables = {
                "profile_qc.csv": qc,
                "completeness.csv": completeness_long,
                "seasonal_climatology.csv": seasonal,
                "inversion_metrics_v2.csv": inv,
                "monthly_inversion_frequency.csv": monthly,
                "annual_inversion_frequency.csv": annual,
            }
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for name, fig in figures.items():
                    zf.writestr(f"figures/{name}.png", fig_bytes(fig, "png", dpi))
                    zf.writestr(f"figures/{name}.svg", fig_bytes(fig, "svg", dpi))
                    plt.close(fig)
                for name, table in tables.items():
                    zf.writestr(f"tables/{name}", table.to_csv(index=False).encode("utf-8-sig"))
            st.download_button("Скачать ZIP", buffer.getvalue(), file_name="aldan_article_figures.zip", mime="application/zip")

with tables_tab:
    st.subheader("QC профилей")
    st.dataframe(qc, use_container_width=True, height=480)
