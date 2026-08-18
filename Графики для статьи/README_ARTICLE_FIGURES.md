# Рисунки ревизии 2026 — Алдан, WMO 31004

Источник: `gdex_outputs/far_east/stations/aldan/profiles_long.csv`.  
Пригодные профили: `eligible_article` (00 и 12 UTC). Период в данных: **1999–2026**.  
Слои: **gap-v3**, без смены алгоритма. AGL = высота минус нижний уровень профиля (не «минус 680 м»).  
Старые папки `sample_output/` и `Алдан_высоты_инверсий_исправлено/` **не перезаписывались**.

Сборка:

```bash
cd "Графики для статьи"
py -3 scripts/build_revision_figures.py ^
  --input "../gdex_outputs/far_east/stations/aldan/profiles_long.csv" ^
  --output revision_2026/output ^
  --config article_figures_config.yaml
```

Две разные γ:

- **γ_sfc-P** = 100 · (T_P − T_sfc) / (H_P − H_sfc) — средний градиент от нижнего уровня зонда до изобары 850/700/500 гПа; интерполяция только внутри наблюдаемого диапазона P.
- **γ_local** = 100 · (T_{i+1} − T_i) / (z_{i+1} − z_i) — все соседние интервалы профиля, и рост, и падение T.

Повторяемость профилей:

F = 100 · N(eligible с условием) / N(все eligible).  
G+E+HE не обязаны давать 100%.

Прогон: 17 991 пригодных профилей, 19 996 валидных слоёв; QC геометрии: depth≤0 = 0, top≤base = 0.

---

## Полка: основная статья

| Файл | Формула / смысл | Поля | Знаменатель | Единицы |
|---|---|---|---|---|
| `figures/article/01_thickness/inversion_depth_vs_base_joint` | Y = depth = top_agl − base_agl | base_height_agl_m, depth_m | N слоёв (фильтр base≥0, depth>0, top>base) | м |
| `.../inversion_depth_vs_base_monthly_12panel` | то же, 12 месяцев, общие оси и шкала log10(N) | месяц | слои | м |
| `.../inversion_depth_boxplots` | распределение thickness | depth_m, month, season, type | N в группе | м |
| `.../hexbin_depth_vs_delta_t` | плотность depth × ΔT | depth_m, delta_t_c | слои | м, °C |
| `figures/article/02_gamma_sfc_P/type03_gamma_annual_cycle_850_700_500` | медиана γ_sfc-P по месяцу | gamma_sfc_850/700/500 | профили с изобарой внутри диапазона | °C/100 м |
| `.../type03_gamma_monthly_850_700_500` | 12 панелей, X=год, 3 линии 850/700/500 | year, month, median/q25/q75 | то же | °C/100 м |
| `figures/article/03_gamma_local/gamma_local_month_height_median` | медиана γ_local, месяц × высота AGL | gamma_local, z_mid_agl_m | интервалы | °C/100 м |
| `.../gamma_local_month_height_p_positive` | P(γ_local>0) | то же | интервалы | % |
| `.../gamma_local_month_height_p_negative` | P(γ_local<0) | то же | интервалы | % |
| `figures/article/04_multilayer/n_layers_monthly_percent` | доли профилей с 0/1/2/3+ слоями | n_inversion_layers | eligible | % |
| `.../heatmap_p_multilayer` | P(n≥2), год × месяц | multilayer | eligible | % |
| `figures/article/05_GEHE/GEHE_summary_three_columns` | F и медианы base/depth/ΔT по типу | flags + layers | eligible для F; слои для медиан | %, м, °C |
| `figures/updated/type01_matrix_G_E_HE_shared_scale` | F_G, F_E, F_HE, **одна** color scale | has_G/E/HE | eligible | % |
| `figures/updated/fig01_completeness_heatmap` | полнота сроков | date+cycle | expected 00+12 | % |
| `figures/updated/fig02_seasonal_temperature_profiles` | медиана T(P) + IQR | interpolated T | eligible | °C, гПа |
| `figures/updated/fig03_monthly_inversion_frequency_v2` | **v2** приземная инверсия | inversion_detected | eligible | % |
| `figures/updated/fig04_annual_inversion_variability_v2` | **v2**, Theil–Sen | то же | eligible | % |

Фильтры слоёв: `valid_layers` — base≥0, top>base, depth = top−base > 0.

---

## Полка: supplementary

`figures/article/01_thickness/inversion_depth_vs_base_month_01` … `_12` — полные joint scatter по месяцам (общие xlim/ylim/vmax).

`figures/article/03_gamma_local/`: histogram (все интервалы), сезоны, 00 vs 12, monthly box, seasonal γ(z).

`figures/article/04_multilayer/`: hist n_layers, seasonal stack, 00/12, mean layers per profile.

`figures/article/05_GEHE/heatmap_any_inversion_year_month` — любой слой v3.

`figures/article/06_extra/` (≥10):

1. hexbin base × depth  
2. hexbin γ слоя × depth  
3. scatter base × ΔT по типу  
4. violin ΔT по G/E/HE  
5. violin depth по месяцам  
6. ridgeline depth  
7. ridgeline ΔT  
8. heatmap месяц × тип (F, %)  
9. bubble год×месяц (size=F, color=median ΔT)  
10. seasonal phase depth–ΔT  
11. 2D density depth–ΔT по типам  
12. ECDF depth  
13. ECDF ΔT  
14. heatmap месяц × бин основания (F профилей)  
15. heatmap месяц × бин толщины (F профилей)

CSV: `tables/article_figures/` (в т.ч. `gamma_sfc_P.csv` с delta_T/delta_H; `gamma_counts_n_intervals.csv` — колонка **n_intervals**, не days).

---

## Полка: только QC

`figures/diagnostic/`

| Файл | Смысл |
|---|---|
| qc_base_vs_top_yeqx | scatter + линия y=x; все точки top>base |
| qc_depth_histogram | thickness |
| qc_dz_histogram | шаг по высоте соседних уровней |
| qc_embedded_gap_depth | суммарный gap внутри слоя v3 |
| qc_embedded_gap_fraction | gap / depth |
| qc_delta_t_vs_source_segments | ΔT vs (embedded_gap_count+1) |
| qc_eligible_counts_year_month | число eligible профилей |

---

## Тесты

```bash
cd "Графики для статьи"
py -3 -m pytest -c pytest.ini --rootdir=. tests/test_revision_2026.py tests/test_smoke.py -q
```

Проверяют: depth=top−base>0; формулы обеих γ; нет экстраполяции 850/700/500; F от eligible, не от числа слоёв; месяцы 1…12; общие ylim и colorbar.
