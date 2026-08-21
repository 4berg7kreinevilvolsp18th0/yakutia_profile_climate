# README_GRAPHICS_REVISED

Ревизия графиков статьи строится из единого набора таблиц `revision_2026/output/tables/article_figures`.

## Source table

- `inversion_layers_valid.csv` — валидные слои v3, включая `base_height_agl_m`, `top_height_agl_m`, `depth_m`.
- `profile_type_flags.csv` — флаги G/E/HE по профилю.
- `gamma_sfc_P.csv` — \(\gamma_{sfc-P}\) для 850/700/500 гПа, без экстраполяции.
- `gamma_local_intervals.csv` — локальные интервальные градиенты \(\gamma_{local}\).
- `article_graphics_master_layers.csv`, `article_graphics_master_profiles.csv` — канонические master tables.

## Формулы

- `depth_m = top_height_agl_m - base_height_agl_m`
- `gamma_layer = 100 * delta_t_c / depth_m`
- `gamma_local = 100 * (T_{i+1} - T_i) / (z_{i+1} - z_i)`
- `gamma_sfc_P = 100 * (T_P - T_sfc) / (H_P - H_sfc)`

## Denominator / frequency

Для частот используется профильный знаменатель внутри группы:

- \(F_{00} = 100 \cdot N_{hit,00} / N_{valid,00}\)
- \(F_{12} = 100 \cdot N_{hit,12} / N_{valid,12}\)

## Cycles

Основные графики сравнения сроков используют только `00` и `12` UTC.  
Цветовой стандарт централизован в `revision_2026/article_colors.py`.

## N / filters / clipping

- На ключевых графиках подписываются `N layers`, `N00`, `N12` (или `N` по уровню давления).
- Геометрический QC: `depth_m > 0`, `top_height_agl_m > base_height_agl_m`.
- Аудит по источникам/фильтрам/статистикам:  
  `revision_2026/output/audit/GRAPHICS_DATA_AUDIT.csv` и `.md`
