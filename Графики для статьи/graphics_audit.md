# Аудит научных графиков для статьи (Алдан, WMO 31004)

Дата аудита: 2026-08-18.  
Исходный pipeline: `Графики для статьи/gdex_bufr/profile_climate/article_figures/`  
Существующие выходы **не удалялись**: `sample_output/`, `Алдан_высоты_инверсий_исправлено/`.

## 1. Каталог существующих рисунков

Источник уровней: `gdex_outputs/far_east/stations/aldan/profiles_long.csv`.  
Пригодные профили: `profile_qc.eligible_article` (статус `good` + опциональный строгий QC поверхности).

| Рисунок | Данные | Формула / знаменатель | Единицы | Замечания |
|---|---|---|---|---|
| fig01 completeness | уникальные date+cycle | observed / expected×100 | % | Корректно; крайние месяцы по окну наблюдений |
| fig02 seasonal T | интерполяция T(P) без экстраполяции | медиана и IQR по сезону | °C, гПа | Корректно |
| fig03 monthly inversion frequency | **v2** `inversion_detected` | N(confirmed v2) / N(eligible) | % | Это **приземная v2**, не G/E/HE v3 |
| fig04 annual variability | v2, годы 2005–2025 | то же; Theil–Sen | % | Не смешивать с v3 |
| extra01 intensity | v2 `inversion_delta_t_c` | по слоям/профилям v2 | °C | |
| extra02 top height | v2 `inversion_top_height_m` | **ASL** (`height_m` верха) | м | Легко спутать с AGL |
| extra03 QC summary | profile_qc | абсолютные счётчики | — | |
| extra04 T at 925/850/700/500 | интерп. T(P) | медиана T по году | °C | Это температура, не γ |
| 00 height QC old vs new | layers pressure-order vs gap-v3 | счётчики depth≤0, top≤base | — | Диагностика |
| 01–03 recurrence G/E/HE | слои v3 по бинам | **N_слоёв в бине / N_eligible** ×100 | % | См. проблему P1 |
| 04 base vs top QC | v3 AGL | scatter | м AGL | |
| 05–06 monthly median IQR | v3 top/base AGL | медиана слоёв | м AGL | Знаменатель — слои, не профили |
| 07 heatmap top height G/E/HE | медиана top AGL | год×месяц | м | Шкалы **разные** по типу (намеренно после фикса G) |
| 08 CDF 00/12 | top AGL | ECDF слоёв | м | |
| 09 annual median top | слои | медиана | м | |
| 10 seasonal quantiles | слои | | м | |
| type01 matrix G/E/HE | `profile_type_flags` | N(профилей с ≥1 слоем типа) / N(eligible) | % | **Корректный** знаменатель профилей |
| type02 height_* | счётчики слоёв по бинам top AGL | N_layers | шт. | Не проценты повторяемости |
| type03 gamma_* | **все соседние интервалы** профиля | γ=100·ΔT/Δz | °C/100 м | После фикса — и + и −; колонка CSV всё ещё называется `days` |
| type03_gamma_monthly_box | те же интервалы | boxplot по месяцам | °C/100 м | Корректно по знаку |

## 2. Найденные проблемы

### P1. «Повторяемость» по бинам высоты (01–03)

`recurrence_percent_table` считает `height_count_table` (число **слоёв** в бине), делит на число пригодных **профилей**, но колонку называет `profiles`.  
Если у одного профиля два слоя в одном бине, процент завышается. Для «повторяемости профилей» нужен unique `profile_id`.

### P2. Две методики инверсии на соседних рисунках

fig03/fig04/extra01/extra02 — **legacy v2** (один приземный слой + confirm-drop).  
type01 и высоты — **gap-v3** (все слои G/E/HE).  
Это не ошибка алгоритма, но в статье нужно явно подписывать метод.

### P3. AGL ≠ «высота минус 680 м»

В gap-v3 `z0` = нижний уровень профиля после сортировки по `height_m`, не отметка станции.  
Конфиг: `station_elevation_m = 679` (Алдан ≈ 680 м н.у.м.). Поле AGL — относительно **поверхности зонда**.

### P4. extra02 в ASL

`inversion_top_height_m` — абсолютная высота уровня, не AGL.

### P5. type03: подпись «дни» vs интервалы

В CSV `gamma_counts_all.days` — это число **интервалов**, не календарных дней. На графиках ось уже «число интервалов», таблица — нет.

### P6. Сравниваемые heatmap G/E/HE

Матрицы повторяемости type01 имеют разные максимумы (G чаще HE). Общая color scale нужна для сравнения панелей; отдельные шкалы — только если это оговорено.

### P7. Нет толщины как Y

type02 и 03_depth recurrence используют бины `depth_m` как «высоту» на той же шкале, что top/base. Нет joint scatter base AGL × thickness.

### P8. Нет среднего γ до 850/700/500

extra04 — температура на изобарах. Локальный γ интервалов ≠ средний градиент слоя станция→P.

### P9. Стиль

Шрифт DejaVu Sans, не serif; часть рисунков с внутренним заголовком (`show_title` в старых сборках).

### P10. Геометрия слоёв (уже чинилось)

Старый pressure-order давал `depth_m ≤ 0`. Текущий height-primary gap-v3 отбрасывает такие слои (`layer_geometry_qc`). Не менять алгоритм детекции.

## 3. Что не трогать

- Алгоритм gap-v3 и пороги G≤30 м / E≤250 м / HE>250 м AGL основания.
- Существующие PNG/SVG в `sample_output` и `Алдан_высоты_инверсий_исправлено`.
- Знаменатель type01 (профили, не слои).

## 4. Куда складывается ревизия

Новые и обновлённые рисунки — отдельно, для сравнения:

- `revision_2026/output/figures/updated/` — restyle существующих
- `revision_2026/output/figures/article/` — новые научные
- `revision_2026/output/figures/diagnostic/` — QC
- `revision_2026/output/tables/article_figures/` — CSV

## 5. Факты после прогона (1999–2026, 00/12 UTC)

- Пригодных профилей: **17 991**
- Валидных слоёв v3: **19 996**
- Геометрия: negative_depth = **0**, top_below_base = **0**, negative_base_agl = **0**
- Повторяемость профилей (знаменатель 17 991): G **36.9%**, E **3.0%**, HE **53.0%** (сумма >100% из‑за нескольких типов на профиле; HE: 12 823 слоя / 9 530 профилей)

P1 исправлен в updated/extra: биновые проценты = unique profile_id / eligible.  
P5: таблица `gamma_counts_n_intervals.csv`.  
P6: `type01_matrix_G_E_HE_shared_scale`.  
γ_sfc-P и γ_local разведены по папкам `02_gamma_sfc_P` и `03_gamma_local`.
