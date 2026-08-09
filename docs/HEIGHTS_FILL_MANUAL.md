# Высоты уровней: полная инструкция по коду

Подробное описание того, **как в проекте появляется высота уровня** `height_m` — от сырого
BUFR-сообщения до колонок в CSV/XLSX/JSON и до оси «высота, м» на графиках.

Смежные документы:
- [HEIGHT_ARCHITECTURE_GUIDE.md](HEIGHT_ARCHITECTURE_GUIDE.md) — обзорная карта проекта (для доклада);
- [METHODS_HEIGHT_INVERSION.md](METHODS_HEIGHT_INVERSION.md) — формулы и методика для статьи.

Здесь — «инженерная» версия: какая функция, в каком файле, в каком порядке, что записывает
и какие есть подводные камни.

---

## 0. Версия на 30 секунд

Высота уровня заполняется **каскадом приоритетов**, и на каждом уровне фиксируется, откуда
она взялась (`height_source`):

```
1. Прямая высота из BUFR      0-10-009, иначе 0-07-007        → height_bufr_m
2. Геопотенциал Φ → z         0-10-008, формула MetPy         → height_phi_m
3. Высота станции (для SFC)   0-07-001 / справочник (679 м)   → station_007001
4. Линейная интерполяция H(P) между известными уровнями зонда → height_interp_m
5. Барометрия от станции      z_st + 44330(1 − (P/P_sfc)^0.1903) → height_baro_m
```

Первый непустой вариант сверху вниз становится `height_m`. Экстраполяции нет: если уровень
выше/ниже всех известных якорей — сразу шаг 5.

---

## 1. Термины и дескрипторы BUFR

| FXY | Константа в коде | Поле объекта / колонка | Смысл |
|-----|------------------|------------------------|-------|
| 0-07-004 | `DESC_PRESSURE` | `pressure_hpa`, `PRES` | Давление уровня (в BUFR — **Па**) |
| 0-10-009 | `DESC_HEIGHT` | `height_010009_m` | Геопотенциальная высота уровня, м |
| 0-07-007 | `DESC_HEIGHT_COORD` | `height_007007_m` | Высота как координата, м (резервный прямой источник) |
| 0-10-008 | `DESC_GEOPOT` | `geopotential_m2s2`, `GEOPOT` | Геопотенциал Φ, м²/с² |
| 0-07-001 | `DESC_STATION_HEIGHT` | `station_elevation_m` | Высота станции н.у.м. (Алдан ≈ 679 м) |
| 0-08-001 | — | `VSIG`, `vertical_significance_code` | Тип уровня: SFC / MANL / SIGT / … |

Обозначения формул:
- \(z_{\mathrm{st}}\) — высота станции н.у.м. (679 м для 31004);
- \(P_{\mathrm{sfc}}\) — давление на поверхности станции (не константа, обычно 920–945 гПа);
- MSL — над уровнем моря, AGL — над уровнем станции (`height_agl_m = height_msl_m − z_st`).

---

## 2. Карта конвейера: кто и когда трогает высоту

| # | Этап | Файл | Функция | Результат |
|---|------|------|---------|-----------|
| 1 | Декод BUFR | `gdex_bufr/bufr_adapter.py` | `_record_height`, `_flush_record` | `height_010009_m`, `height_007007_m`, `height_phi_m`, `geopotential_m2s2`, `geopotential_height_m` |
| 2 | Обогащение профиля | `gdex_bufr/meteo_parser_bridge.py` | `enrich_profile_levels` → `enrich_vertical_level` | Дозаполняет `geopotential_height_m` (SFC → Φ → баро) |
| 3a | Рабочий профиль | `gdex_bufr/profile_climate/extract.py` | `_thermo_levels`, `_pick_station_surface`, `_level_climate_fields` | Уровни до 500 гПа, отсечение секций ниже станции |
| 3b | Полный дамп | `gdex_bufr/profile_climate/extract.py` | `extract_decoded_levels` → `_resolve_height_msl` | `height_msl_m` + `height_msl_source`, `below_station` |
| 4 | Дозаполнение дыр | `gdex_bufr/profile_climate/height_fill.py` | `fill_profile_level_heights` → `_pick_observed_height`, `_choose_final_height` | `height_obs_m`, `height_interp_m`, `height_baro_m`, `height_m`, `height_source` |
| 5 | Тот же каскад по таблице | `gdex_bufr/profile_climate/height_fill.py` | `fill_long_dataframe_heights` | То же, но векторно для DataFrame `profiles_long` |
| 6 | Потребители | `metrics.py`/`inversion.py`, `plots.py`, `obs_qc.py`, `build_daily_profiles.py` | — | `inversion_top_height_m`, ось Y, `heights_m` в JSON |

Порядок вызова в основном прогоне: `process_profile` (extract.py:363) делает 3a → 4 → метрики → 3b.

---

## 3. Этап 1. Декодирование BUFR

Файл: `gdex_bufr/bufr_adapter.py`.

### 3.1. Сборка уровня

ADPUPA-сообщение — плоский поток дескрипторов. Уровень «режется» по 0-08-001: встретили новый
0-08-001 → предыдущая запись сбрасывается функцией `_flush_record`. Уровень без давления
(нет 0-07-004) или с `P > 1100` гПа отбрасывается целиком.

### 3.2. Нормализация давления (без неё высоты «поедут»)

```248:273:gdex_bufr/bufr_adapter.py
def _normalize_pressure(value: Any, registry: BufrTablesRegistry | None = None) -> float | None:
```

0-07-004 в BUFR — **паскали**. По таблице дескрипторов определяется unit и делится на 100.
Если таблицы нет, применяется запасная эвристика `P > 1100 → /100`. Это критично: без деления
уровень 1000 Па (10 гПа, стратосфера) стал бы 1000 гПа (поверхность), и вся привязка H(P)
развалилась бы.

### 3.3. Три кандидата высоты

```368:406:gdex_bufr/bufr_adapter.py
    def _record_height() -> tuple[
```

На каждом уровне:

1. `geopotential_m2s2` = 0-10-008 (если не missing);
2. `height_phi_m` = `geopotential_to_height_m(Φ)`, округление до 0.1 м;
3. `height_010009_m` = 0-10-009;
4. `height_007007_m` = 0-07-007;
5. **рабочая** `geopotential_height_m` = первое непустое из `height_010009_m` → `height_007007_m` → `height_phi_m`.

Ключевое правило: **то, что записано в сообщении как высота, важнее пересчёта из Φ.** Пересчёт
Φ→z всегда сохраняется отдельно в `height_phi_m` — чтобы можно было сравнить и объяснить расхождение.

### 3.4. Высота станции

Читается 0-07-001 на уровне сообщения в `_decode_subset_header` (`bufr_adapter.py:704–716`) и
кладётся в `RadiosondeProfile.station_elevation_m`, а также в метаданные профиля
(`station_elevation_m` + `station_height_fxy`). Если её в файле нет — дальше используется
справочник `STATION_ELEVATION_M` (см. §6.1).

### 3.5. Не-ADPUPA режим

Для `decode_mode != "adpupa"` уровни собираются по параллельным сериям, и там
`geopotential_height_m = height_010009_m` (0-10-009), без Φ. Это запасной режим, для
основного конвейера не используется.

---

## 4. Этап 2. Обогащение профиля

Файл: `gdex_bufr/meteo_parser_bridge.py`, функция `enrich_vertical_level` (строка 118).

Вызывается пакетно из `enrich_profile_levels` после декода. Если `geopotential_height_m`
всё ещё `None`:

```
0) VSIG == "SFC" и известна высота станции  → height = z_st
1) есть Φ                                    → height = Φ → z
2) есть P и P_sfc                            → height = z_st + 44330(1 − (P/P_sfc)^0.1903)
```

Как выбирается `P_sfc` — функция `_resolve_surface_pressure` (`meteo_parser_bridge.py:168`),
которую вызывает `enrich_profile_levels` (строка 210): если есть уровни с VSIG=SFC и известна
высота станции, берётся SFC с **минимальным `seq`** (первый по шаблону, обычно sig-секция),
иначе — просто `max(P)` по профилю.

В метаданные профиля пишутся флаги: `height_from_geopotential`, `height_from_pressure_or_station`,
`station_elevation_from_bufr` — по ним видно, что именно применялось.

---

## 5. Этап 3. Климатический extract

Файл: `gdex_bufr/profile_climate/extract.py`.

### 5.1. Какой SFC считать «станцией» — `_pick_station_surface` (строка 98)

В ADPUPA часто **несколько секций**, и после первой может начаться повторная manl-секция
с P ≈ 992 гПа и H ≈ 68 м — это физически ниже станции Алдан (679 м). Алгоритм выбора:

1. Опорная высота `elev` = 0-07-001 из BUFR → справочник по станции → 679 м как последний фолбэк;
2. берём все уровни с VSIG = SFC;
3. если SFC нет — просто уровень с максимальным P;
4. среди SFC с известной высотой — **тот, чья высота ближе к `elev`**;
5. если ни у одного SFC нет высоты — SFC с минимальным `seq`.

### 5.2. Обрезка рабочего профиля — `_thermo_levels` (строка 51)

- оставляем уровни с `P` и `T`;
- отбрасываем всё, где `P > P_surface + 2` гПа (то есть «ниже станции»);
- если у выбранной поверхности не было T — она всё равно добавляется явно;
- дедуп по давлению (с точностью 0.1 гПа): при равном P выигрывает уровень с температурой,
  затем SFC;
- сортировка по убыванию P (земля → верх).

Затем `extract_temperature_levels` режет диапазон `500 ≤ P ≤ 1000` гПа.

### 5.3. Поля уровня — `_level_climate_fields` (строка 127)

Пишутся сразу все варианты высоты, чтобы ничего не терялось:

| Ключ | Что содержит |
|------|--------------|
| `height_010009_m`, `height_007007_m` | сырые прямые высоты |
| `height_bufr_m` | 0-10-009, иначе 0-07-007 |
| `height_phi_m` | только Φ→z |
| `FLVL` = `geopotential_height_m` = `height_m` | рабочая высота после этапов 1–2 |

### 5.4. Полный дамп — `extract_decoded_levels` (строка 260)

Здесь считается **своя** высота `height_msl_m` со своей меткой источника — она нужна, чтобы
показать *все* уровни, включая забракованные. Сам приоритет вынесен в `_resolve_height_msl`
(строка 210), а два его шага — в `_direct_bufr_height_m` (194) и `_phi_height_m` (201):

| Порядок | Условие | `height_msl_source` |
|---------|---------|---------------------|
| 1 | есть `height_bufr_m` | `direct_bufr` |
| 2 | есть `height_phi_m` | `phi` |
| 3 | уровень ниже поверхности (`P > P_sfc + 2`) и известны `z_st`, `P_sfc` | `baro_below_station` |
| 4 | есть `height_decoded_m` (после enrich) | `enriched` |
| 5 | это выбранный SFC и известна `z_st` | `station_007001` |

Дополнительно:
- `height_agl_m = height_msl_m − station_elevation_m`;
- `below_station = (P > P_sfc + 2) OR (H < z_st − 100)`;
- `in_working_profile` — попал ли уровень в рабочий профиль (не below_station, есть T, 500 ≤ P ≤ 1000);
- `qc_flag = "below_station"` для брака.

Пример из теста `tests/test_actual_dual_tables.py`: уровень 992 гПа с H = 68 м остаётся в
`decoded_levels` с `height_agl_m = −612`, `below_station = True`, но в рабочую таблицу не попадает.

---

## 6. Этап 4. `fill_profile_level_heights` — главный каскад

Файл: `gdex_bufr/profile_climate/height_fill.py`, строка 163.
Вызов: `process_profile` (`extract.py:394`) сразу после `extract_temperature_levels` и **до** метрик,
чтобы инверсия видела уже окончательную высоту.

Сигнатура:

```python
fill_profile_level_heights(
    levels,                              # список dict-уровней рабочего профиля
    surface_pressure_hpa=...,            # P выбранной поверхности станции
    station_id="31004",
    station_elevation_override_m=...,    # 0-07-001 из BUFR, если есть
)
```

### 6.1. Шаг 0 — высота станции

```16:30:gdex_bufr/profile_climate/height_fill.py
STATION_ELEVATION_M: dict[str, float] = {
```

`elev` = `station_elevation_override_m` → `STATION_ELEVATION_M[station_id]` → `None`.
Ключ нормализуется как `str(station_id).zfill(5)[-5:]`. Сейчас в справочнике две станции:
31004 (Алдан, 679 м) и 24959 (Якутск, 103 м).

### 6.2. Шаг 1 — «наблюдённая» высота `height_obs_m`

Функция `_pick_observed_height` (строка 116) для каждого уровня по порядку:

1. `height_010009_m`, иначе `height_007007_m` → источник `level`;
2. иначе `geopotential_m2s2` → `Φ → z`, источник `phi`;
3. иначе `height_m` / `geopotential_height_m` (то, что уже посчитано на этапах 1–2)
   → источник `observed_or_geopot`;
4. иначе, если `VSIG == "SFC"` и известна `elev` → `height = elev`, источник `station_007001`.

Любое значение проходит через `_finite`: `None`, нечисловое и `NaN` считаются отсутствующими.

### 6.3. Шаг 2 — интерполяция `height_interp_m`

`interpolate_heights_on_pressure(pressures, obs_heights)` (строка 77):

- якоря — только уровни с известной `height_obs_m` (включая SFC-высоту станции из шага 1);
- 0 якорей → все `None`; 1 якорь → заполнен только сам якорь, экстраполяции нет;
- давления сортируются по возрастанию, дубликаты P схлопываются **средним H**;
- `np.interp(..., left=nan, right=nan)` — **строго внутри** диапазона якорей;
- результат округляется до 0.1 м.

### 6.4. Шаг 3 — барометрия `height_baro_m`

```62:74:gdex_bufr/profile_climate/height_fill.py
def barometric_height_m(
```

\[
z = z_{\mathrm{st}} + 44330\left(1 - \left(\frac{P}{P_{\mathrm{sfc}}}\right)^{0.1903}\right)
\]

`P_sfc` = переданное `surface_pressure_hpa`, иначе `max(P)` по профилю. Если `elev` неизвестна,
базой служит 0 (тогда высота получается «над поверхностью», а не н.у.м.).

### 6.5. Шаг 4 — итоговая высота

Выбор делает `_choose_final_height` (строка 147), запись колонок — тело цикла:

```209:218:gdex_bufr/profile_climate/height_fill.py
        final_h, final_source = _choose_final_height(h_obs, h_source, h_interp, h_baro)
```

```
height_m = height_obs_m           → height_source = level | phi | observed_or_geopot | station_007001
         иначе height_interp_m    → height_source = "interp"
         иначе height_baro_m      → height_source = "baro"
         иначе None               → height_source = None

height_msl_m = height_m
height_agl_m = height_m − elev    (None, если высота или elev неизвестны)
```

Все три кандидата (`height_obs_m`, `height_interp_m`, `height_baro_m`) сохраняются рядом
даже когда не выбраны — это позволяет потом проверить расхождение методов.

---

## 7. Этап 5. `fill_long_dataframe_heights` — векторная версия

Файл: тот же, строка 223. Используется, когда высоты чинят **по готовой таблице**, а не по
объектам профиля: `scripts/build_daily_profiles.py:401` и `scripts/repair_heights_xlsx.py:55`.

Логика та же (obs → interp → baro), но есть отличия, которые важно знать:

| Аспект | `fill_profile_level_heights` | `fill_long_dataframe_heights` |
|--------|------------------------------|-------------------------------|
| Вход | список dict одного профиля | DataFrame со всеми профилями + `profile_metrics` |
| Источник obs | `height_010009_m` → `007007` → Φ → `height_m` | `height_m` → `geopotential_height_m` → Φ |
| Метка obs-источника | `level` / `phi` / `observed_or_geopot` / `station_007001` | всегда `observed_or_geopot` |
| Фолбэк «SFC = высота станции» | есть | **нет** (SFC отработал раньше, в extract) |
| Отсечение брака | нет | `height_obs < −50 м → NaN` (например, Φ<0 у поверхности), строка 269 |
| Φ→z | через MetPy | аналитическая формула на numpy (без вызова MetPy построчно) |
| `P_sfc` | аргумент | `p_surface_hpa` из метрик, иначе `max(P)` профиля |
| `z_st` | аргумент/справочник | `station_elevation_m` из метрик, иначе справочник |

Группировка по `profile_id` сделана через `argsort` + границы блоков — интерполяция считается
внутри каждого зонда отдельно (высоты **никогда** не интерполируются между разными сроками).

На выходе перезаписываются колонки: `height_obs_m`, `height_interp_m`, `height_baro_m`,
`height_m`, `height_msl_m`, `height_agl_m`, `height_source`.

---

## 8. Формулы с числами

### 8.1. Φ → z (MetPy)

\[
z = \frac{\Phi R_e}{g_0 R_e - \Phi},\qquad g_0 = 9.80665,\ R_e = 6\,371\,229\ \text{м}
\]

Реализация: `geopotential_to_height_m` (`meteo_parser_bridge.py:77`). Сначала пробуется
`metpy.calc.geopotential_to_height`; если MetPy недоступен — та же аналитическая формула.

| Φ, м²/с² | Φ/g₀ («наивно»), м | z по формуле, м | разница |
|----------|--------------------|-----------------|---------|
| 9 805 | 999.8 | 999.99 | +0.2 м |
| 29 406 | 2 998.6 | 2 999.99 | +1.4 м |
| 150 000 | 15 295.7 | ≈ 15 332 | ≈ +36 м |

Именно поэтому «делить на 9.8» нельзя: у стратосферных уровней ошибка десятки метров.

### 8.2. Барометрия

Алдан, `z_st = 679`, `P_sfc = 930` гПа:

| P, гПа | Δz, м | z, м н.у.м. |
|--------|-------|-------------|
| 930 | 0 | 679 |
| 850 | ≈ 752 | ≈ 1 431 |
| 700 | ≈ 2 333 | ≈ 3 012 |
| 500 | ≈ 4 938 | ≈ 5 617 |

При `P = P_sfc` прирост ровно 0 → высота станции (это проверяет `test_baro_uses_station_elevation`).

### 8.3. Интерполяция

Якоря `(900 гПа, 1000 м)` и `(700 гПа, 3000 м)`, целевой уровень 850 гПа:

\[
H = 3000 + \frac{850-700}{900-700}(1000-3000) = 1500\ \text{м}
\]

Уровень 600 гПа лежит **вне** диапазона якорей → `height_interp_m = None` → уходит в барометрию
(проверяется в `test_fill_prefers_obs_then_interp_then_baro`).

---

## 9. Альтернативная реализация: `scripts/aldan_simple_pipeline.py`

Упрощённый однофайловый конвейер со своей функцией `fill_heights` (строка 347). Логика та же,
но набор меток отличается и добавлен QC:

| Приоритет | Условие | `height_source` |
|-----------|---------|-----------------|
| 1 | `height_bufr_m ≥ −50` | `level` |
| 2 | `height_phi_m ≥ −50` | `phi` |
| 3 | уровень помечен как предпочтительный SFC | `station_007001` |
| 4 | интерполяция между известными | `interp` |
| 5 | иначе | `baro` |

QC-флаги строки: `height_negative` (итог < −50 м) и `surface_height_far_from_station`
(поверхность отличается от 679 м более чем на 250 м). Барометрия здесь считается **всегда**,
независимо от того, выбрана она или нет.

Проверяется тестами `tests/test_simple_sounding.py` (`height_source == "level"` / `"phi"`).

---

## 10. Починка уже собранных файлов

`scripts/repair_heights_xlsx.py` — пересчёт высот в готовом XLSX без повторного декода BUFR:

```bash
python scripts/repair_heights_xlsx.py --xlsx "gdex_outputs/результаты-алдан"
```

Что делает: читает листы `profiles_long` + `profile_metrics`, прогоняет
`fill_long_dataframe_heights`, дописывает недостающие колонки из `PROFILES_LONG_COLUMNS`,
сохраняет `<имя>_heights_fixed.xlsx` и печатает JSON-отчёт: сколько строк, сколько пустых высот
до/после, min/max, число `height < 0` и `> 20000`, распределение `height_source`.

Пример отчёта — файл `gdex_outputs/результаты-алдан/aldan_profile_climate_*_heights_fixed.xlsx`.

---

## 11. Где какие колонки лежат

### `profiles_long` (рабочий профиль, `export.PROFILES_LONG_COLUMNS`)

| Колонка | Смысл |
|---------|-------|
| `pressure_hpa` / `PRES` | давление уровня, гПа |
| `geopotential_m2s2` / `GEOPOT` | сырой геопотенциал |
| `height_010009_m`, `height_007007_m` | сырые прямые высоты из BUFR |
| `height_bufr_m` | прямая высота (10009 → 07007) |
| `height_phi_m` | высота только из Φ |
| `FLVL` / `geopotential_height_m` | рабочая высота после декода+enrich |
| **`height_m`** | **итоговая высота для анализа и графиков** |
| `height_msl_m` / `height_agl_m` | над уровнем моря / над станцией |
| `height_obs_m`, `height_interp_m`, `height_baro_m` | кандидаты каждого метода |
| `height_source` | какой метод победил |

### `decoded_levels` (все уровни, `export.DECODED_LEVEL_BASE_COLUMNS`)

Дополнительно: `height_decoded_m`, `height_msl_source`, `station_elevation_m`, `below_station`,
`in_working_profile`, `qc_flag`.

### `profile_metrics`

`station_elevation_m`, `p_surface_hpa`, `inversion_top_height_m` (берётся как `height_m`
уровня верха инверсии — `inversion.py:52`).

### `daily_profiles.json` (схема `observations_v1`)

Массивы `heights_m`, `heights_interp_m`, `heights_baro_m` на каждое наблюдение
(`build_daily_profiles._obs_arrays`).

---

## 12. Кто потребляет высоту дальше

- **Инверсия** — `metrics.py` → `inversion.detect_surface_inversion`; поле
  `inversion_top_height_m` = `height_m` уровня верха. Ошибка в высоте не меняет факт
  обнаружения (он по T и P), но меняет отчётную высоту верха.
- **Графики** — `plots.py`, `article_figures/plots.py`: ось Y в метрах строится по `height_m`.
- **QC кривых** — `obs_qc.py`: `dedupe_levels_by_height` (один уровень на 10 м, при дубле
  оставляется уровень с бо́льшим P), `_enforce_increasing_height_with_falling_pressure`
  (убирает петли H(P)), `remove_temperature_spikes` (скачок T > 10 °C на < 200 м).
  Это фильтры **отображения**, они не меняют исходные таблицы.
- **Дашборд** — `scripts/profile_dashboard.py` читает готовые массивы из JSON и **не**
  пересчитывает высоты.

---

## 13. Крайние случаи и типичные ловушки

| Ситуация | Что происходит | Где смотреть |
|----------|----------------|--------------|
| В зонде нет ни одной известной высоты | Все уровни → `baro`, всё зависит от `z_st` и `P_sfc` | `height_source` полностью `baro` |
| Известен ровно один уровень | Интерполяции нет вообще; остальные → `baro` | `interpolate_heights_on_pressure`, ветка `known.sum() == 1` |
| Уровень выше самого верхнего якоря | Экстраполяции нет → `baro` | `left=nan, right=nan` в `np.interp` |
| Несколько уровней с одинаковым P | H усредняется перед интерполяцией | `uniq_p / uniq_h` |
| Φ < 0 у поверхности | В DF-версии отсекается порогом `< −50 м`; в per-profile версии — нет | `height_fill.py:269` |
| Повторная manl-секция ниже станции | Отбрасывается из рабочего профиля, помечается `below_station` | `_thermo_levels`, `extract_decoded_levels` |
| Давление пришло в Па | Делится на 100 по unit дескриптора | `_normalize_pressure` |
| Станции нет в `STATION_ELEVATION_M` и нет 0-07-001 | `elev = None` → `height_agl_m = None`, барометрия даёт высоту **над поверхностью** | `station_elevation_m()` |
| `height_m` немонотонна по P | Возможны «петли» на оси высоты; чистится только для отображения | `obs_qc._enforce_increasing_height_with_falling_pressure` |
| Старые XLSX собраны прежней версией | Нужен `repair_heights_xlsx.py` или повторный extract | §10 |

---

## 14. Как проверить, что высоты корректны

```bash
# тесты каскада и формул
python -m pytest tests/test_height_fill.py tests/test_geopotential_height.py -q

# тесты рабочей/полной таблицы и упрощённого конвейера
python -m pytest tests/test_actual_dual_tables.py tests/test_simple_sounding.py -q
```

Быстрая ручная проверка по готовой таблице:

1. Распределение источников — `value_counts()` по `height_source`. Если почти всё `baro`,
   значит прямые высоты не читаются (смотреть декод 0-10-009 / 0-10-008).
2. На поверхности `height_agl_m ≈ 0` (для Алдана `height_m ≈ 679`).
3. `height_m` не отрицательна и не превышает ~20 000 м (этот срез уже считает `repair_heights_xlsx`).
4. `height_m` монотонно растёт при падении P внутри одного `profile_id`.
5. Сравнить `height_obs_m` и `height_baro_m` там, где есть оба: систематический сдвиг
   больше ~200 м у поверхности означает неверный `P_sfc` или неверный выбор SFC.

---

## 15. Как менять поведение

- **Добавить станцию**: `STATION_ELEVATION_M` в `height_fill.py` (плюс `elevation_m` в
  `profile_climate_config.yaml`). Ключ — 5-значный WMO ID.
- **Запретить барометрию**: убрать ветку `elif h_baro is not None` в `fill_profile_level_heights`
  — тогда уровни без якорей останутся с `height_m = None` (графики по давлению продолжат работать).
- **Разрешить экстраполяцию**: заменить `left/right=np.nan` в `interpolate_heights_on_pressure`
  на краевые значения. Не рекомендуется: за пределами якорей линейная H(P) физически неверна.
- **Поменять порог отсечения брака**: константа `−50.0` в `height_fill.py:269`.
- **Изменить верх анализа** (сейчас 500 гПа): параметр `pressure_top_hpa` в `process_profile`
  и `profile_climate_config.yaml`.

---

## 16. Шпаргалка «где что»

| Задача | Файл → функция |
|--------|----------------|
| Прочитать высоту из BUFR | `gdex_bufr/bufr_adapter.py` → `_record_height` |
| Па → гПа | `bufr_adapter.py` → `_normalize_pressure` |
| Φ → метры | `meteo_parser_bridge.py` → `geopotential_to_height_m` |
| Барометрическая оценка | `meteo_parser_bridge.py` → `estimate_geopotential_height_m`; `height_fill.py` → `barometric_height_m` |
| Дозаполнить профиль | `height_fill.py` → `fill_profile_level_heights` |
| Дозаполнить таблицу | `height_fill.py` → `fill_long_dataframe_heights` |
| Выбрать поверхность станции | `extract.py` → `_pick_station_surface` |
| MSL/AGL и метка источника | `extract.py` → `extract_decoded_levels` |
| Починить готовый XLSX | `scripts/repair_heights_xlsx.py` |
| Ось высоты на графиках | `profile_climate/plots.py` |
| Чистка петель H(P) | `profile_climate/obs_qc.py`, `plot_filter.py` |
| Тесты | `tests/test_height_fill.py`, `tests/test_geopotential_height.py`, `tests/test_actual_dual_tables.py` |
