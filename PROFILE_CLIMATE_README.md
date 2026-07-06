# Profile Climate — анализ температурных профилей Якутии

## Цель

Анализ приземного климата через вертикальные температурные профили радиозондирования для станций:

- **Алдан** (WMO **31004**)
- **Якутск** (WMO **24959**)

Период: **1999–2026**.

Источник: **GDAS ADPUPA BUFR** из GDEX/RDA [d351000](https://data.rda.ucar.edu/d351000/bufr).

## Что делает v1

- Используется только **температура**.
- Профиль строится от нижнего доступного уровня до **500 гПа** (изобарическая поверхность, не метры).
- График: `temperature_c` по `pressure_hpa`, ось Y перевёрнута (внизу высокое давление).
- Для каждого месяца и станции — один PNG с пучком профилей (~30–40 линий при cycles `00,12`).
- Экспорт long-format уровней и метрик профиля в CSV/XLSX/JSON.

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## 1. Подготовка BUFR-таблиц

```bash
python -m gdex_bufr tables-update
```

## 2. Загрузка данных

Датасет d351000 начинается с **1999-10-01** (раньше файлов на сервере нет).

### Только Алдан (удобный скрипт)

BUFR-файлы GDEX глобальные (все станции в одном файле), поэтому «скачать только
Алдан» с сервера нельзя. По умолчанию скрипт работает **потоково**: скачивает
файл → извлекает Алдан (31004) → **удаляет BUFR**. Диск не забивается — остаются
только CSV/XLSX Алдана.

```bash
python scripts/download_aldan.py --start-date 1999-10-01 --end-date 2026-12-31
```

Выход: `gdex_outputs/profile_climate/aldan/`.

Классический режим (скачать и **хранить** все глобальные BUFR на диске):

```bash
python scripts/download_aldan.py --keep-bufr --start-date 1999-10-01 --end-date 2026-12-31 --daemon
```

Отдельные шаги (только для `--keep-bufr`):

```bash
python scripts/download_aldan.py --manifest-only --start-date 1999-10-01 --end-date 2026-12-31
python scripts/download_aldan.py --download-only --daemon
python scripts/download_aldan.py --extract-only --start-date 1999-10-01 --end-date 2026-12-31
```

### Вручную (все станции в BUFR)

```bash
python -m gdex_bufr manifest --start-date 1999-10-01 --end-date 2026-12-31
python -m gdex_bufr download --daemon
```

Для тестового месяца:

```bash
python -m gdex_bufr manifest --start-date 1999-01-01 --end-date 1999-01-31
python -m gdex_bufr download --limit-files 62
```

Файлы сохраняются в `gdex_data/raw/`.

## 3. Верификация станций

```bash
python -m gdex_bufr discover-stations \
  --start-date 1999-01-01 --end-date 1999-01-31 \
  --include-all-files
```

Команда выводит `station_id`, координаты и число профилей. Сверьте **31004** (Алдан) и **24959** (Якутск).

## 4. Извлечение профилей и метрик

```bash
python -m gdex_bufr station-profiles \
  --station 31004 \
  --start-date 1999-01-01 \
  --end-date 2026-12-31 \
  --pressure-top 500 \
  --cycles 00,12 \
  --output gdex_outputs/profile_climate
```

Параметры можно задать в [`profile_climate_config.yaml`](profile_climate_config.yaml).

### Выходные файлы

| Файл | Описание |
|------|----------|
| `gdex_outputs/profile_climate/profiles_long.csv` | Уровни профилей (long format) |
| `gdex_outputs/profile_climate/profile_metrics.csv` | Метрики каждого профиля |
| `gdex_outputs/profile_climate/monthly_summary.csv` | Сводка по месяцам |
| `gdex_outputs/profile_climate/station_summary.csv` | Сводка по станциям |
| `gdex_outputs/profile_climate/summary.json` | Общая сводка |
| `gdex_outputs/profile_climate/profile_climate.xlsx` | XLSX (если установлен pandas) |

## 5. Месячные графики

```bash
python -m gdex_bufr monthly-profile-plots \
  --station aldan \
  --start-date 1999-10-01 \
  --end-date 1999-10-31 \
  --input gdex_outputs/profile_climate/aldan/profiles_long.csv \
  --metrics gdex_outputs/profile_climate/aldan/profile_metrics.csv \
  --output gdex_outputs/monthly_temperature_profiles
```

На графике:
- ось Y — **высота, м** (не давление);
- каждый профиль — **свой цвет** и подпись в легенде (дата + срок);
- уровни с **P > 1000 гПа** отбрасываются;
- бракованные профили не рисуются (`duplicate_levels`, `no_temp`, `bad_pressure`, …);
- скачки температуры на малой высоте фильтруются.

Настройки в [`profile_climate_config.yaml`](profile_climate_config.yaml):
- `max_surface_pressure_hpa: 1000` — верхняя граница давления у поверхности;
- `plot_only_good: false` — если `true`, только статус `good`;
- `plot_min_levels: 3` — минимум уровней на профиль.

Пример пути PNG:

```text
gdex_outputs/monthly_temperature_profiles/aldan/1999/aldan_1999_01_temperature_profiles_to_500hpa.png
```

## Интерпретация `profile_metrics.csv`

| Поле | Смысл |
|------|-------|
| `n_levels_to_500` | Число уровней от поверхности до 500 гПа |
| `p_surface_hpa`, `t_surface_c` | Нижний доступный уровень |
| `p_top_hpa`, `t_top_c` | Верх анализа (≈500 гПа) |
| `delta_t_top_surface_c` | T_top − T_surface |
| `inversion_detected` | Найдена приземная инверсия |
| `inversion_top_pressure_hpa` | Давление верха инверсии |
| `inversion_delta_t_c` | T_inversion_top − T_surface |
| `profile_status` | `good`, `short`, `no_500`, `no_temp`, … |

## Алгоритм инверсии (v1)

1. Уровни от нижнего до 500 гПа, сортировка по давлению (земля → верх).
2. Идём вверх, пока температура растёт между соседними уровнями.
3. Верх инверсии — последний уровень с ростом T.
4. Порог шума: `min_inversion_delta_c` (по умолчанию **0.2 °C**).
5. Если роста нет → `inversion_detected = false`.

## Статусы профиля

- `good` — пригоден для анализа
- `short` — мало уровней (`< min_levels_to_500`)
- `no_500` — профиль не доходит до 500 гПа
- `no_temp` — нет температуры
- `bad_pressure` — проблемы с давлением
- `duplicate_levels` — дубли уровней
- `no_surface_level` — нет нижнего уровня

## Ограничения v1

- Только температура; влажность, ветер, Skew-T и hodograph не входят в основной анализ.
- Верхний диапазон — до 500 гПа (давление, не высота).
- Инверсия — простой алгоритм по соседним уровням.
- Высота инверсии в метрах — только если есть в BUFR.

## Дальнейшее развитие

- Влажность, ветер, высота инверсии в метрах
- Seasonal climatology и тренды
- Сравнение Алдан / Якутск

## Тесты

```bash
pytest -c gdex_pytest.ini tests/test_profile_climate_*.py
```

Быстрый smoke-тест без реальных BUFR (синтетические профили Алдана, январь 1999):

```bash
python scripts/run_smoke_test.py
```
