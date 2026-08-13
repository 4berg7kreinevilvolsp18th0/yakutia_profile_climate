# Автономный контур Алдана

`scripts/старое/aldan_simple_pipeline.py` — отдельный понятный путь обработки WMO 31004:

```text
BUFR → SFC/MANL/TXPR → поверхность→500 гПа → высоты → инверсия → CSV/графики/дашборд
```

Старые `fast_extract`, `результаты-алдан` и основной дашборд не изменяются. Все новые
продукты находятся в `gdex_outputs/алдан-simple/`.

## Уровни

- `SFC`: поверхность (`1`/`64`);
- `MANL`: обязательный уровень (`2`/`32`);
- `TXPR`: единая выходная метка для исходных `SIGT`, `TROP` и `TXPR`.

Повторные SFC нормальны для секционного шаблона ADPUPA. Поверхностью станции считается
SFC с высотой, ближайшей к `0-07-001`. Поэтому ложный SFC manl-секции около 992 гПа
и 40–70 м не начинает профиль Алдана.

## Высоты

`profiles_long.csv` сохраняет исходные данные и расчёт отдельно:

- `height_bufr_m` — высота непосредственно из BUFR (`0-10-009`/`0-07-007`);
- `geopotential_m2s2` — геопотенциал Φ (`0-10-008`);
- `height_phi_m` — результат Φ→z;
- `station_elevation_m` — `0-07-001`, fallback для Алдана 679 м;
- `height_m` — рабочая высота для графиков и инверсии;
- `height_source` — `level`, `phi`, `station_007001`, `interp` или `baro`.

`height_bufr_m` никогда не заменяется расчётом. Для `height_m` применяется приоритет:
BUFR → Φ→z → высота станции на SFC → интерполяция → барометрическая оценка.

## Инверсия

Алгоритм v2 сначала ищет непрерывный рост температуры от поверхности, затем требует
два последовательных шага падения температуры и слой подтверждения не менее 30 гПа.
В `profile_metrics.csv` записываются:

- `inversion_detected`, `inversion_quality`;
- `inversion_top_pressure_hpa`;
- `inversion_top_height_m`;
- `inversion_top_temp_c`, `inversion_delta_t_c`.

## Запуск

```powershell
cd "B:\Kutunika programmist\yakutia_profile_climate"

# Проверочный зонд
python scripts/старое/aldan_simple_pipeline.py --date 2000-09-14 --cycle 12

# Весь архив на машине с данными
python scripts/старое/aldan_simple_pipeline.py --all --fresh

# Только месячные графики из готового CSV
python scripts/старое/aldan_simple_pipeline.py --plots

# Отдельный дашборд, читающий только алдан-simple
python scripts/старое/aldan_simple_pipeline.py --dashboard

# Decode всего архива, графики, затем дашборд
python scripts/старое/aldan_simple_pipeline.py --all --fresh --plots --dashboard
```

Пути можно переопределить через `--bufr-root` и `--output`. Константы `BUFR_ROOT`,
`OUT_DIR`, `PLOTS_DIR` и готовая `DASHBOARD_CMD` находятся в начале одного скрипта.

## Выход

- `profiles_long.csv` — уровни и все метеопараметры;
- `profile_metrics.csv` — качество и инверсия по зондам;
- `sfc_raw.csv` — диагностика повторных SFC;
- `summary.json` — итоги и ошибки;
- `plots/` — отдельные месячные PNG.
