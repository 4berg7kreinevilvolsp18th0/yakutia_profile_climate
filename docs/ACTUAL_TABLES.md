# Актуальные таблицы Алдана

Каталог `gdex_outputs/актуальное/` содержит два представления одного BUFR-архива.

## Таблицы

- `decoded_all_levels.csv` — все уровни и секции BUFR. MANL/SFC около
  1000/992 гПа не удаляются, но получают `below_station=true` и
  `qc_flag=below_station`.
- `profiles_working.csv` — физический профиль от поверхности станции до
  500 гПа. Он используется для высот, инверсии, графиков и дашборда.
- `profiles_long.csv` — совместимая копия рабочей таблицы для старых команд.
- `profile_metrics.csv` — метрики и подтверждённая инверсия v2.
- `aldan_actual.xlsx` — те же данные на листах `decoded_all_levels`,
  `profiles_working` и `profile_metrics`.

## Высоты

- `height_010009_m`, `height_007007_m` — прямые значения дескрипторов BUFR;
- `geopotential_m2s2` — исходный геопотенциал Φ;
- `height_phi_m` — отдельное преобразование Φ→z;
- `height_msl_m` — рабочая высота над уровнем моря;
- `height_agl_m` — высота над станцией/местом запуска;
- `height_m` — совместимый алиас `height_msl_m`.

Для Алдана правильная поверхность около 927 гПа имеет примерно 680 м MSL и
0 м AGL. Уровни 1000/992 гПа с высотой около 40–70 м имеют отрицательную AGL:
они доступны в полной таблице, но не участвуют в расчёте инверсии.

## Полный запуск на машине с архивом

```powershell
cd "B:\Kutunika programmist\yakutia_profile_climate"

python -m gdex_bufr.run_fast_extract `
  --actual `
  --fresh `
  --station aldan `
  --start-date 1999-10-01 `
  --end-date 2026-07-08 `
  --cycles 00,12

python scripts/build_daily_profiles.py

python -m gdex_bufr monthly-profile-plots

python -m streamlit run scripts/profile_dashboard.py -- `
  --data "gdex_outputs/актуальное/daily_profiles.json"
```

`--fresh` очищает известные файлы только в выбранном выходном каталоге.
Старые каталоги `результаты-алдан` не удаляются.
