# Yakutia Profile Climate

Исследовательский репозиторий для анализа вертикальных температурных профилей станций **Алдан** и **Якутск** по данным **GDAS ADPUPA BUFR** (GDEX/RDA d351000) за 1999–2026.

Репозиторий содержит **копию** пакета [`gdex_bufr`](https://github.com/4berg7kreinevilvolsp18th0/meteo_parser) из  [`meteo_parser`](https://github.com/4berg7kreinevilvolsp18th0/meteo_parser) и новый модуль `gdex_bufr/profile_climate/`.

Подробная инструкция: [PROFILE_CLIMATE_README.md](PROFILE_CLIMATE_README.md)

Гайд по архитектуре и высотам (для объяснений): [docs/HEIGHT_ARCHITECTURE_GUIDE.md](docs/HEIGHT_ARCHITECTURE_GUIDE.md)  
Методы/формулы (статья): [docs/METHODS_HEIGHT_INVERSION.md](docs/METHODS_HEIGHT_INVERSION.md)

## Быстрый старт

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python -m gdex_bufr tables-update

# Скачать и извлечь только Алдан (31004): поток BUFR → извлечь → удалить файл
python scripts/download_aldan.py --start-date 1999-10-01 --end-date 1999-10-31

python -m gdex_bufr manifest --start-date 1999-10-01 --end-date 1999-10-31
python -m gdex_bufr download --limit-files 10

python -m gdex_bufr discover-stations --start-date 1999-01-01 --end-date 1999-01-31 --include-all-files

python -m gdex_bufr station-profiles \
  --station 31004 \
  --start-date 1999-01-01 --end-date 1999-01-31 \
  --cycles 00,12 \
  --output gdex_outputs/profile_climate

python -m gdex_bufr monthly-profile-plots \
  --station aldan \
  --start-date 1999-01-01 --end-date 1999-01-31 \
  --input gdex_outputs/результаты-алдан/profiles_long.csv \
  --metrics gdex_outputs/результаты-алдан/profile_metrics.csv \
  --output gdex_outputs/monthly_temperature_profiles \
  --set актуальное
```

## Тесты

```bash
pytest -c gdex_pytest.ini tests/test_profile_climate_*.py
```

## Станции

| Станция | WMO ID | slug |
|---------|--------|------|
| Алдан | 31004 | `aldan` |
| Якутск | 24959 | `yakutsk` |

## Синхронизация с upstream

Пакет `gdex_bufr/` скопирован из `meteo_parser`. При обновлении upstream сверяйте изменения вручную и переносите только нужные исправления decode/download, не затрагивая `profile_climate/`.

## Лицензия и данные

Данные GDEX d351000 предоставляются NCAR/RDA. Используйте в соответствии с условиями доступа к датасету.
