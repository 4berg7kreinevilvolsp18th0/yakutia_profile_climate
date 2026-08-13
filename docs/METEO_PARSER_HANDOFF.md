# Вынос ядра в meteo_parser

После стабилизации API в `yakutia_profile_climate` перенести в upstream
[meteo_parser](https://github.com/4berg7kreinevilvolsp18th0/meteo_parser) только ingest-слой.

## Переносить в meteo_parser

- `gdex_bufr/bufr_adapter.py`, `meteo_parser_bridge.py`, `bufr_tables.py`
- `gdex_bufr/downloader.py`, `manifest.py`, `state.py`
- `gdex_bufr/profile_climate/extract.py`, `export.py` (базовые CSV, без inversion v3 writers)
- `gdex_bufr/profile_climate/height_fill.py`
- `gdex_bufr/profile_climate/config.py` — `StationsCatalog` / `load_stations_catalog`
- `gdex_bufr/station_index.py` + CLI `station-index`
- CLI download / discover / station-profiles (без климатических PNG)

Контракт long-таблицы: все параметры sounding (P, T, H, RH, wind, dew), не только температура.

## Оставлять в yakutia_profile_climate

- inversion v2 (`inversion.py`) и gap-v3 (`inversion_layers.py`)
- `scripts/profile_dashboard.py`, `build_daily_profiles.py`, `compute_inversion_v3.py`
- `Графики для статьи/`
- региональные выходы `gdex_outputs/far_east/` и эталоны `актуальное` / прогоны
- `stations_catalog.yaml` региона ДВ

После handoff этот репозиторий держит копию/зависимость `gdex_bufr` и только продуктовые скрипты.
