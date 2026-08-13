# Дерево Дальнего Востока

Новые прогоны, индекс и станции живут **здесь**. Старые папки не трогаем:

- `gdex_outputs/актуальное` — эталон дашборда
- `gdex_outputs/результаты-алдан-прогон_20260806_022516` — эталон decode Алдана
- `gdex_outputs/результаты-алдан` — старый baseline

Список станций — только [`stations_catalog.yaml`](../../stations_catalog.yaml).
Новая станция = новая строка YAML (`id`, `slug`, `name`, `elevation_m`, `region: far_east`).
Выключить станцию: `enabled: false`. Код не менять.

```text
far_east/
  stations/{slug}/    # появляется при extract; aldan уже наполнен из прогона
  regions/far_east/   # station_index.csv
```

Без флагов скрипты берут `default_region` / `default_station` из каталога:

```powershell
py -3 -m gdex_bufr station-index --limit-files 8
py -3 run_fast_extract.py --start-date 1999-10-01 --end-date 1999-10-03 --limit-files 4 --workers 2
py -3 run_fast_extract.py --station aldan --limit-files 4
```
