# Справочники метеодиаграмм GDEX BUFR

Краткие описания всех типов графиков, цветовых слотов и BUFR-полей.

## Графики

| Тип | Справочник | Цвет в `gdex_config.yaml` |
|-----|------------|---------------------------|
| skewt | [plots/skewt.md](plots/skewt.md) | `temperature`, `dewpoint`, `barbs` |
| profile | [plots/profile.md](plots/profile.md) | `temperature`, `dewpoint` |
| wind | [plots/wind.md](plots/wind.md) | `wind_speed`, `wind_direction` |
| hodograph | [plots/hodograph.md](plots/hodograph.md) | `hodograph` |
| rh | [plots/rh.md](plots/rh.md) | `rh` |
| thermo | [plots/thermo.md](plots/thermo.md) | `thermo_box` |
| height | [plots/height.md](plots/height.md) | `height` |
| theta_e | [plots/theta_e.md](plots/theta_e.md) | `theta_e` |
| ttd | [plots/ttd.md](plots/ttd.md) | `ttd` |
| wind_shear | [plots/wind_shear.md](plots/wind_shear.md) | `wind_shear` |
| composite | [plots/composite.md](plots/composite.md) | все слоты `theme.colors` |
| map | [plots/map.md](plots/map.md) | `map_stations` |

## Палитра

Все цвета задаются в [`gdex_config.yaml`](../../gdex_config.yaml) → `plots.theme.colors`.  
Пример текущей яркой палитры: `#FF4757` (T), `#00CEC9` (Td), `#00B894` (ветер).

## Ваши дополнения

Папка [`custom/`](custom/) — для ваших справочников (методики, региональные особенности, QC).  
Файлы из `custom/` не перезаписываются при обновлении проекта.

## Программный доступ

```python
from gdex_bufr.plot_guides import PLOT_GUIDES, guide_path, list_guides
```
