"""Диагностика: декодировать один BUFR для Алдана и показать уровни."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import logging
logging.basicConfig(level=logging.ERROR)

from gdex_bufr.bufr_adapter import decode_bufr_file
from gdex_bufr.bufr_tables import get_registry

path = Path("gdex_data/raw/1999/gdas.adpupa.t12z.19991001.bufr")
registry = get_registry()
profiles = decode_bufr_file(path, station_id="31004", registry=registry, decode_mode="adpupa")
print("profiles:", len(profiles))
for p in profiles:
    n_both = sum(1 for lv in p.levels if lv.pressure_hpa and lv.air_temperature_c is not None)
    print("levels:", len(p.levels), "with T+P:", n_both)
    for lv in p.levels[:8]:
        print(f"  P={lv.pressure_hpa:.0f} T={lv.air_temperature_c:.1f}°C" if lv.air_temperature_c else f"  P={lv.pressure_hpa} T=None")
