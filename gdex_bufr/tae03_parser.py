"""Парсер таблиц зондирования ТАЭ-3 (кодировка cp1251)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from gdex_bufr.meteo_parser_bridge import RadiosondeProfile, VerticalLevel

NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
HEADER_PREFIXES = (
    "ТАБЛИЦА",
    "НАЧАЛО",
    "КОНЕЦ",
    "ВЫСОТА",
    "СИНОПТИЧЕСКИЙ",
    "КОД",
    "ПРИЗЕМНАЯ",
    "H",
)


@dataclass
class Tae03Metadata:
    start_datetime: str
    end_datetime: str | None = None
    station_id: str = "31977"
    cloud_code: str | None = None
    temp_bias_c: float | None = None
    rh_bias_percent: float | None = None
    sun_elevation_deg: float | None = None
    extra: dict[str, str] = field(default_factory=dict)


def _is_num(token: str) -> bool:
    return bool(NUM_RE.match(token))


def _parse_float(token: str) -> float:
    return float(token.replace(",", "."))


def parse_tae03_metadata(text: str) -> Tae03Metadata:
    start_match = re.search(r"НАЧАЛО НАБЛЮДЕНИЙ\s*:\s*(\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2})", text)
    end_match = re.search(r"КОНЕЦ НАБЛЮДЕНИЙ\s*:\s*(\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2})", text)
    station_match = re.search(r"СИНОПТИЧЕСКИЙ ИНДЕКС СТАНЦИИ\s*:\s*(\d+)", text)
    cloud_match = re.search(r"КОД ОБЛАЧНОСТИ\s*:\s*(\d+)", text)
    temp_bias_match = re.search(r"ПРИЗЕМНАЯ ОШИБКА ТЕМПЕРАТУРЫ\s*:\s*([-\d.,]+)", text)
    rh_bias_match = re.search(r"ПРИЗЕМНАЯ ОШИБКА ВЛАЖНОСТИ\s*:\s*([-\d.,]+)", text)
    sun_match = re.search(r"ВЫСОТА СОЛНЦА ПРИ ПУСКЕ\s*:\s*([-\d.,]+)", text)

    if not start_match:
        raise ValueError("Не найдено время начала наблюдений в файле ТАЭ-3")

    start_dt = datetime.strptime(start_match.group(1), "%d.%m.%Y %H:%M").strftime("%Y-%m-%dT%H:%M:00")
    end_dt = None
    if end_match:
        end_dt = datetime.strptime(end_match.group(1), "%d.%m.%Y %H:%M").strftime("%Y-%m-%dT%H:%M:00")

    return Tae03Metadata(
        start_datetime=start_dt,
        end_datetime=end_dt,
        station_id=station_match.group(1) if station_match else "31977",
        cloud_code=cloud_match.group(1) if cloud_match else None,
        temp_bias_c=_parse_float(temp_bias_match.group(1)) if temp_bias_match else None,
        rh_bias_percent=_parse_float(rh_bias_match.group(1)) if rh_bias_match else None,
        sun_elevation_deg=_parse_float(sun_match.group(1)) if sun_match else None,
    )


def _parse_level_line(line: str) -> VerticalLevel | None:
    tokens = line.split()
    num_idx = next((i for i, tok in enumerate(tokens) if _is_num(tok)), None)
    if num_idx is None:
        return None

    nums = [_parse_float(t) for t in tokens[num_idx:] if _is_num(t)]
    if len(nums) < 6:
        return None

    h_km, pressure, temp, rh = nums[0], nums[1], nums[2], nums[3]
    if len(nums) >= 7:
        wind_dir, wind_speed, dewpoint_deficit = nums[4], nums[5], nums[6]
    else:
        wind_dir, wind_speed, dewpoint_deficit = 0.0, nums[4], nums[5]
    # В таблице ТАЭ-3 последний столбец — дефицит точки росы (T − Td), не Td.
    dewpoint = temp - dewpoint_deficit

    return VerticalLevel(
        pressure_hpa=pressure,
        geopotential_height_m=h_km * 1000.0,
        air_temperature_c=temp,
        dew_point_temperature_c=dewpoint,
        wind_direction_deg=wind_dir,
        wind_speed=wind_speed,
        relative_humidity_percent=rh,
    )


def parse_tae03(
    path: Path | str,
    *,
    latitude_deg: float | None = None,
    longitude_deg: float | None = None,
) -> RadiosondeProfile:
    path = Path(path)
    text = path.read_text(encoding="cp1251", errors="replace")
    meta = parse_tae03_metadata(text)

    levels: list[VerticalLevel] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(HEADER_PREFIXES):
            continue
        level = _parse_level_line(line)
        if level is not None:
            levels.append(level)

    levels.sort(key=lambda lv: -(lv.pressure_hpa or 0.0))
    return RadiosondeProfile(
        source_file=str(path),
        subset_index=0,
        station_id=meta.station_id,
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
        report_datetime_utc=meta.start_datetime,
        levels=levels,
        metadata={
            "source": "TAE03",
            "end_datetime": meta.end_datetime,
            "cloud_code": meta.cloud_code,
            "temp_bias_c": meta.temp_bias_c,
            "rh_bias_percent": meta.rh_bias_percent,
            "sun_elevation_deg": meta.sun_elevation_deg,
        },
    )
