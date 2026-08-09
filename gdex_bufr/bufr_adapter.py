"""Декодирование BUFR через pybufrkit и адаптация к профилю radiosonde."""
from __future__ import annotations

import json
import logging
import math
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from gdex_bufr.bufr_tables import BufrTablesRegistry, configure_from_app, get_registry, normalize_fxy
from gdex_bufr.meteo_parser_bridge import (
    RadiosondeProfile,
    VerticalLevel,
    assess_profile_data,
    enrich_profile_levels,
    geopotential_to_height_m,
)

logger = logging.getLogger(__name__)

MISSING = 1e11

DESC_LAT = "005002"
DESC_LON = "006002"
DESC_WMO_BLOCK = "001001"
DESC_WMO_STATION = "001002"
DESC_YEAR = "004001"
DESC_MONTH = "004002"
DESC_DAY = "004003"
DESC_HOUR = "004004"
DESC_MINUTE = "004005"
DESC_PRESSURE = "007004"
# NCEP ADPUPA (GDEX): T/Td в 012101/012103 (K) или 012225/012227; 012023/012024 в файлах не встречаются.
DESC_TEMP = "012101"
DESC_TEMP_ALT = "012225"
DESC_TEMP_ALT2 = "012023"
DESC_DEWPOINT = "012103"
DESC_DEWPOINT_ALT = "012227"
DESC_DEWPOINT_ALT2 = "012024"
DESC_WDIR = "011001"
DESC_WSPD = "011002"
DESC_HEIGHT = "010009"
DESC_GEOPOT = "010008"
DESC_HEIGHT_COORD = "007007"
DESC_STATION_HEIGHT = "007001"  # Height of station [m]
DESC_RH = "013003"
DESC_VSIG = "008001"

# NCEP ADPUPA / WMO 0-08-001 (vertical sounding significance).
# Короткие метки для климатического пайплайна; 2 = Standard level = MANL.
ADPUPA_VSIG_LABELS: dict[int, str] = {
    1: "SFC",       # Surface
    2: "MANL",      # Standard / mandatory level
    3: "TROP",
    4: "MAXW",
    5: "SIGT",      # significant T/RH
    6: "SIGW",      # significant wind
    8: "MAXW",
    16: "TROP",
    32: "MANL",     # встречается в части ADPUPA как отдельный код
    64: "SFC",
}

# Битовые флаги WMO (номер бита → метка), схема 1<<(bit-1)
ADPUPA_VSIG_BIT_LABELS: dict[int, str] = {
    1: "SFC",
    2: "MANL",
    3: "TROP",
    4: "MAXW",
    5: "SIGT",
    6: "SIGW",
}

ADPUPA_LEVEL_FIELD_IDS = frozenset({
    8001,  # 008001
    7004,  # 007004
    12101,  # 012101
    12225,  # 012225 — часто в ранних ADPUPA вместо 012101
    12023,  # 012023 — fallback WMO
    12103,  # 012103
    12227,  # 012227 — Td alternate
    12024,  # 012024 — fallback WMO
    11001,  # 011001
    11002,  # 011002
    7007,  # 007007 — height coordinate
    10008,  # 010008 — geopotential
    10009,  # 010009 — geopotential height
})

ADPUPA_TEMP_DIDS = (12101, 12225, 12023)
ADPUPA_DEWPOINT_DIDS = (12103, 12227, 12024)

PROFILE_CODED_DESCRIPTORS = (
    "002001",
    "001011",
    "001012",
    "008010",
    "008021",
    "033007",
)

FULL_DECODE_EXTRA_DESCRIPTORS = (
    "001004",
    "001005",
    "001015",
    "002011",
    "002061",
    "004024",
    "004025",
)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        number = float(value)
        return abs(number - MISSING) < 1.0 or number >= 1e10
    except (TypeError, ValueError):
        return False


def _flatten_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        out: list[Any] = []
        for item in value:
            out.extend(_flatten_values(item))
        return out
    return [value]


def _query_values(
    message: Any,
    path: str,
    *,
    subset_index: int | None = None,
) -> dict[int, list[Any]]:
    from pybufrkit.dataquery import DataQuerent, NodePathParser

    query_path = f"@[{subset_index}]>{path}" if subset_index is not None else path
    result = DataQuerent(NodePathParser()).query(message, query_path)
    raw = result.results or {}
    parsed: dict[int, list[Any]] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            parsed[int(key)] = _flatten_values(value)
    elif isinstance(raw, list):
        parsed[0] = _flatten_values(raw)
    return parsed


def _subset_value(values_by_subset: dict[int, list[Any]], subset_index: int) -> Any | None:
    values = values_by_subset.get(subset_index, [])
    for value in values:
        if not _is_missing(value):
            return value
    return None


def _subset_series(values_by_subset: dict[int, list[Any]], subset_index: int) -> list[Any]:
    return [v for v in values_by_subset.get(subset_index, []) if not _is_missing(v)]


def _subset_series_descriptor(
    message: Any,
    subset_index: int,
    *descriptors: str,
    query_cache: dict[str, dict[int, list[Any]]] | None = None,
) -> list[Any]:
    """Беру первый непустой ряд по списку дескрипторов (fallback для ADPUPA)."""
    for descriptor in descriptors:
        if query_cache is not None:
            series = _subset_series(query_cache.get(descriptor, {}), subset_index)
        else:
            series = _subset_series(_query_values(message, descriptor), subset_index)
        if series:
            return series
    return []


def _values_for_message(
    message: Any,
    path: str,
    query_cache: dict[str, dict[int, list[Any]]] | None,
) -> dict[int, list[Any]]:
    if query_cache is not None:
        return query_cache.get(path, {})
    return _query_values(message, path)


PROFILE_QUERY_DESCRIPTORS = (
    DESC_LAT,
    DESC_LON,
    DESC_WMO_BLOCK,
    DESC_WMO_STATION,
    DESC_YEAR,
    DESC_MONTH,
    DESC_DAY,
    DESC_HOUR,
    DESC_MINUTE,
    DESC_PRESSURE,
    DESC_TEMP,
    DESC_TEMP_ALT,
    DESC_TEMP_ALT2,
    DESC_DEWPOINT,
    DESC_DEWPOINT_ALT,
    DESC_DEWPOINT_ALT2,
    DESC_WDIR,
    DESC_WSPD,
    DESC_HEIGHT,
    DESC_GEOPOT,
    DESC_HEIGHT_COORD,
    DESC_STATION_HEIGHT,
    DESC_RH,
    DESC_VSIG,
)


def _build_query_cache(
    message: Any,
    *,
    subset_index: int | None = None,
    extra_descriptors: Iterable[str] = (),
) -> dict[str, dict[int, list[Any]]]:
    """Один проход по каждому дескриптору; при subset_index — только этот subset."""
    paths = set(PROFILE_QUERY_DESCRIPTORS) | set(PROFILE_CODED_DESCRIPTORS) | set(extra_descriptors)
    return {
        path: _query_values(message, path, subset_index=subset_index)
        for path in paths
    }


def _subset_station_id(
    query_cache: dict[str, dict[int, list[Any]]],
    subset_index: int,
) -> str | None:
    block = _subset_value(query_cache.get(DESC_WMO_BLOCK, {}), subset_index)
    station_num = _subset_value(query_cache.get(DESC_WMO_STATION, {}), subset_index)
    if block is not None and station_num is not None:
        return f"{int(block):02d}{int(station_num):03d}"
    return None


def _normalize_pressure(value: Any, registry: BufrTablesRegistry | None = None) -> float | None:
    """Convert decoded 0-07-004 pressure to hPa using its declared unit.

    pybufrkit returns physical descriptor values. For 0-07-004 that unit is
    Pa, including small upper-air values such as 1000 Pa (= 10 hPa). A
    magnitude-only heuristic therefore aliases upper-air levels onto the
    troposphere (1000 Pa was previously interpreted as 1000 hPa).
    """
    if _is_missing(value):
        return None
    pressure = float(value)
    if not math.isfinite(pressure) or pressure <= 0:
        return None

    if registry:
        info = registry.lookup_descriptor(DESC_PRESSURE)
        unit = info.unit.strip().lower().replace(" ", "")
        if unit in {"pa", "pascal", "pascals"}:
            return pressure / 100.0
        if unit in {"hpa", "mb", "mbar", "millibar", "millibars"}:
            return pressure

    # Compatibility fallback for callers without a usable descriptor table.
    if pressure > 1100:
        return pressure / 100.0
    return pressure


def _adpupa_vsig_label(code: Any) -> str | None:
    if _is_missing(code):
        return None
    try:
        bits = int(code)
    except (TypeError, ValueError):
        return None

    # 1) точное совпадение с известными кодами ADPUPA
    if bits in ADPUPA_VSIG_LABELS:
        return ADPUPA_VSIG_LABELS[bits]

    # 2) WMO flag-биты: bit N → 1<<(N-1) (Surface=1, Standard/MANL=2, …)
    labels: list[str] = []
    for bit, name in ADPUPA_VSIG_BIT_LABELS.items():
        if bits & (1 << (bit - 1)):
            if name not in labels:
                labels.append(name)
    if labels:
        return "+".join(labels)

    return None


def _collect_debufr_elements(
    message: Any,
    subset_index: int,
    registry: BufrTablesRegistry,
) -> list[dict[str, Any]]:
    """Element dump в стиле NCEPLIBS-bufr ufdump: FXY, value, unit, kind, scale."""
    try:
        template_data = message.template_data.value
        template_data.wire()
        descs = template_data.decoded_descriptors_all_subsets[subset_index]
        vals = template_data.decoded_values_all_subsets[subset_index]
    except Exception:
        return []

    rows: list[dict[str, Any]] = []
    for seq, (descriptor, raw) in enumerate(zip(descs, vals), start=1):
        fxy = normalize_fxy(descriptor.id)
        info = registry.lookup_descriptor(fxy)
        value: Any = None if _is_missing(raw) else raw
        value_text = None
        if value is not None:
            if info.kind == "code":
                value_text = registry.decode_code_value(fxy, value)
            elif info.kind == "flag":
                value_text = registry.decode_flag_bits(fxy, value)
        rows.append({
            "seq": seq,
            "fxy": info.fxy,
            "name": info.name,
            "value": value,
            "value_text": value_text,
            "unit": info.unit,
            "kind": info.kind,
            "scale": info.scale,
            "reference": info.reference,
            "nbits": info.nbits,
        })
    return rows


def _decode_adpupa_flat_levels(
    message: Any,
    subset_index: int,
    *,
    registry: BufrTablesRegistry,
) -> list[VerticalLevel]:
    """Decode ADPUPA level records in template order.

    Each record starts with 0-08-001, then pressure, and only afterwards may
    contain temperature, dewpoint, geopotential and wind. Records are flushed
    at the next 0-08-001 instead of attaching post-pressure fields to a
    neighbouring level.
    """
    template_data = message.template_data.value
    template_data.wire()
    descs = template_data.decoded_descriptors_all_subsets[subset_index]
    vals = template_data.decoded_values_all_subsets[subset_index]

    current: dict[int, Any] = {}
    levels: list[VerticalLevel] = []
    seq = 0

    def _first_record_temperature(descriptor_ids: tuple[int, ...]) -> float | None:
        for did in descriptor_ids:
            if did in current:
                return _normalize_temperature(current[did])
        return None

    def _record_height() -> tuple[
        float | None,
        float | None,
        float | None,
        float | None,
        float | None,
    ]:
        geopotential = current.get(10008)
        geopotential_f = None if _is_missing(geopotential) else float(geopotential)
        height_phi_m = (
            None
            if geopotential_f is None
            else round(geopotential_to_height_m(geopotential_f), 1)
        )
        geopotential_height = current.get(10009)
        height_010009_m = (
            None if _is_missing(geopotential_height) else float(geopotential_height)
        )
        height_coordinate = current.get(7007)
        height_007007_m = (
            None if _is_missing(height_coordinate) else float(height_coordinate)
        )

        # Прямые BUFR-высоты имеют приоритет; Φ→z хранится отдельно.
        height_m = next(
            (
                value
                for value in (height_010009_m, height_007007_m, height_phi_m)
                if value is not None
            ),
            None,
        )
        return (
            height_m,
            geopotential_f,
            height_010009_m,
            height_007007_m,
            height_phi_m,
        )

    def _flush_record() -> None:
        nonlocal current, seq
        if 7004 not in current:
            current = {}
            return

        pressure = _normalize_pressure(current.get(7004), registry)
        if pressure is None or pressure > 1100:
            current = {}
            return

        vsig_code = current.get(8001)
        (
            height_m,
            geopotential_m2s2,
            height_010009_m,
            height_007007_m,
            height_phi_m,
        ) = _record_height()
        seq += 1
        levels.append(
            VerticalLevel(
                pressure_hpa=pressure,
                geopotential_height_m=height_m,
                height_010009_m=height_010009_m,
                height_007007_m=height_007007_m,
                height_phi_m=height_phi_m,
                geopotential_m2s2=geopotential_m2s2,
                air_temperature_c=_first_record_temperature(ADPUPA_TEMP_DIDS),
                dew_point_temperature_c=_first_record_temperature(ADPUPA_DEWPOINT_DIDS),
                wind_direction_deg=None
                if _is_missing(current.get(11001))
                else float(current[11001]),
                wind_speed=None
                if _is_missing(current.get(11002))
                else float(current[11002]),
                replication_index=seq - 1,
                seq=seq,
                vertical_significance=_adpupa_vsig_label(vsig_code),
                vertical_significance_code=None
                if _is_missing(vsig_code)
                else int(vsig_code),
            )
        )
        current = {}

    for descriptor, raw in zip(descs, vals):
        did = descriptor.id
        if did not in ADPUPA_LEVEL_FIELD_IDS:
            continue

        if did == 8001:
            _flush_record()
            current[did] = raw
            continue

        # Defensive boundary for malformed/local templates without 0-08-001.
        if did == 7004 and 7004 in current:
            _flush_record()
        current[did] = raw

    _flush_record()
    return levels


def _adpupa_sig_level_count(pressures: list[float | None]) -> int:
    """Первая репликация ADPUPA заканчивается скачком давления вверх (новая секция шаблона)."""
    for idx in range(1, len(pressures)):
        prev_p = pressures[idx - 1]
        curr_p = pressures[idx]
        if prev_p is None or curr_p is None:
            continue
        if curr_p > prev_p + 10:
            return idx
    return len(pressures)


def _normalize_temperature(value: Any) -> float | None:
    if _is_missing(value):
        return None
    temp = float(value)
    if temp > 150:
        return temp - 273.15
    return temp

# Декодирую зашифрованное поле (код или флаг)
def _decode_coded_field(
    message: Any,
    subset_index: int,
    fxy: str,
    registry: BufrTablesRegistry,
    *,
    query_cache: dict[str, dict[int, list[Any]]] | None = None,
) -> dict[str, Any] | None:
    raw = _subset_value(_values_for_message(message, fxy, query_cache), subset_index)
    if raw is None:
        return None
    info = registry.lookup_descriptor(fxy)
    entry: dict[str, Any] = {
        "descriptor": normalize_fxy(fxy),
        "name": info.name,
        "name_ru": info.name_ru,
        "value": raw,
        "unit": info.unit,
        "kind": info.kind,
    }
    if info.kind == "code":
        entry["value_text"] = registry.decode_code_value(fxy, raw)
    elif info.kind == "flag":
        entry["value_text"] = registry.decode_flag_bits(fxy, raw)
    return entry

# Получаю метаданные заголовка сообщения BUFR
def _message_header_metadata(message: Any) -> dict[str, Any]:
    from pybufrkit.mdquery import MetadataExprParser, MetadataQuerent

    mq = MetadataQuerent(MetadataExprParser())
    keys = (
        "%master_table_number",
        "%master_table_version",
        "%local_table_version",
        "%bufr_header_edition",
        "%data_category",
        "%data_subcategory",
        "%originating_centre",
        "%n_subsets",
    )
    meta: dict[str, Any] = {}
    for key in keys:
        try:
            meta[key.lstrip("%")] = mq.query(message, key)
        except Exception:
            meta[key.lstrip("%")] = None
    return meta

# Итерация по сообщениям BUFR (генерация сообщений из байтового потока)
def _iter_messages(decoder, raw: bytes):
    from pybufrkit import decoder as dec_mod

    if hasattr(dec_mod, "generate_bufr_messages"):
        yield from dec_mod.generate_bufr_messages(decoder, raw, continue_on_error=True)
    else:
        yield from dec_mod.generate_bufr_message(decoder, raw, continue_on_error=True)

# Проверяю, является ли сообщение наблюдением (данные категории 2 и есть subset)
def _is_observation_message(message: Any) -> bool:
    from pybufrkit.mdquery import MetadataExprParser, MetadataQuerent

    mq = MetadataQuerent(MetadataExprParser())
    category = mq.query(message, "%data_category")
    n_subsets = int(mq.query(message, "%n_subsets") or 0)
    return category == 2 and n_subsets > 0

# Создаю декодер BUFR (pybufrkit) для декодирования BUFR-файлов с использованием офицального справочника таблиц WMO
def _make_decoder(registry: BufrTablesRegistry):
    from pybufrkit.decoder import Decoder

    tables_root = str(registry.tables_root) if registry.is_ready() else None
    return Decoder(tables_root_dir=tables_root, fallback_or_ignore_missing_tables=True)

# Декодирую BUFR-файл, получая список профилей RadiosondeProfile (RadiosondeProfile - класс для хранения профиля радиолокационного зондирования)
def decode_bufr_file(
    path: Path,
    *,
    max_profiles: int | None = None,
    station_id: str | None = None,
    registry: BufrTablesRegistry | None = None,
    decode_mode: str = "adpupa",
    decoder: Any | None = None,
) -> list[RadiosondeProfile]:
    import contextlib
    import io

    # pybufrkit печатает в stdout при битых дескрипторах — это сильно тормозит массовую расшифровку
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return _decode_bufr_file_impl(
            path,
            max_profiles=max_profiles,
            station_id=station_id,
            registry=registry,
            decode_mode=decode_mode,
            decoder=decoder,
        )


def _message_subset_indices(
    message: Any,
    n_subsets: int,
    *,
    station_id: str | None,
) -> list[int]:
    """Какие subset декодировать: все или только совпадающие с WMO station_id."""
    if station_id is None:
        return list(range(n_subsets))

    block_map = _query_values(message, DESC_WMO_BLOCK)
    station_map = _query_values(message, DESC_WMO_STATION)
    station_maps = {DESC_WMO_BLOCK: block_map, DESC_WMO_STATION: station_map}
    return [
        idx
        for idx in range(n_subsets)
        if _subset_station_id(station_maps, idx) == station_id
    ]


def _decode_bufr_file_impl(
    path: Path,
    *,
    max_profiles: int | None = None,
    station_id: str | None = None,
    registry: BufrTablesRegistry | None = None,
    decode_mode: str = "adpupa",
    decoder: Any | None = None,
) -> list[RadiosondeProfile]:
    registry = registry or get_registry()
    decoder = decoder or _make_decoder(registry)
    raw = path.read_bytes()
    profiles: list[RadiosondeProfile] = []
    extra_descriptors = FULL_DECODE_EXTRA_DESCRIPTORS if decode_mode == "full" else ()

    for message in _iter_messages(decoder, raw):
        if not _is_observation_message(message):
            continue
        from pybufrkit.mdquery import MetadataExprParser, MetadataQuerent

        n_subsets = int(MetadataQuerent(MetadataExprParser()).query(message, "%n_subsets") or 0)
        subset_indices = _message_subset_indices(message, n_subsets, station_id=station_id)
        if not subset_indices:
            continue

        header_meta = _message_header_metadata(message)
        for subset_idx in subset_indices:
            if max_profiles is not None and len(profiles) >= max_profiles:
                return profiles

            query_cache = _build_query_cache(
                message,
                subset_index=subset_idx,
                extra_descriptors=extra_descriptors,
            )
            profile = _decode_subset(
                path,
                message,
                subset_idx,
                registry=registry,
                decode_mode=decode_mode,
                header_meta=header_meta,
                query_cache=query_cache,
            )
            if station_id is not None and profile.station_id != station_id:
                continue
            profiles.append(profile)
            if max_profiles is not None and len(profiles) >= max_profiles:
                return profiles
            # Не выходим после первого hit станции: в файле могут быть
            # несколько сроков/subset одной WMO (дополнения, 00+12 в одном сообщении).
    return profiles


def _decode_subset_header(
    message: Any,
    subset_index: int,
    query_cache: dict[str, dict[int, list[Any]]] | None,
) -> tuple[
    float | None,
    float | None,
    str | None,
    str | None,
    float | None,
]:
    """Шапка subset: координаты, WMO, срок, высота станции."""
    lat = _subset_value(_values_for_message(message, DESC_LAT, query_cache), subset_index)
    lon = _subset_value(_values_for_message(message, DESC_LON, query_cache), subset_index)
    station_id = _subset_station_id(
        {
            DESC_WMO_BLOCK: _values_for_message(message, DESC_WMO_BLOCK, query_cache),
            DESC_WMO_STATION: _values_for_message(message, DESC_WMO_STATION, query_cache),
        },
        subset_index,
    )

    year = _subset_value(_values_for_message(message, DESC_YEAR, query_cache), subset_index)
    month = _subset_value(_values_for_message(message, DESC_MONTH, query_cache), subset_index)
    day = _subset_value(_values_for_message(message, DESC_DAY, query_cache), subset_index)
    hour = _subset_value(_values_for_message(message, DESC_HOUR, query_cache), subset_index)
    minute = _subset_value(_values_for_message(message, DESC_MINUTE, query_cache), subset_index)

    report_dt = None
    if all(v is not None for v in (year, month, day, hour)):
        minute_val = 0 if minute is None else int(minute)
        report_dt = (
            f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
            f"T{int(hour):02d}:{minute_val:02d}:00"
        )

    station_height_raw = _subset_value(
        _values_for_message(message, DESC_STATION_HEIGHT, query_cache),
        subset_index,
    )
    station_elevation_m = None
    if not _is_missing(station_height_raw):
        try:
            station_elevation_m = round(float(station_height_raw), 1)
        except (TypeError, ValueError):
            station_elevation_m = None

    lat_deg = None if lat is None else float(lat)
    lon_deg = None if lon is None else float(lon)
    return lat_deg, lon_deg, station_id, report_dt, station_elevation_m


def _decode_subset_raw_series(
    message: Any,
    subset_index: int,
    *,
    registry: BufrTablesRegistry,
    query_cache: dict[str, dict[int, list[Any]]] | None,
) -> tuple[list[float | None], list[float | None], list[float | None], list[float], list[float], list[float], list[float]]:
    """Параллельные ряды дескрипторов до сборки уровней (full mode и метаданные)."""
    pressures = [
        _normalize_pressure(v, registry)
        for v in _subset_series(_values_for_message(message, DESC_PRESSURE, query_cache), subset_index)
    ]
    temps = [
        _normalize_temperature(v)
        for v in _subset_series_descriptor(
            message, subset_index, DESC_TEMP, DESC_TEMP_ALT, DESC_TEMP_ALT2, query_cache=query_cache
        )
    ]
    dewpoints = [
        _normalize_temperature(v)
        for v in _subset_series_descriptor(
            message,
            subset_index,
            DESC_DEWPOINT,
            DESC_DEWPOINT_ALT,
            DESC_DEWPOINT_ALT2,
            query_cache=query_cache,
        )
    ]
    wind_dirs = [
        float(v) for v in _subset_series(_values_for_message(message, DESC_WDIR, query_cache), subset_index)
    ]
    wind_speeds = [
        float(v) for v in _subset_series(_values_for_message(message, DESC_WSPD, query_cache), subset_index)
    ]
    heights = [
        float(v) for v in _subset_series(_values_for_message(message, DESC_HEIGHT, query_cache), subset_index)
    ]
    rh_series = [
        float(v) for v in _subset_series(_values_for_message(message, DESC_RH, query_cache), subset_index)
    ]
    return pressures, temps, dewpoints, wind_dirs, wind_speeds, heights, rh_series


def _decode_subset_levels(
    message: Any,
    subset_index: int,
    *,
    path: Path,
    registry: BufrTablesRegistry,
    decode_mode: str,
    station_id: str | None,
    station_elevation_m: float | None,
    pressures: list[float | None],
    temps: list[float | None],
    dewpoints: list[float | None],
    wind_dirs: list[float],
    wind_speeds: list[float],
    heights: list[float],
    rh_series: list[float],
) -> tuple[list[VerticalLevel], dict[str, Any]]:
    """Уровни профиля: ADPUPA flat scan или выравнивание рядов (full)."""
    if decode_mode == "adpupa":
        levels = _decode_adpupa_flat_levels(message, subset_index, registry=registry)
        enriched = enrich_profile_levels(
            RadiosondeProfile(
                source_file=str(path),
                subset_index=subset_index,
                station_id=station_id,
                station_elevation_m=station_elevation_m,
                levels=levels,
                metadata=(
                    {"station_elevation_m": station_elevation_m}
                    if station_elevation_m is not None
                    else {}
                ),
            )
        )
        return enriched.levels, dict(enriched.metadata.get("enrichment", {}))

    lengths = [
        len(pressures),
        len(temps),
        len(dewpoints),
        len(wind_dirs),
        len(wind_speeds),
        len(heights),
        len(rh_series),
    ]
    max_len = max(lengths) if lengths else 0
    levels = []
    for idx in range(max_len):
        pressure = pressures[idx] if idx < len(pressures) else None
        temp = temps[idx] if idx < len(temps) else None
        dewpoint = dewpoints[idx] if idx < len(dewpoints) else None
        wind_dir = wind_dirs[idx] if idx < len(wind_dirs) else None
        wind_speed = wind_speeds[idx] if idx < len(wind_speeds) else None
        if pressure is None and temp is None and dewpoint is None and wind_speed is None:
            continue
        levels.append(
            VerticalLevel(
                pressure_hpa=pressure,
                geopotential_height_m=heights[idx] if idx < len(heights) else None,
                height_010009_m=heights[idx] if idx < len(heights) else None,
                air_temperature_c=temp,
                dew_point_temperature_c=dewpoint,
                wind_direction_deg=wind_dir,
                wind_speed=wind_speed,
                relative_humidity_percent=rh_series[idx] if idx < len(rh_series) else None,
                replication_index=idx,
            )
        )
    levels = [lv for lv in levels if lv.pressure_hpa is not None]
    levels.sort(key=lambda lv: -(lv.pressure_hpa or 0.0))
    return levels, {}


def _decode_subset_metadata(
    *,
    path: Path,
    message: Any,
    subset_index: int,
    registry: BufrTablesRegistry,
    decode_mode: str,
    header_meta: dict[str, Any],
    query_cache: dict[str, dict[int, list[Any]]] | None,
    station_id: str | None,
    station_elevation_m: float | None,
    levels: list[VerticalLevel],
    pressures: list[float | None],
    temps: list[float | None],
    wind_speeds: list[float],
    enrichment_meta: dict[str, Any],
) -> dict[str, Any]:
    """Метаданные subset: coded fields, debufr dump, QC-статус."""
    coded_metadata: dict[str, Any] = {}
    for fxy in PROFILE_CODED_DESCRIPTORS:
        decoded = _decode_coded_field(message, subset_index, fxy, registry, query_cache=query_cache)
        if decoded:
            coded_metadata[normalize_fxy(fxy)] = decoded

    all_fields: list[dict[str, Any]] = []
    if decode_mode == "full":
        for fxy in FULL_DECODE_EXTRA_DESCRIPTORS:
            decoded = _decode_coded_field(message, subset_index, fxy, registry, query_cache=query_cache)
            if decoded:
                coded_metadata[normalize_fxy(fxy)] = decoded
        for fxy in sorted(set(PROFILE_CODED_DESCRIPTORS + FULL_DECODE_EXTRA_DESCRIPTORS)):
            raw = _subset_value(_values_for_message(message, fxy, query_cache), subset_index)
            if raw is None:
                continue
            info = registry.lookup_descriptor(fxy)
            all_fields.append({
                "descriptor": normalize_fxy(fxy),
                "name": info.name,
                "name_ru": info.name_ru,
                "value": raw,
                "value_text": coded_metadata.get(normalize_fxy(fxy), {}).get("value_text"),
                "unit": info.unit,
                "kind": info.kind,
            })

    metadata: dict[str, Any] = {
        "decoder": "pybufrkit",
        "template": "ncep_adpupa",
        "decode_mode": decode_mode,
        "table_edition": header_meta.get("master_table_version"),
        "bufr_header": header_meta,
    }
    if station_elevation_m is not None:
        metadata["station_elevation_m"] = station_elevation_m
        metadata["station_height_fxy"] = DESC_STATION_HEIGHT

    debufr_elements = _collect_debufr_elements(message, subset_index, registry)
    if debufr_elements:
        metadata["debufr_elements"] = debufr_elements
    if coded_metadata:
        metadata["coded_metadata"] = coded_metadata
    if enrichment_meta:
        metadata["enrichment"] = enrichment_meta
    if all_fields:
        metadata["all_fields"] = all_fields

    data_status, data_status_reason = assess_profile_data(
        RadiosondeProfile(
            source_file=str(path),
            subset_index=subset_index,
            station_id=station_id,
            levels=levels,
        )
    )
    metadata["data_status"] = data_status
    metadata["data_status_reason"] = data_status_reason
    metadata["n_pressure_raw"] = len(pressures)
    metadata["n_temp_raw"] = len(temps)
    metadata["n_wind_raw"] = len(wind_speeds)
    if decode_mode == "adpupa":
        metadata["n_adpupa_rows"] = len(levels)
        vsig_counts: dict[str, int] = {}
        for lv in levels:
            label = lv.vertical_significance or "?"
            vsig_counts[label] = vsig_counts.get(label, 0) + 1
        metadata["adpupa_vsig_counts"] = vsig_counts
    return metadata


# Декодирую subset (подмножество данных)
def _decode_subset(
    path: Path,
    message: Any,
    subset_index: int,
    *,
    registry: BufrTablesRegistry,
    decode_mode: str,
    header_meta: dict[str, Any],
    query_cache: dict[str, dict[int, list[Any]]] | None = None,
) -> RadiosondeProfile:
    lat_deg, lon_deg, station_id, report_dt, station_elevation_m = _decode_subset_header(
        message,
        subset_index,
        query_cache,
    )
    pressures, temps, dewpoints, wind_dirs, wind_speeds, heights, rh_series = _decode_subset_raw_series(
        message,
        subset_index,
        registry=registry,
        query_cache=query_cache,
    )
    levels, enrichment_meta = _decode_subset_levels(
        message,
        subset_index,
        path=path,
        registry=registry,
        decode_mode=decode_mode,
        station_id=station_id,
        station_elevation_m=station_elevation_m,
        pressures=pressures,
        temps=temps,
        dewpoints=dewpoints,
        wind_dirs=wind_dirs,
        wind_speeds=wind_speeds,
        heights=heights,
        rh_series=rh_series,
    )
    metadata = _decode_subset_metadata(
        path=path,
        message=message,
        subset_index=subset_index,
        registry=registry,
        decode_mode=decode_mode,
        header_meta=header_meta,
        query_cache=query_cache,
        station_id=station_id,
        station_elevation_m=station_elevation_m,
        levels=levels,
        pressures=pressures,
        temps=temps,
        wind_speeds=wind_speeds,
        enrichment_meta=enrichment_meta,
    )

    return RadiosondeProfile(
        source_file=str(path),
        subset_index=subset_index,
        station_id=station_id,
        latitude_deg=lat_deg,
        longitude_deg=lon_deg,
        report_datetime_utc=report_dt,
        station_elevation_m=station_elevation_m,
        levels=levels,
        metadata=metadata,
    )


def init_decoder_tables(bufr_tables_config: dict[str, Any] | None) -> BufrTablesRegistry:
    """Инициализирую глобальный справочник из YAML-конфига."""
    return configure_from_app(bufr_tables_config)


def export_decoded_fields_csv(profile: RadiosondeProfile, output_path: Path) -> Path | None:
    fields = profile.metadata.get("all_fields")
    if not fields:
        return None
    import csv

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["descriptor", "name", "name_ru", "value", "value_text", "unit", "kind"],
        )
        writer.writeheader()
        for row in fields:
            writer.writerow(row)
    return output_path


def pybufrkit_decode_json(path: Path) -> list[dict[str, Any]]:
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "pybufrkit", "decode", "-j", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "pybufrkit decode failed")

    payloads: list[dict[str, Any]] = []
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("["):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, list):
            payloads.append({"sections": payload})
    if payloads:
        return payloads
    raise RuntimeError("No JSON payloads found in pybufrkit stdout")
