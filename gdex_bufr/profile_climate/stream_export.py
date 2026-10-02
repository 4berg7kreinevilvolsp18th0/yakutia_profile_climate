"""Потоковый экспорт порций в общую таблицу и папки станций.

CSV открываются один раз. В памяти остаются только метрики и одна порция,
а подробные уровни всего региона не накапливаются. XLSX — отдельная опция.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

from gdex_bufr.profile_climate.export import (
    PROFILES_LONG_COLUMNS, PROFILE_METRICS_COLUMNS, DECODED_LEVEL_COLUMNS,
    DEBUFR_ELEMENT_COLUMNS, write_monthly_summary, write_station_summary,
    write_field_types_csv, write_xlsx_exports,
)


def export_store(store, output: Path, *, config: dict, station_slugs: dict[str, str],
                 split: bool, output_mode: str, working_copy: bool = False,
                 xlsx: bool = False) -> dict:
    tables = {"long": ("profiles_long.csv", PROFILES_LONG_COLUMNS),
              "metrics": ("profile_metrics.csv", PROFILE_METRICS_COLUMNS)}
    if output_mode in ("decoded", "audit"):
        tables["decoded"] = ("decoded_levels.csv", DECODED_LEVEL_COLUMNS)
    if output_mode == "audit":
        tables["elements"] = ("debufr_elements.csv", DEBUFR_ELEMENT_COLUMNS)
    destinations = {"all": output}
    if split:
        destinations.update({slug: output / "stations" / slug for slug in set(station_slugs.values())})
    metrics = defaultdict(list)
    counts = defaultdict(int)
    temporary = []
    with ExitStack() as stack:
        writers = {}
        for station, directory in destinations.items():
            directory.mkdir(parents=True, exist_ok=True)
            for kind, (name, columns) in tables.items():
                final = directory / name
                temp = final.with_suffix(".csv.pending")
                handle = stack.enter_context(temp.open("w", encoding="utf-8", newline=""))
                writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
                writer.writeheader()
                writers[station, kind] = writer
                temporary.append((temp, final))
        for part in store.results():
            for kind in tables:
                for row in part[kind]:
                    targets = ["all"]
                    if split:
                        targets.append(station_slugs[str(row["station_id"])])
                    for station in targets:
                        writers[station, kind].writerow(row)
                        if kind == "metrics":
                            metrics[station].append(row)
                        elif kind == "long":
                            counts[station] += 1
    # Сбой сборки оставляет старый CSV целым; журнал позволяет повторить экспорт.
    for temp, final in temporary:
        temp.replace(final)
    for station, directory in destinations.items():
        station_metrics = metrics[station]
        write_monthly_summary(station_metrics, directory)
        write_station_summary(station_metrics, directory)
        write_field_types_csv(directory)
        info = {**config, "station_slug": station}
        summary = {"config": info, "profiles_total": len(station_metrics),
                   "levels_total": counts[station],
                   "profiles_good": sum(r["profile_status"] == "good" for r in station_metrics),
                   "profiles_with_inversion": sum(bool(r.get("inversion_detected")) for r in station_metrics),
                   "stations": dict(Counter(str(r["station_id"]) for r in station_metrics)),
                   "errors": store.failures()}
        (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if working_copy:
            import shutil
            shutil.copyfile(directory / "profiles_long.csv", directory / "profiles_working.csv")
        if xlsx:
            # Excel нужен только по запросу; для региона создаём книги по станциям.
            if not split or station != "all":
                from gdex_bufr.profile_climate.cli import _load_csv_rows
                rows = _load_csv_rows(directory / "profiles_long.csv")
                write_xlsx_exports(rows, station_metrics, directory)
    return {"profiles": len(metrics["all"]), "levels": counts["all"],
            "errors": len(store.failures()), "output": str(output)}
