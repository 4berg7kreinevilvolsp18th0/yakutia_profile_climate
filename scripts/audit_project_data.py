"""Проверка сохранённых данных и исправление подтверждённых ошибок рисунков.

Без --apply создаёт только отчёт. С --apply переносит подтверждённые пустые
рисунки и прежние неверные версии в архив с путями и SHA-256, затем пересчитывает
повторяемость из исходных слоёв. Сами наблюдения и сомнительные значения сохраняет.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

from gdex_bufr.profile_climate.article_figures.config import load_yaml_config
from gdex_bufr.profile_climate.article_figures.metrics import recurrence_percent_table
from gdex_bufr.profile_climate.article_figures.pipeline import save_figure
from gdex_bufr.profile_climate.article_figures.plots import plot_recurrence_by_type_bars


def move_to_archive(path: Path, output: Path, reason: str) -> dict:
    """Перенос только конкретного файла внутри проекта, без удаления каталогов."""
    path = path.resolve()
    relative = path.relative_to(ROOT)
    target = (output / "quarantine" / relative).resolve()
    target.relative_to(output.resolve())
    if target.exists():
        raise FileExistsError(f"Архив уже содержит {target}; выберите новый --output")
    record = {"source": str(relative), "archive": str(target), "reason": reason,
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    target.parent.mkdir(parents=True, exist_ok=True)
    # Журнал записывается до переноса: при прерывании путь восстановления известен.
    with (output / "archive_manifest.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    path.replace(target)
    return record


def check_profiles(path: Path) -> dict:
    frame = pd.read_csv(path, low_memory=False)
    p = pd.to_numeric(frame.pressure_hpa, errors="coerce")
    t = pd.to_numeric(frame.temperature_c, errors="coerce")
    stats = {"file": str(path.relative_to(ROOT)), "rows": len(frame),
             "profiles": int(frame.profile_id.nunique()),
             "stations": sorted(frame.station_id.dropna().astype(str).unique().tolist()),
             "invalid_pressure": int((~np.isfinite(p) | (p <= 0) | (p > 1100)).sum()),
             "missing_temperature": int((~np.isfinite(t)).sum()),
             "temperature_review_outside_minus100_plus60": int(((t < -100) | (t > 60)).sum()),
             "duplicate_profile_pressure_rows": int(frame.duplicated(["profile_id", "pressure_hpa"], keep=False).sum())}
    if "height_m" in frame:
        stats["missing_height"] = int((~np.isfinite(pd.to_numeric(frame.height_m, errors="coerce"))).sum())
    if {"source_file", "subset_index"} <= set(frame):
        identities = frame[["profile_id", "source_file", "subset_index"]].drop_duplicates()
        stats["profile_ids_with_multiple_sources_or_subsets"] = int((identities.groupby("profile_id").size() > 1).sum())
    return stats


def audit_graphics(article: Path, output: Path, apply: bool) -> dict:
    analysis, style = load_yaml_config(article / "article_figures_config.yaml")
    style = replace(style, dpi=min(style.dpi, 300))
    actions = []
    checks = []
    specs = [
        ("top_height_recurrence_percent", "top_height_agl_m", False, "01_top_height_recurrence_G_E_HE_fixed"),
        ("base_height_recurrence_percent", "base_height_agl_m", False, "02_base_height_recurrence_G_E_HE_fixed"),
        ("depth_recurrence_percent", "depth_m", False, "03_inversion_depth_recurrence_G_E_HE_fixed"),
        ("top_height_month_recurrence_percent", "top_height_agl_m", True, None),
    ]
    for source in sorted(article.rglob("inversion_layers_height_fixed.csv")):
        if "quarantine" in source.parts:
            continue
        tables = source.parent
        qc_file = tables / "profile_qc.csv"
        if not qc_file.exists():
            continue
        layers = pd.read_csv(source)
        qc = pd.read_csv(qc_file)
        figures = tables.parent / "figures"
        depth = layers.top_height_agl_m - layers.base_height_agl_m
        bad = (~np.isfinite(depth) | (depth <= 0) | (layers.base_height_agl_m < -1e-6)
               | ~np.isclose(depth, layers.depth_m, atol=.001))
        gamma = 100 * layers.delta_t_c / layers.depth_m
        gamma_raw = layers.get("gamma_raw_c_per_100m", layers.gamma_c_per_100m)
        gamma_bad = np.isfinite(gamma_raw) & ~np.isclose(gamma, gamma_raw, atol=1e-6)
        checks.append({"source": str(source.relative_to(ROOT)), "layers": len(layers),
                       "invalid_geometry": int(bad.sum()), "gamma_formula_mismatch": int(gamma_bad.sum())})
        # Пустота отрицательных 3D-инверсий подтверждается исходной таблицей.
        # Отрицательные градиенты обычных профилей сюда не относятся.
        for kind in ("G", "E", "HE"):
            subset = layers[layers.position_type == kind]
            if len(subset) and not (subset.gamma_c_per_100m < 0).any():
                for picture in sorted(figures.rglob(f"top_height_depth_gamma_3d_{kind}_neg_*")):
                    if picture.suffix.lower() not in {".png", ".svg"}:
                        continue
                    reason = "Нет отрицательных градиентов в исходной выборке инверсионных слоёв"
                    actions.append(move_to_archive(picture, output, reason) if apply else {"source": str(picture), "reason": reason})
        for table_name, column, monthly, figure_name in specs:
            path = tables / (table_name + ".csv")
            corrected = recurrence_percent_table(layers, qc, bin_edges=analysis.layers.height_bin_edges_m,
                                                 value_col=column, by_month=monthly)
            if not path.exists():
                continue
            previous = pd.read_csv(path)
            keys = ["month", "position_type", "bin_left"]
            compare = previous.merge(corrected, on=keys, suffixes=("_old", "_new"))
            changes = int((~np.isclose(compare.recurrence_percent_old, compare.recurrence_percent_new,
                                      equal_nan=True, atol=1e-9)).sum())
            checks.append({"table": str(path.relative_to(ROOT)), "changed_bins": changes})
            if not changes:
                continue
            reason = f"Исправлен знаменатель/счётчик уникальных профилей: {changes} бинов"
            if apply:
                actions.append(move_to_archive(path, output, reason))
                corrected.to_csv(path, index=False)
                if figure_name:
                    base = figures / figure_name
                    for suffix in (".png", ".svg"):
                        old = base.with_suffix(suffix)
                        if old.exists():
                            actions.append(move_to_archive(old, output, reason))
                    figure = plot_recurrence_by_type_bars(corrected, style, value_name="Повторяемость профилей, %")
                    if column == "depth_m":
                        for axis in figure.axes:
                            axis.set_xlabel("Толщина слоя, м")
                    save_figure(figure, base, style)
            else:
                actions.append({"source": str(path.relative_to(ROOT)), "reason": reason})
    return {"checks": checks, "actions": actions}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="gdex_outputs/audit_20261002/graphics")
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    output = (ROOT / args.output).resolve()
    output.relative_to(ROOT)
    output.mkdir(parents=True, exist_ok=True)
    profiles = [check_profiles(path) for path in sorted((ROOT / "gdex_outputs/far_east/stations").glob("*/profiles_long.csv"))]
    report = {"profiles": profiles, "graphics": audit_graphics(ROOT / "Графики для статьи", output, args.apply)}
    (output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"profiles": profiles, "checks": report["graphics"]["checks"],
                      "actions": len(report["graphics"]["actions"]), "applied": args.apply}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
