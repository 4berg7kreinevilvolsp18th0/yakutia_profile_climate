"""Скопировать/hardlink BUFR с Aldan в gdex_data/bufr_алдан по индексу."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def _link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Копия BUFR с Aldan")
    p.add_argument("--index", default="gdex_outputs/результаты-алдан-полный/aldan_bufr_index.csv")
    p.add_argument("--dest-root", default="gdex_data/bufr_алдан")
    p.add_argument("--manifest", default="gdex_outputs/результаты-алдан-полный/bufr_алдан_manifest.csv")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    index_path = Path(args.index)
    if not index_path.exists():
        logger.error("Нет индекса: %s", index_path)
        return 1

    # unique source paths
    by_file: dict[str, dict] = {}
    with index_path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            src = row.get("source_path") or ""
            if not src:
                continue
            entry = by_file.setdefault(src, {
                "source_path": src,
                "source_file": row.get("source_file", ""),
                "file_ymd": row.get("file_ymd", ""),
                "obs_datetimes": set(),
            })
            if row.get("obs_datetime"):
                entry["obs_datetimes"].add(row["obs_datetime"])

    dest_root = Path(args.dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict] = []
    counts = {"hardlink": 0, "copy": 0, "exists": 0, "missing": 0}

    for src_str, meta in sorted(by_file.items()):
        src = Path(src_str)
        if not src.exists():
            counts["missing"] += 1
            logger.warning("нет файла %s", src)
            continue
        year = meta["file_ymd"][:4] if len(meta["file_ymd"]) >= 4 else src.parent.name
        dst = dest_root / year / src.name
        mode = _link_or_copy(src, dst)
        counts[mode] = counts.get(mode, 0) + 1
        manifest_rows.append({
            "path_src": str(src),
            "path_dst": str(dst),
            "size": str(src.stat().st_size),
            "mode": mode,
            "obs_datetimes": ";".join(sorted(meta["obs_datetimes"])),
        })

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["path_src", "path_dst", "size", "mode", "obs_datetimes"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "unique_files": len(by_file),
        "manifest": str(manifest_path),
        "dest_root": str(dest_root),
        **counts,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
