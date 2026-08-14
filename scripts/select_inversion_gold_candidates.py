"""Отбор 50–100 проблемных профилей из comparison_v2_v3.csv для ручной разметки."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "да"}


def select_candidates(rows: list[dict[str, str]], *, limit: int) -> list[dict[str, str]]:
    scored: list[tuple[int, dict[str, str]]] = []
    for row in rows:
        n_v3 = int(float(row.get("n_inversion_layers_v3") or 0))
        v2 = _truthy(row.get("inversion_detected_v2", ""))
        pattern = str(row.get("pattern_v3") or "NONE")
        score = 0
        reasons: list[str] = []
        if v2 and n_v3 == 0:
            score += 40
            reasons.append("v2_yes_v3_none")
        if (not v2) and n_v3 >= 2:
            score += 30
            reasons.append("v2_no_v3_multi")
        if pattern == "MULTI":
            score += 15
            reasons.append("pattern_multi")
        if pattern == "NONE" and _truthy(row.get("inversion_candidate_v2", "")):
            score += 20
            reasons.append("v2_candidate_v3_none")
        if score <= 0:
            continue
        out = dict(row)
        out["reason"] = "+".join(reasons)
        out["score"] = str(score)
        scored.append((score, out))
    scored.sort(key=lambda item: (-item[0], item[1].get("profile_id") or ""))
    return [row for _, row in scored[:limit]]


def main() -> int:
    parser = argparse.ArgumentParser(description="Список проблемных профилей v2 vs v3")
    parser.add_argument(
        "--comparison",
        default="gdex_outputs/far_east/stations/aldan/v3/comparison_v2_v3.csv",
    )
    parser.add_argument(
        "--output",
        default="gdex_outputs/far_east/stations/aldan/v3/gold_candidates.csv",
    )
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()

    src = Path(args.comparison)
    if not src.exists():
        print(f"Нет файла {src}", file=sys.stderr)
        return 1
    with src.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    picked = select_candidates(rows, limit=args.limit)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(picked[0].keys()) if picked else ["profile_id", "reason", "score"]
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(picked)
    print(f"candidates={len(picked)} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
