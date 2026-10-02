"""Тонкая обёртка над корневым run_fast_extract (канон Windows ProcessPool).

Сохраняет команду: python -m gdex_bufr.run_fast_extract [--actual ...]
Даты и все остальные значения по умолчанию совпадают с корневым CLI.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def build_parser():
    import run_fast_extract as canon

    parser = canon.build_parser()
    return parser


def _with_default_end_date(argv: list[str] | None) -> list[str]:
    args = list(sys.argv[1:] if argv is None else argv)
    if any(a == "--end-date" or a.startswith("--end-date=") for a in args):
        return args
    return args


def main(argv: list[str] | None = None) -> int:
    import run_fast_extract as canon

    return canon.main(_with_default_end_date(argv))


if __name__ == "__main__":
    raise SystemExit(main())
