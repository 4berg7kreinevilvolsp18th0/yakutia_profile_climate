"""Тонкая обёртка над корневым run_fast_extract (канон Windows ProcessPool).

Сохраняет команду: python -m gdex_bufr.run_fast_extract [--actual ...]
Исторический default --end-date для этой точки входа: 2026-07-08.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_DEFAULT_END_DATE = "2026-07-08"


def build_parser():
    import run_fast_extract as canon

    parser = canon.build_parser()
    for action in parser._actions:
        if getattr(action, "dest", None) == "end_date":
            action.default = _DEFAULT_END_DATE
    return parser


def _with_default_end_date(argv: list[str] | None) -> list[str]:
    args = list(sys.argv[1:] if argv is None else argv)
    if any(a == "--end-date" or a.startswith("--end-date=") for a in args):
        return args
    return ["--end-date", _DEFAULT_END_DATE, *args]


def main(argv: list[str] | None = None) -> int:
    import run_fast_extract as canon

    return canon.main(_with_default_end_date(argv))


if __name__ == "__main__":
    raise SystemExit(main())
