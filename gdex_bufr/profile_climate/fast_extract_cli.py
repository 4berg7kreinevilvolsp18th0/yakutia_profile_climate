"""Тонкая обёртка: делегирует в gdex_bufr.run_fast_extract → корневой канон.

Команда: python -m gdex_bufr.profile_climate.fast_extract_cli
"""
from __future__ import annotations

from gdex_bufr.run_fast_extract import build_parser, main

__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
