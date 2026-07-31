"""Обёртка. Предпочтительно: py -3 -m gdex_bufr.run_fast_extract ..."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gdex_bufr.run_fast_extract import main

if __name__ == "__main__":
    raise SystemExit(main())
