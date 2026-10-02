from __future__ import annotations

import sys
from pathlib import Path

ARTICLE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ARTICLE_ROOT.parent
for directory in (PROJECT_ROOT, ARTICLE_ROOT):
    if str(directory) not in sys.path:
        sys.path.append(str(directory))
